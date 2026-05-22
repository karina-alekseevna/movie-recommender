# backend/nlp/entity_matcher.py
import numpy as np
import json
import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from backend.nlp.embedder import embedder
from backend.nlp.segmenter import ReviewSegment
from backend.nlp.sentiment import SentimentResult
from backend.database.connection import get_supabase


@dataclass
class MatchedEntity:
    entity_type: str
    entity_value: str
    entity_normalized: str
    match_method: str       # 'exact', 'surname', 'fuzzy_name'
    match_score: float      # 0..1
    entity_embedding: np.ndarray = field(repr=False)


@dataclass
class AnalyzedSegment:
    segment: ReviewSegment
    sentiment: SentimentResult
    matched_entities: List[MatchedEntity]
    segment_embedding: np.ndarray = field(repr=False)
    is_general: bool = False


class EntityMatcher:
    """
    Матчер сущностей — только точные и fuzzy-строковые совпадения.
    Никакого семантического матчинга: он даёт слишком много false positives
    для имён на разных языках.
    """

    # Минимальная длина фамилии для surname-матча
    MIN_SURNAME_LENGTH = 5

    # Стоп-лист общих слов, которые не должны триггерить keyword-матч
    KEYWORD_STOPLIST = {
        'мультфильм', 'комедия', 'драма', 'приключения', 'фильм',
        'кино', 'история', 'семейный', 'боевик', 'триллер',
        'злодей', 'месть', 'заговор', 'анимация', 'сюжет',
        'эмоции', 'дружба', 'любовь', 'смерть', 'предательство',
        'юмор', 'детектив', 'шедевр', 'разочарование',
        'надежда', 'грусть', 'восторг', 'музыка', 'песня',
    }

    # Минимальная длина keyword для однословного матча
    MIN_SINGLE_KEYWORD_LENGTH = 6

    def __init__(self):
        self.supabase = get_supabase()
        self._entity_cache: Dict[int, List[Dict]] = {}

    # ----------------------------------------------------------------
    # Загрузка сущностей
    # ----------------------------------------------------------------
    def get_movie_entities(self, movie_id: int) -> List[Dict]:
        """Получает все сущности фильма из БД (с кэшированием)."""
        if movie_id in self._entity_cache:
            return self._entity_cache[movie_id]

        result = self.supabase.table('movie_entities').select(
            'entity_type, entity_value, entity_normalized, embedding'
        ).eq('movie_id', movie_id).execute()

        entities = []
        for row in result.data:
            emb_val = row['embedding']
            if isinstance(emb_val, str):
                emb_val = json.loads(emb_val)
            entity_emb = np.array(emb_val, dtype=np.float32)

            entities.append({
                'entity_type': row['entity_type'],
                'entity_value': row['entity_value'],
                'entity_normalized': row['entity_normalized'],
                'embedding': entity_emb,
            })

        self._entity_cache[movie_id] = entities
        return entities

    # ----------------------------------------------------------------
    # Вспомогательные методы
    # ----------------------------------------------------------------
    @staticmethod
    def _extract_person_name(normalized: str) -> str:
        """Достаёт имя персоны (до скобок)."""
        if '(' in normalized:
            return normalized.split('(')[0].strip()
        return normalized

    @staticmethod
    def _extract_character_names(entity_value: str) -> List[str]:
        """
        Извлекает имена персонажей из entity_value.
        Формат: «Actor Name (Character Name (voice))»
        """
        characters = []
        parts = re.findall(r'$([^)]+)$', entity_value)
        for part in parts:
            name = re.sub(r'\s*$?voice$?', '', part, flags=re.IGNORECASE).strip()
            if name and len(name) > 2:
                characters.append(name.lower())
        return characters

    @staticmethod
    def _word_boundary_search(text: str, phrase: str) -> bool:
        """Проверяет наличие фразы как целого слова/словосочетания."""
        pattern = r'(?<![а-яёa-z])' + re.escape(phrase) + r'(?![а-яёa-z])'
        return bool(re.search(pattern, text, re.IGNORECASE))

    @staticmethod
    def _normalize_for_fuzzy(name: str) -> str:
        """Убирает ё→е, лишние пробелы, приводит к нижнему регистру."""
        name = name.lower().strip()
        name = name.replace('ё', 'е')
        name = re.sub(r'\s+', ' ', name)
        return name

    # ----------------------------------------------------------------
    # Генерация вариантов имени для поиска
    # ----------------------------------------------------------------
    def _build_search_variants(self, entity: Dict) -> List[Tuple[str, str, float]]:
        """
        Возвращает список (variant_name, match_method, match_score)
        для данной сущности.
        """
        entity_type = entity['entity_type']
        normalized = entity['entity_normalized']
        variants = []

        if entity_type in ('actor', 'director', 'writer'):
            person_name = self._normalize_for_fuzzy(
                self._extract_person_name(normalized)
            )

            if person_name:
                # Полное имя — лучший матч
                variants.append((person_name, 'exact', 1.0))

                # Фамилия (последнее слово), если достаточно длинная
                parts = person_name.split()
                if len(parts) >= 2:
                    surname = parts[-1]
                    if len(surname) >= self.MIN_SURNAME_LENGTH:
                        variants.append((surname, 'surname', 0.85))

                    # Имя (первое слово), если достаточно длинное
                    first_name = parts[0]
                    if len(first_name) >= self.MIN_SURNAME_LENGTH:
                        variants.append((first_name, 'surname', 0.7))

            # Имена персонажей (только для актёров)
            if entity_type == 'actor':
                for char_name in self._extract_character_names(entity['entity_value']):
                    char_normalized = self._normalize_for_fuzzy(char_name)
                    if char_normalized:
                        variants.append((char_normalized, 'exact', 1.0))

                        # Части имени персонажа (если многословное)
                        char_parts = char_normalized.split()
                        if len(char_parts) >= 2:
                            for cp in char_parts:
                                if len(cp) >= 4:
                                    variants.append((cp, 'surname', 0.8))

        return variants

    # ----------------------------------------------------------------
    # Точный + fuzzy-строковый матчинг персон
    # ----------------------------------------------------------------
    def _match_persons(self, segment_text: str,
                       person_entities: List[Dict]) -> List[MatchedEntity]:
        """
        Ищет персон в тексте сегмента через точное и fuzzy-строковое совпадение.
        """
        text_normalized = self._normalize_for_fuzzy(segment_text)
        matches = []

        for ent in person_entities:
            variants = self._build_search_variants(ent)
            best_match = None

            for variant_name, method, score in variants:
                if self._word_boundary_search(text_normalized, variant_name):
                    # Дополнительная проверка для коротких фамилий:
                    # не матчим если это часть другого слова
                    if method == 'surname' and len(variant_name) < 6:
                        # Проверяем что это не часть обычного русского слова
                        idx = text_normalized.find(variant_name)
                        if idx == -1:
                            continue

                    if best_match is None or score > best_match.match_score:
                        best_match = MatchedEntity(
                            entity_type=ent['entity_type'],
                            entity_value=ent['entity_value'],
                            entity_normalized=ent['entity_normalized'],
                            match_method=method,
                            match_score=score,
                            entity_embedding=ent['embedding'],
                        )

            if best_match:
                matches.append(best_match)

        # Дедупликация по (тип, normalized) — оставляем лучший score
        best = {}
        for m in matches:
            key = (m.entity_type, m.entity_normalized)
            if key not in best or m.match_score > best[key].match_score:
                best[key] = m
        return list(best.values())

    # ----------------------------------------------------------------
    # Точный матчинг keywords
    # ----------------------------------------------------------------
    def _match_keywords(self, segment_text: str,
                        keyword_entities: List[Dict]) -> List[MatchedEntity]:
        """Точный матчинг ключевых слов."""
        text_normalized = self._normalize_for_fuzzy(segment_text)
        matches = []

        for ent in keyword_entities:
            kw = self._normalize_for_fuzzy(ent['entity_normalized'])

            # Пропускаем стоп-слова
            if kw in self.KEYWORD_STOPLIST:
                continue

            # Для однословных — требуем минимальную длину
            if ' ' not in kw and len(kw) < self.MIN_SINGLE_KEYWORD_LENGTH:
                continue

            if self._word_boundary_search(text_normalized, kw):
                matches.append(MatchedEntity(
                    entity_type=ent['entity_type'],
                    entity_value=ent['entity_value'],
                    entity_normalized=ent['entity_normalized'],
                    match_method='exact',
                    match_score=1.0,
                    entity_embedding=ent['embedding'],
                ))

        # Дедупликация
        best = {}
        for m in matches:
            key = (m.entity_type, m.entity_normalized)
            if key not in best or m.match_score > best[key].match_score:
                best[key] = m
        return list(best.values())

    # ----------------------------------------------------------------
    # Главный метод: матчинг одного сегмента
    # ----------------------------------------------------------------
    def match_segment(
        self,
        segment: ReviewSegment,
        movie_entities: List[Dict]
    ) -> List[MatchedEntity]:
        """
        Матчит сущности к сегменту ТОЛЬКО через точное/строковое совпадение.
        Никакого семантического матчинга.
        """
        # Отделяем жанры — они не матчатся к сегментам
        non_genre = [e for e in movie_entities if e['entity_type'] != 'genre']
        person_entities = [
            e for e in non_genre
            if e['entity_type'] in ('actor', 'director', 'writer')
        ]
        keyword_entities = [
            e for e in non_genre
            if e['entity_type'] == 'keyword'
        ]

        person_matches = self._match_persons(segment.text, person_entities)
        keyword_matches = self._match_keywords(segment.text, keyword_entities)

        return person_matches + keyword_matches

    # ----------------------------------------------------------------
    # Анализ всего отзыва
    # ----------------------------------------------------------------
    def analyze_review(
        self,
        segments: List[ReviewSegment],
        sentiments: List[SentimentResult],
        movie_id: int
    ) -> List[AnalyzedSegment]:
        movie_entities = self.get_movie_entities(movie_id)
        segment_texts = [s.text for s in segments]
        segment_embeddings = embedder.encode(segment_texts)

        analyzed = []
        for i, (segment, sentiment) in enumerate(zip(segments, sentiments)):
            seg_embedding = segment_embeddings[i]
            matched = self.match_segment(segment, movie_entities)
            analyzed.append(AnalyzedSegment(
                segment=segment,
                sentiment=sentiment,
                matched_entities=matched,
                segment_embedding=seg_embedding,
                is_general=(len(matched) == 0),
            ))
        return analyzed


entity_matcher = EntityMatcher()
