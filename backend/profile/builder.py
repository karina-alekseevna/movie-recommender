# backend/profile/builder.py
import numpy as np
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from backend.nlp.entity_matcher import AnalyzedSegment
from backend.database.connection import get_supabase


@dataclass
class EntityPreferenceUpdate:
    entity_type: str
    entity_value: str
    entity_normalized: str
    score_delta: float
    entity_embedding: np.ndarray


class ProfileBuilder:
    """
    Логика обновления профиля:

    1. ПРЯМОЕ УПОМИНАНИЕ (direct mention):
       Если сущность найдена в сегменте через exact/surname матч,
       она получает полный вклад: segment_sentiment × match_score

    2. MICRO-BONUS (ambient signal):
       ВСЕ сущности фильма (актёры, режиссёры, сценаристы, жанры, keywords)
       получают маленький бонус на основе ОБЩЕЙ тональности отзыва.
       Это один раз за отзыв, не за каждый сегмент.
       Коэффициент: overall_sentiment × MICRO_BONUS_FACTOR

    Это позволяет:
    - Точно учитывать, что пользователь хвалит/ругает конкретного актёра
    - Постепенно накапливать предпочтения к жанрам, режиссёрам и т.д.
      даже если пользователь не упоминает их по имени
    """

    # Множитель для micro-bonus (маленький, чтобы не забивал прямые упоминания)
    MICRO_BONUS_FACTOR = 0.05

    # Минимальный |sentiment| для micro-bonus (не даём бонус за нейтральные отзывы)
    MICRO_BONUS_SENTIMENT_THRESHOLD = 0.15

    def __init__(self):
        self.supabase = get_supabase()

    # ----------------------------------------------------------------
    # Применение обновлений к БД
    # ----------------------------------------------------------------
    def apply_entity_updates(self, user_id: str, updates: List[EntityPreferenceUpdate]):
        """Применяет список обновлений к user_entity_preferences."""
        # Группируем по (тип, normalized)
        grouped: Dict[Tuple[str, str], EntityPreferenceUpdate] = {}
        for update in updates:
            key = (update.entity_type, update.entity_normalized)
            if key in grouped:
                grouped[key].score_delta += update.score_delta
            else:
                grouped[key] = EntityPreferenceUpdate(
                    entity_type=update.entity_type,
                    entity_value=update.entity_value,
                    entity_normalized=update.entity_normalized,
                    score_delta=update.score_delta,
                    entity_embedding=update.entity_embedding,
                )

        for key, update in grouped.items():
            # Пропускаем нулевые обновления
            if abs(update.score_delta) < 1e-6:
                continue

            existing = self.supabase.table('user_entity_preferences').select('*').match({
                'user_id': user_id,
                'entity_type': update.entity_type,
                'entity_normalized': update.entity_normalized,
            }).execute()

            if existing.data:
                record = existing.data[0]
                new_score = record['score'] + update.score_delta
                new_count = record['mention_count'] + 1
                new_weight = float(np.clip(new_score / max(new_count, 1), -1.0, 1.0))

                self.supabase.table('user_entity_preferences').update({
                    'score': round(new_score, 4),
                    'mention_count': new_count,
                    'weight': round(new_weight, 4),
                    'updated_at': 'now()',
                }).eq('id', record['id']).execute()
            else:
                initial_weight = float(np.clip(update.score_delta, -1.0, 1.0))
                self.supabase.table('user_entity_preferences').insert({
                    'user_id': user_id,
                    'entity_type': update.entity_type,
                    'entity_value': update.entity_value,
                    'entity_normalized': update.entity_normalized,
                    'score': round(update.score_delta, 4),
                    'weight': round(initial_weight, 4),
                    'mention_count': 1,
                    'entity_embedding': update.entity_embedding.tolist(),
                }).execute()

    # ----------------------------------------------------------------
    # Векторный профиль
    # ----------------------------------------------------------------
    def _parse_embedding(self, val, dim: int = 384):
        if val is None:
            return np.zeros(dim, dtype=np.float32)
        if isinstance(val, str):
            val = json.loads(val)
        return np.array(val, dtype=np.float32)

    def update_vector_profile(self, user_id: str, movie_embedding: np.ndarray,
                              overall_sentiment: float):
        existing = self.supabase.table('user_vector_profiles').select('*').eq(
            'user_id', user_id
        ).execute()

        if existing.data:
            profile = existing.data[0]
            n = profile['reviews_count'] + 1
            old_positive = self._parse_embedding(profile.get('positive_centroid'))
            old_negative = self._parse_embedding(profile.get('negative_centroid'))

            if overall_sentiment > 0.1:
                new_positive = old_positive + (movie_embedding - old_positive) / n
                new_positive = new_positive / (np.linalg.norm(new_positive) + 1e-8)
                new_negative = old_negative
            elif overall_sentiment < -0.1:
                new_negative = old_negative + (movie_embedding - old_negative) / n
                new_negative = new_negative / (np.linalg.norm(new_negative) + 1e-8)
                new_positive = old_positive
            else:
                new_positive = old_positive
                new_negative = old_negative

            overall = new_positive - 0.3 * new_negative
            norm = np.linalg.norm(overall)
            if norm > 0:
                overall = overall / norm

            self.supabase.table('user_vector_profiles').update({
                'positive_centroid': new_positive.tolist(),
                'negative_centroid': new_negative.tolist(),
                'overall_centroid': overall.tolist(),
                'reviews_count': n,
                'updated_at': 'now()',
            }).eq('user_id', user_id).execute()
        else:
            positive = movie_embedding.copy() if overall_sentiment > 0.1 \
                else np.zeros(384, dtype=np.float32)
            negative = movie_embedding.copy() if overall_sentiment < -0.1 \
                else np.zeros(384, dtype=np.float32)
            if overall_sentiment > 0.1:
                overall = positive / (np.linalg.norm(positive) + 1e-8)
            elif overall_sentiment < -0.1:
                overall = -negative / (np.linalg.norm(negative) + 1e-8)
            else:
                overall = np.zeros(384, dtype=np.float32)

            self.supabase.table('user_vector_profiles').insert({
                'user_id': user_id,
                'positive_centroid': positive.tolist(),
                'negative_centroid': negative.tolist(),
                'overall_centroid': overall.tolist(),
                'reviews_count': 1,
            }).execute()

    # ----------------------------------------------------------------
    # Главный метод
    # ----------------------------------------------------------------
    def process_review(
        self,
        user_id: str,
        movie_id: int,
        analyzed_segments: List[AnalyzedSegment],
        movie_embedding: np.ndarray,
        overall_sentiment: float,
        all_movie_entities: List[Dict],
    ) -> Dict:
        """
        Обрабатывает отзыв и обновляет профиль пользователя.

        Два канала обновления:
        1. Direct mentions — из analyzed_segments
        2. Micro-bonus — для ВСЕХ сущностей фильма
        """
        updates: Dict[Tuple[str, str], EntityPreferenceUpdate] = {}

        # ============================================================
        # 1. DIRECT MENTIONS: сущности, найденные в конкретных сегментах
        # ============================================================
        for seg in analyzed_segments:
            if seg.is_general or not seg.matched_entities:
                continue

            seg_sentiment = seg.sentiment.score

            for entity in seg.matched_entities:
                key = (entity.entity_type, entity.entity_normalized)
                contribution = seg_sentiment * entity.match_score

                if key in updates:
                    updates[key].score_delta += contribution
                else:
                    updates[key] = EntityPreferenceUpdate(
                        entity_type=entity.entity_type,
                        entity_value=entity.entity_value,
                        entity_normalized=entity.entity_normalized,
                        score_delta=contribution,
                        entity_embedding=entity.entity_embedding,
                    )

        # Запоминаем ключи прямых упоминаний (для логирования)
        direct_mention_keys = set(updates.keys())

        # ============================================================
        # 2. MICRO-BONUS: все сущности фильма получают маленький бонус
        #    на основе общей тональности отзыва (один раз за отзыв)
        # ============================================================
        if abs(overall_sentiment) >= self.MICRO_BONUS_SENTIMENT_THRESHOLD:
            micro_bonus = overall_sentiment * self.MICRO_BONUS_FACTOR

            for ent in all_movie_entities:
                key = (ent['entity_type'], ent['entity_normalized'])

                if key in updates:
                    # Добавляем micro-bonus к уже существующему direct mention
                    updates[key].score_delta += micro_bonus
                else:
                    # Создаём новое обновление только с micro-bonus
                    updates[key] = EntityPreferenceUpdate(
                        entity_type=ent['entity_type'],
                        entity_value=ent['entity_value'],
                        entity_normalized=ent['entity_normalized'],
                        score_delta=micro_bonus,
                        entity_embedding=ent['embedding'],
                    )

        # Применяем все обновления
        all_updates = list(updates.values())
        self.apply_entity_updates(user_id, all_updates)

        # Обновляем векторный профиль
        self.update_vector_profile(user_id, movie_embedding, overall_sentiment)

        return {
            'overall_sentiment': overall_sentiment,
            'direct_mentions': len(direct_mention_keys),
            'total_entity_updates': len(all_updates),
            'micro_bonus_applied': abs(overall_sentiment) >= self.MICRO_BONUS_SENTIMENT_THRESHOLD,
        }


profile_builder = ProfileBuilder()
