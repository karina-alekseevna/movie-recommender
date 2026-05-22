# backend/ingestion/build_entities.py

from typing import List, Dict, Tuple
from backend.nlp.embedder import embedder
import numpy as np


def build_movie_entities(movie: Dict) -> List[Dict]:
    """
    Из одного фильма извлекает все сущности и создаёт для каждой вектор.
    
    Сущности фильма — это его «семантический отпечаток».
    Каждая сущность (жанр, ключевое слово, актёр, режиссёр) 
    получает свой вектор через SentenceTransformer.
    """
    entities = []
    
    # Жанры
    for genre in movie['genres']:
        entities.append({
            'entity_type': 'genre',
            'entity_value': genre,
            'entity_normalized': genre.lower().strip(),
        })
    
    # Ключевые слова
    for keyword in movie['keywords']:
        entities.append({
            'entity_type': 'keyword',
            'entity_value': keyword,
            'entity_normalized': keyword.lower().strip(),
        })
    
    # Актёры (берём топ-10 для экономии)
    for actor_info in movie['cast'][:10]:
        entities.append({
            'entity_type': 'actor',
            'entity_value': actor_info['name'],
            'entity_normalized': actor_info['name'].lower().strip(),
        })
    
    # Режиссёры
    for director in movie['directors']:
        entities.append({
            'entity_type': 'director',
            'entity_value': director,
            'entity_normalized': director.lower().strip(),
        })
    
    # Сценаристы
    for writer in movie['writers']:
        entities.append({
            'entity_type': 'writer',
            'entity_value': writer,
            'entity_normalized': writer.lower().strip(),
        })
    
    # Векторизуем все сущности батчем (эффективнее)
    if entities:
        texts = [e['entity_value'] for e in entities]
        embeddings = embedder.encode(texts)
        for i, entity in enumerate(entities):
            entity['embedding'] = embeddings[i]
    
    return entities


def build_movie_embedding(movie: Dict) -> np.ndarray:
    """
    Создаёт единый вектор фильма из комбинации его текстовых данных.
    
    Стратегия: объединяем наиболее информативные текстовые поля
    в один «документ» и кодируем его целиком.
    
    Почему так: SentenceTransformer понимает контекст,
    поэтому "драма о дружбе в тюрьме" даст более осмысленный вектор,
    чем среднее от отдельных слов.
    """
    parts = []
    
    # Название
    parts.append(movie['title'])
    
    # Жанры
    if movie['genres']:
        parts.append(f"Жанр: {', '.join(movie['genres'])}")
    
    # Описание (самая информативная часть)
    if movie['overview']:
        parts.append(movie['overview'])
    
    # Ключевые слова
    if movie['keywords']:
        parts.append(f"Темы: {', '.join(movie['keywords'][:15])}")
    
    # Режиссёр
    if movie['directors']:
        parts.append(f"Режиссёр: {', '.join(movie['directors'])}")
    
    # Главные актёры (топ-5)
    actor_names = [a['name'] for a in movie['cast'][:5]]
    if actor_names:
        parts.append(f"В ролях: {', '.join(actor_names)}")
    
    combined_text = '. '.join(parts)
    
    # Sentence Transformer имеет лимит ~256 токенов,
    # обрезаем если слишком длинный текст
    if len(combined_text) > 1000:
        combined_text = combined_text[:1000]
    
    return embedder.encode(combined_text)
