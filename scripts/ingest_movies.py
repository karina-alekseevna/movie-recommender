# scripts/ingest_movies.py

import asyncio
import sys
sys.path.append('.')

from backend.ingestion.load_movies import load_movies_from_csv
from backend.ingestion.build_entities import build_movie_entities, build_movie_embedding
from backend.database.connection import get_supabase
from backend.nlp.embedder import embedder
import numpy as np


async def ingest_all_movies(csv_path: str = 'data/movies.csv'):
    
    print("1. Загрузка фильмов из CSV...")
    movies = load_movies_from_csv(csv_path)
    print(f"   Загружено {len(movies)} фильмов")
    
    supabase = get_supabase()
    
    for i, movie in enumerate(movies):
        print(f"\n2. Обработка [{i+1}/{len(movies)}]: {movie['title']}")
        
        # 2.1 Создаём вектор фильма
        print("   Создание эмбеддинга фильма...")
        movie_embedding = build_movie_embedding(movie)
        
        # 2.2 Вставляем фильм в БД
        movie_data = {
            'tmdb_id': movie['tmdb_id'],
            'title': movie['title'],
            'original_title': movie['original_title'],
            'release_year': movie['release_year'],
            'release_date': movie['release_date'],
            'runtime': movie['runtime'],
            'overview': movie['overview'],
            'rating': movie['rating'],
            'rating_count': movie['rating_count'],
            'popularity': movie['popularity'],
            'budget': movie['budget'],
            'revenue': movie['revenue'],
            'poster_url': movie['poster_url'],
            'backdrop_url': movie['backdrop_url'],
            'tmdb_url': movie['tmdb_url'],
            'imdb_id': movie['imdb_id'],
            'original_language': movie['original_language'],
            'genres_raw': movie['genres_raw'],
            'keywords_raw': movie['keywords_raw'],
            'cast_raw': movie['cast_raw'],
            'directors_raw': movie['directors_raw'],
            'writers_raw': movie['writers_raw'],
            'producers_raw': movie['producers_raw'],
            'embedding': movie_embedding.tolist(),
        }
        
        result = supabase.table('movies').upsert(
            movie_data, on_conflict='tmdb_id'
        ).execute()
        
        movie_id = result.data[0]['id']
        
        # 2.3 Извлекаем и сохраняем сущности
        print("   Извлечение сущностей...")
        entities = build_movie_entities(movie)
        print(f"   Найдено {len(entities)} сущностей")
        
        for entity in entities:
            entity_data = {
                'movie_id': movie_id,
                'entity_type': entity['entity_type'],
                'entity_value': entity['entity_value'],
                'entity_normalized': entity['entity_normalized'],
                'embedding': entity['embedding'].tolist(),
            }
            
            supabase.table('movie_entities').upsert(
                entity_data,
                on_conflict='movie_id,entity_type,entity_normalized'
            ).execute()
        
        print(f"   ✓ Готово: {movie['title']}")
    
    print(f"\n{'='*50}")
    print(f"Загрузка завершена! Обработано {len(movies)} фильмов.")


if __name__ == '__main__':
    asyncio.run(ingest_all_movies())
