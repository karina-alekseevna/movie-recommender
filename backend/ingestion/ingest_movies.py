# backend/ingestion/ingest_movies.py
import asyncio
import sys
import time
sys.path.append('.')

from backend.ingestion.load_movies import load_movies_from_csv
from backend.ingestion.build_entities import build_movie_entities, build_movie_embedding
from backend.database.connection import get_supabase
import numpy as np


def upsert_with_retry(supabase, table, data, on_conflict='tmdb_id', max_retries=3):
    for attempt in range(max_retries):
        try:
            result = supabase.table(table).upsert(data, on_conflict=on_conflict).execute()
            return result
        except Exception as e:
            print(f"   ⚠️ Ошибка при вставке в {table} (попытка {attempt+1}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def get_existing_tmdb_ids(supabase):
    try:
        result = supabase.table('movies').select('tmdb_id').execute()
        return {row['tmdb_id'] for row in result.data}
    except Exception:
        return set()


async def ingest_all_movies(csv_path: str = 'data/movies.csv'):
    print("1. Загрузка фильмов из CSV...")
    movies = load_movies_from_csv(csv_path)
    print(f"   Загружено {len(movies)} фильмов из CSV")

    supabase = get_supabase()
    print("   Проверка существующих фильмов в БД...")
    existing_ids = get_existing_tmdb_ids(supabase)
    print(f"   В БД уже {len(existing_ids)} фильмов")

    processed = 0
    skipped = 0

    for i, movie in enumerate(movies):
        if movie['tmdb_id'] in existing_ids:
            skipped += 1
            if skipped % 100 == 0:
                print(f"   Пропущено {skipped} уже существующих фильмов")
            continue

        print(f"\n2. Обработка [{i+1}/{len(movies)}]: {movie['title']}")
        print("   Создание эмбеддинга фильма...")
        movie_embedding = build_movie_embedding(movie)

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

        print("   Сохранение фильма в БД...")
        result = upsert_with_retry(supabase, 'movies', movie_data, on_conflict='tmdb_id')
        movie_id = result.data[0]['id']

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
            upsert_with_retry(supabase, 'movie_entities', entity_data,
                              on_conflict='movie_id,entity_type,entity_normalized')

        existing_ids.add(movie['tmdb_id'])
        processed += 1
        print(f"   ✓ Готово: {movie['title']} (новых: {processed})")

    print(f"\n{'='*50}")
    print(f"Загрузка завершена! Новых: {processed}, пропущено существующих: {skipped}")
    print(f"Всего в БД: {len(existing_ids)} фильмов")


if __name__ == '__main__':
    asyncio.run(ingest_all_movies())