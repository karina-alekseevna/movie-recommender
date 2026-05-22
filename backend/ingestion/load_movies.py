# backend/ingestion/load_movies.py
import csv
import re
from typing import List, Dict


def parse_cast(cast_raw: str) -> List[Dict[str, str]]:
    if not cast_raw:
        return []

    actors = []
    for entry in cast_raw.split('|'):
        entry = entry.strip()
        # Ищем паттерн "Имя (Роль)"
        match = re.match(r'^(.+?)\s*\((.+?)\)\s*$', entry)
        if match:
            actors.append({
                'name': match.group(1).strip(),
                'character': match.group(2).strip()
            })
        else:
            actors.append({
                'name': entry,
                'character': ''
            })
    return actors


def parse_pipe_separated(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split('|') if item.strip()]


def load_movies_from_csv(filepath: str) -> List[Dict]:
    movies = []

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie = {
                'tmdb_id': int(row['tmdb_id']),
                'title': row['title'],
                'original_title': row['original_title'],
                'release_year': int(row['release_year']) if row['release_year'] else None,
                'release_date': row['release_date'] or None,
                'runtime': int(row['runtime']) if row['runtime'] else None,
                'overview': row['overview'],
                'rating': float(row['rating']) if row['rating'] else None,
                'rating_count': int(row['rating_count']) if row['rating_count'] else None,
                'popularity': float(row['popularity']) if row['popularity'] else None,
                'budget': int(row['budget']) if row['budget'] else None,
                'revenue': int(row['revenue']) if row['revenue'] else None,
                'poster_url': row['poster_url'],
                'backdrop_url': row['backdrop_url'],
                'tmdb_url': row['tmdb_url'],
                'imdb_id': row['imdb_id'],
                'original_language': row['original_language'],
                'genres_raw': row['genres'],
                'keywords_raw': row['keywords'],
                'cast_raw': row['cast'],
                'directors_raw': row['directors'],
                'writers_raw': row['writers'],
                'producers_raw': row.get('producers', ''),
                'genres': parse_pipe_separated(row['genres']),
                'keywords': parse_pipe_separated(row['keywords']),
                'cast': parse_cast(row['cast']),
                'directors': parse_pipe_separated(row['directors']),
                'writers': parse_pipe_separated(row['writers']),
                'producers': parse_pipe_separated(row.get('producers', '')),
            }
            movies.append(movie)
    return movies