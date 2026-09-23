# backend/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import httpx
import json

from backend.nlp.segmenter import segment_review
from backend.nlp.sentiment import sentiment_analyzer
from backend.nlp.entity_matcher import entity_matcher
from backend.nlp.embedder import embedder
from backend.profile.builder import profile_builder
from backend.recommender.engine import recommendation_engine
from backend.database.connection import get_supabase


app = FastAPI(
    title="Film Recommender API",
    description="Рекомендательный сервис на основе семантического анализа отзывов",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TMDB_IMAGE_BASE = "https://image.tmdb.org"


@app.get("/api/images/proxy")
async def proxy_image(url: str = Query(..., description="Полный URL изображения TMDB")):
    if not url.startswith(TMDB_IMAGE_BASE):
        raise HTTPException(status_code=403, detail="Проксирование разрешено только для image.tmdb.org")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Ошибка загрузки изображения: {str(e)}")
    return StreamingResponse(
        content=response.iter_bytes(),
        media_type=response.headers.get("content-type", "image/jpeg"),
        headers={"Cache-Control": "public, max-age=86400"}
    )


class ReviewRequest(BaseModel):
    user_id: str
    movie_id: int
    review_text: str


class ReviewResponse(BaseModel):
    review_id: int
    overall_sentiment: float
    overall_label: str
    segments_analysis: list
    entity_updates_count: int


class LoginRequest(BaseModel):
    email: str


def compute_overall_sentiment(analyzed_segments) -> tuple:
    if not analyzed_segments:
        return 0.0, 'neutral'

    weighted_sum = 0.0
    total_weight = 0.0

    for seg in analyzed_segments:
        text_len = max(len(seg.segment.text), 1)
        weight = text_len

        # Сегменты с высокой уверенностью дают больше вклада
        if seg.sentiment.confidence > 0.7:
            weight *= 1.2

        # Сегменты, в которых были найдены сущности, также немного весомее
        if seg.matched_entities:
            weight *= 1.1

        weighted_sum += seg.sentiment.score * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0, 'neutral'

    overall_score = weighted_sum / total_weight

    # Определяем лейбл
    if overall_score > 0.15:
        label = 'positive'
    elif overall_score < -0.15:
        label = 'negative'
    else:
        label = 'neutral'

    return round(overall_score, 3), label


@app.post("/api/auth/login")
async def login(request: LoginRequest):
    supabase = get_supabase()
    email = request.email.strip().lower()
    existing = supabase.table('users').select('*').eq('email', email).execute()
    if existing.data:
        user = existing.data[0]
        return {
            "user_id": user['id'],
            "email": user['email'],
            "username": user.get('username') or email.split('@')[0]
        }
    new_user = supabase.table('users').insert({
        "email": email,
        "username": email.split('@')[0] if '@' in email else email
    }).execute()
    user = new_user.data[0]
    return {
        "user_id": user['id'],
        "email": user['email'],
        "username": user.get('username')
    }


@app.get("/api/auth/me")
async def get_me(user_id: str):
    supabase = get_supabase()
    user = supabase.table('users').select('*').eq('id', user_id).execute()
    if not user.data:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    u = user.data[0]
    return {"user_id": u['id'], "email": u['email'], "username": u.get('username')}


@app.get("/api/movies")
async def get_movies(
    page: int = 1,
    per_page: int = 20,
    search: Optional[str] = None,
    sort_by: str = 'popularity',
    order: str = 'desc'
):
    supabase = get_supabase()
    valid_sort = ['popularity', 'rating', 'release_year']
    if sort_by not in valid_sort:
        sort_by = 'popularity'

    query = supabase.table('movies').select(
        'id, tmdb_id, title, original_title, release_year, '
        'overview, rating, rating_count, poster_url, genres_raw, directors_raw, popularity'
    )
    if search:
        query = query.ilike('title', f'%{search}%')
    query = query.order(sort_by, desc=(order.lower() == 'desc'))
    offset = (page - 1) * per_page
    query = query.range(offset, offset + per_page - 1)
    result = query.execute()
    return result.data


@app.get("/api/movies/{movie_id}")
async def get_movie(movie_id: int):
    supabase = get_supabase()
    movie = supabase.table('movies').select('*').eq('id', movie_id).execute()
    if not movie.data:
        raise HTTPException(status_code=404, detail="Фильм не найден")
    entities = supabase.table('movie_entities').select(
        'entity_type, entity_value'
    ).eq('movie_id', movie_id).execute()
    result = movie.data[0]
    result['entities'] = entities.data
    result.pop('embedding', None)
    return result


@app.post("/api/reviews", response_model=ReviewResponse)
async def submit_review(request: ReviewRequest):
    supabase = get_supabase()

    movie = supabase.table('movies').select('id, embedding').eq(
        'id', request.movie_id
    ).execute()
    if not movie.data:
        raise HTTPException(status_code=404, detail="Фильм не найден")

    emb_val = movie.data[0]['embedding']
    if isinstance(emb_val, str):
        emb_val = json.loads(emb_val)
    movie_embedding = np.array(emb_val, dtype=np.float32)

    # Сегментация
    segments = segment_review(request.review_text)
    if not segments:
        raise HTTPException(status_code=400, detail="Отзыв слишком короткий")

    # Анализ тональности по сегментам
    sentiments = sentiment_analyzer.analyze_batch([s.text for s in segments])

    # Сопоставление сущностей
    analyzed_segments = entity_matcher.analyze_review(
        segments, sentiments, request.movie_id
    )
    all_movie_entities = entity_matcher.get_movie_entities(request.movie_id)

    # Общая тональность с учётом всех сигналов
    overall_sentiment, overall_label = compute_overall_sentiment(analyzed_segments)

    # Обновление профиля (теперь с корректной работой с сущностями)
    profile_builder.process_review(
        user_id=request.user_id,
        movie_id=request.movie_id,
        analyzed_segments=analyzed_segments,
        movie_embedding=movie_embedding,
        overall_sentiment=overall_sentiment,
        all_movie_entities=all_movie_entities  # <-- новый параметр
    )

    # Сохранение отзыва
    segments_data = [
        {
            'text': seg.segment.text,
            'sentiment': {
                'label': seg.sentiment.label,
                'score': seg.sentiment.score,
            },
            'entities': [
                {
                    'type': e.entity_type,
                    'value': e.entity_value,
                    'match_method': e.match_method,
                    'match_score': e.match_score,
                }
                for e in seg.matched_entities
            ],
            'is_general': seg.is_general,
        }
        for seg in analyzed_segments
    ]

    entity_updates_count = sum(
        1 for seg in analyzed_segments if seg.matched_entities
    )

    review_data = {
        'user_id': request.user_id,
        'movie_id': request.movie_id,
        'review_text': request.review_text,
        'overall_sentiment': overall_sentiment,
        'extracted_entities': {
            'overall_sentiment': overall_sentiment,
            'overall_label': overall_label,
            'segments': segments_data,
            'entity_updates_count': entity_updates_count,
        },
    }

    saved = supabase.table('reviews').insert(review_data).execute()

    return ReviewResponse(
        review_id=saved.data[0]['id'],
        overall_sentiment=overall_sentiment,
        overall_label=overall_label,
        segments_analysis=segments_data,
        entity_updates_count=entity_updates_count,
    )


@app.get("/api/reviews/{review_id}")
async def get_review(review_id: int):
    supabase = get_supabase()
    result = supabase.table('reviews').select('*').eq('id', review_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Отзыв не найден")
    return result.data[0]


@app.get("/api/recommendations/{user_id}")
async def get_recommendations(user_id: str, limit: int = 20):
    recommendations = recommendation_engine.get_recommendations(
        user_id=user_id, limit=limit
    )
    if not recommendations:
        supabase = get_supabase()
        popular = supabase.table('movies').select(
            'id, tmdb_id, title, original_title, release_year, '
            'overview, rating, poster_url, genres_raw'
        ).order('popularity', desc=True).limit(limit).execute()
        for m in popular.data:
            m['recommendation_score'] = 0
            m['explanation'] = (
                "Популярный фильм — напишите отзывы, "
                "чтобы получить персональные рекомендации"
            )
        return popular.data
    return recommendations


@app.get("/api/profile/{user_id}")
async def get_user_profile(user_id: str):
    supabase = get_supabase()
    prefs = supabase.table('user_entity_preferences').select(
        'entity_type, entity_value, weight, mention_count'
    ).eq('user_id', user_id).order('weight', desc=True).execute()

    grouped = {}
    for p in prefs.data:
        t = p['entity_type']
        if t not in grouped:
            grouped[t] = {'positive': [], 'negative': []}
        entry = {
            'value': p['entity_value'],
            'weight': round(p['weight'], 3),
            'mentions': p['mention_count'],
        }
        if p['weight'] > 0:
            grouped[t]['positive'].append(entry)
        else:
            grouped[t]['negative'].append(entry)

    vector_profile = supabase.table('user_vector_profiles').select(
        'reviews_count, updated_at'
    ).eq('user_id', user_id).execute()

    reviews = supabase.table('reviews').select(
        'id, movie_id, overall_sentiment, created_at, movies(title)'
    ).eq('user_id', user_id).order('created_at', desc=True).execute()

    return {
        'entity_preferences': grouped,
        'vector_profile': (
            vector_profile.data[0] if vector_profile.data else None
        ),
        'reviews_count': len(reviews.data),
        'recent_reviews': reviews.data[:10],
    }


@app.get("/api/analyze-preview")
async def analyze_preview(review_text: str, movie_id: int):
    segments = segment_review(review_text)
    if not segments:
        return {"segments": [], "overall": None, "message": "Текст слишком короткий"}

    sentiments = sentiment_analyzer.analyze_batch([s.text for s in segments])
    analyzed = entity_matcher.analyze_review(segments, sentiments, movie_id)

    overall_score, overall_label = compute_overall_sentiment(analyzed)

    return {
        "segments": [
            {
                "text": seg.segment.text,
                "sentiment": {
                    "label": seg.sentiment.label,
                    "score": round(seg.sentiment.score, 3),
                },
                "entities": [
                    {
                        "type": e.entity_type,
                        "value": e.entity_value,
                        "match_method": e.match_method,
                        "match_score": round(e.match_score, 3),
                    }
                    for e in seg.matched_entities
                ],
                "is_general": seg.is_general,
            }
            for seg in analyzed
        ],
        "overall": {
            "score": overall_score,
            "label": overall_label,
        }
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)