# backend/recommender/engine.py
import numpy as np
import json
from typing import List, Dict, Tuple, Set
from backend.database.connection import get_supabase


class RecommendationEngine:
    ENTITY_WEIGHT = 0.45
    VECTOR_WEIGHT = 0.25
    POPULARITY_WEIGHT = 0.1
    GENRE_WEIGHT = 0.2

    ENTITY_MIN_SCORE = 0.0
    GENRE_MIN_WEIGHT = 0.0

    def __init__(self):
        self.supabase = get_supabase()

    def get_liked_genres(self, user_id: str) -> List[Tuple[str, float]]:
        prefs = self.supabase.table('user_entity_preferences').select(
            'entity_normalized, weight'
        ).eq('user_id', user_id).eq('entity_type', 'genre').gt(
            'weight', self.GENRE_MIN_WEIGHT
        ).order('weight', desc=True).execute()

        return [(p['entity_normalized'], p['weight']) for p in (prefs.data or [])]

    def get_disliked_genres(self, user_id: str) -> Set[str]:
        prefs = self.supabase.table('user_entity_preferences').select(
            'entity_normalized'
        ).eq('user_id', user_id).eq('entity_type', 'genre').lt(
            'weight', 0
        ).execute()
        return {p['entity_normalized'] for p in (prefs.data or [])}

    def get_genre_recommendations(
        self, user_id: str, limit: int = 100, exclude_movie_ids: List[int] = None
    ) -> List[Tuple[int, float]]:
        liked_genres = self.get_liked_genres(user_id)
        disliked_genres = self.get_disliked_genres(user_id)

        if not liked_genres:
            return []

        movie_scores: Dict[int, float] = {}
        genre_movies: Dict[str, List[int]] = {}

        for genre, _ in liked_genres:
            movies = self.supabase.table('movie_entities').select('movie_id').eq(
                'entity_type', 'genre'
            ).eq('entity_normalized', genre).execute()
            genre_movies[genre] = [r['movie_id'] for r in (movies.data or [])]

        for genre, weight in liked_genres:
            for mid in genre_movies.get(genre, []):
                if exclude_movie_ids and mid in exclude_movie_ids:
                    continue
                movie_scores[mid] = movie_scores.get(mid, 0.0) + weight

        # Штраф для отрицательных жанров
        for genre in disliked_genres:
            movies = self.supabase.table('movie_entities').select('movie_id').eq(
                'entity_type', 'genre'
            ).eq('entity_normalized', genre).execute()
            for row in (movies.data or []):
                mid = row['movie_id']
                if mid in movie_scores:
                    movie_scores[mid] -= 0.15  # умеренный штраф

        # Оставляем только положительные скоры (даже очень маленькие)
        movie_scores = {mid: s for mid, s in movie_scores.items() if s > 0}
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_movies[:limit]

    def get_entity_recommendations(
        self, user_id: str, limit: int = 30, exclude_movie_ids: List[int] = None
    ) -> List[Tuple[int, float]]:
        pos_prefs = self.supabase.table('user_entity_preferences').select(
            'entity_type, entity_normalized, weight, mention_count'
        ).eq('user_id', user_id).gt('weight', 0).execute()

        movie_scores: Dict[int, float] = {}

        for pref in (pos_prefs.data or []):
            movies_with_entity = self.supabase.table('movie_entities').select(
                'movie_id'
            ).eq('entity_type', pref['entity_type']).eq(
                'entity_normalized', pref['entity_normalized']
            ).execute()
            contribution = (pref['weight'] ** 2) * np.log1p(pref['mention_count'])
            for row in (movies_with_entity.data or []):
                mid = row['movie_id']
                if exclude_movie_ids and mid in exclude_movie_ids:
                    continue
                movie_scores[mid] = movie_scores.get(mid, 0.0) + contribution

        # Отрицательные сущности – теперь все, у кого weight < 0
        neg_prefs = self.supabase.table('user_entity_preferences').select(
            'entity_type, entity_normalized, weight'
        ).eq('user_id', user_id).lt('weight', 0).execute()

        for pref in (neg_prefs.data or []):
            movies_with_entity = self.supabase.table('movie_entities').select(
                'movie_id'
            ).eq('entity_type', pref['entity_type']).eq(
                'entity_normalized', pref['entity_normalized']
            ).execute()
            penalty = pref['weight'] * 2.0   # weight отрицательный, penalty отрицательный
            for row in (movies_with_entity.data or []):
                mid = row['movie_id']
                if exclude_movie_ids and mid in exclude_movie_ids:
                    continue
                movie_scores[mid] = movie_scores.get(mid, 0.0) + penalty

        # Убираем фильмы с отрицательным или нулевым скором (оставляем только > 0)
        movie_scores = {mid: s for mid, s in movie_scores.items() if s > self.ENTITY_MIN_SCORE}
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_movies[:limit]

    def get_vector_recommendations(
        self, user_id: str, limit: int = 30, exclude_movie_ids: List[int] = None
    ) -> List[Tuple[int, float]]:
        profile = self.supabase.table('user_vector_profiles').select('*').eq(
            'user_id', user_id
        ).execute()

        if not profile.data or not profile.data[0].get('overall_centroid'):
            return []

        overall_centroid_raw = profile.data[0]['overall_centroid']
        if isinstance(overall_centroid_raw, str):
            overall_centroid_raw = json.loads(overall_centroid_raw)
        overall_centroid = np.array(overall_centroid_raw, dtype=np.float32)

        negative_centroid_raw = profile.data[0].get('negative_centroid')
        negative_centroid = None
        if negative_centroid_raw:
            if isinstance(negative_centroid_raw, str):
                negative_centroid_raw = json.loads(negative_centroid_raw)
            negative_centroid = np.array(negative_centroid_raw, dtype=np.float32)

        result = self.supabase.rpc('match_movies', {
            'query_embedding': overall_centroid.tolist(),
            'match_count': limit + len(exclude_movie_ids or []),
        }).execute()

        recommendations = []
        for row in (result.data or []):
            mid = row['id']
            if exclude_movie_ids and mid in exclude_movie_ids:
                continue

            similarity = row['similarity']
            similarity = max(0.0, min(1.0, float(similarity)))

            if negative_centroid is not None:
                emb_raw = row.get('embedding')
                if isinstance(emb_raw, str):
                    emb_raw = json.loads(emb_raw)
                movie_emb = np.array(emb_raw, dtype=np.float32)
                neg_sim = float(np.dot(movie_emb, negative_centroid))
                similarity = similarity - 0.5 * max(neg_sim, 0)

            recommendations.append((mid, similarity))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:limit]

    def get_recommendations(self, user_id: str, limit: int = 20) -> List[Dict]:
        reviewed = self.supabase.table('reviews').select('movie_id').eq(
            'user_id', user_id
        ).execute()
        exclude_ids = [r['movie_id'] for r in (reviewed.data or [])]

        entity_recs = self.get_entity_recommendations(user_id, limit=50, exclude_movie_ids=exclude_ids)
        vector_recs = self.get_vector_recommendations(user_id, limit=50, exclude_movie_ids=exclude_ids)
        genre_recs = self.get_genre_recommendations(user_id, limit=100, exclude_movie_ids=exclude_ids)

        def normalize_scores(recs):
            if not recs:
                return {}
            scores = [s for _, s in recs]
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return {mid: 1.0 for mid, _ in recs}
            return {mid: (s - min_s) / (max_s - min_s) for mid, s in recs}

        entity_scores_norm = normalize_scores(entity_recs)
        vector_scores_norm = normalize_scores(vector_recs)
        genre_scores_norm = normalize_scores(genre_recs)

        all_movie_ids = set()
        all_movie_ids.update(entity_scores_norm.keys())
        all_movie_ids.update(vector_scores_norm.keys())
        all_movie_ids.update(genre_scores_norm.keys())

        combined_scores: Dict[int, float] = {}
        for mid in all_movie_ids:
            e_score = entity_scores_norm.get(mid, 0.0)
            v_score = vector_scores_norm.get(mid, 0.0)
            g_score = genre_scores_norm.get(mid, 0.0)
            combined_scores[mid] = (
                self.ENTITY_WEIGHT * e_score +
                self.VECTOR_WEIGHT * v_score +
                self.GENRE_WEIGHT * g_score
            )

        # Добавляем популярность
        if combined_scores:
            movie_ids = list(combined_scores.keys())
            movies_data = self.supabase.table('movies').select(
                'id, rating'
            ).in_('id', movie_ids).execute()
            ratings = {m['id']: m['rating'] or 0 for m in (movies_data.data or [])}
            if ratings:
                max_rating = max(ratings.values())
                if max_rating > 0:
                    for mid in combined_scores:
                        pop = ratings.get(mid, 0) / max_rating
                        combined_scores[mid] += self.POPULARITY_WEIGHT * pop

        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        if not sorted_recs:
            return []

        top_ids = [mid for mid, _ in sorted_recs]
        movies = self.supabase.table('movies').select(
            'id, tmdb_id, title, original_title, release_year, overview, '
            'rating, poster_url, genres_raw, directors_raw'
        ).in_('id', top_ids).execute()
        movie_map = {m['id']: m for m in (movies.data or [])}

        result = []
        for mid, score in sorted_recs:
            movie = movie_map.get(mid)
            if movie:
                movie['recommendation_score'] = round(score, 4)
                movie['explanation'] = self._explain_recommendation(
                    user_id, mid,
                    entity_scores_norm.get(mid, 0),
                    vector_scores_norm.get(mid, 0),
                    genre_scores_norm.get(mid, 0)
                )
                result.append(movie)
        return result

    def _explain_recommendation(
        self, user_id, movie_id, entity_score, vector_score, genre_score
    ):
        reasons = []

        # Объяснение через жанры (показываем только сильные, чтобы не перегружать)
        if genre_score > 0.5:
            movie_genres = self.supabase.table('movie_entities').select(
                'entity_normalized'
            ).eq('movie_id', movie_id).eq('entity_type', 'genre').execute()
            movie_genre_set = {g['entity_normalized'] for g in (movie_genres.data or [])}

            liked_genres = self.get_liked_genres(user_id)
            matching_genres = [
                g for g, _ in liked_genres if g in movie_genre_set
            ]
            if matching_genres:
                reasons.append(f"Вам нравятся жанры: {', '.join(matching_genres[:3])}")

        # Объяснение через сильные сущности (вес > 0.3)
        if entity_score > 0.5:
            movie_entities = self.supabase.table('movie_entities').select(
                'entity_type, entity_normalized'
            ).eq('movie_id', movie_id).execute()

            user_prefs = self.supabase.table('user_entity_preferences').select(
                'entity_type, entity_normalized, entity_value, weight'
            ).eq('user_id', user_id).gt('weight', 0.3).execute()

            pref_dict = {(p['entity_type'], p['entity_normalized']): p for p in (user_prefs.data or [])}
            matching = []
            for ent in (movie_entities.data or []):
                key = (ent['entity_type'], ent['entity_normalized'])
                if key in pref_dict:
                    matching.append(pref_dict[key])
            matching.sort(key=lambda x: x['weight'], reverse=True)

            type_names = {
                'genre': 'жанр', 'actor': 'актёр', 'director': 'режиссёр',
                'keyword': 'тема', 'writer': 'сценарист'
            }
            for m in matching[:2]:
                tn = type_names.get(m['entity_type'], m['entity_type'])
                reasons.append(f"Вам нравится {tn} «{m['entity_value']}»")

        if vector_score > 0.5:
            reasons.append("Фильм похож на те, что вам понравились")
        if not reasons:
            reasons.append("Может вам понравиться на основе ваших предпочтений")
        return "; ".join(reasons)


recommendation_engine = RecommendationEngine()