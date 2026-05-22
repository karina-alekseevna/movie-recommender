import React, { useState, useEffect } from 'react';
import { recommendationsApi } from '../api/client';
import MovieCard from '../components/MovieCard';

export default function RecommendationsPage({ userId }) {
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [limit, setLimit] = useState(20);

  useEffect(() => {
    const fetchRecs = async () => {
      setLoading(true);
      try {
        const res = await recommendationsApi.getRecommendations(userId, limit);
        setMovies(res.data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchRecs();
  }, [userId, limit]);

  return (
    <div className="recommendations-page">
      <h1>Персональные рекомендации</h1>
      <div className="controls">
        <label>
          Показать:{' '}
          <select value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
            <option value={10}>10</option>
            <option value={20}>20</option>
            <option value={50}>50</option>
          </select>
        </label>
      </div>
      {loading ? (
        <p>Подбираем фильмы...</p>
      ) : movies.length === 0 ? (
        <p>Напишите отзывы, чтобы получить персональные рекомендации.</p>
      ) : (
        <div className="movie-grid">
          {movies.map((m) => (
            <MovieCard key={m.id} movie={m} showScore />
          ))}
        </div>
      )}
    </div>
  );
}