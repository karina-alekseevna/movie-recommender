import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { moviesApi } from '../api/client';
import ReviewForm from '../components/ReviewForm';

// Функция для подмены URL на проксированный (обход блокировки TMDB)
const proxyPoster = (url) => {
  if (!url) return null;
  if (url.startsWith('https://image.tmdb.org')) {
    return `/api/images/proxy?url=${encodeURIComponent(url)}`;
  }
  return url;
};

export default function MoviePage({ userId }) {
  const { id } = useParams();
  const [movie, setMovie] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMovie = async () => {
      try {
        const res = await moviesApi.getMovie(id);
        setMovie(res.data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchMovie();
  }, [id]);

  if (loading) return <p>Загрузка...</p>;
  if (!movie) return <p>Фильм не найден</p>;

  // Проксируем постер
  const posterUrl = proxyPoster(movie.poster_url);

  return (
    <div className="movie-page">
      <div className="movie-header">
        <div className="poster-large">
          {posterUrl ? (
            <img src={posterUrl} alt={movie.title} />
          ) : (
            <div className="no-poster large">Нет обложки</div>
          )}
        </div>
        <div className="movie-details">
          <h1>{movie.title}</h1>
          {movie.original_title && movie.original_title !== movie.title && (
            <p className="original-title">{movie.original_title}</p>
          )}
          <div className="meta">
            {movie.release_year && <span>{movie.release_year}</span>}
            {movie.runtime && <span>{movie.runtime} мин</span>}
            {movie.rating && <span>★ {movie.rating}</span>}
          </div>
          {movie.genres_raw && <p className="genres">{movie.genres_raw.replace(/\|/g, ', ')}</p>}
          {movie.overview && <p className="overview">{movie.overview}</p>}
          {movie.directors_raw && <p><strong>Режиссёр:</strong> {movie.directors_raw}</p>}
          {movie.cast_raw && (
            <p><strong>В ролях:</strong> {movie.cast_raw.split('|').slice(0, 5).join(', ')}</p>
          )}
        </div>
      </div>

      <div className="movie-review-section">
        <ReviewForm movieId={movie.id} userId={userId} />
      </div>
    </div>
  );
}