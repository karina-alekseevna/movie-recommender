// MovieCard.jsx
import React from 'react';
import { Link } from 'react-router-dom';

const proxyPoster = (url) => {
  if (!url) return null;
  if (url.startsWith('https://image.tmdb.org')) {
    return `/api/images/proxy?url=${encodeURIComponent(url)}`;
  }
  return url;
};

export default function MovieCard({ movie, showScore }) {
  const posterUrl = proxyPoster(movie.poster_url);

  return (
    <Link to={`/movie/${movie.id}`} className="movie-card">
      <div className="movie-poster">
        {posterUrl ? (
          <img src={posterUrl} alt={movie.title} />
        ) : (
          <div className="no-poster">Нет обложки</div>
        )}
        {showScore && movie.recommendation_score !== undefined && movie.recommendation_score > 0 && (
          <div className="rec-score">{Math.round(movie.recommendation_score * 100)}%</div>
        )}
      </div>
      <div className="movie-info">
        <h3>{movie.title}</h3>
        <div className="meta">
          {movie.release_year && <span className="year">{movie.release_year}</span>}
          {movie.rating && <span className="rating">★ {movie.rating}</span>}
        </div>
        {movie.genres_raw && (
          <p className="genres">{movie.genres_raw.replace(/\|/g, ', ')}</p>
        )}
      </div>
    </Link>
  );
}