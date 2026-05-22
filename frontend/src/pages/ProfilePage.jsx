import React, { useState, useEffect } from 'react';
import { profileApi, reviewsApi, moviesApi } from '../api/client';  // <-- добавлен moviesApi

// Функция прокси для постеров (как в MovieCard)
const proxyPoster = (url) => {
  if (!url) return null;
  if (url.startsWith('https://image.tmdb.org')) {
    return `/api/images/proxy?url=${encodeURIComponent(url)}`;
  }
  return url;
};

// Компонент для одной сущности (повторяет логику из ReviewForm)
const EntityTag = ({ entity }) => {
  const typeLabels = {
    genre: 'Жанр',
    actor: 'Актёр',
    director: 'Режиссёр',
    keyword: 'Тема',
    writer: 'Сценарист',
  };
  return (
    <span className="entity-tag">
      {typeLabels[entity.type] || entity.type}: {entity.value}
    </span>
  );
};

// ==================== Компонент одного отзыва ====================
function ReviewItem({ review }) {
  const [showDetails, setShowDetails] = useState(false);
  const [fullReview, setFullReview] = useState(null);
  const [movieInfo, setMovieInfo] = useState(null); // { title, poster_url }

  const sentimentLabel =
    review.overall_sentiment > 0.3 ? 'Позитивный' :
    review.overall_sentiment < -0.3 ? 'Негативный' : 'Нейтральный';
  const sentimentClass =
    review.overall_sentiment > 0.3 ? 'positive' :
    review.overall_sentiment < -0.3 ? 'negative' : 'neutral';

  // Загружаем информацию о фильме, если её нет в отзыве
  useEffect(() => {
    const movieTitle = review.movie_title;
    const moviePoster = review.movie_poster_url;

    if (!movieTitle && !moviePoster && review.movie_id) {
      // Если нет ни названия, ни постера – возможно, API их не вернул, запрашиваем сами
      moviesApi.getMovie(review.movie_id)
        .then(res => setMovieInfo({
          title: res.data.title,
          poster_url: res.data.poster_url
        }))
        .catch(err => console.error(err));
    } else {
      setMovieInfo({
        title: movieTitle || null,
        poster_url: moviePoster || null
      });
    }
  }, [review]);

  const toggleDetails = async () => {
    if (!showDetails && !fullReview) {
      try {
        const res = await reviewsApi.getReview(review.id);
        setFullReview(res.data);
      } catch (err) {
        console.error(err);
      }
    }
    setShowDetails(!showDetails);
  };

  // Итоговые название и постер, используем загруженные или из отзыва
  const movieTitle = movieInfo?.title || 'Без названия';
  const posterUrl = proxyPoster(movieInfo?.poster_url);

  return (
    <div className="review-item">
      <div className="review-summary" onClick={toggleDetails}>
        {posterUrl && (
          <img
            src={posterUrl}
            alt={movieTitle}
            style={{
              width: '40px',
              height: '60px',
              objectFit: 'cover',
              borderRadius: '4px',
              marginRight: '8px'
            }}
          />
        )}
        <span className="movie-title">{movieTitle}</span>
        <span className={`sentiment-badge ${sentimentClass}`}>
          {sentimentLabel} ({review.overall_sentiment?.toFixed(2)})
        </span>
        <span className="date">
          {new Date(review.created_at).toLocaleDateString()}
        </span>
        <button className="toggle-details-btn">
          {showDetails ? 'Скрыть' : 'Подробнее'}
        </button>
      </div>
      {showDetails && fullReview && (
        <div className="review-details">
          <p className="review-text">{fullReview.review_text}</p>
          <div className="segments-analysis">
            {fullReview.extracted_entities?.segments?.map((seg, idx) => (
              <div key={idx} className="seg-item">
                <p className="seg-text">«{seg.text}»</p>
                <div className="seg-meta">
                  <span className={`sentiment-badge ${
                    seg.sentiment.score > 0.3 ? 'positive' :
                    seg.sentiment.score < -0.3 ? 'negative' : 'neutral'
                  }`}>
                    {seg.sentiment.label} ({seg.sentiment.score.toFixed(2)})
                  </span>
                  {seg.entities?.length > 0 && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '4px' }}>
                      {seg.entities.map((ent, j) => (
                        <EntityTag key={j} entity={ent} />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


// ==================== Компонент предпочтений (сворачиваемый) ====================
function Preferences({ preferences }) {
  const [open, setOpen] = useState(false);
  const typeNames = {
    genre: 'Жанры',
    actor: 'Актёры',
    director: 'Режиссёры',
    keyword: 'Темы',
    writer: 'Сценаристы',
  };

  return (
    <div className="preferences-card">
      <div className="pref-header" onClick={() => setOpen(!open)}>
        <h3>Предпочтения</h3>
        <span className="toggle-icon">{open ? '−' : '+'}</span>
      </div>
      {open && (
        <div className="pref-body">
          {Object.entries(preferences || {}).map(([type, data]) => (
            <div key={type} className="pref-category">
              <h4>{typeNames[type] || type}</h4>
              {data.positive?.length > 0 && (
                <div className="pref-list positive">
                  <span className="pref-label">Нравится</span>
                  <ul>
                    {data.positive.slice(0, 10).map((p, i) => (
                      <li key={i}>{p.value} <span className="weight">+{p.weight.toFixed(2)}</span></li>
                    ))}
                  </ul>
                </div>
              )}
              {data.negative?.length > 0 && (
                <div className="pref-list negative">
                  <span className="pref-label">Не нравится</span>
                  <ul>
                    {data.negative.slice(0, 10).map((p, i) => (
                      <li key={i}>{p.value} <span className="weight">{p.weight.toFixed(2)}</span></li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ==================== Главная страница профиля ====================
export default function ProfilePage({ userId }) {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const res = await profileApi.getProfile(userId);
        setProfile(res.data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchProfile();
  }, [userId]);

  if (loading) return <p>Загрузка профиля...</p>;
  if (!profile) return <p>Профиль не найден</p>;

  return (
    <div className="profile-page">
      <h1>Ваш профиль</h1>

      {/* Статистика */}
      <div className="profile-stats card">
        <div className="stat-item">
          <span className="stat-number">{profile.reviews_count}</span>
          <span className="stat-label">написано отзывов</span>
        </div>
      </div>

      {/* История отзывов */}
      <section className="reviews-section card">
        <h2>Последние отзывы</h2>
        {profile.recent_reviews?.length > 0 ? (
          profile.recent_reviews.map((rev) => (
            <ReviewItem key={rev.id} review={rev} />
          ))
        ) : (
          <p className="empty-text">Пока нет ни одного отзыва</p>
        )}
      </section>

      {/* Предпочтения (сворачиваемый блок) */}
      <Preferences preferences={profile.entity_preferences} />
    </div>
  );
};