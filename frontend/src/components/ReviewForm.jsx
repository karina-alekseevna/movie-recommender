import React, { useState, useCallback } from 'react';
import { reviewsApi } from '../api/client';
import debounce from 'lodash/debounce';

const SentimentBadge = ({ score }) => {
  const label = score > 0.3 ? 'Позитивный' : score < -0.3 ? 'Негативный' : 'Нейтральный';
  const cls = score > 0.3 ? 'positive' : score < -0.3 ? 'negative' : 'neutral';
  return <span className={`sentiment-badge ${cls}`}>{label} ({score.toFixed(2)})</span>;
};

export default function ReviewForm({ movieId, userId, onSuccess }) {
  const [text, setText] = useState('');
  const [preview, setPreview] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);

  const fetchPreview = useCallback(
    debounce(async (input) => {
      if (input.length < 20) {
        setPreview(null);
        return;
      }
      try {
        const res = await reviewsApi.analyzePreview(input, movieId);
        setPreview(res.data);
      } catch (err) {
        console.error(err);
      }
    }, 800),
    [movieId]
  );

  const handleChange = (e) => {
    const val = e.target.value;
    setText(val);
    fetchPreview(val);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (text.length < 20) return;
    setSubmitting(true);
    try {
      const res = await reviewsApi.submitReview(userId, movieId, text);
      setResult(res.data);
      if (onSuccess) onSuccess(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="review-form">
      <h3>Написать отзыв</h3>
      <form onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={handleChange}
          placeholder="Ваше мнение о фильме..."
          rows={6}
          minLength={20}
        />
        {preview && preview.segments && (
          <div className="analysis-preview">
            <h4>Предварительный анализ</h4>
            {preview.segments.map((seg, i) => (
              <div key={i} className="segment-preview">
                <p className="segment-text">«{seg.text}»</p>
                <div className="segment-meta">
                  <SentimentBadge score={seg.sentiment.score} />
                  {/* Сущности и общее впечатление не показываем */}
                </div>
              </div>
            ))}
          </div>
        )}
        <button type="submit" disabled={submitting || text.length < 20}>
          {submitting ? 'Отправка...' : 'Отправить отзыв'}
        </button>
      </form>
      {result && (
        <div className="review-submit-result">
          <h4>Отзыв принят</h4>
          <p>Общая тональность: <SentimentBadge score={result.overall_sentiment} /></p>
          <p>Затронуто сущностей: {result.entity_updates_count}</p>
        </div>
      )}
    </div>
  );
}