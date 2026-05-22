import React, { useState } from 'react';
import { authApi } from '../api/client';

export default function Login({ onLogin }) {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmed = email.trim();
    if (!trimmed) return;
    setLoading(true);
    setError('');
    try {
      const response = await authApi.login(trimmed);
      onLogin(response.data);
    } catch (err) {
      setError('Ошибка входа. Проверьте соединение с сервером.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <h2>Вход в систему</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Введите ваш email"
          required
          className="login-input"
        />
        <button type="submit" disabled={loading} className="login-button">
          {loading ? 'Вход...' : 'Войти'}
        </button>
        {error && <p className="error-message">{error}</p>}
      </form>
      <p className="hint">Новый пользователь будет создан автоматически</p>
    </div>
  );
}