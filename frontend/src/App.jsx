import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import HomePage from './pages/HomePage';
import MoviePage from './pages/MoviePage';
import RecommendationsPage from './pages/RecommendationsPage';
import ProfilePage from './pages/ProfilePage';
import Login from './components/Login';
import './App.css';

function App() {
  const [user, setUser] = useState(null);

  const handleLogin = (userData) => {
    setUser(userData);
  };

  const handleLogout = () => {
    setUser(null);
  };

  if (!user) {
    return (
      <div className="app">
        <div className="login-page">
          <Login onLogin={handleLogin} />
        </div>
      </div>
    );
  }

  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <Link to="/" className="logo">ФильмИИ</Link>
          <div className="nav-links">
            <Link to="/">Каталог</Link>
            <Link to="/recommendations">Рекомендации</Link>
            <Link to="/profile">Профиль</Link>
            <span className="user-info">{user.email}</span>
            <button onClick={handleLogout} className="logout-btn">
              Выйти
            </button>
          </div>
        </nav>
        <main className="content">
          <Routes>
            <Route path="/" element={<HomePage userId={user.user_id} />} />
            <Route path="/movie/:id" element={<MoviePage userId={user.user_id} />} />
            <Route path="/recommendations" element={<RecommendationsPage userId={user.user_id} />} />
            <Route path="/profile" element={<ProfilePage userId={user.user_id} />} />
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;