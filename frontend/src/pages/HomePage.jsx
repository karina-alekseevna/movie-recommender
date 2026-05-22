import React, { useState, useEffect } from 'react';
import { moviesApi } from '../api/client';
import MovieCard from '../components/MovieCard';
import Pagination from '../components/Pagination';

export default function HomePage({ userId }) {
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [search, setSearch] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [sortBy, setSortBy] = useState('popularity');
  const [order, setOrder] = useState('desc');

  useEffect(() => {
    const fetch = async () => {
      setLoading(true);
      try {
        const res = await moviesApi.getMovies(page, 20, search, sortBy, order);
        setMovies(res.data.movies || res.data || []);
        setTotalPages(res.data.total_pages || 1);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetch();
  }, [page, search, sortBy, order]);

  const handleSearch = (e) => {
    e.preventDefault();
    setPage(1);
    setSearch(searchInput);
  };

  const handleSortChange = (field) => {
    if (field === sortBy) {
      setOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortBy(field);
      setOrder('desc');
    }
    setPage(1);
  };

  return (
    <div className="home-page">
      <h1>Каталог фильмов</h1>
      <div className="filters">
        <form onSubmit={handleSearch}>
          <input
            type="text"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            placeholder="Поиск по названию..."
          />
          <button type="submit">Найти</button>
        </form>
        <div className="sort-buttons">
          <span>Сортировка:</span>
          <button onClick={() => handleSortChange('popularity')} className={sortBy === 'popularity' ? 'active' : ''}>
            Популярность {sortBy === 'popularity' && (order === 'asc' ? '▲' : '▼')}
          </button>
          <button onClick={() => handleSortChange('rating')} className={sortBy === 'rating' ? 'active' : ''}>
            Рейтинг {sortBy === 'rating' && (order === 'asc' ? '▲' : '▼')}
          </button>
          <button onClick={() => handleSortChange('release_year')} className={sortBy === 'release_year' ? 'active' : ''}>
            Год {sortBy === 'release_year' && (order === 'asc' ? '▲' : '▼')}
          </button>
        </div>
      </div>

      {loading ? (
        <p>Загрузка...</p>
      ) : (
        <>
          <div className="movie-grid">
            {movies.map((m) => (
              <MovieCard key={m.id} movie={m} />
            ))}
          </div>
          <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />
        </>
      )}
    </div>
  );
}