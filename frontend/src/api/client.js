import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || ''; 
const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
});

export const authApi = {
  login: (email) => api.post('/auth/login', { email }),
};

export const moviesApi = {
  getMovies: (page = 1, perPage = 20, search = '', sortBy = 'popularity', order = 'desc') =>
    api.get('/movies', { params: { page, per_page: perPage, search, sort_by: sortBy, order } }),
  getMovie: (id) => api.get(`/movies/${id}`),
};

export const reviewsApi = {
  submitReview: (userId, movieId, reviewText) =>
    api.post('/reviews', { user_id: userId, movie_id: movieId, review_text: reviewText }),
  getReview: (reviewId) => api.get(`/reviews/${reviewId}`),
  analyzePreview: (reviewText, movieId) =>
    api.get('/analyze-preview', { params: { review_text: reviewText, movie_id: movieId } }),
};

export const recommendationsApi = {
  getRecommendations: (userId, limit = 20) =>
    api.get(`/recommendations/${userId}`, { params: { limit } }),
};

export const profileApi = {
  getProfile: (userId) => api.get(`/profile/${userId}`),
};

export default api;