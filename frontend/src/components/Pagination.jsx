import React from 'react';

export default function Pagination({ page, totalPages, onPageChange }) {
  if (totalPages <= 1) return null;

  const pages = [];
  for (let i = 1; i <= totalPages; i++) {
    pages.push(i);
  }

  return (
    <div className="pagination">
      <button disabled={page === 1} onClick={() => onPageChange(page - 1)}>
        &laquo; Назад
      </button>
      {pages.map((p) => (
        <button
          key={p}
          className={p === page ? 'active' : ''}
          onClick={() => onPageChange(p)}
        >
          {p}
        </button>
      ))}
      <button disabled={page === totalPages} onClick={() => onPageChange(page + 1)}>
        Вперёд &raquo;
      </button>
    </div>
  );
}