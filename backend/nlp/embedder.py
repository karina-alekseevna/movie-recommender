# backend/nlp/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
from backend.config import EMBEDDING_MODEL_PATH


class Embedder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model = SentenceTransformer(EMBEDDING_MODEL_PATH)
        self.dimension = 384
        self._initialized = True
        print(f"Модель загружена: paraphrase-multilingual-MiniLM-L12-v2, dim={self.dimension}")

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100,
            batch_size=64
        )
        if single:
            return embeddings[0]
        return embeddings

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(np.dot(vec1, vec2))

    def average_vectors(self, vectors: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
        if not vectors:
            return np.zeros(self.dimension)

        vectors = np.array(vectors)
        if weights is not None:
            weights = np.array(weights).reshape(-1, 1)
            centroid = np.average(vectors, axis=0, weights=weights.flatten())
        else:
            centroid = np.mean(vectors, axis=0)

        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid


embedder = Embedder()