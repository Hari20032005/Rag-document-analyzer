"""Embedding service with batching, retries, and simple in-memory caching."""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
from collections.abc import Iterable
from typing import Any

# Prevent broken-pipe errors from forked subprocesses inside uvicorn workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def _hash_embedding(text: str, dimension: int = 384) -> list[float]:
    """Deterministic fallback embedding used when model init fails."""

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [(digest[i % len(digest)] / 255.0) for i in range(dimension)]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


class EmbeddingService:
    """Generates embeddings for chunks and queries."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._cache: dict[str, list[float]] = {}
        self._model: Any | None = None
        self._fallback = False

    def _ensure_model(self) -> None:
        if self._model is not None or self._fallback:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.settings.embedding_model_name)
            logger.info("Loaded embedding model: %s", self.settings.embedding_model_name)
        except Exception as exc:
            self._fallback = True
            logger.warning("Embedding model load failed; using hash embeddings: %s", exc)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3), reraise=True)
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        self._ensure_model()
        if self._fallback or self._model is None:
            return [_hash_embedding(text) for text in batch]

        vectors = self._model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=len(batch),
        )
        return [vector.astype(float).tolist() for vector in vectors]

    def embed_texts(self, texts: Iterable[str], batch_size: int = 32) -> list[list[float]]:
        """Embed text list in batches with local cache."""

        items = list(texts)
        result: list[list[float]] = [list() for _ in items]
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for idx, text in enumerate(items):
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cached = self._cache.get(key)
            if cached is not None:
                result[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        for start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[start : start + batch_size]
            embeddings = self._embed_batch(batch)
            for offset, emb in enumerate(embeddings):
                original_idx = uncached_indices[start + offset]
                text = items[original_idx]
                key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                self._cache[key] = emb
                result[original_idx] = emb

        return result

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""

        return self.embed_texts([query], batch_size=1)[0]


embedding_service = EmbeddingService()
