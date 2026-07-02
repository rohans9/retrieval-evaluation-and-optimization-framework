"""Device-aware embedding generation engine.

The rest of the framework interacts only with `EmbeddingEngine`, never with a
specific embedding model, so swapping embedding models is a configuration
change rather than a code change.
"""

from __future__ import annotations

import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from retrieval_evaluation_framework.config.settings import EmbeddingConfig, resolve_device
from retrieval_evaluation_framework.embeddings.backends import build_backend
from retrieval_evaluation_framework.embeddings.cache import EmbeddingCache, hash_text
from retrieval_evaluation_framework.embeddings.store import EmbeddingStore
from retrieval_evaluation_framework.logging import get_logger
from retrieval_evaluation_framework.models import Chunk

LOGGER = get_logger(component="embeddings")


class EmbeddingEngine:
    """Generate, cache, and persist embeddings for chunks and queries."""

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize the embedding engine.

        Args:
            config: Embedding configuration.
        """
        self.config = config
        self.device = resolve_device(config.device)
        self.backend = build_backend(
            model_name=config.model_name,
            device=self.device,
            backend=config.backend,
            fallback_dimension=config.fallback_dimension,
        )
        self.cache = EmbeddingCache(config.cache_directory, model_name=self.backend.name)
        LOGGER.info(
            "embedding_engine_ready",
            model_name=config.model_name,
            backend=self.backend.name,
            device=self.device,
            dimension=self.backend.dimension,
        )

    @property
    def dimension(self) -> int:
        """Return the dimensionality of vectors produced by this engine."""
        return self.backend.dimension

    def embed(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        """Embed a batch of texts, reusing cached vectors where possible.

        Args:
            texts: Texts to embed.
            use_cache: Whether to read from and write to the embedding cache.

        Returns:
            A 2D array of shape ``(len(texts), dimension)``.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        keys = [hash_text(text, self.backend.name) for text in texts]
        cached = self.cache.get_many(keys) if use_cache else {}

        missing_indices = [index for index, key in enumerate(keys) if key not in cached]
        if missing_indices:
            missing_texts = [texts[index] for index in missing_indices]
            encoded = self._encode_with_progress(missing_texts)
            if use_cache:
                self.cache.put_many(
                    {
                        keys[index]: vector
                        for index, vector in zip(
                            missing_indices,
                            encoded,
                            strict=True,
                        )
                    }
                )
            for index, vector in zip(missing_indices, encoded, strict=True):
                cached[keys[index]] = vector

        vectors = np.asarray(np.vstack([cached[key] for key in keys]), dtype=np.float32)
        return self._normalize(vectors) if self.config.normalize_embeddings else vectors

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Args:
            query: Query text.

        Returns:
            A 1D embedding vector.
        """
        return np.asarray(self.embed([query])[0], dtype=np.float32)

    def embed_chunks(self, chunks: list[Chunk]) -> EmbeddingStore:
        """Embed a list of chunks into an ordered embedding store.

        Args:
            chunks: Chunks to embed.

        Returns:
            An embedding store aligned with the input chunk order.
        """
        vectors = self.embed([chunk.text for chunk in chunks])
        return EmbeddingStore(
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            vectors=vectors,
            model_name=self.backend.name,
            dimension=self.dimension,
        )

    def _encode_with_progress(self, texts: list[str]) -> np.ndarray:
        if not self.config.show_progress:
            return np.asarray(
                self.backend.encode(texts, self.config.batch_size, show_progress=False),
                dtype=np.float32,
            )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Embedding chunks", total=len(texts))

            def on_batch_complete(count: int) -> None:
                progress.update(task, advance=count)

            return np.asarray(
                self.backend.encode(
                    texts,
                    self.config.batch_size,
                    show_progress=True,
                    on_batch_complete=on_batch_complete,
                ),
                dtype=np.float32,
            )

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.asarray(vectors / norms, dtype=np.float32)
