"""Disk-backed embedding cache.

Caching avoids re-encoding identical text when experimenting with the same
embedding model across multiple pipeline runs.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from retrieval_evaluation_framework.logging import get_logger

LOGGER = get_logger(component="embeddings")

_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def hash_text(text: str, model_name: str) -> str:
    """Compute a cache key for a piece of text under a specific model.

    Args:
        text: Source text.
        model_name: Embedding model identifier.

    Returns:
        A stable hexadecimal cache key.
    """
    return hashlib.sha256(f"{model_name}::{text}".encode()).hexdigest()


def _sanitize_model_name(model_name: str) -> str:
    return _SANITIZE_PATTERN.sub("_", model_name)


class EmbeddingCache:
    """Persist and retrieve embedding vectors keyed by content hash."""

    def __init__(self, cache_directory: Path, model_name: str) -> None:
        """Initialize the cache for a specific embedding model.

        Args:
            cache_directory: Root directory used for cache artifacts.
            model_name: Embedding model identifier used to namespace the cache.
        """
        self.cache_directory = cache_directory
        self.model_name = model_name
        self._keys_path = cache_directory / f"{_sanitize_model_name(model_name)}.keys.json"
        self._vectors_path = cache_directory / f"{_sanitize_model_name(model_name)}.npy"
        self._keys: list[str] = []
        self._vectors: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        if not (self._keys_path.exists() and self._vectors_path.exists()):
            return
        self._keys = json.loads(self._keys_path.read_text(encoding="utf-8"))
        self._vectors = np.load(self._vectors_path)
        LOGGER.info("embedding_cache_loaded", model_name=self.model_name, entries=len(self._keys))

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        """Fetch cached vectors for the given keys.

        Args:
            keys: Cache keys to look up.

        Returns:
            Mapping of cache key to cached vector for entries found.
        """
        if self._vectors is None:
            return {}
        index_by_key = {key: index for index, key in enumerate(self._keys)}
        return {key: self._vectors[index_by_key[key]] for key in keys if key in index_by_key}

    def put_many(self, entries: dict[str, np.ndarray]) -> None:
        """Persist new cache entries to disk, merging with existing entries.

        Args:
            entries: Mapping of cache key to embedding vector.
        """
        if not entries:
            return

        existing_keys = set(self._keys)
        new_keys = [key for key in entries if key not in existing_keys]
        if new_keys:
            new_vectors = np.stack([entries[key] for key in new_keys])
            if self._vectors is None:
                self._vectors = new_vectors
            else:
                self._vectors = np.vstack([self._vectors, new_vectors])
            self._keys.extend(new_keys)

        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self._keys_path.write_text(json.dumps(self._keys), encoding="utf-8")
        if self._vectors is not None:
            np.save(self._vectors_path, self._vectors)
        LOGGER.info(
            "embedding_cache_updated",
            model_name=self.model_name,
            new_entries=len(new_keys),
        )
