"""Persisted embedding matrix artifact.

An `EmbeddingStore` associates an ordered set of chunk identifiers with their
embedding vectors and can be written to, and read from, disk so that dense
indexes can be rebuilt without recomputing embeddings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class EmbeddingStore:
    """Ordered chunk embeddings produced by the embedding engine."""

    chunk_ids: list[str]
    vectors: np.ndarray
    model_name: str
    dimension: int

    def save(self, directory: Path) -> None:
        """Persist the embedding store to disk.

        Args:
            directory: Destination directory. Created if missing.
        """
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        metadata = {
            "chunk_ids": self.chunk_ids,
            "model_name": self.model_name,
            "dimension": self.dimension,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> EmbeddingStore:
        """Load an embedding store previously written with `save`.

        Args:
            directory: Directory containing the persisted artifact.

        Returns:
            The loaded embedding store.
        """
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        vectors = np.load(directory / "vectors.npy")
        return cls(
            chunk_ids=metadata["chunk_ids"],
            vectors=vectors,
            model_name=metadata["model_name"],
            dimension=metadata["dimension"],
        )
