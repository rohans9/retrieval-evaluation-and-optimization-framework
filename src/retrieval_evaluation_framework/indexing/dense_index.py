"""Dense vector index built on top of FAISS."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from retrieval_evaluation_framework.config.settings import DenseIndexConfig
from retrieval_evaluation_framework.indexing.base import BaseIndex
from retrieval_evaluation_framework.logging import get_logger

LOGGER = get_logger(component="indexing")


class DenseIndex(BaseIndex):
    """FAISS-backed dense vector index using cosine similarity.

    Cosine similarity is computed as an inner product over L2-normalized
    vectors, so both the indexed vectors and query vectors must already be
    normalized before being passed to this index.
    """

    def __init__(self, config: DenseIndexConfig, dimension: int) -> None:
        """Initialize the dense index.

        Args:
            config: Dense index configuration.
            dimension: Dimensionality of the embedding vectors.
        """
        self.config = config
        self.dimension = dimension
        self._index: faiss.Index | None = None
        self._chunk_ids: list[str] = []

    def build(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        """Build the FAISS index from chunk vectors.

        Args:
            chunk_ids: Chunk identifiers aligned with `vectors` rows.
            vectors: Normalized embedding vectors of shape `(n, dimension)`.
        """
        index = faiss.IndexFlatIP(self.dimension)
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._index = index
        self._chunk_ids = chunk_ids
        LOGGER.info("dense_index_built", chunk_count=len(chunk_ids), dimension=self.dimension)

    def rebuild(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        """Rebuild the index from a new set of chunk vectors.

        Args:
            chunk_ids: Chunk identifiers aligned with `vectors` rows.
            vectors: Normalized embedding vectors.
        """
        self.build(chunk_ids, vectors)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Search the index for the most similar chunk vectors.

        Args:
            query_vector: Normalized 1D query embedding.
            top_k: Maximum number of results to return.

        Returns:
            Chunk identifiers paired with cosine similarity scores, ordered
            by descending similarity.
        """
        if self._index is None:
            msg = "Dense index has not been built or loaded"
            raise RuntimeError(msg)

        query_matrix = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        top_k = min(top_k, len(self._chunk_ids))
        scores, indices = self._index.search(query_matrix, top_k)
        return [
            (self._chunk_ids[index], float(score))
            for score, index in zip(scores[0], indices[0], strict=True)
            if index != -1
        ]

    def save(self, directory: Path) -> None:
        """Persist the FAISS index and chunk id mapping to disk."""
        if self._index is None:
            msg = "Cannot save a dense index that has not been built"
            raise RuntimeError(msg)

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        metadata = {"chunk_ids": self._chunk_ids, "dimension": self.dimension}
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        LOGGER.info("dense_index_saved", directory=str(directory))

    def load(self, directory: Path) -> None:
        """Load a previously persisted FAISS index from disk."""
        self._index = faiss.read_index(str(directory / "index.faiss"))
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self._chunk_ids = metadata["chunk_ids"]
        self.dimension = metadata["dimension"]
        LOGGER.info(
            "dense_index_loaded",
            directory=str(directory),
            chunk_count=len(self._chunk_ids),
        )
