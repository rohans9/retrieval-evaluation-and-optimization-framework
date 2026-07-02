"""Hybrid retriever combining lexical and dense retrieval via fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retrieval_evaluation_framework.config.settings import RetrievalConfig
from retrieval_evaluation_framework.models import Chunk, RetrievalResult
from retrieval_evaluation_framework.retrieval.base import BaseRetriever
from retrieval_evaluation_framework.retrieval.bm25_retriever import BM25Retriever
from retrieval_evaluation_framework.retrieval.dense_retriever import DenseRetriever
from retrieval_evaluation_framework.retrieval.fusion import FusionStrategy

_CANDIDATE_POOL_MULTIPLIER = 3


class HybridRetriever(BaseRetriever):
    """Retriever combining BM25 and dense retrieval results via fusion."""

    name = "hybrid"

    def __init__(
        self,
        config: RetrievalConfig,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        fusion: FusionStrategy,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            config: Retrieval configuration (used for the default Top-K).
            bm25_retriever: Lexical retriever component.
            dense_retriever: Dense retriever component.
            fusion: Strategy used to combine ranked lists.
        """
        self.config = config
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.fusion = fusion

    def build_index(self, chunks: list[Chunk]) -> None:
        """Build both the lexical and dense indexes."""
        self.bm25_retriever.build_index(chunks)
        self.dense_retriever.build_index(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve chunks using both strategies and fuse the results."""
        resolved_top_k = top_k or self.config.top_k
        candidate_pool_size = resolved_top_k * _CANDIDATE_POOL_MULTIPLIER
        bm25_results = self.bm25_retriever.retrieve(query, top_k=candidate_pool_size)
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_pool_size)
        return self.fusion.fuse([bm25_results, dense_results], top_k=resolved_top_k)

    def get_configuration(self) -> dict[str, Any]:
        """Return the retriever's effective configuration."""
        return {
            "retriever": self.name,
            "top_k": self.config.top_k,
            "fusion": self.fusion.name,
            "bm25": self.bm25_retriever.get_configuration(),
            "dense": self.dense_retriever.get_configuration(),
        }

    def save_index(self, directory: Path) -> None:
        """Persist both underlying indexes to disk."""
        self.bm25_retriever.save_index(_ensure_subdirectory(directory, "bm25"))
        self.dense_retriever.save_index(_ensure_subdirectory(directory, "dense"))

    def load_index(self, directory: Path) -> None:
        """Load both underlying indexes from disk."""
        self.bm25_retriever.load_index(directory / "bm25")
        self.dense_retriever.load_index(directory / "dense")


def _ensure_subdirectory(directory: Path, name: str) -> Path:
    subdirectory = directory / name
    subdirectory.mkdir(parents=True, exist_ok=True)
    return subdirectory
