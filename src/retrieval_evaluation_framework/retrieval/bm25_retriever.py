"""BM25 lexical retriever."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retrieval_evaluation_framework.config.settings import BM25IndexConfig, RetrievalConfig
from retrieval_evaluation_framework.indexing.bm25_index import BM25Index
from retrieval_evaluation_framework.models import Chunk, RetrievalResult
from retrieval_evaluation_framework.retrieval.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """Retriever backed by a BM25 lexical index."""

    name = "bm25"

    def __init__(self, config: RetrievalConfig, index_config: BM25IndexConfig) -> None:
        """Initialize the BM25 retriever.

        Args:
            config: Retrieval configuration (used for the default Top-K).
            index_config: BM25-specific index configuration.
        """
        self.config = config
        self.index_config = index_config
        self._index = BM25Index(index_config)

    def build_index(self, chunks: list[Chunk]) -> None:
        """Build the BM25 index from chunks."""
        self._index.build(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve the most lexically similar chunks for a query."""
        resolved_top_k = top_k or self.config.top_k
        matches = self._index.search(query, resolved_top_k)
        return [
            RetrievalResult(chunk=chunk, score=score, rank=rank, retriever=self.name)
            for rank, (chunk, score) in enumerate(matches, start=1)
        ]

    def get_configuration(self) -> dict[str, Any]:
        """Return the retriever's effective configuration."""
        return {
            "retriever": self.name,
            "top_k": self.config.top_k,
            "k1": self.index_config.k1,
            "b": self.index_config.b,
        }

    def save_index(self, directory: Path) -> None:
        """Persist the BM25 index to disk."""
        directory = directory / "bm25"
        directory.mkdir(parents=True, exist_ok=True)
        self._index.save(directory)

    def load_index(self, directory: Path) -> None:
        """Load a previously persisted BM25 index from disk."""
        # self._index.load(directory)
        directory = directory / "bm25"
        self._index.load(directory)
