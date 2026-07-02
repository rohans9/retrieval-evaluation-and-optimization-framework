"""Retriever abstractions shared by all retrieval strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from retrieval_evaluation_framework.models import Chunk, RetrievalResult


class BaseRetriever(ABC):
    """Common interface implemented by every retrieval strategy.

    New retrieval algorithms only need to implement this interface to be
    usable everywhere a retriever is expected, including the benchmark
    runner introduced in a later phase.
    """

    name: str

    @abstractmethod
    def build_index(self, chunks: list[Chunk]) -> None:
        """Build the retriever's index from a set of chunks.

        Args:
            chunks: Chunks to index.
        """

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: Query text.
            top_k: Optional override for the number of results to return.

        Returns:
            Ranked retrieval results.
        """

    @abstractmethod
    def get_configuration(self) -> dict[str, Any]:
        """Return the retriever's effective configuration.

        Returns:
            A JSON-serializable configuration snapshot.
        """

    @abstractmethod
    def save_index(self, directory: Path) -> None:
        """Persist the retriever's index to disk.

        Args:
            directory: Destination directory.
        """

    @abstractmethod
    def load_index(self, directory: Path) -> None:
        """Load a previously persisted index from disk.

        Args:
            directory: Directory containing a persisted index.
        """
