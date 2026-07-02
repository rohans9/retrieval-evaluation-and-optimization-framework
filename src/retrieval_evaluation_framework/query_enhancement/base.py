"""Query enhancement abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from retrieval_evaluation_framework.models import Chunk, QueryEnhancementResult


class QueryEnhancer(ABC):
    """Common interface implemented by every query enhancement technique."""

    method: str

    def fit(self, chunks: list[Chunk]) -> None:
        """Optionally prepare corpus-derived state before enhancing queries.

        Args:
            chunks: Indexed chunks. The default implementation is a no-op;
                techniques that need corpus statistics should override this.
        """
        return None

    @abstractmethod
    def enhance(self, query: str) -> QueryEnhancementResult:
        """Enhance a query.

        Args:
            query: Original user query.

        Returns:
            The enhancement result, including the text to search with.
        """
