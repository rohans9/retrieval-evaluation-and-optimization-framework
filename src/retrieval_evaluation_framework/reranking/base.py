"""Reranking abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from retrieval_evaluation_framework.models import RetrievalResult


class BaseReranker(ABC):
    """Common interface implemented by every reranking strategy."""

    name: str

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int,
    ) -> list[RetrievalResult]:
        """Rerank a candidate list of retrieval results.

        Args:
            query: Original user query.
            results: Candidate results produced by a retriever.
            top_n: Maximum number of reranked results to return.

        Returns:
            Reranked results, ordered by descending relevance.
        """
