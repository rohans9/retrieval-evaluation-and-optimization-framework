"""Fusion strategies for combining multiple ranked result lists."""

from __future__ import annotations

from abc import ABC, abstractmethod

from retrieval_evaluation_framework.models import Chunk, RetrievalResult


class FusionStrategy(ABC):
    """Common interface for combining ranked lists from multiple retrievers.

    New fusion strategies can be added by implementing this interface
    without changing `HybridRetriever`.
    """

    name: str

    @abstractmethod
    def fuse(self, ranked_lists: list[list[RetrievalResult]], top_k: int) -> list[RetrievalResult]:
        """Combine multiple ranked result lists into a single ranked list.

        Args:
            ranked_lists: Ranked results produced by individual retrievers.
            top_k: Maximum number of fused results to return.

        Returns:
            A single fused, ranked list of results.
        """


class ReciprocalRankFusion(FusionStrategy):
    """Combine ranked lists using Reciprocal Rank Fusion (RRF).

    RRF scores each chunk by ``sum(1 / (k + rank))`` across all ranked
    lists it appears in, rewarding chunks that rank highly in multiple
    retrieval strategies without requiring score normalization.
    """

    name = "reciprocal_rank_fusion"

    def __init__(self, k: int = 60) -> None:
        """Initialize the fusion strategy.

        Args:
            k: RRF constant that dampens the influence of high ranks.
        """
        self.k = k

    def fuse(self, ranked_lists: list[list[RetrievalResult]], top_k: int) -> list[RetrievalResult]:
        """Fuse ranked lists using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        chunk_by_id: dict[str, Chunk] = {}

        for ranked_list in ranked_lists:
            for result in ranked_list:
                chunk_id = result.chunk.chunk_id
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.k + result.rank)
                chunk_by_id[chunk_id] = result.chunk

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RetrievalResult(chunk=chunk_by_id[chunk_id], score=score, rank=rank, retriever="hybrid")
            for rank, (chunk_id, score) in enumerate(ordered, start=1)
        ]
