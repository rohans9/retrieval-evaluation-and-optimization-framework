"""Factory for constructing the configured reranker."""

from __future__ import annotations

from retrieval_evaluation_framework.config.settings import RerankingConfig
from retrieval_evaluation_framework.reranking.base import BaseReranker
from retrieval_evaluation_framework.reranking.cross_encoder import CrossEncoderReranker


class RerankerFactory:
    """Factory for reranker implementations."""

    @staticmethod
    def create(config: RerankingConfig) -> BaseReranker | None:
        """Create a reranker matching configuration.

        Args:
            config: Reranking configuration.

        Returns:
            A configured reranker, or `None` when reranking is disabled.
        """
        if not config.enabled:
            return None
        return CrossEncoderReranker(config)
