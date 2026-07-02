"""Factory for constructing the configured query enhancer."""

from __future__ import annotations

from retrieval_evaluation_framework.config.settings import QueryEnhancementConfig
from retrieval_evaluation_framework.embeddings.engine import EmbeddingEngine
from retrieval_evaluation_framework.query_enhancement.base import QueryEnhancer
from retrieval_evaluation_framework.query_enhancement.expansion import QueryExpander
from retrieval_evaluation_framework.query_enhancement.hyde import HydeQueryEnhancer


class QueryEnhancerFactory:
    """Factory for query enhancement implementations."""

    @staticmethod
    def create(
        config: QueryEnhancementConfig,
        embedding_engine: EmbeddingEngine | None,
    ) -> QueryEnhancer | None:
        """Create a query enhancer matching configuration.

        Args:
            config: Query enhancement configuration.
            embedding_engine: Embedding engine required by embedding-based
                enhancement techniques such as query expansion.

        Returns:
            A configured query enhancer, or `None` when enhancement is disabled.

        Raises:
            ValueError: If query expansion is requested without an embedding engine.
        """
        if not config.enabled or config.method == "none":
            return None

        if config.method == "expansion":
            if embedding_engine is None:
                msg = "Query expansion requires an embedding engine"
                raise ValueError(msg)
            return QueryExpander(config.expansion, embedding_engine)

        if config.method == "hyde":
            return HydeQueryEnhancer(config.hyde)

        msg = f"Unsupported query enhancement method: {config.method}"
        raise ValueError(msg)
