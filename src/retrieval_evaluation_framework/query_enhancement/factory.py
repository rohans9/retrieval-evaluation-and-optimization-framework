"""Factory for constructing the configured query enhancer."""

from __future__ import annotations

from retrieval_evaluation_framework.config.settings import QueryEnhancementConfig
from retrieval_evaluation_framework.embeddings.engine import EmbeddingEngine
from retrieval_evaluation_framework.query_enhancement.base import QueryEnhancer
from retrieval_evaluation_framework.query_enhancement.expansion import QueryExpander
from retrieval_evaluation_framework.query_enhancement.pipeline import SequentialQueryEnhancer


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
        methods = config.resolved_methods()
        if not methods:
            return None

        enhancers: list[QueryEnhancer] = []
        for method in methods:
            if method == "expansion":
                if embedding_engine is None:
                    msg = "Query expansion requires an embedding engine"
                    raise ValueError(msg)
                enhancers.append(QueryExpander(config.expansion, embedding_engine))
                continue

            msg = f"Unsupported query enhancement method: {method}"
            raise ValueError(msg)

        if len(enhancers) == 1:
            return enhancers[0]
        return SequentialQueryEnhancer(enhancers)
