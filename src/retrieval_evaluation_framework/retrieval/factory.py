"""Factory for constructing the configured retriever."""

from __future__ import annotations

from retrieval_evaluation_framework.config.settings import AppConfig
from retrieval_evaluation_framework.embeddings.engine import EmbeddingEngine
from retrieval_evaluation_framework.retrieval.base import BaseRetriever
from retrieval_evaluation_framework.retrieval.bm25_retriever import BM25Retriever
from retrieval_evaluation_framework.retrieval.dense_retriever import DenseRetriever
from retrieval_evaluation_framework.retrieval.fusion import ReciprocalRankFusion
from retrieval_evaluation_framework.retrieval.hybrid_retriever import HybridRetriever


class RetrieverFactory:
    """Factory for retriever implementations."""

    @staticmethod
    def create(config: AppConfig, embedding_engine: EmbeddingEngine | None) -> BaseRetriever:
        """Create a retriever matching configuration.

        Args:
            config: Application configuration.
            embedding_engine: Shared embedding engine, required by the dense
                and hybrid retrievers.

        Returns:
            A configured retriever instance.

        Raises:
            ValueError: If a dense or hybrid retriever is requested without
                an embedding engine, or the retriever type is unsupported.
        """
        retriever_type = config.retrieval.retriever

        if retriever_type == "bm25":
            return BM25Retriever(config.retrieval, config.index.bm25)

        if retriever_type == "dense":
            if embedding_engine is None:
                msg = "Dense retrieval requires an embedding engine"
                raise ValueError(msg)
            return DenseRetriever(config.retrieval, config.index.dense, embedding_engine)

        if retriever_type == "hybrid":
            if embedding_engine is None:
                msg = "Hybrid retrieval requires an embedding engine"
                raise ValueError(msg)
            bm25_retriever = BM25Retriever(config.retrieval, config.index.bm25)
            dense_retriever = DenseRetriever(config.retrieval, config.index.dense, embedding_engine)
            fusion = ReciprocalRankFusion(k=config.retrieval.fusion.rrf_k)
            return HybridRetriever(config.retrieval, bm25_retriever, dense_retriever, fusion)

        msg = f"Unsupported retriever: {retriever_type}"
        raise ValueError(msg)
