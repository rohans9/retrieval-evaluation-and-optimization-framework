"""End-to-end retrieval pipeline orchestration.

`RetrievalPipeline` is the single entry point CLI commands and API endpoints
use to build indexes and answer queries, mirroring the role
`DocumentProcessingPipeline` plays for phase-1 corpus preparation.
"""

from __future__ import annotations

import time
from pathlib import Path

from retrieval_evaluation_framework.config.settings import AppConfig
from retrieval_evaluation_framework.embeddings.engine import EmbeddingEngine
from retrieval_evaluation_framework.logging import get_logger
from retrieval_evaluation_framework.models import Chunk, ProcessedCorpus, RetrievalResponse
from retrieval_evaluation_framework.query_enhancement.factory import QueryEnhancerFactory
from retrieval_evaluation_framework.reranking.factory import RerankerFactory
from retrieval_evaluation_framework.retrieval.factory import RetrieverFactory

LOGGER = get_logger(component="retrieval_pipeline")

_DENSE_RETRIEVERS = {"dense", "hybrid"}


class RetrievalPipeline:
    """Coordinate embedding, indexing, query enhancement, and reranking."""

    def __init__(self, config: AppConfig) -> None:
        """Initialize the retrieval pipeline.

        Args:
            config: Application configuration.
        """
        self.config = config

        needs_embeddings = config.retrieval.retriever in _DENSE_RETRIEVERS or (
            config.query_enhancement.enabled and config.query_enhancement.method == "expansion"
        )
        self.embedding_engine = EmbeddingEngine(config.embedding) if needs_embeddings else None
        self.retriever = RetrieverFactory.create(config, self.embedding_engine)
        self.query_enhancer = QueryEnhancerFactory.create(
            config.query_enhancement, self.embedding_engine
        )
        self.reranker = RerankerFactory.create(config.reranking)
        self._is_built = False

        LOGGER.info(
            "retrieval_pipeline_initialized",
            retriever=self.retriever.name,
            query_enhancement=(
                ",".join(self.config.query_enhancement.resolved_methods())
                if self.query_enhancer
                else None
            ),
            reranker=(
                type(self.reranker).__name__
                if self.reranker
                else None
            ),
        )

    def load_processed_corpus(self, path: Path) -> ProcessedCorpus:
        """Load a processed corpus produced by the phase-1 pipeline.

        Args:
            path: Path to a processed corpus JSON file.

        Returns:
            The parsed processed corpus.
        """
        return ProcessedCorpus.model_validate_json(path.read_text(encoding="utf-8"))

    def build_index(self, chunks: list[Chunk]) -> None:
        """Build the retriever's index (and any enhancer state) from chunks.

        Args:
            chunks: Chunks to index.
        """
        self.retriever.build_index(chunks)
        if self.query_enhancer is not None:
            self.query_enhancer.fit(chunks)
        self._is_built = True
        LOGGER.info("retrieval_index_built", chunk_count=len(chunks))

    def index_processed_corpus(self, path: Path) -> ProcessedCorpus:
        """Load a processed corpus and build the retriever's index from it.

        Args:
            path: Path to a processed corpus JSON file.

        Returns:
            The processed corpus that was indexed.
        """
        corpus = self.load_processed_corpus(path)
        self.build_index(corpus.chunks)
        return corpus

    def save_index(self, directory: Path) -> None:
        """Persist the retriever's index to disk.

        Args:
            directory: Destination directory.
        """
        directory.mkdir(parents=True, exist_ok=True)
        self.retriever.save_index(directory)
        LOGGER.info("retrieval_index_saved", directory=str(directory))

    def load_index(self, directory: Path) -> None:
        """Load a previously persisted index from disk.

        Args:
            directory: Directory containing a persisted index.
        """
        self.retriever.load_index(directory)
        self._is_built = True
        LOGGER.info("retrieval_index_loaded", directory=str(directory))

    @property
    def is_built(self) -> bool:
        """Return whether the retriever's index is ready to serve queries."""
        return self._is_built

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResponse:
        """Retrieve relevant chunks for a query.

        Args:
            query: User query text.
            top_k: Optional override for the number of results to return.

        Returns:
            The full retrieval response, including latency and provenance.
        """

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not self._is_built:
            msg = "The retrieval index has not been built or loaded"
            raise RuntimeError(msg)

        total_start = time.perf_counter()

        enhanced_query: str | None = None
        enhancement_method: str | None = None
        search_query = query
        enhancement_ms = 0.0
        if self.query_enhancer is not None:
            enhancement_start = time.perf_counter()
            enhancement = self.query_enhancer.enhance(query)
            enhancement_ms = (time.perf_counter() - enhancement_start) * 1000
            enhanced_query = enhancement.enhanced_query
            enhancement_method = enhancement.method
            search_query = enhancement.enhanced_query
            LOGGER.info("query_enhanced", method=enhancement_method, latency_ms=enhancement_ms)

        retrieval_start = time.perf_counter()
        results = self.retriever.retrieve(search_query, top_k=top_k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        LOGGER.info(
            "retrieval_completed",
            retriever=self.retriever.name,
            result_count=len(results),
            latency_ms=retrieval_ms,
        )

        reranked = False
        reranking_ms = 0.0
        if self.reranker is not None:
            reranking_start = time.perf_counter()
            results = self.reranker.rerank(query, results, top_n=self.config.reranking.top_n)
            reranking_ms = (time.perf_counter() - reranking_start) * 1000
            reranked = True
            LOGGER.info("reranking_completed", latency_ms=reranking_ms)

        total_ms = (time.perf_counter() - total_start) * 1000

        LOGGER.info(
            "retrieval_pipeline_summary",
            retriever=self.retriever.name,
            original_query=query,
            effective_query=search_query,
            top_k=top_k or self.config.retrieval.top_k,
            enhancement_ms=round(enhancement_ms, 2),
            retrieval_ms=round(retrieval_ms, 2),
            reranking_ms=round(reranking_ms, 2),
            total_ms=round(total_ms, 2),
        )

        return RetrievalResponse(
            query=query,
            enhanced_query=enhanced_query,
            query_enhancement_method=enhancement_method,
            retriever=self.retriever.name,
            reranked=reranked,
            results=results,
            retrieval_latency_ms=retrieval_ms,
            enhancement_latency_ms=enhancement_ms,
            reranking_latency_ms=reranking_ms,
            total_latency_ms=total_ms,
        )
