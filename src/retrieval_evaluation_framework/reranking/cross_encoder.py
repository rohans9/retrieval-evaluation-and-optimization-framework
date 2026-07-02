"""Cross-encoder reranking with a lexical-overlap offline fallback."""

from __future__ import annotations

from typing import Any

from retrieval_evaluation_framework.config.settings import RerankingConfig
from retrieval_evaluation_framework.logging import get_logger
from retrieval_evaluation_framework.models import RetrievalResult
from retrieval_evaluation_framework.reranking.base import BaseReranker
from retrieval_evaluation_framework.utils.tokenization import tokenize

LOGGER = get_logger(component="reranking")


class CrossEncoderReranker(BaseReranker):
    """Rerank candidates with a cross-encoder model.

    Falls back to a deterministic lexical-overlap score when the configured
    cross-encoder model cannot be loaded (for example, without network
    access), so the pipeline still functions end to end offline.
    """

    name = "cross_encoder"

    def __init__(self, config: RerankingConfig) -> None:
        """Initialize the reranker.

        Args:
            config: Reranking configuration.
        """
        self.config = config
        self._model: Any | None = None
        if config.backend != "lexical":
            self._model = self._load_model()

    def _load_model(self) -> Any | None:
        try:
            from sentence_transformers import CrossEncoder

            return CrossEncoder(self.config.model_name)
        except Exception as error:
            if self.config.backend == "cross_encoder":
                raise
            LOGGER.warning("cross_encoder_unavailable", error=str(error))
            return None

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int,
    ) -> list[RetrievalResult]:
        """Rerank candidates using the cross-encoder model or lexical fallback."""
        if not results:
            return results

        scores = self._score(query, results)
        ordered = sorted(zip(results, scores, strict=True), key=lambda item: item[1], reverse=True)
        return [
            result.model_copy(update={"score": score, "rank": rank})
            for rank, (result, score) in enumerate(ordered[:top_n], start=1)
        ]

    def _score(self, query: str, results: list[RetrievalResult]) -> list[float]:
        if self._model is not None:
            pairs = [(query, result.chunk.text) for result in results]
            return [float(score) for score in self._model.predict(pairs)]
        return [self._lexical_overlap_score(query, result.chunk.text) for result in results]

    @staticmethod
    def _lexical_overlap_score(query: str, text: str) -> float:
        query_terms = set(tokenize(query))
        if not query_terms:
            return 0.0
        text_terms = set(tokenize(text))
        return len(query_terms & text_terms) / len(query_terms)
