"""Independent retrieval quality metrics."""

from __future__ import annotations

import math
from statistics import mean

from retrieval_evaluation_framework.benchmarking.models import QueryMetrics, RetrievalQualityMetrics


class MetricEvaluator:
    """Compute per-query and aggregate retrieval quality metrics."""

    @staticmethod
    def precision_at_k(relevance_flags: list[bool], k: int) -> float:
        """Compute Precision@K.

        Args:
            relevance_flags: Ranked relevance labels.
            k: Cutoff rank.

        Returns:
            Precision at the requested cutoff.
        """
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if not relevance_flags:
            return 0.0
        truncated = relevance_flags[:k]
        return sum(1 for flag in truncated if flag) / k

    @staticmethod
    def recall_at_k(relevance_flags: list[bool], relevant_total: int, k: int) -> float:
        """Compute Recall@K.

        Args:
            relevance_flags: Ranked relevance labels.
            relevant_total: Number of relevant documents in the dataset truth.
            k: Cutoff rank.

        Returns:
            Recall at the requested cutoff.
        """
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if relevant_total <= 0:
            return 0.0
        truncated = relevance_flags[:k]
        return sum(1 for flag in truncated if flag) / relevant_total

    @staticmethod
    def reciprocal_rank(relevance_flags: list[bool]) -> float:
        """Compute reciprocal rank for a single ranked list."""
        for index, flag in enumerate(relevance_flags, start=1):
            if flag:
                return 1.0 / index
        return 0.0

    @staticmethod
    def ndcg_at_k(relevance_flags: list[bool], relevant_total: int, k: int) -> float:
        """Compute NDCG@K using binary relevance labels."""
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if relevant_total <= 0:
            return 0.0

        truncated = relevance_flags[:k]
        dcg = sum(
            (1.0 / math.log2(index + 1))
            for index, flag in enumerate(truncated, start=1)
            if flag
        )
        ideal_hits = min(relevant_total, k)
        idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
        if idcg == 0:
            return 0.0
        return dcg / idcg

    @classmethod
    def evaluate_query(
        cls,
        relevance_flags: list[bool],
        relevant_total: int,
        k: int,
    ) -> QueryMetrics:
        """Compute all required metrics for a single query."""
        return QueryMetrics(
            precision_at_k=cls.precision_at_k(relevance_flags, k),
            recall_at_k=cls.recall_at_k(relevance_flags, relevant_total, k),
            reciprocal_rank=cls.reciprocal_rank(relevance_flags),
            ndcg_at_k=cls.ndcg_at_k(relevance_flags, relevant_total, k),
        )

    @staticmethod
    def aggregate(query_metrics: list[QueryMetrics]) -> RetrievalQualityMetrics:
        """Aggregate per-query metrics into dataset-level averages."""
        if not query_metrics:
            return RetrievalQualityMetrics(
                precision_at_k=0.0,
                recall_at_k=0.0,
                mean_reciprocal_rank=0.0,
                ndcg_at_k=0.0,
            )
        return RetrievalQualityMetrics(
            precision_at_k=mean(metric.precision_at_k for metric in query_metrics),
            recall_at_k=mean(metric.recall_at_k for metric in query_metrics),
            mean_reciprocal_rank=mean(metric.reciprocal_rank for metric in query_metrics),
            ndcg_at_k=mean(metric.ndcg_at_k for metric in query_metrics),
        )
