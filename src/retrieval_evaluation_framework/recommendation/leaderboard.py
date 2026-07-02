"""Experiment leaderboard generation."""

from __future__ import annotations

from typing import Literal

from retrieval_evaluation_framework.benchmarking.models import (
    ExperimentRecord,
    Leaderboard,
    LeaderboardRow,
)

SortableMetric = Literal[
    "mrr",
    "recall",
    "precision",
    "ndcg",
    "average_latency",
    "p95",
    "p99",
    "embedding_time",
    "index_build_time",
    "overall_score",
]

SORTABLE_METRICS: tuple[SortableMetric, ...] = (
    "mrr",
    "recall",
    "precision",
    "ndcg",
    "average_latency",
    "p95",
    "p99",
    "embedding_time",
    "index_build_time",
    "overall_score",
)


def parse_sort_metric(value: str) -> SortableMetric:
    """Validate and normalize a requested leaderboard sort metric."""
    if value not in SORTABLE_METRICS:
        msg = f"Unsupported leaderboard sort metric: {value}"
        raise ValueError(msg)
    return value


class LeaderboardEngine:
    """Build sortable leaderboards from experiment history."""

    def build(
        self,
        experiments: list[ExperimentRecord],
        sort_by: SortableMetric = "overall_score",
    ) -> Leaderboard:
        """Create a leaderboard sorted by the requested metric."""
        rows = [
            self._to_row(experiment)
            for experiment in experiments
            if self._is_rankable(experiment)
        ]
        reverse = sort_by not in {
            "average_latency",
            "p95",
            "p99",
            "embedding_time",
            "index_build_time",
        }
        rows = sorted(rows, key=lambda row: self._metric_value(row, sort_by), reverse=reverse)
        for index, row in enumerate(rows, start=1):
            row.rank = index
        return Leaderboard(rows=rows)

    @staticmethod
    def _is_rankable(experiment: ExperimentRecord) -> bool:
        return (
            experiment.retrieval_quality_metrics is not None
            and experiment.performance_metrics is not None
        )

    @staticmethod
    def _to_row(experiment: ExperimentRecord) -> LeaderboardRow:
        quality = experiment.retrieval_quality_metrics
        performance = experiment.performance_metrics
        if quality is None or performance is None:
            raise ValueError("Experiment is missing quality or performance metrics")
        return LeaderboardRow(
            experiment_id=experiment.experiment_id,
            retriever=experiment.retriever,
            embedding_model=experiment.embedding_model,
            precision_at_k=quality.precision_at_k,
            recall_at_k=quality.recall_at_k,
            mean_reciprocal_rank=quality.mean_reciprocal_rank,
            ndcg_at_k=quality.ndcg_at_k,
            average_latency_ms=performance.average_retrieval_latency_ms,
            p95_latency_ms=performance.p95_latency_ms,
            p99_latency_ms=performance.p99_latency_ms,
            embedding_time_ms=performance.embedding_generation_time_ms,
            index_build_time_ms=performance.index_build_time_ms,
            overall_score=experiment.overall_score or 0.0,
            rank=experiment.leaderboard_rank or 0,
        )

    @staticmethod
    def _metric_value(row: LeaderboardRow, sort_by: SortableMetric) -> float:
        metric_map = {
            "mrr": row.mean_reciprocal_rank,
            "recall": row.recall_at_k,
            "precision": row.precision_at_k,
            "ndcg": row.ndcg_at_k,
            "average_latency": row.average_latency_ms,
            "p95": row.p95_latency_ms,
            "p99": row.p99_latency_ms,
            "embedding_time": row.embedding_time_ms,
            "index_build_time": row.index_build_time_ms,
            "overall_score": row.overall_score,
        }
        return metric_map[sort_by]
