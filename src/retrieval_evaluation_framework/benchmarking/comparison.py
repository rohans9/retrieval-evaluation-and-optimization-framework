"""Experiment comparison utilities."""

from __future__ import annotations

from retrieval_evaluation_framework.benchmarking.models import ComparisonRow, ComparisonTable
from retrieval_evaluation_framework.benchmarking.tracking import ExperimentTracker


class ExperimentComparisonEngine:
    """Build side-by-side views over stored experiments."""

    def __init__(self, tracker: ExperimentTracker) -> None:
        """Initialize the comparison engine.

        Args:
            tracker: Experiment tracker used to load persisted records.
        """
        self.tracker = tracker

    def compare(self, experiment_ids: list[str]) -> ComparisonTable:
        """Compare persisted experiments side by side.

        Args:
            experiment_ids: Experiment IDs to compare.

        Returns:
            Structured comparison table.
        """
        rows: list[ComparisonRow] = []
        for experiment_id in experiment_ids:
            record = self.tracker.get_experiment(experiment_id)
            quality = record.retrieval_quality_metrics
            performance = record.performance_metrics
            rows.append(
                ComparisonRow(
                    experiment_id=record.experiment_id,
                    retriever=record.retriever,
                    embedding_model=record.embedding_model,
                    query_enhancement=record.query_enhancement,
                    reranker=record.reranker,
                    mean_reciprocal_rank=quality.mean_reciprocal_rank if quality else 0.0,
                    ndcg_at_k=quality.ndcg_at_k if quality else 0.0,
                    average_latency_ms=(
                        performance.average_retrieval_latency_ms if performance else 0.0
                    ),
                )
            )
        return ComparisonTable(rows=rows)
