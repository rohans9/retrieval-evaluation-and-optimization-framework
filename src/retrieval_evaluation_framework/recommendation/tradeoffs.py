"""Automatic trade-off analysis over benchmark experiments."""

from __future__ import annotations

from collections import defaultdict

from retrieval_evaluation_framework.benchmarking.models import (
    ExperimentRecord,
    TradeoffAnalysis,
    TradeoffObservation,
)


class TradeoffAnalyzer:
    """Derive explainable trade-off observations from experiment history."""

    def analyze(self, experiments: list[ExperimentRecord]) -> TradeoffAnalysis:
        """Generate structured trade-off observations.

        Args:
            experiments: Benchmark experiments to analyze.

        Returns:
            Structured trade-off analysis.
        """
        observations: list[TradeoffObservation] = []
        rankable = [
            experiment
            for experiment in experiments
            if experiment.retrieval_quality_metrics is not None
            and experiment.performance_metrics is not None
        ]
        if not rankable:
            return TradeoffAnalysis(observations=[])

        observations.extend(
            self._compare_component_groups(rankable, "retriever", "Retriever")
        )
        observations.extend(
            self._compare_component_groups(
                rankable,
                "query_enhancement",
                "Query enhancement",
            )
        )
        observations.extend(self._compare_component_groups(rankable, "reranker", "Reranker"))
        observations.extend(
            self._compare_component_groups(
                rankable,
                "chunking_strategy",
                "Chunking",
            )
        )
        return TradeoffAnalysis(observations=observations)

    def _compare_component_groups(
        self,
        experiments: list[ExperimentRecord],
        attribute: str,
        label: str,
    ) -> list[TradeoffObservation]:
        grouped: dict[str, list[ExperimentRecord]] = defaultdict(list)
        for experiment in experiments:
            grouped[str(getattr(experiment, attribute))].append(experiment)

        if len(grouped) < 2:
            return []

        averages: dict[str, tuple[float, float]] = {}
        for key, records in grouped.items():
            recall = sum(
                record.retrieval_quality_metrics.recall_at_k
                for record in records
                if record.retrieval_quality_metrics is not None
            ) / len(records)
            latency = sum(
                record.performance_metrics.p95_latency_ms
                for record in records
                if record.performance_metrics is not None
            ) / len(records)
            averages[key] = (recall, latency)

        best_quality = max(averages.items(), key=lambda item: item[1][0])
        best_latency = min(averages.items(), key=lambda item: item[1][1])

        observations = [
            TradeoffObservation(
                title=f"{label} quality leader",
                description=(
                    f"{best_quality[0]} achieved the strongest average Recall@K "
                    f"at {best_quality[1][0]:.3f}."
                ),
            )
        ]
        if best_quality[0] != best_latency[0]:
            observations.append(
                TradeoffObservation(
                    title=f"{label} latency trade-off",
                    description=(
                        f"{best_quality[0]} improved average Recall@K to {best_quality[1][0]:.3f} "
                        f"while {best_latency[0]} kept average P95 latency lowest at "
                        f"{best_latency[1][1]:.2f} ms."
                    ),
                )
            )
        return observations
