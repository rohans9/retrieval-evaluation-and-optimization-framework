"""Generate markdown, CSV, and JSON reports from experiment history."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from retrieval_evaluation_framework.benchmarking.models import (
    ExperimentRecord,
    RecommendationResult,
    ReportArtifacts,
    TradeoffAnalysis,
)


class ReportGenerator:
    """Create report artifacts from enriched experiment records."""

    def __init__(self, output_directory: Path) -> None:
        """Initialize the report generator."""
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        experiment: ExperimentRecord,
        recommendation: RecommendationResult,
        tradeoff_analysis: TradeoffAnalysis,
    ) -> ReportArtifacts:
        """Generate markdown, CSV, and JSON reports for one experiment."""
        markdown_path = self.output_directory / f"{experiment.experiment_id}.md"
        csv_path = self.output_directory / f"{experiment.experiment_id}.csv"
        json_path = self.output_directory / f"{experiment.experiment_id}.json"

        markdown_path.write_text(
            self._markdown_report(experiment, recommendation, tradeoff_analysis),
            encoding="utf-8",
        )
        self._csv_report(csv_path, experiment, recommendation)
        json_path.write_text(
            json.dumps(
                {
                    "experiment": experiment.model_dump(mode="json"),
                    "recommendation": recommendation.model_dump(mode="json"),
                    "tradeoff_analysis": tradeoff_analysis.model_dump(mode="json"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return ReportArtifacts(
            markdown_path=str(markdown_path),
            csv_path=str(csv_path),
            json_path=str(json_path),
        )

    def _markdown_report(
        self,
        experiment: ExperimentRecord,
        recommendation: RecommendationResult,
        tradeoff_analysis: TradeoffAnalysis,
    ) -> str:
        quality = experiment.retrieval_quality_metrics
        performance = experiment.performance_metrics
        if quality is None or performance is None:
            raise ValueError("Experiment is missing benchmark metrics")

        summary = experiment.summary.summary if experiment.summary is not None else ""
        observations = "\n".join(
            f"- {observation.description}" for observation in tradeoff_analysis.observations
        ) or "- No trade-off observations available."
        alternative_configurations = ", ".join(recommendation.alternative_configurations) or "None"
        return "\n".join(
            [
                f"# Experiment Report: {experiment.experiment_id}",
                "",
                "## Executive Summary",
                "",
                summary,
                "",
                "## Experiment Configuration",
                "",
                f"- Chunking Strategy: {experiment.chunking_strategy}",
                f"- Chunk Size: {experiment.chunk_size}",
                f"- Embedding Model: {experiment.embedding_model}",
                f"- Retriever: {experiment.retriever}",
                f"- Query Enhancement: {experiment.query_enhancement}",
                f"- Reranker: {experiment.reranker}",
                f"- Top-K: {experiment.top_k}",
                f"- Selected Device: {experiment.selected_device}",
                "",
                "## Retrieval Quality",
                "",
                f"- Precision@K: {quality.precision_at_k:.4f}",
                f"- Recall@K: {quality.recall_at_k:.4f}",
                f"- MRR: {quality.mean_reciprocal_rank:.4f}",
                f"- NDCG@K: {quality.ndcg_at_k:.4f}",
                "",
                "## Performance",
                "",
                (
                    f"- Average Latency: {performance.average_retrieval_latency_ms:.2f} ms"
                ),
                f"- P50: {performance.p50_latency_ms:.2f} ms",
                f"- P95: {performance.p95_latency_ms:.2f} ms",
                f"- P99: {performance.p99_latency_ms:.2f} ms",
                (
                    f"- Embedding Time: {performance.embedding_generation_time_ms:.2f} ms"
                ),
                f"- Index Build Time: {performance.index_build_time_ms:.2f} ms",
                "",
                "## Benchmark Summary",
                "",
                f"- Current Rank: {experiment.leaderboard_rank}",
                f"- Overall Score: {experiment.overall_score:.4f}",
                "- Observations:",
                observations,
                "",
                "## Recommendation",
                "",
                f"- Recommended Pipeline: {recommendation.recommended_pipeline}",
                f"- Reason: {recommendation.reason}",
                f"- Alternative Configurations: {alternative_configurations}",
            ]
        )

    @staticmethod
    def _csv_report(
        path: Path,
        experiment: ExperimentRecord,
        recommendation: RecommendationResult,
    ) -> None:
        quality = experiment.retrieval_quality_metrics
        performance = experiment.performance_metrics
        if quality is None or performance is None:
            raise ValueError("Experiment is missing benchmark metrics")

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["field", "value"])
            writer.writerow(["experiment_id", experiment.experiment_id])
            writer.writerow(["chunking_strategy", experiment.chunking_strategy])
            writer.writerow(["chunk_size", experiment.chunk_size])
            writer.writerow(["embedding_model", experiment.embedding_model])
            writer.writerow(["retriever", experiment.retriever])
            writer.writerow(["query_enhancement", experiment.query_enhancement])
            writer.writerow(["reranker", experiment.reranker])
            writer.writerow(["top_k", experiment.top_k])
            writer.writerow(["selected_device", experiment.selected_device])
            writer.writerow(["precision_at_k", f"{quality.precision_at_k:.4f}"])
            writer.writerow(["recall_at_k", f"{quality.recall_at_k:.4f}"])
            writer.writerow(["mrr", f"{quality.mean_reciprocal_rank:.4f}"])
            writer.writerow(["ndcg_at_k", f"{quality.ndcg_at_k:.4f}"])
            writer.writerow(
                [
                    "average_latency_ms",
                    f"{performance.average_retrieval_latency_ms:.2f}",
                ]
            )
            writer.writerow(["p95_latency_ms", f"{performance.p95_latency_ms:.2f}"])
            writer.writerow(["overall_score", f"{experiment.overall_score:.4f}"])
            writer.writerow(["recommended_pipeline", recommendation.recommended_pipeline])
