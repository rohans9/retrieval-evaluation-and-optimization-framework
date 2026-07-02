"""Structured models for benchmarking results and experiment history."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

ExperimentStatus = Literal["running", "completed", "failed"]
BenchmarkMode = Literal["single", "sweep", "grid_search", "ablation"]


class QueryMetrics(BaseModel):
    """Per-query retrieval quality metrics."""

    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


class RetrievalQualityMetrics(BaseModel):
    """Aggregated retrieval quality metrics for an evaluation run."""

    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float


class PerformanceMetrics(BaseModel):
    """System performance metrics measured during benchmarking."""

    average_retrieval_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    embedding_generation_time_ms: float
    index_build_time_ms: float


class RecommendationResult(BaseModel):
    """Explainable recommendation over benchmarked pipelines."""

    experiment_id: str
    recommended_pipeline: str
    reason: str
    alternative_configurations: list[str] = Field(default_factory=list)
    explainability: list[str] = Field(default_factory=list)
    overall_score: float


class TradeoffObservation(BaseModel):
    """One automatically derived trade-off observation."""

    title: str
    description: str


class TradeoffAnalysis(BaseModel):
    """Structured trade-off analysis over a set of experiments."""

    observations: list[TradeoffObservation]


class ExecutiveSummary(BaseModel):
    """Programmatic executive summary for a report or history entry."""

    summary: str


class QueryBenchmarkResult(BaseModel):
    """Benchmark output for a single dataset query."""

    query: str
    positive_documents: list[str]
    retrieved_chunk_ids: list[str]
    metrics: QueryMetrics
    latency_ms: float


class ExperimentRecord(BaseModel):
    """Structured experiment record persisted to local history."""

    experiment_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    mode: BenchmarkMode
    status: ExperimentStatus
    configuration: dict[str, Any]
    dataset_path: str
    corpus_path: str
    chunking_strategy: str
    chunk_size: int
    embedding_model: str
    retriever: str
    query_enhancement: str
    reranker: str
    top_k: int
    selected_device: str
    retrieval_quality_metrics: RetrievalQualityMetrics | None = None
    performance_metrics: PerformanceMetrics | None = None
    notes: str | None = None
    overall_score: float | None = None
    summary: ExecutiveSummary | None = None
    recommendation: RecommendationResult | None = None
    tradeoff_analysis: TradeoffAnalysis | None = None
    leaderboard_rank: int | None = None
    report_paths: dict[str, str] = Field(default_factory=dict)
    visualization_paths: dict[str, str] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    """Structured result returned by benchmark execution."""

    experiment: ExperimentRecord
    dataset_name: str
    sample_count: int
    query_results: list[QueryBenchmarkResult]

    def save_json(self, destination: Path) -> Path:
        """Persist the benchmark result as JSON.

        Args:
            destination: Output file path.

        Returns:
            The written file path.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return destination


class ComparisonRow(BaseModel):
    """Single row in an experiment comparison table."""

    experiment_id: str
    retriever: str
    embedding_model: str
    query_enhancement: str
    reranker: str
    mean_reciprocal_rank: float
    ndcg_at_k: float
    average_latency_ms: float


class ComparisonTable(BaseModel):
    """Structured comparison view over multiple experiments."""

    rows: list[ComparisonRow]


class LeaderboardRow(BaseModel):
    """Single row in an experiment leaderboard."""

    experiment_id: str
    retriever: str
    embedding_model: str
    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    embedding_time_ms: float
    index_build_time_ms: float
    overall_score: float
    rank: int


class Leaderboard(BaseModel):
    """Sortable experiment leaderboard."""

    rows: list[LeaderboardRow]


class ReportArtifacts(BaseModel):
    """Paths to generated report files for an experiment."""

    markdown_path: str
    csv_path: str
    json_path: str


class VisualizationArtifacts(BaseModel):
    """Paths to generated chart exports."""

    html_paths: dict[str, str] = Field(default_factory=dict)
    png_paths: dict[str, str] = Field(default_factory=dict)


class ExperimentHistoryEntry(BaseModel):
    """Complete history entry for phase-4 analysis outputs."""

    experiment: ExperimentRecord
    report_paths: dict[str, str]
    visualization_paths: dict[str, str]
