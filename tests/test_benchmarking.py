"""Phase-3 benchmarking, tracking, comparison, and ablation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from retrieval_evaluation_framework.benchmarking.analysis import BenchmarkAnalysisService
from retrieval_evaluation_framework.benchmarking.comparison import ExperimentComparisonEngine
from retrieval_evaluation_framework.benchmarking.models import (
    ExperimentRecord,
    PerformanceMetrics,
    RetrievalQualityMetrics,
)
from retrieval_evaluation_framework.benchmarking.runner import BenchmarkRunner
from retrieval_evaluation_framework.benchmarking.tracking import ExperimentTracker
from retrieval_evaluation_framework.config.settings import AppConfig
from retrieval_evaluation_framework.pipeline import DocumentProcessingPipeline
from retrieval_evaluation_framework.reranking.factory import RerankerFactory
from tests.benchmark_helpers import (
    write_benchmark_config,
    write_corpus,
    write_dataset,
)


def test_experiment_tracker_persists_and_loads_records(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    record = ExperimentRecord(
        experiment_id=tracker.next_experiment_id(),
        mode="single",
        status="completed",
        configuration={},
        dataset_path="dataset.json",
        corpus_path="corpus.json",
        chunking_strategy="recursive",
        chunk_size=128,
        embedding_model="local-hash",
        retriever="bm25",
        query_enhancement="none",
        reranker="none",
        top_k=2,
        selected_device="cpu",
        retrieval_quality_metrics=RetrievalQualityMetrics(
            precision_at_k=1.0,
            recall_at_k=1.0,
            mean_reciprocal_rank=1.0,
            ndcg_at_k=1.0,
        ),
        performance_metrics=PerformanceMetrics(
            average_retrieval_latency_ms=1.0,
            p50_latency_ms=1.0,
            p95_latency_ms=1.0,
            p99_latency_ms=1.0,
            embedding_generation_time_ms=0.0,
            index_build_time_ms=0.0,
        ),
    )

    tracker.create_experiment(record)

    assert tracker.get_experiment(record.experiment_id).experiment_id == record.experiment_id
    assert len(tracker.list_experiments()) == 1
    assert tracker.export_index().exists()


def test_benchmark_runner_single_sweep_grid_and_ablation(tmp_path: Path) -> None:
    config_path = write_benchmark_config(tmp_path, retriever="hybrid")
    corpus_path = write_corpus(tmp_path)
    dataset_path = write_dataset(tmp_path)
    config = AppConfig.load_yaml(config_path)
    tracker = ExperimentTracker(config.benchmark.experiment_directory)
    runner = BenchmarkRunner(tracker)

    single_result = runner.run_single_experiment(config, corpus_path, dataset_path)
    sweep_results = runner.parameter_sweep(
        config,
        corpus_path,
        dataset_path,
        sweep_parameters={"retrieval.top_k": [1, 2]},
    )
    grid_results = runner.grid_search(
        config,
        corpus_path,
        dataset_path,
        parameter_grid={"retrieval.top_k": [1, 2], "retrieval.retriever": ["bm25", "hybrid"]},
    )
    ablation_results = runner.ablation_study(
        config,
        corpus_path,
        dataset_path,
        variants={
            "remove_hybrid": {"retrieval.retriever": "bm25"},
            "remove_query_enhancement": {"query_enhancement.enabled": False},
        },
    )

    assert single_result.experiment.status == "completed"
    assert single_result.experiment.performance_metrics is not None
    assert single_result.experiment.performance_metrics.embedding_generation_time_ms >= 0.0
    assert single_result.experiment.performance_metrics.index_build_time_ms >= 0.0
    assert len(sweep_results) == 2
    assert len(grid_results) == 4
    assert len(ablation_results) == 3
    assert len(tracker.list_experiments()) == 10


def test_experiment_comparison_engine_returns_rows(tmp_path: Path) -> None:
    config_path = write_benchmark_config(tmp_path, retriever="bm25")
    corpus_path = write_corpus(tmp_path)
    dataset_path = write_dataset(tmp_path)
    config = AppConfig.load_yaml(config_path)
    tracker = ExperimentTracker(config.benchmark.experiment_directory)
    runner = BenchmarkRunner(tracker)

    first = runner.run_single_experiment(config, corpus_path, dataset_path)
    second = runner.run_single_experiment(config, corpus_path, dataset_path, notes="second")

    comparison = ExperimentComparisonEngine(tracker).compare(
        [first.experiment.experiment_id, second.experiment.experiment_id]
    )

    assert len(comparison.rows) == 2
    assert {row.experiment_id for row in comparison.rows} == {
        first.experiment.experiment_id,
        second.experiment.experiment_id,
    }


def test_analysis_service_generates_leaderboard_reports_and_visualizations(
    tmp_path: Path,
) -> None:
    config_path = write_benchmark_config(tmp_path, retriever="hybrid")
    corpus_path = write_corpus(tmp_path)
    dataset_path = write_dataset(tmp_path)
    config = AppConfig.load_yaml(config_path)
    tracker = ExperimentTracker(config.benchmark.experiment_directory)
    runner = BenchmarkRunner(tracker)

    runner.run_single_experiment(config, corpus_path, dataset_path)
    runner.parameter_sweep(
        config,
        corpus_path,
        dataset_path,
        sweep_parameters={"retrieval.top_k": [1, 2]},
    )

    service = BenchmarkAnalysisService(config, tracker)
    leaderboard = service.get_leaderboard()
    recommendation = service.get_recommendation()
    reports = service.generate_reports()
    visualizations = service.generate_visualizations()
    history = service.get_history()

    assert len(leaderboard.rows) == 3
    assert leaderboard.rows[0].rank == 1
    assert recommendation.experiment_id in reports
    assert Path(reports[recommendation.experiment_id]["markdown_path"]).exists()
    assert Path(visualizations.html_paths["leaderboard"]).exists()
    assert Path(visualizations.png_paths["quality_vs_latency"]).exists()
    assert history[0].experiment.summary is not None


def test_benchmark_runner_raises_for_misaligned_dataset_labels(tmp_path: Path) -> None:
    config_path = write_benchmark_config(tmp_path, retriever="hybrid")
    corpus_path = write_corpus(tmp_path)
    dataset_path = tmp_path / "dataset_misaligned.yaml"
    dataset_path.write_text(
        """
name: misaligned_fixture
examples:
  - query: What is the maternity leave policy?
    positive_documents:
      - unknown_doc:99
""".strip(),
        encoding="utf-8",
    )

    config = AppConfig.load_yaml(config_path)
    tracker = ExperimentTracker(config.benchmark.experiment_directory)
    runner = BenchmarkRunner(tracker)

    with pytest.raises(ValueError, match="Dataset labels do not match corpus identifiers"):
        runner.run_single_experiment(config, corpus_path, dataset_path)


def test_grid_search_rechunks_when_chunking_parameters_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_benchmark_config(tmp_path, retriever="bm25")
    corpus_path = write_corpus(tmp_path)
    dataset_path = write_dataset(tmp_path)
    config = AppConfig.load_yaml(config_path)
    tracker = ExperimentTracker(config.benchmark.experiment_directory)
    runner = BenchmarkRunner(tracker)

    observed_chunk_sizes: list[int] = []
    original_chunk_documents = DocumentProcessingPipeline.chunk_documents

    def spying_chunk_documents(self: DocumentProcessingPipeline, documents: list) -> list:
        observed_chunk_sizes.append(self.config.chunking.chunk_size)
        return original_chunk_documents(self, documents)

    monkeypatch.setattr(DocumentProcessingPipeline, "chunk_documents", spying_chunk_documents)

    runner.grid_search(
        config,
        corpus_path,
        dataset_path,
        parameter_grid={
            "chunking.chunk_size": [80, 120],
            "retrieval.top_k": [2],
        },
    )

    assert observed_chunk_sizes == [80, 120]


def test_parameter_sweep_applies_reranking_enabled_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_benchmark_config(tmp_path, retriever="bm25")
    corpus_path = write_corpus(tmp_path)
    dataset_path = write_dataset(tmp_path)
    config = AppConfig.load_yaml(config_path)
    tracker = ExperimentTracker(config.benchmark.experiment_directory)
    runner = BenchmarkRunner(tracker)

    observed_enabled_flags: list[bool] = []

    def spying_create(config):
        observed_enabled_flags.append(config.enabled)
        return None

    monkeypatch.setattr(RerankerFactory, "create", staticmethod(spying_create))

    runner.parameter_sweep(
        config,
        corpus_path,
        dataset_path,
        sweep_parameters={"reranking.enabled": [False, True]},
    )

    assert observed_enabled_flags == [False, True]
