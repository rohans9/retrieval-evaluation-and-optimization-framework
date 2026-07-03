"""Phase-3 CLI integration tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from retrieval_evaluation_framework.cli.app import app
from tests.benchmark_helpers import (
    write_benchmark_config,
    write_corpus,
    write_dataset,
    write_parameter_file,
)

RUNNER = CliRunner()


def _write_cli_e2e_config(tmp_path: Path, dataset_path: Path) -> Path:
    config_path = tmp_path / "cli_workflow_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "device: cpu",
                "logging:",
                "  level: INFO",
                "ingestion:",
                "  recursive: true",
                "preprocessing:",
                "  normalize_unicode: true",
                "  cleanup_whitespace: true",
                "chunking:",
                "  strategy: recursive",
                "  chunk_size: 120",
                "  overlap: 20",
                "output:",
                f"  output_directory: {tmp_path.as_posix()}",
                "  processed_corpus_filename: processed_corpus.json",
                "embedding:",
                "  model_name: local-hash",
                "  backend: hashing",
                "  device: cpu",
                "retrieval:",
                "  retriever: hybrid",
                "  top_k: 3",
                "query_enhancement:",
                "  enabled: false",
                "  method: none",
                "reranking:",
                "  enabled: false",
                "benchmark:",
                f"  dataset_path: {dataset_path.as_posix()}",
                f"  experiment_directory: {(tmp_path / 'experiments').as_posix()}",
                f"  results_directory: {(tmp_path / 'benchmarks').as_posix()}",
                "reporting:",
                f"  reports_directory: {(tmp_path / 'generated_reports').as_posix()}",
                "visualization:",
                f"  output_directory: {(tmp_path / 'visualizations').as_posix()}",
                "recommendation:",
                "  quality_weight: 0.65",
                "  latency_weight: 0.2",
                "  embedding_cost_weight: 0.075",
                "  index_build_weight: 0.075",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_cli_e2e_dataset(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "cli_e2e_dataset.yaml"
    dataset_path.write_text(
        """
name: cli_e2e_dataset
examples:
  - query: What is the maternity leave policy?
    positive_documents:
            - sample
    negative_documents:
            - unrelated
""".strip(),
        encoding="utf-8",
    )
    return dataset_path


def test_cli_evaluate_and_experiments_commands(tmp_path: Path) -> None:
    config_path = write_benchmark_config(tmp_path, retriever="bm25")
    corpus_path = write_corpus(tmp_path)
    dataset_path = write_dataset(tmp_path)
    experiment_directory = tmp_path / "experiments"

    evaluate_result = RUNNER.invoke(
        app,
        [
            "retrieval",
            "evaluate",
            "--corpus-path",
            str(corpus_path),
            "--dataset-path",
            str(dataset_path),
            "--config",
            str(config_path),
            "--experiment-directory",
            str(experiment_directory),
        ],
    )

    assert evaluate_result.exit_code == 0

    experiments_result = RUNNER.invoke(
        app,
        [
            "retrieval",
            "experiments",
            "--config",
            str(config_path),
            "--experiment-directory",
            str(experiment_directory),
        ],
    )

    assert experiments_result.exit_code == 0


def test_cli_sweep_grid_search_and_compare_commands(tmp_path: Path) -> None:
    config_path = write_benchmark_config(tmp_path, retriever="bm25")
    corpus_path = write_corpus(tmp_path)
    dataset_path = write_dataset(tmp_path)
    parameters_path = write_parameter_file(tmp_path)
    experiment_directory = tmp_path / "experiments"

    sweep_result = RUNNER.invoke(
        app,
        [
            "retrieval",
            "sweep",
            "--corpus-path",
            str(corpus_path),
            "--dataset-path",
            str(dataset_path),
            "--parameters-path",
            str(parameters_path),
            "--config",
            str(config_path),
            "--experiment-directory",
            str(experiment_directory),
        ],
    )

    assert sweep_result.exit_code == 0

    grid_result = RUNNER.invoke(
        app,
        [
            "retrieval",
            "grid-search",
            "--corpus-path",
            str(corpus_path),
            "--dataset-path",
            str(dataset_path),
            "--parameters-path",
            str(parameters_path),
            "--config",
            str(config_path),
            "--experiment-directory",
            str(experiment_directory),
        ],
    )

    assert grid_result.exit_code == 0

    experiment_files = sorted(experiment_directory.glob("exp-*.json"))
    compare_result = RUNNER.invoke(
        app,
        [
            "retrieval",
            "compare",
            "--experiment-ids",
            ",".join(path.stem for path in experiment_files[:2]),
            "--config",
            str(config_path),
            "--experiment-directory",
            str(experiment_directory),
        ],
    )

    assert compare_result.exit_code == 0


def test_cli_phase4_commands(tmp_path: Path) -> None:
    config_path = write_benchmark_config(tmp_path, retriever="hybrid")
    corpus_path = write_corpus(tmp_path)
    dataset_path = write_dataset(tmp_path)
    experiment_directory = tmp_path / "experiments"

    evaluate_result = RUNNER.invoke(
        app,
        [
            "retrieval",
            "evaluate",
            "--corpus-path",
            str(corpus_path),
            "--dataset-path",
            str(dataset_path),
            "--config",
            str(config_path),
            "--experiment-directory",
            str(experiment_directory),
        ],
    )

    assert evaluate_result.exit_code == 0

    leaderboard_result = RUNNER.invoke(
        app,
        [
            "retrieval",
            "leaderboard",
            "--config",
            str(config_path),
            "--experiment-directory",
            str(experiment_directory),
        ],
    )
    recommend_result = RUNNER.invoke(
        app,
        [
            "retrieval",
            "recommend",
            "--config",
            str(config_path),
            "--experiment-directory",
            str(experiment_directory),
        ],
    )
    report_result = RUNNER.invoke(
        app,
        [
            "retrieval",
            "report",
            "--config",
            str(config_path),
            "--experiment-directory",
            str(experiment_directory),
        ],
    )
    visualize_result = RUNNER.invoke(
        app,
        [
            "retrieval",
            "visualize",
            "--config",
            str(config_path),
            "--experiment-directory",
            str(experiment_directory),
        ],
    )
    history_result = RUNNER.invoke(
        app,
        [
            "retrieval",
            "history",
            "--config",
            str(config_path),
            "--experiment-directory",
            str(experiment_directory),
        ],
    )

    assert leaderboard_result.exit_code == 0
    assert recommend_result.exit_code == 0
    assert report_result.exit_code == 0
    assert visualize_result.exit_code == 0
    assert history_result.exit_code == 0


def test_cli_end_to_end_from_ingestion_to_recommendation(
    tmp_path: Path,
    fixture_directory: Path,
) -> None:
    dataset_path = _write_cli_e2e_dataset(tmp_path)
    config_path = _write_cli_e2e_config(tmp_path, dataset_path)
    corpus_path = tmp_path / "processed_corpus.json"

    ingest_result = RUNNER.invoke(
        app,
        ["ingest", str(fixture_directory), "--config", str(config_path)],
    )
    chunk_result = RUNNER.invoke(
        app,
        ["chunk", str(fixture_directory), "--config", str(config_path)],
    )
    benchmark_result = RUNNER.invoke(
        app,
        [
            "retrieval",
            "benchmark",
            "--corpus-path",
            str(corpus_path),
            "--dataset-path",
            str(dataset_path),
            "--config",
            str(config_path),
            "--experiment-directory",
            str(tmp_path / "experiments"),
        ],
    )
    recommendation_result = RUNNER.invoke(
        app,
        [
            "retrieval",
            "recommend",
            "--config",
            str(config_path),
            "--experiment-directory",
            str(tmp_path / "experiments"),
        ],
    )

    assert ingest_result.exit_code == 0
    assert chunk_result.exit_code == 0
    assert benchmark_result.exit_code == 0
    assert recommendation_result.exit_code == 0
