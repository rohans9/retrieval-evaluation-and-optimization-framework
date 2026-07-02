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
