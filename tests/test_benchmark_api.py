"""Phase-3 FastAPI tests for benchmarking endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from retrieval_evaluation_framework.api.app import _PIPELINE_CACHE, app
from tests.benchmark_helpers import (
    write_benchmark_config,
    write_corpus,
    write_dataset,
)

CLIENT = TestClient(app)


def test_evaluate_benchmark_compare_and_experiment_endpoints(tmp_path: Path) -> None:
    _PIPELINE_CACHE.clear()
    config_path = write_benchmark_config(tmp_path, retriever="bm25")
    corpus_path = write_corpus(tmp_path)
    dataset_path = write_dataset(tmp_path)
    experiment_directory = tmp_path / "experiments"

    evaluate_response = CLIENT.post(
        "/evaluate",
        json={
            "config_path": str(config_path),
            "corpus_path": str(corpus_path),
            "dataset_path": str(dataset_path),
            "experiment_directory": str(experiment_directory),
        },
    )

    assert evaluate_response.status_code == 200
    benchmark_result = evaluate_response.json()
    experiment_id = benchmark_result["experiment"]["experiment_id"]

    benchmark_response = CLIENT.post(
        "/benchmark",
        json={
            "config_path": str(config_path),
            "corpus_path": str(corpus_path),
            "dataset_path": str(dataset_path),
            "mode": "sweep",
            "parameters": {"retrieval.top_k": [1, 2]},
            "experiment_directory": str(experiment_directory),
        },
    )

    assert benchmark_response.status_code == 200
    assert len(benchmark_response.json()["results"]) == 2

    experiments_response = CLIENT.get(
        "/experiments",
        params={
            "config_path": str(config_path),
            "experiment_directory": str(experiment_directory),
        },
    )

    assert experiments_response.status_code == 200
    assert len(experiments_response.json()) >= 3

    experiment_response = CLIENT.get(
        f"/experiment/{experiment_id}",
        params={
            "config_path": str(config_path),
            "experiment_directory": str(experiment_directory),
        },
    )

    assert experiment_response.status_code == 200
    assert experiment_response.json()["experiment_id"] == experiment_id

    compare_response = CLIENT.post(
        "/compare",
        json={
            "config_path": str(config_path),
            "experiment_directory": str(experiment_directory),
            "experiment_ids": [experiment_id],
        },
    )

    assert compare_response.status_code == 200
    assert len(compare_response.json()["rows"]) == 1


def test_phase4_benchmarking_endpoints(tmp_path: Path) -> None:
    _PIPELINE_CACHE.clear()
    config_path = write_benchmark_config(tmp_path, retriever="hybrid")
    corpus_path = write_corpus(tmp_path)
    dataset_path = write_dataset(tmp_path)
    experiment_directory = tmp_path / "experiments"

    evaluate_response = CLIENT.post(
        "/evaluate",
        json={
            "config_path": str(config_path),
            "corpus_path": str(corpus_path),
            "dataset_path": str(dataset_path),
            "experiment_directory": str(experiment_directory),
        },
    )
    assert evaluate_response.status_code == 200

    leaderboard_response = CLIENT.get(
        "/leaderboard",
        params={
            "config_path": str(config_path),
            "experiment_directory": str(experiment_directory),
        },
    )
    recommendation_response = CLIENT.get(
        "/recommendation",
        params={
            "config_path": str(config_path),
            "experiment_directory": str(experiment_directory),
        },
    )
    reports_response = CLIENT.get(
        "/reports",
        params={
            "config_path": str(config_path),
            "experiment_directory": str(experiment_directory),
        },
    )
    visualizations_response = CLIENT.get(
        "/visualizations",
        params={
            "config_path": str(config_path),
            "experiment_directory": str(experiment_directory),
        },
    )
    history_response = CLIENT.get(
        "/history",
        params={
            "config_path": str(config_path),
            "experiment_directory": str(experiment_directory),
        },
    )

    assert leaderboard_response.status_code == 200
    assert recommendation_response.status_code == 200
    assert reports_response.status_code == 200
    assert visualizations_response.status_code == 200
    assert history_response.status_code == 200
    assert len(leaderboard_response.json()["rows"]) == 1
    assert "artifacts" in reports_response.json()
    assert "html_paths" in visualizations_response.json()
    assert len(history_response.json()) == 1


def test_leaderboard_validation_error_is_actionable(tmp_path: Path) -> None:
    _PIPELINE_CACHE.clear()
    config_path = write_benchmark_config(tmp_path, retriever="hybrid")
    corpus_path = write_corpus(tmp_path)
    dataset_path = write_dataset(tmp_path)
    experiment_directory = tmp_path / "experiments"

    evaluate_response = CLIENT.post(
        "/evaluate",
        json={
            "config_path": str(config_path),
            "corpus_path": str(corpus_path),
            "dataset_path": str(dataset_path),
            "experiment_directory": str(experiment_directory),
        },
    )
    assert evaluate_response.status_code == 200

    invalid_response = CLIENT.get(
        "/leaderboard",
        params={
            "config_path": str(config_path),
            "experiment_directory": str(experiment_directory),
            "sort_by": "invalid_metric",
        },
    )

    assert invalid_response.status_code == 400
    payload = invalid_response.json()
    assert "detail" in payload
    assert "suggestion" in payload
