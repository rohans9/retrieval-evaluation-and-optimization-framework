"""Phase-5 end-to-end interface workflow tests."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from retrieval_evaluation_framework.api.app import app

CLIENT = TestClient(app)


def _write_full_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "workflow_config.yaml"
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
                "  batch_size: 8",
                "retrieval:",
                "  retriever: hybrid",
                "  top_k: 3",
                "query_enhancement:",
                "  enabled: false",
                "  method: none",
                "reranking:",
                "  enabled: false",
                "benchmark:",
                f"  dataset_path: {(tmp_path / 'eval_dataset.yaml').as_posix()}",
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


def _write_eval_dataset(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "eval_dataset.yaml"
    dataset_path.write_text(
        """
name: e2e_dataset
examples:
  - query: What is the maternity leave policy?
    positive_documents:
            - sample
    negative_documents:
            - unrelated
  - query: Which policy mentions hybrid schedules?
    positive_documents:
            - sample
    negative_documents:
            - unrelated
""".strip(),
        encoding="utf-8",
    )
    return dataset_path


def test_end_to_end_workflow_from_ingestion_to_recommendation(
    tmp_path: Path,
    fixture_directory: Path,
) -> None:
    config_path = _write_full_config(tmp_path)
    dataset_path = _write_eval_dataset(tmp_path)

    ingest_response = CLIENT.post(
        "/ingest",
        json={
            "config_path": str(config_path),
            "source_path": str(fixture_directory),
        },
    )
    assert ingest_response.status_code == 200

    chunk_response = CLIENT.post(
        "/chunk",
        json={
            "config_path": str(config_path),
            "source_path": str(fixture_directory),
        },
    )
    assert chunk_response.status_code == 200
    corpus_path = Path(chunk_response.json()["output_path"])
    assert corpus_path.exists()

    benchmark_response = CLIENT.post(
        "/benchmark",
        json={
            "config_path": str(config_path),
            "corpus_path": str(corpus_path),
            "dataset_path": str(dataset_path),
            "mode": "single",
            "experiment_directory": str(tmp_path / "experiments"),
        },
    )
    assert benchmark_response.status_code == 200

    reports_response = CLIENT.get(
        "/reports",
        params={
            "config_path": str(config_path),
            "experiment_directory": str(tmp_path / "experiments"),
        },
    )
    leaderboard_response = CLIENT.get(
        "/leaderboard",
        params={
            "config_path": str(config_path),
            "experiment_directory": str(tmp_path / "experiments"),
        },
    )
    recommendation_response = CLIENT.get(
        "/recommendation",
        params={
            "config_path": str(config_path),
            "experiment_directory": str(tmp_path / "experiments"),
        },
    )

    assert reports_response.status_code == 200
    assert leaderboard_response.status_code == 200
    assert recommendation_response.status_code == 200
    assert len(leaderboard_response.json()["rows"]) >= 1
    assert recommendation_response.json()["experiment_id"]
