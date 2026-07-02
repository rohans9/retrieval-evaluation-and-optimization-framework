"""Phase-3 dataset and metric tests."""

from __future__ import annotations

from pathlib import Path

from retrieval_evaluation_framework.evaluation.datasets import JsonEvaluationDatasetLoader
from retrieval_evaluation_framework.evaluation.metrics import MetricEvaluator
from tests.benchmark_helpers import write_dataset


def test_json_evaluation_dataset_loader_and_split(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)

    dataset = JsonEvaluationDatasetLoader().load(dataset_path)
    train_dataset, test_dataset = dataset.train_test_split(test_ratio=0.5, seed=7)

    assert dataset.name == "benchmark_fixture"
    assert len(dataset.examples) == 2
    assert len(train_dataset.examples) == 1
    assert len(test_dataset.examples) == 1


def test_metric_evaluator_computes_required_metrics() -> None:
    relevance_flags = [True, False, True]

    query_metrics = MetricEvaluator.evaluate_query(
        relevance_flags=relevance_flags,
        relevant_total=2,
        k=2,
    )

    assert query_metrics.precision_at_k == 0.5
    assert query_metrics.recall_at_k == 0.5
    assert query_metrics.reciprocal_rank == 1.0
    assert 0.61 < query_metrics.ndcg_at_k < 0.62
