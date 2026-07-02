"""Configuration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from retrieval_evaluation_framework.config.settings import AppConfig, ChunkingConfig


def test_load_default_config() -> None:
    config = AppConfig.load_yaml(Path("configs/default.yaml"))

    assert config.chunking.strategy == "recursive"
    assert config.output.processed_corpus_filename == "processed_corpus.json"
    assert config.reporting.include_markdown is True
    assert config.visualization.export_png is True
    assert config.recommendation.quality_weight == 0.65
    assert config.resolved_device in {"cpu", "cuda", "mps"}


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=10, overlap=10)


def test_example_app_configs_are_loadable() -> None:
    app_config_files = [
        "small_collection.yaml",
        "large_collection.yaml",
        "bm25_only.yaml",
        "dense_retrieval.yaml",
        "hybrid_retrieval.yaml",
        "hyde_enabled.yaml",
        "reranker_enabled.yaml",
        "benchmark_experiment.yaml",
    ]
    for name in app_config_files:
        config = AppConfig.load_yaml(Path("configs/examples") / name)
        assert config.retrieval.top_k > 0


def test_example_parameter_templates_are_loadable() -> None:
    for name in ["parameter_sweep.yaml", "grid_search.yaml"]:
        payload = yaml.safe_load((Path("configs/examples") / name).read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert payload
