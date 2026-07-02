"""Configuration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from retrieval_evaluation_framework.config.settings import AppConfig, ChunkingConfig


def test_load_default_config() -> None:
    config = AppConfig.load_yaml(Path("configs/default.yaml"))

    assert config.chunking.strategy == "recursive"
    assert config.output.processed_corpus_filename == "processed_corpus.json"
    assert config.resolved_device in {"cpu", "cuda", "mps"}


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=10, overlap=10)
