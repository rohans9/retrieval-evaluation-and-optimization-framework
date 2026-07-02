"""Shared API dependency helpers and state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from retrieval_evaluation_framework.benchmarking.analysis import BenchmarkAnalysisService
from retrieval_evaluation_framework.benchmarking.tracking import ExperimentTracker
from retrieval_evaluation_framework.config.settings import AppConfig
from retrieval_evaluation_framework.logging import configure_logging
from retrieval_evaluation_framework.models import ProcessedCorpus
from retrieval_evaluation_framework.retrieval.pipeline import RetrievalPipeline

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_EMBEDDINGS_PATH = Path("data/embeddings")
DEFAULT_INDEX_PATH = Path("data/index")

_PIPELINE_CACHE: dict[str, RetrievalPipeline] = {}


def ensure_exists(path: Path, description: str) -> None:
    """Raise an HTTP 404 when a required path does not exist."""
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{description} not found: {path}")


def load_config(config_path: Path) -> AppConfig:
    """Load and validate API configuration."""
    ensure_exists(config_path, "Config file")
    config = AppConfig.load_yaml(config_path)
    configure_logging(config.logging.level)
    return config


def get_retrieval_pipeline(config_path: Path) -> RetrievalPipeline:
    """Return a cached retrieval pipeline for a given config path."""
    resolved_path = str(config_path.resolve())
    pipeline = _PIPELINE_CACHE.get(resolved_path)
    if pipeline is None:
        pipeline = RetrievalPipeline(load_config(config_path))
        _PIPELINE_CACHE[resolved_path] = pipeline
    return pipeline


def get_tracker(config: AppConfig, directory: Path | None = None) -> ExperimentTracker:
    """Return an experiment tracker for benchmark history."""
    return ExperimentTracker(directory or config.benchmark.experiment_directory)


def get_analysis_service(
    config: AppConfig,
    directory: Path | None = None,
) -> BenchmarkAnalysisService:
    """Return the phase-4 analysis service over benchmark history."""
    return BenchmarkAnalysisService(config, get_tracker(config, directory))


def load_parameter_mapping(payload: dict[str, Any] | None) -> dict[str, list[Any]]:
    """Validate benchmark parameter payloads supplied via API."""
    if payload is None:
        return {}
    mapping: dict[str, list[Any]] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, list):
            msg = "Benchmark parameters must map dotted paths to lists of values"
            raise HTTPException(status_code=400, detail=msg)
        mapping[key] = value
    return mapping


def load_corpus(corpus_path: Path) -> ProcessedCorpus:
    """Load a processed corpus from disk."""
    ensure_exists(corpus_path, "Corpus file")
    return ProcessedCorpus.model_validate_json(corpus_path.read_text(encoding="utf-8"))


def ensure_index_loaded(pipeline: RetrievalPipeline, index_path: Path) -> None:
    """Load persisted index artifacts when needed."""
    if pipeline.is_built:
        return

    ensure_exists(index_path, "Index directory")
    try:
        pipeline.load_index(index_path)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=f"Index artifact missing in {index_path}: {error}",
        ) from error
