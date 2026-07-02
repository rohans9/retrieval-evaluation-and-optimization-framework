"""Shared request/response models for API routers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from retrieval_evaluation_framework.api.dependencies import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_INDEX_PATH,
)
from retrieval_evaluation_framework.benchmarking.models import BenchmarkResult
from retrieval_evaluation_framework.models import RetrievalResponse


class EmbedRequest(BaseModel):
    """Request model for embedding generation."""

    corpus_path: Path = Field(..., description="Path to the processed corpus JSON file.")
    config_path: Path = Field(
        default=DEFAULT_CONFIG_PATH,
        description="Path to the YAML configuration file.",
    )
    output_path: Path = Field(
        default=DEFAULT_EMBEDDINGS_PATH,
        description="Directory where embedding artifacts should be written.",
    )


class EmbedResponse(BaseModel):
    """Response model for embedding generation."""

    output_path: str
    chunk_count: int
    model_name: str
    dimension: int
    device: str


class IndexRequest(BaseModel):
    """Request model for index construction."""

    corpus_path: Path = Field(..., description="Path to the processed corpus JSON file.")
    config_path: Path = Field(
        default=DEFAULT_CONFIG_PATH,
        description="Path to the YAML configuration file.",
    )
    index_path: Path = Field(
        default=DEFAULT_INDEX_PATH,
        description="Directory where index artifacts should be written.",
    )


class IndexResponse(BaseModel):
    """Response model for index construction."""

    index_path: str
    retriever: str
    chunk_count: int


class RetrieveRequest(BaseModel):
    """Request model for retrieval endpoints."""

    query: str = Field(..., description="The user query for retrieval.")
    config_path: Path = Field(
        default=DEFAULT_CONFIG_PATH,
        description="Path to the YAML configuration file.",
    )
    index_path: Path = Field(
        default=DEFAULT_INDEX_PATH,
        description="Directory containing the persisted retrieval index.",
    )
    top_k: int | None = Field(
        default=None,
        description="Optional override for the number of chunks to return.",
    )


class EvaluateRequest(BaseModel):
    """Request model for single-run evaluation."""

    corpus_path: Path
    dataset_path: Path
    config_path: Path = Field(default=DEFAULT_CONFIG_PATH)
    notes: str | None = None
    experiment_directory: Path | None = None


class BenchmarkRequest(BaseModel):
    """Request model for benchmark workflows."""

    corpus_path: Path
    dataset_path: Path
    config_path: Path = Field(default=DEFAULT_CONFIG_PATH)
    mode: str = Field(default="single", description="single, sweep, or grid_search")
    parameters: dict[str, list[Any]] | None = None
    notes: str | None = None
    experiment_directory: Path | None = None


class BenchmarkExecutionResponse(BaseModel):
    """Response model for benchmark execution endpoints."""

    mode: str
    results: list[BenchmarkResult]


class CompareRequest(BaseModel):
    """Request model for side-by-side experiment comparison."""

    experiment_ids: list[str]
    config_path: Path = Field(default=DEFAULT_CONFIG_PATH)
    experiment_directory: Path | None = None


class ReportResponse(BaseModel):
    """Response model for generated report artifacts."""

    artifacts: dict[str, dict[str, str]]


class IngestRequest(BaseModel):
    """Request model for ingestion endpoint."""

    source_path: Path
    config_path: Path = Field(default=DEFAULT_CONFIG_PATH)


class PreprocessRequest(BaseModel):
    """Request model for preprocessing endpoint."""

    source_path: Path
    config_path: Path = Field(default=DEFAULT_CONFIG_PATH)


class ChunkRequest(BaseModel):
    """Request model for chunking endpoint."""

    source_path: Path
    config_path: Path = Field(default=DEFAULT_CONFIG_PATH)


class PipelineStageResponse(BaseModel):
    """Standard response for ingestion/preprocessing/chunking endpoints."""

    stage: str
    source_path: str
    output_path: str
    document_count: int
    chunk_count: int = 0


class HealthResponse(BaseModel):
    """Simple service health response."""

    status: str
    service: str


class ErrorResponse(BaseModel):
    """User-facing API error response contract."""

    detail: str
    suggestion: str | None = None


RetrievalEndpointResponse = RetrievalResponse
