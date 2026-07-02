"""API router for ingestion, preprocessing, and chunking workflows."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from retrieval_evaluation_framework.api.dependencies import ensure_exists
from retrieval_evaluation_framework.api.schemas import (
    ChunkRequest,
    IngestRequest,
    PipelineStageResponse,
    PreprocessRequest,
)
from retrieval_evaluation_framework.config.settings import AppConfig
from retrieval_evaluation_framework.pipeline import DocumentProcessingPipeline

router = APIRouter(prefix="", tags=["pipeline"])


def _load_pipeline(config_path: Path) -> DocumentProcessingPipeline:
    ensure_exists(config_path, "Config file")
    config = AppConfig.load_yaml(config_path)
    return DocumentProcessingPipeline(config)


@router.post("/ingest", response_model=PipelineStageResponse)
def ingest(request: IngestRequest) -> PipelineStageResponse:
    """Ingest supported documents and persist raw document artifacts."""
    ensure_exists(request.source_path, "Source path")
    pipeline = _load_pipeline(request.config_path)
    documents = pipeline.ingest_path(request.source_path)
    output_path = pipeline.save_documents(
        documents,
        pipeline.config.output.output_directory / "ingested_documents.json",
    )
    return PipelineStageResponse(
        stage="ingest",
        source_path=str(request.source_path),
        output_path=str(output_path),
        document_count=len(documents),
    )


@router.post("/preprocess", response_model=PipelineStageResponse)
def preprocess(request: PreprocessRequest) -> PipelineStageResponse:
    """Ingest and preprocess supported documents."""
    ensure_exists(request.source_path, "Source path")
    pipeline = _load_pipeline(request.config_path)
    documents = pipeline.ingest_path(request.source_path)
    processed = pipeline.preprocess_documents(documents)
    output_path = pipeline.save_documents(
        processed,
        pipeline.config.output.output_directory / "preprocessed_documents.json",
    )
    return PipelineStageResponse(
        stage="preprocess",
        source_path=str(request.source_path),
        output_path=str(output_path),
        document_count=len(processed),
    )


@router.post("/chunk", response_model=PipelineStageResponse)
def chunk(request: ChunkRequest) -> PipelineStageResponse:
    """Run the complete phase-1 processing pipeline and persist corpus output."""
    ensure_exists(request.source_path, "Source path")
    pipeline = _load_pipeline(request.config_path)
    try:
        corpus = (
            pipeline.process_file(request.source_path)
            if request.source_path.is_file()
            else pipeline.process_directory(request.source_path)
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    output_path = (
        pipeline.config.output.output_directory
        / pipeline.config.output.processed_corpus_filename
    )
    return PipelineStageResponse(
        stage="chunk",
        source_path=str(request.source_path),
        output_path=str(output_path),
        document_count=int(corpus.statistics["document_count"]),
        chunk_count=int(corpus.statistics["chunk_count"]),
    )
