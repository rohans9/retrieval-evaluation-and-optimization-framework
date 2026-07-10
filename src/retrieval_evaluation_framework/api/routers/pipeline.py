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

    document_count = 0
    source_count = 0
    for _, current_source in pipeline.iter_sources(request.source_path):
        source_count += 1
        documents = pipeline.ingest_path(current_source)
        document_count += len(documents)
        pipeline.save_documents_for_source(documents, current_source, "ingested_documents.json")

    output_path = (
        pipeline.config.output.output_directory
        if source_count > 1
        else pipeline.output_path_for(request.source_path, "ingested_documents.json")
    )
    return PipelineStageResponse(
        stage="ingest",
        source_path=str(request.source_path),
        output_path=str(output_path),
        document_count=document_count,
    )


@router.post("/preprocess", response_model=PipelineStageResponse)
def preprocess(request: PreprocessRequest) -> PipelineStageResponse:
    """Ingest and preprocess supported documents."""
    ensure_exists(request.source_path, "Source path")
    pipeline = _load_pipeline(request.config_path)

    document_count = 0
    source_count = 0
    for _, current_source in pipeline.iter_sources(request.source_path):
        source_count += 1
        documents = pipeline.ingest_path(current_source)
        processed = pipeline.preprocess_documents(documents)
        document_count += len(processed)
        pipeline.save_documents_for_source(
            processed,
            current_source,
            "preprocessed_documents.json",
        )

    output_path = (
        pipeline.config.output.output_directory
        if source_count > 1
        else pipeline.output_path_for(request.source_path, "preprocessed_documents.json")
    )
    return PipelineStageResponse(
        stage="preprocess",
        source_path=str(request.source_path),
        output_path=str(output_path),
        document_count=document_count,
    )


@router.post("/chunk", response_model=PipelineStageResponse)
def chunk(request: ChunkRequest) -> PipelineStageResponse:
    """Run the complete phase-1 processing pipeline and persist corpus output."""
    ensure_exists(request.source_path, "Source path")
    pipeline = _load_pipeline(request.config_path)
    try:
        document_count = 0
        chunk_count = 0
        source_count = 0
        for _, current_source in pipeline.iter_sources(request.source_path):
            source_count += 1
            corpus = (
                pipeline.process_file(current_source)
                if current_source.is_file()
                else pipeline.process_directory(current_source)
            )
            document_count += int(corpus.statistics["document_count"])
            chunk_count += int(corpus.statistics["chunk_count"])
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    output_path = pipeline.config.output.output_directory if source_count > 1 else (
        pipeline.output_path_for(
            request.source_path,
            pipeline.config.output.processed_corpus_filename,
        )
    )
    return PipelineStageResponse(
        stage="chunk",
        source_path=str(request.source_path),
        output_path=str(output_path),
        document_count=document_count,
        chunk_count=chunk_count,
    )
