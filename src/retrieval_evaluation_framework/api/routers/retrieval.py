"""API router for embedding, indexing, and retrieval endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from retrieval_evaluation_framework.api.dependencies import (
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_INDEX_PATH,
    ensure_index_loaded,
    get_retrieval_pipeline,
    load_config,
    load_corpus,
)
from retrieval_evaluation_framework.api.schemas import (
    EmbedRequest,
    EmbedResponse,
    IndexRequest,
    IndexResponse,
    RetrievalEndpointResponse,
    RetrieveRequest,
)
from retrieval_evaluation_framework.embeddings.engine import EmbeddingEngine

router = APIRouter(prefix="", tags=["retrieval"])


def _infer_corpus_domain(
    config_output_directory: Path,
    processed_filename: str,
    corpus_path: Path,
) -> str | None:
    if corpus_path.name != processed_filename:
        return None
    try:
        relative_parent = corpus_path.parent.resolve().relative_to(config_output_directory.resolve())
    except ValueError:
        return None
    if len(relative_parent.parts) != 1:
        return None
    return relative_parent.parts[0]


@router.post("/embed", response_model=EmbedResponse)
def embed_corpus(request: EmbedRequest) -> EmbedResponse:
    """Generate and persist embeddings for a processed corpus."""
    config = load_config(request.config_path)
    corpus = load_corpus(request.corpus_path)
    engine = EmbeddingEngine(config.embedding)
    domain = _infer_corpus_domain(
        config.output.output_directory,
        config.output.processed_corpus_filename,
        request.corpus_path,
    )
    output_path = (
        request.output_path / domain
        if domain and request.output_path == DEFAULT_EMBEDDINGS_PATH
        else request.output_path
    )
    embedding_store = engine.embed_chunks(corpus.chunks)
    embedding_store.save(output_path)
    return EmbedResponse(
        output_path=str(output_path),
        chunk_count=len(embedding_store.chunk_ids),
        model_name=embedding_store.model_name,
        dimension=embedding_store.dimension,
        device=engine.device,
    )


@router.post("/index", response_model=IndexResponse)
def index_corpus(request: IndexRequest) -> IndexResponse:
    """Build and persist a retrieval index for a processed corpus."""
    pipeline = get_retrieval_pipeline(request.config_path)
    config = load_config(request.config_path)
    domain = _infer_corpus_domain(
        config.output.output_directory,
        config.output.processed_corpus_filename,
        request.corpus_path,
    )
    index_path = (
        request.index_path / domain
        if domain and request.index_path == DEFAULT_INDEX_PATH
        else request.index_path
    )
    corpus = pipeline.index_processed_corpus(request.corpus_path)
    pipeline.save_index(index_path)
    return IndexResponse(
        index_path=str(index_path),
        retriever=pipeline.retriever.name,
        chunk_count=len(corpus.chunks),
    )


@router.post("/retrieve", response_model=RetrievalEndpointResponse)
def retrieve_chunks(request: RetrieveRequest) -> RetrievalEndpointResponse:
    """Retrieve relevant chunks for a query."""
    pipeline = get_retrieval_pipeline(request.config_path)
    ensure_index_loaded(pipeline, request.index_path)
    try:
        return pipeline.retrieve(request.query, top_k=request.top_k)
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/search", response_model=RetrievalEndpointResponse)
def search(request: RetrieveRequest) -> RetrievalEndpointResponse:
    """Alias endpoint for retrieval, kept for API discoverability."""
    return retrieve_chunks(request)
