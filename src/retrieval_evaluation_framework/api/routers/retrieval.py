"""API router for embedding, indexing, and retrieval endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from retrieval_evaluation_framework.api.dependencies import (
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


@router.post("/embed", response_model=EmbedResponse)
def embed_corpus(request: EmbedRequest) -> EmbedResponse:
    """Generate and persist embeddings for a processed corpus."""
    config = load_config(request.config_path)
    corpus = load_corpus(request.corpus_path)
    engine = EmbeddingEngine(config.embedding)
    embedding_store = engine.embed_chunks(corpus.chunks)
    embedding_store.save(request.output_path)
    return EmbedResponse(
        output_path=str(request.output_path),
        chunk_count=len(embedding_store.chunk_ids),
        model_name=embedding_store.model_name,
        dimension=embedding_store.dimension,
        device=engine.device,
    )


@router.post("/index", response_model=IndexResponse)
def index_corpus(request: IndexRequest) -> IndexResponse:
    """Build and persist a retrieval index for a processed corpus."""
    pipeline = get_retrieval_pipeline(request.config_path)
    corpus = pipeline.index_processed_corpus(request.corpus_path)
    pipeline.save_index(request.index_path)
    return IndexResponse(
        index_path=str(request.index_path),
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
