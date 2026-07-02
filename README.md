# Retrieval Evaluation & Optimization Framework

This repository contains a production-oriented retrieval engineering toolkit for designing and validating retrieval pipelines before they are attached to a RAG application. The current implementation spans corpus preparation, embedding generation, index persistence, multi-strategy retrieval, optional query enhancement, optional reranking, a CLI, and a basic FastAPI surface.

## Features

- YAML-driven configuration with automatic CPU, CUDA, and Apple MPS selection.
- Modular ingestion for PDF, DOCX, TXT, and Markdown.
- Configurable preprocessing plus fixed, recursive, and semantic chunking.
- Embedding engine with batching, caching, disk persistence, and interchangeable backends.
- Supported embedding models:
	- `sentence-transformers/all-MiniLM-L6-v2`
	- `BAAI/bge-small-en-v1.5`
- Optional query enhancement with query expansion and HyDE.
- Optional cross-encoder reranking.
- Typer CLI and FastAPI endpoints for embedding, indexing, and search.

## Installation

```

Some embedding, HyDE, and reranking configurations download Hugging Face model weights on first use.

## Project Layout

```text
configs/
data/
docs/
examples/
reports/
src/retrieval_evaluation_framework/
tests/
```

## Retrieval Workflow

1. Ingest and preprocess source documents.
2. Chunk documents into a processed corpus artifact.
3. Generate embeddings for dense retrieval workflows.
4. Build and persist BM25, FAISS, or hybrid indexes.
5. Optionally enhance incoming queries.
6. Retrieve candidate chunks.
7. Optionally rerank results with a cross encoder.

## CLI Usage

Prepare a processed corpus:

```bash
retrieval chunk ./data/input --config ./configs/default.yaml
```

Persist embeddings for a processed corpus:

```bash
retrieval retrieval embed \
	--corpus-path ./data/processed/processed_corpus.json \
	--config ./configs/default.yaml \
	--output-path ./data/embeddings
```

Build an index:

```bash
retrieval retrieval index \
	--corpus-path ./data/processed/processed_corpus.json \
	--config ./configs/default.yaml \
	--index-path ./data/index
```

Run retrieval:

```bash
retrieval retrieval retrieve \
	--query "What is the maternity leave policy?" \
	--config ./configs/default.yaml \
	--index-path ./data/index \
	--top-k 5
```

`retrieval retrieval search` is an alias for the same retrieval path.

## FastAPI Usage
uvicorn retrieval_evaluation_framework.api.app:app --reload
```
- `POST /embed`
Swagger UI is available at `/docs`, and the OpenAPI schema is available at `/openapi.json`.
- BM25 remains a first-class retriever because lexical matching is still strong on policy, keyword, and exact-term workloads.
- Hybrid retrieval combines BM25 and dense search because the two failure modes are complementary.
- Reciprocal Rank Fusion is the default fusion strategy because it is simple, robust, and model-agnostic.
- HyDE is supported because it often improves recall when user questions are short or underspecified.
- Cross-encoder reranking is optional because it typically improves precision at the cost of latency.
- MiniLM is a strong default for fast, low-cost dense retrieval.
- BGE and E5 are included because they are common retrieval baselines with different trade-offs in quality, size, and instruction behavior.

## Quality Checks

```bash
ruff check .
mypy
pytest
```