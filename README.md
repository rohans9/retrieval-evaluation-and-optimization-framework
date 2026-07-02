# Retrieval Evaluation & Optimization Framework

Retrieval Evaluation & Optimization Framework is a production-oriented toolkit for building, evaluating, and improving retrieval pipelines before connecting them to a RAG or search application.

The repository now implements four complete phases:

- Phase 1: ingestion, preprocessing, chunking, and processed-corpus persistence.
- Phase 2: embedding generation, lexical and dense indexing, hybrid retrieval, query enhancement, and reranking.
- Phase 3: labeled evaluation datasets, benchmark execution, experiment tracking, parameter sweeps, grid search, and experiment comparison.
- Phase 4: leaderboard generation, explainable recommendation, trade-off analysis, HTML and PNG visualizations, and Markdown, CSV, and JSON reporting.

## Features

- YAML-driven configuration with automatic CPU, CUDA, and Apple MPS resolution.
- Modular ingestion for PDF, DOCX, TXT, and Markdown.
- Configurable preprocessing plus fixed, recursive, and semantic chunking.
- Embedding engine with batching, caching, persistence, and offline hashing fallback.
- BM25, FAISS dense, and hybrid retrieval with Reciprocal Rank Fusion.
- Optional query expansion, HyDE, and reranking.
- Evaluation with Precision@K, Recall@K, MRR, and NDCG@K.
- Benchmark execution for single runs, sweeps, grid search, and ablation studies.
- Local experiment history with side-by-side comparison.
- Leaderboards, recommendation summaries, trade-off analysis, and generated artifacts.
- Typer CLI and FastAPI surfaces for end-to-end workflows.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -e .[dev]
```

Some embedding, HyDE, and reranking configurations download Hugging Face model weights on first use.

## Project Layout

```text
configs/
prompts/
reports/
src/retrieval_evaluation_framework/
tests/
```

## End-To-End Workflow

1. Ingest and preprocess source documents.
2. Chunk documents into a processed corpus artifact.
3. Generate embeddings for dense retrieval workflows.
4. Build and persist BM25, dense, or hybrid indexes.
5. Run retrieval with optional query enhancement and reranking.
6. Evaluate against labeled datasets.
7. Benchmark multiple configurations and store experiment history.
8. Generate leaderboards, recommendations, reports, and visualizations.

## CLI Usage

Build a processed corpus:

```bash
retrieval chunk ./data/input --config ./configs/default.yaml
```

Embed a processed corpus:

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

Run one evaluation benchmark:

```bash
retrieval retrieval evaluate \
  --corpus-path ./data/processed/processed_corpus.json \
  --dataset-path ./data/eval/dataset.yaml \
  --config ./configs/default.yaml
```

Generate phase-4 analysis outputs:

```bash
retrieval retrieval leaderboard --config ./configs/default.yaml
retrieval retrieval recommend --config ./configs/default.yaml
retrieval retrieval report --config ./configs/default.yaml
retrieval retrieval visualize --config ./configs/default.yaml
retrieval retrieval history --config ./configs/default.yaml
```

`retrieval retrieval search` remains an alias for the retrieval path.

## FastAPI Usage

Start the API:

```bash
uvicorn retrieval_evaluation_framework.api.app:app --reload
```

Key endpoints:

- `POST /embed`
- `POST /index`
- `POST /retrieve`
- `POST /search`
- `POST /evaluate`
- `POST /benchmark`
- `POST /compare`
- `GET /experiments`
- `GET /experiment/{id}`
- `GET /leaderboard`
- `GET /recommendation`
- `GET /reports`
- `GET /visualizations`
- `GET /history`

Swagger UI is available at `/docs`, and the OpenAPI schema is available at `/openapi.json`.

## Reporting And Visualization Outputs

Phase 4 writes generated artifacts to configurable directories:

- `reporting.reports_directory`: Markdown, CSV, and JSON experiment reports.
- `visualization.output_directory`: HTML dashboards and PNG charts.
- experiment history JSON files: stored summaries, recommendation metadata, leaderboard rank, and artifact paths.

## Quality Checks

```bash
ruff check .
mypy
pytest
```

The repository currently passes the full quality gate with strict typing and automated tests.
