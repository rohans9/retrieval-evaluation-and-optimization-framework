# Retrieval Evaluation & Optimization Framework

Retrieval Evaluation & Optimization Framework is a production-oriented developer tool for building and validating retrieval pipelines before attaching them to a RAG application.

The framework now provides two first-class interfaces:

- a polished Typer CLI for local workflows
- a structured FastAPI REST API for service-based workflows

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -e .[dev]
```

## Quick Start

### 1. Process sample documents

```bash
retrieval ingest examples/sample_data/documents --config configs/examples/small_collection.yaml
retrieval chunk examples/sample_data/documents --config configs/examples/small_collection.yaml
```

### 2. Build retrieval artifacts and query

```bash
retrieval retrieval embed --corpus-path data/processed/processed_corpus.json --config configs/examples/small_collection.yaml
retrieval retrieval index --corpus-path data/processed/processed_corpus.json --config configs/examples/small_collection.yaml --index-path data/index
retrieval retrieval retrieve --query "What is the maternity leave policy?" --config configs/examples/small_collection.yaml --index-path data/index --top-k 3
```

### 3. Benchmark and recommendation flow

```bash
retrieval retrieval benchmark --corpus-path data/processed/processed_corpus.json --dataset-path examples/sample_data/evaluation_dataset.yaml --config configs/examples/benchmark_experiment.yaml
retrieval retrieval report --config configs/examples/benchmark_experiment.yaml
retrieval retrieval leaderboard --config configs/examples/benchmark_experiment.yaml
retrieval retrieval recommend --config configs/examples/benchmark_experiment.yaml
```

## CLI Usage

Main commands:

- `retrieval ingest`
- `retrieval preprocess`
- `retrieval chunk`
- `retrieval retrieval embed`
- `retrieval retrieval index`
- `retrieval retrieval retrieve`
- `retrieval retrieval search`
- `retrieval retrieval evaluate`
- `retrieval retrieval benchmark`
- `retrieval retrieval compare`
- `retrieval retrieval sweep`
- `retrieval retrieval grid-search`
- `retrieval retrieval leaderboard`
- `retrieval retrieval report`
- `retrieval retrieval recommend`
- `retrieval retrieval history`
- `retrieval retrieval visualize`

Get command help:

```bash
retrieval --help
retrieval retrieval --help
retrieval retrieval benchmark --help
```

## API Usage

Run API server:

```bash
uvicorn retrieval_evaluation_framework.api.app:app --reload
```

Core endpoints:

- `POST /ingest`
- `POST /preprocess`
- `POST /chunk`
- `POST /embed`
- `POST /index`
- `POST /retrieve`
- `POST /search`
- `POST /evaluate`
- `POST /benchmark`
- `POST /compare`
- `GET /leaderboard`
- `GET /recommendation`
- `GET /history`
- `GET /reports`
- `GET /visualizations`
- `GET /experiments`
- `GET /experiment/{id}`
- `GET /health`

OpenAPI and docs:

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`

## Configuration Examples

`configs/examples/` includes ready-to-use templates for:

- small document collections
- large document collections
- BM25 only
- dense retrieval
- hybrid retrieval
- HyDE enabled
- reranker enabled
- benchmark experiment runs
- parameter sweep templates
- grid search templates

## Example Workflows

See `examples/workflows/`:

- `workflow_1_retrieval_pipeline.md`
- `workflow_2_benchmark_reporting.md`
- `workflow_3_sweep_comparison_tradeoffs.md`

Each workflow is fully command-driven and runnable without writing Python code.

## Sample Dataset And Corpus

`examples/sample_data/` includes:

- sample corpus documents
- labeled evaluation dataset
- retrieval queries
- benchmark config mapping

The sample assets are designed to work immediately after clone.

## FAQ

### Why both CLI and REST API?

CLI is optimized for local experimentation and scripts. API is optimized for integration into tooling, dashboards, and services.

### Why Typer?

Typer provides type-safe command declarations, rich help output, and production-grade ergonomics for Python CLIs.

### Why FastAPI?

FastAPI provides strict request and response validation, OpenAPI generation, and predictable HTTP behavior for engineering teams.

### Why YAML configs?

YAML keeps retrieval and benchmark settings reproducible, reviewable, and easy to version-control.

### Do I need GPU?

No. The framework supports CPU execution and can automatically select CUDA or MPS when available.

## Troubleshooting

- If retrieval fails because index artifacts are missing, run `retrieval retrieval index` first.
- If benchmarking fails due to dataset paths, verify `dataset_path` in config and command arguments.
- If API requests return validation errors, inspect `/docs` schema and retry with valid payloads.
- If chart export fails in headless environments, ensure the environment can import Matplotlib and keep non-interactive backend settings.

## Quality Gate

```bash
ruff check .
mypy
pytest
```

Current state at phase completion: Ruff, MyPy, and Pytest pass.
