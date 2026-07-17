# Retrieval Evaluation and Optimization Framework

A Python framework for building, evaluating, and optimizing document retrieval pipelines across BM25, dense, and hybrid strategies.

It provides:
- End-to-end data pipeline commands (`ingest`, `preprocess`, `chunk`)
- Retrieval tooling (`embed`, `index`, `retrieve`, `search`)
- Benchmarking and optimization workflows (single run, parameter sweep, grid search, Optuna)
- Report generation, leaderboard views, and visualizations
- REST API endpoints for pipeline and retrieval operations

## Key Capabilities

- Multiple retrievers: BM25, dense, and hybrid reciprocal-rank fusion
- Config-driven execution via YAML in the `configs/` directory
- Query enhancement and reranking support
- Experiment tracking and comparison utilities
- Portable outputs for reports and visual artifacts

## Project Layout

- `src/retrieval_evaluation_framework/`: Core package
- `configs/`: Default and example YAML configurations
- `scripts/`: Convenience scripts for baseline/final workflows
- `tests/`: Automated tests
- `data/`: Input corpora, processed artifacts, indices, and evaluation sets
- `reports/`: Benchmark outputs and generated reports/visualizations

## Installation

Python 3.12+ is required.

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

For development tools:

```bash
pip install -e .[dev]
```

## Quick Start (CLI)

1. Ingest documents:

```bash
retrieval ingest data/raw/insurance --config configs/default.yaml
```

2. Preprocess and chunk:

```bash
retrieval preprocess data/raw/insurance --config configs/default.yaml
retrieval chunk data/raw/insurance --config configs/default.yaml
```

3. Build retrieval artifacts:

```bash
retrieval retrieval embed \
  --corpus-path data/processed/insurance/processed_corpus.json \
  --config configs/examples/dense_retrieval.yaml \
  --output-path data/embeddings/insurance

retrieval retrieval index \
  --corpus-path data/processed/insurance/processed_corpus.json \
  --config configs/examples/hybrid_retrieval.yaml \
  --index-path data/index/insurance/hybrid
```

4. Run benchmark:

```bash
retrieval retrieval benchmark \
  --corpus-path data/processed/insurance/processed_corpus.json \
  --dataset-path data/evaluation/insurance_eval_dataset.json \
  --config configs/examples/hybrid_retrieval.yaml \
  --experiment-directory reports/experiments
```

## API

The FastAPI application object is exposed at:
- `retrieval_evaluation_framework.api.app:app`

Run locally with Uvicorn:

```bash
uvicorn retrieval_evaluation_framework.api.app:app --reload
```

Useful endpoints include:
- `GET /health`
- `POST /ingest`
- `POST /preprocess`
- `POST /chunk`
- `POST /embed`
- `POST /index`
- `POST /retrieve`
- `POST /search`

## Configuration

- Primary defaults: `configs/default.yaml`
- Focused examples: `configs/examples/`

Important sections:
- `chunking`
- `embedding`
- `retrieval`
- `reranking`
- `benchmark`
- `reporting`
- `visualization`

## Quality Checks

```bash
python -m pytest -q
python -m ruff check src tests
python -m mypy src tests
```

## Notes for Releases

- Generated artifacts under `reports/`, `data/cache/`, and `data/processed/` are intentionally gitignored.
- Keep benchmark datasets and curated sample corpora versioned only when they are required for reproducibility.

## License

MIT. See `LICENSE`.
