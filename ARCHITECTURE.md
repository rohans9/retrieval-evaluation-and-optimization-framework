# Architecture

## Overview

The framework is organized into modular, interchangeable layers:

1. Ingestion
2. Preprocessing
3. Chunking
4. Embedding
5. Indexing and retrieval
6. Optional query enhancement and reranking
7. Evaluation and benchmarking
8. Recommendation, reporting, and visualization
9. Interface layer (CLI + API)

Phase 5 focuses on interface quality and user experience while preserving existing retrieval and benchmarking architecture.

## CLI Architecture

The CLI is implemented with Typer and split into:

- root commands for corpus processing: `ingest`, `preprocess`, `chunk`
- retrieval namespace commands under `retrieval`

CLI UX design includes:

- type-driven argument parsing
- command help text and examples
- Rich progress bars and tabular output
- consistent structured command logging
- user-friendly error messaging without stack trace exposure

## API Architecture

FastAPI is now router-based instead of a single endpoint file.

Router modules:

- `api/routers/system.py` for service health
- `api/routers/pipeline.py` for ingestion/preprocessing/chunking
- `api/routers/retrieval.py` for embedding/index/retrieve/search
- `api/routers/benchmarking.py` for evaluation, benchmark, compare, and phase-4 analysis endpoints

Shared API components:

- `api/dependencies.py` for state/cache/config helpers
- `api/schemas.py` for request/response models
- `api/app.py` for app wiring, middleware, and global exception handlers

API behavior includes:

- request and response validation via Pydantic
- standardized user-facing error payloads
- middleware-based request logging and execution-time headers
- OpenAPI/Swagger/ReDoc generation by default

## Configuration Flow

1. User selects a YAML configuration.
2. CLI/API loads the config into `AppConfig`.
3. Config drives pipeline/retrieval/benchmark/recommendation behavior.
4. Example YAML templates provide reusable starting points for common modes.

Configuration is the primary control plane for reproducibility.

## Request Flow

### CLI Request Flow

1. Parse command and typed arguments.
2. Start command span logging.
3. Execute action with Rich progress feedback.
4. Render colored summary table.
5. Log duration and outcome.

### API Request Flow

1. Validate request payload/path/query.
2. Route to appropriate router handler.
3. Execute service logic using shared dependency helpers.
4. Validate response model.
5. Middleware logs method/path/status/latency and returns `X-Execution-Time-Ms`.
6. Exceptions are normalized into actionable user-facing error responses.

## Engineering Decisions

- Typer was selected for a production-quality Python CLI with strong type ergonomics.
- FastAPI was selected for strict validation and built-in API docs.
- YAML was kept as the source of truth for reproducible experimentation.
- Both CLI and API are provided so users can choose interactive local workflows or service integration paths.

## Extensibility

Phase-5 interface changes are additive and do not alter retrieval algorithms, metrics, or benchmark semantics.

New interfaces are built on top of existing pipeline contracts, preserving modularity and backward compatibility.
