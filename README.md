# Retrieval Evaluation & Optimization Framework

This repository contains the phase-1 foundation for a production-grade retrieval evaluation framework. The current implementation covers document ingestion, configurable preprocessing, multiple chunking strategies, a reusable processing pipeline, and a basic CLI for preparing corpora for later retrieval experiments.

## Features

- YAML-driven configuration with automatic device resolution.
- Modular ingestion for PDF, DOCX, TXT, and Markdown.
- Configurable preprocessing for normalization and cleanup.
- Fixed, recursive, and semantic chunking strategies.
- Processed corpus serialization for later retrieval and benchmarking phases.
- Typer CLI with Rich progress output.
- Unit tests, Ruff configuration, and MyPy configuration.

## Installation

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -e .[dev]
```

Optional retrieval dependencies for later phases:

```bash
pip install -e .[dev,retrieval]
```

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

## Phase 1 Usage

Run chunk generation end to end for a file or directory:

```bash
retrieval chunk ./data/input --config ./configs/default.yaml
```

Run only ingestion:

```bash
retrieval ingest ./data/input --config ./configs/default.yaml
```

Run ingestion plus preprocessing:

```bash
retrieval preprocess ./data/input --config ./configs/default.yaml
```

The processed corpus is written to the configured output directory as JSON. Later phases can consume the same corpus without repeating ingestion and cleanup work.

## Quality Checks

```bash
pytest
ruff check .
mypy
```