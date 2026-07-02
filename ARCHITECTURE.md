# Architecture

## Overview

Phase 1 implements the document processing portion of the framework:

1. Ingestion discovers supported files and converts them into standardized `Document` models.
2. Preprocessing applies configurable normalization and cleanup steps.
3. Chunking converts documents into ordered `Chunk` models using interchangeable strategies.
4. The pipeline persists a `ProcessedCorpus` artifact for later retrieval stages.

## Core Components

### Configuration

`AppConfig` is loaded from YAML and owns ingestion, preprocessing, chunking, output, and device settings. The resolved device is recorded in pipeline output metadata.

### Data Models

- `Document`: canonical representation of ingested content.
- `Chunk`: chunk artifact with ordering, metadata, and token counts.
- `ProcessedCorpus`: serialized corpus payload containing documents, chunks, statistics, and configuration snapshot.

### Ingestion Pipeline

The ingestion layer is based on a `DocumentLoader` interface. Each file format implements its own loader while `DocumentIngestor` coordinates discovery, recursive traversal, file-type dispatch, metadata extraction, and graceful skipping of unsupported files.

### Preprocessing Pipeline

`TextPreprocessor` applies independently configurable steps:

- Unicode normalization
- Whitespace cleanup
- Repeated header/footer removal across pages
- Page number removal
- Empty line cleanup

Each processed document stores the applied preprocessing steps in metadata for traceability.

### Chunking Pipeline

Chunkers share a common abstract base and can be swapped through configuration.

- `FixedChunker`: token window chunking with overlap.
- `RecursiveChunker`: structure-aware recursive splitting across paragraphs, lines, and sentences.
- `SemanticChunker`: sentence-boundary chunking with similarity-based breakpoints and an optional sentence-transformers backend.

## Extensibility

The project keeps responsibilities separated so future work can add embedding generation, indexing, retrieval, reranking, benchmarking, and visualization without changing the phase-1 data contracts.