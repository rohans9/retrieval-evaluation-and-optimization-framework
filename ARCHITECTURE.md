# Architecture

## Overview

The framework is organized as a sequence of interchangeable layers:

1. Ingestion discovers supported files and converts them into standardized `Document` models.
2. Preprocessing applies configurable normalization and cleanup steps.
3. Chunking converts documents into ordered `Chunk` models.
4. The document pipeline persists a `ProcessedCorpus` artifact.
5. The embedding layer turns chunks and queries into reusable vectors.
6. The index layer persists lexical, dense, or hybrid retrieval structures.
7. The retrieval layer serves ranked chunk candidates.
8. Optional query enhancement and reranking refine recall and precision.
9. Evaluation and benchmarking layers measure retrieval quality and system performance.
10. CLI and FastAPI surfaces expose both retrieval and benchmarking workflows.

## Core Components

### Configuration

`AppConfig` is loaded from YAML and owns ingestion, preprocessing, chunking, embedding, indexing, retrieval, query enhancement, reranking, benchmarking, output, and device settings. Device resolution is centralized so embeddings, HyDE, reranking, and benchmarking runs can use the same hardware policy.

### Data Models

- `Document`: canonical representation of ingested content.
- `Chunk`: chunk artifact with ordering, metadata, and token counts.
- `ProcessedCorpus`: serialized corpus payload bridging phase 1 and phase 2.
- `RetrievalResult`: a scored retrieved chunk.
- `RetrievalResponse`: a retrieval response with provenance and latency metadata.
- `EvaluationDataset`: a validated custom benchmark dataset.
- `ExperimentRecord`: a persisted benchmark history entry.
- `BenchmarkResult`: a structured benchmark execution artifact.

### Embedding Layer

`EmbeddingEngine` is the only object the rest of the system depends on for vector generation. It hides the underlying backend, batches requests, caches vectors to disk, and persists `EmbeddingStore` artifacts so dense indexes can be rebuilt without recomputing embeddings.

Supported backends and behaviors:

- Sentence-transformers for production dense retrieval.
- A deterministic hashing backend for offline tests and fallbacks.
- Automatic CPU, CUDA, and Apple MPS selection.

### Index Layer

Indexes are responsible for build, save, load, and rebuild operations.

- `BM25Index` persists lexical search structures.
- `DenseIndex` uses FAISS for dense cosine similarity search.

Persistence is part of the design so future benchmarking runs can reuse embedding and index artifacts instead of paying setup cost on every experiment.

### Retriever Layer

All retrievers implement a shared `BaseRetriever` interface with `build_index()`, `retrieve()`, `get_configuration()`, `save_index()`, and `load_index()`.

- `BM25Retriever`: lexical search over chunk text.
- `DenseRetriever`: semantic search over embedding vectors.
- `HybridRetriever`: merges BM25 and dense rankings with Reciprocal Rank Fusion.

The retrieval pipeline depends only on that interface, which keeps later benchmarking and evaluation work decoupled from individual algorithms.

### Query Enhancement Layer

Query enhancement is optional and configuration-driven.

- `QueryExpander` builds a lightweight semantic vocabulary from the indexed corpus.
- `HyDEQueryEnhancer` generates a hypothetical answer and uses it as the effective retrieval query.

These components return a structured enhancement result so retrieval code can log and report which transformation was applied.

### Reranking Layer

Reranking is applied after retrieval over the top candidate set.

- Cross-encoder reranking is used when transformer models are available.
- A lexical fallback exists to keep the pipeline operable in constrained environments.

The retrieval pipeline can operate unchanged with reranking disabled, enabled, or swapped to another reranker implementation later.

### Evaluation Layer

Evaluation is intentionally decoupled from retrieval execution.

- `JsonEvaluationDatasetLoader` validates custom JSON or YAML benchmark datasets.
- `MetricEvaluator` computes Precision@K, Recall@K, MRR, and NDCG@K independently.
- Relevance metrics are aggregated separately from latency metrics so quality and cost trade-offs remain explicit.

### Benchmarking Layer

`BenchmarkRunner` orchestrates dataset evaluation on top of the retrieval pipeline.

- Single experiments benchmark one pipeline configuration.
- Parameter sweeps vary one parameter at a time while the rest stay fixed.
- Grid search executes the full Cartesian product of configuration values.
- Ablation studies compare a baseline configuration against targeted removals or substitutions.

Each experiment records retrieval quality metrics, latency metrics, configuration, device, and notes.

### Experiment Tracking And Comparison

- `ExperimentTracker` stores one JSON file per experiment and maintains a consolidated history index.
- `ExperimentComparisonEngine` builds side-by-side comparison tables over persisted experiments.
- Benchmark result artifacts are saved separately from the lightweight experiment history so later phases can add richer reporting without changing benchmark execution.

### Delivery Surfaces

- The Typer CLI supports corpus embedding, index construction, and retrieval execution.
- The Typer CLI also supports evaluation, benchmark execution, parameter sweeps, grid search, experiment listing, and experiment comparison.
- The FastAPI app exposes retrieval endpoints plus `/evaluate`, `/benchmark`, `/compare`, `/experiments`, and `/experiment/{id}` with OpenAPI documentation.

## Design Rationale

- FAISS was chosen for dense indexing because it is widely used, fast, and easy to persist.
- BM25 remains necessary because exact-term matching is often decisive on enterprise text.
- Hybrid retrieval and Reciprocal Rank Fusion provide a strong default without coupling to one retriever family.
- HyDE and reranking are optional because they can improve quality, but they carry real latency and model cost.

## Extensibility

The project keeps responsibilities separated so future work can add visualization, recommendation logic, richer reporting, and production-facing orchestration without changing the established ingestion, corpus, embedding, retrieval, or benchmark contracts.