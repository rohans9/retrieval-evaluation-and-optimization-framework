# Architecture

## Overview

The framework is organized as a sequence of interchangeable layers:

1. Ingestion discovers supported files and converts them into normalized `Document` models.
2. Preprocessing applies configurable cleanup and normalization steps.
3. Chunking converts documents into ordered `Chunk` models.
4. The document pipeline persists a `ProcessedCorpus` artifact.
5. The embedding layer turns chunks and queries into reusable vectors.
6. The index layer persists lexical, dense, or hybrid retrieval structures.
7. The retrieval layer serves ranked chunk candidates.
8. Optional query enhancement and reranking refine recall and precision.
9. Evaluation and benchmarking layers measure retrieval quality and system performance.
10. Recommendation, reporting, and visualization layers convert experiment history into decisions and artifacts.
11. CLI and FastAPI surfaces expose the full workflow.

## Core Components

### Configuration

`AppConfig` is loaded from YAML and owns ingestion, preprocessing, chunking, embedding, indexing, retrieval, query enhancement, reranking, benchmarking, reporting, visualization, recommendation, output, and device settings.

Device resolution is centralized so embeddings, HyDE, reranking, benchmarking, and artifact generation use the same hardware policy.

### Data Models

Key repository-wide models include:

- `Document`: canonical representation of ingested content.
- `Chunk`: chunk artifact with ordering, metadata, and token counts.
- `ProcessedCorpus`: serialized corpus payload bridging preparation and retrieval.
- `RetrievalResult`: a scored retrieved chunk.
- `RetrievalResponse`: a retrieval response with provenance and latency metadata.
- `EvaluationDataset`: a validated custom benchmark dataset.
- `ExperimentRecord`: a persisted benchmark and analysis history entry.
- `BenchmarkResult`: a structured benchmark execution artifact.
- `Leaderboard`: ranked experiment summary.
- `RecommendationResult`: explainable best-pipeline selection.
- `TradeoffAnalysis`: structured observations about quality, latency, and component choices.

### Embedding Layer

`EmbeddingEngine` is the only object the rest of the system depends on for vector generation. It hides the underlying backend, batches requests, caches vectors to disk, and persists embedding artifacts so dense indexes can be rebuilt without recomputing embeddings.

Supported behaviors include:

- sentence-transformers for production dense retrieval.
- a deterministic hashing backend for offline tests and fallbacks.
- automatic CPU, CUDA, and Apple MPS selection.

### Index Layer

Indexes are responsible for build, save, load, and rebuild operations.

- `BM25Index` persists lexical search structures.
- `DenseIndex` uses FAISS for dense cosine similarity search.

Persistence is part of the design so later benchmarking and reporting workflows can reuse index artifacts instead of paying setup cost on every experiment.

### Retriever Layer

All retrievers implement a shared `BaseRetriever` interface with `build_index()`, `retrieve()`, `get_configuration()`, `save_index()`, and `load_index()`.

- `BM25Retriever`: lexical search over chunk text.
- `DenseRetriever`: semantic search over embedding vectors.
- `HybridRetriever`: merges BM25 and dense rankings with Reciprocal Rank Fusion.

The retrieval pipeline depends only on that interface, which keeps benchmarking and downstream analysis decoupled from individual retrieval algorithms.

### Query Enhancement Layer

Query enhancement is optional and configuration-driven.

- `QueryExpander` builds a lightweight semantic vocabulary from the indexed corpus.
- `HyDEQueryEnhancer` generates a hypothetical answer and uses it as the effective retrieval query.

These components return structured outputs so retrieval code can record which transformation was applied and how much latency it introduced.

### Reranking Layer

Reranking is applied after retrieval over the top candidate set.

- cross-encoder reranking is used when transformer models are available.
- a lexical fallback keeps the pipeline operable in constrained environments.

The retrieval pipeline can operate unchanged with reranking disabled, enabled, or swapped to another reranker implementation later.

### Evaluation Layer

Evaluation is intentionally decoupled from retrieval execution.

- `JsonEvaluationDatasetLoader` validates custom JSON or YAML benchmark datasets.
- `MetricEvaluator` computes Precision@K, Recall@K, MRR, and NDCG@K independently.
- relevance metrics are aggregated separately from latency metrics so quality and cost trade-offs remain explicit.

### Benchmarking Layer

`BenchmarkRunner` orchestrates dataset evaluation on top of the retrieval pipeline.

- single experiments benchmark one pipeline configuration.
- parameter sweeps vary one parameter at a time while the rest stay fixed.
- grid search executes the full Cartesian product of configuration values.
- ablation studies compare a baseline configuration against targeted removals or substitutions.

Each experiment records retrieval quality metrics, latency metrics, configuration, device, notes, and generated analysis metadata.

### Analysis And Decision Layer

Phase 4 adds a dedicated analysis layer over stored experiment history.

- `RecommendationEngine` computes a weighted overall score balancing quality, retrieval latency, embedding cost, and index build cost.
- `LeaderboardEngine` ranks stored experiments by overall score or individual metrics.
- `TradeoffAnalyzer` derives explainable observations across retrievers, query enhancement strategies, rerankers, and chunking choices.
- `BenchmarkAnalysisService` coordinates enrichment, persistence, report generation, and visualization generation.

This layer is intentionally built on experiment history rather than benchmark execution so reports can be regenerated without re-running retrieval experiments.

### Reporting Layer

`ReportGenerator` emits three portable formats per experiment:

- Markdown for human-readable summaries.
- CSV for spreadsheet and BI workflows.
- JSON for machine-readable automation.

Reports include executive summary text, configuration snapshot, retrieval quality metrics, latency metrics, ranking position, recommendation details, and trade-off observations.

### Visualization Layer

`VisualizationGenerator` emits:

- HTML dashboards for portable review.
- PNG charts for reports and presentations.

The current charts include leaderboard score views and quality-versus-latency comparisons. Rendering uses a non-interactive backend so artifact generation works in CLI, tests, and FastAPI worker threads.

### Delivery Surfaces

- The Typer CLI supports ingestion, preprocessing, chunking, embedding, indexing, retrieval, evaluation, benchmarking, sweeps, grid search, experiment listing, comparison, leaderboard generation, recommendation, reporting, visualization, and history inspection.
- The FastAPI app exposes retrieval endpoints plus benchmarking and phase-4 analysis endpoints with OpenAPI documentation.

## Design Rationale

- FAISS was chosen for dense indexing because it is widely used, fast, and easy to persist.
- BM25 remains necessary because exact-term matching is often decisive on enterprise text.
- Hybrid retrieval and Reciprocal Rank Fusion provide a strong default without coupling to one retriever family.
- HyDE and reranking are optional because they can improve quality, but they carry real latency and model cost.
- Phase-4 analysis is history-driven because recommendations and reports should be reproducible without re-executing experiments.
- Report and visualization artifacts are persisted by path in experiment history so downstream workflows can reference them consistently.

## Extensibility

The project keeps responsibilities separated so future work can add larger benchmark datasets, richer dashboards, deployment packaging, and production orchestration without changing the established ingestion, corpus, embedding, retrieval, evaluation, or history contracts.
