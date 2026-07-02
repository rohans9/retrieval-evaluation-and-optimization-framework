"""Benchmark execution, parameter sweeps, grid search, and ablations."""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np

from retrieval_evaluation_framework.benchmarking.models import (
    BenchmarkMode,
    BenchmarkResult,
    ExperimentRecord,
    PerformanceMetrics,
    QueryBenchmarkResult,
)
from retrieval_evaluation_framework.benchmarking.tracking import ExperimentTracker
from retrieval_evaluation_framework.config.settings import AppConfig
from retrieval_evaluation_framework.evaluation.datasets import JsonEvaluationDatasetLoader
from retrieval_evaluation_framework.evaluation.metrics import MetricEvaluator
from retrieval_evaluation_framework.logging import get_logger
from retrieval_evaluation_framework.models import Chunk, RetrievalResult
from retrieval_evaluation_framework.retrieval.pipeline import RetrievalPipeline

LOGGER = get_logger(component="benchmark_runner")


class BenchmarkRunner:
    """Run retrieval evaluation experiments against a labeled dataset."""

    def __init__(self, tracker: ExperimentTracker) -> None:
        """Initialize the benchmark runner.

        Args:
            tracker: Experiment tracker for persisting history.
        """
        self.tracker = tracker
        self.dataset_loader = JsonEvaluationDatasetLoader()

    def run_single_experiment(
        self,
        config: AppConfig,
        corpus_path: Path,
        dataset_path: Path,
        notes: str | None = None,
    ) -> BenchmarkResult:
        """Run one benchmark experiment."""
        return self._run(config, corpus_path, dataset_path, mode="single", notes=notes)

    def parameter_sweep(
        self,
        config: AppConfig,
        corpus_path: Path,
        dataset_path: Path,
        sweep_parameters: dict[str, list[Any]],
        notes: str | None = None,
    ) -> list[BenchmarkResult]:
        """Run a parameter sweep varying one parameter at a time."""
        results: list[BenchmarkResult] = []
        for parameter_name, values in sweep_parameters.items():
            for value in values:
                derived_config = self._with_overrides(config, {parameter_name: value})
                run_notes = self._merge_notes(notes, f"sweep:{parameter_name}={value}")
                results.append(
                    self._run(
                        derived_config,
                        corpus_path,
                        dataset_path,
                        mode="sweep",
                        notes=run_notes,
                    )
                )
        return results

    def grid_search(
        self,
        config: AppConfig,
        corpus_path: Path,
        dataset_path: Path,
        parameter_grid: dict[str, list[Any]],
        notes: str | None = None,
    ) -> list[BenchmarkResult]:
        """Run a Cartesian-product grid search over configuration values."""
        keys = list(parameter_grid.keys())
        results: list[BenchmarkResult] = []
        for combination in itertools.product(*(parameter_grid[key] for key in keys)):
            overrides = dict(zip(keys, combination, strict=True))
            derived_config = self._with_overrides(config, overrides)
            run_notes = self._merge_notes(
                notes,
                "grid:" + ", ".join(f"{key}={value}" for key, value in overrides.items()),
            )
            results.append(
                self._run(
                    derived_config,
                    corpus_path,
                    dataset_path,
                    mode="grid_search",
                    notes=run_notes,
                )
            )
        return results

    def ablation_study(
        self,
        config: AppConfig,
        corpus_path: Path,
        dataset_path: Path,
        variants: dict[str, dict[str, Any]],
        notes: str | None = None,
    ) -> list[BenchmarkResult]:
        """Run baseline-plus-variant ablation experiments."""
        results = [
            self._run(
                config,
                corpus_path,
                dataset_path,
                mode="ablation",
                notes=self._merge_notes(notes, "ablation:baseline"),
            )
        ]
        for variant_name, overrides in variants.items():
            derived_config = self._with_overrides(config, overrides)
            results.append(
                self._run(
                    derived_config,
                    corpus_path,
                    dataset_path,
                    mode="ablation",
                    notes=self._merge_notes(notes, f"ablation:{variant_name}"),
                )
            )
        return results

    def _run(
        self,
        config: AppConfig,
        corpus_path: Path,
        dataset_path: Path,
        mode: BenchmarkMode,
        notes: str | None,
    ) -> BenchmarkResult:
        dataset = self.dataset_loader.load(dataset_path)
        experiment = self._create_experiment_record(config, corpus_path, dataset_path, mode, notes)
        self.tracker.create_experiment(experiment)
        LOGGER.info("benchmark_started", experiment_id=experiment.experiment_id, mode=mode)

        try:
            pipeline = RetrievalPipeline(config)
            corpus = pipeline.load_processed_corpus(corpus_path)
            experiment.selected_device = (
                pipeline.embedding_engine.device
                if pipeline.embedding_engine is not None
                else config.resolved_device
            )

            embedding_time_ms = self._measure_embedding_time(pipeline, corpus.chunks)
            index_build_time_ms = self._measure_index_build_time(pipeline, corpus.chunks)

            query_results: list[QueryBenchmarkResult] = []
            query_metrics = []
            latencies_ms: list[float] = []
            for sample in dataset.examples:
                response = pipeline.retrieve(sample.query, top_k=config.retrieval.top_k)
                latencies_ms.append(response.total_latency_ms)
                relevance_flags = [
                    self._is_relevant(result, sample.positive_documents)
                    for result in response.results[: config.retrieval.top_k]
                ]
                metrics = MetricEvaluator.evaluate_query(
                    relevance_flags=relevance_flags,
                    relevant_total=len(sample.positive_documents),
                    k=config.retrieval.top_k,
                )
                query_metrics.append(metrics)
                query_results.append(
                    QueryBenchmarkResult(
                        query=sample.query,
                        positive_documents=sample.positive_documents,
                        retrieved_chunk_ids=[
                            result.chunk.chunk_id
                            for result in response.results[: config.retrieval.top_k]
                        ],
                        metrics=metrics,
                        latency_ms=response.total_latency_ms,
                    )
                )

            quality_metrics = MetricEvaluator.aggregate(query_metrics)
            performance_metrics = self._build_performance_metrics(
                latencies_ms,
                embedding_time_ms,
                index_build_time_ms,
            )
            experiment.status = "completed"
            experiment.retrieval_quality_metrics = quality_metrics
            experiment.performance_metrics = performance_metrics
            self.tracker.save(experiment)
            self.tracker.export_index()
            LOGGER.info(
                "benchmark_completed",
                experiment_id=experiment.experiment_id,
                query_count=len(dataset.examples),
            )
            result = BenchmarkResult(
                experiment=experiment,
                dataset_name=dataset.name,
                sample_count=len(dataset.examples),
                query_results=query_results,
            )
            result.save_json(
                config.benchmark.results_directory / f"{experiment.experiment_id}_benchmark.json"
            )
            return result
        except Exception:
            experiment.status = "failed"
            self.tracker.save(experiment)
            raise

    @staticmethod
    def _measure_embedding_time(
        pipeline: RetrievalPipeline,
        chunks: list[Chunk],
    ) -> float:
        if pipeline.embedding_engine is None:
            return 0.0
        start = time.perf_counter()
        pipeline.embedding_engine.embed_chunks(chunks)
        return (time.perf_counter() - start) * 1000

    @staticmethod
    def _measure_index_build_time(
        pipeline: RetrievalPipeline,
        chunks: list[Chunk],
    ) -> float:
        start = time.perf_counter()
        pipeline.build_index(chunks)
        return (time.perf_counter() - start) * 1000

    @staticmethod
    def _build_performance_metrics(
        latencies_ms: list[float],
        embedding_time_ms: float,
        index_build_time_ms: float,
    ) -> PerformanceMetrics:
        if not latencies_ms:
            latencies_ms = [0.0]
        values = np.asarray(latencies_ms, dtype=np.float64)
        return PerformanceMetrics(
            average_retrieval_latency_ms=float(values.mean()),
            p50_latency_ms=float(np.percentile(values, 50)),
            p95_latency_ms=float(np.percentile(values, 95)),
            p99_latency_ms=float(np.percentile(values, 99)),
            embedding_generation_time_ms=embedding_time_ms,
            index_build_time_ms=index_build_time_ms,
        )

    def _create_experiment_record(
        self,
        config: AppConfig,
        corpus_path: Path,
        dataset_path: Path,
        mode: BenchmarkMode,
        notes: str | None,
    ) -> ExperimentRecord:
        enhancement_method = (
            config.query_enhancement.method if config.query_enhancement.enabled else "none"
        )
        reranker_name = config.reranking.model_name if config.reranking.enabled else "none"
        return ExperimentRecord(
            experiment_id=self.tracker.next_experiment_id(),
            mode=mode,
            status="running",
            configuration=config.to_metadata(),
            dataset_path=str(dataset_path),
            corpus_path=str(corpus_path),
            chunking_strategy=config.chunking.strategy,
            chunk_size=config.chunking.chunk_size,
            embedding_model=config.embedding.model_name,
            retriever=config.retrieval.retriever,
            query_enhancement=enhancement_method,
            reranker=reranker_name,
            top_k=config.retrieval.top_k,
            selected_device=config.resolved_device,
            notes=notes,
        )

    @staticmethod
    def _is_relevant(result: RetrievalResult, positive_documents: list[str]) -> bool:
        chunk = result.chunk
        identifiers = {
            chunk.chunk_id,
            chunk.document_id,
            chunk.text,
            chunk.metadata.get("title", ""),
            chunk.metadata.get("source", ""),
        }
        return any(positive in identifiers for positive in positive_documents)

    @staticmethod
    def _merge_notes(notes: str | None, suffix: str) -> str:
        if not notes:
            return suffix
        return f"{notes} | {suffix}"

    @staticmethod
    def _with_overrides(config: AppConfig, overrides: dict[str, Any]) -> AppConfig:
        payload = config.to_metadata()
        for path, value in overrides.items():
            BenchmarkRunner._apply_override(payload, path, value)
        return AppConfig.model_validate(payload)

    @staticmethod
    def _apply_override(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
        keys = dotted_path.split(".")
        cursor: dict[str, Any] = payload
        for key in keys[:-1]:
            next_value = cursor.get(key)
            if not isinstance(next_value, dict):
                next_value = {}
                cursor[key] = next_value
            cursor = next_value
        cursor[keys[-1]] = value
