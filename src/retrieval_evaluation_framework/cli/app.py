"""CLI entrypoints for the document processing and retrieval pipelines."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from retrieval_evaluation_framework.benchmarking.analysis import BenchmarkAnalysisService
from retrieval_evaluation_framework.benchmarking.comparison import ExperimentComparisonEngine
from retrieval_evaluation_framework.benchmarking.runner import BenchmarkRunner
from retrieval_evaluation_framework.benchmarking.tracking import ExperimentTracker
from retrieval_evaluation_framework.config.settings import AppConfig
from retrieval_evaluation_framework.logging import configure_logging, get_logger
from retrieval_evaluation_framework.pipeline import DocumentProcessingPipeline
from retrieval_evaluation_framework.recommendation.leaderboard import parse_sort_metric
from retrieval_evaluation_framework.retrieval.pipeline import RetrievalPipeline

app = typer.Typer(help="Retrieval Evaluation & Optimization Framework CLI")
retrieval_app = typer.Typer(help="Retrieval engine commands")
app.add_typer(retrieval_app, name="retrieval")
console = Console()
LOGGER = get_logger(component="cli")
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_INDEX_PATH = Path("data/index")
DEFAULT_EMBEDDINGS_PATH = Path("data/embeddings")
SOURCE_PATH_ARGUMENT = typer.Argument(..., exists=True, readable=True)
CONFIG_PATH_OPTION = typer.Option("--config", "-c", exists=True, readable=True)
CORPUS_PATH_OPTION = typer.Option("--corpus-path", exists=True, readable=True)
DATASET_PATH_OPTION = typer.Option("--dataset-path", exists=True, readable=True)
INDEX_PATH_OPTION = typer.Option("--index-path")
QUERY_OPTION = typer.Option("--query", "-q")
TOP_K_OPTION = typer.Option("--top-k")
EMBEDDINGS_PATH_OPTION = typer.Option("--output-path")
PARAMETERS_PATH_OPTION = typer.Option("--parameters-path", exists=True, readable=True)
EXPERIMENT_DIRECTORY_OPTION = typer.Option("--experiment-directory")
NOTES_OPTION = typer.Option("--notes")
EXPERIMENT_IDS_OPTION = typer.Option("--experiment-ids")
SORT_BY_OPTION = typer.Option("--sort-by")

SourcePath = Annotated[Path, SOURCE_PATH_ARGUMENT]
ConfigPath = Annotated[Path, CONFIG_PATH_OPTION]
CorpusPath = Annotated[Path, CORPUS_PATH_OPTION]
DatasetPath = Annotated[Path, DATASET_PATH_OPTION]
def _run_with_progress[T](description: str, action: Callable[[], T]) -> T:
    """Execute an action while rendering a Rich progress bar."""
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=1)
        result = action()
        progress.advance(task)
    return result


@contextmanager
def _command_span(command_name: str) -> Iterator[None]:
    """Log command start/end and render user-friendly errors."""
    start = time.perf_counter()
    LOGGER.info("command_started", command=command_name)
    try:
        yield
    except FileNotFoundError as error:
        LOGGER.error("command_failed", command=command_name, reason=str(error))
        console.print(f"[red]Error:[/red] {error}")
        console.print("[yellow]Suggestion:[/yellow] Verify the input path and retry.")
        raise typer.Exit(code=1) from error
    except (RuntimeError, ValueError, typer.BadParameter) as error:
        LOGGER.error("command_failed", command=command_name, reason=str(error))
        console.print(f"[red]Error:[/red] {error}")
        console.print("[yellow]Suggestion:[/yellow] Run with --help and check argument values.")
        raise typer.Exit(code=1) from error
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        LOGGER.info(
            "command_completed",
            command=command_name,
            duration_ms=round(duration_ms, 2),
        )


def _load_pipeline(config_path: Path) -> DocumentProcessingPipeline:
    config = AppConfig.load_yaml(config_path)
    configure_logging(config.logging.level)
    return DocumentProcessingPipeline(config)


def _load_retrieval_pipeline(config_path: Path) -> RetrievalPipeline:
    config = AppConfig.load_yaml(config_path)
    configure_logging(config.logging.level)
    return RetrievalPipeline(config)


def _render_summary(title: str, rows: list[tuple[str, str]]) -> None:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value")
    for label, value in rows:
        table.add_row(label, value)
    console.print(table)


def _load_benchmark_runner(
    config_path: Path,
    experiment_directory: Path | None = None,
) -> tuple[AppConfig, BenchmarkRunner, ExperimentTracker]:
    config = AppConfig.load_yaml(config_path)
    configure_logging(config.logging.level)
    tracker = ExperimentTracker(experiment_directory or config.benchmark.experiment_directory)
    return config, BenchmarkRunner(tracker), tracker


def _load_analysis_service(
    config_path: Path,
    experiment_directory: Path | None = None,
) -> BenchmarkAnalysisService:
    config, _, tracker = _load_benchmark_runner(config_path, experiment_directory)
    return BenchmarkAnalysisService(config, tracker)


def _load_parameter_mapping(path: Path) -> dict[str, list[Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = "Parameter file must contain a mapping of dotted paths to value lists"
        raise typer.BadParameter(msg)

    mapping: dict[str, list[Any]] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, list):
            msg = "Each parameter entry must map a dotted string path to a list of values"
            raise typer.BadParameter(msg)
        mapping[key] = value
    return mapping


def _render_benchmark_result(title: str, result: Any) -> None:
    _render_summary(
        title,
        [
            ("Experiment ID", result.experiment.experiment_id),
            ("Mode", result.experiment.mode),
            ("Dataset", result.dataset_name),
            (
                "Precision@K",
                f"{result.experiment.retrieval_quality_metrics.precision_at_k:.4f}",
            ),
            (
                "Recall@K",
                f"{result.experiment.retrieval_quality_metrics.recall_at_k:.4f}",
            ),
            (
                "MRR",
                f"{result.experiment.retrieval_quality_metrics.mean_reciprocal_rank:.4f}",
            ),
            (
                "NDCG@K",
                f"{result.experiment.retrieval_quality_metrics.ndcg_at_k:.4f}",
            ),
            (
                "Avg Latency (ms)",
                f"{result.experiment.performance_metrics.average_retrieval_latency_ms:.2f}",
            ),
            (
                "P95 Latency (ms)",
                f"{result.experiment.performance_metrics.p95_latency_ms:.2f}",
            ),
        ],
    )


def _run_evaluation_command(
    corpus_path: Path,
    dataset_path: Path,
    config_path: Path,
    experiment_directory: Path | None,
    notes: str | None,
) -> None:
    config, runner, _ = _load_benchmark_runner(config_path, experiment_directory)
    result = _run_with_progress(
        "Running evaluation benchmark",
        lambda: runner.run_single_experiment(
            config=config,
            corpus_path=corpus_path,
            dataset_path=dataset_path,
            notes=notes or config.benchmark.notes,
        ),
    )
    _render_benchmark_result("Evaluation Summary", result)


@app.command()
def ingest(
    source_path: SourcePath,
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
) -> None:
    """Ingest supported documents and persist the raw corpus.

    Example:
        retrieval ingest ./data/input --config ./configs/default.yaml
    """
    with _command_span("ingest"):
        pipeline = _load_pipeline(config_path)

        def action() -> tuple[list[Any], Path]:
            documents = pipeline.ingest_path(source_path)
            output_path = pipeline.save_documents(
                documents,
                pipeline.config.output.output_directory / "ingested_documents.json",
            )
            return documents, output_path

        documents, output_path = _run_with_progress("Ingesting documents", action)

        _render_summary(
            "Ingestion Summary",
            [("Documents", str(len(documents))), ("Output", str(output_path))],
        )


@app.command()
def preprocess(
    source_path: SourcePath,
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
) -> None:
    """Ingest and preprocess documents.

    Example:
        retrieval preprocess ./data/input --config ./configs/default.yaml
    """
    with _command_span("preprocess"):
        pipeline = _load_pipeline(config_path)

        def action() -> tuple[list[Any], Path]:
            documents = pipeline.preprocess_documents(pipeline.ingest_path(source_path))
            output_path = pipeline.save_documents(
                documents,
                pipeline.config.output.output_directory / "preprocessed_documents.json",
            )
            return documents, output_path

        documents, output_path = _run_with_progress("Preprocessing documents", action)

        _render_summary(
            "Preprocessing Summary",
            [("Documents", str(len(documents))), ("Output", str(output_path))],
        )


@app.command()
def chunk(
    source_path: SourcePath,
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
) -> None:
    """Run phase-1 processing and persist a processed corpus.

    Example:
        retrieval chunk ./data/input --config ./configs/default.yaml
    """
    with _command_span("chunk"):
        pipeline = _load_pipeline(config_path)

        def action() -> Any:
            if source_path.is_file():
                corpus = pipeline.process_file(source_path)
            else:
                corpus = pipeline.process_directory(source_path)
            output_path = (
                pipeline.config.output.output_directory
                / pipeline.config.output.processed_corpus_filename
            )
            return corpus, output_path

        corpus, output_path = _run_with_progress("Generating chunks", action)

        _render_summary(
            "Chunking Summary",
            [
                ("Documents", str(corpus.statistics["document_count"])),
                ("Chunks", str(corpus.statistics["chunk_count"])),
                ("Output", str(output_path)),
            ],
        )


@retrieval_app.command("embed")
def retrieval_embed(
    corpus_path: CorpusPath,
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
    output_path: Annotated[Path, EMBEDDINGS_PATH_OPTION] = DEFAULT_EMBEDDINGS_PATH,
) -> None:
    """Embed a processed corpus and persist the embeddings to disk.

    Example:
        retrieval retrieval embed --corpus-path ./data/processed/processed_corpus.json
    """
    with _command_span("retrieval.embed"):
        config = AppConfig.load_yaml(config_path)
        configure_logging(config.logging.level)

        from retrieval_evaluation_framework.embeddings.engine import EmbeddingEngine
        from retrieval_evaluation_framework.models import ProcessedCorpus

        corpus = ProcessedCorpus.model_validate_json(corpus_path.read_text(encoding="utf-8"))
        embedding_engine = EmbeddingEngine(config.embedding)

        def action() -> Any:
            embedding_store = embedding_engine.embed_chunks(corpus.chunks)
            embedding_store.save(output_path)
            return embedding_store

        embedding_store = _run_with_progress("Embedding chunks", action)

        _render_summary(
            "Embedding Summary",
            [
                ("Chunks", str(len(embedding_store.chunk_ids))),
                ("Model", embedding_store.model_name),
                ("Dimension", str(embedding_store.dimension)),
                ("Output", str(output_path)),
            ],
        )


@retrieval_app.command("index")
def retrieval_index(
    corpus_path: CorpusPath,
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
    index_path: Annotated[Path, INDEX_PATH_OPTION] = DEFAULT_INDEX_PATH,
) -> None:
    """Build and persist the configured retriever's index.

    Example:
        retrieval retrieval index --corpus-path ./data/processed/processed_corpus.json
    """
    with _command_span("retrieval.index"):
        pipeline = _load_retrieval_pipeline(config_path)

        def action() -> Any:
            corpus = pipeline.index_processed_corpus(corpus_path)
            pipeline.save_index(index_path)
            return corpus

        corpus = _run_with_progress("Building index", action)

        _render_summary(
            "Indexing Summary",
            [
                ("Retriever", pipeline.retriever.name),
                ("Chunks", str(len(corpus.chunks))),
                ("Index Path", str(index_path)),
            ],
        )


def _run_retrieve(
    config_path: Path,
    index_path: Path,
    query: str,
    top_k: int | None,
) -> None:
    pipeline = _load_retrieval_pipeline(config_path)
    _run_with_progress("Loading index", lambda: pipeline.load_index(index_path))

    start = time.perf_counter()
    response = pipeline.retrieve(query, top_k=top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    table = Table(title=f"Retrieved Chunks (retriever={response.retriever})")
    table.add_column("Rank")
    table.add_column("Score")
    table.add_column("Chunk ID")
    table.add_column("Text")
    for result in response.results:
        preview = result.chunk.text[:120].replace("\n", " ")
        table.add_row(str(result.rank), f"{result.score:.4f}", result.chunk.chunk_id, preview)
    console.print(table)

    _render_summary(
        "Retrieval Metadata",
        [
            ("Retriever", response.retriever),
            ("Reranked", str(response.reranked)),
            ("Enhanced Query", response.enhanced_query or "-"),
            ("Retrieval Latency (ms)", f"{response.retrieval_latency_ms:.2f}"),
            ("Enhancement Latency (ms)", f"{response.enhancement_latency_ms:.2f}"),
            ("Reranking Latency (ms)", f"{response.reranking_latency_ms:.2f}"),
            ("Total Latency (ms)", f"{response.total_latency_ms:.2f}"),
            ("Wall Clock (ms)", f"{elapsed_ms:.2f}"),
        ],
    )


@retrieval_app.command("retrieve")
def retrieval_retrieve(
    query: Annotated[str, QUERY_OPTION],
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
    index_path: Annotated[Path, INDEX_PATH_OPTION] = DEFAULT_INDEX_PATH,
    top_k: Annotated[int | None, TOP_K_OPTION] = None,
) -> None:
    """Retrieve the most relevant chunks for a query.

    Example:
        retrieval retrieval retrieve --query "What is the policy?" --index-path ./data/index
    """
    with _command_span("retrieval.retrieve"):
        _run_retrieve(config_path, index_path, query, top_k)


@retrieval_app.command("search")
def retrieval_search(
    query: Annotated[str, QUERY_OPTION],
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
    index_path: Annotated[Path, INDEX_PATH_OPTION] = DEFAULT_INDEX_PATH,
    top_k: Annotated[int | None, TOP_K_OPTION] = None,
) -> None:
    """Alias for retrieval retrieve.

    Example:
        retrieval retrieval search --query "hybrid schedules" --index-path ./data/index
    """
    with _command_span("retrieval.search"):
        _run_retrieve(config_path, index_path, query, top_k)


@retrieval_app.command("evaluate")
def retrieval_evaluate(
    corpus_path: CorpusPath,
    dataset_path: DatasetPath,
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
    experiment_directory: Annotated[Path | None, EXPERIMENT_DIRECTORY_OPTION] = None,
    notes: Annotated[str | None, NOTES_OPTION] = None,
) -> None:
    """Evaluate one retrieval pipeline against a labeled dataset."""
    with _command_span("retrieval.evaluate"):
        _run_evaluation_command(
            corpus_path=corpus_path,
            dataset_path=dataset_path,
            config_path=config_path,
            experiment_directory=experiment_directory,
            notes=notes,
        )


@retrieval_app.command("benchmark")
def retrieval_benchmark(
    corpus_path: CorpusPath,
    dataset_path: DatasetPath,
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
    experiment_directory: Annotated[Path | None, EXPERIMENT_DIRECTORY_OPTION] = None,
    notes: Annotated[str | None, NOTES_OPTION] = None,
) -> None:
    """Alias for single-experiment evaluation benchmarking."""
    with _command_span("retrieval.benchmark"):
        _run_evaluation_command(
            corpus_path=corpus_path,
            dataset_path=dataset_path,
            config_path=config_path,
            experiment_directory=experiment_directory,
            notes=notes,
        )


@retrieval_app.command("sweep")
def retrieval_sweep(
    corpus_path: CorpusPath,
    dataset_path: DatasetPath,
    parameters_path: Annotated[Path, PARAMETERS_PATH_OPTION],
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
    experiment_directory: Annotated[Path | None, EXPERIMENT_DIRECTORY_OPTION] = None,
    notes: Annotated[str | None, NOTES_OPTION] = None,
) -> None:
    """Run a parameter sweep from a YAML or JSON parameter mapping."""
    with _command_span("retrieval.sweep"):
        config, runner, _ = _load_benchmark_runner(config_path, experiment_directory)
        sweep_parameters = _load_parameter_mapping(parameters_path)
        results = _run_with_progress(
            "Running parameter sweep",
            lambda: runner.parameter_sweep(
                config=config,
                corpus_path=corpus_path,
                dataset_path=dataset_path,
                sweep_parameters=sweep_parameters,
                notes=notes or config.benchmark.notes,
            ),
        )
        if any(result.experiment.retrieval_quality_metrics is None for result in results):
            raise RuntimeError("Parameter sweep did not produce retrieval quality metrics")
        sweep_metrics = [
            result.experiment.retrieval_quality_metrics for result in results
        ]
        best_mrr = max(
            metric.mean_reciprocal_rank for metric in sweep_metrics if metric is not None
        )

        _render_summary(
            "Sweep Summary",
            [("Experiments", str(len(results))), ("Best MRR", f"{best_mrr:.4f}")],
        )


@retrieval_app.command("grid-search")
def retrieval_grid_search(
    corpus_path: CorpusPath,
    dataset_path: DatasetPath,
    parameters_path: Annotated[Path, PARAMETERS_PATH_OPTION],
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
    experiment_directory: Annotated[Path | None, EXPERIMENT_DIRECTORY_OPTION] = None,
    notes: Annotated[str | None, NOTES_OPTION] = None,
) -> None:
    """Run a full grid search from a YAML or JSON parameter mapping."""
    with _command_span("retrieval.grid_search"):
        config, runner, _ = _load_benchmark_runner(config_path, experiment_directory)
        parameter_grid = _load_parameter_mapping(parameters_path)
        results = _run_with_progress(
            "Running grid search",
            lambda: runner.grid_search(
                config=config,
                corpus_path=corpus_path,
                dataset_path=dataset_path,
                parameter_grid=parameter_grid,
                notes=notes or config.benchmark.notes,
            ),
        )
        if any(result.experiment.retrieval_quality_metrics is None for result in results):
            raise RuntimeError("Grid search did not produce retrieval quality metrics")
        grid_metrics = [
            result.experiment.retrieval_quality_metrics for result in results
        ]
        best_ndcg = max(
            metric.ndcg_at_k for metric in grid_metrics if metric is not None
        )

        _render_summary(
            "Grid Search Summary",
            [("Experiments", str(len(results))), ("Best NDCG@K", f"{best_ndcg:.4f}")],
        )


@retrieval_app.command("experiments")
def retrieval_experiments(
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
    experiment_directory: Annotated[Path | None, EXPERIMENT_DIRECTORY_OPTION] = None,
) -> None:
    """List locally tracked experiments."""
    with _command_span("retrieval.experiments"):
        _, _, tracker = _load_benchmark_runner(config_path, experiment_directory)
        records = _run_with_progress("Loading experiments", tracker.list_experiments)

    table = Table(title="Experiments")
    table.add_column("Experiment ID")
    table.add_column("Mode")
    table.add_column("Status")
    table.add_column("Retriever")
    table.add_column("MRR")
    for record in records:
        table.add_row(
            record.experiment_id,
            record.mode,
            record.status,
            record.retriever,
            (
                f"{record.retrieval_quality_metrics.mean_reciprocal_rank:.4f}"
                if record.retrieval_quality_metrics is not None
                else "-"
            ),
        )
    console.print(table)


@retrieval_app.command("compare")
def retrieval_compare(
    experiment_ids: Annotated[str, EXPERIMENT_IDS_OPTION],
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
    experiment_directory: Annotated[Path | None, EXPERIMENT_DIRECTORY_OPTION] = None,
) -> None:
    """Compare stored experiments side by side."""
    with _command_span("retrieval.compare"):
        _, _, tracker = _load_benchmark_runner(config_path, experiment_directory)
        comparison = _run_with_progress(
            "Comparing experiments",
            lambda: ExperimentComparisonEngine(tracker).compare(
                [item.strip() for item in experiment_ids.split(",") if item.strip()]
            ),
        )

    table = Table(title="Experiment Comparison")
    table.add_column("Experiment ID")
    table.add_column("Retriever")
    table.add_column("Embedding")
    table.add_column("MRR")
    table.add_column("NDCG@K")
    table.add_column("Avg Latency (ms)")
    for row in comparison.rows:
        table.add_row(
            row.experiment_id,
            row.retriever,
            row.embedding_model,
            f"{row.mean_reciprocal_rank:.4f}",
            f"{row.ndcg_at_k:.4f}",
            f"{row.average_latency_ms:.2f}",
        )
    console.print(table)


@retrieval_app.command("leaderboard")
def retrieval_leaderboard(
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
    experiment_directory: Annotated[Path | None, EXPERIMENT_DIRECTORY_OPTION] = None,
    sort_by: Annotated[str, SORT_BY_OPTION] = "overall_score",
) -> None:
    """Rank stored experiments across quality and performance metrics."""
    with _command_span("retrieval.leaderboard"):
        try:
            metric = parse_sort_metric(sort_by)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        leaderboard = _run_with_progress(
            "Generating leaderboard",
            lambda: _load_analysis_service(config_path, experiment_directory).get_leaderboard(
                sort_by=metric
            ),
        )

    table = Table(title="Experiment Leaderboard")
    table.add_column("Rank")
    table.add_column("Experiment ID")
    table.add_column("Retriever")
    table.add_column("MRR")
    table.add_column("NDCG@K")
    table.add_column("Avg Latency (ms)")
    table.add_column("Score")
    for row in leaderboard.rows:
        table.add_row(
            str(row.rank),
            row.experiment_id,
            row.retriever,
            f"{row.mean_reciprocal_rank:.4f}",
            f"{row.ndcg_at_k:.4f}",
            f"{row.average_latency_ms:.2f}",
            f"{row.overall_score:.4f}",
        )
    console.print(table)


@retrieval_app.command("recommend")
def retrieval_recommend(
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
    experiment_directory: Annotated[Path | None, EXPERIMENT_DIRECTORY_OPTION] = None,
) -> None:
    """Recommend the strongest production-ready experiment."""
    with _command_span("retrieval.recommend"):
        recommendation = _run_with_progress(
            "Computing recommendation",
            lambda: _load_analysis_service(config_path, experiment_directory).get_recommendation(),
        )
        _render_summary(
            "Recommendation Summary",
            [
                ("Experiment ID", recommendation.experiment_id),
                ("Pipeline", recommendation.recommended_pipeline),
                ("Score", f"{recommendation.overall_score:.4f}"),
                ("Reason", recommendation.reason),
                (
                    "Alternatives",
                    ", ".join(recommendation.alternative_configurations) or "None",
                ),
            ],
        )


@retrieval_app.command("report")
def retrieval_report(
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
    experiment_directory: Annotated[Path | None, EXPERIMENT_DIRECTORY_OPTION] = None,
    experiment_ids: Annotated[str | None, EXPERIMENT_IDS_OPTION] = None,
) -> None:
    """Generate markdown, CSV, and JSON reports for experiment history."""
    with _command_span("retrieval.report"):
        selected_ids = None
        if experiment_ids:
            selected_ids = [item.strip() for item in experiment_ids.split(",") if item.strip()]
        artifacts = _run_with_progress(
            "Generating reports",
            lambda: _load_analysis_service(config_path, experiment_directory).generate_reports(
                selected_ids
            ),
        )
        rows = [
            (experiment_id, ", ".join(paths.values()))
            for experiment_id, paths in artifacts.items()
        ]
        _render_summary("Report Artifacts", rows)


@retrieval_app.command("visualize")
def retrieval_visualize(
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
    experiment_directory: Annotated[Path | None, EXPERIMENT_DIRECTORY_OPTION] = None,
) -> None:
    """Generate HTML and PNG visualizations for stored experiments."""
    with _command_span("retrieval.visualize"):
        artifacts = _run_with_progress(
            "Generating visualizations",
            lambda: _load_analysis_service(
                config_path,
                experiment_directory,
            ).generate_visualizations(),
        )
        rows = [(name, path) for name, path in artifacts.html_paths.items()]
        rows.extend((name, path) for name, path in artifacts.png_paths.items())
        _render_summary("Visualization Artifacts", rows)


@retrieval_app.command("history")
def retrieval_history(
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
    experiment_directory: Annotated[Path | None, EXPERIMENT_DIRECTORY_OPTION] = None,
) -> None:
    """List enriched experiment history, including reports and visualizations."""
    with _command_span("retrieval.history"):
        history = _run_with_progress(
            "Loading experiment history",
            lambda: _load_analysis_service(config_path, experiment_directory).get_history(),
        )

    table = Table(title="Experiment History")
    table.add_column("Experiment ID")
    table.add_column("Rank")
    table.add_column("Score")
    table.add_column("Reports")
    table.add_column("Visualizations")
    for entry in history:
        table.add_row(
            entry.experiment.experiment_id,
            str(entry.experiment.leaderboard_rank or "-"),
            (
                f"{entry.experiment.overall_score:.4f}"
                if entry.experiment.overall_score is not None
                else "-"
            ),
            str(len(entry.report_paths)),
            str(len(entry.visualization_paths)),
        )
    console.print(table)


if __name__ == "__main__":
    app()
