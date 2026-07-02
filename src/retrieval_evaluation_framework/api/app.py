"""FastAPI application for the retrieval framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from retrieval_evaluation_framework.benchmarking.analysis import BenchmarkAnalysisService
from retrieval_evaluation_framework.benchmarking.comparison import ExperimentComparisonEngine
from retrieval_evaluation_framework.benchmarking.models import (
    BenchmarkResult,
    ComparisonTable,
    ExperimentHistoryEntry,
    ExperimentRecord,
    Leaderboard,
    RecommendationResult,
    VisualizationArtifacts,
)
from retrieval_evaluation_framework.benchmarking.runner import BenchmarkRunner
from retrieval_evaluation_framework.benchmarking.tracking import ExperimentTracker
from retrieval_evaluation_framework.config.settings import AppConfig
from retrieval_evaluation_framework.embeddings.engine import EmbeddingEngine
from retrieval_evaluation_framework.logging import configure_logging
from retrieval_evaluation_framework.models import ProcessedCorpus, RetrievalResponse
from retrieval_evaluation_framework.recommendation.leaderboard import parse_sort_metric
from retrieval_evaluation_framework.retrieval.pipeline import RetrievalPipeline

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_EMBEDDINGS_PATH = Path("data/embeddings")
DEFAULT_INDEX_PATH = Path("data/index")

app = FastAPI(
    title="Retrieval Evaluation & Optimization Framework",
    description="API for indexing processed corpora and serving retrieval requests.",
)

_PIPELINE_CACHE: dict[str, RetrievalPipeline] = {}


def _ensure_exists(path: Path, description: str) -> None:
    """Raise an HTTP 404 when the requested path does not exist.

    Args:
        path: Path to validate.
        description: Human-readable path description.
    """
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{description} not found: {path}")


def _load_config(config_path: Path) -> AppConfig:
    """Load API configuration and configure logging.

    Args:
        config_path: YAML configuration path.

    Returns:
        Parsed application configuration.
    """
    _ensure_exists(config_path, "Config file")
    config = AppConfig.load_yaml(config_path)
    configure_logging(config.logging.level)
    return config


def _get_retrieval_pipeline(config_path: Path) -> RetrievalPipeline:
    """Return a cached retrieval pipeline for a config path.

    Args:
        config_path: YAML configuration path.

    Returns:
        Cached or newly created retrieval pipeline.
    """
    resolved_path = str(config_path.resolve())
    pipeline = _PIPELINE_CACHE.get(resolved_path)
    if pipeline is None:
        pipeline = RetrievalPipeline(_load_config(config_path))
        _PIPELINE_CACHE[resolved_path] = pipeline
    return pipeline


def _get_tracker(config: AppConfig, directory: Path | None = None) -> ExperimentTracker:
    """Return an experiment tracker for benchmark history.

    Args:
        config: Loaded application configuration.
        directory: Optional override for the experiment directory.

    Returns:
        Configured experiment tracker.
    """
    return ExperimentTracker(directory or config.benchmark.experiment_directory)


def _get_analysis_service(
    config: AppConfig,
    directory: Path | None = None,
) -> BenchmarkAnalysisService:
    """Return a phase-4 analysis service over benchmark history."""
    return BenchmarkAnalysisService(config, _get_tracker(config, directory))


def _load_parameter_mapping(payload: dict[str, Any] | None) -> dict[str, list[Any]]:
    """Validate an API-supplied parameter mapping.

    Args:
        payload: Raw parameter mapping from an API request.

    Returns:
        Validated parameter mapping.
    """
    if payload is None:
        return {}
    mapping: dict[str, list[Any]] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, list):
            msg = "Benchmark parameters must map dotted paths to lists of values"
            raise HTTPException(status_code=400, detail=msg)
        mapping[key] = value
    return mapping


def _load_corpus(corpus_path: Path) -> ProcessedCorpus:
    """Load a processed corpus from disk.

    Args:
        corpus_path: Path to a processed corpus JSON file.

    Returns:
        Parsed processed corpus.
    """
    _ensure_exists(corpus_path, "Corpus file")
    return ProcessedCorpus.model_validate_json(corpus_path.read_text(encoding="utf-8"))


def _ensure_index_loaded(pipeline: RetrievalPipeline, index_path: Path) -> None:
    """Load a persisted retrieval index when the pipeline is not yet built.

    Args:
        pipeline: Retrieval pipeline to ready for search.
        index_path: Directory containing the persisted index.
    """
    if pipeline.is_built:
        return

    _ensure_exists(index_path, "Index directory")
    try:
        pipeline.load_index(index_path)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=f"Index artifact missing in {index_path}: {error}",
        ) from error


class EmbedRequest(BaseModel):
    """Request model for the ``/embed`` endpoint."""

    corpus_path: Path = Field(..., description="Path to the processed corpus JSON file.")
    config_path: Path = Field(
        default=DEFAULT_CONFIG_PATH,
        description="Path to the YAML configuration file.",
    )
    output_path: Path = Field(
        default=DEFAULT_EMBEDDINGS_PATH,
        description="Directory where embedding artifacts should be written.",
    )


class EmbedResponse(BaseModel):
    """Response model for the ``/embed`` endpoint."""

    output_path: str
    chunk_count: int
    model_name: str
    dimension: int
    device: str


class IndexRequest(BaseModel):
    """Request model for the ``/index`` endpoint."""

    corpus_path: Path = Field(..., description="Path to the processed corpus JSON file.")
    config_path: Path = Field(
        default=DEFAULT_CONFIG_PATH,
        description="Path to the YAML configuration file.",
    )
    index_path: Path = Field(
        default=DEFAULT_INDEX_PATH,
        description="Directory where index artifacts should be written.",
    )


class IndexResponse(BaseModel):
    """Response model for the ``/index`` endpoint."""

    index_path: str
    retriever: str
    chunk_count: int


class RetrieveRequest(BaseModel):
    """Request model for the ``/retrieve`` and ``/search`` endpoints."""

    query: str = Field(..., description="The user query for retrieval.")
    config_path: Path = Field(
        default=DEFAULT_CONFIG_PATH,
        description="Path to the YAML configuration file.",
    )
    index_path: Path = Field(
        default=DEFAULT_INDEX_PATH,
        description="Directory containing the persisted retrieval index.",
    )
    top_k: int | None = Field(
        default=None,
        description="Optional override for the number of chunks to return.",
    )


class EvaluateRequest(BaseModel):
    """Request model for the ``/evaluate`` endpoint."""

    corpus_path: Path
    dataset_path: Path
    config_path: Path = Field(default=DEFAULT_CONFIG_PATH)
    notes: str | None = None
    experiment_directory: Path | None = None


class BenchmarkRequest(BaseModel):
    """Request model for the ``/benchmark`` endpoint."""

    corpus_path: Path
    dataset_path: Path
    config_path: Path = Field(default=DEFAULT_CONFIG_PATH)
    mode: str = Field(default="single", description="single, sweep, or grid_search")
    parameters: dict[str, list[Any]] | None = None
    notes: str | None = None
    experiment_directory: Path | None = None


class BenchmarkExecutionResponse(BaseModel):
    """Response model for benchmark execution endpoints."""

    mode: str
    results: list[BenchmarkResult]


class CompareRequest(BaseModel):
    """Request model for the ``/compare`` endpoint."""

    experiment_ids: list[str]
    config_path: Path = Field(default=DEFAULT_CONFIG_PATH)
    experiment_directory: Path | None = None


class ReportResponse(BaseModel):
    """Response model for generated report artifacts."""

    artifacts: dict[str, dict[str, str]]


@app.post("/embed", response_model=EmbedResponse, tags=["retrieval"])
def embed_corpus(request: EmbedRequest) -> EmbedResponse:
    """Generate and persist embeddings for a processed corpus."""
    config = _load_config(request.config_path)
    corpus = _load_corpus(request.corpus_path)
    engine = EmbeddingEngine(config.embedding)
    embedding_store = engine.embed_chunks(corpus.chunks)
    embedding_store.save(request.output_path)
    return EmbedResponse(
        output_path=str(request.output_path),
        chunk_count=len(embedding_store.chunk_ids),
        model_name=embedding_store.model_name,
        dimension=embedding_store.dimension,
        device=engine.device,
    )


@app.post("/index", response_model=IndexResponse, tags=["retrieval"])
def index_corpus(request: IndexRequest) -> IndexResponse:
    """Build and persist a retrieval index for a processed corpus."""
    pipeline = _get_retrieval_pipeline(request.config_path)
    corpus = pipeline.index_processed_corpus(request.corpus_path)
    pipeline.save_index(request.index_path)
    return IndexResponse(
        index_path=str(request.index_path),
        retriever=pipeline.retriever.name,
        chunk_count=len(corpus.chunks),
    )


@app.post("/retrieve", response_model=RetrievalResponse, tags=["retrieval"])
def retrieve_chunks(request: RetrieveRequest) -> RetrievalResponse:
    """Retrieve relevant chunks for a query."""
    pipeline = _get_retrieval_pipeline(request.config_path)
    _ensure_index_loaded(pipeline, request.index_path)
    try:
        return pipeline.retrieve(request.query, top_k=request.top_k)
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/search", response_model=RetrievalResponse, tags=["retrieval"])
def search(request: RetrieveRequest) -> RetrievalResponse:
    """Alias endpoint for retrieval, kept for API discoverability."""
    return retrieve_chunks(request)


@app.post("/evaluate", response_model=BenchmarkResult, tags=["benchmarking"])
def evaluate(request: EvaluateRequest) -> BenchmarkResult:
    """Run a single retrieval evaluation against a labeled dataset."""
    config = _load_config(request.config_path)
    tracker = _get_tracker(config, request.experiment_directory)
    runner = BenchmarkRunner(tracker)
    _ensure_exists(request.corpus_path, "Corpus file")
    _ensure_exists(request.dataset_path, "Dataset file")
    return runner.run_single_experiment(
        config=config,
        corpus_path=request.corpus_path,
        dataset_path=request.dataset_path,
        notes=request.notes or config.benchmark.notes,
    )


@app.post(
    "/benchmark",
    response_model=BenchmarkExecutionResponse,
    tags=["benchmarking"],
)
def benchmark(request: BenchmarkRequest) -> BenchmarkExecutionResponse:
    """Run single, sweep, or grid-search benchmark workflows."""
    config = _load_config(request.config_path)
    tracker = _get_tracker(config, request.experiment_directory)
    runner = BenchmarkRunner(tracker)
    _ensure_exists(request.corpus_path, "Corpus file")
    _ensure_exists(request.dataset_path, "Dataset file")
    parameter_mapping = _load_parameter_mapping(request.parameters)

    if request.mode == "single":
        result = runner.run_single_experiment(
            config=config,
            corpus_path=request.corpus_path,
            dataset_path=request.dataset_path,
            notes=request.notes or config.benchmark.notes,
        )
        return BenchmarkExecutionResponse(mode=request.mode, results=[result])

    if request.mode == "sweep":
        results = runner.parameter_sweep(
            config=config,
            corpus_path=request.corpus_path,
            dataset_path=request.dataset_path,
            sweep_parameters=parameter_mapping,
            notes=request.notes or config.benchmark.notes,
        )
        return BenchmarkExecutionResponse(mode=request.mode, results=results)

    if request.mode == "grid_search":
        results = runner.grid_search(
            config=config,
            corpus_path=request.corpus_path,
            dataset_path=request.dataset_path,
            parameter_grid=parameter_mapping,
            notes=request.notes or config.benchmark.notes,
        )
        return BenchmarkExecutionResponse(mode=request.mode, results=results)

    raise HTTPException(status_code=400, detail=f"Unsupported benchmark mode: {request.mode}")


@app.post("/compare", response_model=ComparisonTable, tags=["benchmarking"])
def compare(request: CompareRequest) -> ComparisonTable:
    """Compare stored experiments side by side."""
    config = _load_config(request.config_path)
    tracker = _get_tracker(config, request.experiment_directory)
    return ExperimentComparisonEngine(tracker).compare(request.experiment_ids)


@app.get("/experiments", response_model=list[ExperimentRecord], tags=["benchmarking"])
def list_experiments(
    config_path: Path = DEFAULT_CONFIG_PATH,
    experiment_directory: Path | None = None,
) -> list[ExperimentRecord]:
    """List persisted experiments."""
    config = _load_config(config_path)
    tracker = _get_tracker(config, experiment_directory)
    return tracker.list_experiments()


@app.get(
    "/experiment/{experiment_id}",
    response_model=ExperimentRecord,
    tags=["benchmarking"],
)
def get_experiment(
    experiment_id: str,
    config_path: Path = DEFAULT_CONFIG_PATH,
    experiment_directory: Path | None = None,
) -> ExperimentRecord:
    """Get one persisted experiment by ID."""
    config = _load_config(config_path)
    tracker = _get_tracker(config, experiment_directory)
    try:
        return tracker.get_experiment(experiment_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/leaderboard", response_model=Leaderboard, tags=["benchmarking"])
def leaderboard(
    config_path: Path = DEFAULT_CONFIG_PATH,
    experiment_directory: Path | None = None,
    sort_by: str = "overall_score",
) -> Leaderboard:
    """Return a leaderboard over benchmark history."""
    config = _load_config(config_path)
    try:
        metric = parse_sort_metric(sort_by)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _get_analysis_service(config, experiment_directory).get_leaderboard(sort_by=metric)


@app.get(
    "/recommendation",
    response_model=RecommendationResult,
    tags=["benchmarking"],
)
def recommendation(
    config_path: Path = DEFAULT_CONFIG_PATH,
    experiment_directory: Path | None = None,
) -> RecommendationResult:
    """Return the best overall pipeline recommendation."""
    config = _load_config(config_path)
    return _get_analysis_service(config, experiment_directory).get_recommendation()


@app.get("/reports", response_model=ReportResponse, tags=["benchmarking"])
def reports(
    config_path: Path = DEFAULT_CONFIG_PATH,
    experiment_directory: Path | None = None,
    experiment_ids: str | None = None,
) -> ReportResponse:
    """Generate reports for all or selected experiments."""
    config = _load_config(config_path)
    selected_ids = None
    if experiment_ids:
        selected_ids = [item.strip() for item in experiment_ids.split(",") if item.strip()]
    artifacts = _get_analysis_service(config, experiment_directory).generate_reports(selected_ids)
    return ReportResponse(artifacts=artifacts)


@app.get(
    "/visualizations",
    response_model=VisualizationArtifacts,
    tags=["benchmarking"],
)
def visualizations(
    config_path: Path = DEFAULT_CONFIG_PATH,
    experiment_directory: Path | None = None,
) -> VisualizationArtifacts:
    """Generate shared HTML and PNG benchmark visualizations."""
    config = _load_config(config_path)
    return _get_analysis_service(config, experiment_directory).generate_visualizations()


@app.get(
    "/history",
    response_model=list[ExperimentHistoryEntry],
    tags=["benchmarking"],
)
def history(
    config_path: Path = DEFAULT_CONFIG_PATH,
    experiment_directory: Path | None = None,
) -> list[ExperimentHistoryEntry]:
    """Return enriched experiment history with report and visualization metadata."""
    config = _load_config(config_path)
    return _get_analysis_service(config, experiment_directory).get_history()
