"""API router for benchmarking and phase-4 analysis endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from retrieval_evaluation_framework.api.dependencies import (
    DEFAULT_CONFIG_PATH,
    ensure_exists,
    get_analysis_service,
    get_tracker,
    load_config,
    load_parameter_mapping,
)
from retrieval_evaluation_framework.api.schemas import (
    BenchmarkExecutionResponse,
    BenchmarkRequest,
    CompareRequest,
    EvaluateRequest,
    ReportResponse,
)
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
from retrieval_evaluation_framework.recommendation.leaderboard import parse_sort_metric

router = APIRouter(prefix="", tags=["benchmarking"])


@router.post("/evaluate", response_model=BenchmarkResult)
def evaluate(request: EvaluateRequest) -> BenchmarkResult:
    """Run a single retrieval evaluation against a labeled dataset."""
    config = load_config(request.config_path)
    tracker = get_tracker(config, request.experiment_directory)
    runner = BenchmarkRunner(tracker)
    ensure_exists(request.corpus_path, "Corpus file")
    ensure_exists(request.dataset_path, "Dataset file")
    return runner.run_single_experiment(
        config=config,
        corpus_path=request.corpus_path,
        dataset_path=request.dataset_path,
        notes=request.notes or config.benchmark.notes,
    )


@router.post("/benchmark", response_model=BenchmarkExecutionResponse)
def benchmark(request: BenchmarkRequest) -> BenchmarkExecutionResponse:
    """Run single, sweep, or grid-search benchmark workflows."""
    config = load_config(request.config_path)
    tracker = get_tracker(config, request.experiment_directory)
    runner = BenchmarkRunner(tracker)
    ensure_exists(request.corpus_path, "Corpus file")
    ensure_exists(request.dataset_path, "Dataset file")
    parameter_mapping = load_parameter_mapping(request.parameters)

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


@router.post("/compare", response_model=ComparisonTable)
def compare(request: CompareRequest) -> ComparisonTable:
    """Compare stored experiments side by side."""
    config = load_config(request.config_path)
    tracker = get_tracker(config, request.experiment_directory)
    return ExperimentComparisonEngine(tracker).compare(request.experiment_ids)


@router.get("/experiments", response_model=list[ExperimentRecord])
def list_experiments(
    config_path: Path = DEFAULT_CONFIG_PATH,
    experiment_directory: Path | None = None,
) -> list[ExperimentRecord]:
    """List persisted experiments."""
    config = load_config(config_path)
    tracker = get_tracker(config, experiment_directory)
    return tracker.list_experiments()


@router.get("/experiment/{experiment_id}", response_model=ExperimentRecord)
def get_experiment(
    experiment_id: str,
    config_path: Path = DEFAULT_CONFIG_PATH,
    experiment_directory: Path | None = None,
) -> ExperimentRecord:
    """Get one persisted experiment by ID."""
    config = load_config(config_path)
    tracker = get_tracker(config, experiment_directory)
    try:
        return tracker.get_experiment(experiment_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/leaderboard", response_model=Leaderboard)
def leaderboard(
    config_path: Path = DEFAULT_CONFIG_PATH,
    experiment_directory: Path | None = None,
    sort_by: str = "overall_score",
) -> Leaderboard:
    """Return a leaderboard over benchmark history."""
    config = load_config(config_path)
    try:
        metric = parse_sort_metric(sort_by)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return get_analysis_service(config, experiment_directory).get_leaderboard(sort_by=metric)


@router.get("/recommendation", response_model=RecommendationResult)
def recommendation(
    config_path: Path = DEFAULT_CONFIG_PATH,
    experiment_directory: Path | None = None,
) -> RecommendationResult:
    """Return the best overall pipeline recommendation."""
    config = load_config(config_path)
    return get_analysis_service(config, experiment_directory).get_recommendation()


@router.get("/reports", response_model=ReportResponse)
def reports(
    config_path: Path = DEFAULT_CONFIG_PATH,
    experiment_directory: Path | None = None,
    experiment_ids: str | None = None,
) -> ReportResponse:
    """Generate reports for all or selected experiments."""
    config = load_config(config_path)
    selected_ids = None
    if experiment_ids:
        selected_ids = [item.strip() for item in experiment_ids.split(",") if item.strip()]
    artifacts = get_analysis_service(config, experiment_directory).generate_reports(selected_ids)
    return ReportResponse(artifacts=artifacts)


@router.get("/visualizations", response_model=VisualizationArtifacts)
def visualizations(
    config_path: Path = DEFAULT_CONFIG_PATH,
    experiment_directory: Path | None = None,
) -> VisualizationArtifacts:
    """Generate shared HTML and PNG benchmark visualizations."""
    config = load_config(config_path)
    return get_analysis_service(config, experiment_directory).generate_visualizations()


@router.get("/history", response_model=list[ExperimentHistoryEntry])
def history(
    config_path: Path = DEFAULT_CONFIG_PATH,
    experiment_directory: Path | None = None,
) -> list[ExperimentHistoryEntry]:
    """Return enriched experiment history with report and visualization metadata."""
    config = load_config(config_path)
    return get_analysis_service(config, experiment_directory).get_history()
