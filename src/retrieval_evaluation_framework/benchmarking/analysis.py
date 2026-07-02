"""Phase-4 analysis orchestration over benchmark history."""

from __future__ import annotations

from retrieval_evaluation_framework.benchmarking.models import (
    ExperimentHistoryEntry,
    ExperimentRecord,
    Leaderboard,
    RecommendationResult,
    VisualizationArtifacts,
)
from retrieval_evaluation_framework.benchmarking.tracking import ExperimentTracker
from retrieval_evaluation_framework.config.settings import AppConfig
from retrieval_evaluation_framework.recommendation.engine import RecommendationEngine
from retrieval_evaluation_framework.recommendation.leaderboard import (
    LeaderboardEngine,
    SortableMetric,
)
from retrieval_evaluation_framework.reporting.generator import ReportGenerator
from retrieval_evaluation_framework.visualization.charts import VisualizationGenerator


class BenchmarkAnalysisService:
    """Coordinate recommendations, reports, visualizations, and history views."""

    def __init__(self, config: AppConfig, tracker: ExperimentTracker) -> None:
        """Initialize the analysis service."""
        self.config = config
        self.tracker = tracker
        self.recommendation_engine = RecommendationEngine(config.recommendation)
        self.leaderboard_engine = LeaderboardEngine()
        self.report_generator = ReportGenerator(config.reporting.reports_directory)
        self.visualization_generator = VisualizationGenerator(config.visualization.output_directory)

    def get_history(self) -> list[ExperimentHistoryEntry]:
        """Return enriched history entries."""
        experiments = self._enriched_experiments()
        return [
            ExperimentHistoryEntry(
                experiment=experiment,
                report_paths=experiment.report_paths,
                visualization_paths=experiment.visualization_paths,
            )
            for experiment in experiments
        ]

    def get_leaderboard(self, sort_by: SortableMetric = "overall_score") -> Leaderboard:
        """Return a leaderboard for stored experiments."""
        return self.leaderboard_engine.build(self._enriched_experiments(), sort_by=sort_by)

    def get_recommendation(self) -> RecommendationResult:
        """Return the current best recommendation across experiments."""
        experiments = self._enriched_experiments()
        return self.recommendation_engine.recommend(experiments)

    def generate_reports(
        self,
        experiment_ids: list[str] | None = None,
    ) -> dict[str, dict[str, str]]:
        """Generate reports for one or more experiments and persist their paths."""
        experiments = self._enriched_experiments()
        recommendation = self.recommendation_engine.recommend(experiments)
        selected = self._filter_experiments(experiments, experiment_ids)
        artifacts: dict[str, dict[str, str]] = {}
        for experiment in selected:
            if experiment.tradeoff_analysis is None:
                raise ValueError("Trade-off analysis missing for experiment history")
            report_artifacts = self.report_generator.generate(
                experiment,
                recommendation,
                experiment.tradeoff_analysis,
            )
            experiment.report_paths = report_artifacts.model_dump()
            self.tracker.save(experiment)
            artifacts[experiment.experiment_id] = experiment.report_paths
        self.tracker.export_index()
        return artifacts

    def generate_visualizations(self) -> VisualizationArtifacts:
        """Generate shared visualizations and attach them to stored experiments."""
        experiments = self._enriched_experiments()
        artifacts = self.visualization_generator.generate(experiments)
        payload = {
            **artifacts.html_paths,
            **artifacts.png_paths,
        }
        for experiment in experiments:
            experiment.visualization_paths = payload
            self.tracker.save(experiment)
        self.tracker.export_index()
        return artifacts

    def _enriched_experiments(self) -> list[ExperimentRecord]:
        experiments = self.tracker.list_experiments()
        enriched = self.recommendation_engine.enrich(experiments)
        for experiment in enriched:
            self.tracker.save(experiment)
        self.tracker.export_index()
        return enriched

    @staticmethod
    def _filter_experiments(
        experiments: list[ExperimentRecord],
        experiment_ids: list[str] | None,
    ) -> list[ExperimentRecord]:
        if not experiment_ids:
            return experiments
        wanted = set(experiment_ids)
        return [experiment for experiment in experiments if experiment.experiment_id in wanted]
