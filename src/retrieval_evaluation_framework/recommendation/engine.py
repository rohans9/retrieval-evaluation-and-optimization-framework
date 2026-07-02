"""Explainable recommendation engine for retrieval pipelines."""

from __future__ import annotations

from retrieval_evaluation_framework.benchmarking.models import (
    ExecutiveSummary,
    ExperimentRecord,
    RecommendationResult,
)
from retrieval_evaluation_framework.config.settings import RecommendationConfig
from retrieval_evaluation_framework.recommendation.leaderboard import LeaderboardEngine
from retrieval_evaluation_framework.recommendation.tradeoffs import TradeoffAnalyzer


class RecommendationEngine:
    """Recommend the best pipeline using multiple quality and cost signals."""

    def __init__(self, config: RecommendationConfig) -> None:
        """Initialize the recommendation engine.

        Args:
            config: Recommendation scoring weights.
        """
        self.config = config
        self.leaderboard_engine = LeaderboardEngine()
        self.tradeoff_analyzer = TradeoffAnalyzer()

    def enrich(self, experiments: list[ExperimentRecord]) -> list[ExperimentRecord]:
        """Populate overall scores, rankings, summaries, and trade-off analysis.

        Args:
            experiments: Experiments to enrich.

        Returns:
            Enriched experiment records.
        """
        rankable = [
            experiment
            for experiment in experiments
            if experiment.retrieval_quality_metrics is not None
            and experiment.performance_metrics is not None
        ]
        if not rankable:
            return experiments

        for experiment in rankable:
            experiment.overall_score = self._score(experiment, rankable)

        leaderboard = self.leaderboard_engine.build(rankable, sort_by="overall_score")
        for row in leaderboard.rows:
            experiment = next(item for item in rankable if item.experiment_id == row.experiment_id)
            experiment.leaderboard_rank = row.rank

        recommendation = self.recommend(rankable)
        tradeoff_analysis = self.tradeoff_analyzer.analyze(rankable)
        for experiment in rankable:
            experiment.tradeoff_analysis = tradeoff_analysis
            experiment.summary = ExecutiveSummary(
                summary=self._build_summary(experiment, recommendation)
            )
            if experiment.experiment_id == recommendation.experiment_id:
                experiment.recommendation = recommendation
        return experiments

    def recommend(self, experiments: list[ExperimentRecord]) -> RecommendationResult:
        """Recommend the strongest production candidate from history.

        Args:
            experiments: Candidate experiments.

        Returns:
            Explainable recommendation result.
        """
        rankable = [
            experiment
            for experiment in experiments
            if experiment.retrieval_quality_metrics is not None
            and experiment.performance_metrics is not None
            and experiment.overall_score is not None
        ]
        if not rankable:
            raise ValueError("No rankable experiments available for recommendation")

        def score_value(experiment: ExperimentRecord) -> float:
            if experiment.overall_score is None:
                raise ValueError("Experiment is missing an overall score")
            return experiment.overall_score

        best = max(rankable, key=score_value)
        alternatives = [
            experiment.experiment_id
            for experiment in sorted(
                rankable,
                key=score_value,
                reverse=True,
            )[1:3]
        ]
        explainability = [
            f"Retriever: {best.retriever}",
            f"Embedding model: {best.embedding_model}",
            f"Chunking strategy: {best.chunking_strategy}",
            f"Query enhancement: {best.query_enhancement}",
            f"Reranker: {best.reranker}",
        ]
        return RecommendationResult(
            experiment_id=best.experiment_id,
            recommended_pipeline=self._pipeline_name(best),
            reason=(
                "This pipeline achieved the best balanced score across retrieval quality, "
                "latency, embedding cost, and index build time."
            ),
            alternative_configurations=alternatives,
            explainability=explainability,
            overall_score=score_value(best),
        )

    def _score(
        self,
        experiment: ExperimentRecord,
        experiments: list[ExperimentRecord],
    ) -> float:
        quality = experiment.retrieval_quality_metrics
        performance = experiment.performance_metrics
        if quality is None or performance is None:
            return 0.0

        quality_score = (
            quality.mean_reciprocal_rank
            + quality.ndcg_at_k
            + quality.recall_at_k
            + quality.precision_at_k
        ) / 4
        latency_values = [
            item.performance_metrics.average_retrieval_latency_ms
            for item in experiments
            if item.performance_metrics is not None
        ]
        embedding_values = [
            item.performance_metrics.embedding_generation_time_ms
            for item in experiments
            if item.performance_metrics is not None
        ]
        index_values = [
            item.performance_metrics.index_build_time_ms
            for item in experiments
            if item.performance_metrics is not None
        ]
        latency_penalty = self._normalized_lower_is_better(
            performance.average_retrieval_latency_ms,
            latency_values,
        )
        embedding_penalty = self._normalized_lower_is_better(
            performance.embedding_generation_time_ms,
            embedding_values,
        )
        index_penalty = self._normalized_lower_is_better(
            performance.index_build_time_ms,
            index_values,
        )
        return (
            self.config.quality_weight * quality_score
            + self.config.latency_weight * (1 - latency_penalty)
            + self.config.embedding_cost_weight * (1 - embedding_penalty)
            + self.config.index_build_weight * (1 - index_penalty)
        )

    @staticmethod
    def _normalized_lower_is_better(value: float, values: list[float]) -> float:
        if not values:
            return 0.0
        low = min(values)
        high = max(values)
        if high == low:
            return 0.0
        return (value - low) / (high - low)

    @staticmethod
    def _pipeline_name(experiment: ExperimentRecord) -> str:
        return (
            f"{experiment.chunking_strategy} chunking + {experiment.embedding_model} + "
            f"{experiment.retriever} + {experiment.query_enhancement} + {experiment.reranker}"
        )

    @staticmethod
    def _build_summary(
        experiment: ExperimentRecord,
        recommendation: RecommendationResult,
    ) -> str:
        quality = experiment.retrieval_quality_metrics
        performance = experiment.performance_metrics
        if quality is None or performance is None:
            return "This experiment has incomplete metrics and cannot be summarized."
        recommendation_clause = (
            "It is the recommended production candidate."
            if experiment.experiment_id == recommendation.experiment_id
            else "It remains an alternative configuration."
        )
        return (
            f"{experiment.retriever} retrieval with {experiment.embedding_model} achieved "
            f"MRR {quality.mean_reciprocal_rank:.3f}, NDCG@K {quality.ndcg_at_k:.3f}, and "
            f"average latency {performance.average_retrieval_latency_ms:.2f} ms. "
            f"{recommendation_clause}"
        )
