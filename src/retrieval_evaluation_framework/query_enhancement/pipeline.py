"""Composable query enhancement pipeline."""

from __future__ import annotations

from retrieval_evaluation_framework.models import Chunk, QueryEnhancementResult
from retrieval_evaluation_framework.query_enhancement.base import QueryEnhancer


class SequentialQueryEnhancer(QueryEnhancer):
    """Apply multiple query enhancers in sequence."""

    method = "sequential"

    def __init__(self, enhancers: list[QueryEnhancer]) -> None:
        if not enhancers:
            msg = "SequentialQueryEnhancer requires at least one enhancer"
            raise ValueError(msg)
        self.enhancers = enhancers

    def fit(self, chunks: list[Chunk]) -> None:
        for enhancer in self.enhancers:
            enhancer.fit(chunks)

    def enhance(self, query: str) -> QueryEnhancementResult:
        current_query = query
        steps: list[dict[str, str]] = []

        for enhancer in self.enhancers:
            result = enhancer.enhance(current_query)
            steps.append(
                {
                    "method": result.method,
                    "input_query": current_query,
                    "output_query": result.enhanced_query,
                }
            )
            current_query = result.enhanced_query

        return QueryEnhancementResult(
            original_query=query,
            enhanced_query=current_query,
            method="+".join(step["method"] for step in steps),
            metadata={"steps": steps},
        )
