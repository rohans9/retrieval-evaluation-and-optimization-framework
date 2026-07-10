"""Query rewriting enhancement.

Rewriting transforms user questions into retrieval-optimized search phrases
while preserving intent. A template fallback keeps rewriting available even
without local generator models.
"""

from __future__ import annotations

from typing import Any, cast

from retrieval_evaluation_framework.config.settings import QueryRewriteConfig
from retrieval_evaluation_framework.logging import get_logger
from retrieval_evaluation_framework.models import QueryEnhancementResult
from retrieval_evaluation_framework.query_enhancement.base import QueryEnhancer

LOGGER = get_logger(component="query_enhancement")


class RewriteQueryEnhancer(QueryEnhancer):
    """Rewrite user queries into retrieval-oriented phrasing."""

    method = "rewrite"

    def __init__(self, config: QueryRewriteConfig) -> None:
        self.config = config
        self._generator: Any | None = None
        if config.backend != "template":
            self._generator = self._load_generator()

    def _load_generator(self) -> Any | None:
        try:
            from transformers import pipeline

            return pipeline("text-generation", model=self.config.generator_model)
        except Exception as error:
            if self.config.backend == "transformers":
                raise
            LOGGER.warning("rewrite_generator_unavailable", error=str(error))
            return None

    def enhance(self, query: str) -> QueryEnhancementResult:
        rewritten_query, backend_used = self._rewrite(query)
        return QueryEnhancementResult(
            original_query=query,
            enhanced_query=rewritten_query,
            method=self.method,
            metadata={"backend": backend_used},
        )

    def _rewrite(self, query: str) -> tuple[str, str]:
        if self._generator is not None:
            try:
                return self._rewrite_with_model(query), "transformers"
            except Exception as error:
                LOGGER.warning("rewrite_generation_failed", error=str(error))
        return self._template_rewrite(query), "template"

    def _rewrite_with_model(self, query: str) -> str:
        if self._generator is None:
            msg = "Rewrite generator is not available"
            raise RuntimeError(msg)

        prompt = (
            "Rewrite the following query for semantic retrieval. "
            "Preserve intent and key entities. Keep it concise.\n"
            f"Query: {query}\n"
            "Rewritten Query:"
        )
        tokenizer = getattr(self._generator, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
        outputs = cast(
            list[dict[str, str]],
            self._generator(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                num_return_sequences=1,
                pad_token_id=pad_token_id,
            ),
        )
        generated_text = outputs[0]["generated_text"]
        rewritten = generated_text.split("Rewritten Query:")[-1].strip()
        return rewritten or query

    @staticmethod
    def _template_rewrite(query: str) -> str:
        normalized = " ".join(query.split()).rstrip("?")
        if not normalized:
            return query
        return f"Find relevant passages that answer: {normalized}"
