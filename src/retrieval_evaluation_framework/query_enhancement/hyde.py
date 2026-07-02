"""HyDE (Hypothetical Document Embeddings) query enhancement.

HyDE generates a short hypothetical answer to the query and appends it to the
search text so that retrieval matches against the language of a plausible
answer rather than only the (often terse) question. When a local
text-generation model is unavailable, a deterministic template-based
fallback keeps the technique usable offline.
"""

from __future__ import annotations

from typing import Any, cast

from retrieval_evaluation_framework.config.settings import HydeConfig
from retrieval_evaluation_framework.logging import get_logger
from retrieval_evaluation_framework.models import QueryEnhancementResult
from retrieval_evaluation_framework.query_enhancement.base import QueryEnhancer

LOGGER = get_logger(component="query_enhancement")


class HydeQueryEnhancer(QueryEnhancer):
    """Generate a hypothetical answer document and append it to the query."""

    method = "hyde"

    def __init__(self, config: HydeConfig) -> None:
        """Initialize the HyDE enhancer.

        Args:
            config: HyDE configuration.
        """
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
            LOGGER.warning("hyde_generator_unavailable", error=str(error))
            return None

    def enhance(self, query: str) -> QueryEnhancementResult:
        """Generate a hypothetical answer and append it to the query."""
        hypothetical_document, backend_used = self._generate(query)
        enhanced_query = f"{query} {hypothetical_document}".strip()
        return QueryEnhancementResult(
            original_query=query,
            enhanced_query=enhanced_query,
            method=self.method,
            metadata={"hypothetical_document": hypothetical_document, "backend": backend_used},
        )

    def _generate(self, query: str) -> tuple[str, str]:
        if self._generator is not None:
            try:
                return self._generate_with_model(query), "transformers"
            except Exception as error:
                LOGGER.warning("hyde_generation_failed", error=str(error))
        return self._template_answer(query), "template"

    def _generate_with_model(self, query: str) -> str:
        if self._generator is None:
            msg = "HyDE generator is not available"
            raise RuntimeError(msg)
        prompt = f"Answer the question in one short passage.\nQuestion: {query}\nAnswer:"
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
        return generated_text.split("Answer:")[-1].strip()

    @staticmethod
    def _template_answer(query: str) -> str:
        topic = query.strip().rstrip("?").lower()
        return (
            f"This passage explains {topic} by describing the relevant background, "
            "key considerations, and a direct answer to the question."
        )
