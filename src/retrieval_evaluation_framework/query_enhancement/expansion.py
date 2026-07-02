"""Corpus-vocabulary based query expansion.

Rather than depending on an external lexical resource such as WordNet, the
expander builds a small vocabulary from the indexed corpus and selects terms
whose embeddings are most similar to the query, keeping the framework's
dependency footprint minimal while remaining fully corpus-aware.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from retrieval_evaluation_framework.config.settings import QueryExpansionConfig
from retrieval_evaluation_framework.embeddings.engine import EmbeddingEngine
from retrieval_evaluation_framework.models import Chunk, QueryEnhancementResult
from retrieval_evaluation_framework.query_enhancement.base import QueryEnhancer
from retrieval_evaluation_framework.utils.tokenization import tokenize

_MIN_TERM_LENGTH = 3


class QueryExpander(QueryEnhancer):
    """Expand queries with semantically related corpus terms."""

    method = "expansion"

    def __init__(self, config: QueryExpansionConfig, embedding_engine: EmbeddingEngine) -> None:
        """Initialize the query expander.

        Args:
            config: Query expansion configuration.
            embedding_engine: Embedding engine used to score vocabulary terms.
        """
        self.config = config
        self.embedding_engine = embedding_engine
        self._vocabulary: list[str] = []
        self._vocabulary_vectors: np.ndarray | None = None

    def fit(self, chunks: list[Chunk]) -> None:
        """Build a term vocabulary from the indexed corpus and embed it."""
        term_frequencies: Counter[str] = Counter()
        for chunk in chunks:
            term_frequencies.update(
                term for term in tokenize(chunk.text) if len(term) >= _MIN_TERM_LENGTH
            )

        self._vocabulary = [
            term
            for term, _ in term_frequencies.most_common(self.config.vocabulary_size)
        ]
        self._vocabulary_vectors = (
            self.embedding_engine.embed(self._vocabulary) if self._vocabulary else None
        )

    def enhance(self, query: str) -> QueryEnhancementResult:
        """Expand the query with related vocabulary terms."""
        if not self._vocabulary or self._vocabulary_vectors is None:
            return QueryEnhancementResult(
                original_query=query,
                enhanced_query=query,
                method=self.method,
                metadata={"expansion_terms": []},
            )

        query_vector = self.embedding_engine.embed_query(query)
        similarities = self._vocabulary_vectors @ query_vector
        query_terms = set(tokenize(query))

        candidates = [
            (term, float(score))
            for term, score in zip(self._vocabulary, similarities, strict=True)
            if term not in query_terms and score >= self.config.similarity_threshold
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        selected_terms = [term for term, _ in candidates[: self.config.max_expansion_terms]]

        enhanced_query = query if not selected_terms else f"{query} {' '.join(selected_terms)}"
        return QueryEnhancementResult(
            original_query=query,
            enhanced_query=enhanced_query,
            method=self.method,
            metadata={"expansion_terms": selected_terms},
        )
