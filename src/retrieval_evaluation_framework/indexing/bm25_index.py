"""BM25 lexical index built on top of `rank-bm25`."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from retrieval_evaluation_framework.config.settings import BM25IndexConfig
from retrieval_evaluation_framework.indexing.base import BaseIndex
from retrieval_evaluation_framework.logging import get_logger
from retrieval_evaluation_framework.models import Chunk
from retrieval_evaluation_framework.utils.tokenization import tokenize

LOGGER = get_logger(component="indexing")


class BM25Index(BaseIndex):
    """BM25 index over chunk text, ranked by lexical similarity."""

    def __init__(self, config: BM25IndexConfig) -> None:
        """Initialize the index.

        Args:
            config: BM25 index configuration.
        """
        self.config = config
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Build the BM25 index from chunks.

        Args:
            chunks: Chunks to index.
        """
        tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus, k1=self.config.k1, b=self.config.b)
        self._chunks = chunks
        LOGGER.info("bm25_index_built", chunk_count=len(chunks))

    def rebuild(self, chunks: list[Chunk]) -> None:
        """Rebuild the index from a new set of chunks.

        Args:
            chunks: Chunks to index.
        """
        self.build(chunks)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        """Search the index for the most lexically similar chunks.

        Args:
            query: Query text.
            top_k: Maximum number of results to return.

        Returns:
            Chunks paired with their BM25 score, ordered by descending score.
        """
        if self._bm25 is None:
            msg = "BM25 index has not been built or loaded"
            raise RuntimeError(msg)

        scores = self._bm25.get_scores(tokenize(query))
        ranked_indices = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )[:top_k]
        return [(self._chunks[index], float(scores[index])) for index in ranked_indices]

    def save(self, directory: Path) -> None:
        """Persist the BM25 model and indexed chunks to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "bm25.pkl").open("wb") as handle:
            pickle.dump(self._bm25, handle)
        payload = json.dumps([chunk.model_dump(mode="json") for chunk in self._chunks])
        (directory / "chunks.json").write_text(payload, encoding="utf-8")
        LOGGER.info("bm25_index_saved", directory=str(directory))

    def load(self, directory: Path) -> None:
        """Load a previously persisted BM25 index from disk."""
        with (directory / "bm25.pkl").open("rb") as handle:
            self._bm25 = pickle.load(handle)
        payload = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        self._chunks = [Chunk.model_validate(item) for item in payload]
        LOGGER.info("bm25_index_loaded", directory=str(directory), chunk_count=len(self._chunks))
