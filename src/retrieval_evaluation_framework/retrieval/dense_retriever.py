"""Dense vector retriever backed by FAISS and an embedding engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from retrieval_evaluation_framework.config.settings import DenseIndexConfig, RetrievalConfig
from retrieval_evaluation_framework.embeddings.engine import EmbeddingEngine
from retrieval_evaluation_framework.indexing.dense_index import DenseIndex
from retrieval_evaluation_framework.models import Chunk, RetrievalResult
from retrieval_evaluation_framework.retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Retriever backed by a dense embedding index."""

    name = "dense"

    def __init__(
        self,
        config: RetrievalConfig,
        index_config: DenseIndexConfig,
        embedding_engine: EmbeddingEngine,
    ) -> None:
        """Initialize the dense retriever.

        Args:
            config: Retrieval configuration (used for the default Top-K).
            index_config: Dense-index-specific configuration.
            embedding_engine: Embedding engine used to encode chunks and queries.
        """
        self.config = config
        self.index_config = index_config
        self.embedding_engine = embedding_engine
        self._index = DenseIndex(index_config, dimension=embedding_engine.dimension)
        self._chunk_by_id: dict[str, Chunk] = {}

    def build_index(self, chunks: list[Chunk]) -> None:
        """Embed chunks and build the dense index."""
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        embedding_store = self.embedding_engine.embed_chunks(chunks)
        self._index.build(embedding_store.chunk_ids, embedding_store.vectors)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve the most semantically similar chunks for a query."""
        resolved_top_k = top_k or self.config.top_k
        query_vector = self.embedding_engine.embed_query(query)
        matches = self._index.search(query_vector, resolved_top_k)
        return [
            RetrievalResult(
                chunk=self._chunk_by_id[chunk_id],
                score=score,
                rank=rank,
                retriever=self.name,
            )
            for rank, (chunk_id, score) in enumerate(matches, start=1)
        ]

    def get_configuration(self) -> dict[str, Any]:
        """Return the retriever's effective configuration."""
        return {
            "retriever": self.name,
            "top_k": self.config.top_k,
            "model_name": self.embedding_engine.config.model_name,
            "device": self.embedding_engine.device,
            "metric": self.index_config.metric,
        }

    def save_index(self, directory: Path) -> None:
        """Persist the dense index and chunk metadata to disk."""
        directory = directory / "dense"
        directory.mkdir(parents=True, exist_ok=True)

        self._index.save(directory)
        chunk_payload = json.dumps(
            [chunk.model_dump(mode="json") for chunk in self._chunk_by_id.values()]
        )
        (directory / "chunk_metadata.json").write_text(chunk_payload, encoding="utf-8")

    def load_index(self, directory: Path) -> None:
        """Load a previously persisted dense index and chunk metadata."""
        directory = directory / "dense"

        self._index.load(directory)

        payload = json.loads(
            (directory / "chunk_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        chunks = [Chunk.model_validate(item) for item in payload]
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
