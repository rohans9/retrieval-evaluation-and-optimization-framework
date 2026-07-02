"""Chunking abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from retrieval_evaluation_framework.config.settings import ChunkingConfig
from retrieval_evaluation_framework.models import Chunk, Document
from retrieval_evaluation_framework.utils.tokenization import TOKEN_PATTERN, count_tokens

__all__ = ["TOKEN_PATTERN", "BaseChunker", "ChunkerFactory", "count_tokens", "tail_tokens"]


def tail_tokens(text: str, limit: int) -> str:
    """Return the last `limit` tokens from text."""
    tokens = TOKEN_PATTERN.findall(text)
    if not tokens or limit <= 0:
        return ""
    return " ".join(tokens[-limit:])


class BaseChunker(ABC):
    """Abstract chunker interface."""

    def __init__(self, config: ChunkingConfig) -> None:
        """Initialize the chunker.

        Args:
            config: Chunking configuration.
        """
        self.config = config

    @abstractmethod
    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""

    def make_chunk(self, document: Document, text: str, position: int) -> Chunk:
        """Create a chunk model from raw text."""
        return Chunk(
            chunk_id=f"{document.id}:{position}",
            document_id=document.id,
            text=text.strip(),
            metadata={**document.metadata, "source": document.source, "title": document.title},
            position=position,
            token_count=count_tokens(text),
        )


class ChunkerFactory:
    """Factory for chunker implementations."""

    @staticmethod
    def create(config: ChunkingConfig) -> BaseChunker:
        """Create a chunker matching configuration.

        Args:
            config: Chunking configuration.

        Returns:
            Configured chunker instance.
        """
        from retrieval_evaluation_framework.chunking.strategies import (
            FixedChunker,
            RecursiveChunker,
            SemanticChunker,
        )

        strategy_map: dict[str, type[BaseChunker]] = {
            "fixed": FixedChunker,
            "recursive": RecursiveChunker,
            "semantic": SemanticChunker,
        }
        return strategy_map[config.strategy](config)
