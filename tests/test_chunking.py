"""Chunking tests."""

from __future__ import annotations

from retrieval_evaluation_framework.chunking.base import ChunkerFactory, count_tokens
from retrieval_evaluation_framework.config.settings import ChunkingConfig
from retrieval_evaluation_framework.models import Document, FileType


def _document(text: str) -> Document:
    return Document(id="doc", title="doc", text=text, source="memory", file_type=FileType.TXT)


def test_fixed_chunker_applies_overlap() -> None:
    chunker = ChunkerFactory.create(ChunkingConfig(strategy="fixed", chunk_size=5, overlap=2))

    chunks = chunker.chunk_document(_document("one two three four five six seven eight nine"))

    assert len(chunks) == 3
    assert chunks[0].text.endswith("five")
    assert chunks[1].text.startswith("four")


def test_recursive_chunker_respects_chunk_size() -> None:
    chunker = ChunkerFactory.create(ChunkingConfig(strategy="recursive", chunk_size=12, overlap=2))

    chunks = chunker.chunk_document(
        _document(
            "Paragraph one has several words.\n\n"
            "Paragraph two has enough text to require another chunk."
        )
    )

    assert len(chunks) >= 2
    assert all(count_tokens(chunk.text) <= 12 for chunk in chunks)


def test_semantic_chunker_splits_topic_shift() -> None:
    chunker = ChunkerFactory.create(
        ChunkingConfig(
            strategy="semantic",
            chunk_size=20,
            overlap=0,
            semantic_similarity_threshold=0.2,
        )
    )

    chunks = chunker.chunk_document(
        _document(
            "Cats purr softly and like warm sunlight. Cats enjoy playful movement. "
            "Databases use indexes for fast retrieval. Query planners optimize execution paths."
        )
    )

    assert len(chunks) >= 2
