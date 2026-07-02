"""Shared data models for documents and chunks."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class FileType(StrEnum):
    """Supported input file types."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"


class Document(BaseModel):
    """Canonical document representation used across pipeline stages."""

    id: str
    title: str
    text: str
    source: str
    file_type: FileType
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Chunk artifact produced by chunking strategies."""

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    position: int
    token_count: int


class ProcessedCorpus(BaseModel):
    """Serialized corpus artifact persisted between framework phases."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    device: str
    documents: list[Document]
    chunks: list[Chunk]
    statistics: dict[str, int | float]
    config_snapshot: dict[str, Any]

    def save_json(self, destination: Path) -> Path:
        """Persist the processed corpus as JSON.

        Args:
            destination: Destination file path.

        Returns:
            The written file path.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return destination
