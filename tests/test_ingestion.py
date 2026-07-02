"""Ingestion tests."""

from __future__ import annotations

from pathlib import Path

from retrieval_evaluation_framework.ingestion.loaders import DocumentIngestor
from retrieval_evaluation_framework.models import FileType


def test_directory_ingestion_supports_all_phase_one_formats(fixture_directory: Path) -> None:
    ingestor = DocumentIngestor()

    documents = ingestor.ingest_directory(fixture_directory)

    assert len(documents) == 4
    assert {document.file_type for document in documents} == {
        FileType.PDF,
        FileType.DOCX,
        FileType.TXT,
        FileType.MARKDOWN,
    }


def test_recursive_discovery_finds_nested_markdown(fixture_directory: Path) -> None:
    ingestor = DocumentIngestor()

    discovered = ingestor.discover_files(fixture_directory, recursive=True)

    assert any(path.name == "sample.md" for path in discovered)
