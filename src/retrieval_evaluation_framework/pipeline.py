"""End-to-end document processing pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from retrieval_evaluation_framework.chunking.base import ChunkerFactory
from retrieval_evaluation_framework.config.settings import AppConfig
from retrieval_evaluation_framework.ingestion.loaders import DocumentIngestor
from retrieval_evaluation_framework.logging import get_logger
from retrieval_evaluation_framework.models import Chunk, Document, ProcessedCorpus
from retrieval_evaluation_framework.preprocessing.pipeline import TextPreprocessor

LOGGER = get_logger(component="pipeline")


class DocumentProcessingPipeline:
    """Coordinate ingestion, preprocessing, chunking, and persistence."""

    def __init__(self, config: AppConfig) -> None:
        """Initialize the pipeline.

        Args:
            config: Application configuration.
        """
        self.config = config
        self.ingestor = DocumentIngestor()
        self.preprocessor = TextPreprocessor(config.preprocessing)
        self.chunker = ChunkerFactory.create(config.chunking)

    def ingest_path(self, source_path: Path) -> list[Document]:
        """Ingest documents from a file or directory.

        Args:
            source_path: Source file or directory.

        Returns:
            Parsed documents.
        """
        if source_path.is_file():
            documents = [self.ingestor.ingest_file(source_path)]
        else:
            documents = self.ingestor.ingest_directory(
                source_path,
                recursive=self.config.ingestion.recursive,
            )
        self._annotate_documents_with_domain(documents, source_path)
        return documents

    def preprocess_documents(self, documents: list[Document]) -> list[Document]:
        """Preprocess ingested documents."""
        return self.preprocessor.preprocess_documents(documents)

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk preprocessed documents."""
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunker.chunk_document(document))
        return chunks

    def process_file(self, source_path: Path, persist: bool = True) -> ProcessedCorpus:
        """Process a single file from ingestion through chunking."""
        return self._process(source_path, persist=persist)

    def process_directory(self, source_path: Path, persist: bool = True) -> ProcessedCorpus:
        """Process a directory from ingestion through chunking."""
        return self._process(source_path, persist=persist)

    def save_documents(self, documents: list[Document], destination: Path) -> Path:
        """Persist a document list as JSON."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = [document.model_dump(mode="json") for document in documents]
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

    def iter_sources(self, source_path: Path) -> list[tuple[str | None, Path]]:
        """Return one or more source paths to process.

        When the configured ingestion root is passed (for example ``data/raw``)
        and it contains domain subdirectories, each domain is processed
        independently.
        """
        if not source_path.is_dir() or not self._is_ingestion_root(source_path):
            return [(self.resolve_domain(source_path), source_path)]

        recursive = self.config.ingestion.recursive
        domain_sources: list[tuple[str, Path]] = []
        for candidate in sorted(source_path.iterdir()):
            if not candidate.is_dir():
                continue
            if self.ingestor.discover_files(candidate, recursive=recursive):
                domain_sources.append((candidate.name, candidate))

        if domain_sources:
            return domain_sources
        return [(None, source_path)]

    def resolve_domain(self, source_path: Path) -> str | None:
        """Infer the domain from a source path relative to ingestion root."""
        ingestion_root = self.config.ingestion.input_directory
        try:
            relative = source_path.resolve().relative_to(ingestion_root.resolve())
        except ValueError:
            return None

        if not relative.parts:
            return None
        return relative.parts[0]

    def output_path_for(self, source_path: Path, filename: str) -> Path:
        """Build a destination path for pipeline artifacts."""
        domain = self.resolve_domain(source_path)
        base = self.config.output.output_directory
        return (base / domain / filename) if domain else (base / filename)

    def save_documents_for_source(
        self,
        documents: list[Document],
        source_path: Path,
        filename: str,
    ) -> Path:
        """Persist documents to the domain-aware output location."""
        return self.save_documents(documents, self.output_path_for(source_path, filename))

    def _process(self, source_path: Path, persist: bool) -> ProcessedCorpus:
        documents = self.ingest_path(source_path)
        processed_documents = self.preprocess_documents(documents)
        chunks = self.chunk_documents(processed_documents)
        corpus = ProcessedCorpus(
            device=self.config.resolved_device,
            documents=processed_documents,
            chunks=chunks,
            statistics=self._build_statistics(processed_documents, chunks),
            config_snapshot=self.config.to_metadata(),
        )
        if persist:
            output_path = self.output_path_for(
                source_path,
                self.config.output.processed_corpus_filename,
            )
            corpus.save_json(output_path)
            LOGGER.info("processed_corpus_saved", path=str(output_path), chunk_count=len(chunks))
        return corpus

    def _is_ingestion_root(self, source_path: Path) -> bool:
        try:
            return source_path.resolve() == self.config.ingestion.input_directory.resolve()
        except FileNotFoundError:
            return source_path == self.config.ingestion.input_directory

    def _annotate_documents_with_domain(
        self,
        documents: list[Document],
        source_path: Path,
    ) -> None:
        domain = self.resolve_domain(source_path)
        if domain is None:
            return
        for document in documents:
            document.metadata.setdefault("domain", domain)

    def _build_statistics(
        self,
        documents: list[Document],
        chunks: list[Chunk],
    ) -> dict[str, int | float]:
        average_chunk_tokens = mean(chunk.token_count for chunk in chunks) if chunks else 0.0
        return {
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "average_chunk_tokens": round(average_chunk_tokens, 2),
        }
