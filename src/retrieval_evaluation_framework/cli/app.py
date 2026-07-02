"""CLI entrypoints for the document processing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from retrieval_evaluation_framework.config.settings import AppConfig
from retrieval_evaluation_framework.logging import configure_logging
from retrieval_evaluation_framework.pipeline import DocumentProcessingPipeline

app = typer.Typer(help="Retrieval Evaluation & Optimization Framework CLI")
console = Console()
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
SOURCE_PATH_ARGUMENT = typer.Argument(..., exists=True, readable=True)
CONFIG_PATH_OPTION = typer.Option("--config", "-c", exists=True, readable=True)

SourcePath = Annotated[Path, SOURCE_PATH_ARGUMENT]
ConfigPath = Annotated[Path, CONFIG_PATH_OPTION]


def _load_pipeline(config_path: Path) -> DocumentProcessingPipeline:
    config = AppConfig.load_yaml(config_path)
    configure_logging(config.logging.level)
    return DocumentProcessingPipeline(config)


def _render_summary(title: str, rows: list[tuple[str, str]]) -> None:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value")
    for label, value in rows:
        table.add_row(label, value)
    console.print(table)


@app.command()
def ingest(
    source_path: SourcePath,
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
) -> None:
    """Ingest supported documents and persist the raw corpus."""
    pipeline = _load_pipeline(config_path)
    with console.status("Ingesting documents..."):
        documents = pipeline.ingest_path(source_path)
        output_path = pipeline.save_documents(
            documents,
            pipeline.config.output.output_directory / "ingested_documents.json",
        )

    _render_summary(
        "Ingestion Summary",
        [("Documents", str(len(documents))), ("Output", str(output_path))],
    )


@app.command()
def preprocess(
    source_path: SourcePath,
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
) -> None:
    """Ingest and preprocess documents."""
    pipeline = _load_pipeline(config_path)
    with console.status("Preprocessing documents..."):
        documents = pipeline.preprocess_documents(pipeline.ingest_path(source_path))
        output_path = pipeline.save_documents(
            documents,
            pipeline.config.output.output_directory / "preprocessed_documents.json",
        )

    _render_summary(
        "Preprocessing Summary",
        [("Documents", str(len(documents))), ("Output", str(output_path))],
    )


@app.command()
def chunk(
    source_path: SourcePath,
    config_path: ConfigPath = DEFAULT_CONFIG_PATH,
) -> None:
    """Run the complete phase-1 pipeline and persist a processed corpus."""
    pipeline = _load_pipeline(config_path)
    with console.status("Generating chunks..."):
        if source_path.is_file():
            corpus = pipeline.process_file(source_path)
        else:
            corpus = pipeline.process_directory(source_path)
        output_path = (
            pipeline.config.output.output_directory
            / pipeline.config.output.processed_corpus_filename
        )

    _render_summary(
        "Chunking Summary",
        [
            ("Documents", str(corpus.statistics["document_count"])),
            ("Chunks", str(corpus.statistics["chunk_count"])),
            ("Output", str(output_path)),
        ],
    )


if __name__ == "__main__":
    app()
