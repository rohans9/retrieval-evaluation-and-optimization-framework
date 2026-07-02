"""Pipeline and CLI tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from retrieval_evaluation_framework.cli.app import app
from retrieval_evaluation_framework.config.settings import AppConfig
from retrieval_evaluation_framework.pipeline import DocumentProcessingPipeline

RUNNER = CliRunner()


def test_pipeline_process_directory_persists_corpus(
    fixture_directory: Path,
    tmp_path: Path,
) -> None:
    config = AppConfig.load_yaml(Path("configs/default.yaml"))
    config.output.output_directory = tmp_path
    pipeline = DocumentProcessingPipeline(config)

    corpus = pipeline.process_directory(fixture_directory)

    assert corpus.statistics["document_count"] == 4
    assert (tmp_path / config.output.processed_corpus_filename).exists()


def test_cli_chunk_command_runs(fixture_directory: Path, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""device: cpu
logging:
  level: INFO
ingestion:
  input_directory: data/input
  recursive: true
  supported_extensions:
    - .pdf
    - .docx
    - .txt
    - .md
preprocessing:
  normalize_unicode: true
  cleanup_whitespace: true
  remove_headers: true
  remove_footers: true
  remove_page_numbers: true
  remove_empty_lines: true
chunking:
  strategy: recursive
  chunk_size: 50
  overlap: 10
  semantic_similarity_threshold: 0.58
  semantic_min_sentences: 2
  semantic_encoder_model:
output:
  output_directory: {tmp_path.as_posix()}
  processed_corpus_filename: processed_corpus.json
""",
        encoding="utf-8",
    )

    result = RUNNER.invoke(app, ["chunk", str(fixture_directory), "--config", str(config_path)])

    assert result.exit_code == 0
    assert (tmp_path / "processed_corpus.json").exists()
