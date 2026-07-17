"""Shared fixtures for phase-3 benchmarking tests."""

from __future__ import annotations

from pathlib import Path

from retrieval_evaluation_framework.models import Chunk, Document, FileType, ProcessedCorpus


def write_benchmark_config(
    tmp_path: Path,
    *,
    retriever: str = "hybrid",
) -> Path:
    """Create a benchmark-friendly config file using local hashing embeddings."""
    config_path = tmp_path / "benchmark_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "device: cpu",
                "logging:",
                "  level: INFO",
                "embedding:",
                "  model_name: local-hash",
                "  backend: hashing",
                "  device: cpu",
                "  batch_size: 4",
                "  show_progress: false",
                "index:",
                f"  index_directory: {(tmp_path / 'index').as_posix()}",
                "retrieval:",
                f"  retriever: {retriever}",
                "  top_k: 2",
                "  fusion:",
                "    strategy: reciprocal_rank_fusion",
                "    rrf_k: 60",
                "reranking:",
                "  enabled: false",
                "benchmark:",
                f"  dataset_path: {(tmp_path / 'dataset.json').as_posix()}",
                f"  experiment_directory: {(tmp_path / 'experiments').as_posix()}",
                f"  results_directory: {(tmp_path / 'benchmarks').as_posix()}",
                "  test_split_ratio: 0.2",
                "reporting:",
                f"  reports_directory: {(tmp_path / 'generated_reports').as_posix()}",
                "  include_csv: true",
                "  include_json: true",
                "  include_markdown: true",
                "visualization:",
                f"  output_directory: {(tmp_path / 'visualizations').as_posix()}",
                "  export_html: true",
                "  export_png: true",
                "recommendation:",
                "  quality_weight: 0.65",
                "  latency_weight: 0.2",
                "  embedding_cost_weight: 0.075",
                "  index_build_weight: 0.075",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def write_corpus(tmp_path: Path) -> Path:
    """Create a processed corpus for retrieval evaluation tests."""
    corpus_path = tmp_path / "processed_corpus.json"
    document = Document(
        id="doc-handbook",
        title="employee_handbook",
        text=(
            "Maternity leave policy offers sixteen weeks of paid leave. "
            "Remote work policy allows hybrid schedules. "
            "Travel reimbursement covers flights and hotels."
        ),
        source="memory",
        file_type=FileType.TXT,
    )
    chunks = [
        Chunk(
            chunk_id="doc-handbook:0",
            document_id=document.id,
            text="Maternity leave policy offers sixteen weeks of paid leave.",
            position=0,
            token_count=9,
            metadata={"title": document.title, "source": document.source},
        ),
        Chunk(
            chunk_id="doc-handbook:1",
            document_id=document.id,
            text="Remote work policy allows hybrid schedules.",
            position=1,
            token_count=6,
            metadata={"title": document.title, "source": document.source},
        ),
        Chunk(
            chunk_id="doc-handbook:2",
            document_id=document.id,
            text="Travel reimbursement covers flights and hotels.",
            position=2,
            token_count=6,
            metadata={"title": document.title, "source": document.source},
        ),
    ]
    ProcessedCorpus(
        device="cpu",
        documents=[document],
        chunks=chunks,
        statistics={
            "document_count": 1,
            "chunk_count": 3,
            "average_chunk_tokens": 7.0,
        },
        config_snapshot={},
    ).save_json(corpus_path)
    return corpus_path


def write_dataset(tmp_path: Path) -> Path:
    """Create a small labeled retrieval evaluation dataset."""
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        """
name: benchmark_fixture
metadata:
  domain: hr
examples:
  - query: What is the maternity leave policy?
    positive_documents:
      - doc-handbook:0
    negative_documents:
      - doc-handbook:2
  - query: Which policy mentions hybrid schedules?
    positive_documents:
      - doc-handbook:1
    negative_documents:
      - doc-handbook:2
""".strip(),
        encoding="utf-8",
    )
    return dataset_path


def write_parameter_file(tmp_path: Path) -> Path:
    """Create a sweep/grid parameter file."""
    parameters_path = tmp_path / "parameters.yaml"
    parameters_path.write_text(
        """
retrieval.top_k:
  - 1
  - 2
retrieval.retriever:
  - bm25
  - hybrid
""".strip(),
        encoding="utf-8",
    )
    return parameters_path
