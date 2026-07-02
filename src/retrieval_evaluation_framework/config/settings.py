"""YAML-backed application configuration models."""

from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

DeviceName = Literal["auto", "cuda", "mps", "cpu"]
ResolvedDeviceName = Literal["cuda", "mps", "cpu"]


def resolve_device(device: DeviceName) -> ResolvedDeviceName:
    """Resolve the best available device for the current environment.

    Args:
        device: Requested device configuration.

    Returns:
        The resolved device name.
    """
    if device != "auto":
        return device

    if importlib.util.find_spec("torch") is None:
        return "cpu"

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({"
                "'cuda': bool(torch.cuda.is_available()), "
                "'mps': bool("
                "getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()"
                ")"
                "}))"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if probe.returncode != 0:
        return "cpu"

    try:
        payload = yaml.safe_load(probe.stdout) or {}
    except Exception:
        payload = {}

    if payload.get("cuda") is True:
        return "cuda"
    if platform.system() == "Darwin" and payload.get("mps") is True:
        return "mps"
    return "cpu"


class LoggingConfig(BaseModel):
    """Logging-related configuration."""

    level: str = "INFO"


class IngestionConfig(BaseModel):
    """Document ingestion configuration."""

    input_directory: Path = Path("data/input")
    recursive: bool = True
    supported_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".txt", ".md"]
    )


class PreprocessingConfig(BaseModel):
    """Text preprocessing configuration."""

    normalize_unicode: bool = True
    cleanup_whitespace: bool = True
    remove_headers: bool = True
    remove_footers: bool = True
    remove_page_numbers: bool = True
    remove_empty_lines: bool = True


class ChunkingConfig(BaseModel):
    """Chunking configuration."""

    strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = 250
    overlap: int = 40
    semantic_similarity_threshold: float = 0.58
    semantic_min_sentences: int = 2
    semantic_encoder_model: str | None = None

    @model_validator(mode="after")
    def validate_sizes(self) -> ChunkingConfig:
        """Validate chunk and overlap settings.

        Returns:
            The validated model instance.
        """
        if self.chunk_size <= 0:
            msg = "chunk_size must be greater than zero"
            raise ValueError(msg)
        if self.overlap < 0:
            msg = "overlap must be greater than or equal to zero"
            raise ValueError(msg)
        if self.overlap >= self.chunk_size:
            msg = "overlap must be smaller than chunk_size"
            raise ValueError(msg)
        return self


class OutputConfig(BaseModel):
    """Output artifact configuration."""

    output_directory: Path = Path("data/processed")
    processed_corpus_filename: str = "processed_corpus.json"


EmbeddingBackendName = Literal["auto", "sentence_transformers", "hashing"]


class EmbeddingConfig(BaseModel):
    """Embedding generation configuration."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    backend: EmbeddingBackendName = "auto"
    device: DeviceName = "auto"
    batch_size: int = 32
    normalize_embeddings: bool = True
    fallback_dimension: int = 384
    cache_directory: Path = Path("data/cache/embeddings")
    show_progress: bool = True

    @model_validator(mode="after")
    def validate_batch_size(self) -> EmbeddingConfig:
        """Validate the configured batch size.

        Returns:
            The validated model instance.
        """
        if self.batch_size <= 0:
            msg = "batch_size must be greater than zero"
            raise ValueError(msg)
        if self.fallback_dimension <= 0:
            msg = "fallback_dimension must be greater than zero"
            raise ValueError(msg)
        return self


class BM25IndexConfig(BaseModel):
    """BM25 index configuration."""

    k1: float = 1.5
    b: float = 0.75


class DenseIndexConfig(BaseModel):
    """Dense (FAISS) index configuration."""

    metric: Literal["cosine"] = "cosine"


class IndexConfig(BaseModel):
    """Index construction and persistence configuration."""

    index_directory: Path = Path("data/index")
    bm25: BM25IndexConfig = Field(default_factory=BM25IndexConfig)
    dense: DenseIndexConfig = Field(default_factory=DenseIndexConfig)


class FusionConfig(BaseModel):
    """Hybrid retrieval fusion configuration."""

    strategy: Literal["reciprocal_rank_fusion"] = "reciprocal_rank_fusion"
    rrf_k: int = 60


class RetrievalConfig(BaseModel):
    """Retriever selection and behavior configuration."""

    retriever: Literal["bm25", "dense", "hybrid"] = "hybrid"
    top_k: int = 10
    fusion: FusionConfig = Field(default_factory=FusionConfig)

    @model_validator(mode="after")
    def validate_top_k(self) -> RetrievalConfig:
        """Validate the configured Top-K value.

        Returns:
            The validated model instance.
        """
        if self.top_k <= 0:
            msg = "top_k must be greater than zero"
            raise ValueError(msg)
        return self


QueryEnhancementMethod = Literal["none", "expansion", "hyde"]
HydeBackendName = Literal["auto", "transformers", "template"]


class QueryExpansionConfig(BaseModel):
    """Query expansion configuration."""

    max_expansion_terms: int = 3
    similarity_threshold: float = 0.35
    vocabulary_size: int = 512


class HydeConfig(BaseModel):
    """HyDE (Hypothetical Document Embeddings) configuration."""

    backend: HydeBackendName = "auto"
    generator_model: str = "gpt2"
    max_new_tokens: int = 48


class QueryEnhancementConfig(BaseModel):
    """Optional query enhancement configuration."""

    enabled: bool = False
    method: QueryEnhancementMethod = "none"
    expansion: QueryExpansionConfig = Field(default_factory=QueryExpansionConfig)
    hyde: HydeConfig = Field(default_factory=HydeConfig)


RerankerBackendName = Literal["auto", "cross_encoder", "lexical"]


class RerankingConfig(BaseModel):
    """Optional reranking configuration."""

    enabled: bool = False
    backend: RerankerBackendName = "auto"
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5


class BenchmarkConfig(BaseModel):
    """Benchmarking and experiment tracking configuration."""

    dataset_path: Path | None = None
    experiment_directory: Path = Path("reports/experiments")
    results_directory: Path = Path("reports/benchmarks")
    test_split_ratio: float = 0.2
    notes: str | None = None

    @model_validator(mode="after")
    def validate_ratio(self) -> BenchmarkConfig:
        """Validate optional train/test split configuration.

        Returns:
            The validated benchmark configuration.
        """
        if not 0 < self.test_split_ratio < 1:
            msg = "test_split_ratio must be between 0 and 1"
            raise ValueError(msg)
        return self


class ReportingConfig(BaseModel):
    """Reporting output configuration."""

    reports_directory: Path = Path("reports/generated")
    include_csv: bool = True
    include_json: bool = True
    include_markdown: bool = True


class VisualizationConfig(BaseModel):
    """Visualization output configuration."""

    output_directory: Path = Path("reports/visualizations")
    export_html: bool = True
    export_png: bool = True


class RecommendationConfig(BaseModel):
    """Recommendation scoring configuration."""

    quality_weight: float = 0.65
    latency_weight: float = 0.2
    embedding_cost_weight: float = 0.075
    index_build_weight: float = 0.075

    @model_validator(mode="after")
    def validate_weights(self) -> RecommendationConfig:
        """Validate recommendation engine weights.

        Returns:
            The validated recommendation configuration.
        """
        total = (
            self.quality_weight
            + self.latency_weight
            + self.embedding_cost_weight
            + self.index_build_weight
        )
        if not 0.99 <= total <= 1.01:
            msg = "recommendation weights must sum to approximately 1.0"
            raise ValueError(msg)
        return self


class AppConfig(BaseModel):
    """Top-level application configuration."""

    device: DeviceName = "auto"
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    query_enhancement: QueryEnhancementConfig = Field(default_factory=QueryEnhancementConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    recommendation: RecommendationConfig = Field(default_factory=RecommendationConfig)

    @classmethod
    def load_yaml(cls, path: Path) -> AppConfig:
        """Load application configuration from YAML.

        Args:
            path: Configuration file path.

        Returns:
            Parsed application configuration.
        """
        raw_data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw_data)

    @property
    def resolved_device(self) -> ResolvedDeviceName:
        """Return the resolved execution device."""
        return resolve_device(self.device)

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration snapshot."""
        return self.model_dump(mode="json")
