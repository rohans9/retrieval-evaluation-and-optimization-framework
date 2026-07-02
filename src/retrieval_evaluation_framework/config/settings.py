"""YAML-backed application configuration models."""

from __future__ import annotations

import importlib.util
import platform
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

    import torch  # type: ignore[import-not-found]

    if torch.cuda.is_available():
        return "cuda"
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
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


class AppConfig(BaseModel):
    """Top-level application configuration."""

    device: DeviceName = "auto"
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

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
