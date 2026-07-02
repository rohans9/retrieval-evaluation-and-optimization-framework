"""Base interfaces for document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from retrieval_evaluation_framework.models import Document


class DocumentLoader(ABC):
    """Abstract interface for file-format-specific document loaders."""

    supported_extensions: tuple[str, ...]

    def supports(self, file_path: Path) -> bool:
        """Return whether this loader supports a file path.

        Args:
            file_path: Candidate file path.

        Returns:
            `True` when the file suffix is supported.
        """
        return file_path.suffix.lower() in self.supported_extensions

    @abstractmethod
    def load(self, file_path: Path) -> Document:
        """Parse a file into a document model.

        Args:
            file_path: File to parse.

        Returns:
            Parsed document.
        """
