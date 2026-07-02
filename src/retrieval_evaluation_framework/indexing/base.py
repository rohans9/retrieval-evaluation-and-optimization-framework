"""Index persistence abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseIndex(ABC):
    """Common persistence interface shared by all index implementations."""

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Persist the index to disk.

        Args:
            directory: Destination directory.
        """

    @abstractmethod
    def load(self, directory: Path) -> None:
        """Load a previously persisted index from disk.

        Args:
            directory: Directory containing a persisted index.
        """
