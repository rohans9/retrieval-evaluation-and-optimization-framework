"""Embedding backend abstractions.

The embedding engine depends only on this interface so the rest of the
framework never needs to know which model actually produced a vector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np


class EmbeddingBackend(ABC):
    """Common interface implemented by every embedding backend."""

    name: str
    dimension: int

    @abstractmethod
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int,
        show_progress: bool,
        on_batch_complete: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """Encode a sequence of texts into embedding vectors.

        Args:
            texts: Texts to encode.
            batch_size: Number of texts encoded per batch.
            show_progress: Whether to display a progress indicator.
            on_batch_complete: Optional callback invoked with the number of
                newly encoded texts after each batch.

        Returns:
            A 2D array of shape ``(len(texts), dimension)``.
        """
