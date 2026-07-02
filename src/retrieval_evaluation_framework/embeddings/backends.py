"""Concrete embedding backend implementations."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

import numpy as np

from retrieval_evaluation_framework.embeddings.base import EmbeddingBackend
from retrieval_evaluation_framework.logging import get_logger
from retrieval_evaluation_framework.utils.tokenization import tokenize

LOGGER = get_logger(component="embeddings")


class SentenceTransformerBackend(EmbeddingBackend):
    """Embedding backend backed by a `sentence-transformers` model."""

    def __init__(self, model_name: str, device: str) -> None:
        """Load a sentence-transformers model.

        Args:
            model_name: Hugging Face model identifier.
            device: Torch device string (``cpu``, ``cuda``, or ``mps``).
        """
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            msg = "Sentence-transformers backend did not expose an embedding dimension"
            raise ValueError(msg)
        self.dimension = int(dimension)

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int,
        show_progress: bool,
        on_batch_complete: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """Encode texts using the underlying sentence-transformers model."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            batch_vectors = self._model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            vectors.append(np.asarray(batch_vectors, dtype=np.float32))
            if on_batch_complete is not None:
                on_batch_complete(len(batch))
        return np.vstack(vectors)


class HashingEmbeddingBackend(EmbeddingBackend):
    """Deterministic, dependency-free fallback embedding backend.

    Produces a signed hashing (feature-hashing) bag-of-words vector for each
    text. This backend requires no model download or network access, which
    makes it useful both as an offline fallback and as a fast, reproducible
    backend for automated tests.
    """

    def __init__(self, dimension: int = 384) -> None:
        """Initialize the hashing backend.

        Args:
            dimension: Size of the produced embedding vectors.
        """
        self.name = "hashing-fallback"
        self.dimension = dimension

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int,
        show_progress: bool,
        on_batch_complete: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """Encode texts using deterministic feature hashing."""
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, index] += sign
            if on_batch_complete is not None:
                on_batch_complete(1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def build_backend(
    model_name: str,
    device: str,
    backend: str,
    fallback_dimension: int,
) -> EmbeddingBackend:
    """Construct an embedding backend from configuration.

    Args:
        model_name: Sentence-transformers model identifier.
        device: Resolved compute device.
        backend: Requested backend strategy (``auto``, ``sentence_transformers``, or ``hashing``).
        fallback_dimension: Vector size used by the hashing fallback backend.

    Returns:
        A ready-to-use embedding backend.
    """
    if backend == "hashing":
        return HashingEmbeddingBackend(dimension=fallback_dimension)

    try:
        return SentenceTransformerBackend(model_name=model_name, device=device)
    except Exception as error:
        if backend == "sentence_transformers":
            raise
        LOGGER.warning(
            "sentence_transformer_backend_unavailable",
            model_name=model_name,
            error=str(error),
        )
        return HashingEmbeddingBackend(dimension=fallback_dimension)
