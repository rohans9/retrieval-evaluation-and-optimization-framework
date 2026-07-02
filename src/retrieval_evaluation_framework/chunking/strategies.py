"""Concrete chunking strategies."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, cast

from retrieval_evaluation_framework.chunking.base import TOKEN_PATTERN, BaseChunker, count_tokens
from retrieval_evaluation_framework.config.settings import ChunkingConfig
from retrieval_evaluation_framework.logging import get_logger
from retrieval_evaluation_framework.models import Chunk, Document

LOGGER = get_logger(component="chunking")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(text) if sentence.strip()]


def _token_chunks(tokens: list[str], chunk_size: int, overlap: int) -> list[str]:
    step = chunk_size - overlap
    return [" ".join(tokens[index : index + chunk_size]) for index in range(0, len(tokens), step)]


def _split_recursively(text: str, chunk_size: int, separators: Sequence[str]) -> list[str]:
    if count_tokens(text) <= chunk_size:
        return [text.strip()]

    for separator in separators:
        if separator and separator in text:
            raw_parts = [part.strip() for part in text.split(separator) if part.strip()]
            if len(raw_parts) <= 1:
                continue

            parts: list[str] = []
            for part in raw_parts:
                normalized_part = part
                if separator == ". " and not normalized_part.endswith("."):
                    normalized_part = f"{normalized_part}."
                parts.extend(_split_recursively(normalized_part, chunk_size, separators[1:]))
            return parts

    tokens = TOKEN_PATTERN.findall(text)
    return _token_chunks(tokens, chunk_size, 0)


def _merge_segments(segments: Sequence[str], chunk_size: int) -> list[str]:
    merged: list[str] = []
    current = ""
    for segment in segments:
        candidate = segment if not current else f"{current}\n\n{segment}".strip()
        if count_tokens(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            merged.append(current.strip())
        current = segment
    if current:
        merged.append(current.strip())
    return merged


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(left[token] * right[token] for token in left.keys() & right.keys())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _lexical_sentence_vectors(sentences: Sequence[str]) -> list[Counter[str]]:
    return [
        Counter(token.lower() for token in TOKEN_PATTERN.findall(sentence) if token.isalnum())
        for sentence in sentences
    ]


class FixedChunker(BaseChunker):
    """Chunk documents using fixed token windows."""

    def chunk_document(self, document: Document) -> list[Chunk]:
        tokens = TOKEN_PATTERN.findall(document.text)
        if not tokens:
            return []
        chunk_texts = _token_chunks(tokens, self.config.chunk_size, self.config.overlap)
        return [
            self.make_chunk(document, chunk_text, position)
            for position, chunk_text in enumerate(chunk_texts)
        ]


class RecursiveChunker(BaseChunker):
    """Chunk documents by recursively splitting structural boundaries."""

    def chunk_document(self, document: Document) -> list[Chunk]:
        segments = _split_recursively(
            document.text,
            self.config.chunk_size,
            ["\n\n", "\n", ". ", " "],
        )
        chunk_texts = _merge_segments(segments, self.config.chunk_size)
        chunks = [
            self.make_chunk(document, chunk_text, position)
            for position, chunk_text in enumerate(chunk_texts)
        ]
        LOGGER.info("recursive_chunks_generated", document_id=document.id, chunk_count=len(chunks))
        return chunks


class SemanticChunker(BaseChunker):
    """Chunk documents using sentence-level semantic breakpoints."""

    def __init__(self, config: ChunkingConfig) -> None:
        super().__init__(config)
        self._model: Any | None = None
        if config.semantic_encoder_model:
            try:
                from sentence_transformers import (
                    SentenceTransformer,
                )

                self._model = SentenceTransformer(config.semantic_encoder_model)
            except Exception as error:
                LOGGER.warning("semantic_encoder_unavailable", error=str(error))

    def chunk_document(self, document: Document) -> list[Chunk]:
        sentences = _sentences(document.text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return RecursiveChunker(self.config).chunk_document(document)

        sentence_vectors = self._encode_sentences(sentences)
        chunk_texts: list[str] = []
        current_sentences = [sentences[0]]

        for index in range(1, len(sentences)):
            next_sentence = sentences[index]
            similarity = self._adjacent_similarity(
                sentence_vectors[index - 1],
                sentence_vectors[index],
            )
            candidate_text = " ".join(current_sentences + [next_sentence])
            should_split = count_tokens(candidate_text) > self.config.chunk_size or (
                len(current_sentences) >= self.config.semantic_min_sentences
                and similarity < self.config.semantic_similarity_threshold
            )
            if should_split:
                chunk_texts.append(" ".join(current_sentences))
                current_sentences = [next_sentence]
            else:
                current_sentences.append(next_sentence)

        if current_sentences:
            chunk_texts.append(" ".join(current_sentences))

        normalized_chunks: list[str] = []
        for chunk_text in chunk_texts:
            if count_tokens(chunk_text) <= self.config.chunk_size:
                normalized_chunks.append(chunk_text)
                continue
            normalized_chunks.extend(
                _merge_segments(
                    _split_recursively(
                        chunk_text,
                        self.config.chunk_size,
                        [". ", " "],
                    ),
                    self.config.chunk_size,
                )
            )

        chunks = [
            self.make_chunk(document, chunk_text, position)
            for position, chunk_text in enumerate(normalized_chunks)
        ]
        LOGGER.info("semantic_chunks_generated", document_id=document.id, chunk_count=len(chunks))
        return chunks

    def _encode_sentences(self, sentences: Sequence[str]) -> Sequence[object]:
        if self._model is not None:
            encoded = self._model.encode(list(sentences), normalize_embeddings=True)
            return cast(Sequence[object], encoded)
        return _lexical_sentence_vectors(sentences)

    def _adjacent_similarity(self, left: object, right: object) -> float:
        if self._model is not None:
            left_vector = cast(Sequence[float], left)
            right_vector = cast(Sequence[float], right)
            return float(
                sum(
                    left_value * right_value
                    for left_value, right_value in zip(
                        left_vector,
                        right_vector,
                        strict=False,
                    )
                )
            )
        return _cosine_similarity(left, right)  # type: ignore[arg-type]
