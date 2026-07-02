"""Configurable text preprocessing pipeline."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from retrieval_evaluation_framework.config.settings import PreprocessingConfig
from retrieval_evaluation_framework.models import Document

PAGE_NUMBER_PATTERN = re.compile(r"^\s*(?:page\s+)?\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE)


def normalize_unicode(text: str) -> str:
    """Normalize Unicode text using NFKC."""
    return unicodedata.normalize("NFKC", text)


def cleanup_whitespace(text: str) -> str:
    """Trim noisy whitespace while preserving line structure."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def remove_page_numbers(text: str) -> str:
    """Remove standalone page number lines."""
    lines = [line for line in text.splitlines() if not PAGE_NUMBER_PATTERN.match(line)]
    return "\n".join(lines)


def _remove_repeated_page_edges(text: str, remove_headers: bool, remove_footers: bool) -> str:
    pages = [page for page in text.split("\f") if page.strip()]
    if len(pages) < 2:
        return text

    page_lines = [[line.strip() for line in page.splitlines() if line.strip()] for page in pages]
    first_lines = Counter(lines[0] for lines in page_lines if lines)
    last_lines = Counter(lines[-1] for lines in page_lines if lines)
    threshold = max(2, len(page_lines) // 2 + 1)

    repeated_header = next(
        (line for line, count in first_lines.items() if count >= threshold),
        None,
    )
    repeated_footer = next(
        (line for line, count in last_lines.items() if count >= threshold),
        None,
    )

    cleaned_pages: list[str] = []
    for lines in page_lines:
        candidate_lines = lines[:]
        if (
            remove_headers
            and repeated_header
            and candidate_lines
            and candidate_lines[0] == repeated_header
        ):
            candidate_lines = candidate_lines[1:]
        if (
            remove_footers
            and repeated_footer
            and candidate_lines
            and candidate_lines[-1] == repeated_footer
        ):
            candidate_lines = candidate_lines[:-1]
        cleaned_pages.append("\n".join(candidate_lines))
    return "\n\f\n".join(cleaned_pages)


def remove_empty_lines(text: str) -> str:
    """Collapse consecutive empty lines."""
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class TextPreprocessor:
    """Apply independently configurable preprocessing steps to documents."""

    def __init__(self, config: PreprocessingConfig) -> None:
        """Initialize the preprocessor.

        Args:
            config: Preprocessing configuration.
        """
        self.config = config

    def preprocess_text(self, text: str) -> tuple[str, list[str]]:
        """Apply the configured preprocessing steps to raw text.

        Args:
            text: Source text.

        Returns:
            A tuple containing processed text and applied step names.
        """
        applied_steps: list[str] = []
        processed_text = text

        if self.config.normalize_unicode:
            processed_text = normalize_unicode(processed_text)
            applied_steps.append("normalize_unicode")
        if self.config.remove_headers or self.config.remove_footers:
            processed_text = _remove_repeated_page_edges(
                processed_text,
                remove_headers=self.config.remove_headers,
                remove_footers=self.config.remove_footers,
            )
            applied_steps.append("remove_headers_footers")
        if self.config.remove_page_numbers:
            processed_text = remove_page_numbers(processed_text)
            applied_steps.append("remove_page_numbers")
        if self.config.cleanup_whitespace:
            processed_text = cleanup_whitespace(processed_text)
            applied_steps.append("cleanup_whitespace")
        if self.config.remove_empty_lines:
            processed_text = remove_empty_lines(processed_text)
            applied_steps.append("remove_empty_lines")

        return processed_text, applied_steps

    def preprocess_document(self, document: Document) -> Document:
        """Apply preprocessing to a document.

        Args:
            document: Document to process.

        Returns:
            Updated document.
        """
        processed_text, applied_steps = self.preprocess_text(document.text)
        metadata = {**document.metadata, "preprocessing_steps": applied_steps}
        return document.model_copy(update={"text": processed_text, "metadata": metadata})

    def preprocess_documents(self, documents: list[Document]) -> list[Document]:
        """Apply preprocessing to multiple documents.

        Args:
            documents: Documents to preprocess.

        Returns:
            Preprocessed documents.
        """
        return [self.preprocess_document(document) for document in documents]
