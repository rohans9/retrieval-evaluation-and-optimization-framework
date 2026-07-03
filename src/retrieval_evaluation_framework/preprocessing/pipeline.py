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
    """Clean whitespace while preserving structured content."""

    # Normalize spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # Restore email addresses
    text = re.sub(r"\s*@\s*", "@", text)

    # Restore domains
    text = re.sub(r"\s*\.\s*", ".", text)

    # Restore URLs
    text = re.sub(r"https?\s*:\s*/\s*/", "https://", text)

    # Remove spaces around slashes
    text = re.sub(r"\s*/\s*", "/", text)

    # Restore phone numbers
    text = re.sub(r"(\d)\s*-\s*(\d)", r"\1-\2", text)

    # Trim each line
    lines = [line.strip() for line in text.splitlines()]

    return "\n".join(lines).strip()


def remove_page_numbers(text: str) -> str:
    """Remove standalone page number lines."""
    lines = [line for line in text.splitlines() if not PAGE_NUMBER_PATTERN.match(line)]
    return "\n".join(lines)


def _remove_repeated_page_edges(text: str, remove_headers: bool, remove_footers: bool) -> str:
    pages = [page for page in text.split("\f") if page.strip()]
    if len(pages) < 2:
        return text

    # Normalize lines for robust comparison (collapse whitespace, lowercase)
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip().lower()

    page_lines = [page.splitlines() for page in pages]

    # Helper to get first/last non-empty line
    def first_non_empty(lines: list[str]) -> str:
        for l in lines:
            if l.strip():
                return l
        return ""

    def last_non_empty(lines: list[str]) -> str:
        for l in reversed(lines):
            if l.strip():
                return l
        return ""

    # Collect normalized first/last non-empty lines counts
    first_lines = Counter(norm(first_non_empty(lines)) for lines in page_lines if first_non_empty(lines))
    last_lines = Counter(norm(last_non_empty(lines)) for lines in page_lines if last_non_empty(lines))

    # require at least half the pages (or 2) to agree to consider repeated
    threshold = max(2, len(page_lines) // 2)

    repeated_header_norm = next((line for line, count in first_lines.items() if count >= threshold), None)
    repeated_footer_norm = next((line for line, count in last_lines.items() if count >= threshold), None)

    cleaned_pages: list[str] = []
    for lines in page_lines:
        candidate = lines[:]
        # Remove first non-empty line if it matches the repeated header
        if remove_headers and repeated_header_norm:
            for i, l in enumerate(candidate):
                if l.strip():
                    if norm(l) == repeated_header_norm:
                        candidate.pop(i)
                    break
        # Remove last non-empty line if it matches the repeated footer
        if remove_footers and repeated_footer_norm:
            for i in range(len(candidate) - 1, -1, -1):
                if candidate[i].strip():
                    if norm(candidate[i]) == repeated_footer_norm:
                        candidate.pop(i)
                    break
        cleaned_pages.append("\n".join(candidate).rstrip())

    # Rejoin using a single form-feed character to preserve page boundaries
    return "\f".join(cleaned_pages)


def remove_empty_lines(text: str) -> str:
    """Collapse consecutive empty lines."""
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class PageEdgeRemover:
    """Detect and remove repeated page headers/footers across a document.

    The remover is page-aware: it splits on form-feeds, inspects the first
    and last non-empty lines of each page, normalizes them and removes only
    those that repeat across a configurable fraction of pages.
    """

    def __init__(self, threshold: float = 0.7, top_k: int = 4, bottom_k: int = 4) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if top_k < 1 or bottom_k < 1:
            raise ValueError("top_k and bottom_k must be >= 1")
        self.threshold = threshold
        self.top_k = top_k
        self.bottom_k = bottom_k

    def _norm(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip().lower()

    def remove_edges(self, text: str, remove_headers: bool, remove_footers: bool) -> str:
        pages = [page for page in text.split("\f")]
        if len(pages) < 2:
            return text

        page_lines = [page.splitlines() for page in pages]

        # Efficiently build header/footer candidates as tuples of normalized lines
        def header_candidate(lines: list[str]) -> tuple[str, ...]:
            out: list[str] = []
            for l in lines:
                normalized = self._norm(l)
                if len(normalized) < 5:
                    continue
                out.append(normalized)
                if len(out) >= self.top_k:
                    break
            return tuple(out)

        def footer_candidate(lines: list[str]) -> tuple[str, ...]:
            out: list[str] = []
            for l in reversed(lines):
                normalized = self._norm(l)
                if len(normalized) < 5:
                    continue
                out.append(normalized)
                if len(out) >= self.bottom_k:
                    break
            return tuple(reversed(out))

        # Build header/footer candidate tuples for lengths 1..top_k/bottom_k
        headers_by_len: dict[int, list[tuple[str, ...]]] = {}
        footers_by_len: dict[int, list[tuple[str, ...]]] = {}
        total_pages = len(page_lines)
        required = max(2, int(total_pages * self.threshold))

        for k in range(1, self.top_k + 1):
            headers_by_len[k] = []
            for lines in page_lines:
                out: list[str] = []
                for l in lines:
                    normalized = self._norm(l)
                    if len(normalized) < 5:
                        continue
                    out.append(normalized)
                    if len(out) >= k:
                        break
                headers_by_len[k].append(tuple(out))

        for k in range(1, self.bottom_k + 1):
            footers_by_len[k] = []
            for lines in page_lines:
                out: list[str] = []
                for l in reversed(lines):
                    normalized = self._norm(l)
                    if len(normalized) < 5:
                        continue
                    out.append(normalized)
                    if len(out) >= k:
                        break
                footers_by_len[k].append(tuple(reversed(out)))

        # Find the best (longest) repeated header/footer candidate that meets threshold
        repeated_header = None
        repeated_footer = None

        for k in range(self.top_k, 0, -1):
            counts = Counter(h for h in headers_by_len[k] if h)
            rep = next((h for h, c in counts.items() if c >= required), None)
            if rep:
                repeated_header = (rep, k)
                break

        for k in range(self.bottom_k, 0, -1):
            counts = Counter(f for f in footers_by_len[k] if f)
            rep = next((f for f, c in counts.items() if c >= required), None)
            if rep:
                repeated_footer = (rep, k)
                break

        # If neither header nor footer meets threshold, return early
        if not repeated_header and not repeated_footer:
            return text

        cleaned_pages: list[str] = []
        for idx, lines in enumerate(page_lines):
            candidate = lines[:]

            # remove header tuple if matching (use the selected k)
            if remove_headers and repeated_header:
                rep, rep_k = repeated_header
                # compare the page's header tuple for rep_k
                page_header = headers_by_len[rep_k][idx]
                if page_header == rep:
                    # drop first rep_k non-empty lines
                    removed = 0
                    new_candidate: list[str] = []
                    for l in candidate:
                        if l.strip() and removed < rep_k:
                            removed += 1
                            continue
                        new_candidate.append(l)
                    candidate = new_candidate

            # remove footer tuple if matching
            if remove_footers and repeated_footer:
                rep, rep_k = repeated_footer
                page_footer = footers_by_len[rep_k][idx]
                if page_footer == rep:
                    removed = 0
                    new_candidate = []
                    for l in reversed(candidate):
                        if l.strip() and removed < rep_k:
                            removed += 1
                            continue
                        new_candidate.append(l)
                    candidate = list(reversed(new_candidate))

            cleaned_pages.append("\n".join(candidate).rstrip())

        return "\f".join(cleaned_pages)


class TextPreprocessor:
    """Apply independently configurable preprocessing steps to documents."""

    def __init__(self, config: PreprocessingConfig) -> None:
        """Initialize the preprocessor.

        Args:
            config: Preprocessing configuration.
        """
        self.config = config
        # page-edge remover configured from preprocessing settings
        self.page_edge_remover = PageEdgeRemover(
            threshold=self.config.header_footer_removal_threshold,
            top_k=4,
            bottom_k=4,
        )

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
            processed_text = self.page_edge_remover.remove_edges(
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
            processed_text = processed_text.replace("\f", "\n")
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
