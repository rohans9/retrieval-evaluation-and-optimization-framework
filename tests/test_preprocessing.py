"""Preprocessing tests."""

from __future__ import annotations

from retrieval_evaluation_framework.config.settings import PreprocessingConfig
from retrieval_evaluation_framework.models import Document, FileType
from retrieval_evaluation_framework.preprocessing.pipeline import TextPreprocessor


def test_preprocessor_removes_repeated_headers_footers_and_page_numbers() -> None:
    document = Document(
        id="doc-1",
        title="sample",
        text="Report Header\nIntro line\n1\n\f\nReport Header\nSecond line\n2",
        source="memory",
        file_type=FileType.TXT,
    )
    preprocessor = TextPreprocessor(PreprocessingConfig())

    processed = preprocessor.preprocess_document(document)

    assert "Report Header" not in processed.text
    assert "\n1\n" not in f"\n{processed.text}\n"
    assert "\n2\n" not in f"\n{processed.text}\n"
    assert processed.metadata["preprocessing_steps"]
