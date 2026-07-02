"""Lightweight, dependency-free text tokenization helpers.

These helpers back token counting in the chunking pipeline as well as
lexical retrieval (BM25) and corpus-vocabulary based query expansion, so all
components agree on what a "token" is without depending on a heavyweight NLP
library.
"""

from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def count_tokens(text: str) -> int:
    """Count tokens using a lightweight regex tokenizer.

    Args:
        text: Source text.

    Returns:
        Number of tokens found in the text.
    """
    return len(TOKEN_PATTERN.findall(text))


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens, excluding punctuation and digits.

    Args:
        text: Source text.

    Returns:
        List of lowercase word tokens.
    """
    return [token.lower() for token in WORD_PATTERN.findall(text)]
