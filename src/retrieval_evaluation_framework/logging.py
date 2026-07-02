"""Logging helpers for the retrieval evaluation framework."""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    """Configure process-wide structured logging.

    Args:
        level: Minimum log level to emit.
    """
    logger.remove()
    logger.add(sys.stderr, level=level.upper(), serialize=True)


def get_logger(**context: str) -> Any:
    """Return a contextualized logger.

    Args:
        **context: Key-value metadata bound to all emitted records.

    Returns:
        A bound Loguru logger.
    """
    return logger.bind(**context)
