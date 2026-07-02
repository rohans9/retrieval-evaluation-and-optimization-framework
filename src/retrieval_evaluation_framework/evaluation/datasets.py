"""Evaluation dataset loading and validation."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class EvaluationDatasetExample(BaseModel):
    """Single evaluation example for retrieval benchmarking."""

    query: str
    positive_documents: list[str]
    negative_documents: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_example(self) -> EvaluationDatasetExample:
        """Validate required example fields.

        Returns:
            The validated example.
        """
        if not self.query.strip():
            msg = "query must not be empty"
            raise ValueError(msg)
        if not self.positive_documents:
            msg = "positive_documents must contain at least one item"
            raise ValueError(msg)
        return self


class EvaluationDataset(BaseModel):
    """Collection of evaluation examples plus dataset metadata."""

    name: str = "custom_json"
    examples: list[EvaluationDatasetExample]
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_path: str | None = None

    def train_test_split(
        self,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> tuple[EvaluationDataset, EvaluationDataset]:
        """Split the dataset into train and test partitions.

        Args:
            test_ratio: Fraction of examples placed in the test split.
            seed: Random seed for deterministic shuffling.

        Returns:
            Train and test dataset objects.
        """
        if not 0 < test_ratio < 1:
            msg = "test_ratio must be between 0 and 1"
            raise ValueError(msg)

        indices = list(range(len(self.examples)))
        random.Random(seed).shuffle(indices)
        test_count = max(1, int(len(indices) * test_ratio)) if len(indices) > 1 else 1
        test_indices = set(indices[:test_count])

        train_examples = [
            example
            for index, example in enumerate(self.examples)
            if index not in test_indices
        ]
        test_examples = [
            example for index, example in enumerate(self.examples) if index in test_indices
        ]

        if not train_examples:
            train_examples, test_examples = test_examples[:-1], test_examples[-1:]

        return (
            EvaluationDataset(
                name=f"{self.name}_train",
                examples=train_examples,
                metadata={**self.metadata, "split": "train"},
                source_path=self.source_path,
            ),
            EvaluationDataset(
                name=f"{self.name}_test",
                examples=test_examples,
                metadata={**self.metadata, "split": "test"},
                source_path=self.source_path,
            ),
        )


class BaseEvaluationDatasetLoader(ABC):
    """Abstract loader for evaluation dataset formats."""

    @abstractmethod
    def load(self, path: Path) -> EvaluationDataset:
        """Load and validate an evaluation dataset from disk.

        Args:
            path: Dataset file path.

        Returns:
            Parsed dataset.
        """


class JsonEvaluationDatasetLoader(BaseEvaluationDatasetLoader):
    """Loader for custom JSON or YAML evaluation datasets."""

    def load(self, path: Path) -> EvaluationDataset:
        """Load a dataset from JSON or YAML.

        Args:
            path: Dataset file path.

        Returns:
            Parsed evaluation dataset.
        """
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            dataset = EvaluationDataset.model_validate(
                {"examples": payload, "source_path": str(path)}
            )
        elif isinstance(payload, dict):
            if "examples" in payload:
                dataset = EvaluationDataset.model_validate({**payload, "source_path": str(path)})
            else:
                dataset = EvaluationDataset.model_validate(
                    {"examples": [payload], "source_path": str(path)}
                )
        else:
            msg = f"Unsupported dataset payload in {path}"
            raise ValueError(msg)
        return dataset
