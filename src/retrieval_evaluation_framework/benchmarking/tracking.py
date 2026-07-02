"""Local experiment tracking for benchmark runs."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from retrieval_evaluation_framework.benchmarking.models import ExperimentRecord
from retrieval_evaluation_framework.logging import get_logger

LOGGER = get_logger(component="experiment_tracking")


class ExperimentTracker:
    """Persist and query local experiment history."""

    def __init__(self, directory: Path) -> None:
        """Initialize the experiment tracker.

        Args:
            directory: Directory where experiment JSON files are stored.
        """
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def create_experiment(self, record: ExperimentRecord) -> ExperimentRecord:
        """Persist a new experiment record.

        Args:
            record: Experiment metadata to persist.

        Returns:
            The same experiment record.
        """
        self.save(record)
        LOGGER.info("experiment_created", experiment_id=record.experiment_id, mode=record.mode)
        return record

    def next_experiment_id(self) -> str:
        """Generate a unique experiment identifier."""
        return f"exp-{uuid.uuid4().hex[:12]}"

    def save(self, record: ExperimentRecord) -> Path:
        """Write an experiment record to disk.

        Args:
            record: Experiment record to persist.

        Returns:
            The written file path.
        """
        path = self.directory / f"{record.experiment_id}.json"
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return path

    def list_experiments(self) -> list[ExperimentRecord]:
        """Return all persisted experiments sorted by newest first."""
        records = [
            ExperimentRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.directory.glob("exp-*.json"))
        ]
        return sorted(records, key=lambda record: record.timestamp, reverse=True)

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        """Load one persisted experiment by ID.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            Parsed experiment record.
        """
        path = self.directory / f"{experiment_id}.json"
        if not path.exists():
            msg = f"Experiment not found: {experiment_id}"
            raise FileNotFoundError(msg)
        return ExperimentRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def export_index(self) -> Path:
        """Write a consolidated history index for external tooling.

        Returns:
            The written index path.
        """
        records = [record.model_dump(mode="json") for record in self.list_experiments()]
        path = self.directory / "experiments_index.json"
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return path
