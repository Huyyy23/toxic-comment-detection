"""MLflow local (file store) experiment tracking.

Implements ADR-08: MLflow local is the tracking center.
All training runs log: Experiment ID, Run ID, dataset version/hash,
split config, model architecture, hyperparameters, random seed,
training duration, train/val metrics, environment metadata,
checkpoint/artifacts, git commit (if available).

Uses a local file store at `./mlruns/` — no MLflow server required.
"""

from __future__ import annotations

import os
import time
import json
import hashlib
import subprocess
from contextlib import contextmanager
from typing import Any, Dict, Optional

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
except Exception:  # pragma: no cover - mlflow installed at runtime
    mlflow = None  # type: ignore


class MLflowTracker:
    """Thin wrapper around MLflow local file-store tracking."""

    def __init__(
        self,
        experiment_name: str = "vietnamese-toxic-detection",
        tracking_uri: Optional[str] = None,
    ) -> None:
        if mlflow is None:
            raise ImportError(
                "mlflow is not installed. Run: pip install mlflow>=2.0.0,<3.0.0"
            )
        self.tracking_uri = tracking_uri or os.path.join(
            os.getcwd(), "mlruns"
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        self.experiment = mlflow.set_experiment(experiment_name)
        self.experiment_id = self.experiment.experiment_id
        self._run: Optional[Any] = None
        self._run_id: Optional[str] = None
        self._start_time: Optional[float] = None

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start a new MLflow run and return its Run ID."""
        if self._run is not None:
            raise RuntimeError("A run is already active. Call end_run() first.")
        self._run = mlflow.start_run(run_name=run_name, tags=tags or {})
        self._run_id = self._run.info.run_id
        self._start_time = time.time()
        return self._run_id

    def end_run(self) -> None:
        """End the active MLflow run."""
        if self._run is not None:
            mlflow.end_run()
            self._run = None
            self._run_id = None
            self._start_time = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to the active run."""
        if self._run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        for key, value in params.items():
            mlflow.log_param(str(key), value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to the active run."""
        if self._run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        for key, value in metrics.items():
            mlflow.log_metric(str(key), float(value), step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a local file/directory as an artifact."""
        if self._run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        mlflow.log_artifact(local_path, artifact_path)

    def log_training_duration(self) -> Optional[float]:
        """Log elapsed time since start_run() was called."""
        if self._start_time is None:
            return None
        duration = time.time() - self._start_time
        self.log_metrics({"training_duration_seconds": duration})
        return duration

    def log_dataset_info(
        self,
        dataset_name: str,
        dataset_path: str,
        split_config: Dict[str, float],
        label_distribution: Optional[Dict[str, int]] = None,
    ) -> None:
        """Log dataset version/hash and split configuration."""
        file_hash = self._hash_file(dataset_path) if dataset_path and os.path.exists(dataset_path) else None
        self.log_params(
            {
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "dataset_hash": file_hash,
                "train_split": split_config.get("train", 0.0),
                "val_split": split_config.get("val", 0.0),
                "test_split": split_config.get("test", 0.0),
            }
        )
        if label_distribution:
            for label, count in label_distribution.items():
                mlflow.log_metric(f"dataset_{label}_count", float(count))

    def log_model_architecture(self, architecture: str, model_type: str) -> None:
        """Log model architecture and type."""
        self.log_params({"model_architecture": architecture, "model_type": model_type})

    def log_environment(self) -> None:
        """Log environment metadata (Python version, git commit if available)."""
        import sys

        self.log_params({"python_version": sys.version})
        git_commit = self._get_git_commit()
        if git_commit:
            self.log_params({"git_commit": git_commit})

    @staticmethod
    def _hash_file(path: str) -> Optional[str]:
        """Return SHA-256 of a file for dataset versioning."""
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    @staticmethod
    def _get_git_commit() -> Optional[str]:
        """Return short git commit hash if available."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    @contextmanager
    def run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Context manager for an MLflow run."""
        self.start_run(run_name=run_name, tags=tags)
        try:
            yield self
        finally:
            self.end_run()


# Module-level convenience functions backed by a default tracker instance.
_default_tracker: Optional[MLflowTracker] = None


def _get_default_tracker(experiment_name: str = "vietnamese-toxic-detection") -> MLflowTracker:
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = MLflowTracker(experiment_name=experiment_name)
    return _default_tracker


def start_run(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
    return _get_default_tracker().start_run(run_name=run_name, tags=tags)


def end_run() -> None:
    _get_default_tracker().end_run()


def log_params(params: Dict[str, Any]) -> None:
    _get_default_tracker().log_params(params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    _get_default_tracker().log_metrics(metrics, step=step)


def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    _get_default_tracker().log_artifact(local_path, artifact_path)
