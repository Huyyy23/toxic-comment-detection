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
from pathlib import Path
import hashlib
import subprocess
from contextlib import contextmanager
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from urllib.request import url2pathname

try:
    import mlflow
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name != "mlflow":
        raise
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
                "mlflow is not installed. Run: pip install 'mlflow>=2.0.0,<3.0.0'"
            )
        uri = os.fspath(tracking_uri) if tracking_uri is not None else os.path.join(os.getcwd(), "mlruns")
        parsed = urlparse(uri)
        if parsed.scheme == "file":
            local_path = url2pathname(parsed.path)
            if parsed.netloc and parsed.netloc != "localhost":
                local_path = f"//{parsed.netloc}{local_path}"
            self.tracking_uri = Path(local_path).resolve().as_uri()
        elif not parsed.scheme or (len(parsed.scheme) == 1 and uri[1:2] == ":"):
            self.tracking_uri = Path(uri).resolve().as_uri()
        else:
            self.tracking_uri = uri
        client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
        self.experiment = client.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            experiment_id = client.create_experiment(experiment_name)
            self.experiment = client.get_experiment(experiment_id)
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
        if mlflow.active_run() is not None:
            raise RuntimeError("Another MLflow run is active. End it before starting a run.")
        mlflow.set_tracking_uri(self.tracking_uri)
        self._run = mlflow.start_run(
            experiment_id=self.experiment_id, run_name=run_name, tags=tags or {}
        )
        self._run_id = self._run.info.run_id
        self._start_time = time.time()
        return self._run_id

    def _activate_run(self) -> None:
        """Restore this store without logging to or ending another owner's run."""
        if self._run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        active = mlflow.active_run()
        if active is None or active.info.run_id != self._run_id:
            raise RuntimeError("This tracker's run is not the active MLflow run.")
        mlflow.set_tracking_uri(self.tracking_uri)

    def end_run(self, status: str = "FINISHED") -> None:
        """End this tracker's active run with the supplied MLflow status."""
        if self._run is not None:
            self._activate_run()
            mlflow.end_run(status=status)
            self._run = None
            self._run_id = None
            self._start_time = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to the active run."""
        self._activate_run()
        for key, value in params.items():
            mlflow.log_param(str(key), value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to the active run."""
        self._activate_run()
        for key, value in metrics.items():
            mlflow.log_metric(str(key), float(value), step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a local file/directory as an artifact."""
        self._activate_run()
        if os.path.isdir(local_path):
            mlflow.log_artifacts(local_path, artifact_path)
        else:
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
            self.log_metrics({f"dataset_{label}_count": float(count)
                              for label, count in label_distribution.items()})

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
        except BaseException:
            self.end_run(status="FAILED")
            raise
        else:
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


def end_run(status: str = "FINISHED") -> None:
    _get_default_tracker().end_run(status=status)


def log_params(params: Dict[str, Any]) -> None:
    _get_default_tracker().log_params(params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    _get_default_tracker().log_metrics(metrics, step=step)


def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    _get_default_tracker().log_artifact(local_path, artifact_path)
