"""MLflow artifact contract for recurrent experiment runs; no training logic."""
import json
import platform
from pathlib import Path
from urllib.parse import urlparse

import torch

from src.tracking.mlflow_tracker import MLflowTracker


LABELS = ("CLEAN", "OFFENSIVE", "HATE")
BINARY_LABELS = ("CLEAN", "TOXIC")


def _label_names(num_classes: int) -> tuple[str, ...]:
    if num_classes == 2:
        return BINARY_LABELS
    if num_classes == 3:
        return LABELS
    return tuple(f"CLASS_{index}" for index in range(num_classes))


def log_epoch(tracker, state):
    """Record metrics at completed epoch, without prediction rows or raw text."""
    row = state["history"][-1]
    metrics = {}
    for split in ("train", "val"):
        values = row[split]
        metrics.update({f"{split}_{name}": values[name] for name in ("loss", "macro_f1")})
        labels = _label_names(len(values["per_class"]))
        for label, scores in zip(labels, values["per_class"]):
            for name in ("precision", "recall", "f1", "support"):
                metrics[f"{split}_{label}_{name}"] = scores[name]
    tracker.log_metrics(metrics, step=state["epoch"])


def start_tracking(tracking_uri, *, architecture, resume, official=False, smoke=True):
    """Create EXP-03 run. Synthetic by default; official when official=True."""
    uri = str(tracking_uri)
    parsed = urlparse(uri)
    if ("://" in uri and parsed.scheme != "file") or parsed.netloc or uri.startswith("\\\\"):
        raise ValueError("tracking requires a local path or file URI")
    if official:
        experiment_name = "EXP-03-official"
        result_type = "official_smoke" if smoke else "official"
        entry_point = "train_recurrent_official"
    else:
        experiment_name = "EXP-03-synthetic-smoke"
        result_type = "synthetic"
        entry_point = "smoke_recurrent"
    tracker = MLflowTracker(experiment_name=experiment_name, tracking_uri=uri)
    tags = {"exp": "EXP-03", "result_type": result_type,
            "research_result": "false" if smoke else "true",
            "entry_point": entry_point, "architecture": architecture,
            "pretrained": "false", "prediction_checkpoint": "best"}
    if resume is not None:
        tags["resumed_from_checkpoint"] = Path(resume).name
    tracker.start_run(run_name=f"{result_type}-{architecture}", tags=tags)
    return tracker


def log_run_metadata(tracker, *, config, model_config, data_metadata, device):
    tracker.log_params({**config, **model_config, **data_metadata,
                        "torch_version": str(torch.__version__), "device_resolved": str(device),
                        "platform": platform.platform(), "probabilities_calibrated": False})
    tracker.log_environment()


def finish_tracking(tracker, output, summary, *, config, data_metadata, duration_seconds,
                    write_artifacts=True):
    """Record run provenance and metrics; optionally write the legacy smoke artifacts."""
    output = Path(output)
    summary["run_id"] = tracker.run_id
    summary["experiment_id"] = tracker.experiment_id
    summary["data_metadata"] = data_metadata
    summary["prediction_checkpoint"] = "best"
    summary["training_duration_seconds"] = duration_seconds
    for prediction in summary["last_validation"]["predictions"]:
        prediction["run_id"] = tracker.run_id
    tracker.log_metrics({"training_duration_seconds": duration_seconds,
                         "best_val_macro_f1": summary["training"]["best_macro_f1"],
                         "last_val_macro_f1": summary["last_validation"]["macro_f1"]})
    if write_artifacts:
        (output / "validation_predictions.json").write_text(
            json.dumps(
                summary["last_validation"]["predictions"], ensure_ascii=False, indent=2
            ) + "\n",
            encoding="utf-8",
        )
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (output / "data_metadata.json").write_text(
            json.dumps(data_metadata, indent=2) + "\n", encoding="utf-8")


def log_run_artifacts(tracker, output):
    for name in ("smoke_results.json", "validation_predictions.json", "config.json",
                 "data_metadata.json", "vocabulary.json", "last.pt", "best.pt"):
        path = Path(output) / name
        if path.is_file():
            tracker.log_artifact(str(path))
