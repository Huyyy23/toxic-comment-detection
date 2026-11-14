"""Read measured EXP-03 artifacts and render headless analysis, never train."""

from __future__ import annotations

import json
import math
from pathlib import Path


LABELS = ("CLEAN", "OFFENSIVE", "HATE")
STATUSES = ("SYNTHETIC_NOT_RESEARCH", "OFFICIAL")


def _number(value, name, *, unit=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (unit and value > 1):
        raise ValueError(f"{name} is outside its valid range")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _metrics(metrics, name):
    if not isinstance(metrics, dict):
        raise ValueError(f"{name} must be an object")
    _number(metrics.get("loss"), f"{name}.loss")
    _number(metrics.get("macro_f1"), f"{name}.macro_f1", unit=True)
    matrix = metrics.get("confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise ValueError(f"{name}.confusion_matrix must be 3x3")
    for row in matrix:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"{name}.confusion_matrix must be 3x3")
        for count in row:
            _integer(count, f"{name}.confusion_matrix count")
    per_class = metrics.get("per_class")
    if not isinstance(per_class, list) or len(per_class) != 3:
        raise ValueError(f"{name}.per_class must contain three class records")
    for index, record in enumerate(per_class):
        if not isinstance(record, dict):
            raise ValueError(f"{name}.per_class record must be an object")
        for key in ("precision", "recall", "f1"):
            _number(record.get(key), f"{name}.per_class.{key}", unit=True)
        _integer(record.get("support"), f"{name}.per_class.support")
        if record["support"] != sum(matrix[index]):
            raise ValueError(f"{name}.per_class support disagrees with confusion matrix")


def _validate(result):
    if not isinstance(result, dict):
        raise ValueError("Report must be an object")
    if result.get("status") not in STATUSES:
        raise ValueError(f"Explicit report status required: {STATUSES}")
    if not isinstance(result.get("architecture"), str) or not result["architecture"].strip():
        raise ValueError("Report architecture must be a nonempty string")
    if "run_id" in result and (not isinstance(result["run_id"], str) or not result["run_id"].strip()):
        raise ValueError("run_id must be a nonempty string when supplied")
    training = result.get("training")
    if not isinstance(training, dict):
        raise ValueError("training must be an object")
    history = training.get("history", [])
    if not isinstance(history, list):
        raise ValueError("training.history must be a list")
    previous_epoch = 0
    for epoch in history:
        if not isinstance(epoch, dict):
            raise ValueError("History epoch must be an object")
        _integer(epoch.get("epoch"), "epoch", minimum=1)
        if epoch["epoch"] <= previous_epoch:
            raise ValueError("History epochs must be strictly increasing")
        previous_epoch = epoch["epoch"]
        for split in ("train", "val"):
            _metrics(epoch.get(split), f"epoch {previous_epoch}.{split}")
    if training.get("best_epoch") is not None:
        _integer(training["best_epoch"], "best_epoch", minimum=1)
    if training.get("best_macro_f1") is not None:
        _number(training["best_macro_f1"], "best_macro_f1", unit=True)
    validation = result.get("last_validation")
    if validation is None:
        if history:
            raise ValueError("last_validation required when epochs exist")
        return result
    _metrics(validation, "last_validation")
    predictions = validation.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("last_validation.predictions must be a list")
    observed = [[0] * 3 for _ in range(3)]
    ids = set()
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise ValueError("Prediction must be an object")
        identifier = prediction.get("id")
        if type(identifier) is not int and not (
            type(identifier) is str and identifier
            and not any(character.isspace() for character in identifier)
        ):
            raise ValueError("Prediction IDs must be integers or nonempty whitespace-free strings")
        key = (type(identifier), identifier)
        if key in ids:
            raise ValueError("Prediction IDs must be unique")
        ids.add(key)
        if "run_id" in prediction:
            if not isinstance(prediction["run_id"], str) or not prediction["run_id"].strip():
                raise ValueError("Prediction run_id must be a nonempty string")
            if result.get("run_id") is not None and prediction["run_id"] != result["run_id"]:
                raise ValueError("Prediction run_id disagrees with report run_id")
        for key in ("true_label", "predicted_label"):
            _integer(prediction.get(key), key)
            if prediction[key] >= 3:
                raise ValueError(f"{key} must be a class index in [0, 2]")
        observed[prediction["true_label"]][prediction["predicted_label"]] += 1
        probabilities = prediction.get("probabilities_if_available")
        if probabilities is not None:
            if not isinstance(probabilities, list) or len(probabilities) != 3:
                raise ValueError("Prediction probabilities must contain three values")
            for value in probabilities:
                _number(value, "Prediction probability", unit=True)
            if not math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-5):
                raise ValueError("Prediction probabilities must sum to one")
    if observed != validation["confusion_matrix"]:
        raise ValueError("Predictions disagree with last_validation.confusion_matrix")
    return result


def load_report(path) -> dict:
    """Load smoke_results.json or its directory; reject corrupt/mislabeled evidence."""
    path = Path(path)
    if path.is_dir():
        path = path / "smoke_results.json"
    with path.open(encoding="utf-8") as handle:
        return _validate(json.load(handle))


def render_report(result, output_dir) -> dict:
    """Return {figure stem: Path}; no epochs means no figures and an explanation."""
    _validate(result)
    history = result["training"].get("history", [])
    if not history:
        print("No epochs recorded; no plots generated. CHƯA THỰC HIỆN: epoch analysis.")
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    title = f"{result['architecture']}\n{result['status']}"
    if result.get("run_id"):
        title += f"\nrun {result['run_id']}"
    paths = {}
    epochs = [record["epoch"] for record in history]
    for name, metric, ylabel in (
        ("loss_curves", "loss", "Loss"),
        ("macro_f1", "macro_f1", "Macro-F1"),
    ):
        figure, axis = plt.subplots()
        try:
            for split in ("train", "val"):
                axis.plot(epochs, [record[split][metric] for record in history], label=split)
            axis.set(xlabel="Epoch", ylabel=ylabel, title=title)
            if metric == "macro_f1":
                axis.set_ylim(0, 1)
            axis.legend()
            figure.tight_layout()
            paths[name] = output_dir / f"{name}.png"
            figure.savefig(paths[name], dpi=120)
        finally:
            plt.close(figure)

    figure, axis = plt.subplots()
    try:
        matrix = result["last_validation"]["confusion_matrix"]
        image = axis.imshow(matrix, cmap="Blues", vmin=0)
        figure.colorbar(image, ax=axis, label="Count")
        axis.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
                 xlabel="Predicted label", ylabel="True label", title=f"{title}\nLast validation (not best checkpoint)")
        for row in range(3):
            for column in range(3):
                axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
        figure.tight_layout()
        paths["confusion_matrix"] = output_dir / "confusion_matrix.png"
        figure.savefig(paths["confusion_matrix"], dpi=120)
    finally:
        plt.close(figure)
    return paths


def compare_reports(paths) -> list[dict]:
    """Preserve input order/status; absent measurements stay None, never rank runs."""
    rows = []
    for path in paths:
        result = load_report(path)
        validation = result.get("last_validation") or {}
        training = result["training"]
        rows.append({
            "architecture": result["architecture"],
            "run_id": result.get("run_id"),
            "status": result["status"],
            "val_loss": validation.get("loss"),
            "val_macro_f1": validation.get("macro_f1"),
            "best_epoch": training.get("best_epoch"),
            "best_macro_f1": training.get("best_macro_f1"),
        })
    return rows
