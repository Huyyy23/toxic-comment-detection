from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


METHODS = {
    "n6_lr_d": "platt",
    "v3_tri_gated_ft": "temperature",
    "v3_phobert_v2": "temperature",
}
THRESHOLD = 0.5


def apply_calibration(prob_toxic: float, method: str, params: dict[str, float]) -> float:
    if not math.isfinite(prob_toxic) or not 0.0 <= prob_toxic <= 1.0:
        raise ValueError("P(TOXIC) must be finite and between 0 and 1")
    p = min(1.0 - 1e-7, max(1e-7, prob_toxic))
    logit = math.log(p / (1.0 - p))
    if method == "platt":
        value = params["coef"] * logit + params["intercept"]
    elif method == "temperature":
        if params["temperature"] <= 0:
            raise ValueError("temperature must be positive")
        value = logit / params["temperature"]
    else:
        raise ValueError(f"Unsupported calibration method: {method}")
    return min(1.0 - 1e-7, max(1e-7, 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))))


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


def _validate_metrics(metrics: Any, model_id: str, phase: str) -> None:
    if not isinstance(metrics, dict) or set(metrics) != {"ece", "brier_score", "macro_f1"}:
        raise ValueError(f"{model_id} {phase} metrics are malformed")
    for name, value in metrics.items():
        number = _finite_number(value, f"{model_id} {phase} {name}")
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{model_id} {phase} {name} is outside [0, 1]")


def load_calibration(path: Path, model_hashes: dict[str, str]) -> dict[str, dict[str, Any]]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or set(registry) != {"schema_version", "provenance", "models"}:
        raise ValueError("Calibration registry has an invalid top-level shape")
    if registry["schema_version"] != 1:
        raise ValueError("Unsupported calibration registry schema")
    provenance = registry["provenance"]
    expected_provenance = {
        "dataset_sha256", "split", "fit_count", "evaluation_count", "fit_ids_sha256",
        "evaluation_ids_sha256", "seed", "ece_bins", "threshold", "evaluation_note",
    }
    if not isinstance(provenance, dict) or set(provenance) != expected_provenance:
        raise ValueError("Calibration provenance is malformed")
    for key in ("dataset_sha256", "fit_ids_sha256", "evaluation_ids_sha256"):
        value = provenance[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Calibration provenance {key} is invalid")
    if provenance["split"] != "stratified_source_label_80_20" or provenance["seed"] != 42 or provenance["ece_bins"] != 10 or provenance["threshold"] != THRESHOLD:
        raise ValueError("Calibration provenance does not match the required protocol")
    if provenance["fit_count"] != 4838 or provenance["evaluation_count"] != 1210:
        raise ValueError("Calibration split counts do not match 6,048 rows at 80/20")
    note = provenance["evaluation_note"]
    if note != "Reserved validation holdout exposed to model selection; not independent.":
        raise ValueError("Calibration evaluation caveat is missing")
    models = registry["models"]
    if not isinstance(models, dict) or set(models) != set(METHODS):
        raise ValueError("Calibration registry must cover every deployment model")
    for model_id, expected_method in METHODS.items():
        entry = models[model_id]
        expected_keys = {
            "method", "model_sha256", "prediction_sha256", "params", "applied", "threshold",
            "metrics_before", "metrics_after",
        }
        if isinstance(entry, dict) and entry.get("applied") is False:
            expected_keys.add("reason")
        if not isinstance(entry, dict) or set(entry) != expected_keys:
            raise ValueError(f"{model_id} calibration entry is malformed")
        if entry["method"] != expected_method:
            raise ValueError(f"{model_id} calibration method is incorrect")
        if entry["model_sha256"] != model_hashes[model_id]:
            raise ValueError(f"{model_id} calibration model hash mismatch")
        prediction_hash = entry["prediction_sha256"]
        if not isinstance(prediction_hash, str) or len(prediction_hash) != 64 or any(c not in "0123456789abcdef" for c in prediction_hash):
            raise ValueError(f"{model_id} prediction hash is invalid")
        if type(entry["applied"]) is not bool or entry["threshold"] != THRESHOLD:
            raise ValueError(f"{model_id} calibration activation or threshold is invalid")
        params = entry["params"]
        names = {"coef", "intercept"} if expected_method == "platt" else {"temperature"}
        if not isinstance(params, dict) or set(params) != names:
            raise ValueError(f"{model_id} calibration parameters are malformed")
        values = {name: _finite_number(value, f"{model_id} parameter {name}") for name, value in params.items()}
        if expected_method == "temperature" and values["temperature"] <= 0:
            raise ValueError(f"{model_id} temperature must be positive")
        _validate_metrics(entry["metrics_before"], model_id, "before")
        _validate_metrics(entry["metrics_after"], model_id, "after")
        before, after = entry["metrics_before"], entry["metrics_after"]
        gate = after["ece"] < before["ece"] and after["brier_score"] < before["brier_score"] and after["macro_f1"] >= before["macro_f1"]
        if entry["applied"] and not gate:
            raise ValueError(f"{model_id} calibration is applied despite failing the activation gate")
        if not entry["applied"] and (gate or not isinstance(entry.get("reason"), str) or not entry["reason"]):
            raise ValueError(f"{model_id} calibration disabled state is inconsistent")
    return models
