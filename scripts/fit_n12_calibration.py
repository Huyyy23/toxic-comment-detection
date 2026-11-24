from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.evaluation.cnn_report import (
    apply_platt_scaling,
    apply_temperature_scaling,
    compute_calibration_metrics,
    fit_platt_scaling,
    fit_temperature_scaling,
)

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "deployment" / "hf-gradio"
sys.path.insert(0, str(BUNDLE))
from calibration import METHODS, THRESHOLD, apply_calibration

PREDICTION_PATHS = {
    "n6_lr_d": ROOT / "models/baselines/N6-LR-run-ea238c630a2f41b4b5be6b58edf4cdcd/EXP-N6-LR-D/seed_42/val_predictions.json",
    "v3_tri_gated_ft": ROOT / "models/retrained/v3/retrained_20261124/cnn/V3-TRI-GATED-FT/seed_43/val_predictions.json",
    "v3_phobert_v2": ROOT / "models/retrained/v3/retrained_20261124/transformers/V3-PHOBERT-V2/seed_44/val_predictions.json",
}
WEIGHT_PATHS = {
    "n6_lr_d": BUNDLE / "models/n6_lr_d.joblib",
    "v3_tri_gated_ft": BUNDLE / "models/tri_best.pt",
    "v3_phobert_v2": BUNDLE / "models/phobert_best.pt",
}
EXPECTED_MODEL_HASHES = {
    "n6_lr_d": "2b693b43eb74ef062ab9e0b278032abd1cfd1a1f101754a1dd379a32729b81be",
    "v3_tri_gated_ft": "97cf7e7393b613a76310d0fc69a2e6f38091d08cb54a99de4e07357cab3376d8",
    "v3_phobert_v2": "fd8db1d1c030abfae5e00ef0b0b825731dca5a0c99e9d3e504d4670ce6bf3b59",
}
DATASET_SHA256 = "4c33430a520818078aef6bcc34b3e2573873445dc7040c5729313e11bb60197a"
OUTPUT_PATH = BUNDLE / "models" / "calibration.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_id_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def load_rows() -> tuple[dict[str, list[dict[str, object]]], dict[str, str]]:
    rows: dict[str, list[dict[str, object]]] = {}
    hashes = {}
    for model_id, path in PREDICTION_PATHS.items():
        raw = path.read_bytes()
        hashes[model_id] = hashlib.sha256(raw).hexdigest()
        model_rows = json.loads(raw)
        if not isinstance(model_rows, list) or len(model_rows) != 6048:
            raise ValueError(f"{model_id} must contain exactly 6,048 prediction rows")
        rows[model_id] = model_rows
    reference = rows["n6_lr_d"]
    reference_by_id = {row["id"]: row for row in reference}
    if len(reference_by_id) != 6048 or set(reference_by_id) != {row["id"] for row in reference}:
        raise ValueError("N6 prediction IDs must be unique")
    for model_id, model_rows in rows.items():
        if len({row["id"] for row in model_rows}) != 6048:
            raise ValueError(f"{model_id} prediction IDs must be unique")
        for row in model_rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError(f"{model_id} contains an invalid prediction ID")
            base = reference_by_id.get(row["id"])
            if base is None or (row.get("true_label"), row.get("source")) != (base.get("true_label"), base.get("source")):
                raise ValueError(f"{model_id} IDs, labels, or sources do not align")
            if isinstance(row.get("true_label"), bool) or not isinstance(row.get("true_label"), int) or row["true_label"] not in (0, 1):
                raise ValueError(f"{model_id} contains an invalid label")
            if not isinstance(row.get("source"), str) or not row["source"]:
                raise ValueError(f"{model_id} contains an invalid source")
            probabilities = row.get("probabilities")
            if not isinstance(probabilities, dict) or set(probabilities) != {"CLEAN", "TOXIC"}:
                raise ValueError(f"{model_id} contains malformed class probabilities")
            clean, toxic = probabilities["CLEAN"], probabilities["TOXIC"]
            if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not np.isfinite(p) or not 0 <= p <= 1 for p in (clean, toxic)) or abs(clean + toxic - 1.0) > 2e-6:
                raise ValueError(f"{model_id} contains invalid class probabilities")
    return rows, hashes


def split_ids(rows: list[dict[str, object]]) -> tuple[list[str], list[str]]:
    ids = sorted(row["id"] for row in rows)
    by_id = {row["id"]: row for row in rows}
    strata = [f"{by_id[item]['source']}\0{by_id[item]['true_label']}" for item in ids]
    fit_ids, evaluation_ids = train_test_split(ids, test_size=0.2, random_state=42, stratify=strata)
    fit_ids, evaluation_ids = sorted(fit_ids), sorted(evaluation_ids)
    if len(fit_ids) != 4838 or len(evaluation_ids) != 1210 or set(fit_ids) & set(evaluation_ids) or set(fit_ids) | set(evaluation_ids) != set(ids):
        raise ValueError("Deterministic 80/20 calibration partition is invalid")
    return fit_ids, evaluation_ids


def metric_summary(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    metrics = compute_calibration_metrics(labels, probabilities, n_bins=10)
    return {
        "ece": float(metrics["ece"]),
        "brier_score": float(metrics["brier_score"]),
        "macro_f1": float(metrics["macro_f1_at_05"]),
    }


def main() -> None:
    weight_hashes_before = {model_id: sha256_file(path) for model_id, path in WEIGHT_PATHS.items()}
    if weight_hashes_before != EXPECTED_MODEL_HASHES:
        raise ValueError("Deployment model weights do not match the known artifact hashes")
    rows, prediction_hashes = load_rows()
    reference = {row["id"]: row for row in rows["n6_lr_d"]}
    fit_ids, evaluation_ids = split_ids(rows["n6_lr_d"])
    fit_set, evaluation_set = set(fit_ids), set(evaluation_ids)
    assert not fit_set & evaluation_set
    assert fit_set | evaluation_set == set(reference)
    assert split_ids(rows["n6_lr_d"]) == (fit_ids, evaluation_ids)

    y_fit = np.asarray([reference[item]["true_label"] for item in fit_ids], dtype=np.int64)
    y_evaluation = np.asarray([reference[item]["true_label"] for item in evaluation_ids], dtype=np.int64)
    if set(np.unique(y_fit)) != {0, 1} or set(np.unique(y_evaluation)) != {0, 1}:
        raise ValueError("Fit and evaluation partitions must each contain both labels")
    model_entries = {}
    for model_id, method in METHODS.items():
        by_id = {row["id"]: row for row in rows[model_id]}
        p_fit = np.asarray([by_id[item]["probabilities"]["TOXIC"] for item in fit_ids], dtype=np.float64)
        p_evaluation = np.asarray([by_id[item]["probabilities"]["TOXIC"] for item in evaluation_ids], dtype=np.float64)
        if method == "platt":
            params = fit_platt_scaling(p_fit, y_fit)
            calibrated_evaluation = apply_platt_scaling(p_evaluation, params["coef"], params["intercept"])
        else:
            fitted = fit_temperature_scaling(p_fit, y_fit)
            params = {"temperature": fitted["temperature"]}
            calibrated_evaluation = apply_temperature_scaling(p_evaluation, params["temperature"])
        p_all = np.asarray([by_id[item]["probabilities"]["TOXIC"] for item in sorted(by_id)], dtype=np.float64)
        if method == "platt":
            parity = np.asarray([apply_calibration(float(p), method, params) for p in p_all])
            expected_parity = apply_platt_scaling(p_all, params["coef"], params["intercept"])
        else:
            parity = np.asarray([apply_calibration(float(p), method, params) for p in p_all])
            expected_parity = apply_temperature_scaling(p_all, params["temperature"])
        assert np.allclose(expected_parity, parity, rtol=0.0, atol=1e-12)
        raw_metrics = metric_summary(y_evaluation, p_evaluation)
        calibrated_metrics = metric_summary(y_evaluation, calibrated_evaluation)
        applied = (
            calibrated_metrics["ece"] < raw_metrics["ece"]
            and calibrated_metrics["brier_score"] < raw_metrics["brier_score"]
            and calibrated_metrics["macro_f1"] >= raw_metrics["macro_f1"]
        )
        clean = 1.0 - calibrated_evaluation
        assert np.all(np.isfinite(calibrated_evaluation))
        assert np.all((calibrated_evaluation >= 0.0) & (calibrated_evaluation <= 1.0))
        assert np.allclose(clean + calibrated_evaluation, 1.0, rtol=0.0, atol=1e-15)
        entry = {
            "method": method,
            "model_sha256": weight_hashes_before[model_id],
            "prediction_sha256": prediction_hashes[model_id],
            "params": {name: float(value) for name, value in params.items()},
            "applied": applied,
            "threshold": THRESHOLD,
            "metrics_before": raw_metrics,
            "metrics_after": calibrated_metrics,
        }
        if not applied:
            entry["reason"] = "Reserved evaluation did not satisfy the activation gate."
        model_entries[model_id] = entry

    weight_hashes_after = {model_id: sha256_file(path) for model_id, path in WEIGHT_PATHS.items()}
    assert weight_hashes_after == weight_hashes_before == EXPECTED_MODEL_HASHES
    registry = {
        "schema_version": 1,
        "provenance": {
            "dataset_sha256": DATASET_SHA256,
            "split": "stratified_source_label_80_20",
            "fit_count": len(fit_ids),
            "evaluation_count": len(evaluation_ids),
            "fit_ids_sha256": ordered_id_hash(fit_ids),
            "evaluation_ids_sha256": ordered_id_hash(evaluation_ids),
            "seed": 42,
            "ece_bins": 10,
            "threshold": THRESHOLD,
            "evaluation_note": "Reserved validation holdout exposed to model selection; not independent.",
        },
        "models": model_entries,
    }
    OUTPUT_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
