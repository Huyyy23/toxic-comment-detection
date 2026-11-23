"""Phase 4 calibration, visualization, integrity audit, and error analysis for TV4 CNN runs."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, log_loss
import yaml

from src.models.cnn_pipeline import load_cnn_artifacts, predict_cnn_from_artifacts
from src.models.svm_binary_pipeline import (
    BINARY_LABEL_NAMES,
    _build_aggregate_error_summary,
    compute_ece,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CNN_CONFIG_PATH = ROOT / "configs" / "cnn_config.yaml"
DEFAULT_CNN_MODELS_DIR = ROOT / "models" / "cnn"
DEFAULT_REPORT_OUT_DIR = ROOT / "artifacts" / "TV4-cnn-analysis"
BINARY_N2_DIR = ROOT / "data" / "processed" / "binary_n2_v1"
BINARY_N4_DIR = ROOT / "data" / "processed" / "binary_n4_training_v1"

_URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE)
_MENTION_RE = re.compile(r"@[A-Za-z0-9_.\-]+")
_CAMEL_HANDLE_RE = re.compile(
    r"\b[A-Za-zÀ-ỹ]*[a-zà-ỹ][A-Za-zÀ-ỹ]*[0-9A-Z][A-Za-z0-9_]*\b"
)
_LONG_NUM_RE = re.compile(r"\b\d{5,}\b")
_TITLE_NAME_RE = re.compile(
    r"\b[A-ZÀ-Ỹ][a-zà-ỹ]{1,18}(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]{1,18})+\b"
)
_MID_PROPER_RE = re.compile(r"(?<![.!?\n]\s)(?<!^)(?<!<)\b[A-ZÀ-Ỹ][a-zà-ỹ]{1,18}\b")
_WHITESPACE_RE = re.compile(r"\s+")

_TEENCODE_RE = re.compile(
    r"\b(?:ko|k|kh|hok|hem|k0|dc|đc|vs|ntn|vcl|vl|vkl|cl|clgt|cmn|cmm|dcm|dm|đm|"
    r"cc|sml|bn|j|z|r|ng|nx|ad|m|t|đb|vc|đkm|dkm)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:không|ko|k|chẳng|chả|chưa|đừng|đéo|đâu|chớ|ít|hiếm|nào)\b",
    re.IGNORECASE,
)
_OBFUSCATED_RE = re.compile(
    r"(?:[a-zà-ỹ][.*_\-][a-zà-ỹ]|[a-zà-ỹ]{1,2}\*{1,4}[a-zà-ỹ]{0,2}|(.)\1{2,})",
    re.IGNORECASE,
)
_EMOJI_PUNCT_RE = re.compile(r"(?:[!?~]{2,}|[\U00010000-\U0010ffff]|[\u2600-\u27bf])")


def redact_comment_snippet(text: Any, max_len: int = 84) -> str:
    """Return a compacted, privacy-redacted comment snippet safe for reports."""
    clean = "" if text is None else str(text)
    clean = _URL_RE.sub("<URL>", clean)
    clean = _EMAIL_RE.sub("<EMAIL>", clean)
    clean = _MENTION_RE.sub("<USER>", clean)
    clean = _CAMEL_HANDLE_RE.sub("<USER>", clean)
    clean = _LONG_NUM_RE.sub("<NUM>", clean)
    clean = _TITLE_NAME_RE.sub("<NAME>", clean)
    clean = _MID_PROPER_RE.sub("<NAME>", clean)
    clean = _WHITESPACE_RE.sub(" ", clean).strip()
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3].rstrip() + "..."


def classify_error_tags(text: Any) -> list[str]:
    """Tag linguistic and structural error phenomena present in a comment."""
    raw = "" if text is None else str(text)
    words = raw.split()
    tags: list[str] = []
    if _TEENCODE_RE.search(raw):
        tags.append("contains_teencode")
    if _NEGATION_RE.search(raw):
        tags.append("contains_negation")
    if len(words) <= 5:
        tags.append("short_text_le_5_words")
    elif len(words) >= 35:
        tags.append("long_text_ge_35_words")
    if _URL_RE.search(raw) or _MENTION_RE.search(raw) or _EMAIL_RE.search(raw):
        tags.append("contains_url_or_mention")
    if _EMOJI_PUNCT_RE.search(raw):
        tags.append("contains_emoji_or_punct")
    if _OBFUSCATED_RE.search(raw):
        tags.append("obfuscated_or_repeated_chars")
    if not tags:
        tags.append("context_or_lexical_nuance")
    return tags


def clip_probs(probs: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Clip probability predictions away from {0, 1} and extract 1D P(TOXIC)."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError(f"Expected binary 2-column probability array, got {arr.shape}")
        arr = arr[:, 1]
    elif arr.ndim != 1:
        raise ValueError(f"Expected 1D or 2D array, got ndim={arr.ndim}")
    return np.clip(arr, eps, 1.0 - eps)


def probs_to_logits(prob_toxic: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Convert binary P(TOXIC) probabilities into 1D log-odds logits."""
    clipped = clip_probs(prob_toxic, eps=eps)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def fit_temperature_scaling(
    prob_toxic: np.ndarray, y_true: np.ndarray
) -> dict[str, float]:
    """Fit single-scalar Temperature Scaling T > 0 on binary validation log-odds."""
    y = np.asarray(y_true, dtype=np.float64)
    logits = probs_to_logits(prob_toxic)
    if len(np.unique(y)) < 2:
        p = clip_probs(prob_toxic)
        return {
            "temperature": 1.0,
            "nll": float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))),
        }

    def _nll(temp: float) -> float:
        p = np.clip(_sigmoid(logits / temp), 1e-7, 1.0 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    res = minimize_scalar(_nll, bounds=(0.05, 10.0), method="bounded")
    best_temp = float(res.x)
    return {"temperature": best_temp, "nll": float(res.fun)}


def apply_temperature_scaling(prob_toxic: np.ndarray, temperature: float) -> np.ndarray:
    """Apply fitted Temperature Scaling parameter T to P(TOXIC) log-odds."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    logits = probs_to_logits(prob_toxic)
    return np.clip(_sigmoid(logits / float(temperature)), 1e-7, 1.0 - 1e-7)


def fit_platt_scaling(prob_toxic: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Fit 1D Platt Scaling (LogisticRegression on log-odds) on validation set."""
    y = np.asarray(y_true, dtype=np.int64)
    logits = probs_to_logits(prob_toxic)
    if len(np.unique(y)) < 2:
        return {"coef": 1.0, "intercept": 0.0}
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(logits[:, None], y)
    return {
        "coef": float(lr.coef_[0, 0]),
        "intercept": float(lr.intercept_[0]),
    }


def apply_platt_scaling(
    prob_toxic: np.ndarray, coef: float, intercept: float
) -> np.ndarray:
    """Apply fitted Platt scaling affine transform to binary P(TOXIC) log-odds."""
    logits = probs_to_logits(prob_toxic)
    return np.clip(_sigmoid(float(coef) * logits + float(intercept)), 1e-7, 1.0 - 1e-7)


def compute_calibration_metrics(
    y_true: np.ndarray, prob_toxic: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    """Compute ECE, log-loss, Brier score, threshold-0.50 F1s, and reliability bins."""
    y = np.asarray(y_true, dtype=np.int64)
    p = clip_probs(prob_toxic)
    pred_05 = (p >= 0.5).astype(np.int64)
    ece = float(compute_ece(y, p, n_bins=n_bins))
    nll = float(log_loss(y, p, labels=[0, 1]))
    brier = float(brier_score_loss(y, p))
    macro_05 = float(f1_score(y, pred_05, average="macro", zero_division=0))
    toxic_05 = float(f1_score(y, pred_05, pos_label=1, average="binary", zero_division=0))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float | int | None]] = []
    for idx in range(n_bins):
        low, high = float(bin_edges[idx]), float(bin_edges[idx + 1])
        mask = (p >= low) & (p <= high if idx == n_bins - 1 else p < high)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin_low": low,
                    "bin_high": high,
                    "center": (low + high) / 2.0,
                    "mean_conf": None,
                    "emp_pos_rate": None,
                    "count": 0,
                }
            )
        else:
            bins.append(
                {
                    "bin_low": low,
                    "bin_high": high,
                    "center": (low + high) / 2.0,
                    "mean_conf": float(np.mean(p[mask])),
                    "emp_pos_rate": float(np.mean(y[mask])),
                    "count": count,
                }
            )
    return {
        "ece": ece,
        "log_loss": nll,
        "brier_score": brier,
        "macro_f1_at_05": macro_05,
        "f1_toxic_at_05": toxic_05,
        "n_bins": n_bins,
        "reliability_bins": bins,
    }


def sweep_probability_thresholds(
    y_true: np.ndarray,
    prob_toxic: np.ndarray,
    t_min: float = 0.10,
    t_max: float = 0.90,
    t_step: float = 0.01,
) -> dict[str, Any]:
    """Sweep probability thresholds on validation (diagnostic in-sample only)."""
    y = np.asarray(y_true, dtype=np.int64)
    p = clip_probs(prob_toxic)
    grid = [round(float(x), 4) for x in np.arange(t_min, t_max + t_step * 0.25, t_step)]
    macro_scores: list[float] = []
    toxic_scores: list[float] = []
    for thr in grid:
        pred = (p >= thr).astype(np.int64)
        macro_scores.append(float(f1_score(y, pred, average="macro", zero_division=0)))
        toxic_scores.append(
            float(f1_score(y, pred, pos_label=1, average="binary", zero_division=0))
        )
    idx_macro = int(np.argmax(macro_scores))
    idx_toxic = int(np.argmax(toxic_scores))
    return {
        "in_sample_validation_only": True,
        "grid": grid,
        "macro_f1_scores": macro_scores,
        "f1_toxic_scores": toxic_scores,
        "best_macro_f1_threshold": float(grid[idx_macro]),
        "best_macro_f1": float(macro_scores[idx_macro]),
        "f1_toxic_at_macro_opt": float(toxic_scores[idx_macro]),
        "best_toxic_threshold": float(grid[idx_toxic]),
        "best_f1_toxic": float(toxic_scores[idx_toxic]),
        "macro_f1_at_toxic_opt": float(macro_scores[idx_toxic]),
    }


def evaluate_run_calibration(seed_dir: Path) -> dict[str, Any]:
    """Compute Raw, Temperature Scaling, and Platt Scaling calibration diagnostics."""
    pred_csv = pd.read_csv(seed_dir / "predictions_val.csv")
    metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))

    y_true = pred_csv["y_true"].to_numpy(dtype=np.int64)
    p_raw = pred_csv["prob_TOXIC"].to_numpy(dtype=np.float64)

    raw_metrics = compute_calibration_metrics(y_true, p_raw)
    raw_sweep = sweep_probability_thresholds(y_true, p_raw)

    temp_fit = fit_temperature_scaling(p_raw, y_true)
    p_temp = apply_temperature_scaling(p_raw, temp_fit["temperature"])
    temp_metrics = compute_calibration_metrics(y_true, p_temp)

    platt_fit = fit_platt_scaling(p_raw, y_true)
    p_platt = apply_platt_scaling(p_raw, platt_fit["coef"], platt_fit["intercept"])
    platt_metrics = compute_calibration_metrics(y_true, p_platt)
    platt_sweep = sweep_probability_thresholds(y_true, p_platt)

    return {
        "variant": cfg["variant"],
        "seed": int(cfg["seed"]),
        "run_id": cfg["run_id"],
        "seed_dir": str(seed_dir),
        "in_sample_validation_only": True,
        "selection_ranking_frozen_pre_calibration": True,
        "uncalibrated_val_macro_f1": float(metrics["macro_f1"]),
        "uncalibrated_val_f1_toxic": float(metrics["f1_toxic"]),
        "raw": raw_metrics,
        "temperature_scaling": {
            "params": temp_fit,
            "metrics": temp_metrics,
        },
        "platt_scaling": {
            "params": platt_fit,
            "metrics": platt_metrics,
        },
        "threshold_sweep_raw": raw_sweep,
        "threshold_sweep_platt": platt_sweep,
    }


def load_expected_dataset_hashes(root: Path = ROOT) -> dict[str, dict[str, str]]:
    """Load official SHA-256 hashes for train/val parquet from dataset_info.json."""
    expected: dict[str, dict[str, str]] = {}
    for key, ds_dir in (("n2", root / "data" / "processed" / "binary_n2_v1"),
                        ("n4", root / "data" / "processed" / "binary_n4_training_v1")):
        info_path = ds_dir / "dataset_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            sha_map = info.get("output_sha256", {})
            expected[key] = {
                "train_parquet_sha256": str(sha_map.get("train.parquet", "")),
                "val_parquet_sha256": str(sha_map.get("val.parquet", "")),
            }
    return expected


def _load_mlflow_statuses(mlruns_dir: Path) -> dict[str, str]:
    """Query local MLflow file store for run statuses without modifying tracking state."""
    if not mlruns_dir.exists():
        return {}
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=mlruns_dir.resolve().as_uri())
        statuses: dict[str, str] = {}
        for exp in client.search_experiments():
            runs = client.search_runs([exp.experiment_id], max_results=500)
            for run in runs:
                statuses[run.info.run_id] = str(run.info.status)
        return statuses
    except Exception:
        return {}


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _verify_seed_artifacts(
    seed_dir: Path,
    expected_hashes: dict[str, dict[str, str]],
    mlflow_statuses: dict[str, str],
    val_df: pd.DataFrame | None = None,
    verify_reload: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    required_files = [
        "best.pt",
        "last.pt",
        "config.json",
        "metrics.json",
        "history.json",
        "val_predictions.json",
        "predictions_val.csv",
        "error_summary.json",
        "vocab.json",
        "run_id.txt",
    ]
    missing = [name for name in required_files if not (seed_dir / name).exists()]
    if missing:
        return {"ok": False, "missing_files": missing}

    cfg = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
    val_preds = json.loads((seed_dir / "val_predictions.json").read_text(encoding="utf-8"))
    csv_preds = pd.read_csv(seed_dir / "predictions_val.csv")
    run_id = (seed_dir / "run_id.txt").read_text(encoding="utf-8").strip()

    ds_key = "n4" if cfg.get("use_augmentation") else "n2"
    exp_ds = expected_hashes.get(ds_key, {})
    actual_hashes = cfg.get("data_hashes", {})
    hash_ok = (
        not exp_ds
        or (
            actual_hashes.get("train_parquet_sha256") == exp_ds.get("train_parquet_sha256")
            and actual_hashes.get("val_parquet_sha256") == exp_ds.get("val_parquet_sha256")
        )
    )
    prob_sums_ok = all(
        abs(row["probabilities"]["CLEAN"] + row["probabilities"]["TOXIC"] - 1.0) <= 1e-4
        for row in val_preds
    )
    mlflow_status = mlflow_statuses.get(run_id, "UNKNOWN")
    reload_max_diff: float | None = None
    reload_exact_match: int | None = None

    if verify_reload and val_df is not None:
        artifacts = load_cnn_artifacts(seed_dir, device=device)
        preds, probs = predict_cnn_from_artifacts(artifacts, val_df["text"].astype(str).tolist())
        saved_probs = csv_preds[["prob_CLEAN", "prob_TOXIC"]].to_numpy(dtype=np.float64)
        saved_preds = csv_preds["y_pred"].tolist()
        reload_max_diff = float(np.max(np.abs(probs - saved_probs)))
        reload_exact_match = int(sum(int(a == b) for a, b in zip(preds, saved_preds)))

    ok = (
        hash_ok
        and prob_sums_ok
        and len(val_preds) == len(csv_preds) == int(metrics["n_samples"])
        and metrics.get("is_calibrated") is False
        and (reload_exact_match is None or reload_exact_match == len(csv_preds))
        and (reload_max_diff is None or reload_max_diff <= 5e-4)
    )
    return {
        "ok": bool(ok),
        "run_id": run_id,
        "dataset_split_key": ds_key,
        "hash_match": bool(hash_ok),
        "prob_sums_ok": bool(prob_sums_ok),
        "n_val_rows": int(len(csv_preds)),
        "mlflow_status": mlflow_status,
        "reload_exact_match": reload_exact_match,
        "reload_max_diff": reload_max_diff,
        "reload_tol": 5e-4,
    }


def _summarize_variant(
    variant_entry: dict[str, Any],
    v_cfg: dict[str, Any],
    seed_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    runs = sorted(variant_entry["runs"], key=lambda r: int(r["seed"]))
    best_run = max(runs, key=lambda r: float(r["metrics"]["macro_f1"]))
    by_source: dict[str, dict[str, Any]] = {}
    for src in ("victsd", "vihsd"):
        by_source[src] = {
            "macro_f1": _mean_std(
                [float(r["metrics"]["by_source"][src]["macro_f1"]) for r in runs]
            ),
            "f1_toxic": _mean_std(
                [float(r["metrics"]["by_source"][src]["f1_toxic"]) for r in runs]
            ),
            "pr_auc": _mean_std(
                [float(r["metrics"]["by_source"][src]["pr_auc"]) for r in runs]
            ),
        }
    return {
        "variant": variant_entry["variant"],
        "architecture": variant_entry["architecture"],
        "preprocessing": variant_entry["preprocessing"],
        "use_augmentation": bool(variant_entry.get("use_augmentation", False)),
        "imbalance_method": v_cfg.get("imbalance_method", "none"),
        "weight_decay": float(v_cfg.get("weight_decay", 0.0)),
        "description": str(v_cfg.get("description", "")),
        "hypothesis": str(v_cfg.get("hypothesis", "")).strip(),
        "model_config": dict(v_cfg.get("model", {})),
        "variant_dir": variant_entry["variant_dir"],
        "n_seeds": len(runs),
        "seeds": [int(r["seed"]) for r in runs],
        "macro_f1": _mean_std([float(r["metrics"]["macro_f1"]) for r in runs]),
        "f1_toxic": _mean_std([float(r["metrics"]["f1_toxic"]) for r in runs]),
        "weighted_f1": _mean_std([float(r["metrics"]["weighted_f1"]) for r in runs]),
        "balanced_accuracy": _mean_std([float(r["metrics"]["balanced_accuracy"]) for r in runs]),
        "pr_auc": _mean_std([float(r["metrics"]["pr_auc"]) for r in runs]),
        "raw_softmax_ece": _mean_std([float(r["metrics"]["raw_softmax_ece"]) for r in runs]),
        "train_time_s": _mean_std([float(r["metrics"]["train_time_s"]) for r in runs]),
        "inference_ms_per_sample": _mean_std(
            [float(r["metrics"]["inference_ms_per_sample"]) for r in runs]
        ),
        "peak_vram_mb": float(max(float(r["metrics"]["peak_vram_mb"]) for r in runs)),
        "model_size_mb": float(runs[0]["metrics"]["model_size_mb"]),
        "num_params": int(runs[0]["metrics"]["num_params"]),
        "by_source": by_source,
        "best_seed": int(best_run["seed"]),
        "best_seed_macro_f1": float(best_run["metrics"]["macro_f1"]),
        "best_seed_f1_toxic": float(best_run["metrics"]["f1_toxic"]),
        "best_seed_dir": str(best_run["seed_dir"]),
        "seed_runs": runs,
        "seed_checks": seed_checks,
    }


def discover_tv4_variants(
    models_dir: Path = DEFAULT_CNN_MODELS_DIR,
    cnn_config_path: Path = DEFAULT_CNN_CONFIG_PATH,
    root: Path = ROOT,
    check_mlflow: bool = True,
    verify_reload: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load, verify, and aggregate all completed TV4 CNN variants from run summaries."""
    cfg_raw = yaml.safe_load(cnn_config_path.read_text(encoding="utf-8"))
    variant_cfgs: dict[str, Any] = cfg_raw.get("variants", {})
    expected_hashes = load_expected_dataset_hashes(root=root)
    mlflow_statuses = _load_mlflow_statuses(root / "mlruns") if check_mlflow else {}

    val_df: pd.DataFrame | None = None
    if verify_reload:
        val_df = pd.read_parquet(root / "data" / "processed" / "binary_n2_v1" / "val.parquet")

    summary_files = sorted(
        models_dir.glob("TV4-run-*/TV4-CNN-summary.json"),
        key=lambda p: p.stat().st_mtime,
    )
    latest_by_variant: dict[str, dict[str, Any]] = {}
    for summary_path in summary_files:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            v_name = entry["variant"]
            if len(entry.get("runs", [])) >= 1:
                latest_by_variant[v_name] = entry

    summaries: list[dict[str, Any]] = []
    ordered_names = list(variant_cfgs.keys()) + [
        name for name in latest_by_variant if name not in variant_cfgs
    ]
    all_checks_ok = True
    for name in ordered_names:
        entry = latest_by_variant.get(name)
        if entry is None:
            continue
        seed_checks: list[dict[str, Any]] = []
        for run_info in sorted(entry["runs"], key=lambda r: int(r["seed"])):
            chk = _verify_seed_artifacts(
                seed_dir=Path(run_info["seed_dir"]),
                expected_hashes=expected_hashes,
                mlflow_statuses=mlflow_statuses,
                val_df=val_df,
                verify_reload=verify_reload,
                device=device,
            )
            chk["variant"] = name
            chk["seed"] = int(run_info["seed"])
            seed_checks.append(chk)
            if not chk["ok"]:
                all_checks_ok = False
        summaries.append(
            _summarize_variant(entry, variant_cfgs.get(name, {}), seed_checks)
        )

    ranked = sorted(
        summaries,
        key=lambda item: (item["macro_f1"]["mean"], item["f1_toxic"]["mean"]),
        reverse=True,
    )
    return {
        "n_variants_found": len(ranked),
        "n_runs_total": sum(item["n_seeds"] for item in ranked),
        "all_integrity_checks_passed": bool(all_checks_ok and len(ranked) > 0),
        "variants_by_name": {item["variant"]: item for item in ranked},
        "ranking": ranked,
    }


def _length_bucket(word_count: int) -> str:
    if word_count <= 5:
        return "le_5_words"
    if word_count <= 15:
        return "6_to_15_words"
    if word_count <= 30:
        return "16_to_30_words"
    return "gt_30_words"


def build_privacy_safe_error_analysis(
    seed_dir: Path,
    val_df: pd.DataFrame,
    max_examples: int = 40,
    seed: int = 42,
) -> dict[str, Any]:
    """Build privacy-safe validation error analysis with >=30 redacted cases."""
    pred_df = pd.read_csv(seed_dir / "predictions_val.csv")
    cfg = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))
    val_lookup = val_df[["id", "text"]].copy()
    merged = pred_df.merge(
        val_lookup,
        left_on="sample_id",
        right_on="id",
        how="inner",
    )
    if len(merged) != len(val_df):
        raise ValueError(
            f"Validation merge count mismatch: {len(merged)} vs {len(val_df)}"
        )

    agg_summary = _build_aggregate_error_summary(val_df, pred_df["y_pred"].tolist())
    merged["is_error"] = merged["y_true"] != merged["y_pred"]
    merged["word_count"] = merged["text"].astype(str).apply(lambda s: len(s.split()))
    merged["char_count"] = merged["text"].astype(str).apply(len)
    merged["length_bucket"] = merged["word_count"].apply(_length_bucket)
    merged["tags"] = merged["text"].astype(str).apply(classify_error_tags)

    tag_error_rates: dict[str, dict[str, Any]] = {}
    all_tags = sorted({tag for tags in merged["tags"] for tag in tags})
    for tag in all_tags:
        mask = merged["tags"].apply(lambda tags: tag in tags)
        n_tag = int(mask.sum())
        n_err = int((mask & merged["is_error"]).sum())
        tag_error_rates[tag] = {
            "n_samples": n_tag,
            "n_errors": n_err,
            "error_rate": round(float(n_err / max(1, n_tag)), 4),
        }

    length_stats: dict[str, dict[str, Any]] = {}
    for bucket, grp in merged.groupby("length_bucket"):
        n_b = int(len(grp))
        n_e = int(grp["is_error"].sum())
        length_stats[str(bucket)] = {
            "n_samples": n_b,
            "n_errors": n_e,
            "error_rate": round(float(n_e / max(1, n_b)), 4),
        }

    errors = merged[merged["is_error"]].copy()
    errors["error_type"] = np.where(
        errors["y_true"] == 1, "TOXIC -> CLEAN", "CLEAN -> TOXIC"
    )
    errors["confidence_margin"] = (errors["prob_TOXIC"] - 0.5).abs()
    high_conf_fn = int(((errors["y_true"] == 1) & (errors["prob_TOXIC"] <= 0.20)).sum())
    high_conf_fp = int(((errors["y_true"] == 0) & (errors["prob_TOXIC"] >= 0.80)).sum())

    rng = np.random.default_rng(seed)
    chosen_indices: list[int] = []
    for err_type in ("TOXIC -> CLEAN", "CLEAN -> TOXIC"):
        for src in ("vihsd", "victsd"):
            sub = errors[(errors["error_type"] == err_type) & (errors["source"] == src)]
            if sub.empty:
                continue
            high_c = sub.sort_values("confidence_margin", ascending=False).head(3)
            border = sub.sort_values("confidence_margin", ascending=True).head(2)
            for idx in list(high_c.index) + list(border.index):
                if idx not in chosen_indices:
                    chosen_indices.append(int(idx))

    remaining = [int(i) for i in errors.index if int(i) not in chosen_indices]
    if remaining and len(chosen_indices) < max_examples:
        rng.shuffle(remaining)
        for idx in remaining:
            if len(chosen_indices) >= max_examples:
                break
            chosen_indices.append(idx)

    examples: list[dict[str, Any]] = []
    for idx in chosen_indices[:max_examples]:
        row = errors.loc[idx]
        examples.append(
            {
                "sample_id": str(row["sample_id"]),
                "source": str(row["source"]),
                "true_label": BINARY_LABEL_NAMES[int(row["y_true"])],
                "pred_label": BINARY_LABEL_NAMES[int(row["y_pred"])],
                "error_type": str(row["error_type"]),
                "prob_toxic": round(float(row["prob_TOXIC"]), 6),
                "word_count": int(row["word_count"]),
                "char_count": int(row["char_count"]),
                "tags": list(row["tags"]),
                "redacted_snippet": redact_comment_snippet(row["text"]),
            }
        )

    tag_counts_sampled = Counter(tag for ex in examples for tag in ex["tags"])
    return {
        "variant": cfg["variant"],
        "seed": int(cfg["seed"]),
        "run_id": cfg["run_id"],
        "privacy_note": (
            "All representative examples are redacted (<USER>, <URL>, <EMAIL>, "
            "<NUM>, <NAME>) and capped at 84 chars with SHA-256 sample_id only."
        ),
        "aggregate_error_summary": agg_summary,
        "high_confidence_error_counts": {
            "fn_prob_le_020": high_conf_fn,
            "fp_prob_ge_080": high_conf_fp,
        },
        "error_rate_by_tag": tag_error_rates,
        "error_rate_by_length_bucket": length_stats,
        "n_representative_examples": len(examples),
        "sampled_example_tag_counts": dict(tag_counts_sampled),
        "representative_examples": examples,
    }


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fcfcfd",
            "axes.edgecolor": "#cbd5e1",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 9.5,
        }
    )
    return plt


def _plot_ablation_macro_f1(
    ranking: list[dict[str, Any]], out_path: Path, plt: Any
) -> Path:
    ordered = list(reversed(ranking))
    names = [item["variant"] for item in ordered]
    macro_means = [item["macro_f1"]["mean"] for item in ordered]
    macro_stds = [item["macro_f1"]["std"] for item in ordered]
    toxic_means = [item["f1_toxic"]["mean"] for item in ordered]

    fig, ax = plt.subplots(figsize=(10.5, max(5.8, 0.36 * len(ordered) + 1.4)))
    y = np.arange(len(names))
    colors = [
        "#0f766e" if "HYBRID" in n else ("#7c3aed" if "CHAR" in n else "#2563eb")
        for n in names
    ]
    bars = ax.barh(
        y, macro_means, xerr=macro_stds, height=0.64, color=colors, alpha=0.88, capsize=3
    )
    ax.scatter(toxic_means, y, color="#dc2626", zorder=4, s=36, label="F1 TOXIC (mean)")
    for bar, m_val, std_val, t_val in zip(bars, macro_means, macro_stds, toxic_means):
        ax.text(
            m_val + std_val + 0.004,
            bar.get_y() + bar.get_height() / 2.0,
            f"{m_val:.4f} (TOXIC {t_val:.4f})",
            va="center",
            fontsize=8,
            color="#0f172a",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlim(0.50, 0.87)
    ax.set_xlabel("Validation Score (3 seeds mean +/- std)")
    ax.set_title("TV4 CNN / Char-CNN / Hybrid Ablation Ranking (Validation Macro-F1 & F1-TOXIC)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_learning_curves(
    top_variants: list[dict[str, Any]], out_path: Path, plt: Any
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax_loss, ax_f1 = axes
    palette = ["#0f766e", "#0284c7", "#2563eb", "#7c3aed", "#d97706", "#dc2626"]
    for idx, var in enumerate(top_variants[:5]):
        color = palette[idx % len(palette)]
        seed_dir = Path(var["best_seed_dir"])
        hist = json.loads((seed_dir / "history.json").read_text(encoding="utf-8"))
        epochs = [int(item["epoch"]) for item in hist]
        tr_loss = [float(item["train"]["loss"]) for item in hist]
        va_loss = [float(item["val"]["loss"]) for item in hist]
        tr_f1 = [float(item["train"]["macro_f1"]) for item in hist]
        va_f1 = [float(item["val"]["macro_f1"]) for item in hist]
        label = f"{var['variant']} (s{var['best_seed']})"
        ax_loss.plot(epochs, va_loss, color=color, linewidth=2.0, label=f"{label} val")
        ax_loss.plot(epochs, tr_loss, color=color, linestyle="--", alpha=0.45, linewidth=1.2)
        ax_f1.plot(epochs, va_f1, color=color, linewidth=2.0, label=f"{label} val")
        ax_f1.plot(epochs, tr_f1, color=color, linestyle="--", alpha=0.45, linewidth=1.2)

    ax_loss.set_title("Loss Curves (Solid=Val, Dashed=Train)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=7.5)
    ax_f1.set_title("Macro-F1 Curves (Solid=Val, Dashed=Train)")
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("Macro-F1")
    ax_f1.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("TV4 Top Variants Training vs Validation Curves (Best Validation Seed)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_confusion_matrices(
    selected_variants: list[dict[str, Any]], out_path: Path, plt: Any
) -> Path:
    items = selected_variants[:4]
    fig, axes = plt.subplots(1, len(items), figsize=(3.6 * len(items), 3.6))
    if len(items) == 1:
        axes = [axes]
    for ax, var in zip(axes, items):
        seed_dir = Path(var["best_seed_dir"])
        pred_df = pd.read_csv(seed_dir / "predictions_val.csv")
        cm = confusion_matrix(pred_df["y_true"], pred_df["y_pred"], labels=[0, 1])
        cm_pct = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cm_pct, cmap="Blues", vmin=0.0, vmax=1.0)
        for r in range(2):
            for c in range(2):
                txt_color = "white" if cm_pct[r, c] > 0.55 else "#0f172a"
                ax.text(
                    c,
                    r,
                    f"{cm[r, c]}\n({cm_pct[r, c] * 100:.1f}%)",
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=8.5,
                    fontweight="bold",
                )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["CLEAN", "TOXIC"], fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["CLEAN", "TOXIC"], fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(
            f"{var['variant']} (s{var['best_seed']})\nMacro-F1={var['best_seed_macro_f1']:.4f}",
            fontsize=8.8,
        )
    fig.suptitle("Validation Confusion Matrices (n=6,048, Uncalibrated Argmax)")
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_calibration_reliability(
    top_cal: dict[str, Any], out_path: Path, plt: Any
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    ax_rel, ax_thr = axes
    ax_rel.plot([0, 1], [0, 1], "--", color="#64748b", label="Perfect Calibration")

    series_specs = [
        ("Raw Softmax", top_cal["raw"], "#dc2626", "o"),
        (
            f"Temperature (T={top_cal['temperature_scaling']['params']['temperature']:.2f})",
            top_cal["temperature_scaling"]["metrics"],
            "#2563eb",
            "s",
        ),
        ("Platt Scaling", top_cal["platt_scaling"]["metrics"], "#0f766e", "^"),
    ]
    for label, m_dict, color, marker in series_specs:
        bins = [b for b in m_dict["reliability_bins"] if b["count"] > 0]
        xs = [b["mean_conf"] for b in bins]
        ys = [b["emp_pos_rate"] for b in bins]
        ax_rel.plot(
            xs, ys, marker=marker, color=color, linewidth=1.8,
            label=f"{label} (ECE={m_dict['ece']:.4f})",
        )
    ax_rel.set_xlim(0.0, 1.0)
    ax_rel.set_ylim(0.0, 1.0)
    ax_rel.set_xlabel("Mean Predicted P(TOXIC)")
    ax_rel.set_ylabel("Empirical TOXIC Fraction")
    ax_rel.set_title(
        f"Reliability Diagram — {top_cal['variant']} s{top_cal['seed']} (Val In-Sample)"
    )
    ax_rel.legend(fontsize=8, loc="upper left")

    sw = top_cal["threshold_sweep_platt"]
    ax_thr.plot(sw["grid"], sw["macro_f1_scores"], color="#0f766e", lw=2.0, label="Macro-F1")
    ax_thr.plot(sw["grid"], sw["f1_toxic_scores"], color="#dc2626", lw=2.0, label="F1 TOXIC")
    ax_thr.axvline(0.50, color="#64748b", linestyle=":", label="Default 0.50")
    ax_thr.axvline(
        sw["best_macro_f1_threshold"],
        color="#2563eb",
        linestyle="--",
        label=f"Best Macro-F1 @{sw['best_macro_f1_threshold']:.2f} ({sw['best_macro_f1']:.4f})",
    )
    ax_thr.set_xlabel("Threshold on Platt P(TOXIC)")
    ax_thr.set_ylabel("Validation Score")
    ax_thr.set_title("Threshold Sweep (Diagnostic Val In-Sample Only)")
    ax_thr.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_source_comparison(
    variants: list[dict[str, Any]], out_path: Path, plt: Any
) -> Path:
    names = [v["variant"] for v in variants]
    vihsd_m = [v["by_source"]["vihsd"]["macro_f1"]["mean"] for v in variants]
    victsd_m = [v["by_source"]["victsd"]["macro_f1"]["mean"] for v in variants]
    victsd_tox = [v["by_source"]["victsd"]["f1_toxic"]["mean"] for v in variants]

    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(9.5, 0.9 * len(names) + 2.0), 4.8))
    ax.bar(x - w, vihsd_m, width=w, color="#2563eb", label="ViHSD Macro-F1")
    ax.bar(x, victsd_m, width=w, color="#0f766e", label="ViCTSD Macro-F1")
    ax.bar(x + w, victsd_tox, width=w, color="#f59e0b", label="ViCTSD F1-TOXIC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha="right", fontsize=8)
    ax.set_ylim(0.20, 0.94)
    ax.set_ylabel("Validation Score (3-seed mean)")
    ax.set_title("Domain Generalization: ViHSD (n=4,559) vs ViCTSD (n=1,489)")
    ax.legend(ncol=3, loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def generate_tv4_plots(
    discovery: dict[str, Any],
    calibrations: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, str]:
    """Render Phase 4 comparison, learning curve, confusion, calibration, and source charts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = _configure_matplotlib()
    ranking = discovery["ranking"]
    by_name = discovery["variants_by_name"]

    top_diverse: list[dict[str, Any]] = []
    seen_arch: set[str] = set()
    for item in ranking:
        if item["architecture"] not in seen_arch:
            top_diverse.append(item)
            seen_arch.add(item["architecture"])
    for item in ranking:
        if item not in top_diverse and len(top_diverse) < 5:
            top_diverse.append(item)

    source_focus = ranking[:6]
    for extra in ("CNN-AUG-N4", "CNN-IMB-FOCAL", "CNN-IMB-SAMPLER", "CHAR-02"):
        if extra in by_name and by_name[extra] not in source_focus:
            source_focus.append(by_name[extra])

    paths = {
        "ablation_macro_f1": str(
            _plot_ablation_macro_f1(ranking, out_dir / "ablation_macro_f1.png", plt)
        ),
        "learning_curves": str(
            _plot_learning_curves(top_diverse, out_dir / "learning_curves.png", plt)
        ),
        "confusion_matrices": str(
            _plot_confusion_matrices(top_diverse, out_dir / "confusion_matrices.png", plt)
        ),
        "calibration_reliability": str(
            _plot_calibration_reliability(
                calibrations[0], out_dir / "calibration_reliability.png", plt
            )
        ),
        "source_comparison": str(
            _plot_source_comparison(source_focus, out_dir / "source_comparison.png", plt)
        ),
    }
    return paths


def generate_tv4_phase4_report(
    models_dir: Path = DEFAULT_CNN_MODELS_DIR,
    out_dir: Path = DEFAULT_REPORT_OUT_DIR,
    cnn_config_path: Path = DEFAULT_CNN_CONFIG_PATH,
    root: Path = ROOT,
    verify_reload: bool = False,
    device: str = "cpu",
    max_error_examples: int = 40,
) -> dict[str, Any]:
    """Run complete TV4 Phase 4 calibration, error analysis, charts, and JSON outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    discovery = discover_tv4_variants(
        models_dir=models_dir,
        cnn_config_path=cnn_config_path,
        root=root,
        check_mlflow=True,
        verify_reload=verify_reload,
        device=device,
    )
    ranking = discovery["ranking"]
    if not ranking:
        raise RuntimeError(f"No completed TV4 variants found under {models_dir}")

    calibrations = [
        evaluate_run_calibration(Path(item["best_seed_dir"])) for item in ranking
    ]

    val_df = pd.read_parquet(root / "data" / "processed" / "binary_n2_v1" / "val.parquet")
    best_overall = ranking[0]
    by_name = discovery["variants_by_name"]
    error_analysis_primary = build_privacy_safe_error_analysis(
        Path(best_overall["best_seed_dir"]), val_df, max_examples=max_error_examples, seed=42
    )
    secondary_errors: dict[str, Any] = {}
    for ref_name in ("CNN-01", "CNN-01-WD", "CHAR-02"):
        if ref_name in by_name and ref_name != best_overall["variant"]:
            secondary_errors[ref_name] = build_privacy_safe_error_analysis(
                Path(by_name[ref_name]["best_seed_dir"]), val_df, max_examples=15, seed=43
            )

    plots = generate_tv4_plots(discovery, calibrations, out_dir)
    compact_ranking = []
    for item in ranking:
        compact_ranking.append(
            {
                key: item[key]
                for key in (
                    "variant",
                    "architecture",
                    "preprocessing",
                    "use_augmentation",
                    "imbalance_method",
                    "weight_decay",
                    "description",
                    "hypothesis",
                    "macro_f1",
                    "f1_toxic",
                    "weighted_f1",
                    "balanced_accuracy",
                    "pr_auc",
                    "raw_softmax_ece",
                    "train_time_s",
                    "inference_ms_per_sample",
                    "peak_vram_mb",
                    "model_size_mb",
                    "num_params",
                    "by_source",
                    "best_seed",
                    "best_seed_macro_f1",
                    "best_seed_f1_toxic",
                    "best_seed_dir",
                )
            }
        )

    report_payload = {
        "n_variants_found": discovery["n_variants_found"],
        "n_runs_total": discovery["n_runs_total"],
        "all_integrity_checks_passed": discovery["all_integrity_checks_passed"],
        "best_variant": best_overall["variant"],
        "best_seed_dir": best_overall["best_seed_dir"],
        "ranking": compact_ranking,
        "calibrations": calibrations,
        "plots": plots,
        "integrity_checks": [
            chk for item in ranking for chk in item["seed_checks"]
        ],
    }
    error_payload = {
        "primary": error_analysis_primary,
        "secondary_references": secondary_errors,
    }
    (out_dir / "tv4_phase4_report.json").write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "tv4_error_analysis_val.json").write_text(
        json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "report": report_payload,
        "error_analysis": error_payload,
        "out_dir": str(out_dir),
    }


DEFAULT_NOTEBOOK_PATH = ROOT / "notebooks" / "EXP-04-cnn-analysis.ipynb"


def _build_exp04_notebook_cells() -> list[dict[str, Any]]:
    """Build markdown and executable code cell templates for EXP-04-cnn-analysis.ipynb."""
    return [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# EXP-04 — Phân tích Thực nghiệm CNN, Char-CNN & Hybrid RNN+Char-CNN (TV4)\n",
                "\n",
                "- **Phạm vi:** Binary classification (`CLEAN=0`, `TOXIC=1`) theo SPEC v5.0.0.\n",
                "- **Dữ liệu:** `data/processed/binary_n2_v1/` (`train=28,231`, `val=6,048`) và ",
                "`binary_n4_training_v1/` (`train=33,906`, `val=6,048`).\n",
                "- **Chống rò rỉ:** Tuyệt đối không đọc/chạm `test.parquet` trước N10.\n",
                "- **Nguồn sự thật logic:** `src/models/cnn.py`, `src/models/cnn_pipeline.py`, ",
                "và `src/evaluation/cnn_report.py`.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "from pathlib import Path\n",
                "import sys\n",
                "import pandas as pd\n",
                "\n",
                "ROOT = Path.cwd().resolve()\n",
                "if not (ROOT / 'src').exists() and (ROOT.parent / 'src').exists():\n",
                "    ROOT = ROOT.parent\n",
                "if str(ROOT) not in sys.path:\n",
                "    sys.path.insert(0, str(ROOT))\n",
                "\n",
                "REPORT_DIR = (\n",
                "    Path(OVERRIDE_REPORT_DIR)\n",
                "    if 'OVERRIDE_REPORT_DIR' in globals()\n",
                "    else ROOT / 'artifacts' / 'TV4-cnn-analysis'\n",
                ")\n",
                "report = json.loads(\n",
                "    (REPORT_DIR / 'tv4_phase4_report.json').read_text(encoding='utf-8')\n",
                ")\n",
                "err_report = json.loads(\n",
                "    (REPORT_DIR / 'tv4_error_analysis_val.json').read_text(encoding='utf-8')\n",
                ")\n",
                "print(\n",
                "    f\"Variants={report['n_variants_found']} | \"\n",
                "    f\"Runs={report['n_runs_total']} | \"\n",
                "    f\"Integrity={report['all_integrity_checks_passed']} | \"\n",
                "    f\"Best={report['best_variant']}\"\n",
                ")\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Bảng xếp hạng tổng hợp 18 biến thể × 3 seeds trên Validation\n",
                "\n",
                "Xếp hạng chính thức dựa trên **Uncalibrated Validation Macro-F1** (`argmax` / ",
                "ngưỡng `0.50`). Mọi mô hình đều đạt peak VRAM `< 250 MB` (`<< 7 GB`).\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "rows = []\n",
                "for rank_idx, item in enumerate(report['ranking'], start=1):\n",
                "    m_mean = item['macro_f1']['mean']\n",
                "    m_std = item['macro_f1']['std']\n",
                "    t_mean = item['f1_toxic']['mean']\n",
                "    t_std = item['f1_toxic']['std']\n",
                "    b_f1 = item['best_seed_macro_f1']\n",
                "    rows.append({\n",
                "        'rank': rank_idx,\n",
                "        'variant': item['variant'],\n",
                "        'arch': item['architecture'],\n",
                "        'prep': item['preprocessing'],\n",
                "        'macro_f1': f'{m_mean:.4f}±{m_std:.4f}',\n",
                "        'f1_toxic': f'{t_mean:.4f}±{t_std:.4f}',\n",
                "        'pr_auc': round(item['pr_auc']['mean'], 4),\n",
                "        'raw_ece': round(item['raw_softmax_ece']['mean'], 4),\n",
                "        'victsd_f1': round(item['by_source']['victsd']['macro_f1']['mean'], 4),\n",
                "        'vihsd_f1': round(item['by_source']['vihsd']['macro_f1']['mean'], 4),\n",
                "        'params': item['num_params'],\n",
                "        'vram_mb': round(item['peak_vram_mb'], 1),\n",
                "        'best_seed': f\"s{item['best_seed']} ({b_f1:.4f})\",\n",
                "    })\n",
                "ranking_df = pd.DataFrame(rows)\n",
                "print(ranking_df.to_string(index=False))\n",
                "_cell_charts = ['ablation_macro_f1.png', 'source_comparison.png']\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Phân tích các vòng cải tiến theo Hướng 2 (Ablation Studies)\n",
                "\n",
                "So sánh có đối chứng từng nhóm giả thuyết (chỉ thay đổi 1 yếu tố mỗi bước): ",
                "Kernel sizes, Filters, Regularization (`dropout`/`weight_decay`), Imbalance D3, ",
                "Augmentation N4, Char-CNN và Hybrid Word/BiLSTM/BiGRU + Char-CNN.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "by_var = {x['variant']: x for x in report['ranking']}\n",
                "groups = [\n",
                "    ('G1-Kernel', ['CNN-01', 'CNN-02']),\n",
                "    ('G2-Filters', ['CNN-02', 'CNN-03a', 'CNN-03b']),\n",
                "    ('G3-Reg', ['CNN-02', 'CNN-04a', 'CNN-04b', 'CNN-01', 'CNN-01-WD']),\n",
                "    (\n",
                "        'G4-Imbalance-D3',\n",
                "        ['CNN-02', 'CNN-IMB-CW', 'CNN-IMB-FOCAL', 'CNN-IMB-SAMPLER'],\n",
                "    ),\n",
                "    ('G5-Augment-N4', ['CNN-02', 'CNN-AUG-N4']),\n",
                "    ('G6-CharCNN-N7', ['CHAR-01', 'CHAR-02', 'CHAR-03']),\n",
                "    (\n",
                "        'G7-Hybrid-N7',\n",
                "        ['HYBRID-01', 'HYBRID-BILSTM-01', 'HYBRID-RNN-01', 'HYBRID-RNN-WD'],\n",
                "    ),\n",
                "]\n",
                "for g_name, v_list in groups:\n",
                "    present = [by_var[v] for v in v_list if v in by_var]\n",
                "    if not present:\n",
                "        continue\n",
                "    print(f'=== {g_name} ===')\n",
                "    for x in present:\n",
                "        print(\n",
                "            f\"  {x['variant']:17s} | Macro-F1={x['macro_f1']['mean']:.4f} \"\n",
                "            f\"| F1-TOXIC={x['f1_toxic']['mean']:.4f} \"\n",
                "            f\"| ViCTSD={x['by_source']['victsd']['macro_f1']['mean']:.4f} \"\n",
                "            f\"| ViHSD={x['by_source']['vihsd']['macro_f1']['mean']:.4f}\"\n",
                "        )\n",
                "_cell_charts = ['learning_curves.png', 'confusion_matrices.png']\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Chẩn đoán Calibration (Temperature vs Platt) & Threshold Sweep\n",
                "\n",
                "Tất cả phép hiệu chuẩn xác suất và quét ngưỡng đều gắn cờ ",
                "`in_sample_validation_only = True` và **không** dùng để đảo ",
                "thứ tự chọn mô hình.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cal_rows = []\n",
                "for c in report['calibrations']:\n",
                "    sw = c['threshold_sweep_platt']\n",
                "    cal_rows.append({\n",
                "        'variant': c['variant'],\n",
                "        'seed': c['seed'],\n",
                "        'uncal_macro_f1': round(c['uncalibrated_val_macro_f1'], 4),\n",
                "        'raw_ece': round(c['raw']['ece'], 4),\n",
                "        'temp_T': round(c['temperature_scaling']['params']['temperature'], 3),\n",
                "        'temp_ece': round(c['temperature_scaling']['metrics']['ece'], 4),\n",
                "        'platt_ece': round(c['platt_scaling']['metrics']['ece'], 4),\n",
                "        'platt_opt_thr': sw['best_macro_f1_threshold'],\n",
                "        'in_sample_macro_f1': round(sw['best_macro_f1'], 4),\n",
                "        'in_sample_f1_toxic': round(sw['f1_toxic_at_macro_opt'], 4),\n",
                "    })\n",
                "cal_df = pd.DataFrame(cal_rows)\n",
                "print(cal_df.to_string(index=False))\n",
                "_cell_charts = ['calibration_reliability.png']\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Phân tích lỗi Validation ẩn danh (`>= 30` mẫu đã che PII)\n",
                "\n",
                "Toàn bộ mẫu lỗi đại diện đều được che `<USER>`, `<URL>`, `<EMAIL>`, `<NUM>`, ",
                "`<NAME>` và giới hạn tối đa 84 ký tự theo quy tắc bảo vệ quyền riêng tư.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "prim = err_report['primary']\n",
                "agg = prim['aggregate_error_summary']\n",
                "n_err = agg['n_errors_total']\n",
                "n_tot = agg['n_val_total']\n",
                "print(\n",
                "    f\"Primary={prim['variant']} (s{prim['seed']}) | \"\n",
                "    f\"Errors={n_err}/{n_tot} ({agg['error_rate']:.2%}) | \"\n",
                "    f\"Pairs={agg['confusion_pair_counts']}\"\n",
                ")\n",
                "tag_df = pd.DataFrame.from_dict(prim['error_rate_by_tag'], orient='index')\n",
                "print('\\n--- Error Rate by Linguistic Tag ---')\n",
                "print(tag_df.to_string())\n",
                "len_buckets = prim['error_rate_by_length_bucket']\n",
                "len_df = pd.DataFrame.from_dict(len_buckets, orient='index')\n",
                "print('\\n--- Error Rate by Length Bucket ---')\n",
                "print(len_df.to_string())\n",
                "ex_rows = [\n",
                "    {\n",
                "        'id12': ex['sample_id'][:12],\n",
                "        'src': ex['source'],\n",
                "        'type': ex['error_type'],\n",
                "        'p_tox': round(ex['prob_toxic'], 4),\n",
                "        'words': ex['word_count'],\n",
                "        'tags': ','.join(ex['tags']),\n",
                "        'redacted_snippet': ex['redacted_snippet'],\n",
                "    }\n",
                "    for ex in prim['representative_examples']\n",
                "]\n",
                "print(f'\\n--- {len(ex_rows)} Privacy-Redacted Validation Error Examples ---')\n",
                "print(pd.DataFrame(ex_rows).to_string(index=False))\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Kết luận chọn ứng viên cho Phần 5 (Ensemble) & Phần 6 (Locked Test N10)\n",
                "\n",
                "1. **Ứng viên #1 (Hybrid RNN+Char-CNN):** `HYBRID-RNN-01` ",
                "(Macro-F1 `0.7733 ± 0.0072`, best seed 42 `0.7806`, Platt ECE `0.0078`) và ",
                "`HYBRID-RNN-WD` (`0.7731 ± 0.0052`, F1-TOXIC `0.6155`).\n",
                "2. **Ứng viên #2 (Pure Word TextCNN nhanh/nhẹ):** `CNN-01-WD` ",
                "(`0.7705 ± 0.0027`, Platt ECE `0.0064`, inference `0.050 ms/sample`) và ",
                "`CNN-01` (`0.7702 ± 0.0041`).\n",
                "3. **Ứng viên #3 (Pure Char-CNN hình thái ký tự):** `CHAR-02` ",
                "(`0.7550 ± 0.0040`, `345,666` params, `3.98 MB`) bổ sung đa dạng kiến trúc.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Phần 5 (EXP-06) — Error Diversity & Conditional Ensemble (TV4 + TV2)\n",
                "\n",
                "Kiểm định độ đa dạng lỗi (`disagreement_rate`, `double_fault_rate`, ",
                "`oracle_macro_f1`) và đánh giá Soft/Weighted Probability Voting cùng ",
                "5-Fold Out-of-Fold Platt Calibrated Voting đối chiếu ngưỡng ADR `+1.0%`.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ens_path = REPORT_DIR.parent / 'TV4-ensemble-analysis' / (\n",
                "    'tv4_phase5_ensemble_report.json'\n",
                ")\n",
                "if ens_path.exists():\n",
                "    ens_rep = json.loads(ens_path.read_text(encoding='utf-8'))\n",
                "    adr = ens_rep['adr_decision']\n",
                "    print('ADR Gate Summary:', json.dumps(adr, ensure_ascii=False))\n",
                "    e_rows = [\n",
                "        {\n",
                "            'ensemble': e['name'],\n",
                "            'scope': e['scope'],\n",
                "            'protocol': e['protocol'],\n",
                "            'macro_f1': (\n",
                "                f\"{e['macro_f1']['mean']:.4f}±{e['macro_f1']['std']:.4f}\"\n",
                "            ),\n",
                "            'f1_toxic': (\n",
                "                f\"{e['f1_toxic']['mean']:.4f}±{e['f1_toxic']['std']:.4f}\"\n",
                "            ),\n",
                "            'victsd_f1': round(\n",
                "                e['by_source']['victsd']['macro_f1']['mean'], 4\n",
                "            ),\n",
                "            'vihsd_f1': round(\n",
                "                e['by_source']['vihsd']['macro_f1']['mean'], 4\n",
                "            ),\n",
                "            'gain_vs_tv4': e['gain_vs_best_tv4_single'],\n",
                "            'gain_vs_n6_cal': e['gain_vs_n6_cal_050'],\n",
                "            'gain_vs_n6_uncal': e['gain_vs_n6_uncal_margin'],\n",
                "        }\n",
                "        for e in ens_rep['ensembles']\n",
                "    ]\n",
                "    print(pd.DataFrame(e_rows).to_string(index=False))\n",
                "    _cell_charts = [\n",
                "        '../TV4-ensemble-analysis/error_diversity_heatmap.png',\n",
                "        '../TV4-ensemble-analysis/ensemble_comparison.png',\n",
                "        '../TV4-ensemble-analysis/ensemble_source_gains.png',\n",
                "    ]\n",
                "else:\n",
                "    print('Phase 5 ensemble report not generated in this directory.')\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. One-Shot Locked Test Evaluation (`N10` / `EXP-08`)\n",
                "\n",
                "Đánh giá một lần duy nhất (`one-shot`) danh sách mô hình đơn và ensemble "
                "đã đóng băng từ validation trên `data/processed/binary_n2_v1/test.parquet` "
                "(`n = 6,048`, SHA-256 "
                "`d200be4a7ff2f2e725ba8e6c2c0ab835afd73612b9a5b2f14b528cb5c577be4e`). "
                "Không có bất kỳ bước chọn mô hình, tinh chỉnh trọng số hay tối ưu ngưỡng "
                "nào được thực hiện trên tập test.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "locked_report_path = (\n",
                "    REPORT_DIR.parent\n",
                "    / 'TV4-locked-test'\n",
                "    / 'tv4_n10_locked_test_report.json'\n",
                ")\n",
                "if locked_report_path.exists():\n",
                "    locked_rep = json.loads(locked_report_path.read_text(encoding='utf-8'))\n",
                "    locked_rows = []\n",
                "    for m in locked_rep['single_models']:\n",
                "        at050 = m['at_050']\n",
                "        platt = m['at_frozen_platt']\n",
                "        gap = m['generalization_gap_macro_f1']\n",
                "        locked_rows.append({\n",
                "            'group': 'Single (TV4)',\n",
                "            'candidate': m['variant'],\n",
                "            'val_macro_f1': round(m['val_macro_f1_mean'], 4),\n",
                "            'test_macro_f1_050': round(at050['test_macro_f1']['mean'], 4),\n",
                "            'test_f1_toxic_050': round(at050['test_f1_toxic']['mean'], 4),\n",
                "            'test_macro_f1_platt': round(platt['test_macro_f1']['mean'], 4),\n",
                "            'test_ece_platt': round(platt['test_ece']['mean'], 4),\n",
                "            'delta_test_minus_val': round(gap, 4),\n",
                "        })\n",
                "    tv2_ref = locked_rep['n6_reference']\n",
                "    tv2_050 = tv2_ref['at_050']\n",
                "    tv2_034 = tv2_ref['at_frozen_034']\n",
                "    locked_rows.append({\n",
                "        'group': 'Single (TV2 Ref)',\n",
                "        'candidate': 'EXP-N6-C-calibrated (@0.50 / @0.34)',\n",
                "        'val_macro_f1': 0.7836,\n",
                "        'test_macro_f1_050': round(tv2_050['macro_f1'], 4),\n",
                "        'test_f1_toxic_050': round(tv2_050['f1_toxic'], 4),\n",
                "        'test_macro_f1_platt': round(tv2_034['macro_f1'], 4),\n",
                "        'test_ece_platt': round(tv2_050['ece'], 4),\n",
                "        'delta_test_minus_val': round(tv2_050['macro_f1'] - 0.7836, 4),\n",
                "    })\n",
                "    for e in locked_rep['ensembles']:\n",
                "        locked_rows.append({\n",
                "            'group': f\"Ensemble ({e['scope']})\",\n",
                "            'candidate': e['name'],\n",
                "            'val_macro_f1': round(e['val_macro_f1_mean'], 4),\n",
                "            'test_macro_f1_050': round(e['test_macro_f1']['mean'], 4),\n",
                "            'test_f1_toxic_050': round(e['test_f1_toxic']['mean'], 4),\n",
                "            'test_macro_f1_platt': round(e['test_macro_f1']['mean'], 4),\n",
                "            'test_ece_platt': round(e['test_ece']['mean'], 4),\n",
                "            'delta_test_minus_val': round(e['generalization_gap_macro_f1'], 4),\n",
                "        })\n",
                "    print(pd.DataFrame(locked_rows).to_string(index=False))\n",
                "    _locked_charts = [\n",
                "        '../TV4-locked-test/locked_test_val_vs_test.png',\n",
                "        '../TV4-locked-test/locked_test_confusion_matrices.png',\n",
                "        '../TV4-locked-test/locked_test_source_breakdown.png',\n",
                "    ]\n",
                "else:\n",
                "    print('Phase 6 locked test report not generated in this directory.')\n",
            ],
        },
    ]


def _execute_notebook_cells(
    cells: list[dict[str, Any]], report_dir: Path
) -> list[dict[str, Any]]:
    executed: list[dict[str, Any]] = []
    ns: dict[str, Any] = {"OVERRIDE_REPORT_DIR": str(report_dir.resolve())}
    exec_idx = 0
    for cell in cells:
        cloned = json.loads(json.dumps(cell))
        if cloned["cell_type"] != "code":
            executed.append(cloned)
            continue
        exec_idx += 1
        code_str = "".join(cloned["source"]).replace("_ cell_charts", "_cell_charts")
        cloned["source"] = [
            line.replace("_ cell_charts", "_cell_charts") for line in cloned["source"]
        ]
        ns["_cell_charts"] = []
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code_str, ns)  # noqa: S102
        stdout_text = buf.getvalue()
        outputs: list[dict[str, Any]] = []
        if stdout_text:
            outputs.append(
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": stdout_text.splitlines(keepends=True),
                }
            )
        for chart_name in ns.get("_cell_charts", []):
            chart_path = report_dir / str(chart_name)
            if chart_path.exists():
                b64_png = base64.b64encode(chart_path.read_bytes()).decode("ascii")
                outputs.append(
                    {
                        "data": {
                            "image/png": b64_png,
                            "text/plain": [f"<Figure {chart_name}>"],
                        },
                        "metadata": {},
                        "output_type": "display_data",
                    }
                )
        cloned["execution_count"] = exec_idx
        cloned["outputs"] = outputs
        executed.append(cloned)
    return executed


def export_executed_notebook(
    report_dir: Path = DEFAULT_REPORT_OUT_DIR,
    notebook_path: Path = DEFAULT_NOTEBOOK_PATH,
) -> Path:
    """Build, execute, verify Restart-and-Run-All equivalence, and save EXP-04 notebook."""
    cells_pass1 = _execute_notebook_cells(_build_exp04_notebook_cells(), report_dir)
    cells_pass2 = _execute_notebook_cells(_build_exp04_notebook_cells(), report_dir)
    if cells_pass1 != cells_pass2:
        raise RuntimeError("Notebook execution is not deterministic across Restart & Run All")

    notebook_payload = {
        "cells": cells_pass1,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(
        json.dumps(notebook_payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return notebook_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TV4 CNN Phase 4 analysis & calibration.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_CNN_MODELS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_OUT_DIR)
    parser.add_argument("--notebook-path", type=Path, default=DEFAULT_NOTEBOOK_PATH)
    parser.add_argument("--verify-reload", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-error-examples", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    res = generate_tv4_phase4_report(
        models_dir=args.models_dir,
        out_dir=args.out_dir,
        verify_reload=bool(args.verify_reload),
        device=args.device,
        max_error_examples=int(args.max_error_examples),
    )
    nb_path = export_executed_notebook(
        report_dir=args.out_dir, notebook_path=args.notebook_path
    )
    rep = res["report"]
    print(
        f"TV4 Phase 4 OK | variants={rep['n_variants_found']} | runs={rep['n_runs_total']} | "
        f"integrity={rep['all_integrity_checks_passed']} | best={rep['best_variant']} | "
        f"notebook={nb_path}"
    )


if __name__ == "__main__":
    main()
