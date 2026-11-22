"""
SVM Binary Pipeline — N6 TV2 (Vinh) theo SPEC v5.0.0.

Đặc tả:
  - Dataset: data/processed/binary_n2_v1/ (train.parquet, val.parquet, manifest.json).
  - Ablation tăng cường train-only: data/processed/binary_n4_v1/augmentation.parquet.
  - Nhãn nhị phân: 0 = CLEAN, 1 = TOXIC.
  - Chạy đa seed (mặc định [42, 43, 44]), báo mean ± std và metrics theo từng seed.
  - Metric chọn mô hình: Validation Macro-F1; bổ sung F1 TOXIC, PR-AUC, per-class,
    metrics tách riêng theo source ('vihsd', 'victsd'), latency và size (< 15 MB).
  - Calibration + threshold + ECE chỉ chạy sau bước chọn model (không dùng để xếp hạng).

Chạy:
  python -m src.models.svm_binary_pipeline --variant A|B|C|D|all [--tune] [--calibrate]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from src.features.preprocessing import TEENCODE_DICT, preprocess_series
from src.features.tfidf_pipeline import build_tfidf, fit_tfidf, save_tfidf, transform_texts
from src.models.svm_pipeline import (
    build_feature_selector,
    build_svm_estimator,
    sha256_file,
    sha256_texts,
)
from src.tracking.mlflow_tracker import MLflowTracker

BINARY_LABELS: list[int] = [0, 1]
BINARY_LABEL_NAMES: dict[int, str] = {0: "CLEAN", 1: "TOXIC"}
DEFAULT_BINARY_CONFIG_PATH = ROOT / "configs" / "svm_binary_config.yaml"

_NEGATION_WORDS = {"không", "ko", "k", "chẳng", "chả", "đừng", "chưa", "đéo", "éo"}
_URL_OR_MENTION_RE = re.compile(r"https?://|www\.|@\w+")
_EMOJI_OR_PUNCT_RE = re.compile(r"[!?]{2,}|[\U0001F300-\U0001FAFF]")


def load_binary_config(config_path: Path = DEFAULT_BINARY_CONFIG_PATH) -> dict[str, Any]:
    """Đọc cấu hình SVM nhị phân từ YAML."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_binary_frame(df: pd.DataFrame, expected_split: str, path: Path) -> None:
    required = {"id", "text", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: thiếu các cột bắt buộc {sorted(missing)}")
    splits = set(df["split"].astype(str).unique())
    if splits != {expected_split}:
        raise ValueError(f"{path}: chỉ được chứa split={expected_split!r}, nhận {sorted(splits)}")
    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError(f"{path}: ID rỗng hoặc trùng lặp")
    if df["text"].isna().any() or (df["text"].astype(str).str.strip() == "").any():
        raise ValueError(f"{path}: tồn tại văn bản rỗng")
    labels = set(df["label"].tolist())
    if not labels.issubset({0, 1}) or any(isinstance(x, bool) for x in df["label"].tolist()):
        raise ValueError(f"{path}: nhãn phải thuộc {{0, 1}}, nhận {sorted(labels)}")


def load_binary_train_val(
    dataset_dir: Path,
    augmentation_path: Path | None = None,
    use_augmentation: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load train.parquet và val.parquet (tuyệt đối không mở test.parquet)."""
    train_path = dataset_dir / "train.parquet"
    val_path = dataset_dir / "val.parquet"
    manifest_path = dataset_dir / "manifest.json"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    _validate_binary_frame(train_df, "train", train_path)
    _validate_binary_frame(val_df, "val", val_path)

    if "source" not in val_df.columns or "source" not in train_df.columns:
        raise ValueError("train.parquet và val.parquet phải có cột 'source'")

    overlap = set(train_df["text"]).intersection(set(val_df["text"]))
    if overlap:
        raise ValueError(f"Phát hiện {len(overlap)} văn bản trùng giữa train và val")

    hashes = {
        "train_parquet_sha256": sha256_file(train_path),
        "val_parquet_sha256": sha256_file(val_path),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else "",
    }

    if use_augmentation:
        if augmentation_path is None or not augmentation_path.exists():
            raise FileNotFoundError(f"Không tìm thấy augmentation file: {augmentation_path}")
        aug_df = pd.read_parquet(augmentation_path)
        _validate_binary_frame(aug_df, "train", augmentation_path)
        if set(aug_df["label"].tolist()) != {1}:
            raise ValueError(f"{augmentation_path}: augmentation D3 chỉ được phép chứa label=1 (TOXIC)")
        aug_val_overlap = set(aug_df["text"]).intersection(set(val_df["text"]))
        if aug_val_overlap:
            raise ValueError(f"Augmentation trùng {len(aug_val_overlap)} văn bản với validation")
        if "source" not in aug_df.columns:
            aug_df = aug_df.copy()
            aug_df["source"] = "augmented_train"
        train_df = pd.concat(
            [train_df[["id", "text", "label", "source", "split"]], aug_df[["id", "text", "label", "source", "split"]]],
            ignore_index=True,
        )
        hashes["augmentation_parquet_sha256"] = sha256_file(augmentation_path)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), hashes


def compute_ece(y_true: np.ndarray, y_prob_pos: np.ndarray, n_bins: int = 10) -> float:
    """Tính Expected Calibration Error (ECE) cho bài toán nhị phân."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for idx in range(n_bins):
        low, high = bin_edges[idx], bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob_pos >= low) & (y_prob_pos <= high)
        else:
            mask = (y_prob_pos >= low) & (y_prob_pos < high)
        count = int(np.sum(mask))
        if count == 0:
            continue
        acc = float(np.mean(y_true[mask] == 1))
        conf = float(np.mean(y_prob_pos[mask]))
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def _compute_subset_metrics(
    y_true: list[int],
    y_pred: list[int],
    scores_or_probs: np.ndarray | None = None,
) -> dict[str, Any]:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_toxic = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true,
        y_pred,
        labels=BINARY_LABELS,
        target_names=[BINARY_LABEL_NAMES[i] for i in BINARY_LABELS],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=BINARY_LABELS).tolist()
    per_class = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": float(report[name]["support"]),
        }
        for name in [BINARY_LABEL_NAMES[i] for i in BINARY_LABELS]
    }
    pr_auc: float | None = None
    if scores_or_probs is not None and len(set(y_true)) == 2:
        pr_auc = float(average_precision_score(y_true, scores_or_probs))

    return {
        "n_samples": len(y_true),
        "macro_f1": macro_f1,
        "f1_toxic": f1_toxic,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_acc,
        "pr_auc": pr_auc,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def compute_binary_metrics(
    y_true: list[int],
    y_pred: list[int],
    sources: list[str],
    scores_or_probs: np.ndarray | None = None,
) -> dict[str, Any]:
    """Tính metrics tổng thể và tách riêng theo source ('vihsd', 'victsd')."""
    overall = _compute_subset_metrics(y_true, y_pred, scores_or_probs)
    by_source: dict[str, Any] = {}
    for src in sorted(set(sources)):
        idxs = [i for i, s in enumerate(sources) if s == src]
        sub_y = [y_true[i] for i in idxs]
        sub_p = [y_pred[i] for i in idxs]
        sub_s = scores_or_probs[idxs] if scores_or_probs is not None else None
        by_source[src] = _compute_subset_metrics(sub_y, sub_p, sub_s)
    overall["by_source"] = by_source
    return overall


def _build_aggregate_error_summary(
    val_df: pd.DataFrame,
    val_pred: list[int],
) -> dict[str, Any]:
    """Tạo phân tích lỗi tổng hợp (không lưu id hay text gốc để bảo vệ riêng tư)."""
    pair_counts: Counter[str] = Counter()
    source_error_counts: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    total_errors = 0

    for text, true_lbl, pred_lbl, src in zip(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        val_pred,
        val_df["source"].tolist(),
    ):
        if int(true_lbl) == int(pred_lbl):
            continue
        total_errors += 1
        pair_counts[f"{BINARY_LABEL_NAMES[int(true_lbl)]} -> {BINARY_LABEL_NAMES[int(pred_lbl)]}"] += 1
        source_error_counts[str(src)] += 1

        tokens = str(text).lower().split()
        tagged = False
        if len(tokens) <= 5:
            pattern_counts["short_text_le_5_words"] += 1
            tagged = True
        if any(tok in TEENCODE_DICT for tok in tokens):
            pattern_counts["contains_teencode"] += 1
            tagged = True
        if any(tok in _NEGATION_WORDS for tok in tokens):
            pattern_counts["contains_negation"] += 1
            tagged = True
        if _URL_OR_MENTION_RE.search(str(text)):
            pattern_counts["contains_url_or_mention"] += 1
            tagged = True
        if _EMOJI_OR_PUNCT_RE.search(str(text)):
            pattern_counts["contains_emoji_or_punct"] += 1
            tagged = True
        if not tagged:
            pattern_counts["context_or_lexical_nuance"] += 1

    return {
        "n_val_total": len(val_pred),
        "n_errors_total": total_errors,
        "error_rate": round(total_errors / max(1, len(val_pred)), 4),
        "confusion_pair_counts": dict(pair_counts.most_common()),
        "errors_by_source": dict(source_error_counts.most_common()),
        "pattern_tag_counts_among_errors": dict(pattern_counts.most_common()),
    }


def _find_best_threshold(
    y_true: np.ndarray,
    prob_toxic: np.ndarray,
    t_min: float = 0.10,
    t_max: float = 0.90,
    t_step: float = 0.01,
) -> tuple[float, float, float]:
    """Tìm ngưỡng quyết định trên validation tối ưu Macro-F1 (cho inference artifact)."""
    best_t = 0.5
    best_macro = -1.0
    best_toxic_f1 = 0.0
    grid = np.arange(t_min, t_max + 1e-9, t_step)
    for thr in grid:
        preds = (prob_toxic >= float(thr)).astype(int)
        macro = float(f1_score(y_true, preds, average="macro", zero_division=0))
        if macro > best_macro:
            best_macro = macro
            best_t = float(round(float(thr), 4))
            best_toxic_f1 = float(f1_score(y_true, preds, pos_label=1, zero_division=0))
    return best_t, best_macro, best_toxic_f1


def _fit_and_eval_binary_single(
    X_train_raw: Any,
    y_train: list[int],
    X_val_raw: Any,
    y_val: list[int],
    val_sources: list[str],
    fs_cfg: dict[str, Any],
    svm_cfg: dict[str, Any],
    seed: int,
    calibrate: bool,
    cal_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Fit selector (nếu bật) + SVM nhị phân trên train, chấm điểm trên val."""
    t_fs = time.perf_counter()
    selector = None
    if fs_cfg.get("enabled", False) and fs_cfg.get("k") is not None:
        selector = build_feature_selector(
            method=fs_cfg.get("method", "chi2"),
            k=fs_cfg.get("k"),
            n_available_features=X_train_raw.shape[1],
        )
        if selector is not None:
            X_train = selector.fit_transform(X_train_raw, y_train)
            X_val = selector.transform(X_val_raw)
        else:
            X_train, X_val = X_train_raw, X_val_raw
    else:
        X_train, X_val = X_train_raw, X_val_raw
    fs_time_s = time.perf_counter() - t_fs

    class_counts = {c: y_train.count(c) for c in set(y_train)}
    min_class_count = min(class_counts.values()) if class_counts else 2

    estimator, model_name = build_svm_estimator(
        svm_cfg=svm_cfg,
        seed=seed,
        calibrate=calibrate,
        calibration_method=str(cal_cfg.get("method", "sigmoid")),
        calibration_cv=int(cal_cfg.get("cv", 5)),
        n_min_class_samples=min_class_count,
    )

    t_train = time.perf_counter()
    estimator.fit(X_train, y_train)
    train_time_s = time.perf_counter() - t_train

    t_inf = time.perf_counter()
    raw_pred = [int(x) for x in estimator.predict(X_val)]
    val_proba_ordered: np.ndarray | None = None
    score_for_prauc: np.ndarray | None = None

    if calibrate:
        raw_proba = estimator.predict_proba(X_val)
        val_proba_ordered = np.zeros((len(y_val), 2), dtype=float)
        for col_idx, cls in enumerate(list(estimator.classes_)):
            val_proba_ordered[:, int(cls)] = raw_proba[:, col_idx]
        score_for_prauc = val_proba_ordered[:, 1]
    elif hasattr(estimator, "decision_function"):
        margins = np.asarray(estimator.decision_function(X_val), dtype=float)
        if margins.ndim == 1:
            # Nếu classes_ == [0, 1], margin dương tương ứng lớp 1
            if list(estimator.classes_) == [1, 0]:
                margins = -margins
            score_for_prauc = margins
    inference_ms = ((time.perf_counter() - t_inf) / max(1, len(y_val))) * 1000.0

    metrics = compute_binary_metrics(y_val, raw_pred, val_sources, score_for_prauc)
    metrics["train_time_s"] = round(train_time_s, 4)
    metrics["feature_selection_time_s"] = round(fs_time_s, 4)
    metrics["inference_ms_per_sample"] = round(inference_ms, 4)
    metrics["n_features_raw"] = int(X_train_raw.shape[1])
    metrics["n_features_selected"] = int(X_train.shape[1])

    threshold = 0.5
    cal_diagnostics: dict[str, Any] | None = None
    final_pred = raw_pred

    if calibrate and val_proba_ordered is not None:
        y_arr = np.asarray(y_val, dtype=int)
        p_pos = val_proba_ordered[:, 1]
        ece = compute_ece(y_arr, p_pos, n_bins=int(cal_cfg.get("n_ece_bins", 10)))
        clipped = np.clip(val_proba_ordered, 1e-12, 1.0)
        clipped = clipped / clipped.sum(axis=1, keepdims=True)
        ll = float(log_loss(y_arr, clipped, labels=BINARY_LABELS))
        brier = float(brier_score_loss(y_arr, p_pos))
        best_t, tuned_macro, tuned_toxic_f1 = _find_best_threshold(
            y_arr,
            p_pos,
            t_min=float(cal_cfg.get("threshold_grid_min", 0.10)),
            t_max=float(cal_cfg.get("threshold_grid_max", 0.90)),
            t_step=float(cal_cfg.get("threshold_grid_step", 0.01)),
        )
        threshold = best_t
        final_pred = [int(p >= threshold) for p in p_pos]
        cal_diagnostics = {
            "in_sample_validation_only": True,
            "ece": round(ece, 6),
            "log_loss": round(ll, 6),
            "brier_score": round(brier, 6),
            "default_threshold_0_5_macro_f1": round(metrics["macro_f1"], 6),
            "optimal_threshold": threshold,
            "optimal_threshold_macro_f1": round(tuned_macro, 6),
            "optimal_threshold_f1_toxic": round(tuned_toxic_f1, 6),
        }
        metrics["calibration_diagnostics"] = cal_diagnostics

    return {
        "estimator": estimator,
        "selector": selector,
        "model_name": model_name,
        "val_pred": final_pred,
        "val_pred_default_0_5": raw_pred,
        "val_proba": val_proba_ordered,
        "threshold": threshold,
        "metrics": metrics,
        "effective_fs_cfg": {
            "enabled": bool(selector is not None),
            "method": fs_cfg.get("method", "chi2") if selector is not None else None,
            "k_requested": fs_cfg.get("k"),
            "k_effective": int(X_train.shape[1]) if selector is not None else None,
        },
        "effective_svm_cfg": dict(svm_cfg),
    }


def _tune_binary_grid(
    X_train_raw: Any,
    y_train: list[int],
    X_val_raw: Any,
    y_val: list[int],
    val_sources: list[str],
    base_fs_cfg: dict[str, Any],
    base_svm_cfg: dict[str, Any],
    tuning_grid: dict[str, Any],
    seed: int,
    cal_cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Quét tuning_grid theo raw Validation Macro-F1 (không calibrate trong lúc chọn)."""
    k_list = tuning_grid.get("k_candidates", [base_fs_cfg.get("k")])
    c_list = tuning_grid.get("C_candidates", [base_svm_cfg.get("C", 1.0)])
    cw_list = tuning_grid.get("class_weight_candidates", [base_svm_cfg.get("class_weight")])

    best_outcome: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for k_cand in k_list:
        fs_cand = {
            "enabled": k_cand is not None,
            "method": base_fs_cfg.get("method", "chi2"),
            "k": k_cand,
        }
        for c_cand in c_list:
            for cw_cand in cw_list:
                svm_cand = dict(base_svm_cfg)
                svm_cand["C"] = float(c_cand)
                svm_cand["class_weight"] = cw_cand

                outcome = _fit_and_eval_binary_single(
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    X_val_raw=X_val_raw,
                    y_val=y_val,
                    val_sources=val_sources,
                    fs_cfg=fs_cand,
                    svm_cfg=svm_cand,
                    seed=seed,
                    calibrate=False,
                    cal_cfg=cal_cfg,
                )
                history.append(
                    {
                        "k_requested": k_cand,
                        "k_effective": outcome["metrics"]["n_features_selected"],
                        "C": float(c_cand),
                        "class_weight": cw_cand,
                        "val_macro_f1": outcome["metrics"]["macro_f1"],
                        "val_f1_toxic": outcome["metrics"]["f1_toxic"],
                        "val_pr_auc": outcome["metrics"]["pr_auc"],
                        "train_time_s": outcome["metrics"]["train_time_s"],
                    }
                )
                if (
                    best_outcome is None
                    or outcome["metrics"]["macro_f1"] > best_outcome["metrics"]["macro_f1"]
                ):
                    best_outcome = outcome

    assert best_outcome is not None
    return best_outcome, history


def load_binary_svm_artifacts(seed_dir: Path) -> dict[str, Any]:
    """Load lại artifacts đã lưu của một seed nhị phân."""
    model = joblib.load(seed_dir / "svm_model.joblib")
    tfidf_data = joblib.load(seed_dir / "tfidf.joblib")
    selector_path = seed_dir / "feature_selector.joblib"
    selector = joblib.load(selector_path) if selector_path.exists() else None
    config = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
    return {
        "model": model,
        "tfidf": tfidf_data["transformer"],
        "selector": selector,
        "config": config,
        "metrics": metrics,
    }


def predict_binary_from_artifacts(artifacts: dict[str, Any], texts: list[str]) -> list[int]:
    """Suy luận nhị phân từ artifacts đã lưu (áp dụng threshold nếu calibrated)."""
    processed = preprocess_series(texts, variant=artifacts["config"]["preprocessing"])
    X = transform_texts(artifacts["tfidf"], processed)
    if artifacts["selector"] is not None:
        X = artifacts["selector"].transform(X)
    if artifacts["config"].get("calibrated", False):
        proba = artifacts["model"].predict_proba(X)
        classes = list(artifacts["model"].classes_)
        pos_col = classes.index(1)
        thr = float(artifacts["config"].get("threshold", 0.5))
        return [int(p >= thr) for p in proba[:, pos_col]]
    return [int(x) for x in artifacts["model"].predict(X)]


def run_binary_seed(
    variant: str,
    seed: int,
    full_cfg: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    data_hashes: dict[str, str],
    variant_root_dir: Path,
    tracking_uri: str | None = None,
    calibrate: bool = False,
    tune: bool = False,
    result_type: str = "official",
    svm_overrides: dict[str, Any] | None = None,
    fs_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Chạy đúng 1 seed cho một variant nhị phân N6."""
    v_cfg = full_cfg["variants"][variant]
    cal_cfg = full_cfg.get("calibration", {})
    max_mb = float(full_cfg.get("max_fallback_size_mb", 15.0))

    train_texts_raw = train_df["text"].astype(str).tolist()
    train_labels = [int(x) for x in train_df["label"].tolist()]
    val_texts_raw = val_df["text"].astype(str).tolist()
    val_labels = [int(x) for x in val_df["label"].tolist()]
    val_sources = val_df["source"].astype(str).tolist()

    t_pre = time.perf_counter()
    train_texts = preprocess_series(train_texts_raw, variant=v_cfg["preprocessing"])
    val_texts = preprocess_series(val_texts_raw, variant=v_cfg["preprocessing"])
    preprocess_time_s = time.perf_counter() - t_pre

    tfidf_cfg = v_cfg["tfidf"]
    tfidf = build_tfidf(
        mode=tfidf_cfg["mode"],
        word_ngram_range=tuple(tfidf_cfg["word_ngram_range"]),
        char_ngram_range=tuple(tfidf_cfg["char_ngram_range"]),
        max_features=tfidf_cfg.get("max_features"),
        sublinear_tf=bool(tfidf_cfg.get("sublinear_tf", True)),
        min_df=int(tfidf_cfg.get("min_df", 2)),
    )
    t_tfidf = time.perf_counter()
    tfidf = fit_tfidf(tfidf, train_texts)
    X_train_raw = transform_texts(tfidf, train_texts)
    X_val_raw = transform_texts(tfidf, val_texts)
    tfidf_fit_time_s = time.perf_counter() - t_tfidf

    fs_cfg = dict(v_cfg.get("feature_selection", {"enabled": False, "k": None}))
    if fs_overrides:
        fs_cfg.update({k: v for k, v in fs_overrides.items() if v is not None})
        if "k" in fs_overrides:
            fs_cfg["enabled"] = fs_overrides["k"] is not None

    svm_cfg = dict(v_cfg["svm"])
    if svm_overrides:
        svm_cfg.update(svm_overrides)

    tuning_history: list[dict[str, Any]] | None = None
    if tune and "tuning_grid" in v_cfg:
        outcome, tuning_history = _tune_binary_grid(
            X_train_raw=X_train_raw,
            y_train=train_labels,
            X_val_raw=X_val_raw,
            y_val=val_labels,
            val_sources=val_sources,
            base_fs_cfg=fs_cfg,
            base_svm_cfg=svm_cfg,
            tuning_grid=v_cfg["tuning_grid"],
            seed=seed,
            cal_cfg=cal_cfg,
        )
        # Nếu đồng thời yêu cầu calibrate sau khi chọn bằng raw Macro-F1
        if calibrate:
            outcome = _fit_and_eval_binary_single(
                X_train_raw=X_train_raw,
                y_train=train_labels,
                X_val_raw=X_val_raw,
                y_val=val_labels,
                val_sources=val_sources,
                fs_cfg=outcome["effective_fs_cfg"],
                svm_cfg=outcome["effective_svm_cfg"],
                seed=seed,
                calibrate=True,
                cal_cfg=cal_cfg,
            )
    else:
        outcome = _fit_and_eval_binary_single(
            X_train_raw=X_train_raw,
            y_train=train_labels,
            X_val_raw=X_val_raw,
            y_val=val_labels,
            val_sources=val_sources,
            fs_cfg=fs_cfg,
            svm_cfg=svm_cfg,
            seed=seed,
            calibrate=calibrate,
            cal_cfg=cal_cfg,
        )

    metrics = outcome["metrics"]
    metrics["preprocess_time_s"] = round(preprocess_time_s, 4)
    metrics["tfidf_fit_time_s"] = round(tfidf_fit_time_s, 4)
    metrics["n_train"] = len(train_labels)
    metrics["n_val"] = len(val_labels)
    metrics["train_toxic_share"] = round(sum(train_labels) / max(1, len(train_labels)), 4)

    seed_dir = variant_root_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    model_path = seed_dir / "svm_model.joblib"
    joblib.dump(outcome["estimator"], model_path)

    tfidf_path = seed_dir / "tfidf.joblib"
    save_tfidf(tfidf, tfidf_path, metadata={"mode": tfidf_cfg["mode"], "variant": f"N6-{variant}", "seed": seed})

    selector_path = seed_dir / "feature_selector.joblib"
    if outcome["selector"] is not None:
        joblib.dump(outcome["selector"], selector_path)

    total_bytes = model_path.stat().st_size + tfidf_path.stat().st_size
    if selector_path.exists():
        total_bytes += selector_path.stat().st_size
    model_size_mb = round(total_bytes / (1024 * 1024), 4)
    metrics["model_size_mb"] = model_size_mb
    metrics["within_15mb_fallback_limit"] = bool(model_size_mb < max_mb)

    if tuning_history is not None:
        (seed_dir / "tuning_results.json").write_text(
            json.dumps(tuning_history, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    error_summary = _build_aggregate_error_summary(val_df, outcome["val_pred"])
    (seed_dir / "error_summary.json").write_text(
        json.dumps(error_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    exp_name = str(full_cfg.get("experiment", "N6-SVM-Binary"))
    tracker = MLflowTracker(experiment_name=exp_name, tracking_uri=tracking_uri)
    run_name = f"N6-{variant}-seed{seed}" + ("-cal" if calibrate else "")
    run_id = tracker.start_run(
        run_name=run_name,
        tags={
            "experiment": exp_name,
            "variant": f"N6-{variant}",
            "seed": str(seed),
            "owner": str(full_cfg.get("owner", "TV2")),
            "result_type": result_type,
            "model": outcome["model_name"],
            "calibrated": str(calibrate).lower(),
            "use_augmentation": str(bool(v_cfg.get("use_augmentation", False))).lower(),
        },
    )

    try:
        predictions: list[dict[str, Any]] = []
        val_proba = outcome["val_proba"]
        for idx, (rec_id, true_lbl, src, pred_lbl) in enumerate(
            zip(val_df["id"].tolist(), val_labels, val_sources, outcome["val_pred"])
        ):
            proba_entry: dict[str, float] | None = None
            if val_proba is not None:
                proba_entry = {
                    "CLEAN": round(float(val_proba[idx, 0]), 6),
                    "TOXIC": round(float(val_proba[idx, 1]), 6),
                }
            predictions.append(
                {
                    "id": rec_id,
                    "source": src,
                    "true_label": true_lbl,
                    "predicted_label": pred_lbl,
                    "probabilities": proba_entry,
                    "is_calibrated": bool(calibrate),
                    "threshold": outcome["threshold"] if calibrate else None,
                    "run_id": run_id,
                }
            )
        (seed_dir / "val_predictions.json").write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        config_snapshot = {
            "experiment": exp_name,
            "variant": f"N6-{variant}",
            "seed": seed,
            "run_id": run_id,
            "owner": full_cfg.get("owner", "TV2"),
            "data_contract_version": full_cfg.get("data_contract_version", "binary_n2_v1"),
            "use_augmentation": bool(v_cfg.get("use_augmentation", False)),
            "preprocessing": v_cfg["preprocessing"],
            "tfidf": tfidf_cfg,
            "feature_selection": outcome["effective_fs_cfg"],
            "svm": outcome["effective_svm_cfg"],
            "calibrated": bool(calibrate),
            "calibration_method": cal_cfg.get("method", "sigmoid") if calibrate else None,
            "calibration_cv": cal_cfg.get("cv", 5) if calibrate else None,
            "threshold": outcome["threshold"] if calibrate else None,
            "data_hashes": data_hashes,
            "train_text_sha256": sha256_texts(train_texts_raw),
            "val_text_sha256": sha256_texts(val_texts_raw),
            "n_train": len(train_labels),
            "n_val": len(val_labels),
        }
        (seed_dir / "config.json").write_text(
            json.dumps(config_snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (seed_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (seed_dir / "run_id.txt").write_text(run_id, encoding="utf-8")

        tracker.log_environment()
        tracker.log_model_architecture(architecture=outcome["model_name"], model_type="svm_binary_scratch")
        tracker.log_params(
            {
                "variant": f"N6-{variant}",
                "seed": seed,
                "use_augmentation": bool(v_cfg.get("use_augmentation", False)),
                "preprocessing": v_cfg["preprocessing"],
                "tfidf_mode": tfidf_cfg["mode"],
                "fs_k": outcome["effective_fs_cfg"]["k_effective"],
                "svm_C": outcome["effective_svm_cfg"].get("C"),
                "svm_class_weight": str(outcome["effective_svm_cfg"].get("class_weight")),
                "calibrated": bool(calibrate),
                "train_parquet_sha256": data_hashes["train_parquet_sha256"][:16],
                "val_parquet_sha256": data_hashes["val_parquet_sha256"][:16],
                "n_train": len(train_labels),
                "n_val": len(val_labels),
            }
        )
        scalar_metrics: dict[str, float] = {
            "val_macro_f1": metrics["macro_f1"],
            "val_f1_toxic": metrics["f1_toxic"],
            "val_weighted_f1": metrics["weighted_f1"],
            "val_balanced_accuracy": metrics["balanced_accuracy"],
            "train_time_s": metrics["train_time_s"],
            "inference_ms_per_sample": metrics["inference_ms_per_sample"],
            "model_size_mb": model_size_mb,
            "n_features_selected": float(metrics["n_features_selected"]),
        }
        if metrics.get("pr_auc") is not None:
            scalar_metrics["val_pr_auc"] = float(metrics["pr_auc"])
        for src_name, src_m in metrics["by_source"].items():
            scalar_metrics[f"val_macro_f1_{src_name}"] = src_m["macro_f1"]
            scalar_metrics[f"val_f1_toxic_{src_name}"] = src_m["f1_toxic"]
        if "calibration_diagnostics" in metrics:
            scalar_metrics["val_ece"] = metrics["calibration_diagnostics"]["ece"]
            scalar_metrics["val_optimal_threshold"] = metrics["calibration_diagnostics"]["optimal_threshold"]
        tracker.log_metrics(scalar_metrics)
        tracker.log_artifact(str(seed_dir))
        tracker.end_run("FINISHED")
    except Exception:
        tracker.end_run("FAILED")
        raise

    return {
        "variant": variant,
        "seed": seed,
        "run_id": run_id,
        "seed_dir": str(seed_dir),
        "metrics": metrics,
        "effective_svm_cfg": outcome["effective_svm_cfg"],
        "effective_fs_cfg": outcome["effective_fs_cfg"],
    }


def _aggregate_seed_results(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Tính mean ± std trên các seeds cho một variant."""
    def _mean_std(vals: list[float]) -> dict[str, float]:
        arr = np.asarray(vals, dtype=float)
        return {"mean": round(float(np.mean(arr)), 4), "std": round(float(np.std(arr)), 4)}

    macro_f1s = [r["metrics"]["macro_f1"] for r in seed_runs]
    toxic_f1s = [r["metrics"]["f1_toxic"] for r in seed_runs]
    pr_aucs = [r["metrics"]["pr_auc"] for r in seed_runs if r["metrics"].get("pr_auc") is not None]
    times = [r["metrics"]["train_time_s"] for r in seed_runs]
    sizes = [r["metrics"]["model_size_mb"] for r in seed_runs]

    by_source_agg: dict[str, Any] = {}
    sources = sorted(seed_runs[0]["metrics"]["by_source"].keys()) if seed_runs else []
    for src in sources:
        by_source_agg[src] = {
            "macro_f1": _mean_std([r["metrics"]["by_source"][src]["macro_f1"] for r in seed_runs]),
            "f1_toxic": _mean_std([r["metrics"]["by_source"][src]["f1_toxic"] for r in seed_runs]),
        }

    return {
        "n_seeds": len(seed_runs),
        "seeds": [r["seed"] for r in seed_runs],
        "macro_f1": _mean_std(macro_f1s),
        "f1_toxic": _mean_std(toxic_f1s),
        "pr_auc": _mean_std(pr_aucs) if pr_aucs else None,
        "train_time_s": _mean_std(times),
        "model_size_mb": _mean_std(sizes),
        "by_source": by_source_agg,
    }


def run_binary_variant(
    variant: str,
    config_path: Path = DEFAULT_BINARY_CONFIG_PATH,
    dataset_dir: Path | None = None,
    augmentation_path: Path | None = None,
    artifacts_root: Path | None = None,
    tracking_uri: str | None = None,
    seeds: list[int] | None = None,
    calibrate: bool = False,
    tune: bool = False,
    result_type: str = "official",
    svm_overrides: dict[str, Any] | None = None,
    fs_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Chạy một variant nhị phân trên toàn bộ danh sách seeds (mặc định 3 seeds)."""
    full_cfg = load_binary_config(config_path)
    if variant not in full_cfg["variants"]:
        raise KeyError(f"Variant không hợp lệ: {variant!r}")

    v_cfg = full_cfg["variants"][variant]
    resolved_seeds = seeds or [int(s) for s in full_cfg.get("seeds", [42, 43, 44])]
    resolved_dataset_dir = dataset_dir or (ROOT / full_cfg["dataset_dir"])
    resolved_aug_path = augmentation_path or (ROOT / full_cfg["augmentation_path"])
    resolved_tracking = tracking_uri if tracking_uri is not None else full_cfg.get("tracking_uri")

    root_out = artifacts_root or (ROOT / "models" / "baselines" / f"N6-run-{uuid.uuid4().hex}")
    suffix = "-calibrated" if calibrate else ""
    variant_root_dir = root_out / f"EXP-N6-{variant}{suffix}"
    variant_root_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, data_hashes = load_binary_train_val(
        dataset_dir=resolved_dataset_dir,
        augmentation_path=resolved_aug_path,
        use_augmentation=bool(v_cfg.get("use_augmentation", False)),
    )

    seed_runs: list[dict[str, Any]] = []
    for s in resolved_seeds:
        run_res = run_binary_seed(
            variant=variant,
            seed=s,
            full_cfg=full_cfg,
            train_df=train_df,
            val_df=val_df,
            data_hashes=data_hashes,
            variant_root_dir=variant_root_dir,
            tracking_uri=resolved_tracking,
            calibrate=calibrate,
            tune=tune,
            result_type=result_type,
            svm_overrides=svm_overrides,
            fs_overrides=fs_overrides,
        )
        seed_runs.append(run_res)

    agg = _aggregate_seed_results(seed_runs)
    summary = {
        "variant": variant,
        "calibrated": calibrate,
        "use_augmentation": bool(v_cfg.get("use_augmentation", False)),
        "variant_dir": str(variant_root_dir),
        "aggregate": agg,
        "runs": seed_runs,
    }
    (variant_root_dir / "variant_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TV2 N6: Binary TF-IDF + LinearSVC Pipeline")
    p.add_argument("--variant", choices=["A", "B", "C", "D", "all"], default="all")
    p.add_argument("--config", type=Path, default=DEFAULT_BINARY_CONFIG_PATH)
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--augmentation-path", type=Path, default=None)
    p.add_argument("--artifacts-dir", type=Path, default=None)
    p.add_argument("--tracking-uri", type=str, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--C", type=float, default=None)
    p.add_argument("--class-weight", choices=["balanced", "none"], default=None)
    p.add_argument("--k", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.artifacts_dir or (ROOT / "models" / "baselines")
    run_group_dir = base_dir / f"N6-run-{uuid.uuid4().hex}"
    run_group_dir.mkdir(parents=True, exist_ok=False)

    variants = ["A", "B", "C", "D"] if args.variant == "all" else [args.variant]
    svm_overrides: dict[str, Any] = {}
    if args.C is not None:
        svm_overrides["C"] = args.C
    if args.class_weight is not None:
        svm_overrides["class_weight"] = None if args.class_weight == "none" else "balanced"

    fs_overrides: dict[str, Any] | None = {"k": args.k} if args.k is not None else None

    all_summaries = []
    for v in variants:
        res = run_binary_variant(
            variant=v,
            config_path=args.config,
            dataset_dir=args.dataset_dir,
            augmentation_path=args.augmentation_path,
            artifacts_root=run_group_dir,
            tracking_uri=args.tracking_uri,
            seeds=args.seeds,
            calibrate=bool(args.calibrate),
            tune=bool(args.tune),
            svm_overrides=svm_overrides or None,
            fs_overrides=fs_overrides,
        )
        all_summaries.append(res)
        agg = res["aggregate"]
        print(
            f"[N6-{v}] macro_f1={agg['macro_f1']['mean']:.4f}±{agg['macro_f1']['std']:.4f} "
            f"| f1_toxic={agg['f1_toxic']['mean']:.4f}±{agg['f1_toxic']['std']:.4f} "
            f"| size={agg['model_size_mb']['mean']:.3f}MB"
        )

    summary_path = run_group_dir / "N6-SVM-summary.json"
    summary_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
