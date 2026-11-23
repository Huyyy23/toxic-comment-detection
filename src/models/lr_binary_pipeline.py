"""
LR Binary Pipeline — N6 TV1 (Tam) theo SPEC v5.0.0.

Đặc tả:
  - Dataset: data/processed/binary_n2_v1/ (train.parquet, val.parquet, manifest.json).
  - Ablation tăng cường train-only: data/processed/binary_n4_v1/augmentation.parquet.
  - Nhãn nhị phân: 0 = CLEAN, 1 = TOXIC.
  - Chạy đa seed (mặc định [42, 43, 44]), báo mean ± std và metrics theo từng seed.
  - Metric chọn mô hình: Validation Macro-F1; bổ sung F1 TOXIC, PR-AUC, per-class,
    metrics tách riêng theo source ('vihsd', 'victsd'), ECE, latency và size (< 15 MB).
  - Calibration + threshold chỉ chạy trên train/val (không dùng test).

Chạy:
  python -m src.models.lr_binary_pipeline --variant A|B|C|D|all [--tune] [--calibrate]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline

from src.features.preprocessing import preprocess_series
from src.features.tfidf_pipeline import build_tfidf, fit_tfidf, save_tfidf, transform_texts
from src.models.svm_binary_pipeline import (
    BINARY_LABEL_NAMES,
    BINARY_LABELS,
    _build_aggregate_error_summary,
    _find_best_threshold,
    compute_binary_metrics,
    compute_ece,
    load_binary_train_val,
    sha256_texts,
)
from src.tracking.mlflow_tracker import MLflowTracker

DEFAULT_LR_BINARY_CONFIG_PATH = ROOT / "configs" / "lr_binary_config.yaml"


def load_lr_binary_config(config_path: Path = DEFAULT_LR_BINARY_CONFIG_PATH) -> dict[str, Any]:
    """Đọc cấu hình LR nhị phân từ YAML."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_lr_estimator(lr_cfg: dict[str, Any], seed: int) -> LogisticRegression:
    """Khởi tạo LogisticRegression từ config."""
    return LogisticRegression(
        C=float(lr_cfg.get("C", 1.0)),
        class_weight=lr_cfg.get("class_weight"),
        max_iter=int(lr_cfg.get("max_iter", 1000)),
        solver=str(lr_cfg.get("solver", "lbfgs")),
        random_state=seed,
    )


def _fit_and_eval_lr_single(
    X_train_raw: Any,
    y_train: list[int],
    X_val_raw: Any,
    y_val: list[int],
    val_sources: list[str],
    lr_cfg: dict[str, Any],
    seed: int,
    calibrate: bool,
    cal_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Fit LogisticRegression trên train, chấm điểm trên val (tuỳ chọn Platt calibration)."""
    t0 = time.perf_counter()
    base_clf = build_lr_estimator(lr_cfg, seed=seed)
    base_clf.fit(X_train_raw, y_train)
    train_time_s = round(time.perf_counter() - t0, 3)

    t_inf = time.perf_counter()
    raw_probs = base_clf.predict_proba(X_val_raw)[:, 1]
    val_pred = base_clf.predict(X_val_raw).tolist()
    inf_ms = round((time.perf_counter() - t_inf) * 1000 / max(1, len(y_val)), 4)

    metrics = compute_binary_metrics(y_val, val_pred, val_sources, scores_or_probs=raw_probs)
    y_val_arr = np.asarray(y_val, dtype=int)
    n_bins = int(cal_cfg.get("n_ece_bins", 10))
    metrics["raw_ece"] = float(compute_ece(y_val_arr, raw_probs, n_bins=n_bins))
    metrics["raw_log_loss"] = float(log_loss(y_val_arr, raw_probs, labels=BINARY_LABELS))
    metrics["raw_brier_score"] = float(brier_score_loss(y_val_arr, raw_probs))
    metrics["train_time_s"] = train_time_s
    metrics["inference_latency_ms_per_sample"] = inf_ms
    metrics["n_features_final"] = int(X_train_raw.shape[1])

    fitted_model: Any = base_clf
    val_probs: np.ndarray = base_clf.predict_proba(X_val_raw)
    cal_diagnostics: dict[str, Any] | None = None

    if calibrate:
        t_cal = time.perf_counter()
        cv_folds = int(cal_cfg.get("cv", 5))
        method = str(cal_cfg.get("method", "sigmoid"))
        cal_clf = CalibratedClassifierCV(
            estimator=build_lr_estimator(lr_cfg, seed=seed),
            method=method,
            cv=cv_folds,
        )
        cal_clf.fit(X_train_raw, y_train)
        cal_time_s = round(time.perf_counter() - t_cal, 3)
        val_probs = cal_clf.predict_proba(X_val_raw)
        prob_pos = val_probs[:, 1]

        best_thr, thr_macro, thr_toxic_f1 = _find_best_threshold(
            y_val_arr,
            prob_pos,
            t_min=float(cal_cfg.get("threshold_grid_min", 0.10)),
            t_max=float(cal_cfg.get("threshold_grid_max", 0.90)),
            t_step=float(cal_cfg.get("threshold_grid_step", 0.01)),
        )
        default_pred = (prob_pos >= 0.5).astype(int).tolist()
        default_metrics = compute_binary_metrics(y_val, default_pred, val_sources, scores_or_probs=prob_pos)
        cal_diagnostics = {
            "method": method,
            "cv": cv_folds,
            "calibration_time_s": cal_time_s,
            "ece": float(compute_ece(y_val_arr, prob_pos, n_bins=n_bins)),
            "log_loss": float(log_loss(y_val_arr, prob_pos, labels=BINARY_LABELS)),
            "brier_score": float(brier_score_loss(y_val_arr, prob_pos)),
            "pr_auc": default_metrics["pr_auc"],
            "default_threshold_0_50": {
                "macro_f1": default_metrics["macro_f1"],
                "f1_toxic": default_metrics["f1_toxic"],
                "weighted_f1": default_metrics["weighted_f1"],
            },
            "val_selected_threshold": {
                "threshold": best_thr,
                "macro_f1": thr_macro,
                "f1_toxic": thr_toxic_f1,
                "note": "Ngưỡng chọn và đánh giá trên cùng validation (in-sample diagnostic; không dùng để chọn model).",
            },
        }
        fitted_model = cal_clf

    return {
        "model": fitted_model,
        "metrics": metrics,
        "val_pred": val_pred,
        "val_probs": val_probs,
        "calibration_diagnostics": cal_diagnostics,
    }


def _aggregate_seed_metrics(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Tổng hợp mean ± std qua nhiều seed."""
    def _stat(vals: list[float]) -> dict[str, float]:
        arr = np.asarray(vals, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    metrics_list = [r["metrics"] for r in seed_results]
    agg: dict[str, Any] = {
        "n_seeds": len(seed_results),
        "macro_f1": _stat([m["macro_f1"] for m in metrics_list]),
        "f1_toxic": _stat([m["f1_toxic"] for m in metrics_list]),
        "weighted_f1": _stat([m["weighted_f1"] for m in metrics_list]),
        "balanced_accuracy": _stat([m["balanced_accuracy"] for m in metrics_list]),
        "raw_ece": _stat([m["raw_ece"] for m in metrics_list]),
        "train_time_s": _stat([m["train_time_s"] for m in metrics_list]),
        "fallback_size_mb": _stat([r["fallback_size_mb"] for r in seed_results]),
    }
    pr_aucs = [m["pr_auc"] for m in metrics_list if m.get("pr_auc") is not None]
    if pr_aucs:
        agg["pr_auc"] = _stat(pr_aucs)

    by_source: dict[str, Any] = {}
    all_sources = sorted({src for m in metrics_list for src in m.get("by_source", {}).keys()})
    for src in all_sources:
        by_source[src] = {
            "macro_f1": _stat([m["by_source"][src]["macro_f1"] for m in metrics_list]),
            "f1_toxic": _stat([m["by_source"][src]["f1_toxic"] for m in metrics_list]),
            "weighted_f1": _stat([m["by_source"][src]["weighted_f1"] for m in metrics_list]),
        }
    agg["by_source"] = by_source

    cal_list = [r["calibration_diagnostics"] for r in seed_results if r.get("calibration_diagnostics")]
    if cal_list:
        agg["calibration_diagnostics"] = {
            "ece": _stat([c["ece"] for c in cal_list]),
            "log_loss": _stat([c["log_loss"] for c in cal_list]),
            "brier_score": _stat([c["brier_score"] for c in cal_list]),
            "pr_auc": _stat([c["pr_auc"] for c in cal_list if c.get("pr_auc") is not None]),
            "default_0_50_macro_f1": _stat([c["default_threshold_0_50"]["macro_f1"] for c in cal_list]),
            "val_selected_threshold_mean": float(
                np.mean([c["val_selected_threshold"]["threshold"] for c in cal_list])
            ),
            "val_selected_macro_f1": _stat([c["val_selected_threshold"]["macro_f1"] for c in cal_list]),
            "val_selected_f1_toxic": _stat([c["val_selected_threshold"]["f1_toxic"] for c in cal_list]),
        }
    return agg


def run_lr_binary_variant(
    variant: str,
    config_path: Path = DEFAULT_LR_BINARY_CONFIG_PATH,
    dataset_dir: Path | None = None,
    augmentation_path: Path | None = None,
    artifacts_root: Path | None = None,
    tracking_uri: str | None = None,
    seeds: list[int] | None = None,
    tune: bool = False,
    calibrate: bool = False,
    result_type: str = "official",
) -> dict[str, Any]:
    """Chạy đầy đủ một biến thể LR nhị phân (A, B, C, hoặc D) trên multi-seed."""
    cfg_all = load_lr_binary_config(config_path)
    if variant not in cfg_all["variants"]:
        raise ValueError(f"Variant không hợp lệ: {variant!r}. Chọn từ {list(cfg_all['variants'].keys())}")

    v_cfg = dict(cfg_all["variants"][variant])
    ds_dir = dataset_dir or (ROOT / cfg_all["dataset_dir"])
    aug_path = augmentation_path or (ROOT / cfg_all["augmentation_path"])
    use_aug = bool(v_cfg.get("use_augmentation", False))
    seeds_to_run = seeds or [int(s) for s in cfg_all.get("seeds", [42, 43, 44])]
    cal_cfg = cfg_all.get("calibration", {})
    max_size_mb = float(cfg_all.get("max_fallback_size_mb", 15.0))

    train_df, val_df, file_hashes = load_binary_train_val(
        dataset_dir=ds_dir,
        augmentation_path=aug_path,
        use_augmentation=use_aug,
    )

    prep_mode = v_cfg["preprocessing"]
    train_texts = preprocess_series(train_df["text"].astype(str).tolist(), variant=prep_mode)
    val_texts = preprocess_series(val_df["text"].astype(str).tolist(), variant=prep_mode)
    y_train = [int(x) for x in train_df["label"].tolist()]
    y_val = [int(x) for x in val_df["label"].tolist()]
    val_sources = [str(x) for x in val_df["source"].tolist()]

    tfidf_cfg = dict(v_cfg["tfidf"])
    tfidf = build_tfidf(
        mode=tfidf_cfg["mode"],
        word_ngram_range=tuple(tfidf_cfg["word_ngram_range"]),
        char_ngram_range=tuple(tfidf_cfg["char_ngram_range"]),
        max_features=tfidf_cfg.get("max_features"),
        sublinear_tf=bool(tfidf_cfg.get("sublinear_tf", True)),
        min_df=int(tfidf_cfg.get("min_df", 2)),
    )
    fit_tfidf(tfidf, train_texts)
    X_train_raw = transform_texts(tfidf, train_texts)
    X_val_raw = transform_texts(tfidf, val_texts)

    lr_cfg = dict(v_cfg["lr"])
    tuning_history: list[dict[str, Any]] = []

    if tune and "tuning_grid" in v_cfg:
        grid = v_cfg["tuning_grid"]
        best_macro = -1.0
        best_pair: tuple[float, Any] | None = None
        tune_seed = seeds_to_run[0]

        for c_val in grid.get("C_candidates", [lr_cfg["C"]]):
            for cw in grid.get("class_weight_candidates", [lr_cfg.get("class_weight")]):
                trial_lr = dict(lr_cfg, C=c_val, class_weight=cw)
                res = _fit_and_eval_lr_single(
                    X_train_raw, y_train, X_val_raw, y_val, val_sources,
                    trial_lr, seed=tune_seed, calibrate=False, cal_cfg=cal_cfg,
                )
                m = res["metrics"]
                tuning_history.append({
                    "C": c_val,
                    "class_weight": cw,
                    "macro_f1": m["macro_f1"],
                    "f1_toxic": m["f1_toxic"],
                    "weighted_f1": m["weighted_f1"],
                    "raw_ece": m["raw_ece"],
                    "train_time_s": m["train_time_s"],
                })
                if m["macro_f1"] > best_macro:
                    best_macro = m["macro_f1"]
                    best_pair = (c_val, cw)

        if best_pair is not None:
            lr_cfg["C"], lr_cfg["class_weight"] = best_pair
            print(f"[N6-LR-{variant}] Best tuned: C={lr_cfg['C']}, class_weight={lr_cfg['class_weight']}, macro_f1={best_macro:.4f}")

    run_base = artifacts_root or (ROOT / "models" / "baselines" / f"N6-LR-run-{uuid.uuid4().hex}")
    variant_name = f"EXP-N6-LR-{variant}{'-calibrated' if calibrate else ''}"
    variant_dir = run_base / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    tracker = MLflowTracker(
        experiment_name=f"N6-LR-Binary-{variant}",
        tracking_uri=tracking_uri or cfg_all.get("tracking_uri"),
    )

    data_hashes = {
        **file_hashes,
        "train_texts_sha256": sha256_texts(train_df["text"].astype(str).tolist()),
        "val_texts_sha256": sha256_texts(val_df["text"].astype(str).tolist()),
    }

    seed_results: list[dict[str, Any]] = []
    for seed in seeds_to_run:
        seed_dir = variant_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        eval_out = _fit_and_eval_lr_single(
            X_train_raw, y_train, X_val_raw, y_val, val_sources,
            lr_cfg, seed=seed, calibrate=calibrate, cal_cfg=cal_cfg,
        )

        # Lưu artifacts & kiểm tra size < 15 MB
        save_tfidf(tfidf, seed_dir / "tfidf.joblib", metadata={"variant": variant, "mode": tfidf_cfg["mode"]})
        joblib.dump(eval_out["model"], seed_dir / "model.joblib")
        fallback_pipe = Pipeline([("tfidf", tfidf), ("clf", eval_out["model"])])
        fallback_path = seed_dir / "fallback_pipeline.joblib"
        joblib.dump(fallback_pipe, fallback_path, compress=3)
        fallback_size_mb = round(fallback_path.stat().st_size / (1024 * 1024), 4)

        full_metrics = {
            **eval_out["metrics"],
            "seed": seed,
            "variant": variant_name,
            "use_augmentation": use_aug,
            "fallback_size_mb": fallback_size_mb,
            "within_15mb_budget": fallback_size_mb <= max_size_mb,
        }
        if eval_out["calibration_diagnostics"] is not None:
            full_metrics["calibration_diagnostics"] = eval_out["calibration_diagnostics"]

        config_record = {
            "experiment": cfg_all.get("experiment", "N6-LR-Binary"),
            "owner": cfg_all.get("owner", "TV1"),
            "variant": variant_name,
            "result_type": result_type,
            "data_contract_version": cfg_all.get("data_contract_version", "binary_n2_v1"),
            "use_augmentation": use_aug,
            "seed": seed,
            "preprocessing": prep_mode,
            "tfidf": tfidf_cfg,
            "lr": lr_cfg,
            "calibrated": calibrate,
            "data_hashes": data_hashes,
            "n_train": len(train_df),
            "n_val": len(val_df),
        }

        val_probs = eval_out["val_probs"]
        predictions = []
        for idx, (sid, src, true_lbl, pred_lbl) in enumerate(
            zip(val_df["id"].tolist(), val_sources, y_val, eval_out["val_pred"])
        ):
            predictions.append({
                "id": str(sid),
                "source": src,
                "true_label": int(true_lbl),
                "true_label_name": BINARY_LABEL_NAMES[int(true_lbl)],
                "predicted_label": int(pred_lbl),
                "predicted_label_name": BINARY_LABEL_NAMES[int(pred_lbl)],
                "probabilities": {
                    "CLEAN": float(val_probs[idx, 0]),
                    "TOXIC": float(val_probs[idx, 1]),
                },
                "is_calibrated": calibrate,
            })

        error_summary = _build_aggregate_error_summary(val_df, eval_out["val_pred"])

        (seed_dir / "config.json").write_text(json.dumps(config_record, indent=2, ensure_ascii=False), encoding="utf-8")
        (seed_dir / "metrics.json").write_text(json.dumps(full_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        (seed_dir / "val_predictions.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
        (seed_dir / "error_summary.json").write_text(json.dumps(error_summary, indent=2, ensure_ascii=False), encoding="utf-8")

        run_id = tracker.start_run(run_name=f"{variant_name}-seed-{seed}")
        try:
            tracker.log_params({
                "variant": variant_name,
                "seed": seed,
                "preprocessing": prep_mode,
                "tfidf_mode": tfidf_cfg["mode"],
                "lr_C": lr_cfg["C"],
                "lr_class_weight": str(lr_cfg.get("class_weight")),
                "use_augmentation": use_aug,
                "calibrated": calibrate,
            })
            tracker.log_metrics({
                "val_macro_f1": full_metrics["macro_f1"],
                "val_f1_toxic": full_metrics["f1_toxic"],
                "val_weighted_f1": full_metrics["weighted_f1"],
                "val_raw_ece": full_metrics["raw_ece"],
                "fallback_size_mb": fallback_size_mb,
            })
            tracker.log_artifact(str(seed_dir))
            tracker.end_run(status="FINISHED")
        except Exception:
            tracker.end_run(status="FAILED")
            raise

        (seed_dir / "run_id.txt").write_text(run_id, encoding="utf-8")
        seed_results.append({
            "seed": seed,
            "run_id": run_id,
            "seed_dir": str(seed_dir),
            "metrics": full_metrics,
            "fallback_size_mb": fallback_size_mb,
            "calibration_diagnostics": eval_out["calibration_diagnostics"],
        })

    aggregate = _aggregate_seed_metrics(seed_results)
    variant_summary = {
        "variant": variant_name,
        "description": v_cfg.get("description", ""),
        "hypothesis": v_cfg.get("hypothesis", ""),
        "lr": lr_cfg,
        "use_augmentation": use_aug,
        "calibrated": calibrate,
        "tuning_history": tuning_history,
        "aggregate": aggregate,
        "runs": seed_results,
    }
    (variant_dir / "variant_summary.json").write_text(
        json.dumps(variant_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return variant_summary


def load_lr_binary_artifacts(artifact_dir: Path) -> Pipeline:
    """Load fallback_pipeline.joblib để dự đoán lại mà không cần train lại."""
    return joblib.load(artifact_dir / "fallback_pipeline.joblib")


def predict_lr_binary_from_artifacts(
    pipeline: Pipeline,
    texts: list[str],
    preprocessing_mode: str = "raw",
) -> list[int]:
    """Dự đoán nhãn 0/1 từ pipeline đã load (nhớ preprocess đúng mode của variant)."""
    cleaned = preprocess_series(texts, variant=preprocessing_mode)
    return [int(x) for x in pipeline.predict(cleaned).tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description="N6 Binary Logistic Regression Pipeline (TV1)")
    parser.add_argument("--variant", default="all", choices=["A", "B", "C", "D", "all"])
    parser.add_argument("--config", type=Path, default=DEFAULT_LR_BINARY_CONFIG_PATH)
    parser.add_argument("--artifacts-root", type=Path, default=None)
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    variants = ["A", "B", "C", "D"] if args.variant == "all" else [args.variant]
    run_root = args.artifacts_root or (ROOT / "models" / "baselines" / f"N6-LR-run-{uuid.uuid4().hex}")

    summaries = {}
    for v in variants:
        print(f"\n=== Running N6 LR Binary Variant {v} ===")
        res = run_lr_binary_variant(
            variant=v,
            config_path=args.config,
            artifacts_root=run_root,
            tracking_uri=args.tracking_uri,
            tune=args.tune,
            calibrate=args.calibrate,
        )
        summaries[v] = res["aggregate"]
        agg = res["aggregate"]
        print(
            f"[Variant {v}] Macro-F1={agg['macro_f1']['mean']:.4f}±{agg['macro_f1']['std']:.4f} | "
            f"F1_TOXIC={agg['f1_toxic']['mean']:.4f} | Size={agg['fallback_size_mb']['mean']:.3f} MB"
        )

    (run_root / "N6-LR-summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved N6 LR artifacts to: {run_root}")


if __name__ == "__main__":
    main()
