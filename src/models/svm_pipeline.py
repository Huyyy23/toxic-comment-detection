"""
SVM Pipeline — TV2 EXP-02.

Variants trong configs/svm_config.yaml:
  EXP-02-A: raw + word TF-IDF + LinearSVC baseline
  EXP-02-B: basic + word+char TF-IDF + LinearSVC
  EXP-02-C: teencode + word+char TF-IDF + chi2 feature selection + tune C/class_weight
  EXP-02-D: (Optional P2) RBF SVC

Quy tắc khoa học:
  1. Đọc train_val.jsonl qua src.data.loader.load_records + manifest.json (chặn test).
  2. Fit TF-IDF, SelectKBest và CalibratedClassifierCV CHỈ trên train / train-fold.
  3. Không bao giờ biến raw LinearSVC.decision_function thành xác suất.
  4. Ghi đầy đủ metrics, artifacts, predictions (kèm run_id) và MLflow lineage.

Chạy: python -m src.models.svm_pipeline --variant A|B|C|all [--tune] [--calibrate]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import SVC, LinearSVC

from src.data.loader import load_records
from src.features.preprocessing import preprocess_series
from src.features.tfidf_pipeline import build_tfidf, fit_tfidf, save_tfidf, transform_texts
from src.tracking.mlflow_tracker import MLflowTracker

LABEL_NAMES: dict[int, str] = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
LABELS: list[int] = [0, 1, 2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "svm_config.yaml"


def load_svm_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Đọc cấu hình SVM từ YAML."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: Path) -> str:
    """Tính SHA-256 cho một file bất kỳ."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_texts(texts: list[str]) -> str:
    """Tính SHA-256 cho danh sách văn bản."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
    return h.hexdigest()


def load_g0_train_val(
    data_path: Path,
    manifest_path: Path,
) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Load và kiểm chứng train/val theo Data Contract v1."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    records = load_records(data_path, manifest=manifest)
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]
    if not train_records or not val_records:
        raise ValueError(
            f"Dữ liệu phải chứa cả train và val (hiện có train={len(train_records)}, val={len(val_records)})"
        )
    return train_records, val_records, manifest


def build_feature_selector(
    method: str,
    k: int | None,
    n_available_features: int,
) -> SelectKBest | None:
    """Tạo feature selector chưa fit (hiện hỗ trợ chi2)."""
    if k is None:
        return None
    if method != "chi2":
        raise ValueError(f"Unsupported feature selection method: {method!r}")
    effective_k = min(int(k), int(n_available_features))
    if effective_k <= 0:
        raise ValueError(f"k phải dương, nhận: {k}")
    return SelectKBest(score_func=chi2, k=effective_k)


def build_svm_estimator(
    svm_cfg: dict[str, Any],
    seed: int,
    calibrate: bool = False,
    calibration_method: str = "sigmoid",
    calibration_cv: int = 5,
    n_min_class_samples: int | None = None,
) -> tuple[Any, str]:
    """Khởi tạo LinearSVC hoặc SVC(rbf), bọc CalibratedClassifierCV nếu calibrate=True."""
    kernel = svm_cfg.get("kernel", "linear")
    if kernel == "linear":
        base = LinearSVC(
            C=float(svm_cfg.get("C", 1.0)),
            class_weight=svm_cfg.get("class_weight"),
            max_iter=int(svm_cfg.get("max_iter", 2000)),
            dual=svm_cfg.get("dual", "auto"),
            random_state=seed,
        )
        model_name = "LinearSVC"
    elif kernel == "rbf":
        base = SVC(
            kernel="rbf",
            C=float(svm_cfg.get("C", 1.0)),
            gamma=svm_cfg.get("gamma", "scale"),
            class_weight=svm_cfg.get("class_weight"),
            probability=False,
            random_state=seed,
        )
        model_name = "SVC_RBF"
    else:
        raise ValueError(f"Unsupported SVM kernel: {kernel!r}")

    if not calibrate:
        return base, model_name

    effective_cv = int(calibration_cv)
    if n_min_class_samples is not None and n_min_class_samples < effective_cv:
        effective_cv = max(2, int(n_min_class_samples))
    calibrated = CalibratedClassifierCV(
        estimator=base,
        cv=effective_cv,
        method=calibration_method,
    )
    return calibrated, f"Calibrated_{model_name}"


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """Tính tập metrics chuẩn hóa theo Data Contract / SPEC."""
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=[LABEL_NAMES[i] for i in LABELS],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    per_class: dict[str, dict[str, float]] = {}
    for label_name in [LABEL_NAMES[i] for i in LABELS]:
        per_class[label_name] = {
            "precision": float(report[label_name]["precision"]),
            "recall": float(report[label_name]["recall"]),
            "f1": float(report[label_name]["f1-score"]),
            "support": float(report[label_name]["support"]),
        }

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_acc,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def _fit_and_eval_single(
    X_train_raw: Any,
    y_train: list[int],
    X_val_raw: Any,
    y_val: list[int],
    fs_cfg: dict[str, Any],
    svm_cfg: dict[str, Any],
    seed: int,
    calibrate: bool,
    cal_method: str,
    cal_cv: int,
) -> dict[str, Any]:
    """Fit feature selector (nếu bật) + SVM trên train, đánh giá trên val."""
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
        calibration_method=cal_method,
        calibration_cv=cal_cv,
        n_min_class_samples=min_class_count,
    )

    t_train = time.perf_counter()
    estimator.fit(X_train, y_train)
    train_time_s = time.perf_counter() - t_train

    t_inf = time.perf_counter()
    val_pred = [int(x) for x in estimator.predict(X_val)]
    val_proba_ordered: np.ndarray | None = None
    if calibrate:
        raw_proba = estimator.predict_proba(X_val)
        val_proba_ordered = np.zeros((len(y_val), len(LABELS)), dtype=float)
        for col_idx, cls in enumerate(list(estimator.classes_)):
            val_proba_ordered[:, int(cls)] = raw_proba[:, col_idx]
    inference_ms = ((time.perf_counter() - t_inf) / max(1, len(y_val))) * 1000.0

    metrics = compute_metrics(y_val, val_pred)
    metrics["train_time_s"] = round(train_time_s, 4)
    metrics["feature_selection_time_s"] = round(fs_time_s, 4)
    metrics["inference_ms_per_sample"] = round(inference_ms, 4)
    metrics["n_features_raw"] = int(X_train_raw.shape[1])
    metrics["n_features_selected"] = int(X_train.shape[1])

    return {
        "estimator": estimator,
        "selector": selector,
        "model_name": model_name,
        "val_pred": val_pred,
        "val_proba": val_proba_ordered,
        "metrics": metrics,
        "effective_fs_cfg": {
            "enabled": bool(selector is not None),
            "method": fs_cfg.get("method", "chi2") if selector is not None else None,
            "k_requested": fs_cfg.get("k"),
            "k_effective": int(X_train.shape[1]) if selector is not None else None,
        },
        "effective_svm_cfg": dict(svm_cfg),
    }


def _run_controlled_tuning(
    X_train_raw: Any,
    y_train: list[int],
    X_val_raw: Any,
    y_val: list[int],
    base_fs_cfg: dict[str, Any],
    base_svm_cfg: dict[str, Any],
    tuning_grid: dict[str, Any],
    seed: int,
    calibrate: bool,
    cal_method: str,
    cal_cv: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Chạy tuning có kiểm soát trên train->val, trả về kết quả tốt nhất và bảng lịch sử."""
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

                outcome = _fit_and_eval_single(
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    X_val_raw=X_val_raw,
                    y_val=y_val,
                    fs_cfg=fs_cand,
                    svm_cfg=svm_cand,
                    seed=seed,
                    calibrate=calibrate,
                    cal_method=cal_method,
                    cal_cv=cal_cv,
                )
                entry = {
                    "k_requested": k_cand,
                    "k_effective": outcome["metrics"]["n_features_selected"],
                    "C": float(c_cand),
                    "class_weight": cw_cand,
                    "val_macro_f1": outcome["metrics"]["macro_f1"],
                    "val_weighted_f1": outcome["metrics"]["weighted_f1"],
                    "train_time_s": outcome["metrics"]["train_time_s"],
                }
                history.append(entry)
                if (
                    best_outcome is None
                    or outcome["metrics"]["macro_f1"] > best_outcome["metrics"]["macro_f1"]
                ):
                    best_outcome = outcome

    assert best_outcome is not None
    return best_outcome, history


def load_svm_artifacts(variant_dir: Path) -> dict[str, Any]:
    """Load lại đầy đủ artifacts của một run SVM để tái lập hoặc đánh giá."""
    model = joblib.load(variant_dir / "svm_model.joblib")
    tfidf_data = joblib.load(variant_dir / "tfidf.joblib")
    selector_path = variant_dir / "feature_selector.joblib"
    selector = joblib.load(selector_path) if selector_path.exists() else None
    config = json.loads((variant_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((variant_dir / "metrics.json").read_text(encoding="utf-8"))
    return {
        "model": model,
        "tfidf": tfidf_data["transformer"],
        "tfidf_metadata": tfidf_data.get("metadata", {}),
        "selector": selector,
        "config": config,
        "metrics": metrics,
    }


def predict_from_artifacts(artifacts: dict[str, Any], texts: list[str]) -> list[int]:
    """Suy luận nhãn từ bộ artifacts đã lưu (không fit lại bất kỳ thành phần nào)."""
    preprocessing_variant = artifacts["config"]["preprocessing"]
    processed = preprocess_series(texts, variant=preprocessing_variant)
    X = transform_texts(artifacts["tfidf"], processed)
    if artifacts["selector"] is not None:
        X = artifacts["selector"].transform(X)
    return [int(x) for x in artifacts["model"].predict(X)]


def run_experiment(
    variant: str,
    config_path: Path = DEFAULT_CONFIG_PATH,
    data_path: Path | None = None,
    manifest_path: Path | None = None,
    artifacts_dir: Path | None = None,
    tracking_uri: str | None = None,
    calibrate: bool | None = None,
    tune: bool = False,
    result_type: str = "official",
    artifact_name: str | None = None,
    svm_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Chạy một variant EXP-02, lưu artifacts và log MLflow."""
    full_cfg = load_svm_config(config_path)
    if variant not in full_cfg["variants"]:
        raise KeyError(f"Variant không hợp lệ: {variant!r}")

    v_cfg = full_cfg["variants"][variant]
    seed = int(full_cfg.get("seed", 42))

    resolved_data = data_path or (ROOT / full_cfg["dataset"])
    resolved_manifest = manifest_path or (ROOT / full_cfg["manifest"])
    resolved_artifacts_root = artifacts_dir or (ROOT / "models" / "baselines")
    resolved_tracking = tracking_uri if tracking_uri is not None else full_cfg.get("tracking_uri")

    cal_cfg = full_cfg.get("calibration", {})
    do_calibrate = bool(cal_cfg.get("enabled", False)) if calibrate is None else bool(calibrate)
    cal_method = str(cal_cfg.get("method", "sigmoid"))
    cal_cv = int(cal_cfg.get("cv", 5))

    # 1. Load data theo G0 contract (ngăn test leakage)
    train_records, val_records, _ = load_g0_train_val(resolved_data, resolved_manifest)
    train_texts_raw = [r["text"] for r in train_records]
    train_labels = [int(r["label"]) for r in train_records]
    val_texts_raw = [r["text"] for r in val_records]
    val_labels = [int(r["label"]) for r in val_records]

    data_file_sha256 = sha256_file(resolved_data)
    manifest_file_sha256 = sha256_file(resolved_manifest)
    train_text_sha256 = sha256_texts(train_texts_raw)
    val_text_sha256 = sha256_texts(val_texts_raw)

    # 2. Preprocessing tĩnh
    t_pre = time.perf_counter()
    train_texts = preprocess_series(train_texts_raw, variant=v_cfg["preprocessing"])
    val_texts = preprocess_series(val_texts_raw, variant=v_cfg["preprocessing"])
    preprocess_time_s = time.perf_counter() - t_pre

    # 3. TF-IDF: FIT CHỈ TRÊN TRAIN
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

    # 4. Feature selection + Train + Eval (đơn lẻ hoặc tuning có kiểm soát)
    fs_cfg = v_cfg.get("feature_selection", {"enabled": False, "k": None})
    svm_cfg = dict(v_cfg["svm"])
    if svm_overrides:
        svm_cfg.update({k: v for k, v in svm_overrides.items() if v is not None})
    tuning_history: list[dict[str, Any]] | None = None

    if tune and "tuning_grid" in v_cfg:
        outcome, tuning_history = _run_controlled_tuning(
            X_train_raw=X_train_raw,
            y_train=train_labels,
            X_val_raw=X_val_raw,
            y_val=val_labels,
            base_fs_cfg=fs_cfg,
            base_svm_cfg=svm_cfg,
            tuning_grid=v_cfg["tuning_grid"],
            seed=seed,
            calibrate=do_calibrate,
            cal_method=cal_method,
            cal_cv=cal_cv,
        )
    else:
        outcome = _fit_and_eval_single(
            X_train_raw=X_train_raw,
            y_train=train_labels,
            X_val_raw=X_val_raw,
            y_val=val_labels,
            fs_cfg=fs_cfg,
            svm_cfg=svm_cfg,
            seed=seed,
            calibrate=do_calibrate,
            cal_method=cal_method,
            cal_cv=cal_cv,
        )

    metrics = outcome["metrics"]
    metrics["preprocess_time_s"] = round(preprocess_time_s, 4)
    metrics["tfidf_fit_time_s"] = round(tfidf_fit_time_s, 4)
    metrics["n_train"] = len(train_labels)
    metrics["n_val"] = len(val_labels)

    resolved_name = artifact_name or f"EXP-02-{variant}"
    variant_dir = resolved_artifacts_root / resolved_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    # 5. Lưu model, TF-IDF và selector để đo kích thước thực tế
    model_path = variant_dir / "svm_model.joblib"
    joblib.dump(outcome["estimator"], model_path)

    tfidf_path = variant_dir / "tfidf.joblib"
    save_tfidf(
        tfidf,
        tfidf_path,
        metadata={"mode": tfidf_cfg["mode"], "variant": resolved_name},
    )

    selector_path = variant_dir / "feature_selector.joblib"
    if outcome["selector"] is not None:
        joblib.dump(outcome["selector"], selector_path)
    elif selector_path.exists():
        selector_path.unlink()

    total_bytes = model_path.stat().st_size + tfidf_path.stat().st_size
    if selector_path.exists():
        total_bytes += selector_path.stat().st_size
    metrics["model_size_mb"] = round(total_bytes / (1024 * 1024), 4)

    if tuning_history is not None:
        with open(variant_dir / "tuning_results.json", "w", encoding="utf-8") as f:
            json.dump(tuning_history, f, ensure_ascii=False, indent=2)

    # 6. Mở MLflow run trước khi ghi val_predictions.json để gắn đúng run_id
    experiment_name = f"{full_cfg.get('experiment', 'EXP-02')}-SVM"
    tracker = MLflowTracker(experiment_name=experiment_name, tracking_uri=resolved_tracking)
    run_id = tracker.start_run(
        run_name=resolved_name,
        tags={
            "experiment": str(full_cfg.get("experiment", "EXP-02")),
            "variant": resolved_name,
            "owner": str(full_cfg.get("owner", "TV2")),
            "result_type": result_type,
            "model": outcome["model_name"],
            "calibrated": str(do_calibrate).lower(),
            "tuned": str(bool(tuning_history is not None)).lower(),
        },
    )

    try:
        # Predictions theo đúng contract (id, true_label, predicted_label, probabilities, run_id)
        predictions: list[dict[str, Any]] = []
        val_proba = outcome["val_proba"]
        for idx, (rec, pred) in enumerate(zip(val_records, outcome["val_pred"])):
            proba_entry: dict[str, float] | None = None
            if val_proba is not None:
                row_p = val_proba[idx]
                proba_entry = {
                    LABEL_NAMES[0]: round(float(row_p[0]), 6),
                    LABEL_NAMES[1]: round(float(row_p[1]), 6),
                    LABEL_NAMES[2]: round(float(row_p[2]), 6),
                }
            predictions.append(
                {
                    "id": rec["id"],
                    "true_label": int(rec["label"]),
                    "predicted_label": int(pred),
                    "probabilities": proba_entry,
                    "run_id": run_id,
                }
            )

        with open(variant_dir / "val_predictions.json", "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        config_snapshot = {
            "experiment": full_cfg.get("experiment", "EXP-02"),
            "variant": resolved_name,
            "run_id": run_id,
            "owner": full_cfg.get("owner", "TV2"),
            "description": v_cfg["description"],
            "hypothesis": v_cfg["hypothesis"].strip(),
            "preprocessing": v_cfg["preprocessing"],
            "tfidf": tfidf_cfg,
            "feature_selection": outcome["effective_fs_cfg"],
            "svm": outcome["effective_svm_cfg"],
            "calibrated": do_calibrate,
            "calibration_method": cal_method if do_calibrate else None,
            "calibration_cv": cal_cv if do_calibrate else None,
            "tuned": bool(tuning_history is not None),
            "seed": seed,
            "data_contract_version": full_cfg.get("data_contract_version", "1.0"),
            "dataset_path": str(resolved_data),
            "manifest_path": str(resolved_manifest),
            "dataset_file_sha256": data_file_sha256,
            "manifest_file_sha256": manifest_file_sha256,
            "train_text_sha256": train_text_sha256,
            "val_text_sha256": val_text_sha256,
            "n_train": len(train_labels),
            "n_val": len(val_labels),
        }
        with open(variant_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_snapshot, f, ensure_ascii=False, indent=2)

        with open(variant_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        with open(variant_dir / "run_id.txt", "w", encoding="utf-8") as f:
            f.write(run_id)

        tracker.log_environment()
        tracker.log_model_architecture(
            architecture=outcome["model_name"],
            model_type="svm_scratch",
        )
        tracker.log_params(
            {
                "preprocessing": v_cfg["preprocessing"],
                "tfidf_mode": tfidf_cfg["mode"],
                "word_ngram_range": str(tfidf_cfg["word_ngram_range"]),
                "char_ngram_range": str(tfidf_cfg["char_ngram_range"]),
                "max_features": tfidf_cfg.get("max_features"),
                "min_df": tfidf_cfg.get("min_df"),
                "fs_enabled": outcome["effective_fs_cfg"]["enabled"],
                "fs_k": outcome["effective_fs_cfg"]["k_effective"],
                "svm_kernel": outcome["effective_svm_cfg"].get("kernel", "linear"),
                "svm_C": outcome["effective_svm_cfg"].get("C"),
                "svm_class_weight": str(outcome["effective_svm_cfg"].get("class_weight")),
                "calibrated": do_calibrate,
                "seed": seed,
                "dataset_file_sha256": data_file_sha256[:16],
                "manifest_file_sha256": manifest_file_sha256[:16],
                "train_text_sha256": train_text_sha256[:16],
                "n_train": len(train_labels),
                "n_val": len(val_labels),
            }
        )
        scalar_metrics: dict[str, float] = {
            "val_macro_f1": metrics["macro_f1"],
            "val_weighted_f1": metrics["weighted_f1"],
            "val_balanced_accuracy": metrics["balanced_accuracy"],
            "train_time_s": metrics["train_time_s"],
            "inference_ms_per_sample": metrics["inference_ms_per_sample"],
            "model_size_mb": metrics["model_size_mb"],
            "n_features_raw": float(metrics["n_features_raw"]),
            "n_features_selected": float(metrics["n_features_selected"]),
        }
        for label_name, pc in metrics["per_class"].items():
            scalar_metrics[f"val_f1_{label_name}"] = pc["f1"]
            scalar_metrics[f"val_precision_{label_name}"] = pc["precision"]
            scalar_metrics[f"val_recall_{label_name}"] = pc["recall"]
        tracker.log_metrics(scalar_metrics)
        tracker.log_artifact(str(variant_dir))
        tracker.end_run("FINISHED")
    except Exception:
        tracker.end_run("FAILED")
        raise

    return {
        "variant": variant,
        "run_id": run_id,
        "metrics": metrics,
        "artifacts_dir": str(variant_dir),
        "calibrated": do_calibrate,
        "effective_svm_cfg": outcome["effective_svm_cfg"],
        "effective_fs_cfg": outcome["effective_fs_cfg"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TV2 EXP-02: TF-IDF + SVM Pipeline")
    p.add_argument(
        "--variant",
        choices=["A", "B", "C", "D", "all"],
        default="all",
        help="Variant chạy (A|B|C|D|all). 'all' chạy mặc định A, B, C.",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument("--manifest-path", type=Path, default=None)
    p.add_argument("--artifacts-dir", type=Path, default=None)
    p.add_argument("--tracking-uri", type=str, default=None)
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="Bọc LinearSVC bằng CalibratedClassifierCV (Platt scaling trên train CV).",
    )
    p.add_argument(
        "--tune",
        action="store_true",
        help="Chạy tuning_grid có kiểm soát trên train->val cho variant có khai báo grid (vd: C).",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Tên thư mục/run tùy chỉnh (mặc định: EXP-02-<variant>).",
    )
    p.add_argument(
        "--C",
        type=float,
        default=None,
        help="Override giá trị C cho SVM khi chạy đơn lẻ.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    variants = ["A", "B", "C"] if args.variant == "all" else [args.variant]
    svm_overrides = {"C": args.C} if args.C is not None else None
    results = []
    for v in variants:
        res = run_experiment(
            variant=v,
            config_path=args.config,
            data_path=args.data_path,
            manifest_path=args.manifest_path,
            artifacts_dir=args.artifacts_dir,
            tracking_uri=args.tracking_uri,
            calibrate=True if args.calibrate else None,
            tune=args.tune,
            artifact_name=args.output_name if len(variants) == 1 else None,
            svm_overrides=svm_overrides,
        )
        results.append(res)
        m = res["metrics"]
        print(
            f"[EXP-02-{v}] macro_f1={m['macro_f1']:.4f} "
            f"| time={m['train_time_s']:.3f}s | size={m['model_size_mb']:.3f}MB "
            f"| run_id={res['run_id']}"
        )

    if args.variant == "all":
        out_dir = args.artifacts_dir or (ROOT / "models" / "baselines")
        summary_path = out_dir / "EXP-02-summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
