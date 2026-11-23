"""
LR Pipeline — TV1 EXP-01.

Ba variant chính:
  EXP-01-A: raw + word TF-IDF + LR (C=1)
  EXP-01-B: basic + word+char TF-IDF + LR (tune C)
  EXP-01-C: teencode + word+char TF-IDF + LR + class_weight='balanced'

Quy trình:
  1. Load data từ JSONL contract v4 (đã có manifest)
  2. Preprocess theo variant
  3. Fit TF-IDF CHỈ TRÊN TRAIN
  4. Train LR
  5. Evaluate trên validation
  6. Log MLflow: config, seed, dataset hash, metrics, artifacts, predictions
  7. Lưu artifacts: model, tfidf, predictions

Chạy: python -m src.models.lr_pipeline --variant A|B|C [--tracking-uri <path>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Cho phép MLflow file store local mà không văng exception trên MLflow 3.x
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.features.preprocessing import preprocess_series
from src.features.tfidf_pipeline import build_tfidf, fit_tfidf, save_tfidf


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
LABEL_NAMES = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
LABELS = [0, 1, 2]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_data(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Tách train và val từ records; từ chối nếu phát hiện split=='test'."""
    test_records = [r for r in records if r.get("split") == "test"]
    if test_records:
        raise ValueError(
            f"Phát hiện {len(test_records)} test records trong input. "
            "Không được phép nạp locked test trong EXP-01."
        )
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    return train, val


def records_to_xy(records: list[dict]) -> tuple[list[str], list[int]]:
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    return texts, labels


def sha256_texts(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Variant configs
# ---------------------------------------------------------------------------

VARIANT_CONFIGS = {
    "A": {
        "description": "Raw text + word TF-IDF (1-2gram) + LR C=1",
        "preprocessing": "raw",
        "tfidf_mode": "word",
        "word_ngram_range": (1, 2),
        "char_ngram_range": (2, 4),
        "max_features": 100_000,
        "lr_C": 1.0,
        "lr_class_weight": None,
        "seed": 42,
        "hypothesis": (
            "Baseline: LR với TF-IDF word n-gram trên text gốc; "
            "không có preprocessing đặc thù tiếng Việt."
        ),
    },
    "B": {
        "description": "Basic cleanup + word+char TF-IDF + LR (tune C=5)",
        "preprocessing": "basic",
        "tfidf_mode": "word+char",
        "word_ngram_range": (1, 2),
        "char_ngram_range": (2, 4),
        "max_features": 100_000,
        "lr_C": 5.0,
        "lr_class_weight": None,
        "seed": 42,
        "hypothesis": (
            "Thêm basic preprocessing (lowercase, URL, HTML) và char n-gram "
            "giúp bắt được subword patterns tiếng Việt; tune C=5 tránh underfitting."
        ),
    },
    "C": {
        "description": "Teencode norm + word+char TF-IDF + LR + class_weight=balanced",
        "preprocessing": "teencode",
        "tfidf_mode": "word+char",
        "word_ngram_range": (1, 3),
        "char_ngram_range": (2, 5),
        "max_features": 150_000,
        "lr_C": 5.0,
        "lr_class_weight": "balanced",
        "seed": 42,
        "hypothesis": (
            "Chuẩn hóa teencode giúp giảm sparsity từ vựng; "
            "class_weight=balanced bù đắp class imbalance (HATE ít hơn nhiều); "
            "mở rộng n-gram range để bắt thêm context."
        ),
    },
}


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: list[int], y_pred: list[int], y_proba: np.ndarray) -> dict:
    """Tính các metric theo contract v4."""
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred,
        labels=LABELS,
        target_names=list(LABEL_NAMES.values()),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    per_class = {}
    for label_id, label_name in LABEL_NAMES.items():
        per_class[label_name] = {
            "precision": report[label_name]["precision"],
            "recall": report[label_name]["recall"],
            "f1": report[label_name]["f1-score"],
            "support": report[label_name]["support"],
        }

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def run_experiment(
    variant: str,
    data_path: Path,
    artifacts_dir: Path,
    tracking_uri: str | None = None,
    dataset_info_path: Path | None = None,
) -> dict:
    """Chạy một variant của EXP-01 và log vào MLflow.

    Args:
        variant: "A", "B", hoặc "C"
        data_path: Đường dẫn đến vihsd.jsonl (processed)
        artifacts_dir: Thư mục lưu artifact
        tracking_uri: MLflow tracking URI (None = mlflow default)
        dataset_info_path: dataset_info.json (để log provenance)

    Returns:
        Dict kết quả gồm metrics và đường dẫn artifacts.
    """
    cfg = VARIANT_CONFIGS[variant]
    print(f"\n{'='*60}")
    print(f"EXP-01-{variant}: {cfg['description']}")
    print(f"Hypothesis: {cfg['hypothesis']}")
    print(f"{'='*60}")

    # --- Load data ---
    print("Loading data...")
    records = load_jsonl(data_path)
    train_records, val_records = split_data(records)
    print(f"  Train: {len(train_records)} | Val: {len(val_records)}")

    train_texts_raw, train_labels = records_to_xy(train_records)
    val_texts_raw, val_labels = records_to_xy(val_records)

    dataset_hash = sha256_texts(train_texts_raw + val_texts_raw)
    train_hash = sha256_texts(train_texts_raw)
    val_hash = sha256_texts(val_texts_raw)
    data_file_sha256 = sha256_file(data_path)

    dataset_info: dict = {}
    if dataset_info_path and dataset_info_path.exists():
        with open(dataset_info_path, encoding="utf-8") as f:
            dataset_info = json.load(f)

    # --- Preprocessing (fit không cần thiết — deterministic) ---
    print(f"Preprocessing variant: {cfg['preprocessing']}...")
    t0 = time.time()
    train_texts = preprocess_series(train_texts_raw, variant=cfg["preprocessing"])
    val_texts = preprocess_series(val_texts_raw, variant=cfg["preprocessing"])
    preprocess_time = time.time() - t0

    # --- TF-IDF: FIT CHỈ TRÊN TRAIN ---
    print(f"Building TF-IDF ({cfg['tfidf_mode']})...")
    tfidf = build_tfidf(
        mode=cfg["tfidf_mode"],
        word_ngram_range=tuple(cfg["word_ngram_range"]),
        char_ngram_range=tuple(cfg["char_ngram_range"]),
        max_features=cfg["max_features"],
    )
    t0 = time.time()
    tfidf = fit_tfidf(tfidf, train_texts)
    X_train = tfidf.transform(train_texts)
    X_val = tfidf.transform(val_texts)
    tfidf_time = time.time() - t0
    print(f"  Feature shape: {X_train.shape}")

    # --- Logistic Regression ---
    print(f"Training LR (C={cfg['lr_C']}, class_weight={cfg['lr_class_weight']})...")
    lr = LogisticRegression(
        C=cfg["lr_C"],
        class_weight=cfg["lr_class_weight"],
        max_iter=1000,
        random_state=cfg["seed"],
        solver="lbfgs",
    )
    t0 = time.time()
    lr.fit(X_train, train_labels)
    train_time = time.time() - t0

    # --- Evaluation ---
    print("Evaluating...")
    t0 = time.time()
    val_pred = lr.predict(X_val).tolist()
    val_proba = lr.predict_proba(X_val)  # shape (n, 3), thứ tự theo lr.classes_
    inference_time_ms = (time.time() - t0) / len(val_labels) * 1000

    # Đảm bảo thứ tự probabilities là [CLEAN, OFFENSIVE, HATE]
    classes = list(lr.classes_)
    prob_ordered = np.zeros((len(val_labels), 3))
    for col_idx, cls in enumerate(classes):
        prob_ordered[:, int(cls)] = val_proba[:, col_idx]

    metrics = compute_metrics(val_labels, val_pred, prob_ordered)
    metrics["train_time_s"] = round(train_time, 3)
    metrics["preprocess_time_s"] = round(preprocess_time, 3)
    metrics["tfidf_fit_time_s"] = round(tfidf_time, 3)
    metrics["inference_ms_per_sample"] = round(inference_time_ms, 4)
    metrics["n_features"] = X_train.shape[1]
    metrics["n_train"] = len(train_labels)
    metrics["n_val"] = len(val_labels)

    print(f"  Macro-F1 (val): {metrics['macro_f1']:.4f}")
    print(f"  Per-class F1: { {k: round(v['f1'], 4) for k, v in metrics['per_class'].items()} }")

    # --- Lưu artifacts ---
    artifacts_dir = artifacts_dir / f"EXP-01-{variant}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = artifacts_dir / "lr_model.joblib"
    joblib.dump(lr, model_path)

    # TF-IDF
    tfidf_path = artifacts_dir / "tfidf.joblib"
    save_tfidf(tfidf, tfidf_path, metadata={"mode": cfg["tfidf_mode"], "variant": variant})

    metrics["model_size_bytes"] = int(model_path.stat().st_size + tfidf_path.stat().st_size)

    # Validation predictions (schema v4 JSON)
    predictions = []
    csv_rows = []
    for rec, pred, proba in zip(val_records, val_pred, prob_ordered.tolist()):
        predictions.append({
            "id": rec["id"],
            "true_label": rec["label"],
            "predicted_label": pred,
            "probabilities": {
                "CLEAN": round(proba[0], 6),
                "OFFENSIVE": round(proba[1], 6),
                "HATE": round(proba[2], 6),
            },
        })
        csv_rows.append({
            "sample_id": rec["id"],
            "y_true": rec["label"],
            "y_pred": pred,
            "prob_CLEAN": round(proba[0], 6),
            "prob_OFFENSIVE": round(proba[1], 6),
            "prob_HATE": round(proba[2], 6),
        })
    pred_path = artifacts_dir / "val_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    pred_csv_path = artifacts_dir / "predictions_val.csv"
    pd.DataFrame(csv_rows).to_csv(pred_csv_path, index=False, encoding="utf-8")

    # Config snapshot
    config_path = artifacts_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "variant": f"EXP-01-{variant}",
                "description": cfg["description"],
                "hypothesis": cfg["hypothesis"],
                "preprocessing": cfg["preprocessing"],
                "tfidf_mode": cfg["tfidf_mode"],
                "word_ngram_range": cfg["word_ngram_range"],
                "char_ngram_range": cfg["char_ngram_range"],
                "max_features": cfg["max_features"],
                "lr_C": cfg["lr_C"],
                "lr_class_weight": cfg["lr_class_weight"],
                "seed": cfg["seed"],
                "dataset_path": str(data_path),
                "data_contract_version": dataset_info.get("data_contract_version"),
                "source_revision": dataset_info.get("source_revision"),
                "data_file_sha256": data_file_sha256,
                "dataset_hash": dataset_hash,
                "train_hash": train_hash,
                "val_hash": val_hash,
                "n_train": len(train_labels),
                "n_val": len(val_labels),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Metrics
    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # --- MLflow logging ---
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("EXP-01-LR")

    with mlflow.start_run(run_name=f"EXP-01-{variant}") as run:
        run_id = run.info.run_id

        # Tags
        mlflow.set_tags({
            "variant": f"EXP-01-{variant}",
            "owner": "TV1",
            "result_type": "official",
            "research_result": "true",
            "preprocessing": cfg["preprocessing"],
            "model": "LogisticRegression",
        })

        # Params
        mlflow.log_params({
            "preprocessing": cfg["preprocessing"],
            "tfidf_mode": cfg["tfidf_mode"],
            "word_ngram_range": str(cfg["word_ngram_range"]),
            "char_ngram_range": str(cfg["char_ngram_range"]),
            "max_features": cfg["max_features"],
            "lr_C": cfg["lr_C"],
            "lr_class_weight": str(cfg["lr_class_weight"]),
            "seed": cfg["seed"],
            "data_contract_version": str(dataset_info.get("data_contract_version", "unknown")),
            "data_file_sha256": data_file_sha256[:16],
            "dataset_hash": dataset_hash[:16],
            "train_hash": train_hash[:16],
            "val_hash": val_hash[:16],
            "n_train": len(train_labels),
            "n_val": len(val_labels),
        })

        # Metrics
        mlflow.log_metric("val_macro_f1", metrics["macro_f1"])
        mlflow.log_metric("val_weighted_f1", metrics["weighted_f1"])
        mlflow.log_metric("n_features", metrics["n_features"])
        mlflow.log_metric("train_time_s", metrics["train_time_s"])
        mlflow.log_metric("inference_ms_per_sample", metrics["inference_ms_per_sample"])
        mlflow.log_metric("model_size_bytes", metrics["model_size_bytes"])

        for label_name, pc in metrics["per_class"].items():
            mlflow.log_metric(f"val_f1_{label_name}", pc["f1"])
            mlflow.log_metric(f"val_precision_{label_name}", pc["precision"])
            mlflow.log_metric(f"val_recall_{label_name}", pc["recall"])

        # Artifacts
        mlflow.log_artifacts(str(artifacts_dir))

        print(f"\n  MLflow run_id: {run_id}")

    # Lưu run_id vào artifact dir
    with open(artifacts_dir / "run_id.txt", "w") as f:
        f.write(run_id)

    return {
        "variant": variant,
        "run_id": run_id,
        "metrics": metrics,
        "artifacts_dir": str(artifacts_dir),
    }


def load_and_predict(artifacts_dir: Path, texts: list[str]) -> tuple[list[int], np.ndarray]:
    """Load lại model, TF-IDF và config đã lưu để dự đoán không cần train lại."""
    from src.features.tfidf_pipeline import load_tfidf

    with open(artifacts_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    lr = joblib.load(artifacts_dir / "lr_model.joblib")
    tfidf, _ = load_tfidf(artifacts_dir / "tfidf.joblib")

    processed = preprocess_series(texts, variant=cfg["preprocessing"])
    X = tfidf.transform(processed)
    preds = lr.predict(X).tolist()
    raw_proba = lr.predict_proba(X)

    prob_ordered = np.zeros((len(texts), 3))
    for col_idx, cls in enumerate(list(lr.classes_)):
        prob_ordered[:, int(cls)] = raw_proba[:, col_idx]
    return preds, prob_ordered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TV1 EXP-01: TF-IDF + Logistic Regression")
    p.add_argument(
        "--variant",
        choices=["A", "B", "C", "all"],
        default="all",
        help="Variant để chạy (A|B|C|all). Default: all",
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "data" / "processed" / "vihsd_g0_v1" / "train_val.jsonl",
        help="Đường dẫn JSONL G0 train_val",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ROOT / "models" / "baselines",
        help="Thư mục lưu artifacts",
    )
    p.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (vd: file:///path/to/mlruns)",
    )
    p.add_argument(
        "--dataset-info",
        type=Path,
        default=ROOT / "data" / "processed" / "vihsd_g0_v1" / "dataset_info.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_path.exists():
        print(f"ERROR: Không tìm thấy data tại {args.data_path}")
        print("Chạy trước: python -m src.data.build_g0_dataset --raw-dir data/raw --output-dir data/processed/vihsd_g0_v1")
        sys.exit(1)

    variants = ["A", "B", "C"] if args.variant == "all" else [args.variant]
    results = []
    for v in variants:
        result = run_experiment(
            variant=v,
            data_path=args.data_path,
            artifacts_dir=args.artifacts_dir,
            tracking_uri=args.tracking_uri,
            dataset_info_path=args.dataset_info,
        )
        results.append(result)

    # In bảng so sánh
    print(f"\n{'='*60}")
    print("KẾT QUẢ SO SÁNH EXP-01")
    print(f"{'='*60}")
    print(f"{'Variant':<12} {'Macro-F1':>10} {'CLEAN-F1':>10} {'OFF-F1':>10} {'HATE-F1':>10}")
    print("-" * 55)
    for r in results:
        m = r["metrics"]
        pc = m["per_class"]
        print(
            f"EXP-01-{r['variant']:<5} "
            f"{m['macro_f1']:>10.4f} "
            f"{pc['CLEAN']['f1']:>10.4f} "
            f"{pc['OFFENSIVE']['f1']:>10.4f} "
            f"{pc['HATE']['f1']:>10.4f}"
        )

    # Lưu summary
    summary_path = args.artifacts_dir / "EXP-01-summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSummary lưu tại: {summary_path}")


if __name__ == "__main__":
    main()
