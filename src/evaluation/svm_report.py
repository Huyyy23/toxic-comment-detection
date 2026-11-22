"""
Tổng hợp phân tích P1 cho TV2 EXP-02 (chỉ đọc artifacts & validation G0, không train, không mở test):
  1. Tổng hợp tuning / ablation LinearSVC (A vs B vs C vs C-calibrated + tuning_results).
  2. Kiểm tra run lineage, MLflow store, SHA-256 hash, reload artifact và đối chiếu 100% predictions.
  3. Phân tích lỗi đại diện trên Validation bàn giao cho EXP-05 (TV3) và metadata bàn giao cho EXP-06 (TV4).

Chạy: python -m src.evaluation.svm_report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.preprocessing import TEENCODE_DICT
from src.models.svm_pipeline import (
    LABEL_NAMES,
    load_g0_train_val,
    load_svm_artifacts,
    predict_from_artifacts,
    sha256_file,
)

NEGATION_WORDS = {"không", "ko", "k", "chẳng", "chả", "đừng", "chưa", "đéo", "éo"}
URL_OR_MENTION_RE = re.compile(r"https?://|www\.|@\w+")
EMOJI_OR_PUNCT_RE = re.compile(r"[!?]{2,}|[\U0001F300-\U0001FAFF]")


def verify_run_lineage(
    variant_dir: Path,
    val_records: list[dict],
    expected_dataset_sha256: str,
    expected_manifest_sha256: str,
    mlflow_client: mlflow.tracking.MlflowClient,
) -> dict[str, Any]:
    """Kiểm tra đầy đủ lineage, schema predictions, probability contract và reload artifact."""
    variant_name = variant_dir.name
    config = json.loads((variant_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((variant_dir / "metrics.json").read_text(encoding="utf-8"))
    run_id_txt = (variant_dir / "run_id.txt").read_text(encoding="utf-8").strip()
    preds = json.loads((variant_dir / "val_predictions.json").read_text(encoding="utf-8"))

    # 1. Hash & run_id consistency
    assert config["run_id"] == run_id_txt, f"{variant_name}: run_id mismatch"
    assert config["dataset_file_sha256"] == expected_dataset_sha256, f"{variant_name}: dataset hash mismatch"
    assert config["manifest_file_sha256"] == expected_manifest_sha256, f"{variant_name}: manifest hash mismatch"

    # 2. MLflow store verification
    ml_run = mlflow_client.get_run(run_id_txt)
    ml_macro_f1 = ml_run.data.metrics.get("val_macro_f1")
    assert ml_macro_f1 is not None and abs(ml_macro_f1 - metrics["macro_f1"]) < 1e-6, (
        f"{variant_name}: MLflow val_macro_f1 mismatch"
    )

    # 3. Prediction IDs & G0 validation alignment (no test rows)
    expected_ids = [r["id"] for r in val_records]
    expected_labels = [int(r["label"]) for r in val_records]
    pred_ids = [p["id"] for p in preds]
    pred_true = [int(p["true_label"]) for p in preds]
    pred_labels = [int(p["predicted_label"]) for p in preds]

    assert len(preds) == len(val_records), f"{variant_name}: prediction count mismatch"
    assert pred_ids == expected_ids, f"{variant_name}: val IDs disagree with G0"
    assert pred_true == expected_labels, f"{variant_name}: true_label disagrees with G0"

    # 4. Probability contract verification
    is_calibrated = bool(config.get("calibrated", False))
    for p in preds:
        assert p["run_id"] == run_id_txt, f"{variant_name}: prediction run_id mismatch"
        if not is_calibrated:
            assert p["probabilities"] is None, (
                f"{variant_name}: uncalibrated LinearSVC must not emit probabilities"
            )
        else:
            proba = p["probabilities"]
            assert isinstance(proba, dict) and set(proba.keys()) == {"CLEAN", "OFFENSIVE", "HATE"}
            total = sum(proba.values())
            assert abs(total - 1.0) < 1e-3, f"{variant_name}: probabilities sum={total}"

    # 5. Reload saved joblib artifacts & verify identical predictions
    artifacts = load_svm_artifacts(variant_dir)
    reloaded_preds = predict_from_artifacts(artifacts, [r["text"] for r in val_records])
    exact_matches = sum(int(a == b) for a, b in zip(reloaded_preds, pred_labels))
    assert exact_matches == len(val_records), (
        f"{variant_name}: reloaded artifact predictions matched {exact_matches}/{len(val_records)}"
    )

    extra_prob_metrics: dict[str, float] = {}
    if is_calibrated:
        proba_mat = np.array(
            [
                [p["probabilities"]["CLEAN"], p["probabilities"]["OFFENSIVE"], p["probabilities"]["HATE"]]
                for p in preds
            ],
            dtype=float,
        )
        proba_mat = np.clip(proba_mat, 1e-12, 1.0)
        proba_mat = proba_mat / proba_mat.sum(axis=1, keepdims=True)
        extra_prob_metrics["val_log_loss"] = round(float(log_loss(expected_labels, proba_mat, labels=[0, 1, 2])), 4)
        brier_per_class = []
        for cls_idx in range(3):
            y_bin = [1 if y == cls_idx else 0 for y in expected_labels]
            brier_per_class.append(brier_score_loss(y_bin, proba_mat[:, cls_idx]))
        extra_prob_metrics["val_brier_macro"] = round(float(np.mean(brier_per_class)), 4)

    return {
        "variant": variant_name,
        "run_id": run_id_txt,
        "mlflow_status": ml_run.info.status,
        "calibrated": is_calibrated,
        "n_val_checked": len(preds),
        "reload_exact_match": f"{exact_matches}/{len(val_records)}",
        "dataset_sha256_verified": True,
        "manifest_sha256_verified": True,
        **extra_prob_metrics,
    }


def summarize_tuning_ablation(tuning_path: Path) -> dict[str, Any]:
    """Tổng hợp tác động của C, class_weight và feature selection k từ tuning_results.json."""
    rows = json.loads(tuning_path.read_text(encoding="utf-8"))

    by_cw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_c: dict[float, list[dict[str, Any]]] = defaultdict(list)
    by_k: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in rows:
        cw_key = str(r["class_weight"]) if r["class_weight"] is not None else "None"
        by_cw[cw_key].append(r)
        by_c[float(r["C"])].append(r)
        k_key = str(r["k_requested"]) if r["k_requested"] is not None else "all(88044)"
        by_k[k_key].append(r)

    def _agg(group: list[dict[str, Any]]) -> dict[str, float]:
        f1s = [x["val_macro_f1"] for x in group]
        times = [x["train_time_s"] for x in group]
        return {
            "count": len(group),
            "best_macro_f1": round(max(f1s), 4),
            "mean_macro_f1": round(float(np.mean(f1s)), 4),
            "mean_train_time_s": round(float(np.mean(times)), 4),
        }

    best_row = max(rows, key=lambda x: x["val_macro_f1"])
    return {
        "total_configs_evaluated": len(rows),
        "best_config": best_row,
        "by_class_weight": {k: _agg(v) for k, v in by_cw.items()},
        "by_C": {str(k): _agg(v) for k, v in sorted(by_c.items())},
        "by_k": {k: _agg(v) for k, v in by_k.items()},
    }


def _tag_error_patterns(text: str) -> list[str]:
    tags: list[str] = []
    tokens = text.lower().split()
    if len(tokens) <= 5:
        tags.append("short_text_le_5_words")
    if any(tok in TEENCODE_DICT for tok in tokens):
        tags.append("contains_teencode")
    if any(tok in NEGATION_WORDS for tok in tokens):
        tags.append("contains_negation")
    if URL_OR_MENTION_RE.search(text):
        tags.append("contains_url_or_mention")
    if EMOJI_OR_PUNCT_RE.search(text):
        tags.append("contains_emoji_or_punct")
    if not tags:
        tags.append("context_or_lexical_nuance")
    return tags


def build_error_handover(
    best_variant_dir: Path,
    val_records: list[dict],
) -> dict[str, Any]:
    """Tổng hợp lỗi validation mà không xuất ID hoặc nội dung bình luận."""
    preds = json.loads((best_variant_dir / "val_predictions.json").read_text(encoding="utf-8"))
    rec_by_id = {r["id"]: r for r in val_records}

    pair_counts: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    total_errors = 0

    for p in preds:
        t_lbl = int(p["true_label"])
        p_lbl = int(p["predicted_label"])
        if t_lbl == p_lbl:
            continue
        total_errors += 1
        pair_key = f"{LABEL_NAMES[t_lbl]} -> {LABEL_NAMES[p_lbl]}"
        pair_counts[pair_key] += 1

        text = rec_by_id[p["id"]]["text"]
        tags = _tag_error_patterns(text)
        if {t_lbl, p_lbl} == {1, 2}:
            tags.append("offensive_hate_boundary")
        pattern_counts.update(tags)

    return {
        "source_variant": best_variant_dir.name,
        "run_id": preds[0]["run_id"] if preds else None,
        "n_val_total": len(preds),
        "n_errors_total": total_errors,
        "error_rate": round(total_errors / max(1, len(preds)), 4),
        "confusion_pair_counts": dict(pair_counts.most_common()),
        "pattern_tag_counts_among_errors": dict(pattern_counts.most_common()),
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    """Xuất báo cáo Markdown cho EXP-02 P1 (ablation, lineage, handover)."""
    lines: list[str] = [
        "# Báo cáo Phân tích & Bàn giao TV2 — EXP-02 (LinearSVC)",
        "",
        f"**Dataset G0:** `data/processed/vihsd_g0_v1/train_val.jsonl` (`n_train={report['dataset']['n_train']}`, `n_val={report['dataset']['n_val']}`)",
        f"**Dataset SHA-256:** `{report['dataset']['dataset_file_sha256']}`",
        f"**Manifest SHA-256:** `{report['dataset']['manifest_file_sha256']}`",
        "",
        "## 1. So sánh hiệu năng & chi phí các biến thể EXP-02",
        "",
        "| Variant | Cấu hình chính | Val Macro-F1 | Weighted-F1 | Balanced Acc | CLEAN F1 | OFFENSIVE F1 | HATE F1 | Features | Train Time | Latency (ms/sample) | Size (MB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for v in report["variants_comparison"]:
        lines.append(
            f"| **{v['variant']}** | {v['summary']} | `{v['macro_f1']:.4f}` | `{v['weighted_f1']:.4f}` | "
            f"`{v['balanced_accuracy']:.4f}` | `{v['CLEAN_f1']:.4f}` | `{v['OFFENSIVE_f1']:.4f}` | "
            f"`{v['HATE_f1']:.4f}` | `{v['n_features_selected']}`/`{v['n_features_raw']}` | "
            f"`{v['train_time_s']:.3f}s` | `{v['inference_ms_per_sample']:.4f}` | `{v['model_size_mb']:.3f}` |"
        )

    ab = report["tuning_ablation"]
    lines.extend(
        [
            "",
            "## 2. Phân tích Ablation & Đánh đổi (`EXP-02-C`, 40 cấu hình)",
            "",
            f"- **Cấu hình tối ưu (chọn theo Val Macro-F1):** `k={ab['best_config']['k_requested']}`, `C={ab['best_config']['C']}`, `class_weight={ab['best_config']['class_weight']}` $\\rightarrow$ `val_macro_f1={ab['best_config']['val_macro_f1']:.4f}` (`train_time_s={ab['best_config']['train_time_s']:.3f}s`).",
            "",
            "### 2.1. Tác động của `class_weight`",
            "| `class_weight` | Số cấu hình | Best Macro-F1 | Mean Macro-F1 | Mean Train Time |",
            "|---|---|---|---|---|",
        ]
    )
    for k, stats in ab["by_class_weight"].items():
        lines.append(
            f"| `{k}` | {stats['count']} | `{stats['best_macro_f1']:.4f}` | `{stats['mean_macro_f1']:.4f}` | `{stats['mean_train_time_s']:.3f}s` |"
        )

    lines.extend(
        [
            "",
            "### 2.2. Tác động của tham số điều chuẩn `C`",
            "| `C` | Số cấu hình | Best Macro-F1 | Mean Macro-F1 | Mean Train Time |",
            "|---|---|---|---|---|",
        ]
    )
    for k, stats in ab["by_C"].items():
        lines.append(
            f"| `{k}` | {stats['count']} | `{stats['best_macro_f1']:.4f}` | `{stats['mean_macro_f1']:.4f}` | `{stats['mean_train_time_s']:.3f}s` |"
        )

    lines.extend(
        [
            "",
            "### 2.3. Tác động của chọn lọc đặc trưng `SelectKBest(chi2)` (`k`)",
            "| `k` | Số cấu hình | Best Macro-F1 | Mean Macro-F1 | Mean Train Time |",
            "|---|---|---|---|---|",
        ]
    )
    for k, stats in ab["by_k"].items():
        lines.append(
            f"| `{k}` | {stats['count']} | `{stats['best_macro_f1']:.4f}` | `{stats['mean_macro_f1']:.4f}` | `{stats['mean_train_time_s']:.3f}s` |"
        )

    lines.extend(
        [
            "",
            "### 2.4. Giới hạn thiết kế của LinearSVC",
            "- Mô hình tuyến tính trên bag-of-n-grams (TF-IDF) không mô hình hóa được thứ tự từ dài, đảo ngữ phức tạp hay ngữ cảnh hội thoại ngoài cửa sổ 1–3 từ.",
            "- `LinearSVC` chỉ tạo lề phân tách (`decision_function`), không sinh xác suất tự nhiên; muốn dùng cho soft voting/API bắt buộc phải bọc `CalibratedClassifierCV` trên train folds.",
            "- `class_weight='balanced'` cải thiện mạnh recall cho `OFFENSIVE`/`HATE` nhưng đánh đổi nhẹ precision của `CLEAN` (nhiều bình luận sạch có từ nhạy cảm bị đẩy sang `OFFENSIVE`/`HATE`).",
            "",
            "## 3. Kiểm chứng Run Lineage & Reproducibility",
            "",
            "| Variant | MLflow Run ID | Status | Val Count | Reload Exact Match | Calibrated | Extra Calibration Metrics |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for l in report["lineage_checks"]:
        extra = "-"
        if l.get("calibrated"):
            extra = f"log_loss={l.get('val_log_loss')}, brier_macro={l.get('val_brier_macro')}"
        lines.append(
            f"| **{l['variant']}** | `{l['run_id']}` | `{l['mlflow_status']}` | `{l['n_val_checked']}` | "
            f"`{l['reload_exact_match']}` | `{l['calibrated']}` | {extra} |"
        )

    ea = report["error_handover_exp05"]
    lines.extend(
        [
            "",
            "## 4. Bàn giao Phân tích Lỗi cho EXP-05 (TV3) & Ensemble cho EXP-06 (TV4)",
            "",
            f"- **Tổng lỗi của `{ea['source_variant']}` trên Validation:** `{ea['n_errors_total']} / {ea['n_val_total']}` (`error_rate={ea['error_rate']:.4f}`).",
            "- **Phân bố cặp nhầm lẫn:**",
        ]
    )
    for pair, cnt in ea["confusion_pair_counts"].items():
        lines.append(f"  - `{pair}`: **{cnt}** mẫu")

    lines.append("- **Nhóm đặc trưng xuất hiện trong các ca lỗi (heuristic):**")
    for tag, cnt in ea["pattern_tag_counts_among_errors"].items():
        lines.append(f"  - `{tag}`: **{cnt}** lần")

    ho = report["ensemble_handover_exp06"]
    lines.extend(
        [
            "",
            "### Metadata bàn giao TV4 (EXP-06)",
            f"- **Hard predictions (LinearSVC tốt nhất):** `{ho['hard_predictions_path']}` (`Macro-F1={ho['hard_macro_f1']:.4f}`).",
            f"- **Calibrated probabilities (Platt scaling `cv=5` chỉ trên train):** `{ho['calibrated_predictions_path']}` (`Macro-F1={ho['calibrated_macro_f1']:.4f}`, `log_loss={ho['calibrated_log_loss']:.4f}`, `brier_macro={ho['calibrated_brier_macro']:.4f}`).",
            f"- **File lỗi đại diện cho TV3 (EXP-05):** `{ho['error_samples_path']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_p1_report(
    baselines_dir: Path,
    data_path: Path,
    manifest_path: Path,
    tracking_uri: str | None = None,
    report_dir: Path | None = None,
    md_out_path: Path | None = None,
) -> dict[str, Any]:
    """Thực thi toàn bộ kiểm tra P1 và xuất báo cáo JSON + Markdown."""
    train_records, val_records, _ = load_g0_train_val(data_path, manifest_path)
    expected_dataset_sha = sha256_file(data_path)
    expected_manifest_sha = sha256_file(manifest_path)

    uri = tracking_uri or (ROOT / "mlruns").resolve().as_uri()
    ml_client = mlflow.tracking.MlflowClient(tracking_uri=uri)

    candidate_names = ["EXP-02-A", "EXP-02-B", "EXP-02-C", "EXP-02-C-calibrated"]
    if (baselines_dir / "EXP-02-D").exists():
        candidate_names.append("EXP-02-D")

    lineage_checks: list[dict[str, Any]] = []
    variants_comparison: list[dict[str, Any]] = []

    summaries = {
        "EXP-02-A": "raw + word(1-2) + LinearSVC(C=1.0)",
        "EXP-02-B": "basic + word+char + LinearSVC(C=1.0)",
        "EXP-02-C": "teencode + word+char + chi2(50k) + LinearSVC(C=0.1, balanced)",
        "EXP-02-C-calibrated": "EXP-02-C + CalibratedClassifierCV(cv=5, sigmoid)",
        "EXP-02-D": "teencode + word+char + chi2(10k) + SVC(kernel=rbf)",
    }

    for name in candidate_names:
        v_dir = baselines_dir / name
        if not v_dir.exists():
            continue
        lineage_checks.append(
            verify_run_lineage(
                variant_dir=v_dir,
                val_records=val_records,
                expected_dataset_sha256=expected_dataset_sha,
                expected_manifest_sha256=expected_manifest_sha,
                mlflow_client=ml_client,
            )
        )
        m = json.loads((v_dir / "metrics.json").read_text(encoding="utf-8"))
        variants_comparison.append(
            {
                "variant": name,
                "summary": summaries.get(name, name),
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "balanced_accuracy": m["balanced_accuracy"],
                "CLEAN_f1": m["per_class"]["CLEAN"]["f1"],
                "OFFENSIVE_f1": m["per_class"]["OFFENSIVE"]["f1"],
                "HATE_f1": m["per_class"]["HATE"]["f1"],
                "n_features_raw": m["n_features_raw"],
                "n_features_selected": m["n_features_selected"],
                "train_time_s": m["train_time_s"],
                "inference_ms_per_sample": m["inference_ms_per_sample"],
                "model_size_mb": m["model_size_mb"],
            }
        )

    tuning_ablation = summarize_tuning_ablation(baselines_dir / "EXP-02-C" / "tuning_results.json")
    error_handover = build_error_handover(baselines_dir / "EXP-02-C", val_records)

    out_dir = report_dir or (baselines_dir / "EXP-02-report")
    out_dir.mkdir(parents=True, exist_ok=True)
    error_path = out_dir / "svm_error_analysis_val.json"
    error_path.write_text(json.dumps(error_handover, ensure_ascii=False, indent=2), encoding="utf-8")

    hard_m = next(x for x in variants_comparison if x["variant"] == "EXP-02-C")
    cal_m = next((x for x in variants_comparison if x["variant"] == "EXP-02-C-calibrated"), None)
    cal_lineage = next((x for x in lineage_checks if x["variant"] == "EXP-02-C-calibrated"), None)

    report: dict[str, Any] = {
        "dataset": {
            "n_train": len(train_records),
            "n_val": len(val_records),
            "dataset_file_sha256": expected_dataset_sha,
            "manifest_file_sha256": expected_manifest_sha,
        },
        "variants_comparison": variants_comparison,
        "tuning_ablation": tuning_ablation,
        "lineage_checks": lineage_checks,
        "error_handover_exp05": error_handover,
        "ensemble_handover_exp06": {
            "hard_predictions_path": "models/baselines/EXP-02-C/val_predictions.json",
            "hard_macro_f1": hard_m["macro_f1"],
            "calibrated_predictions_path": "models/baselines/EXP-02-C-calibrated/val_predictions.json",
            "calibrated_macro_f1": cal_m["macro_f1"] if cal_m else None,
            "calibrated_log_loss": cal_lineage.get("val_log_loss") if cal_lineage else None,
            "calibrated_brier_macro": cal_lineage.get("val_brier_macro") if cal_lineage else None,
            "error_samples_path": "models/baselines/EXP-02-report/svm_error_analysis_val.json",
        },
    }

    (out_dir / "svm_p1_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_target = md_out_path or (ROOT / "docs" / "TV2_EXP02_REPORT.md")
    md_target.write_text(render_markdown_report(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TV2 EXP-02 P1 Analysis, Lineage & Handover Report")
    p.add_argument("--baselines-dir", type=Path, default=ROOT / "models" / "baselines")
    p.add_argument("--data-path", type=Path, default=ROOT / "data" / "processed" / "vihsd_g0_v1" / "train_val.jsonl")
    p.add_argument("--manifest-path", type=Path, default=ROOT / "data" / "processed" / "vihsd_g0_v1" / "manifest.json")
    p.add_argument("--tracking-uri", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = generate_p1_report(
        baselines_dir=args.baselines_dir,
        data_path=args.data_path,
        manifest_path=args.manifest_path,
        tracking_uri=args.tracking_uri,
    )
    print("Lineage checks PASSED:", [x["variant"] for x in report["lineage_checks"]])
    print("Report written to: docs/TV2_EXP02_REPORT.md and models/baselines/EXP-02-report/")


if __name__ == "__main__":
    main()
