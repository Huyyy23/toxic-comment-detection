"""Unit tests for TV4 CNN Phase 4 calibration, error analysis, and charts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.cnn_report import (
    apply_platt_scaling,
    apply_temperature_scaling,
    build_privacy_safe_error_analysis,
    classify_error_tags,
    compute_calibration_metrics,
    discover_tv4_variants,
    evaluate_run_calibration,
    export_executed_notebook,
    fit_platt_scaling,
    fit_temperature_scaling,
    generate_tv4_phase4_report,
    generate_tv4_plots,
    redact_comment_snippet,
    sweep_probability_thresholds,
)
from src.models.svm_binary_pipeline import _build_aggregate_error_summary, compute_binary_metrics


def test_redact_comment_snippet_masks_pii_and_caps_length() -> None:
    raw = (
        "Nghe @NguyenAnh123 noi chuyen voi Tran Minh Tuan o https://example.com/a "
        "sdt 0912345678 email test@mail.com that su qua te hai nhieu lan roi!"
    )
    redacted = redact_comment_snippet(raw, max_len=96)
    assert "@NguyenAnh123" not in redacted
    assert "https://example.com" not in redacted
    assert "0912345678" not in redacted
    assert "test@mail.com" not in redacted
    assert "Tran Minh Tuan" not in redacted
    assert "<USER>" in redacted
    assert "<URL>" in redacted
    assert "<NUM>" in redacted
    assert "<EMAIL>" in redacted
    assert "<NAME>" in redacted
    assert len(redacted) <= 96


def test_classify_error_tags_captures_linguistic_phenomena() -> None:
    assert "contains_teencode" in classify_error_tags("ko biet clgt luon")
    assert "contains_negation" in classify_error_tags("không hề tệ như bạn nghĩ đâu")
    assert "short_text_le_5_words" in classify_error_tags("quá tệ")
    assert "obfuscated_or_repeated_chars" in classify_error_tags("nói đ.m ngu quááá")
    assert "contains_emoji_or_punct" in classify_error_tags("xem cái gì thế???")
    assert classify_error_tags(
        "một câu nhận xét trung tính có độ dài bình thường trên diễn đàn tin tức"
    ) == ["context_or_lexical_nuance"]


def test_temperature_and_platt_calibration_improve_overconfident_probs() -> None:
    rng = np.random.default_rng(7)
    logits = rng.normal(0.0, 1.5, size=600)
    true_probs = 1.0 / (1.0 + np.exp(-(0.6 * logits - 0.8)))
    y_true = (rng.random(600) < true_probs).astype(np.int64)
    # Artificially sharp & biased raw probabilities
    raw_prob = 1.0 / (1.0 + np.exp(-2.2 * logits))

    raw_cal = compute_calibration_metrics(y_true, raw_prob)
    temp_fit = fit_temperature_scaling(raw_prob, y_true)
    p_temp = apply_temperature_scaling(raw_prob, temp_fit["temperature"])
    temp_cal = compute_calibration_metrics(y_true, p_temp)

    platt_fit = fit_platt_scaling(raw_prob, y_true)
    p_platt = apply_platt_scaling(raw_prob, platt_fit["coef"], platt_fit["intercept"])
    platt_cal = compute_calibration_metrics(y_true, p_platt)

    assert temp_fit["temperature"] > 1.0
    assert temp_cal["log_loss"] <= raw_cal["log_loss"] + 1e-6
    assert platt_cal["log_loss"] < raw_cal["log_loss"]
    assert platt_cal["ece"] < raw_cal["ece"]

    sweep = sweep_probability_thresholds(y_true, p_platt)
    assert sweep["in_sample_validation_only"] is True
    assert sweep["best_macro_f1"] >= platt_cal["macro_f1_at_05"] - 1e-6


def _write_synthetic_seed_dir(
    seed_dir: Path,
    val_df: pd.DataFrame,
    variant: str,
    seed: int,
    arch: str,
    train_sha: str,
    val_sha: str,
    rng_seed: int,
) -> dict:
    seed_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(rng_seed)
    n = len(val_df)
    y_true = val_df["label"].to_numpy(dtype=np.int64)
    # Create imperfect probabilities producing >= 35 misclassifications
    base = np.where(y_true == 1, 0.62, 0.35) + rng.normal(0.0, 0.38, size=n)
    p_tox = np.clip(1.0 / (1.0 + np.exp(-3.0 * (base - 0.5))), 1e-4, 1.0 - 1e-4)
    y_pred = (p_tox >= 0.5).astype(np.int64)

    metrics = compute_binary_metrics(
        y_true.tolist(),
        y_pred.tolist(),
        val_df["source"].tolist(),
        scores_or_probs=p_tox,
    )
    metrics.update(
        {
            "loss": 0.41,
            "best_epoch": 3,
            "epochs_completed": 6,
            "raw_softmax_ece": float(
                compute_calibration_metrics(y_true, p_tox)["ece"]
            ),
            "is_calibrated": False,
            "train_time_s": 12.5,
            "inference_ms_per_sample": 0.2,
            "peak_vram_mb": 42.0,
            "num_params": 1024,
            "model_size_mb": 0.5,
            "n_train": 100,
            "n_val": n,
            "device": "cpu",
        }
    )
    run_id = f"run-{variant}-{seed}"
    (seed_dir / "best.pt").write_bytes(b"pt")
    (seed_dir / "last.pt").write_bytes(b"pt")
    (seed_dir / "vocab.json").write_text("{}", encoding="utf-8")
    (seed_dir / "run_id.txt").write_text(run_id, encoding="utf-8")
    (seed_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (seed_dir / "history.json").write_text(
        json.dumps(
            [
                {
                    "epoch": e,
                    "train": {
                        "loss": 0.6 / e,
                        "macro_f1": 0.55 + 0.06 * e,
                        "f1_toxic": 0.3 + 0.07 * e,
                    },
                    "val": {
                        "loss": 0.65 / e,
                        "macro_f1": 0.52 + 0.05 * e,
                        "f1_toxic": 0.28 + 0.06 * e,
                    },
                }
                for e in (1, 2, 3, 4)
            ]
        ),
        encoding="utf-8",
    )
    cfg = {
        "experiment": "TV4-CNN-Binary",
        "variant": variant,
        "seed": seed,
        "run_id": run_id,
        "architecture": arch,
        "use_augmentation": False,
        "data_hashes": {
            "train_parquet_sha256": train_sha,
            "val_parquet_sha256": val_sha,
        },
    }
    (seed_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    pd.DataFrame(
        {
            "sample_id": val_df["id"],
            "source": val_df["source"],
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_CLEAN": np.round(1.0 - p_tox, 6),
            "prob_TOXIC": np.round(p_tox, 6),
        }
    ).to_csv(seed_dir / "predictions_val.csv", index=False)
    val_preds_json = [
        {
            "id": str(sid),
            "source": str(src),
            "true_label": int(yt),
            "predicted_label": int(yp),
            "probabilities": {
                "CLEAN": round(float(1.0 - pt), 6),
                "TOXIC": round(float(pt), 6),
            },
            "is_calibrated": False,
            "threshold": None,
            "run_id": run_id,
        }
        for sid, src, yt, yp, pt in zip(
            val_df["id"], val_df["source"], y_true, y_pred, p_tox
        )
    ]
    (seed_dir / "val_predictions.json").write_text(
        json.dumps(val_preds_json), encoding="utf-8"
    )
    err_sum = _build_aggregate_error_summary(val_df, y_pred.tolist())
    (seed_dir / "error_summary.json").write_text(json.dumps(err_sum), encoding="utf-8")
    return {
        "variant": variant,
        "seed": seed,
        "run_id": run_id,
        "seed_dir": str(seed_dir),
        "metrics": metrics,
    }


def test_discover_error_analysis_and_plots_end_to_end(tmp_path: Path) -> None:
    ds_dir = tmp_path / "data" / "processed" / "binary_n2_v1"
    ds_dir.mkdir(parents=True, exist_ok=True)
    texts = [
        f"bình luận @User{i} https://a.vn/{i} 09123456{i:02d} ko ổn đâu!!"
        if i % 3 == 0
        else (f"ngu vcl đồ đ.m {i}" if i % 2 == 1 else f"bài viết hay quá bạn ơi {i}")
        for i in range(180)
    ]
    labels = [1 if i % 4 == 0 else 0 for i in range(180)]
    sources = ["victsd" if i % 3 == 0 else "vihsd" for i in range(180)]
    val_df = pd.DataFrame(
        {
            "id": [hashlib.sha256(f"id-{i}".encode()).hexdigest() for i in range(180)],
            "text": texts,
            "label": labels,
            "source": sources,
            "split": ["val"] * 180,
        }
    )
    val_df.to_parquet(ds_dir / "val.parquet", index=False)
    train_sha = "train_sha_dummy"
    val_sha = "val_sha_dummy"
    (ds_dir / "dataset_info.json").write_text(
        json.dumps({"output_sha256": {"train.parquet": train_sha, "val.parquet": val_sha}}),
        encoding="utf-8",
    )

    cfg_path = tmp_path / "cnn_config.yaml"
    cfg_path.write_text(
        "variants:\n  CNN-01:\n    description: test\n    hypothesis: hyp\n"
        "  CHAR-01:\n    description: char\n    hypothesis: hyp2\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "models" / "cnn" / "TV4-run-synth"
    r1 = _write_synthetic_seed_dir(
        run_dir / "CNN-01" / "seed_42", val_df, "CNN-01", 42, "textcnn", train_sha, val_sha, 11
    )
    r2 = _write_synthetic_seed_dir(
        run_dir / "CNN-01" / "seed_43", val_df, "CNN-01", 43, "textcnn", train_sha, val_sha, 12
    )
    c1 = _write_synthetic_seed_dir(
        run_dir / "CHAR-01" / "seed_42", val_df, "CHAR-01", 42, "char_cnn", train_sha, val_sha, 13
    )
    summary = [
        {
            "variant": "CNN-01",
            "architecture": "textcnn",
            "preprocessing": "basic",
            "use_augmentation": False,
            "dataset_dir": str(ds_dir),
            "variant_dir": str(run_dir / "CNN-01"),
            "aggregate": {},
            "runs": [r1, r2],
        },
        {
            "variant": "CHAR-01",
            "architecture": "char_cnn",
            "preprocessing": "raw",
            "use_augmentation": False,
            "dataset_dir": str(ds_dir),
            "variant_dir": str(run_dir / "CHAR-01"),
            "aggregate": {},
            "runs": [c1],
        },
    ]
    (run_dir / "TV4-CNN-summary.json").write_text(json.dumps(summary), encoding="utf-8")

    disc = discover_tv4_variants(
        models_dir=tmp_path / "models" / "cnn",
        cnn_config_path=cfg_path,
        root=tmp_path,
        check_mlflow=False,
        verify_reload=False,
    )
    assert disc["n_variants_found"] == 2
    assert disc["all_integrity_checks_passed"] is True

    best_seed_dir = Path(disc["ranking"][0]["best_seed_dir"])
    cal = evaluate_run_calibration(best_seed_dir)
    assert cal["in_sample_validation_only"] is True
    assert cal["raw"]["ece"] >= 0.0

    err = build_privacy_safe_error_analysis(best_seed_dir, val_df, max_examples=35, seed=42)
    assert err["n_representative_examples"] >= 30
    serialized = json.dumps(err["representative_examples"])
    assert "https://a.vn" not in serialized
    assert "@User" not in serialized
    assert "09123456" not in serialized

    plots_dir = tmp_path / "out_plots"
    plots = generate_tv4_plots(disc, [cal], plots_dir)
    assert len(plots) == 5
    for plot_path in plots.values():
        assert Path(plot_path).exists()
        assert Path(plot_path).stat().st_size > 1000

    generate_tv4_phase4_report(
        models_dir=tmp_path / "models" / "cnn",
        out_dir=plots_dir,
        cnn_config_path=cfg_path,
        root=tmp_path,
        verify_reload=False,
        max_error_examples=35,
    )
    nb_path = export_executed_notebook(
        report_dir=plots_dir,
        notebook_path=tmp_path / "notebooks" / "EXP-04-cnn-analysis.ipynb",
    )
    nb_json = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb_json["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 7
    assert all(c["execution_count"] is not None and len(c["outputs"]) > 0 for c in code_cells)
