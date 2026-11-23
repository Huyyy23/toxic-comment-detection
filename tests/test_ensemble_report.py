"""Unit tests for TV4 Phase 5 (EXP-06) conditional ensemble and error diversity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ensemble_report import (
    blend_validated_probabilities,
    compute_pairwise_error_diversity,
    generate_tv4_phase5_ensemble_report,
)
from src.models.svm_binary_pipeline import compute_binary_metrics


def test_blend_validated_probabilities_rejects_non_probabilities() -> None:
    p_a = np.array([0.2, 0.8, 0.6], dtype=np.float64)
    p_b = np.array([0.4, 0.6, 0.2], dtype=np.float64)
    blended = blend_validated_probabilities({"A": p_a, "B": p_b}, {"A": 0.5, "B": 0.5})
    assert np.allclose(blended, np.array([0.3, 0.7, 0.4]))

    with pytest.raises(ValueError, match="does not provide valid probabilities"):
        blend_validated_probabilities({"A": p_a, "SVM_RAW": None}, {"A": 0.5, "SVM_RAW": 0.5})

    with pytest.raises(ValueError, match="Negative weight"):
        blend_validated_probabilities({"A": p_a}, {"A": -1.0})


def test_compute_pairwise_error_diversity_complementary_oracle() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    sources = ["vihsd", "victsd", "vihsd", "victsd", "vihsd", "victsd"]
    # Model A wrong on index 0 (0->1); Model B wrong on index 3 (1->0); complementary!
    p_a = np.array([0.8, 0.1, 0.9, 0.85, 0.2, 0.9], dtype=np.float64)
    p_b = np.array([0.1, 0.2, 0.8, 0.20, 0.1, 0.8], dtype=np.float64)
    loaded = {
        "y_true": y_true,
        "sources": sources,
        "candidates": {
            "MODEL_A": {
                "owner": "TV4",
                "by_seed": {
                    42: {
                        "prob_toxic": p_a,
                        "y_pred": (p_a >= 0.5).astype(np.int64),
                    }
                },
            },
            "MODEL_B": {
                "owner": "TV2",
                "by_seed": {
                    42: {
                        "prob_toxic": p_b,
                        "y_pred": (p_b >= 0.5).astype(np.int64),
                    }
                },
            },
        },
    }
    pairs = compute_pairwise_error_diversity(loaded, seed=42)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["disagreement_count"] == 2
    assert pair["only_a_correct"] == 1
    assert pair["only_b_correct"] == 1
    assert pair["both_wrong"] == 0
    assert pair["oracle_macro_f1"] == 1.0


def _write_synth_candidate_seed(
    seed_dir: Path,
    val_df: pd.DataFrame,
    variant: str,
    seed: int,
    arch: str,
    train_sha: str,
    val_sha: str,
    p_tox: np.ndarray | None,
    y_pred: np.ndarray,
    is_calibrated: bool = False,
) -> dict:
    seed_dir.mkdir(parents=True, exist_ok=True)
    y_true = val_df["label"].to_numpy(dtype=np.int64)
    metrics = compute_binary_metrics(
        y_true.tolist(),
        y_pred.tolist(),
        val_df["source"].tolist(),
        scores_or_probs=p_tox,
    )
    metrics.update(
        {
            "loss": 0.35,
            "best_epoch": 2,
            "epochs_completed": 4,
            "raw_softmax_ece": 0.04,
            "is_calibrated": is_calibrated,
            "train_time_s": 10.0,
            "inference_ms_per_sample": 0.05,
            "peak_vram_mb": 64.0,
            "num_params": 1024,
            "model_size_mb": 1.5,
            "n_train": 100,
            "n_val": len(val_df),
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
                    "epoch": 1,
                    "train": {"loss": 0.5, "macro_f1": 0.7, "f1_toxic": 0.5},
                    "val": {"loss": 0.4, "macro_f1": 0.7, "f1_toxic": 0.5},
                }
            ]
        ),
        encoding="utf-8",
    )
    (seed_dir / "error_summary.json").write_text("{}", encoding="utf-8")
    (seed_dir / "config.json").write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    if p_tox is not None:
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
    val_json = [
        {
            "id": str(sid),
            "source": str(src),
            "true_label": int(yt),
            "predicted_label": int(yp),
            "probabilities": (
                {
                    "CLEAN": round(float(1.0 - float(p_tox[idx])), 6),
                    "TOXIC": round(float(p_tox[idx]), 6),
                }
                if p_tox is not None
                else None
            ),
            "is_calibrated": is_calibrated,
            "threshold": 0.34 if is_calibrated else None,
            "run_id": run_id,
        }
        for idx, (sid, src, yt, yp) in enumerate(
            zip(val_df["id"], val_df["source"], y_true, y_pred)
        )
    ]
    (seed_dir / "val_predictions.json").write_text(
        json.dumps(val_json), encoding="utf-8"
    )
    return {
        "variant": variant,
        "seed": seed,
        "run_id": run_id,
        "seed_dir": str(seed_dir),
        "metrics": metrics,
    }


def test_generate_tv4_phase5_ensemble_report_end_to_end(tmp_path: Path) -> None:
    ds_dir = tmp_path / "data" / "processed" / "binary_n2_v1"
    ds_dir.mkdir(parents=True, exist_ok=True)
    n = 120
    y_true = np.array([1 if i % 4 == 0 else 0 for i in range(n)], dtype=np.int64)
    val_df = pd.DataFrame(
        {
            "id": [hashlib.sha256(f"s-{i}".encode()).hexdigest() for i in range(n)],
            "text": [f"binh luan {i}" for i in range(n)],
            "label": y_true,
            "source": ["victsd" if i % 3 == 0 else "vihsd" for i in range(n)],
            "split": ["val"] * n,
        }
    )
    val_df.to_parquet(ds_dir / "val.parquet", index=False)
    train_sha, val_sha = "sha_tr", "sha_va"
    (ds_dir / "dataset_info.json").write_text(
        json.dumps({"output_sha256": {"train.parquet": train_sha, "val.parquet": val_sha}}),
        encoding="utf-8",
    )

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        ("HYBRID-RNN-01", "bigru_char_cnn"),
        ("HYBRID-RNN-WD", "bigru_char_cnn"),
        ("CNN-01-WD", "textcnn"),
        ("CNN-01", "textcnn"),
        ("CNN-04b", "textcnn"),
        ("HYBRID-BILSTM-01", "bilstm_char_cnn"),
        ("CHAR-02", "char_cnn"),
    ]
    (cfg_dir / "cnn_config.yaml").write_text(
        "variants:\n"
        + "".join(f"  {v}:\n    description: d\n    hypothesis: h\n" for v, _ in variants),
        encoding="utf-8",
    )

    run_dir = tmp_path / "models" / "cnn" / "TV4-run-1"
    summary = []
    for idx_v, (v_name, arch) in enumerate(variants):
        runs = []
        for s in (42, 43, 44):
            rng = np.random.default_rng(100 + idx_v * 10 + s)
            p_t = np.clip(
                np.where(y_true == 1, 0.72, 0.26) + rng.normal(0.0, 0.22, size=n),
                0.02,
                0.98,
            )
            y_p = (p_t >= 0.5).astype(np.int64)
            runs.append(
                _write_synth_candidate_seed(
                    run_dir / v_name / f"seed_{s}",
                    val_df,
                    v_name,
                    s,
                    arch,
                    train_sha,
                    val_sha,
                    p_t,
                    y_p,
                )
            )
        summary.append(
            {
                "variant": v_name,
                "architecture": arch,
                "preprocessing": "raw",
                "use_augmentation": False,
                "dataset_dir": str(ds_dir),
                "variant_dir": str(run_dir / v_name),
                "aggregate": {},
                "runs": runs,
            }
        )
    (run_dir / "TV4-CNN-summary.json").write_text(json.dumps(summary), encoding="utf-8")

    b_dir = tmp_path / "models" / "baselines"
    for s in (42, 43, 44):
        rng = np.random.default_rng(500 + s)
        p_t = np.clip(
            np.where(y_true == 1, 0.76, 0.22) + rng.normal(0.0, 0.18, size=n),
            0.02,
            0.98,
        )
        y_p = (p_t >= 0.34).astype(np.int64)
        _write_synth_candidate_seed(
            b_dir / "N6-run-cal" / "EXP-N6-C-calibrated" / f"seed_{s}",
            val_df,
            "EXP-N6-C-calibrated",
            s,
            "svm",
            train_sha,
            val_sha,
            p_t,
            y_p,
            is_calibrated=True,
        )
        _write_synth_candidate_seed(
            b_dir / "N6-run-raw" / "EXP-N6-C" / f"seed_{s}",
            val_df,
            "EXP-N6-C",
            s,
            "svm",
            train_sha,
            val_sha,
            None,
            y_p,
            is_calibrated=False,
        )

    out_dir = tmp_path / "artifacts" / "TV4-ensemble-analysis"
    ens_models_dir = tmp_path / "models" / "ensemble" / "TV4-EXP06"
    rep = generate_tv4_phase5_ensemble_report(
        root=tmp_path,
        models_dir=tmp_path / "models" / "cnn",
        baselines_dir=b_dir,
        ensemble_models_dir=ens_models_dir,
        out_dir=out_dir,
    )
    assert rep["n_val_samples"] == n
    assert len(rep["ensembles"]) >= 8
    assert (out_dir / "tv4_phase5_ensemble_report.json").exists()
    for p_path in rep["plots"].values():
        assert Path(p_path).exists()
        assert Path(p_path).stat().st_size > 1000
    assert (ens_models_dir / "ENS-CROSS-OOF-PLATT" / "seed_42" / "val_predictions.json").exists()
