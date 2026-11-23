"""Phase 6 (N10 / EXP-08) freeze manifest and one-shot locked test evaluation for TV4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.cnn_report import (
    _configure_matplotlib,
    _mean_std,
    apply_platt_scaling,
    discover_tv4_variants,
    fit_platt_scaling,
    sweep_probability_thresholds,
)
from src.evaluation.ensemble_report import (
    TV4_ENSEMBLE_CANDIDATES,
    _find_latest_n6_variant_dir,
    blend_validated_probabilities,
)
from src.features.preprocessing import preprocess_series
from src.features.tfidf_pipeline import transform_texts
from src.models.cnn_pipeline import load_cnn_artifacts, predict_cnn_from_artifacts
from src.models.svm_binary_pipeline import (
    compute_binary_metrics,
    compute_ece,
    load_binary_svm_artifacts,
    run_binary_variant,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CNN_MODELS_DIR = ROOT / "models" / "cnn"
DEFAULT_BASELINES_DIR = ROOT / "models" / "baselines"
DEFAULT_LOCKED_MODELS_DIR = ROOT / "models" / "locked_test" / "TV4-N10"
DEFAULT_LOCKED_OUT_DIR = ROOT / "artifacts" / "TV4-locked-test"
DEFAULT_SEEDS = (42, 43, 44)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_n10_freeze_manifest(
    root: Path = ROOT,
    models_dir: Path = DEFAULT_CNN_MODELS_DIR,
    baselines_dir: Path = DEFAULT_BASELINES_DIR,
    out_dir: Path = DEFAULT_LOCKED_OUT_DIR,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Build and write frozen N10 candidate manifest BEFORE touching test.parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_dir = root / "data" / "processed" / "binary_n2_v1"
    info = json.loads((ds_dir / "dataset_info.json").read_text(encoding="utf-8"))
    sha_map = info.get("output_sha256", {})

    disc = discover_tv4_variants(
        models_dir=models_dir,
        cnn_config_path=root / "configs" / "cnn_config.yaml",
        root=root,
        check_mlflow=False,
        verify_reload=False,
    )
    by_name = disc["variants_by_name"]

    frozen_single_tv4: dict[str, Any] = {}
    for v_name in TV4_ENSEMBLE_CANDIDATES:
        if v_name not in by_name:
            continue
        v_info = by_name[v_name]
        v_dir = Path(v_info["variant_dir"])
        per_seed: dict[str, Any] = {}
        for s in seeds:
            s_dir = v_dir / f"seed_{s}"
            if not s_dir.exists():
                continue
            cfg = json.loads((s_dir / "config.json").read_text(encoding="utf-8"))
            metrics = json.loads((s_dir / "metrics.json").read_text(encoding="utf-8"))
            pred_csv = pd.read_csv(s_dir / "predictions_val.csv")
            y_val = pred_csv["y_true"].to_numpy(dtype=np.int64)
            p_val = pred_csv["prob_TOXIC"].to_numpy(dtype=np.float64)

            platt_fit = fit_platt_scaling(p_val, y_val)
            p_val_platt = apply_platt_scaling(
                p_val, platt_fit["coef"], platt_fit["intercept"]
            )
            sw = sweep_probability_thresholds(y_val, p_val_platt)
            per_seed[str(s)] = {
                "seed": int(s),
                "run_id": cfg["run_id"],
                "seed_dir": str(s_dir),
                "checkpoint_path": str(s_dir / "best.pt"),
                "checkpoint_sha256": _sha256_file(s_dir / "best.pt"),
                "vocab_sha256": cfg.get("vocab_sha256", ""),
                "val_macro_f1_at_050": round(float(metrics["macro_f1"]), 6),
                "val_f1_toxic_at_050": round(float(metrics["f1_toxic"]), 6),
                "val_raw_ece": round(float(metrics["raw_softmax_ece"]), 6),
                "frozen_platt_coef": round(float(platt_fit["coef"]), 8),
                "frozen_platt_intercept": round(float(platt_fit["intercept"]), 8),
                "frozen_platt_val_ece": round(
                    float(compute_ece(y_val, p_val_platt, n_bins=10)), 6
                ),
                "frozen_platt_threshold": float(sw["best_macro_f1_threshold"]),
                "val_macro_f1_at_frozen_platt_thr": round(
                    float(sw["best_macro_f1"]), 6
                ),
            }
        if len(per_seed) == len(seeds):
            frozen_single_tv4[v_name] = {
                "variant": v_name,
                "owner": "TV4",
                "architecture": v_info["architecture"],
                "preprocessing": v_info["preprocessing"],
                "num_params": int(v_info["num_params"]),
                "model_size_mb": float(v_info["model_size_mb"]),
                "val_macro_f1_mean": round(float(v_info["macro_f1"]["mean"]), 6),
                "val_f1_toxic_mean": round(float(v_info["f1_toxic"]["mean"]), 6),
                "best_val_seed": int(v_info["best_seed"]),
                "seeds": per_seed,
            }

    n6_cal_dir = _find_latest_n6_variant_dir(baselines_dir, "EXP-N6-C-calibrated")
    frozen_n6_reference: dict[str, Any] | None = None
    if n6_cal_dir is not None and (n6_cal_dir / f"seed_{seeds[0]}" / "config.json").exists():
        n6_cfg = json.loads(
            (n6_cal_dir / f"seed_{seeds[0]}" / "config.json").read_text(encoding="utf-8")
        )
        n6_m = json.loads(
            (n6_cal_dir / f"seed_{seeds[0]}" / "metrics.json").read_text(encoding="utf-8")
        )
        frozen_n6_reference = {
            "variant": "EXP-N6-C-calibrated",
            "owner": "TV2",
            "seed_dir": str(n6_cal_dir / f"seed_{seeds[0]}"),
            "config": n6_cfg,
            "frozen_default_threshold": 0.50,
            "frozen_val_tuned_threshold": float(n6_cfg.get("threshold", 0.34)),
            "val_macro_f1_at_050": 0.783641,
            "val_macro_f1_at_034": round(float(n6_m.get("macro_f1", 0.7954)), 6),
        }

    # Pre-compute frozen validation thresholds for calibrated ensembles per seed
    val_df = pd.read_parquet(ds_dir / "val.parquet")
    y_val_all = val_df["label"].to_numpy(dtype=np.int64)
    diverse6_weights = {
        "HYBRID-RNN-01": 0.22,
        "HYBRID-RNN-WD": 0.22,
        "CNN-01-WD": 0.18,
        "CNN-04b": 0.16,
        "HYBRID-BILSTM-01": 0.12,
        "CHAR-02": 0.10,
    }
    cross_weights = {
        "EXP-N6-C-calibrated": 0.48,
        "HYBRID-RNN-01": 0.18,
        "HYBRID-RNN-WD": 0.14,
        "CNN-01-WD": 0.10,
        "CNN-04b": 0.05,
        "CHAR-02": 0.05,
    }
    tv4_platt_thr_by_seed: dict[str, float] = {}
    cross_platt_thr_by_seed: dict[str, float] = {}

    for s in seeds:
        s_key = str(s)
        if all(k in frozen_single_tv4 for k in diverse6_weights):
            cal_probs_tv4: dict[str, np.ndarray | None] = {}
            for k in diverse6_weights:
                s_info = frozen_single_tv4[k]["seeds"][s_key]
                p_raw = pd.read_csv(Path(s_info["seed_dir"]) / "predictions_val.csv")[
                    "prob_TOXIC"
                ].to_numpy(dtype=np.float64)
                cal_probs_tv4[k] = apply_platt_scaling(
                    p_raw,
                    float(s_info["frozen_platt_coef"]),
                    float(s_info["frozen_platt_intercept"]),
                )
            p_blend_tv4 = blend_validated_probabilities(cal_probs_tv4, diverse6_weights)
            tv4_platt_thr_by_seed[s_key] = float(
                sweep_probability_thresholds(y_val_all, p_blend_tv4)[
                    "best_macro_f1_threshold"
                ]
            )
            if n6_cal_dir is not None and (
                n6_cal_dir / f"seed_{s}" / "val_predictions.json"
            ).exists():
                vp_n6 = json.loads(
                    (n6_cal_dir / f"seed_{s}" / "val_predictions.json").read_text(
                        encoding="utf-8"
                    )
                )
                cal_probs_cross = dict(cal_probs_tv4)
                cal_probs_cross["EXP-N6-C-calibrated"] = np.array(
                    [float(r["probabilities"]["TOXIC"]) for r in vp_n6],
                    dtype=np.float64,
                )
                p_blend_cross = blend_validated_probabilities(
                    cal_probs_cross,
                    {k: v for k, v in cross_weights.items() if k in cal_probs_cross},
                )
                cross_platt_thr_by_seed[s_key] = float(
                    sweep_probability_thresholds(y_val_all, p_blend_cross)[
                        "best_macro_f1_threshold"
                    ]
                )

    frozen_ensembles = [
        {
            "name": "ENS-TV4-INTRA-4",
            "scope": "intra_tv4",
            "protocol": "frozen_raw_softmax_cutoff_050",
            "weights": {
                "HYBRID-RNN-01": 0.25,
                "HYBRID-RNN-WD": 0.25,
                "CNN-01-WD": 0.25,
                "CNN-04b": 0.25,
            },
            "use_platt_calibrators": False,
            "frozen_threshold_by_seed": {str(s): 0.50 for s in seeds},
        },
        {
            "name": "ENS-TV4-DIVERSE-6",
            "scope": "intra_tv4",
            "protocol": "frozen_raw_softmax_cutoff_050",
            "weights": diverse6_weights,
            "use_platt_calibrators": False,
            "frozen_threshold_by_seed": {str(s): 0.50 for s in seeds},
        },
        {
            "name": "ENS-TV4-FROZEN-PLATT",
            "scope": "intra_tv4",
            "protocol": "frozen_val_platt_and_val_threshold",
            "weights": diverse6_weights,
            "use_platt_calibrators": True,
            "frozen_threshold_by_seed": tv4_platt_thr_by_seed,
        },
    ]
    if frozen_n6_reference is not None:
        frozen_ensembles.extend(
            [
                {
                    "name": "ENS-CROSS-DIVERSE-050",
                    "scope": "cross_team",
                    "protocol": "frozen_raw_softmax_cutoff_050",
                    "weights": cross_weights,
                    "use_platt_calibrators": False,
                    "frozen_threshold_by_seed": {str(s): 0.50 for s in seeds},
                },
                {
                    "name": "ENS-CROSS-FROZEN-PLATT",
                    "scope": "cross_team",
                    "protocol": "frozen_val_platt_and_val_threshold",
                    "weights": cross_weights,
                    "use_platt_calibrators": True,
                    "frozen_threshold_by_seed": cross_platt_thr_by_seed,
                },
            ]
        )

    manifest = {
        "experiment": "TV4-N10-One-Shot-Locked-Test",
        "status": "FROZEN_BEFORE_TEST_READ",
        "data_contract_version": "binary_n2_v1",
        "expected_sha256": {
            "train.parquet": str(sha_map.get("train.parquet", "")),
            "val.parquet": str(sha_map.get("val.parquet", "")),
            "test.parquet": str(sha_map.get("test.parquet", "")),
        },
        "test_blind": bool(info.get("test_blind", False)),
        "historical_exposure_disclosure": (
            "dataset_info.json records test_blind=false due to historical upstream "
            "exposure prior to binary_n2_v1 split freeze. In TV4's D1-D4 pipeline "
            "(Parts 0-5), test.parquet was never opened, read, or used for vocabulary, "
            "training, hyperparameter tuning, calibration, threshold selection, or "
            "ensemble weighting."
        ),
        "protocol_rules": [
            "Evaluate only the pre-declared frozen single models and ensembles.",
            "Do not refit vocabularies, weights, Platt calibrators, or thresholds on test.",
            "Report both default @0.50 and frozen Validation-Platt threshold metrics.",
        ],
        "seeds": list(seeds),
        "frozen_single_tv4": frozen_single_tv4,
        "frozen_n6_reference": frozen_n6_reference,
        "frozen_ensembles": frozen_ensembles,
    }
    manifest_path = out_dir / "tv4_n10_freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _predict_n6_calibrated_probs(
    artifacts: dict[str, Any], texts: list[str]
) -> np.ndarray:
    processed = preprocess_series(
        pd.Series(texts), variant=artifacts["config"]["preprocessing"]
    )
    x_mat = transform_texts(artifacts["tfidf"], processed)
    if artifacts["selector"] is not None:
        x_mat = artifacts["selector"].transform(x_mat)
    proba = artifacts["model"].predict_proba(x_mat)
    pos_col = list(artifacts["model"].classes_).index(1)
    return np.asarray(proba[:, pos_col], dtype=np.float64)


def _evaluate_n6_reference_on_test(
    root: Path,
    frozen_n6: dict[str, Any],
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seeds: list[int],
) -> dict[str, Any] | None:
    """Reproduce or load Train-only EXP-N6-C-calibrated, verify on Val, and evaluate on Test."""
    ds_dir = root / "data" / "processed" / "binary_n2_v1"
    aug_path = root / "data" / "processed" / "binary_n4_v1" / "augmentation.parquet"
    saved_seed_dir = Path(frozen_n6["seed_dir"])
    if not (ds_dir / "train.parquet").exists():
        return None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        if (saved_seed_dir / "model.joblib").exists():
            art_seed_dir = saved_seed_dir
        else:
            res = run_binary_variant(
                variant="C",
                dataset_dir=ds_dir,
                augmentation_path=(
                    aug_path if aug_path.exists() else (ds_dir / "train.parquet")
                ),
                artifacts_root=tmp_path / "n6_repro",
                tracking_uri=(tmp_path / "mlruns").as_uri(),
                seeds=[int(seeds[0])],
                calibrate=True,
                result_type="n10_locked_test_reference",
            )
            art_seed_dir = Path(res["runs"][0]["seed_dir"])

        artifacts = load_binary_svm_artifacts(art_seed_dir)
        val_probs_check = _predict_n6_calibrated_probs(
            artifacts, val_df["text"].astype(str).tolist()
        )
        vp_saved = json.loads(
            (saved_seed_dir / "val_predictions.json").read_text(encoding="utf-8")
        )
        p_saved = np.array(
            [float(r["probabilities"]["TOXIC"]) for r in vp_saved], dtype=np.float64
        )
        val_max_diff = float(np.max(np.abs(val_probs_check - p_saved)))

        t0 = time.perf_counter()
        p_test_tox = _predict_n6_calibrated_probs(
            artifacts, test_df["text"].astype(str).tolist()
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(test_df))

    y_test = test_df["label"].to_numpy(dtype=np.int64)
    sources_test = test_df["source"].astype(str).tolist()
    thr_34 = float(frozen_n6["frozen_val_tuned_threshold"])

    pred_050 = (p_test_tox >= 0.50).astype(np.int64)
    m_050 = compute_binary_metrics(
        y_test.tolist(), pred_050.tolist(), sources_test, scores_or_probs=p_test_tox
    )
    m_050["ece"] = float(compute_ece(y_test, p_test_tox, n_bins=10))
    m_050["threshold"] = 0.50
    m_050["inference_ms_per_sample"] = round(float(elapsed_ms), 4)

    pred_034 = (p_test_tox >= thr_34).astype(np.int64)
    m_034 = compute_binary_metrics(
        y_test.tolist(), pred_034.tolist(), sources_test, scores_or_probs=p_test_tox
    )
    m_034["ece"] = float(compute_ece(y_test, p_test_tox, n_bins=10))
    m_034["threshold"] = thr_34
    m_034["inference_ms_per_sample"] = round(float(elapsed_ms), 4)

    return {
        "variant": "EXP-N6-C-calibrated",
        "owner": "TV2",
        "val_repro_max_prob_diff": round(val_max_diff, 6),
        "prob_toxic_by_seed": {str(s): p_test_tox for s in seeds},
        "at_050": m_050,
        "at_frozen_034": m_034,
    }


def _save_locked_test_predictions(
    out_dir: Path,
    model_name: str,
    seed: int,
    test_ids: list[str],
    sources: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob_toxic: np.ndarray,
    threshold: float,
    is_calibrated: bool,
    metrics: dict[str, Any],
) -> Path:
    seed_dir = out_dir / model_name / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"tv4-n10-{model_name.lower()}-s{seed}"
    pd.DataFrame(
        {
            "sample_id": test_ids,
            "source": sources,
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_CLEAN": np.round(1.0 - prob_toxic, 6),
            "prob_TOXIC": np.round(prob_toxic, 6),
        }
    ).to_csv(seed_dir / "predictions_test.csv", index=False)

    test_json = [
        {
            "id": str(sid),
            "source": str(src),
            "true_label": int(yt),
            "predicted_label": int(yp),
            "probabilities": {
                "CLEAN": round(float(1.0 - pt), 6),
                "TOXIC": round(float(pt), 6),
            },
            "is_calibrated": bool(is_calibrated),
            "threshold": round(float(threshold), 4),
            "run_id": run_id,
        }
        for sid, src, yt, yp, pt in zip(
            test_ids, sources, y_true, y_pred, prob_toxic
        )
    ]
    (seed_dir / "test_predictions.json").write_text(
        json.dumps(test_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (seed_dir / "metrics_test.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return seed_dir


def _fast_binary_f1s(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tp1 = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp1 = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn1 = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn1 = int(len(y_true) - tp1 - fp1 - fn1)
    f1_tox = (2.0 * tp1) / max(2 * tp1 + fp1 + fn1, 1)
    f1_cln = (2.0 * tn1) / max(2 * tn1 + fn1 + fp1, 1)
    return 0.5 * (f1_cln + f1_tox), f1_tox


def _compute_bootstrap_ci_and_mcnemar(
    y_true: np.ndarray,
    preds_by_candidate: dict[str, np.ndarray],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute 95% bootstrap CIs and exact/chi2 McNemar comparisons (D4 / N10)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    y_boot = y_true[boot_indices]

    bootstrap_ci: dict[str, dict[str, Any]] = {}
    for cand, y_hat in preds_by_candidate.items():
        yh_boot = y_hat[boot_indices]
        tp1 = np.sum((y_boot == 1) & (yh_boot == 1), axis=1)
        fp1 = np.sum((y_boot == 0) & (yh_boot == 1), axis=1)
        fn1 = np.sum((y_boot == 1) & (yh_boot == 0), axis=1)
        tn1 = n - tp1 - fp1 - fn1
        f1_tox = (2.0 * tp1) / np.maximum(2 * tp1 + fp1 + fn1, 1)
        f1_cln = (2.0 * tn1) / np.maximum(2 * tn1 + fn1 + fp1, 1)
        macro_f1 = 0.5 * (f1_cln + f1_tox)
        pt_macro, pt_tox = _fast_binary_f1s(y_true, y_hat)
        bootstrap_ci[cand] = {
            "macro_f1_point": round(float(pt_macro), 4),
            "macro_f1_ci95": [
                round(float(np.percentile(macro_f1, 2.5)), 4),
                round(float(np.percentile(macro_f1, 97.5)), 4),
            ],
            "f1_toxic_point": round(float(pt_tox), 4),
            "f1_toxic_ci95": [
                round(float(np.percentile(f1_tox, 2.5)), 4),
                round(float(np.percentile(f1_tox, 97.5)), 4),
            ],
        }

    pair_specs = [
        ("ENS-TV4-INTRA-4", "HYBRID-RNN-01"),
        ("ENS-TV4-FROZEN-PLATT", "HYBRID-RNN-01"),
        ("ENS-TV4-FROZEN-PLATT", "CNN-01"),
        ("ENS-CROSS-DIVERSE-050", "HYBRID-RNN-01"),
        ("ENS-CROSS-FROZEN-PLATT", "HYBRID-RNN-01"),
        ("ENS-CROSS-FROZEN-PLATT", "EXP-N6-C-calibrated@0.50"),
        ("ENS-CROSS-FROZEN-PLATT", "EXP-N6-C-calibrated@0.34"),
        ("HYBRID-RNN-01", "CNN-01-WD"),
    ]
    mcnemar_pairs: list[dict[str, Any]] = []
    for cand_a, cand_b in pair_specs:
        if cand_a not in preds_by_candidate or cand_b not in preds_by_candidate:
            continue
        ya = preds_by_candidate[cand_a]
        yb = preds_by_candidate[cand_b]
        a_ok = ya == y_true
        b_ok = yb == y_true
        b_only_a = int(np.sum(a_ok & (~b_ok)))
        c_only_b = int(np.sum((~a_ok) & b_ok))
        denom = b_only_a + c_only_b
        chi2_cc = (
            ((abs(b_only_a - c_only_b) - 1.0) ** 2) / denom if denom > 0 else 0.0
        )
        p_val = float(math.erfc(math.sqrt(chi2_cc / 2.0))) if denom > 0 else 1.0
        mcnemar_pairs.append(
            {
                "candidate_a": cand_a,
                "candidate_b": cand_b,
                "b_only_a_correct": b_only_a,
                "c_only_b_correct": c_only_b,
                "net_correct_gain_a_minus_b": b_only_a - c_only_b,
                "mcnemar_chi2_cc": round(float(chi2_cc), 4),
                "p_value": round(p_val, 6),
                "significant_at_005": bool(p_val < 0.05),
            }
        )
    return {
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "bootstrap_95ci_seed42": bootstrap_ci,
        "mcnemar_comparisons_seed42": mcnemar_pairs,
    }


def _summarize_3seed_test_runs(
    per_seed: dict[str, dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    seeds = sorted(per_seed.keys(), key=int)
    runs = [per_seed[s] for s in seeds]
    by_source: dict[str, Any] = {}
    for src in ("victsd", "vihsd"):
        by_source[src] = {
            "macro_f1": _mean_std([float(r["by_source"][src]["macro_f1"]) for r in runs]),
            "f1_toxic": _mean_std([float(r["by_source"][src]["f1_toxic"]) for r in runs]),
            "pr_auc": _mean_std(
                [
                    float(r["by_source"][src]["pr_auc"])
                    for r in runs
                    if r["by_source"][src].get("pr_auc") is not None
                ]
                or [0.0]
            ),
        }
    return {
        **meta,
        "seeds": [int(s) for s in seeds],
        "test_macro_f1": _mean_std([float(r["macro_f1"]) for r in runs]),
        "test_f1_toxic": _mean_std([float(r["f1_toxic"]) for r in runs]),
        "test_weighted_f1": _mean_std([float(r["weighted_f1"]) for r in runs]),
        "test_balanced_accuracy": _mean_std([float(r["balanced_accuracy"]) for r in runs]),
        "test_pr_auc": _mean_std([float(r["pr_auc"]) for r in runs]),
        "test_ece": _mean_std([float(r["ece"]) for r in runs]),
        "test_by_source": by_source,
        "per_seed": per_seed,
    }


def _plot_val_vs_test_comparison(
    single_results: list[dict[str, Any]],
    ensemble_results: list[dict[str, Any]],
    n6_res: dict[str, Any] | None,
    out_path: Path,
    plt: Any,
) -> Path:
    rows: list[tuple[str, float, float, float]] = []
    for item in single_results:
        rows.append(
            (
                item["variant"],
                float(item["val_macro_f1_mean"]),
                float(item["at_050"]["test_macro_f1"]["mean"]),
                float(item["at_050"]["test_f1_toxic"]["mean"]),
            )
        )
    if n6_res is not None:
        rows.append(
            (
                "EXP-N6-C-cal@0.50 (TV2)",
                0.7836,
                float(n6_res["at_050"]["macro_f1"]),
                float(n6_res["at_050"]["f1_toxic"]),
            )
        )
        rows.append(
            (
                "EXP-N6-C-cal@0.34 (TV2)",
                0.7954,
                float(n6_res["at_frozen_034"]["macro_f1"]),
                float(n6_res["at_frozen_034"]["f1_toxic"]),
            )
        )
    for ens in ensemble_results:
        rows.append(
            (
                ens["name"],
                float(ens["val_macro_f1_mean"]),
                float(ens["test_macro_f1"]["mean"]),
                float(ens["test_f1_toxic"]["mean"]),
            )
        )

    labels = [r[0] for r in rows]
    val_m = [r[1] for r in rows]
    test_m = [r[2] for r in rows]
    test_t = [r[3] for r in rows]

    y = np.arange(len(labels))
    h = 0.36
    fig, ax = plt.subplots(figsize=(11.8, max(5.8, 0.44 * len(labels) + 1.5)))
    ax.barh(y + h / 2, val_m, height=h, color="#94a3b8", label="Validation Macro-F1")
    bars = ax.barh(
        y - h / 2, test_m, height=h, color="#0f766e", label="Locked Test Macro-F1"
    )
    ax.scatter(
        test_t, y - h / 2, color="#dc2626", zorder=4, s=34, label="Locked Test F1-TOXIC"
    )
    for bar, v_f1, t_f1, tox_f1 in zip(bars, val_m, test_m, test_t):
        gap = t_f1 - v_f1
        ax.text(
            t_f1 + 0.003,
            bar.get_y() + bar.get_height() / 2.0,
            f"Test={t_f1:.4f} (Val={v_f1:.4f}, Δ={gap:+.4f}, TOX={tox_f1:.4f})",
            va="center",
            fontsize=7.8,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlim(0.55, 0.92)
    ax.set_ylim(-2.1, len(labels) - 0.2)
    ax.set_xlabel("Score (3-seed mean)")
    ax.set_title("Phase 6 (N10): Validation vs Locked Test (n=6,048) Generalization Audit")
    ax.legend(loc="lower right", fontsize=8.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_locked_test_confusion_matrices(
    cm_items: list[tuple[str, list[list[int]], float]],
    out_path: Path,
    plt: Any,
) -> Path:
    items = cm_items[:4]
    fig, axes = plt.subplots(1, len(items), figsize=(3.6 * len(items), 3.7))
    if len(items) == 1:
        axes = [axes]
    for ax, (title, cm_list, macro_f1) in zip(axes, items):
        cm = np.asarray(cm_list, dtype=np.int64)
        cm_pct = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cm_pct, cmap="Blues", vmin=0.0, vmax=1.0)
        for r in range(2):
            for c in range(2):
                color = "white" if cm_pct[r, c] > 0.55 else "#0f172a"
                ax.text(
                    c,
                    r,
                    f"{cm[r, c]}\n({cm_pct[r, c] * 100:.1f}%)",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8.5,
                    fontweight="bold",
                )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["CLEAN", "TOXIC"], fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["CLEAN", "TOXIC"], fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title}\nTest Macro-F1={macro_f1:.4f}", fontsize=8.6)
    fig.suptitle("Locked Test Confusion Matrices (n=6,048, Seed 42)")
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_locked_test_source_breakdown(
    source_items: list[tuple[str, dict[str, Any]]],
    out_path: Path,
    plt: Any,
) -> Path:
    labels = [x[0] for x in source_items]
    vih_m = [float(x[1]["vihsd"]["macro_f1"]["mean"]) for x in source_items]
    vic_m = [float(x[1]["victsd"]["macro_f1"]["mean"]) for x in source_items]
    vic_t = [float(x[1]["victsd"]["f1_toxic"]["mean"]) for x in source_items]

    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.bar(x - w, vih_m, width=w, color="#2563eb", label="Test ViHSD Macro-F1")
    ax.bar(x, vic_m, width=w, color="#0f766e", label="Test ViCTSD Macro-F1")
    ax.bar(x + w, vic_t, width=w, color="#f59e0b", label="Test ViCTSD F1-TOXIC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=8.2)
    ax.set_ylim(0.25, 0.94)
    ax.set_ylabel("Locked Test Score (3-seed mean)")
    ax.set_title("Locked Test Domain Generalization: ViHSD vs ViCTSD")
    ax.legend(ncol=3, loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def run_n10_locked_test_evaluation(
    root: Path = ROOT,
    models_dir: Path = DEFAULT_CNN_MODELS_DIR,
    baselines_dir: Path = DEFAULT_BASELINES_DIR,
    locked_models_dir: Path = DEFAULT_LOCKED_MODELS_DIR,
    out_dir: Path = DEFAULT_LOCKED_OUT_DIR,
    device: str = "cpu",
    include_n6_reference: bool = True,
) -> dict[str, Any]:
    """Freeze candidates, verify test.parquet SHA-256, and run one-shot N10 locked test."""
    manifest = build_n10_freeze_manifest(
        root=root,
        models_dir=models_dir,
        baselines_dir=baselines_dir,
        out_dir=out_dir,
    )
    ds_dir = root / "data" / "processed" / "binary_n2_v1"
    test_path = ds_dir / "test.parquet"
    val_path = ds_dir / "val.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing locked test file: {test_path}")

    actual_test_sha = _sha256_file(test_path)
    expected_test_sha = manifest["expected_sha256"]["test.parquet"]
    if expected_test_sha and actual_test_sha != expected_test_sha:
        raise ValueError(
            f"test.parquet SHA-256 mismatch: {actual_test_sha} != {expected_test_sha}"
        )

    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    if set(test_df["split"].unique()) != {"test"}:
        raise ValueError(f"Unexpected split labels in test.parquet: {test_df['split'].unique()}")
    if set(val_df["id"]).intersection(set(test_df["id"])):
        raise ValueError("Leakage detected: ID overlap between val.parquet and test.parquet")

    test_ids = test_df["id"].astype(str).tolist()
    test_texts = test_df["text"].astype(str).tolist()
    y_test = test_df["label"].to_numpy(dtype=np.int64)
    sources_test = test_df["source"].astype(str).tolist()
    seeds: list[int] = manifest["seeds"]

    raw_probs_by_model_seed: dict[str, dict[str, np.ndarray]] = {}
    platt_probs_by_model_seed: dict[str, dict[str, np.ndarray]] = {}
    single_results: list[dict[str, Any]] = []

    for v_name, v_frozen in manifest["frozen_single_tv4"].items():
        raw_probs_by_model_seed[v_name] = {}
        platt_probs_by_model_seed[v_name] = {}
        per_seed_050: dict[str, dict[str, Any]] = {}
        per_seed_platt: dict[str, dict[str, Any]] = {}

        for s_key, s_info in v_frozen["seeds"].items():
            seed_dir = Path(s_info["seed_dir"])
            t0 = time.perf_counter()
            artifacts = load_cnn_artifacts(seed_dir, device=device)
            preds_050, probs_2d = predict_cnn_from_artifacts(artifacts, test_texts)
            infer_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(test_texts))

            p_raw = probs_2d[:, 1].astype(np.float64)
            p_platt = apply_platt_scaling(
                p_raw,
                float(s_info["frozen_platt_coef"]),
                float(s_info["frozen_platt_intercept"]),
            )
            raw_probs_by_model_seed[v_name][s_key] = p_raw
            platt_probs_by_model_seed[v_name][s_key] = p_platt

            y_hat_050 = np.asarray(preds_050, dtype=np.int64)
            m_050 = compute_binary_metrics(
                y_test.tolist(), y_hat_050.tolist(), sources_test, scores_or_probs=p_raw
            )
            m_050["ece"] = float(compute_ece(y_test, p_raw, n_bins=10))
            m_050["threshold"] = 0.50
            m_050["inference_ms_per_sample"] = round(float(infer_ms), 4)
            per_seed_050[s_key] = m_050

            thr_platt = float(s_info["frozen_platt_threshold"])
            y_hat_platt = (p_platt >= thr_platt).astype(np.int64)
            m_platt = compute_binary_metrics(
                y_test.tolist(),
                y_hat_platt.tolist(),
                sources_test,
                scores_or_probs=p_platt,
            )
            m_platt["ece"] = float(compute_ece(y_test, p_platt, n_bins=10))
            m_platt["threshold"] = thr_platt
            m_platt["inference_ms_per_sample"] = round(float(infer_ms), 4)
            per_seed_platt[s_key] = m_platt

            if v_name in ("HYBRID-RNN-01", "CNN-01-WD", "CHAR-02"):
                _save_locked_test_predictions(
                    out_dir=locked_models_dir,
                    model_name=v_name,
                    seed=int(s_key),
                    test_ids=test_ids,
                    sources=sources_test,
                    y_true=y_test,
                    y_pred=y_hat_050,
                    prob_toxic=p_raw,
                    threshold=0.50,
                    is_calibrated=False,
                    metrics=m_050,
                )

        agg_050 = _summarize_3seed_test_runs(per_seed_050, {})
        agg_platt = _summarize_3seed_test_runs(per_seed_platt, {})
        single_results.append(
            {
                "variant": v_name,
                "owner": "TV4",
                "architecture": v_frozen["architecture"],
                "preprocessing": v_frozen["preprocessing"],
                "num_params": v_frozen["num_params"],
                "model_size_mb": v_frozen["model_size_mb"],
                "val_macro_f1_mean": v_frozen["val_macro_f1_mean"],
                "val_f1_toxic_mean": v_frozen["val_f1_toxic_mean"],
                "at_050": agg_050,
                "at_frozen_platt": agg_platt,
                "generalization_gap_macro_f1": round(
                    float(agg_050["test_macro_f1"]["mean"]) - float(v_frozen["val_macro_f1_mean"]),
                    4,
                ),
            }
        )

    n6_res: dict[str, Any] | None = None
    if include_n6_reference and manifest.get("frozen_n6_reference") is not None:
        n6_res = _evaluate_n6_reference_on_test(
            root=root,
            frozen_n6=manifest["frozen_n6_reference"],
            val_df=val_df,
            test_df=test_df,
            seeds=seeds,
        )
        if n6_res is not None:
            raw_probs_by_model_seed["EXP-N6-C-calibrated"] = n6_res["prob_toxic_by_seed"]
            platt_probs_by_model_seed["EXP-N6-C-calibrated"] = n6_res["prob_toxic_by_seed"]

    val_ens_lookup = {
        "ENS-TV4-INTRA-4": 0.7855,
        "ENS-TV4-DIVERSE-6": 0.7859,
        "ENS-TV4-FROZEN-PLATT": 0.7900,
        "ENS-CROSS-DIVERSE-050": 0.7928,
        "ENS-CROSS-FROZEN-PLATT": 0.7989,
    }
    ensemble_results: list[dict[str, Any]] = []
    for ens_spec in manifest["frozen_ensembles"]:
        w_map = ens_spec["weights"]
        source_pool = (
            platt_probs_by_model_seed
            if ens_spec["use_platt_calibrators"]
            else raw_probs_by_model_seed
        )
        if not all(k in source_pool for k in w_map):
            continue
        per_seed_ens: dict[str, dict[str, Any]] = {}
        for s in seeds:
            s_key = str(s)
            m_probs = {k: source_pool[k][s_key] for k in w_map}
            p_blend = blend_validated_probabilities(m_probs, w_map)
            thr = float(ens_spec["frozen_threshold_by_seed"].get(s_key, 0.50))
            y_hat = (p_blend >= thr).astype(np.int64)
            m_ens = compute_binary_metrics(
                y_test.tolist(), y_hat.tolist(), sources_test, scores_or_probs=p_blend
            )
            m_ens["ece"] = float(compute_ece(y_test, p_blend, n_bins=10))
            m_ens["threshold"] = thr
            per_seed_ens[s_key] = m_ens

            _save_locked_test_predictions(
                out_dir=locked_models_dir,
                model_name=ens_spec["name"],
                seed=s,
                test_ids=test_ids,
                sources=sources_test,
                y_true=y_test,
                y_pred=y_hat,
                prob_toxic=p_blend,
                threshold=thr,
                is_calibrated=bool(ens_spec["use_platt_calibrators"]),
                metrics=m_ens,
            )

        val_ref = float(val_ens_lookup.get(ens_spec["name"], 0.7859))
        agg_ens = _summarize_3seed_test_runs(
            per_seed_ens,
            {
                "name": ens_spec["name"],
                "scope": ens_spec["scope"],
                "protocol": ens_spec["protocol"],
                "weights": w_map,
                "val_macro_f1_mean": val_ref,
            },
        )
        agg_ens["generalization_gap_macro_f1"] = round(
            float(agg_ens["test_macro_f1"]["mean"]) - val_ref, 4
        )
        ensemble_results.append(agg_ens)

    plt = _configure_matplotlib()
    s0_key = str(seeds[0])
    cm_items: list[tuple[str, list[list[int]], float]] = []
    for item in single_results:
        if item["variant"] in ("HYBRID-RNN-01", "CNN-01-WD"):
            r_s0 = item["at_050"]["per_seed"][s0_key]
            cm_items.append(
                (
                    f"{item['variant']} @0.50 (s{s0_key})",
                    r_s0["confusion_matrix"],
                    float(r_s0["macro_f1"]),
                )
            )
    for ens in ensemble_results:
        if ens["name"] in ("ENS-TV4-DIVERSE-6", "ENS-CROSS-FROZEN-PLATT"):
            r_s0 = ens["per_seed"][s0_key]
            cm_items.append(
                (
                    f"{ens['name']} (s{s0_key})",
                    r_s0["confusion_matrix"],
                    float(r_s0["macro_f1"]),
                )
            )

    source_items: list[tuple[str, dict[str, Any]]] = [
        (item["variant"], item["at_050"]["test_by_source"])
        for item in single_results
        if item["variant"] in ("HYBRID-RNN-01", "HYBRID-RNN-WD", "CNN-01-WD", "CHAR-02")
    ]
    for ens in ensemble_results:
        source_items.append((ens["name"], ens["test_by_source"]))

    plots = {
        "locked_test_val_vs_test": str(
            _plot_val_vs_test_comparison(
                single_results,
                ensemble_results,
                n6_res,
                out_dir / "locked_test_val_vs_test.png",
                plt,
            )
        ),
        "locked_test_confusion_matrices": str(
            _plot_locked_test_confusion_matrices(
                cm_items, out_dir / "locked_test_confusion_matrices.png", plt
            )
        ),
        "locked_test_source_breakdown": str(
            _plot_locked_test_source_breakdown(
                source_items, out_dir / "locked_test_source_breakdown.png", plt
            )
        ),
    }

    n6_serializable = None
    if n6_res is not None:
        n6_serializable = {
            k: v for k, v in n6_res.items() if k != "prob_toxic_by_seed"
        }

    preds_for_stats: dict[str, np.ndarray] = {}
    for v_name, by_seed in raw_probs_by_model_seed.items():
        if v_name == "EXP-N6-C-calibrated":
            p_s0 = by_seed[s0_key]
            preds_for_stats["EXP-N6-C-calibrated@0.50"] = (p_s0 >= 0.50).astype(
                np.int64
            )
            preds_for_stats["EXP-N6-C-calibrated@0.34"] = (p_s0 >= 0.34).astype(
                np.int64
            )
        elif s0_key in by_seed:
            preds_for_stats[v_name] = (by_seed[s0_key] >= 0.50).astype(np.int64)
    for ens_spec in manifest["frozen_ensembles"]:
        w_map = ens_spec["weights"]
        source_pool = (
            platt_probs_by_model_seed
            if ens_spec["use_platt_calibrators"]
            else raw_probs_by_model_seed
        )
        if all(k in source_pool and s0_key in source_pool[k] for k in w_map):
            p_blend = blend_validated_probabilities(
                {k: source_pool[k][s0_key] for k in w_map}, w_map
            )
            thr = float(ens_spec["frozen_threshold_by_seed"].get(s0_key, 0.50))
            preds_for_stats[ens_spec["name"]] = (p_blend >= thr).astype(np.int64)

    stats_ci_mcnemar = _compute_bootstrap_ci_and_mcnemar(
        y_true=y_test,
        preds_by_candidate=preds_for_stats,
        n_bootstrap=1000,
        seed=42,
    )

    report_payload = {
        "experiment": "TV4-N10-One-Shot-Locked-Test",
        "n_test_samples": len(test_df),
        "test_parquet_sha256": actual_test_sha,
        "test_blind": manifest["test_blind"],
        "historical_exposure_disclosure": manifest["historical_exposure_disclosure"],
        "test_source_counts": {
            str(k): int(v) for k, v in test_df["source"].value_counts().items()
        },
        "test_label_counts": {
            "CLEAN": int((y_test == 0).sum()),
            "TOXIC": int((y_test == 1).sum()),
        },
        "single_models": single_results,
        "n6_reference": n6_serializable,
        "ensembles": ensemble_results,
        "bootstrap_95ci_and_mcnemar": stats_ci_mcnemar,
        "plots": plots,
    }
    (out_dir / "tv4_n10_locked_test_report.json").write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TV4 Phase 6 (N10) one-shot locked test evaluation."
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_CNN_MODELS_DIR)
    parser.add_argument("--baselines-dir", type=Path, default=DEFAULT_BASELINES_DIR)
    parser.add_argument(
        "--locked-models-dir", type=Path, default=DEFAULT_LOCKED_MODELS_DIR
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_LOCKED_OUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-n6-reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rep = run_n10_locked_test_evaluation(
        models_dir=args.models_dir,
        baselines_dir=args.baselines_dir,
        locked_models_dir=args.locked_models_dir,
        out_dir=args.out_dir,
        device=args.device,
        include_n6_reference=not bool(args.skip_n6_reference),
    )
    best_single = max(
        rep["single_models"], key=lambda x: x["at_050"]["test_macro_f1"]["mean"]
    )
    best_ens = max(
        rep["ensembles"], key=lambda x: x["test_macro_f1"]["mean"]
    )
    print(
        f"TV4 N10 Locked Test OK | n_test={rep['n_test_samples']} | "
        f"best_single={best_single['variant']} "
        f"({best_single['at_050']['test_macro_f1']['mean']:.4f}) | "
        f"best_ens={best_ens['name']} ({best_ens['test_macro_f1']['mean']:.4f})"
    )


if __name__ == "__main__":
    main()
