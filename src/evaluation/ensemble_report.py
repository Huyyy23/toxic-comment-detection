"""Phase 5 (EXP-06) error diversity, conditional probability voting, and ADR audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

from src.evaluation.cnn_report import (
    _configure_matplotlib,
    _mean_std,
    apply_platt_scaling,
    clip_probs,
    discover_tv4_variants,
    fit_platt_scaling,
    load_expected_dataset_hashes,
    probs_to_logits,
    sweep_probability_thresholds,
)
from src.models.svm_binary_pipeline import compute_binary_metrics, compute_ece

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CNN_MODELS_DIR = ROOT / "models" / "cnn"
DEFAULT_BASELINES_DIR = ROOT / "models" / "baselines"
DEFAULT_ENSEMBLE_MODELS_DIR = ROOT / "models" / "ensemble" / "TV4-EXP06"
DEFAULT_ENSEMBLE_OUT_DIR = ROOT / "artifacts" / "TV4-ensemble-analysis"
DEFAULT_SEEDS = (42, 43, 44)
ADR_MIN_MACRO_F1_GAIN = 0.0100

TV4_ENSEMBLE_CANDIDATES = (
    "HYBRID-RNN-01",
    "HYBRID-RNN-WD",
    "CNN-01-WD",
    "CNN-01",
    "CNN-04b",
    "HYBRID-BILSTM-01",
    "CHAR-02",
)


def _find_latest_n6_variant_dir(baselines_dir: Path, folder_name: str) -> Path | None:
    matches = sorted(
        baselines_dir.glob(f"N6-run-*/{folder_name}"),
        key=lambda p: p.stat().st_mtime,
    )
    return matches[-1] if matches else None


def _load_val_predictions_json(
    seed_dir: Path, expected_ids: list[str]
) -> dict[str, Any]:
    vp_path = seed_dir / "val_predictions.json"
    metrics_path = seed_dir / "metrics.json"
    cfg_path = seed_dir / "config.json"
    rows = json.loads(vp_path.read_text(encoding="utf-8"))
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else {}
    )
    cfg = (
        json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg_path.exists()
        else {}
    )
    ids = [str(r["id"]) for r in rows]
    if ids != expected_ids:
        raise ValueError(f"Validation ID order mismatch in {vp_path}")
    y_true = np.array([int(r["true_label"]) for r in rows], dtype=np.int64)
    y_pred = np.array([int(r["predicted_label"]) for r in rows], dtype=np.int64)
    sources = [str(r["source"]) for r in rows]
    has_probs = rows[0].get("probabilities") is not None
    prob_toxic: np.ndarray | None = None
    if has_probs:
        for r in rows:
            p_c = float(r["probabilities"]["CLEAN"])
            p_t = float(r["probabilities"]["TOXIC"])
            if abs(p_c + p_t - 1.0) > 1e-3:
                raise ValueError(f"Invalid probability sum in {vp_path}: {p_c + p_t}")
        prob_toxic = np.array(
            [float(r["probabilities"]["TOXIC"]) for r in rows], dtype=np.float64
        )
    return {
        "seed_dir": str(seed_dir),
        "run_id": str(rows[0].get("run_id", cfg.get("run_id", ""))),
        "is_calibrated": bool(rows[0].get("is_calibrated", False)),
        "threshold": rows[0].get("threshold"),
        "has_probabilities": bool(has_probs),
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_toxic": prob_toxic,
        "sources": sources,
        "metrics": metrics,
        "config": cfg,
    }


def load_candidate_validation_predictions(
    root: Path = ROOT,
    models_dir: Path = DEFAULT_CNN_MODELS_DIR,
    baselines_dir: Path = DEFAULT_BASELINES_DIR,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Load and verify aligned binary validation predictions across TV2 and TV4."""
    val_path = root / "data" / "processed" / "binary_n2_v1" / "val.parquet"
    val_df = pd.read_parquet(val_path)
    expected_ids = val_df["id"].astype(str).tolist()
    y_true = val_df["label"].to_numpy(dtype=np.int64)
    sources = val_df["source"].astype(str).tolist()
    expected_hashes = load_expected_dataset_hashes(root=root).get("n2", {})

    disc = discover_tv4_variants(
        models_dir=models_dir,
        cnn_config_path=root / "configs" / "cnn_config.yaml",
        root=root,
        check_mlflow=False,
        verify_reload=False,
    )
    by_name = disc["variants_by_name"]

    candidates: dict[str, dict[str, Any]] = {}
    for v_name in TV4_ENSEMBLE_CANDIDATES:
        if v_name not in by_name:
            continue
        v_info = by_name[v_name]
        v_dir = Path(v_info["variant_dir"])
        by_seed: dict[int, dict[str, Any]] = {}
        for s in seeds:
            s_dir = v_dir / f"seed_{s}"
            if s_dir.exists():
                by_seed[int(s)] = _load_val_predictions_json(s_dir, expected_ids)
        if len(by_seed) == len(seeds):
            candidates[v_name] = {
                "name": v_name,
                "owner": "TV4",
                "family": v_info["architecture"],
                "has_probabilities": True,
                "is_calibrated": False,
                "model_size_mb": float(v_info["model_size_mb"]),
                "num_params": int(v_info["num_params"]),
                "inference_ms_per_sample": float(
                    v_info["inference_ms_per_sample"]["mean"]
                ),
                "by_seed": by_seed,
            }

    n6_cal_dir = _find_latest_n6_variant_dir(baselines_dir, "EXP-N6-C-calibrated")
    if n6_cal_dir is not None:
        by_seed_n6c: dict[int, dict[str, Any]] = {}
        for s in seeds:
            s_dir = n6_cal_dir / f"seed_{s}"
            if s_dir.exists():
                by_seed_n6c[int(s)] = _load_val_predictions_json(s_dir, expected_ids)
        if len(by_seed_n6c) == len(seeds):
            m0 = by_seed_n6c[int(seeds[0])]["metrics"]
            candidates["EXP-N6-C-calibrated"] = {
                "name": "EXP-N6-C-calibrated",
                "owner": "TV2",
                "family": "tfidf_linearsvc_calibrated_cv3",
                "has_probabilities": True,
                "is_calibrated": True,
                "model_size_mb": float(m0.get("model_size_mb", 7.82)),
                "num_params": 75000,
                "inference_ms_per_sample": float(
                    m0.get("inference_ms_per_sample", 0.085)
                ),
                "by_seed": by_seed_n6c,
            }

    n6_raw_dir = _find_latest_n6_variant_dir(baselines_dir, "EXP-N6-C")
    if n6_raw_dir is not None:
        by_seed_n6: dict[int, dict[str, Any]] = {}
        for s in seeds:
            s_dir = n6_raw_dir / f"seed_{s}"
            if s_dir.exists():
                by_seed_n6[int(s)] = _load_val_predictions_json(s_dir, expected_ids)
        if len(by_seed_n6) == len(seeds):
            m0 = by_seed_n6[int(seeds[0])]["metrics"]
            candidates["EXP-N6-C-uncal"] = {
                "name": "EXP-N6-C-uncal",
                "owner": "TV2",
                "family": "tfidf_linearsvc_margin",
                "has_probabilities": False,
                "is_calibrated": False,
                "model_size_mb": float(m0.get("model_size_mb", 3.21)),
                "num_params": 75000,
                "inference_ms_per_sample": float(
                    m0.get("inference_ms_per_sample", 0.045)
                ),
                "by_seed": by_seed_n6,
            }

    return {
        "n_val_samples": len(expected_ids),
        "val_ids": expected_ids,
        "y_true": y_true,
        "sources": sources,
        "expected_hashes": expected_hashes,
        "seeds": list(seeds),
        "candidates": candidates,
        "other_members_status": {
            "TV1_EXP01_LR": (
                "models/baselines/EXP-01-A/B/C are on historical 3-class vihsd.jsonl "
                "(n=2,672); binary D1-D4 LR predictions not yet checked into models/baselines/."
            ),
            "TV3_N5_FastText": (
                "Documented in docs/N5_TOXIC_TUNING_REPORT.md (Val Macro-F1 0.6222 baseline / "
                "0.6615 threshold-tuned, >11% below N6-C & TV4); local prediction files "
                "not checked into models/baselines/."
            ),
            "TV3_EXP03_Recurrent": (
                "models/recurrent_official/EXP-03 contains G0 placeholder; TV4's N7 "
                "HYBRID-RNN-01 (BiGRU+Char-CNN) & HYBRID-BILSTM-01 (BiLSTM+Char-CNN) "
                "provide the recurrent+char binary predictions."
            ),
        },
    }


def compute_pairwise_error_diversity(
    loaded: dict[str, Any], seed: int = 42
) -> list[dict[str, Any]]:
    """Compute pairwise error diversity, double-fault, kappa, and oracle upper bounds."""
    y_true = loaded["y_true"]
    sources = loaded["sources"]
    cands = loaded["candidates"]
    names = list(cands.keys())
    n_total = len(y_true)
    pairs: list[dict[str, Any]] = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            run_a = cands[name_a]["by_seed"][seed]
            run_b = cands[name_b]["by_seed"][seed]

            pred_a = (
                (run_a["prob_toxic"] >= 0.5).astype(np.int64)
                if run_a["prob_toxic"] is not None
                else run_a["y_pred"]
            )
            pred_b = (
                (run_b["prob_toxic"] >= 0.5).astype(np.int64)
                if run_b["prob_toxic"] is not None
                else run_b["y_pred"]
            )

            corr_a = pred_a == y_true
            corr_b = pred_b == y_true
            disagree = int(np.sum(pred_a != pred_b))
            only_a = int(np.sum(corr_a & (~corr_b)))
            only_b = int(np.sum((~corr_a) & corr_b))
            both_wrong = int(np.sum((~corr_a) & (~corr_b)))
            both_right = int(np.sum(corr_a & corr_b))

            oracle_pred = np.where(corr_a | corr_b, y_true, pred_a)
            oracle_m = compute_binary_metrics(
                y_true.tolist(), oracle_pred.tolist(), sources
            )
            kappa = float(cohen_kappa_score(pred_a, pred_b))

            prob_corr: float | None = None
            if run_a["prob_toxic"] is not None and run_b["prob_toxic"] is not None:
                prob_corr = float(
                    np.corrcoef(run_a["prob_toxic"], run_b["prob_toxic"])[0, 1]
                )

            pairs.append(
                {
                    "model_a": name_a,
                    "model_b": name_b,
                    "owner_pair": f"{cands[name_a]['owner']}+{cands[name_b]['owner']}",
                    "seed": int(seed),
                    "n_samples": n_total,
                    "disagreement_count": disagree,
                    "disagreement_rate": round(float(disagree / n_total), 4),
                    "only_a_correct": only_a,
                    "only_b_correct": only_b,
                    "complementary_gain_potential": only_a + only_b,
                    "both_wrong": both_wrong,
                    "double_fault_rate": round(float(both_wrong / n_total), 4),
                    "both_correct": both_right,
                    "cohen_kappa": round(kappa, 4),
                    "prob_pearson_corr": (
                        round(prob_corr, 4) if prob_corr is not None else None
                    ),
                    "oracle_macro_f1": round(float(oracle_m["macro_f1"]), 4),
                    "oracle_f1_toxic": round(float(oracle_m["f1_toxic"]), 4),
                }
            )
    return pairs


def blend_validated_probabilities(
    model_probs: dict[str, np.ndarray | None],
    weights: dict[str, float],
) -> np.ndarray:
    """Compute weighted probability average, rejecting uncalibrated non-probability inputs."""
    if not weights:
        raise ValueError("weights cannot be empty")
    for name, w in weights.items():
        if w < 0:
            raise ValueError(f"Negative weight not allowed for {name}: {w}")
    total_w = float(sum(weights.values()))
    if total_w <= 0:
        raise ValueError("weights sum must be positive")
    blended: np.ndarray | None = None
    for name, w in weights.items():
        if name not in model_probs or model_probs[name] is None:
            raise ValueError(
                f"Model {name!r} does not provide valid probabilities for Soft/Weighted Voting"
            )
        p = clip_probs(model_probs[name])
        term = (float(w) / total_w) * p
        blended = term if blended is None else (blended + term)
    assert blended is not None
    return np.clip(blended, 1e-7, 1.0 - 1e-7)


def _evaluate_seed_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob_toxic: np.ndarray,
    sources: list[str],
    threshold: float,
    is_calibrated: bool,
    in_sample_tuned: bool,
) -> dict[str, Any]:
    m = compute_binary_metrics(
        y_true.tolist(),
        y_pred.tolist(),
        sources,
        scores_or_probs=prob_toxic,
    )
    m["ece"] = float(compute_ece(y_true, prob_toxic, n_bins=10))
    m["threshold"] = float(threshold)
    m["is_calibrated"] = bool(is_calibrated)
    m["in_sample_validation_only"] = bool(in_sample_tuned)
    return m


def _aggregate_seed_metrics(
    seed_results: dict[int, dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    seeds = sorted(seed_results.keys())
    runs = [seed_results[s] for s in seeds]
    best_s = max(seeds, key=lambda s: float(seed_results[s]["macro_f1"]))
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
    pr_list = [float(r["pr_auc"]) for r in runs if r.get("pr_auc") is not None]
    return {
        **meta,
        "seeds": seeds,
        "macro_f1": _mean_std([float(r["macro_f1"]) for r in runs]),
        "f1_toxic": _mean_std([float(r["f1_toxic"]) for r in runs]),
        "weighted_f1": _mean_std([float(r["weighted_f1"]) for r in runs]),
        "balanced_accuracy": _mean_std([float(r["balanced_accuracy"]) for r in runs]),
        "pr_auc": _mean_std(pr_list if pr_list else [0.0]),
        "ece": _mean_std([float(r["ece"]) for r in runs]),
        "by_source": by_source,
        "best_seed": int(best_s),
        "best_seed_macro_f1": float(seed_results[best_s]["macro_f1"]),
        "best_seed_f1_toxic": float(seed_results[best_s]["f1_toxic"]),
        "per_seed": {str(s): seed_results[s] for s in seeds},
    }


def _run_oof_platt_voting_seed(
    y_true: np.ndarray,
    model_probs: dict[str, np.ndarray],
    weights: dict[str, float],
    already_calibrated: set[str],
    seed: int,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_prob = np.zeros(len(y_true), dtype=np.float64)
    oof_pred = np.zeros(len(y_true), dtype=np.int64)
    fold_thresholds: list[float] = []

    for tr_idx, te_idx in skf.split(y_true, y_true):
        tr_cal: dict[str, np.ndarray | None] = {}
        te_cal: dict[str, np.ndarray | None] = {}
        for name in weights:
            arr = model_probs[name]
            if name in already_calibrated:
                tr_cal[name] = arr[tr_idx]
                te_cal[name] = arr[te_idx]
            else:
                fit = fit_platt_scaling(arr[tr_idx], y_true[tr_idx])
                tr_cal[name] = apply_platt_scaling(
                    arr[tr_idx], fit["coef"], fit["intercept"]
                )
                te_cal[name] = apply_platt_scaling(
                    arr[te_idx], fit["coef"], fit["intercept"]
                )
        tr_blend = blend_validated_probabilities(tr_cal, weights)
        te_blend = blend_validated_probabilities(te_cal, weights)
        thr = sweep_probability_thresholds(y_true[tr_idx], tr_blend)[
            "best_macro_f1_threshold"
        ]
        fold_thresholds.append(float(thr))
        oof_prob[te_idx] = te_blend
        oof_pred[te_idx] = (te_blend >= thr).astype(np.int64)

    return oof_pred, oof_prob, float(np.mean(fold_thresholds))


def _run_oof_stacking_lr_seed(
    y_true: np.ndarray,
    model_probs: dict[str, np.ndarray],
    feature_models: list[str],
    seed: int,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    x_logits = np.column_stack([probs_to_logits(model_probs[m]) for m in feature_models])
    oof_prob = np.zeros(len(y_true), dtype=np.float64)
    oof_pred = np.zeros(len(y_true), dtype=np.int64)
    fold_thresholds: list[float] = []

    for tr_idx, te_idx in skf.split(x_logits, y_true):
        meta_lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        meta_lr.fit(x_logits[tr_idx], y_true[tr_idx])
        tr_p = meta_lr.predict_proba(x_logits[tr_idx])[:, 1]
        te_p = meta_lr.predict_proba(x_logits[te_idx])[:, 1]
        thr = sweep_probability_thresholds(y_true[tr_idx], tr_p)[
            "best_macro_f1_threshold"
        ]
        fold_thresholds.append(float(thr))
        oof_prob[te_idx] = te_p
        oof_pred[te_idx] = (te_p >= thr).astype(np.int64)

    return oof_pred, oof_prob, float(np.mean(fold_thresholds))


def _save_ensemble_seed_predictions(
    out_dir: Path,
    ensemble_name: str,
    seed: int,
    val_ids: list[str],
    sources: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob_toxic: np.ndarray,
    threshold: float,
    is_calibrated: bool,
    metrics: dict[str, Any],
    weights: dict[str, float],
    protocol: str,
) -> Path:
    seed_dir = out_dir / ensemble_name / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"tv4-exp06-{ensemble_name.lower()}-s{seed}"
    pd.DataFrame(
        {
            "sample_id": val_ids,
            "source": sources,
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_CLEAN": np.round(1.0 - prob_toxic, 6),
            "prob_TOXIC": np.round(prob_toxic, 6),
        }
    ).to_csv(seed_dir / "predictions_val.csv", index=False)

    val_json = [
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
            val_ids, sources, y_true, y_pred, prob_toxic
        )
    ]
    (seed_dir / "val_predictions.json").write_text(
        json.dumps(val_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (seed_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (seed_dir / "config.json").write_text(
        json.dumps(
            {
                "experiment": "TV4-EXP06-Ensemble",
                "ensemble_name": ensemble_name,
                "seed": int(seed),
                "run_id": run_id,
                "protocol": protocol,
                "weights": weights,
                "threshold": round(float(threshold), 4),
                "is_calibrated": bool(is_calibrated),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return seed_dir


def evaluate_ensemble_protocols(
    loaded: dict[str, Any],
    ensemble_models_dir: Path = DEFAULT_ENSEMBLE_MODELS_DIR,
) -> dict[str, Any]:
    """Evaluate single baselines, Intra-TV4 voting, Cross-Team voting, and OOF protocols."""
    y_true = loaded["y_true"]
    sources = loaded["sources"]
    val_ids = loaded["val_ids"]
    seeds: list[int] = loaded["seeds"]
    cands = loaded["candidates"]

    single_baselines: list[dict[str, Any]] = []
    for name, c_info in cands.items():
        seed_res: dict[int, dict[str, Any]] = {}
        for s in seeds:
            run = c_info["by_seed"][s]
            if run["prob_toxic"] is not None:
                p_tox = run["prob_toxic"]
                pred_05 = (p_tox >= 0.5).astype(np.int64)
                seed_res[s] = _evaluate_seed_predictions(
                    y_true, pred_05, p_tox, sources, 0.50, c_info["is_calibrated"], False
                )
            else:
                m = compute_binary_metrics(
                    y_true.tolist(), run["y_pred"].tolist(), sources
                )
                m["ece"] = 0.0
                m["threshold"] = 0.0
                m["is_calibrated"] = False
                m["in_sample_validation_only"] = False
                seed_res[s] = m
        single_baselines.append(
            _aggregate_seed_metrics(
                seed_res,
                {
                    "name": name,
                    "owner": c_info["owner"],
                    "kind": "single_baseline",
                    "protocol": "single_default_050_or_margin",
                    "model_size_mb": c_info["model_size_mb"],
                    "num_params": c_info["num_params"],
                    "inference_ms_per_sample": c_info["inference_ms_per_sample"],
                },
            )
        )

    if "EXP-N6-C-calibrated" in cands:
        c_info = cands["EXP-N6-C-calibrated"]
        seed_res_34: dict[int, dict[str, Any]] = {}
        for s in seeds:
            run = c_info["by_seed"][s]
            p_tox = run["prob_toxic"]
            thr = float(run["threshold"] or 0.34)
            pred_thr = (p_tox >= thr).astype(np.int64)
            seed_res_34[s] = _evaluate_seed_predictions(
                y_true, pred_thr, p_tox, sources, thr, True, True
            )
        single_baselines.append(
            _aggregate_seed_metrics(
                seed_res_34,
                {
                    "name": "EXP-N6-C-calibrated@0.34 (in-sample thr)",
                    "owner": "TV2",
                    "kind": "single_baseline_in_sample_thr",
                    "protocol": "in_sample_threshold_diagnostic",
                    "model_size_mb": c_info["model_size_mb"],
                    "num_params": c_info["num_params"],
                    "inference_ms_per_sample": c_info["inference_ms_per_sample"],
                },
            )
        )

    ensemble_specs: list[dict[str, Any]] = [
        {
            "name": "ENS-TV4-INTRA-3",
            "scope": "intra_tv4",
            "protocol": "fixed_cutoff_050",
            "description": (
                "Soft Voting 3 mô hình TV4 "
                "(HYBRID-RNN-01 + HYBRID-RNN-WD + CNN-01-WD) @0.50"
            ),
            "weights": {
                "HYBRID-RNN-01": 1.0 / 3.0,
                "HYBRID-RNN-WD": 1.0 / 3.0,
                "CNN-01-WD": 1.0 / 3.0,
            },
        },
        {
            "name": "ENS-TV4-INTRA-4",
            "scope": "intra_tv4",
            "protocol": "fixed_cutoff_050",
            "description": "Soft Voting 4 mô hình TV4 (+CNN-04b multi-kernel [3,4,5]) @0.50",
            "weights": {
                "HYBRID-RNN-01": 0.25,
                "HYBRID-RNN-WD": 0.25,
                "CNN-01-WD": 0.25,
                "CNN-04b": 0.25,
            },
        },
        {
            "name": "ENS-TV4-DIVERSE-6",
            "scope": "intra_tv4",
            "protocol": "fixed_cutoff_050",
            "description": (
                "Weighted Soft Voting 6 mô hình TV4 (BiGRU+Char, BiLSTM+Char, "
                "TextCNN [3], TextCNN [3,4,5], Char-CNN) @0.50"
            ),
            "weights": {
                "HYBRID-RNN-01": 0.22,
                "HYBRID-RNN-WD": 0.22,
                "CNN-01-WD": 0.18,
                "CNN-04b": 0.16,
                "HYBRID-BILSTM-01": 0.12,
                "CHAR-02": 0.10,
            },
        },
        {
            "name": "ENS-TV4-OOF-PLATT",
            "scope": "intra_tv4",
            "protocol": "oof_5fold_platt_calibrated",
            "description": "5-Fold Out-of-Fold Platt Calibrated Voting trên 6 mô hình TV4",
            "weights": {
                "HYBRID-RNN-01": 0.22,
                "HYBRID-RNN-WD": 0.22,
                "CNN-01-WD": 0.18,
                "CNN-04b": 0.16,
                "HYBRID-BILSTM-01": 0.12,
                "CHAR-02": 0.10,
            },
        },
    ]

    if "EXP-N6-C-calibrated" in cands:
        ensemble_specs.extend(
            [
                {
                    "name": "ENS-CROSS-50-50",
                    "scope": "cross_team",
                    "protocol": "fixed_cutoff_050",
                    "description": (
                        "Soft Voting 50/50 "
                        "(TV2 EXP-N6-C-calibrated + TV4 HYBRID-RNN-01) @0.50"
                    ),
                    "weights": {
                        "EXP-N6-C-calibrated": 0.50,
                        "HYBRID-RNN-01": 0.50,
                    },
                },
                {
                    "name": "ENS-CROSS-WEIGHTED-050",
                    "scope": "cross_team",
                    "protocol": "fixed_cutoff_050",
                    "description": (
                        "Weighted Voting (0.50 N6-C-cal + 0.20 HR01 + "
                        "0.15 HRWD + 0.15 CNN01WD) @0.50"
                    ),
                    "weights": {
                        "EXP-N6-C-calibrated": 0.50,
                        "HYBRID-RNN-01": 0.20,
                        "HYBRID-RNN-WD": 0.15,
                        "CNN-01-WD": 0.15,
                    },
                },
                {
                    "name": "ENS-CROSS-DIVERSE-050",
                    "scope": "cross_team",
                    "protocol": "fixed_cutoff_050",
                    "description": (
                        "Weighted Voting Đa kiến trúc (0.48 N6-C-cal + 0.18 HR01 + "
                        "0.14 HRWD + 0.10 CNN01WD + 0.05 CNN04b + 0.05 CHAR02) @0.50"
                    ),
                    "weights": {
                        "EXP-N6-C-calibrated": 0.48,
                        "HYBRID-RNN-01": 0.18,
                        "HYBRID-RNN-WD": 0.14,
                        "CNN-01-WD": 0.10,
                        "CNN-04b": 0.05,
                        "CHAR-02": 0.05,
                    },
                },
                {
                    "name": "ENS-CROSS-OOF-PLATT",
                    "scope": "cross_team",
                    "protocol": "oof_5fold_platt_calibrated",
                    "description": (
                        "5-Fold Out-of-Fold Calibrated Voting (TV2 N6-C Train-CV Calibrated + "
                        "TV4 Fold-Platt Calibrated + OOF Threshold)"
                    ),
                    "weights": {
                        "EXP-N6-C-calibrated": 0.48,
                        "HYBRID-RNN-01": 0.18,
                        "HYBRID-RNN-WD": 0.14,
                        "CNN-01-WD": 0.10,
                        "CNN-04b": 0.05,
                        "CHAR-02": 0.05,
                    },
                },
                {
                    "name": "ENS-CROSS-OOF-STACKING-LR",
                    "scope": "cross_team",
                    "protocol": "oof_5fold_stacking_diagnostic",
                    "description": (
                        "Chẩn đoán 5-Fold Val-OOF Logistic Meta-Learner trên log-odds "
                        "(N6-C-cal + HYBRID-RNN-01 + CNN-01-WD + CHAR-02)"
                    ),
                    "weights": {
                        "EXP-N6-C-calibrated": 0.25,
                        "HYBRID-RNN-01": 0.25,
                        "CNN-01-WD": 0.25,
                        "CHAR-02": 0.25,
                    },
                },
                {
                    "name": "ENS-CROSS-IN-SAMPLE-DIAG",
                    "scope": "cross_team",
                    "protocol": "in_sample_threshold_diagnostic",
                    "description": (
                        "Chẩn đoán ngưỡng In-Sample trên ENS-CROSS-DIVERSE "
                        "(chỉ để đối chiếu công bằng với N6-C-cal @0.34)"
                    ),
                    "weights": {
                        "EXP-N6-C-calibrated": 0.48,
                        "HYBRID-RNN-01": 0.18,
                        "HYBRID-RNN-WD": 0.14,
                        "CNN-01-WD": 0.10,
                        "CNN-04b": 0.05,
                        "CHAR-02": 0.05,
                    },
                },
            ]
        )

    ensembles_out: list[dict[str, Any]] = []
    for spec in ensemble_specs:
        w_map = spec["weights"]
        if not all(k in cands for k in w_map):
            continue
        tot_size = sum(float(cands[k]["model_size_mb"]) for k in w_map)
        tot_params = sum(int(cands[k]["num_params"]) for k in w_map)
        tot_latency = sum(float(cands[k]["inference_ms_per_sample"]) for k in w_map)

        seed_res_ens: dict[int, dict[str, Any]] = {}
        for s in seeds:
            m_probs = {k: cands[k]["by_seed"][s]["prob_toxic"] for k in w_map}
            proto = spec["protocol"]
            if proto == "fixed_cutoff_050":
                p_blend = blend_validated_probabilities(m_probs, w_map)
                y_hat = (p_blend >= 0.50).astype(np.int64)
                thr = 0.50
                is_cal = all(cands[k]["is_calibrated"] for k in w_map)
                in_samp = False
            elif proto == "oof_5fold_platt_calibrated":
                already_cal = {k for k in w_map if cands[k]["is_calibrated"]}
                y_hat, p_blend, thr = _run_oof_platt_voting_seed(
                    y_true, m_probs, w_map, already_cal, seed=s
                )
                is_cal = True
                in_samp = False
            elif proto == "oof_5fold_stacking_diagnostic":
                y_hat, p_blend, thr = _run_oof_stacking_lr_seed(
                    y_true, m_probs, list(w_map.keys()), seed=s
                )
                is_cal = True
                in_samp = False
            else:
                p_blend = blend_validated_probabilities(m_probs, w_map)
                sw = sweep_probability_thresholds(y_true, p_blend)
                thr = float(sw["best_macro_f1_threshold"])
                y_hat = (p_blend >= thr).astype(np.int64)
                is_cal = False
                in_samp = True

            m_eval = _evaluate_seed_predictions(
                y_true, y_hat, p_blend, sources, thr, is_cal, in_samp
            )
            seed_res_ens[s] = m_eval
            if spec["name"] in (
                "ENS-TV4-DIVERSE-6",
                "ENS-TV4-OOF-PLATT",
                "ENS-CROSS-DIVERSE-050",
                "ENS-CROSS-OOF-PLATT",
            ):
                _save_ensemble_seed_predictions(
                    out_dir=ensemble_models_dir,
                    ensemble_name=spec["name"],
                    seed=s,
                    val_ids=val_ids,
                    sources=sources,
                    y_true=y_true,
                    y_pred=y_hat,
                    prob_toxic=p_blend,
                    threshold=thr,
                    is_calibrated=is_cal,
                    metrics=m_eval,
                    weights=w_map,
                    protocol=proto,
                )

        ensembles_out.append(
            _aggregate_seed_metrics(
                seed_res_ens,
                {
                    "name": spec["name"],
                    "scope": spec["scope"],
                    "protocol": spec["protocol"],
                    "description": spec["description"],
                    "weights": w_map,
                    "n_models": len(w_map),
                    "model_size_mb": round(tot_size, 4),
                    "num_params": tot_params,
                    "inference_ms_per_sample": round(tot_latency, 4),
                },
            )
        )

    single_by_name = {b["name"]: b for b in single_baselines}
    ens_by_name = {e["name"]: e for e in ensembles_out}
    best_tv4_single = float(single_by_name["HYBRID-RNN-01"]["macro_f1"]["mean"])
    best_n6_cal_050 = (
        float(single_by_name["EXP-N6-C-calibrated"]["macro_f1"]["mean"])
        if "EXP-N6-C-calibrated" in single_by_name
        else best_tv4_single
    )
    best_n6_uncal = (
        float(single_by_name["EXP-N6-C-uncal"]["macro_f1"]["mean"])
        if "EXP-N6-C-uncal" in single_by_name
        else best_n6_cal_050
    )

    for e in ensembles_out:
        m_mean = float(e["macro_f1"]["mean"])
        e["gain_vs_best_tv4_single"] = round(m_mean - best_tv4_single, 4)
        e["gain_vs_n6_cal_050"] = round(m_mean - best_n6_cal_050, 4)
        e["gain_vs_n6_uncal_margin"] = round(m_mean - best_n6_uncal, 4)
        ref = best_tv4_single if e["scope"] == "intra_tv4" else best_n6_uncal
        e["passes_adr_1pct_gate"] = bool((m_mean - ref) >= ADR_MIN_MACRO_F1_GAIN)

    adr_summary = {
        "adr_min_gain_required": ADR_MIN_MACRO_F1_GAIN,
        "best_tv4_single_macro_f1": round(best_tv4_single, 4),
        "best_n6_cal_050_macro_f1": round(best_n6_cal_050, 4),
        "best_n6_uncal_macro_f1": round(best_n6_uncal, 4),
        "intra_tv4_winner": "ENS-TV4-DIVERSE-6",
        "intra_tv4_winner_macro_f1": (
            round(float(ens_by_name["ENS-TV4-DIVERSE-6"]["macro_f1"]["mean"]), 4)
            if "ENS-TV4-DIVERSE-6" in ens_by_name
            else None
        ),
        "intra_tv4_gain": (
            ens_by_name["ENS-TV4-DIVERSE-6"]["gain_vs_best_tv4_single"]
            if "ENS-TV4-DIVERSE-6" in ens_by_name
            else None
        ),
        "intra_tv4_passes_adr": (
            ens_by_name["ENS-TV4-DIVERSE-6"]["passes_adr_1pct_gate"]
            if "ENS-TV4-DIVERSE-6" in ens_by_name
            else False
        ),
        "cross_team_winner": (
            "ENS-CROSS-OOF-PLATT"
            if "ENS-CROSS-OOF-PLATT" in ens_by_name
            else "ENS-TV4-DIVERSE-6"
        ),
        "cross_team_winner_macro_f1": (
            round(float(ens_by_name["ENS-CROSS-OOF-PLATT"]["macro_f1"]["mean"]), 4)
            if "ENS-CROSS-OOF-PLATT" in ens_by_name
            else None
        ),
        "cross_team_gain_vs_n6_cal_050": (
            ens_by_name["ENS-CROSS-OOF-PLATT"]["gain_vs_n6_cal_050"]
            if "ENS-CROSS-OOF-PLATT" in ens_by_name
            else None
        ),
        "cross_team_gain_vs_n6_uncal": (
            ens_by_name["ENS-CROSS-OOF-PLATT"]["gain_vs_n6_uncal_margin"]
            if "ENS-CROSS-OOF-PLATT" in ens_by_name
            else None
        ),
        "stacking_train_oof_decision": (
            "SKIP_TRAIN_OOF_FOR_PRODUCTION: Generating 5-fold Train-OOF across 6 neural "
            "architectures x 3 seeds requires 90 GPU training runs (~2.5h GPU) while "
            "5-fold Val-OOF Logistic Stacking offers no gain over calibrated Weighted "
            "Probability Voting. Valid Probability Voting is selected per AGENTS.md §6."
        ),
    }

    return {
        "single_baselines": single_baselines,
        "ensembles": ensembles_out,
        "adr_decision": adr_summary,
    }


def _plot_error_diversity_heatmap(
    loaded: dict[str, Any],
    diversity_pairs: list[dict[str, Any]],
    out_path: Path,
    plt: Any,
) -> Path:
    focus = [
        name
        for name in (
            "EXP-N6-C-calibrated",
            "HYBRID-RNN-01",
            "HYBRID-RNN-WD",
            "CNN-01-WD",
            "CNN-04b",
            "HYBRID-BILSTM-01",
            "CHAR-02",
        )
        if name in loaded["candidates"]
    ]
    n = len(focus)
    dis_mat = np.zeros((n, n), dtype=np.float64)
    pair_lookup = {
        (p["model_a"], p["model_b"]): p for p in diversity_pairs
    }
    pair_lookup.update(
        {(p["model_b"], p["model_a"]): p for p in diversity_pairs}
    )
    for i in range(n):
        for j in range(n):
            if i == j:
                dis_mat[i, j] = 0.0
            else:
                p = pair_lookup.get((focus[i], focus[j]))
                dis_mat[i, j] = float(p["disagreement_rate"]) * 100.0 if p else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    ax_hm, ax_bar = axes
    im = ax_hm.imshow(dis_mat, cmap="YlGnBu", vmin=0.0, vmax=max(14.0, float(dis_mat.max())))
    for r in range(n):
        for c in range(n):
            val = dis_mat[r, c]
            color = "white" if val > 8.5 else "#0f172a"
            ax_hm.text(
                c, r, f"{val:.1f}%", ha="center", va="center",
                color=color, fontsize=8, fontweight="bold",
            )
    ax_hm.set_xticks(range(n))
    ax_hm.set_xticklabels(focus, rotation=28, ha="right", fontsize=8)
    ax_hm.set_yticks(range(n))
    ax_hm.set_yticklabels(focus, fontsize=8)
    ax_hm.set_title("Pairwise Prediction Disagreement Rate (%) on Val (Seed 42)")
    fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    top_pairs = sorted(
        [
            p
            for p in diversity_pairs
            if p["model_a"] in focus and p["model_b"] in focus
        ],
        key=lambda x: x["oracle_macro_f1"],
        reverse=True,
    )[:6]
    labels = [f"{p['model_a']}\n+ {p['model_b']}" for p in reversed(top_pairs)]
    oracles = [p["oracle_macro_f1"] for p in reversed(top_pairs)]
    y_pos = np.arange(len(labels))
    bars = ax_bar.barh(y_pos, oracles, color="#0f766e", height=0.58, alpha=0.88)
    for bar, p in zip(bars, reversed(top_pairs)):
        ax_bar.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2.0,
            f"Oracle={p['oracle_macro_f1']:.4f} (A+B={p['complementary_gain_potential']})",
            va="center",
            fontsize=8,
        )
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=7.8)
    ax_bar.set_xlim(0.75, 0.93)
    ax_bar.set_xlabel("Pairwise Oracle Validation Macro-F1")
    ax_bar.set_title("Top Complementary Pairs (Oracle Upper Bound & Recovery Count)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_ensemble_comparison(
    eval_res: dict[str, Any], out_path: Path, plt: Any
) -> Path:
    items: list[tuple[str, float, float, float, str]] = []
    for b in eval_res["single_baselines"]:
        if b["name"] in (
            "CHAR-02",
            "CNN-01-WD",
            "HYBRID-RNN-01",
            "EXP-N6-C-calibrated",
            "EXP-N6-C-uncal",
            "EXP-N6-C-calibrated@0.34 (in-sample thr)",
        ):
            items.append(
                (
                    f"[Single] {b['name']}",
                    float(b["macro_f1"]["mean"]),
                    float(b["macro_f1"]["std"]),
                    float(b["f1_toxic"]["mean"]),
                    "#64748b",
                )
            )
    for e in eval_res["ensembles"]:
        color = "#2563eb" if e["scope"] == "intra_tv4" else "#0f766e"
        items.append(
            (
                f"[Ensemble] {e['name']}",
                float(e["macro_f1"]["mean"]),
                float(e["macro_f1"]["std"]),
                float(e["f1_toxic"]["mean"]),
                color,
            )
        )

    fig, ax = plt.subplots(figsize=(11.5, max(5.6, 0.42 * len(items) + 1.4)))
    y = np.arange(len(items))
    means = [x[1] for x in items]
    stds = [x[2] for x in items]
    tox = [x[3] for x in items]
    colors = [x[4] for x in items]

    bars = ax.barh(y, means, xerr=stds, color=colors, height=0.62, alpha=0.88, capsize=3)
    ax.scatter(tox, y, color="#dc2626", zorder=4, s=36, label="F1 TOXIC (3-seed mean)")
    adr = eval_res["adr_decision"]
    tv4_gate = float(adr["best_tv4_single_macro_f1"]) + ADR_MIN_MACRO_F1_GAIN
    ax.axvline(
        tv4_gate,
        color="#2563eb",
        linestyle="--",
        linewidth=1.5,
        label=f"TV4 ADR +1.0% Gate ({tv4_gate:.4f})",
    )
    if adr["best_n6_uncal_macro_f1"]:
        n6_gate = float(adr["best_n6_uncal_macro_f1"]) + ADR_MIN_MACRO_F1_GAIN
        ax.axvline(
            n6_gate,
            color="#0f766e",
            linestyle=":",
            linewidth=1.6,
            label=f"Cross-Team ADR +1.0% vs N6-C ({n6_gate:.4f})",
        )
    for bar, m_val, s_val, t_val in zip(bars, means, stds, tox):
        ax.text(
            m_val + s_val + 0.003,
            bar.get_y() + bar.get_height() / 2.0,
            f"{m_val:.4f} (TOXIC {t_val:.4f})",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([x[0] for x in items], fontsize=8.2)
    ax.set_xlim(0.55, 0.88)
    ax.set_ylim(-1.8, len(items) - 0.3)
    ax.set_xlabel("Validation Score (3-seed mean +/- std)")
    ax.set_title("Phase 5 (EXP-06): Single Models vs Intra-TV4 & Cross-Team Ensembles")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_ensemble_source_gains(
    eval_res: dict[str, Any], out_path: Path, plt: Any
) -> Path:
    single_map = {b["name"]: b for b in eval_res["single_baselines"]}
    ens_map = {e["name"]: e for e in eval_res["ensembles"]}
    ordered_keys = [
        ("HYBRID-RNN-01", single_map.get("HYBRID-RNN-01")),
        ("CNN-01-WD", single_map.get("CNN-01-WD")),
        ("EXP-N6-C-cal@0.50", single_map.get("EXP-N6-C-calibrated")),
        ("EXP-N6-C-uncal", single_map.get("EXP-N6-C-uncal")),
        ("ENS-TV4-INTRA-4", ens_map.get("ENS-TV4-INTRA-4")),
        ("ENS-TV4-DIVERSE-6", ens_map.get("ENS-TV4-DIVERSE-6")),
        ("ENS-CROSS-DIVERSE@0.50", ens_map.get("ENS-CROSS-DIVERSE-050")),
        ("ENS-CROSS-OOF-PLATT", ens_map.get("ENS-CROSS-OOF-PLATT")),
    ]
    present = [(label, obj) for label, obj in ordered_keys if obj is not None]
    labels = [x[0] for x in present]
    vih_m = [float(x[1]["by_source"]["vihsd"]["macro_f1"]["mean"]) for x in present]
    vic_m = [float(x[1]["by_source"]["victsd"]["macro_f1"]["mean"]) for x in present]
    vic_t = [float(x[1]["by_source"]["victsd"]["f1_toxic"]["mean"]) for x in present]

    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.bar(x - w, vih_m, width=w, color="#2563eb", label="ViHSD Macro-F1")
    ax.bar(x, vic_m, width=w, color="#0f766e", label="ViCTSD Macro-F1")
    ax.bar(x + w, vic_t, width=w, color="#f59e0b", label="ViCTSD F1-TOXIC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=8.2)
    ax.set_ylim(0.25, 0.94)
    ax.set_ylabel("Validation Score (3-seed mean)")
    ax.set_title("Domain Gains from Ensemble: ViHSD (n=4,559) vs ViCTSD (n=1,489)")
    ax.legend(ncol=3, loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def generate_tv4_phase5_ensemble_report(
    root: Path = ROOT,
    models_dir: Path = DEFAULT_CNN_MODELS_DIR,
    baselines_dir: Path = DEFAULT_BASELINES_DIR,
    ensemble_models_dir: Path = DEFAULT_ENSEMBLE_MODELS_DIR,
    out_dir: Path = DEFAULT_ENSEMBLE_OUT_DIR,
) -> dict[str, Any]:
    """Run complete Phase 5 (EXP-06) error diversity audit, voting protocols, and plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ensemble_models_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_candidate_validation_predictions(
        root=root, models_dir=models_dir, baselines_dir=baselines_dir
    )
    diversity_s42 = compute_pairwise_error_diversity(loaded, seed=int(loaded["seeds"][0]))
    eval_res = evaluate_ensemble_protocols(
        loaded, ensemble_models_dir=ensemble_models_dir
    )
    plt = _configure_matplotlib()
    plots = {
        "error_diversity_heatmap": str(
            _plot_error_diversity_heatmap(
                loaded, diversity_s42, out_dir / "error_diversity_heatmap.png", plt
            )
        ),
        "ensemble_comparison": str(
            _plot_ensemble_comparison(
                eval_res, out_dir / "ensemble_comparison.png", plt
            )
        ),
        "ensemble_source_gains": str(
            _plot_ensemble_source_gains(
                eval_res, out_dir / "ensemble_source_gains.png", plt
            )
        ),
    }

    payload = {
        "experiment": "TV4-EXP06-Conditional-Ensemble",
        "n_val_samples": loaded["n_val_samples"],
        "seeds": loaded["seeds"],
        "expected_hashes": loaded["expected_hashes"],
        "candidates_inventory": [
            {
                "name": c["name"],
                "owner": c["owner"],
                "family": c["family"],
                "has_probabilities": c["has_probabilities"],
                "is_calibrated": c["is_calibrated"],
                "model_size_mb": c["model_size_mb"],
                "num_params": c["num_params"],
                "inference_ms_per_sample": c["inference_ms_per_sample"],
            }
            for c in loaded["candidates"].values()
        ],
        "other_members_status": loaded["other_members_status"],
        "pairwise_error_diversity_seed42": diversity_s42,
        "single_baselines": eval_res["single_baselines"],
        "ensembles": eval_res["ensembles"],
        "adr_decision": eval_res["adr_decision"],
        "plots": plots,
    }
    (out_dir / "tv4_phase5_ensemble_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TV4 Phase 5 (EXP-06) conditional ensemble & error diversity audit."
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_CNN_MODELS_DIR)
    parser.add_argument("--baselines-dir", type=Path, default=DEFAULT_BASELINES_DIR)
    parser.add_argument(
        "--ensemble-models-dir", type=Path, default=DEFAULT_ENSEMBLE_MODELS_DIR
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ENSEMBLE_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rep = generate_tv4_phase5_ensemble_report(
        models_dir=args.models_dir,
        baselines_dir=args.baselines_dir,
        ensemble_models_dir=args.ensemble_models_dir,
        out_dir=args.out_dir,
    )
    adr = rep["adr_decision"]
    print(
        f"TV4 Phase 5 OK | candidates={len(rep['candidates_inventory'])} | "
        f"intra_winner={adr['intra_tv4_winner']} ({adr['intra_tv4_winner_macro_f1']}, "
        f"gain={adr['intra_tv4_gain']:+.4f}, pass={adr['intra_tv4_passes_adr']}) | "
        f"cross_winner={adr['cross_team_winner']} ({adr['cross_team_winner_macro_f1']}, "
        f"gain_vs_n6={adr['cross_team_gain_vs_n6_uncal']:+.4f})"
    )


if __name__ == "__main__":
    main()
