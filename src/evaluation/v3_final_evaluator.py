"""Final V3 evaluation, calibration, error diversity, ensemble, and chart generator.

Performs inference and statistical evaluation ONLY (zero new training):
- Aggregates V1, V2, V3 Scratch (12 runs), and V3 Pretrained Transformer (PhoBERT-v2 3 seeds + XLM-R s42).
- Computes pairwise error diversity, 5-fold Val-OOF calibration (Platt, Temperature, Isotonic),
  weighted probability voting, and Logistic Regression stacking.
- Evaluates frozen V3 models and ensembles on Historical Benchmark Test (test.parquet, test_blind=False)
  with 95% Bootstrap CIs and McNemar significance tests.
- Exports JSON report and 4 high-DPI PNG figures under artifacts/TV4-v3-research/.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

from src.evaluation.cnn_report import (
    _configure_matplotlib,
    apply_platt_scaling,
    apply_temperature_scaling,
    fit_platt_scaling,
    fit_temperature_scaling,
    sweep_probability_thresholds,
)
from sklearn.metrics import cohen_kappa_score
from src.models.cnn_pipeline import load_cnn_artifacts, predict_cnn_from_artifacts


def _compute_pairwise_diversity(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
) -> dict[str, Any]:
    ok_a = pred_a == y_true
    ok_b = pred_b == y_true
    disagree = pred_a != pred_b
    only_a = int(np.sum(ok_a & (~ok_b)))
    only_b = int(np.sum((~ok_a) & ok_b))
    double_fault = int(np.sum((~ok_a) & (~ok_b)))
    oracle_pred = np.where(ok_a | ok_b, y_true, pred_a)
    n = len(y_true)
    tp1 = int(np.sum((y_true == 1) & (oracle_pred == 1)))
    fp1 = int(np.sum((y_true == 0) & (oracle_pred == 1)))
    fn1 = int(np.sum((y_true == 1) & (oracle_pred == 0)))
    tn1 = n - tp1 - fp1 - fn1
    f1_t = (2.0 * tp1) / max(1, 2 * tp1 + fp1 + fn1)
    f1_c = (2.0 * tn1) / max(1, 2 * tn1 + fn1 + fp1)
    return {
        "disagreement_rate": round(float(disagree.mean()), 4),
        "disagreement_count": int(disagree.sum()),
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "double_fault_rate": round(float(double_fault / max(1, n)), 4),
        "cohen_kappa": round(float(cohen_kappa_score(pred_a, pred_b)), 4),
        "prob_pearson_corr": round(float(np.corrcoef(prob_a, prob_b)[0, 1]), 4),
        "oracle_macro_f1": round(float(0.5 * (f1_c + f1_t)), 4),
        "oracle_f1_toxic": round(float(f1_t), 4),
    }
from src.models.svm_binary_pipeline import (
    compute_binary_metrics,
    compute_ece,
)
from src.models.transformer_pipeline import load_and_predict_transformer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "binary_n2_v1"
V3_SCRATCH_DIR = ROOT / "models" / "cnn_v3" / "TV4-V3-scratch"
V3_TRANS_DIR = ROOT / "models" / "transformers_v3"
V2_SCRATCH_DIR = ROOT / "models" / "cnn_v2" / "TV4-run-4e7ab33dcccb446a9cca550579e71213"
N6_VAL_DIR = (
    ROOT
    / "models"
    / "baselines"
    / "N6-run-faffb8025fbc41fb8d9b463468bdd5f3"
    / "EXP-N6-C-calibrated"
)
OUT_DIR = ROOT / "artifacts" / "TV4-v3-research"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_prob_json(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["predictions"] if isinstance(payload, dict) and "predictions" in payload else payload
    return np.array([float(r["probabilities"]["TOXIC"]) for r in rows], dtype=np.float64)


def _eval_probs(
    y: np.ndarray,
    probs: np.ndarray,
    sources: list[str],
    threshold: float = 0.50,
) -> dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    m = compute_binary_metrics(y.tolist(), preds.tolist(), sources, probs)
    p_clip = np.clip(probs, 1e-7, 1.0 - 1e-7)
    m["ece"] = round(float(compute_ece(y, probs, n_bins=10)), 6)
    m["brier_score"] = round(float(brier_score_loss(y, probs)), 6)
    m["log_loss"] = round(float(log_loss(y, p_clip, labels=[0, 1])), 6)
    m["threshold"] = round(float(threshold), 4)
    return m


def _oof_calibrate_and_threshold(
    y_val: np.ndarray,
    p_val: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    sources_val: list[str],
    sources_test: list[str],
    method: str = "platt",
) -> dict[str, Any]:
    """5-fold stratified Val-OOF calibration + threshold selection, plus full-val freeze for test."""
    strat_key = [f"{s}_{l}" for s, l in zip(sources_val, y_val)]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_cal = np.zeros_like(p_val)
    oof_pred = np.zeros(len(y_val), dtype=int)
    fold_thrs: list[float] = []

    for tr_idx, va_idx in skf.split(np.zeros(len(y_val)), strat_key):
        y_tr, p_tr = y_val[tr_idx], p_val[tr_idx]
        p_va = p_val[va_idx]
        if method == "platt":
            pl = fit_platt_scaling(p_tr, y_tr)
            p_tr_c = apply_platt_scaling(p_tr, pl["coef"], pl["intercept"])
            p_va_c = apply_platt_scaling(p_va, pl["coef"], pl["intercept"])
        elif method == "temperature":
            tp = fit_temperature_scaling(p_tr, y_tr)
            p_tr_c = apply_temperature_scaling(p_tr, tp["temperature"])
            p_va_c = apply_temperature_scaling(p_va, tp["temperature"])
        else:  # isotonic
            iso = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
            iso.fit(p_tr, y_tr)
            p_tr_c = iso.predict(p_tr)
            p_va_c = iso.predict(p_va)
        sw = sweep_probability_thresholds(y_tr, p_tr_c)
        thr = float(sw.get("best_threshold_macro_f1", sw.get("best_threshold", 0.40)))
        fold_thrs.append(thr)
        oof_cal[va_idx] = p_va_c
        oof_pred[va_idx] = (p_va_c >= thr).astype(int)

    val_m = compute_binary_metrics(y_val.tolist(), oof_pred.tolist(), sources_val, oof_cal)
    p_clip = np.clip(oof_cal, 1e-7, 1.0 - 1e-7)
    val_m["ece"] = round(float(compute_ece(y_val, oof_cal, n_bins=10)), 6)
    val_m["brier_score"] = round(float(brier_score_loss(y_val, oof_cal)), 6)
    val_m["log_loss"] = round(float(log_loss(y_val, p_clip, labels=[0, 1])), 6)
    val_m["mean_oof_threshold"] = round(float(np.mean(fold_thrs)), 4)

    # Fit once on full val_val and apply frozen to test.parquet
    if method == "platt":
        pl_full = fit_platt_scaling(p_val, y_val)
        p_val_c = apply_platt_scaling(p_val, pl_full["coef"], pl_full["intercept"])
        p_test_c = apply_platt_scaling(p_test, pl_full["coef"], pl_full["intercept"])
        cal_params: dict[str, Any] = pl_full
    elif method == "temperature":
        tp_full = fit_temperature_scaling(p_val, y_val)
        p_val_c = apply_temperature_scaling(p_val, tp_full["temperature"])
        p_test_c = apply_temperature_scaling(p_test, tp_full["temperature"])
        cal_params = tp_full
    else:
        iso_full = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
        iso_full.fit(p_val, y_val)
        p_val_c = iso_full.predict(p_val)
        p_test_c = iso_full.predict(p_test)
        cal_params = {"method": "isotonic"}

    sw_full = sweep_probability_thresholds(y_val, p_val_c)
    frozen_thr = float(sw_full.get("best_threshold_macro_f1", sw_full.get("best_threshold", 0.40)))
    test_m = _eval_probs(y_test, p_test_c, sources_test, threshold=frozen_thr)
    return {
        "method": method,
        "val_oof_metrics": val_m,
        "test_frozen_metrics": test_m,
        "frozen_threshold": round(frozen_thr, 4),
        "cal_params": cal_params,
        "val_oof_probs": oof_cal,
        "test_frozen_probs": p_test_c,
    }


def _bootstrap_ci_and_mcnemar(
    y: np.ndarray,
    preds_map: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, size=(n_boot, n))
    yb = y[idx]
    ci_out: dict[str, Any] = {}
    for name, yhat in preds_map.items():
        yhb = yhat[idx]
        tp1 = np.sum((yb == 1) & (yhb == 1), axis=1)
        fp1 = np.sum((yb == 0) & (yhb == 1), axis=1)
        fn1 = np.sum((yb == 1) & (yhb == 0), axis=1)
        tn1 = n - tp1 - fp1 - fn1
        f1_tox = (2.0 * tp1) / np.maximum(2 * tp1 + fp1 + fn1, 1)
        f1_cln = (2.0 * tn1) / np.maximum(2 * tn1 + fn1 + fp1, 1)
        mf1 = 0.5 * (f1_cln + f1_tox)
        ci_out[name] = {
            "macro_f1_ci95": [round(float(np.percentile(mf1, 2.5)), 4), round(float(np.percentile(mf1, 97.5)), 4)],
            "f1_toxic_ci95": [round(float(np.percentile(f1_tox, 2.5)), 4), round(float(np.percentile(f1_tox, 97.5)), 4)],
        }
    mc_out = []
    for a, b in pairs:
        if a not in preds_map or b not in preds_map:
            continue
        ok_a = preds_map[a] == y
        ok_b = preds_map[b] == y
        b_cnt = int(np.sum(ok_a & (~ok_b)))
        c_cnt = int(np.sum((~ok_a) & ok_b))
        den = b_cnt + c_cnt
        chi2 = ((abs(b_cnt - c_cnt) - 1.0) ** 2) / den if den > 0 else 0.0
        pval = float(math.erfc(math.sqrt(chi2 / 2.0))) if den > 0 else 1.0
        mc_out.append(
            {
                "a": a,
                "b": b,
                "only_a_correct": b_cnt,
                "only_b_correct": c_cnt,
                "net_gain": b_cnt - c_cnt,
                "chi2_cc": round(float(chi2), 4),
                "p_value": round(pval, 6),
                "significant_005": bool(pval < 0.05),
            }
        )
    return {"bootstrap_95ci": ci_out, "mcnemar": mc_out}


def _reproduce_n6_val_and_test(
    val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, float]:
    """Re-fit TV2 EXP-N6-C-calibrated strictly on train.parquet and verify 0.0 diff on val."""
    from src.evaluation.locked_test_report import _evaluate_n6_reference_on_test

    saved_val_p = _load_prob_json(N6_VAL_DIR / "seed_42" / "val_predictions.json")
    n6_res = _evaluate_n6_reference_on_test(
        root=ROOT,
        frozen_n6={
            "seed_dir": str(N6_VAL_DIR / "seed_42"),
            "frozen_val_tuned_threshold": 0.34,
        },
        val_df=val_df,
        test_df=test_df,
        seeds=[42],
    )
    assert n6_res is not None
    return (
        saved_val_p,
        n6_res["prob_toxic_by_seed"]["42"].astype(np.float64),
        float(n6_res["val_repro_max_prob_diff"]),
    )


def _error_analysis_comparison(
    val_df: pd.DataFrame,
    p_phobert: np.ndarray,
    p_scratch: np.ndarray,
    p_cross: np.ndarray,
    thr_cross: float,
) -> dict[str, Any]:
    y = val_df["label"].astype(int).to_numpy()
    texts = val_df["text"].astype(str).tolist()
    n_words = np.array([len(t.split()) for t in texts])
    flags = {
        "non_nfc": np.array([t != unicodedata.normalize("NFC", t) for t in texts]),
        "no_accent": np.array(
            [
                bool(re.search(r"[a-zA-Z]{3,}", t))
                and not bool(
                    re.search(
                        r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
                        t.lower(),
                    )
                )
                for t in texts
            ]
        ),
        "has_negation": np.array(
            [bool(re.search(r"\b(không|ko|k|chẳng|chưa|đừng|chả|hông)\b", t.lower())) for t in texts]
        ),
        "has_teencode": np.array(
            [bool(re.search(r"\b(vl|vcl|đm|dm|đkm|dkm|vkl|cc|ko|dc|đc|cx|méo|éo|cmnr|clgt)\b", t.lower())) for t in texts]
        ),
        "has_quote": np.array([('"' in t) or ("“" in t) or ("chửi" in t.lower()) for t in texts]),
        "has_emoji": np.array([bool(re.search(r"[\U0001F300-\U0001FAFF]|:\)+|=+\)+", t)) for t in texts]),
        "long_ge_30w": n_words >= 30,
        "short_le_5w": n_words <= 5,
    }
    models = {
        "V3-BIGRU-GATED-FT": (p_scratch >= 0.50).astype(int),
        "V3-PHOBERT-V2": (p_phobert >= 0.50).astype(int),
        "V3-ENS-CROSS-PLATT": (p_cross >= thr_cross).astype(int),
    }
    summary: dict[str, Any] = {}
    for m_name, yhat in models.items():
        err = yhat != y
        fn = (y == 1) & (yhat == 0)
        fp = (y == 0) & (yhat == 1)
        by_phenom = {}
        for k, mask in flags.items():
            sub_n = int(mask.sum())
            by_phenom[k] = {
                "n": sub_n,
                "err_rate_pct": round(float(err[mask].mean() * 100.0), 2) if sub_n else 0.0,
                "fn": int(fn[mask].sum()),
                "fp": int(fp[mask].sum()),
            }
        summary[m_name] = {
            "total_errors": int(err.sum()),
            "error_rate_pct": round(float(err.mean() * 100.0), 2),
            "fn": int(fn.sum()),
            "fp": int(fp.sum()),
            "by_phenomenon": by_phenom,
        }
    return summary


def run_v3_final_evaluation() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    y_val = val_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()
    src_val = val_df["source"].astype(str).tolist()
    src_test = test_df["source"].astype(str).tolist()
    txt_test = test_df["text"].astype(str).tolist()

    # 1. Load & predict all V3 Scratch models (4 variants x 3 seeds)
    scratch_variants = [
        "V3-BIGRU-GATED-MSD",
        "V3-BIGRU-GATED-FT",
        "V3-BIGRU-PYVI-FT",
        "V3-TRI-GATED-FT",
    ]
    val_probs: dict[str, dict[int, np.ndarray]] = {}
    test_probs: dict[str, dict[int, np.ndarray]] = {}
    single_summary: dict[str, Any] = {}

    for v in scratch_variants:
        val_probs[v], test_probs[v] = {}, {}
        per_seed_out = {}
        for s in [42, 43, 44]:
            sdir = V3_SCRATCH_DIR / v / f"seed_{s}"
            m_disk = json.loads((sdir / "metrics.json").read_text(encoding="utf-8"))
            p_va = _load_prob_json(sdir / "val_predictions.json")
            art = load_cnn_artifacts(sdir, device="cuda")
            _, pr_te = predict_cnn_from_artifacts(art, txt_test, batch_size=512)
            p_te = pr_te[:, 1].astype(np.float64)
            val_probs[v][s] = p_va
            test_probs[v][s] = p_te
            per_seed_out[str(s)] = {
                "val_050": _eval_probs(y_val, p_va, src_val, 0.50),
                "test_050": _eval_probs(y_test, p_te, src_test, 0.50),
                "train_time_s": m_disk["train_time_s"],
                "peak_vram_mb": m_disk["peak_vram_mb"],
                "num_params": m_disk["num_params"],
                "model_size_mb": m_disk["model_size_mb"],
            }
        single_summary[v] = _agg_single(v, "scratch_v3", per_seed_out)
        print(
            f"[V3-SCRATCH] {v:20s} | Val={single_summary[v]['val_macro_f1_mean']:.4f}±{single_summary[v]['val_macro_f1_std']:.4f} "
            f"| Test={single_summary[v]['test_macro_f1_mean']:.4f}±{single_summary[v]['test_macro_f1_std']:.4f}",
            flush=True,
        )

    # 2. Load & predict V3 Pretrained Transformers (PhoBERT-v2 3 seeds + XLM-R seed 42)
    trans_specs = [
        ("V3-PHOBERT-V2", [42, 43, 44]),
        ("V3-XLMR-BASE", [42]),
    ]
    for v, seeds in trans_specs:
        val_probs[v], test_probs[v] = {}, {}
        per_seed_out = {}
        for s in seeds:
            sdir = V3_TRANS_DIR / v / f"seed_{s}"
            m_disk = json.loads((sdir / "metrics.json").read_text(encoding="utf-8"))
            p_va = _load_prob_json(sdir / "val_predictions.json")
            _, pr_te = load_and_predict_transformer(sdir, txt_test, batch_size=64, device="cuda")
            p_te = pr_te[:, 1].astype(np.float64)
            val_probs[v][s] = p_va
            test_probs[v][s] = p_te
            per_seed_out[str(s)] = {
                "val_050": _eval_probs(y_val, p_va, src_val, 0.50),
                "test_050": _eval_probs(y_test, p_te, src_test, 0.50),
                "train_time_s": m_disk["train_time_s"],
                "peak_vram_mb": m_disk["peak_vram_mb"],
                "num_params": m_disk["num_params"],
                "model_size_mb": m_disk["model_size_mb"],
            }
        single_summary[v] = _agg_single(v, "pretrained_transformer", per_seed_out)
        print(
            f"[V3-TRANS]   {v:20s} | Val={single_summary[v]['val_macro_f1_mean']:.4f}±{single_summary[v]['val_macro_f1_std']:.4f} "
            f"| ViCTSD={single_summary[v]['val_victsd_mean']:.4f} | Test={single_summary[v]['test_macro_f1_mean']:.4f}±{single_summary[v]['test_macro_f1_std']:.4f}",
            flush=True,
        )

    # 3. Reproduce TV2 EXP-N6-C-calibrated (train-only fit, 0.0 diff verified on val)
    p_va_n6, p_te_n6, n6_diff = _reproduce_n6_val_and_test(val_df, test_df)
    val_probs["EXP-N6-C-calibrated"] = {42: p_va_n6, 43: p_va_n6, 44: p_va_n6}
    test_probs["EXP-N6-C-calibrated"] = {42: p_te_n6, 43: p_te_n6, 44: p_te_n6}
    print(f"[TV2-REF]    EXP-N6-C-calibrated | val_repro_max_diff={n6_diff:.2e}", flush=True)

    # 4. Pairwise Error Diversity on seed_42 across 5 diverse families
    div_models = {
        "V3-PHOBERT-V2": val_probs["V3-PHOBERT-V2"][42],
        "V3-XLMR-BASE": val_probs["V3-XLMR-BASE"][42],
        "V3-BIGRU-GATED-FT": val_probs["V3-BIGRU-GATED-FT"][42],
        "V3-TRI-GATED-FT": val_probs["V3-TRI-GATED-FT"][42],
        "EXP-N6-C-calibrated": p_va_n6,
    }
    div_names = list(div_models.keys())
    diversity_pairs = []
    for i in range(len(div_names)):
        for j in range(i + 1, len(div_names)):
            a, b = div_names[i], div_names[j]
            pa, pb = div_models[a], div_models[b]
            d = _compute_pairwise_diversity(
                y_val,
                (pa >= 0.50).astype(int),
                (pb >= 0.50).astype(int),
                pa,
                pb,
            )
            diversity_pairs.append({"model_a": a, "model_b": b, **d})

    # 5. Calibration Comparison (Platt vs Temperature vs Isotonic) on V3-PHOBERT-V2 (3-seed avg)
    p_va_pho_3s = np.mean([val_probs["V3-PHOBERT-V2"][s] for s in [42, 43, 44]], axis=0)
    p_te_pho_3s = np.mean([test_probs["V3-PHOBERT-V2"][s] for s in [42, 43, 44]], axis=0)
    cal_bench = {}
    for meth in ("platt", "temperature", "isotonic"):
        cal_res = _oof_calibrate_and_threshold(
            y_val, p_va_pho_3s, y_test, p_te_pho_3s, src_val, src_test, method=meth
        )
        cal_bench[meth] = {
            k: v for k, v in cal_res.items() if k not in ("val_oof_probs", "test_frozen_probs")
        }

    # 6. Ensemble Suite (Using 3-seed means for multi-seed families + s42 XLM-R + TV2 N6)
    p_va_bigru = np.mean([val_probs["V3-BIGRU-GATED-FT"][s] for s in [42, 43, 44]], axis=0)
    p_te_bigru = np.mean([test_probs["V3-BIGRU-GATED-FT"][s] for s in [42, 43, 44]], axis=0)
    p_va_tri = np.mean([val_probs["V3-TRI-GATED-FT"][s] for s in [42, 43, 44]], axis=0)
    p_te_tri = np.mean([test_probs["V3-TRI-GATED-FT"][s] for s in [42, 43, 44]], axis=0)
    p_va_pyvi = np.mean([val_probs["V3-BIGRU-PYVI-FT"][s] for s in [42, 43, 44]], axis=0)
    p_te_pyvi = np.mean([test_probs["V3-BIGRU-PYVI-FT"][s] for s in [42, 43, 44]], axis=0)
    p_va_msd = np.mean([val_probs["V3-BIGRU-GATED-MSD"][s] for s in [42, 43, 44]], axis=0)
    p_te_msd = np.mean([test_probs["V3-BIGRU-GATED-MSD"][s] for s in [42, 43, 44]], axis=0)
    p_va_xlmr = val_probs["V3-XLMR-BASE"][42]
    p_te_xlmr = test_probs["V3-XLMR-BASE"][42]

    ensemble_defs = {
        "V3-ENS-SCRATCH-INTRA": (
            "Scratch-Only TV4 Ensemble (4 V3 Scratch architectures)",
            0.35 * p_va_bigru + 0.35 * p_va_tri + 0.20 * p_va_pyvi + 0.10 * p_va_msd,
            0.35 * p_te_bigru + 0.35 * p_te_tri + 0.20 * p_te_pyvi + 0.10 * p_te_msd,
            {"V3-BIGRU-GATED-FT": 0.35, "V3-TRI-GATED-FT": 0.35, "V3-BIGRU-PYVI-FT": 0.20, "V3-BIGRU-GATED-MSD": 0.10},
        ),
        "V3-ENS-TV4-ALL": (
            "TV4 Intra-Ensemble (PhoBERT-v2 3s + XLM-R + V3-BIGRU-GATED-FT + V3-TRI-GATED-FT)",
            0.52 * p_va_pho_3s + 0.18 * p_va_xlmr + 0.15 * p_va_bigru + 0.15 * p_va_tri,
            0.52 * p_te_pho_3s + 0.18 * p_te_xlmr + 0.15 * p_te_bigru + 0.15 * p_te_tri,
            {"V3-PHOBERT-V2": 0.52, "V3-XLMR-BASE": 0.18, "V3-BIGRU-GATED-FT": 0.15, "V3-TRI-GATED-FT": 0.15},
        ),
        "V3-ENS-CROSS-VOTING": (
            "Cross-Member Ensemble (TV4 PhoBERT-v2 + XLM-R + V3-Scratch + TV2 EXP-N6-C-calibrated)",
            0.44 * p_va_pho_3s + 0.14 * p_va_xlmr + 0.13 * p_va_bigru + 0.11 * p_va_tri + 0.18 * p_va_n6,
            0.44 * p_te_pho_3s + 0.14 * p_te_xlmr + 0.13 * p_te_bigru + 0.11 * p_te_tri + 0.18 * p_te_n6,
            {"V3-PHOBERT-V2": 0.44, "V3-XLMR-BASE": 0.14, "V3-BIGRU-GATED-FT": 0.13, "V3-TRI-GATED-FT": 0.11, "EXP-N6-C-calibrated": 0.18},
        ),
    }

    ensembles_out: dict[str, Any] = {}
    ens_preds_val_for_err: dict[str, tuple[np.ndarray, float]] = {}
    ens_preds_test_for_mc: dict[str, np.ndarray] = {
        "V3-BIGRU-GATED-FT": (p_te_bigru >= 0.50).astype(int),
        "V3-PHOBERT-V2": (p_te_pho_3s >= 0.50).astype(int),
        "EXP-N6-C-calibrated": (p_te_n6 >= 0.50).astype(int),
    }

    for ename, (desc, pv, pt, wmap) in ensemble_defs.items():
        raw_val = _eval_probs(y_val, pv, src_val, 0.50)
        raw_test = _eval_probs(y_test, pt, src_test, 0.50)
        platt_res = _oof_calibrate_and_threshold(
            y_val, pv, y_test, pt, src_val, src_test, method="platt"
        )
        ensembles_out[ename] = {
            "description": desc,
            "weights": wmap,
            "val_at_050": raw_val,
            "test_at_050": raw_test,
            "val_5fold_oof_platt": platt_res["val_oof_metrics"],
            "test_frozen_platt": platt_res["test_frozen_metrics"],
            "frozen_threshold": platt_res["frozen_threshold"],
            "platt_params": platt_res["cal_params"],
        }
        ens_preds_val_for_err[ename] = (
            platt_res["val_oof_probs"],
            platt_res["val_oof_metrics"]["mean_oof_threshold"],
        )
        ens_preds_test_for_mc[f"{ename}-PLATT"] = (
            platt_res["test_frozen_probs"] >= platt_res["frozen_threshold"]
        ).astype(int)
        print(
            f"[V3-ENS] {ename:22s} | Val@0.50={raw_val['macro_f1']:.4f} | "
            f"Val-OOF-Platt={platt_res['val_oof_metrics']['macro_f1']:.4f} "
            f"(TOX={platt_res['val_oof_metrics']['f1_toxic']:.4f}, "
            f"ViCTSD={platt_res['val_oof_metrics']['by_source']['victsd']['macro_f1']:.4f}, "
            f"ECE={platt_res['val_oof_metrics']['ece']:.4f}) | "
            f"Test-Platt={platt_res['test_frozen_metrics']['macro_f1']:.4f} "
            f"(TOX={platt_res['test_frozen_metrics']['f1_toxic']:.4f}, "
            f"ViCTSD={platt_res['test_frozen_metrics']['by_source']['victsd']['macro_f1']:.4f}, "
            f"ECE={platt_res['test_frozen_metrics']['ece']:.4f})",
            flush=True,
        )

    # 7. Diagnostic 5-Fold Val-OOF Logistic Regression Stacking Meta-Learner
    X_va_stack = np.column_stack([p_va_pho_3s, p_va_xlmr, p_va_bigru, p_va_tri, p_va_n6])
    X_te_stack = np.column_stack([p_te_pho_3s, p_te_xlmr, p_te_bigru, p_te_tri, p_te_n6])
    strat_key = [f"{s}_{l}" for s, l in zip(src_val, y_val)]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_stack_p = np.zeros(len(y_val), dtype=np.float64)
    oof_stack_pred = np.zeros(len(y_val), dtype=int)
    stack_thrs = []
    for tr_idx, va_idx in skf.split(X_va_stack, strat_key):
        meta = LogisticRegression(C=1.0, max_iter=1000)
        meta.fit(X_va_stack[tr_idx], y_val[tr_idx])
        p_tr_m = meta.predict_proba(X_va_stack[tr_idx])[:, 1]
        p_va_m = meta.predict_proba(X_va_stack[va_idx])[:, 1]
        sw = sweep_probability_thresholds(y_val[tr_idx], p_tr_m)
        thr = float(sw.get("best_threshold_macro_f1", sw.get("best_threshold", 0.40)))
        stack_thrs.append(thr)
        oof_stack_p[va_idx] = p_va_m
        oof_stack_pred[va_idx] = (p_va_m >= thr).astype(int)
    meta_full = LogisticRegression(C=1.0, max_iter=1000)
    meta_full.fit(X_va_stack, y_val)
    p_va_full_m = meta_full.predict_proba(X_va_stack)[:, 1]
    sw_m = sweep_probability_thresholds(y_val, p_va_full_m)
    thr_m = float(sw_m.get("best_threshold_macro_f1", sw_m.get("best_threshold", 0.40)))
    p_te_stack = meta_full.predict_proba(X_te_stack)[:, 1]
    val_stack_m = compute_binary_metrics(y_val.tolist(), oof_stack_pred.tolist(), src_val, oof_stack_p)
    val_stack_m["ece"] = round(float(compute_ece(y_val, oof_stack_p, n_bins=10)), 6)
    val_stack_m["brier_score"] = round(float(brier_score_loss(y_val, oof_stack_p)), 6)
    val_stack_m["mean_oof_threshold"] = round(float(np.mean(stack_thrs)), 4)
    test_stack_m = _eval_probs(y_test, p_te_stack, src_test, threshold=thr_m)
    ensembles_out["V3-ENS-CROSS-STACKING-LR"] = {
        "description": "Diagnostic 5-Fold Val-OOF Logistic Regression Stacking Meta-Learner (5 families)",
        "features": ["V3-PHOBERT-V2", "V3-XLMR-BASE", "V3-BIGRU-GATED-FT", "V3-TRI-GATED-FT", "EXP-N6-C-calibrated"],
        "meta_coefs": [round(float(c), 4) for c in meta_full.coef_[0]],
        "meta_intercept": round(float(meta_full.intercept_[0]), 4),
        "val_5fold_oof": val_stack_m,
        "test_frozen": test_stack_m,
        "frozen_threshold": round(thr_m, 4),
    }
    ens_preds_test_for_mc["V3-ENS-CROSS-STACKING-LR"] = (p_te_stack >= thr_m).astype(int)
    print(
        f"[V3-ENS] V3-ENS-CROSS-STACKING  | Val-OOF={val_stack_m['macro_f1']:.4f} "
        f"(TOX={val_stack_m['f1_toxic']:.4f}, ViCTSD={val_stack_m['by_source']['victsd']['macro_f1']:.4f}, "
        f"ECE={val_stack_m['ece']:.4f}) | Test={test_stack_m['macro_f1']:.4f} "
        f"(TOX={test_stack_m['f1_toxic']:.4f}, ViCTSD={test_stack_m['by_source']['victsd']['macro_f1']:.4f}, "
        f"ECE={test_stack_m['ece']:.4f})",
        flush=True,
    )

    # 8. Bootstrap 95% CI & McNemar on Historical Benchmark Test
    stats_test = _bootstrap_ci_and_mcnemar(
        y_test,
        ens_preds_test_for_mc,
        pairs=[
            ("V3-PHOBERT-V2", "V3-BIGRU-GATED-FT"),
            ("V3-ENS-TV4-ALL-PLATT", "V3-PHOBERT-V2"),
            ("V3-ENS-CROSS-VOTING-PLATT", "V3-PHOBERT-V2"),
            ("V3-ENS-CROSS-VOTING-PLATT", "EXP-N6-C-calibrated"),
            ("V3-ENS-CROSS-STACKING-LR", "V3-PHOBERT-V2"),
            ("V3-ENS-SCRATCH-INTRA-PLATT", "V3-BIGRU-GATED-FT"),
        ],
    )

    # 9. Error Analysis comparison on Validation
    p_va_cross_pl, thr_cross_pl = ens_preds_val_for_err["V3-ENS-CROSS-VOTING"]
    err_comp = _error_analysis_comparison(
        val_df, p_va_pho_3s, p_va_bigru, p_va_cross_pl, thr_cross_pl
    )

    # 10. Generate 4 PNG charts
    plots = _generate_v3_charts(
        single_summary=single_summary,
        ensembles_out=ensembles_out,
        diversity_pairs=diversity_pairs,
        div_names=div_names,
        y_val=y_val,
        p_va_pho=p_va_pho_3s,
        p_va_cross_raw=ensemble_defs["V3-ENS-CROSS-VOTING"][1],
        p_va_cross_platt=p_va_cross_pl,
    )

    report = {
        "experiment": "TV4-V3-Final-Evaluation",
        "val_parquet_sha256": _sha256_file(DATA_DIR / "val.parquet"),
        "test_parquet_sha256": _sha256_file(DATA_DIR / "test.parquet"),
        "test_blind": False,
        "n6_val_repro_max_diff": n6_diff,
        "targets": {
            "single_val_macro_f1": 0.790,
            "ensemble_val_macro_f1": 0.805,
            "f1_toxic": 0.700,
            "cross_ensemble_val_macro_f1": 0.815,
            "victsd_macro_f1": 0.710,
            "calibration_ece": 0.015,
        },
        "single_models": single_summary,
        "pairwise_error_diversity_val_seed42": diversity_pairs,
        "calibration_comparison_phobert_3s": cal_bench,
        "ensembles": ensembles_out,
        "test_bootstrap_95ci_and_mcnemar": stats_test,
        "error_analysis_comparison_val": err_comp,
        "plots": plots,
    }
    (OUT_DIR / "tv4_v3_final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"V3 Final Report saved: {OUT_DIR / 'tv4_v3_final_report.json'}", flush=True)
    return report


def _agg_single(name: str, family: str, per_seed: dict[str, Any]) -> dict[str, Any]:
    seeds = list(per_seed.keys())
    v_f1 = [per_seed[s]["val_050"]["macro_f1"] for s in seeds]
    v_tox = [per_seed[s]["val_050"]["f1_toxic"] for s in seeds]
    v_vic = [per_seed[s]["val_050"]["by_source"]["victsd"]["macro_f1"] for s in seeds]
    v_vih = [per_seed[s]["val_050"]["by_source"]["vihsd"]["macro_f1"] for s in seeds]
    v_pr = [per_seed[s]["val_050"]["pr_auc"] for s in seeds]
    v_ece = [per_seed[s]["val_050"]["ece"] for s in seeds]
    v_br = [per_seed[s]["val_050"]["brier_score"] for s in seeds]
    v_ll = [per_seed[s]["val_050"]["log_loss"] for s in seeds]

    t_f1 = [per_seed[s]["test_050"]["macro_f1"] for s in seeds]
    t_tox = [per_seed[s]["test_050"]["f1_toxic"] for s in seeds]
    t_vic = [per_seed[s]["test_050"]["by_source"]["victsd"]["macro_f1"] for s in seeds]
    t_vih = [per_seed[s]["test_050"]["by_source"]["vihsd"]["macro_f1"] for s in seeds]
    t_pr = [per_seed[s]["test_050"]["pr_auc"] for s in seeds]
    t_ece = [per_seed[s]["test_050"]["ece"] for s in seeds]

    return {
        "variant": name,
        "family": family,
        "n_seeds": len(seeds),
        "val_macro_f1_mean": round(float(np.mean(v_f1)), 4),
        "val_macro_f1_std": round(float(np.std(v_f1)), 4),
        "val_f1_toxic_mean": round(float(np.mean(v_tox)), 4),
        "val_f1_toxic_std": round(float(np.std(v_tox)), 4),
        "val_victsd_mean": round(float(np.mean(v_vic)), 4),
        "val_vihsd_mean": round(float(np.mean(v_vih)), 4),
        "val_pr_auc_mean": round(float(np.mean(v_pr)), 4),
        "val_raw_ece_mean": round(float(np.mean(v_ece)), 4),
        "val_brier_mean": round(float(np.mean(v_br)), 4),
        "val_log_loss_mean": round(float(np.mean(v_ll)), 4),
        "test_macro_f1_mean": round(float(np.mean(t_f1)), 4),
        "test_macro_f1_std": round(float(np.std(t_f1)), 4),
        "test_f1_toxic_mean": round(float(np.mean(t_tox)), 4),
        "test_victsd_mean": round(float(np.mean(t_vic)), 4),
        "test_vihsd_mean": round(float(np.mean(t_vih)), 4),
        "test_pr_auc_mean": round(float(np.mean(t_pr)), 4),
        "test_raw_ece_mean": round(float(np.mean(t_ece)), 4),
        "num_params": per_seed[seeds[0]]["num_params"],
        "peak_vram_mb_mean": round(float(np.mean([per_seed[s]["peak_vram_mb"] for s in seeds])), 1),
        "train_time_s_mean": round(float(np.mean([per_seed[s]["train_time_s"] for s in seeds])), 1),
        "per_seed": per_seed,
    }


def _generate_v3_charts(
    *,
    single_summary: dict[str, Any],
    ensembles_out: dict[str, Any],
    diversity_pairs: list[dict[str, Any]],
    div_names: list[str],
    y_val: np.ndarray,
    p_va_pho: np.ndarray,
    p_va_cross_raw: np.ndarray,
    p_va_cross_platt: np.ndarray,
) -> dict[str, str]:
    plt = _configure_matplotlib()
    plots: dict[str, str] = {}

    # 1. V1 -> V2 -> V3 Progression Bar Chart
    items = [
        ("V1: HYBRID-RNN-01\n(Scratch)", 0.7733, 0.7855, 0.6149),
        ("V2: BIGRU-ATTN-MAX\n(Scratch)", 0.7855, 0.7925, 0.6333),
        ("V3: BIGRU-GATED-FT\n(Scratch)", single_summary["V3-BIGRU-GATED-FT"]["val_macro_f1_mean"], single_summary["V3-BIGRU-GATED-FT"]["test_macro_f1_mean"], single_summary["V3-BIGRU-GATED-FT"]["val_f1_toxic_mean"]),
        ("V3: TRI-GATED-FT\n(Scratch)", single_summary["V3-TRI-GATED-FT"]["val_macro_f1_mean"], single_summary["V3-TRI-GATED-FT"]["test_macro_f1_mean"], single_summary["V3-TRI-GATED-FT"]["val_f1_toxic_mean"]),
        ("V3: SCRATCH-ENS\n(4 Scratch)", ensembles_out["V3-ENS-SCRATCH-INTRA"]["val_5fold_oof_platt"]["macro_f1"], ensembles_out["V3-ENS-SCRATCH-INTRA"]["test_frozen_platt"]["macro_f1"], ensembles_out["V3-ENS-SCRATCH-INTRA"]["val_5fold_oof_platt"]["f1_toxic"]),
        ("V3: PHOBERT-V2\n(Pretrained)", single_summary["V3-PHOBERT-V2"]["val_macro_f1_mean"], single_summary["V3-PHOBERT-V2"]["test_macro_f1_mean"], single_summary["V3-PHOBERT-V2"]["val_f1_toxic_mean"]),
        ("V3: TV4-ENS-ALL\n(Platt)", ensembles_out["V3-ENS-TV4-ALL"]["val_5fold_oof_platt"]["macro_f1"], ensembles_out["V3-ENS-TV4-ALL"]["test_frozen_platt"]["macro_f1"], ensembles_out["V3-ENS-TV4-ALL"]["val_5fold_oof_platt"]["f1_toxic"]),
        ("V3: CROSS-ENS\n(TV4+TV2 Platt)", ensembles_out["V3-ENS-CROSS-VOTING"]["val_5fold_oof_platt"]["macro_f1"], ensembles_out["V3-ENS-CROSS-VOTING"]["test_frozen_platt"]["macro_f1"], ensembles_out["V3-ENS-CROSS-VOTING"]["val_5fold_oof_platt"]["f1_toxic"]),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=160)
    x = np.arange(len(items))
    w = 0.27
    ax.bar(x - w, [i[1] for i in items], w, label="Validation Macro-F1", color="#2B6CB0")
    ax.bar(x, [i[2] for i in items], w, label="Historical Test Macro-F1", color="#0F766E")
    ax.bar(x + w, [i[3] for i in items], w, label="Validation F1-TOXIC", color="#D97706")
    ax.axhline(0.790, color="#2B6CB0", linestyle="--", linewidth=1.0, alpha=0.65, label="Target Single (0.790)")
    ax.axhline(0.815, color="#0F766E", linestyle=":", linewidth=1.2, alpha=0.8, label="Target Cross-Ens (0.815)")
    ax.axhline(0.700, color="#D97706", linestyle="-.", linewidth=1.0, alpha=0.7, label="Target F1-TOXIC (0.700)")
    for xi, it in zip(x, items):
        ax.text(xi - w, it[1] + 0.004, f"{it[1]:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax.text(xi, it[2] + 0.004, f"{it[2]:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(xi + w, it[3] + 0.004, f"{it[3]:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([i[0] for i in items], fontsize=8.2)
    ax.set_ylim(0.55, 0.90)
    ax.set_ylabel("Score")
    ax.set_title("V1 → V2 → V3 Progression: Scratch Neural vs Pretrained Transformer & Cross-Ensemble")
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    p1 = OUT_DIR / "v1_v2_v3_progression.png"
    fig.tight_layout()
    fig.savefig(p1)
    plt.close(fig)
    plots["v1_v2_v3_progression"] = str(p1)

    # 2. Error Diversity Heatmap
    n_d = len(div_names)
    mat = np.zeros((n_d, n_d))
    for p in diversity_pairs:
        i, j = div_names.index(p["model_a"]), div_names.index(p["model_b"])
        mat[i, j] = p["disagreement_rate"] * 100.0
        mat[j, i] = p["disagreement_rate"] * 100.0
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=160)
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=12)
    ax.set_xticks(range(n_d))
    ax.set_yticks(range(n_d))
    ax.set_xticklabels(div_names, rotation=25, ha="right", fontsize=8.5)
    ax.set_yticklabels(div_names, fontsize=8.5)
    for i in range(n_d):
        for j in range(n_d):
            ax.text(j, i, f"{mat[i, j]:.2f}%", ha="center", va="center", fontsize=8.5, color="white" if mat[i, j] > 6.5 else "black")
    fig.colorbar(im, ax=ax, label="Disagreement Rate (%) on Validation (n=6,048)")
    ax.set_title("V3 Pairwise Prediction Disagreement Across 5 Model Families")
    p2 = OUT_DIR / "v3_error_diversity_heatmap.png"
    fig.tight_layout()
    fig.savefig(p2)
    plt.close(fig)
    plots["v3_error_diversity_heatmap"] = str(p2)

    # 3. Calibration Reliability Diagram
    fig, ax = plt.subplots(figsize=(6.8, 5.5), dpi=160)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.1, label="Perfect Calibration")
    for lbl, probs, col in [
        ("V3-PHOBERT-V2 Raw Softmax", p_va_pho, "#DC2626"),
        ("V3-ENS-CROSS Raw Voting", p_va_cross_raw, "#D97706"),
        ("V3-ENS-CROSS Val-OOF Platt", p_va_cross_platt, "#0F766E"),
    ]:
        bins = np.linspace(0.0, 1.0, 11)
        xs, ys = [], []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            mask = (probs >= b0) & (probs < b1 if b1 < 1.0 else probs <= b1)
            if mask.sum() > 0:
                xs.append(float(probs[mask].mean()))
                ys.append(float(y_val[mask].mean()))
        ece_v = compute_ece(y_val, probs, n_bins=10)
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=col, label=f"{lbl} (ECE={ece_v:.4f})")
    ax.set_xlabel("Predicted Probability P(TOXIC)")
    ax.set_ylabel("Empirical Positive Rate")
    ax.set_title("V3 Calibration Reliability Diagram (Validation n=6,048)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.3)
    p3 = OUT_DIR / "v3_calibration_reliability.png"
    fig.tight_layout()
    fig.savefig(p3)
    plt.close(fig)
    plots["v3_calibration_reliability"] = str(p3)

    # 4. ViCTSD vs ViHSD Source Comparison
    src_items = [
        ("V1 HYBRID-RNN-01", 0.6542, 0.7961),
        ("V2 BIGRU-ATTN-MAX", 0.6593, 0.8071),
        ("V3 TRI-GATED-FT", single_summary["V3-TRI-GATED-FT"]["val_victsd_mean"], single_summary["V3-TRI-GATED-FT"]["val_vihsd_mean"]),
        ("V3 XLMR-BASE (s42)", single_summary["V3-XLMR-BASE"]["val_victsd_mean"], single_summary["V3-XLMR-BASE"]["val_vihsd_mean"]),
        ("V3 PHOBERT-V2 (3s)", single_summary["V3-PHOBERT-V2"]["val_victsd_mean"], single_summary["V3-PHOBERT-V2"]["val_vihsd_mean"]),
        ("V3 TV4-ENS-ALL", ensembles_out["V3-ENS-TV4-ALL"]["val_5fold_oof_platt"]["by_source"]["victsd"]["macro_f1"], ensembles_out["V3-ENS-TV4-ALL"]["val_5fold_oof_platt"]["by_source"]["vihsd"]["macro_f1"]),
        ("V3 CROSS-ENS", ensembles_out["V3-ENS-CROSS-VOTING"]["val_5fold_oof_platt"]["by_source"]["victsd"]["macro_f1"], ensembles_out["V3-ENS-CROSS-VOTING"]["val_5fold_oof_platt"]["by_source"]["vihsd"]["macro_f1"]),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=160)
    x = np.arange(len(src_items))
    w = 0.35
    ax.bar(x - w / 2, [s[1] for s in src_items], w, label="ViCTSD Macro-F1 (News, n=1,489)", color="#7C3AED")
    ax.bar(x + w / 2, [s[2] for s in src_items], w, label="ViHSD Macro-F1 (Social, n=4,559)", color="#0284C7")
    ax.axhline(0.710, color="#7C3AED", linestyle="--", linewidth=1.2, label="Target ViCTSD (0.710)")
    for xi, it in zip(x, src_items):
        ax.text(xi - w / 2, it[1] + 0.005, f"{it[1]:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(xi + w / 2, it[2] + 0.005, f"{it[2]:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in src_items], rotation=15, ha="right", fontsize=8.5)
    ax.set_ylim(0.58, 0.90)
    ax.set_ylabel("Validation Macro-F1")
    ax.set_title("Domain Generalization Breakthrough: ViCTSD (Hard News) vs ViHSD (Social)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    p4 = OUT_DIR / "v3_source_comparison.png"
    fig.tight_layout()
    fig.savefig(p4)
    plt.close(fig)
    plots["v3_source_comparison"] = str(p4)
    return plots


if __name__ == "__main__":
    run_v3_final_evaluation()
