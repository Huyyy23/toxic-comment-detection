from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from gensim.models import FastText
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.svm import LinearSVC
from src.tracking.mlflow_tracker import MLflowTracker

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_PATH = ROOT / "data" / "processed" / "binary_n2_v1" / "train.parquet"
DEFAULT_VAL_PATH = ROOT / "data" / "processed" / "binary_n2_v1" / "val.parquet"
SEEDS = (42, 123, 2026)
CONFIG: dict[str, Any] = {
    "vector_size": 100,
    "window": 5,
    "min_count": 2,
    "epochs": 20,
    "workers": 1,
    "pooling": "mean",
    "lr_C": 1.0,
    "lr_class_weight": "balanced",
    "lr_max_iter": 1000,
    "lr_solver": "lbfgs",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frame(path: Path, expected_split: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"id", "text", "label", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    if (
        frame.empty
        or frame["split"].isna().any()
        or set(frame["split"].unique()) != {expected_split}
    ):
        raise ValueError(f"{path}: expected only split={expected_split!r}")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError(f"{path}: IDs must be present and unique")
    if frame["text"].isna().any() or frame["text"].map(lambda text: not isinstance(text, str) or not text.strip()).any():
        raise ValueError(f"{path}: text must be nonblank strings")
    labels = frame["label"]
    if labels.isna().any() or not labels.isin([0, 1]).all() or labels.map(lambda label: isinstance(label, (bool, np.bool_))).any():
        raise ValueError(f"{path}: labels must be binary integers 0 or 1")
    if set(labels.astype(int)) != {0, 1}:
        raise ValueError(f"{path}: both binary classes must be present")
    return frame[["id", "text", "label"]].copy()


def _pool(model: FastText, texts: list[str]) -> np.ndarray:
    features = np.zeros((len(texts), int(model.vector_size)), dtype=np.float32)
    for row, text in enumerate(texts):
        vectors = []
        for token in text.split():
            try:
                vector = model.wv[token]
            except KeyError:
                continue
            if np.any(vector):
                vectors.append(vector)
        if vectors:
            features[row] = np.mean(vectors, axis=0, dtype=np.float32)
    return features


def _run_seed(
    seed: int,
    train: pd.DataFrame,
    val: pd.DataFrame,
    hashes: dict[str, str],
    seed_dir: Path,
    tracking_uri: str | None,
) -> dict[str, Any]:
    seed_dir.mkdir(parents=True, exist_ok=False)
    config = {**CONFIG, "seed": seed}
    config_path = seed_dir / "config.json"
    hash_path = seed_dir / "dataset_hashes.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    hash_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    tracker = MLflowTracker(experiment_name="N5-FastText-mean-LR", tracking_uri=tracking_uri)
    with tracker.run(run_name=f"N5-seed-{seed}"):
        tracker.log_params(config)
        tracker.log_params(hashes)
        started = time.perf_counter()
        corpus = [text.split() for text in train["text"].tolist()]
        fasttext = FastText(
            sentences=corpus,
            vector_size=config["vector_size"],
            window=config["window"],
            min_count=config["min_count"],
            epochs=config["epochs"],
            workers=config["workers"],
            seed=seed,
        )
        x_train = _pool(fasttext, train["text"].tolist())
        x_val = _pool(fasttext, val["text"].tolist())
        classifier = LogisticRegression(
            C=config["lr_C"],
            class_weight=config["lr_class_weight"],
            max_iter=config["lr_max_iter"],
            solver=config["lr_solver"],
            random_state=seed,
        )
        classifier.fit(x_train, train["label"].astype(int).tolist())
        predictions = classifier.predict(x_val).astype(int).tolist()
        duration = time.perf_counter() - started
        truth = val["label"].astype(int).tolist()
        metrics = {
            "macro_f1": float(f1_score(truth, predictions, labels=[0, 1], average="macro", zero_division=0)),
            "f1_clean": float(f1_score(truth, predictions, labels=[0, 1], average=None, zero_division=0)[0]),
            "f1_toxic": float(f1_score(truth, predictions, labels=[0, 1], average=None, zero_division=0)[1]),
            "training_duration_seconds": float(duration),
        }
        fasttext_path = seed_dir / "fasttext.model"
        lr_path = seed_dir / "logistic_regression.joblib"
        prediction_path = seed_dir / "val_predictions.csv"
        metrics_path = seed_dir / "metrics.json"
        run_id_path = seed_dir / "run_id.txt"
        fasttext.save(str(fasttext_path))
        joblib.dump(classifier, lr_path)
        pd.DataFrame({"id": val["id"], "true_label": truth, "predicted_label": predictions}).to_csv(prediction_path, index=False)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        tracker.log_metrics(metrics)
        for artifact in (config_path, hash_path, fasttext_path, lr_path, prediction_path, metrics_path):
            tracker.log_artifact(str(artifact))
        run_id = tracker.run_id
        if run_id is None:
            raise RuntimeError("MLflow tracker did not return a run ID")
        run_id_path.write_text(run_id + "\n", encoding="utf-8")
        tracker.log_artifact(str(run_id_path))
    return {"seed": seed, "macro_f1": metrics["macro_f1"], "run_id": run_id, "seed_dir": str(seed_dir)}


def _summarize(results: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    scores = np.asarray([result["macro_f1"] for result in results], dtype=np.float64)
    summary = {
        "seeds": results,
        "macro_f1": {"mean": float(scores.mean()), "std": float(scores.std(ddof=0))},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_experiment(
    train_path: Path = DEFAULT_TRAIN_PATH,
    val_path: Path = DEFAULT_VAL_PATH,
    output_dir: Path | None = None,
    tracking_uri: str | None = None,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    """Train FastText and logistic regression using train and validation parquet files."""
    train_path, val_path = Path(train_path), Path(val_path)
    train = _load_frame(train_path, "train")
    val = _load_frame(val_path, "val")
    hashes = {"train_parquet_sha256": _sha256_file(train_path), "val_parquet_sha256": _sha256_file(val_path)}
    run_dir = Path(output_dir) if output_dir is not None else ROOT / "models" / "baselines"
    run_dir = run_dir / f"N5-run-{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)
    results = [
        _run_seed(seed, train, val, hashes, run_dir / f"seed-{seed}", tracking_uri)
        for seed in seeds
    ]
    return _summarize(results, run_dir)


@dataclass(frozen=True)
class ToxicCandidate:
    name: str
    score_kind: Literal["probability", "margin"]
    default_cutoff: float
    estimator_kind: Literal["saved", "logistic_regression", "linear_svc"]
    params: dict[str, Any]


CANDIDATES: tuple[ToxicCandidate, ...] = (
    ToxicCandidate(
        "lr_balanced",
        "probability",
        0.5,
        "saved",
        {
            "C": CONFIG["lr_C"],
            "class_weight": CONFIG["lr_class_weight"],
            "max_iter": CONFIG["lr_max_iter"],
            "solver": CONFIG["lr_solver"],
        },
    ),
    ToxicCandidate(
        "lr_unweighted",
        "probability",
        0.5,
        "logistic_regression",
        {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "class_weight": None},
    ),
    ToxicCandidate(
        "linear_svc",
        "margin",
        0.0,
        "linear_svc",
        {"C": 1.0, "max_iter": 2000, "class_weight": "balanced"},
    ),
)


def select_toxic_cutoff(scores: Any, labels: Any, default_cutoff: float = 0.5) -> float:
    """Return the highest cutoff among validation TOXIC-F1 maxima."""
    score_values = np.asarray(scores, dtype=np.float64).ravel()
    label_values = np.asarray(labels).astype(int).ravel()
    if score_values.shape != label_values.shape or score_values.size == 0:
        raise ValueError("scores and labels must be nonempty arrays of equal shape")
    precision, recall, thresholds = precision_recall_curve(label_values, score_values, pos_label=1)
    curve_f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds, dtype=np.float64),
        where=(precision[:-1] + recall[:-1]) != 0,
    )
    default_predictions = score_values >= default_cutoff
    default_f1 = float(f1_score(label_values, default_predictions, pos_label=1, zero_division=0))
    all_negative_cutoff = float(np.nextafter(score_values.max(), np.inf))
    best_f1 = max(float(curve_f1.max()), default_f1, 0.0)
    best_cutoffs = [
        float(cutoff)
        for cutoff, f1_toxic in zip(thresholds, curve_f1)
        if np.isclose(f1_toxic, best_f1, rtol=0.0, atol=1e-12)
    ]
    if np.isclose(default_f1, best_f1, rtol=0.0, atol=1e-12):
        best_cutoffs.append(float(default_cutoff))
    if np.isclose(0.0, best_f1, rtol=0.0, atol=1e-12):
        best_cutoffs.append(all_negative_cutoff)
    return max(best_cutoffs)




def _toxic_metrics(y_true: list[int], y_pred: Any, cutoff: float) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    per_class = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "f1_clean": float(per_class[0]),
        "f1_toxic": float(per_class[1]),
        "precision_toxic": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_toxic": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "cutoff": float(cutoff),
    }


def _run_toxic_seed(
    seed: int,
    train: pd.DataFrame,
    val: pd.DataFrame,
    hashes: dict[str, str],
    baseline_run_dir: Path,
    seed_dir: Path,
    tracking_uri: str | None,
) -> dict[str, Any]:
    seed_dir.mkdir(parents=True, exist_ok=False)
    baseline_seed_dir = baseline_run_dir / f"seed-{seed}"
    if not baseline_seed_dir.is_dir():
        raise FileNotFoundError(f"missing baseline seed directory: {baseline_seed_dir}")
    baseline_config_path = baseline_seed_dir / "config.json"
    config = {
        "seed": seed,
        "baseline_run_dir": str(baseline_run_dir),
        "baseline_seed_dir": str(baseline_seed_dir),
        "fasttext_model_path": str(baseline_seed_dir / "fasttext.model"),
        "baseline_classifier_path": str(baseline_seed_dir / "logistic_regression.joblib"),
        "baseline_config_path": str(baseline_config_path),
        "baseline_config": json.loads(baseline_config_path.read_text(encoding="utf-8")),
        "candidates": [candidate.__dict__.copy() for candidate in CANDIDATES],
        "threshold_selection": "max validation F1 TOXIC; ties select the highest cutoff",
        "validation_selection_bias": "Thresholds are selected and scored on the same validation set; validation F1 is optimistic.",
    }
    candidates_path = seed_dir / "candidates.json"
    config_path = seed_dir / "config.json"
    hash_path = seed_dir / "dataset_hashes.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    hash_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    candidates_path.write_text(json.dumps([candidate.__dict__.copy() for candidate in CANDIDATES], indent=2), encoding="utf-8")
    fasttext = FastText.load(str(baseline_seed_dir / "fasttext.model"), mmap="r")
    x_train = _pool(fasttext, train["text"].tolist())
    x_val = _pool(fasttext, val["text"].tolist())
    del fasttext

    train_labels = train["label"].astype(int).tolist()
    val_labels = val["label"].astype(int).tolist()
    classifiers: dict[str, Any] = {}
    for candidate in CANDIDATES:
        if candidate.estimator_kind == "saved":
            classifier = joblib.load(baseline_seed_dir / "logistic_regression.joblib")
        elif candidate.estimator_kind == "logistic_regression":
            classifier = LogisticRegression(random_state=seed, **candidate.params)
            classifier.fit(x_train, train_labels)
        else:
            classifier = LinearSVC(random_state=seed, **candidate.params)
            classifier.fit(x_train, train_labels)
        classifiers[candidate.name] = classifier

    baseline_predictions = pd.read_csv(baseline_seed_dir / "val_predictions.csv")
    if baseline_predictions["id"].astype(str).tolist() != val["id"].astype(str).tolist():
        raise ValueError(f"seed {seed}: baseline val_predictions.csv IDs do not match validation frame")
    if baseline_predictions["true_label"].astype(int).tolist() != val_labels:
        raise ValueError(f"seed {seed}: baseline val_predictions.csv labels do not match validation frame")
    if baseline_predictions["predicted_label"].astype(int).tolist() != classifiers["lr_balanced"].predict(x_val).astype(int).tolist():
        raise ValueError(f"seed {seed}: saved balanced LogisticRegression predictions do not match val_predictions.csv")

    scores: dict[str, np.ndarray] = {}
    default_predictions: dict[str, np.ndarray] = {}
    for candidate in CANDIDATES:
        classifier = classifiers[candidate.name]
        score = classifier.predict_proba(x_val)[:, 1] if candidate.score_kind == "probability" else classifier.decision_function(x_val)
        scores[candidate.name] = np.asarray(score, dtype=np.float64)
        default_predictions[candidate.name] = np.asarray(classifier.predict(x_val), dtype=int)

    cutoffs: dict[str, float] = {}
    for candidate in CANDIDATES:
        cutoffs[candidate.name] = select_toxic_cutoff(scores[candidate.name], val_labels, candidate.default_cutoff)
    config["selected_thresholds"] = cutoffs
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    for candidate in CANDIDATES:
        if candidate.estimator_kind != "saved":
            joblib.dump(classifiers[candidate.name], seed_dir / f"{candidate.name}.joblib")

    score_frame = pd.DataFrame({"id": val["id"].tolist(), "true_label": val["label"].astype(int).tolist()})
    for candidate in CANDIDATES:
        score_frame[f"{candidate.name}_score"] = scores[candidate.name]
        score_frame[f"{candidate.name}_pred_default"] = default_predictions[candidate.name]
        score_frame[f"{candidate.name}_pred_tuned"] = (scores[candidate.name] >= cutoffs[candidate.name]).astype(int)
    scores_path = seed_dir / "val_scores.csv"
    score_frame.to_csv(scores_path, index=False)

    y_true = val["label"].astype(int).tolist()
    metrics_by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        default_metrics = _toxic_metrics(y_true, default_predictions[candidate.name].tolist(), candidate.default_cutoff)
        tuned_metrics = _toxic_metrics(y_true, (scores[candidate.name] >= cutoffs[candidate.name]).astype(int).tolist(), cutoffs[candidate.name])
        metrics_by_candidate[candidate.name] = {"default": default_metrics, "tuned": tuned_metrics}
        for mode, metrics in (("default", default_metrics), ("tuned", tuned_metrics)):
            rows.append({"candidate": candidate.name, "mode": mode, **metrics})
    metrics_path = seed_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"candidates": metrics_by_candidate, "rows": rows}, indent=2), encoding="utf-8")

    tracker = MLflowTracker(experiment_name="N5-toxic-F1-tuning", tracking_uri=tracking_uri)
    with tracker.run(run_name=f"N5-toxic-tuning-seed-{seed}"):
        tracker.log_params(
            {
                "seed": seed,
                "baseline_run_dir": str(baseline_run_dir),
                "fasttext_model_path": str(baseline_seed_dir / "fasttext.model"),
                "baseline_classifier_path": str(baseline_seed_dir / "logistic_regression.joblib"),
                "baseline_config_path": str(baseline_config_path),
                "threshold_selection": config["threshold_selection"],
            }
        )
        tracker.log_params(hashes)
        for candidate in CANDIDATES:
            params = {
                f"{candidate.name}_score_kind": candidate.score_kind,
                f"{candidate.name}_default_cutoff": candidate.default_cutoff,
                f"{candidate.name}_tuned_cutoff": cutoffs[candidate.name],
            }
            params.update(
                {
                    f"{candidate.name}_{key}": "none" if value is None else value
                    for key, value in candidate.params.items()
                }
            )
            tracker.log_params(params)
        per_seed_metrics = {
            f"{candidate.name}_{mode}_{key}": float(value)
            for candidate in CANDIDATES
            for mode in ("default", "tuned")
            for key, value in metrics_by_candidate[candidate.name][mode].items()
        }
        tracker.log_metrics(per_seed_metrics)
        for artifact in (config_path, hash_path, candidates_path, scores_path, metrics_path):
            tracker.log_artifact(str(artifact))
        for candidate in CANDIDATES:
            if candidate.estimator_kind != "saved":
                tracker.log_artifact(str(seed_dir / f"{candidate.name}.joblib"))
        run_id = tracker.run_id
        if run_id is None:
            raise RuntimeError("MLflow tracker did not return a run ID")
        run_id_path = seed_dir / "run_id.txt"
        run_id_path.write_text(run_id + "\n", encoding="utf-8")
        tracker.log_artifact(str(run_id_path))
    return {"seed": seed, "run_id": run_id, "seed_dir": str(seed_dir), "metrics": metrics_by_candidate, "cutoffs": cutoffs}


def run_toxic_tuning(
    baseline_run_dir: Path,
    train_path: Path = DEFAULT_TRAIN_PATH,
    val_path: Path = DEFAULT_VAL_PATH,
    output_dir: Path | None = None,
    tracking_uri: str | None = None,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    """Tune TOXIC decision cutoffs using only N2 train and validation data."""
    baseline_run_dir = Path(baseline_run_dir)
    train_path, val_path = Path(train_path), Path(val_path)
    train = _load_frame(train_path, "train")
    val = _load_frame(val_path, "val")
    hashes = {"train_parquet_sha256": _sha256_file(train_path), "val_parquet_sha256": _sha256_file(val_path)}
    base_output = Path(output_dir) if output_dir is not None else ROOT / "models" / "baselines"
    run_dir = base_output / f"N5-toxic-tuning-run-{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)
    results = [
        _run_toxic_seed(seed, train, val, hashes, baseline_run_dir, run_dir / f"seed-{seed}", tracking_uri)
        for seed in seeds
    ]
    metric_names = ["macro_f1", "f1_clean", "f1_toxic", "precision_toxic", "recall_toxic", "tp", "tn", "fp", "fn", "cutoff"]
    aggregate_candidates: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for candidate in CANDIDATES:
        aggregate_candidates[candidate.name] = {}
        for mode in ("default", "tuned"):
            aggregate_candidates[candidate.name][mode] = {}
            for metric in metric_names:
                values = np.asarray([result["metrics"][candidate.name][mode][metric] for result in results], dtype=np.float64)
                aggregate_candidates[candidate.name][mode][metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    summary = {
        "baseline_run_dir": str(baseline_run_dir),
        "run_dir": str(run_dir),
        "seeds": results,
        "candidates": aggregate_candidates,
        "validation_selection_bias": "Thresholds are selected and scored on the same validation set; validation F1 is optimistic.",
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tracker = MLflowTracker(experiment_name="N5-toxic-F1-tuning", tracking_uri=tracking_uri)
    with tracker.run(run_name="N5-toxic-tuning-summary"):
        summary_run_id = tracker.run_id
        if summary_run_id is None:
            raise RuntimeError("MLflow tracker did not return a summary run ID")
        summary["summary_run_id"] = summary_run_id
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary_metrics = {}
        for candidate in CANDIDATES:
            for mode in ("default", "tuned"):
                for metric in metric_names:
                    stats = aggregate_candidates[candidate.name][mode][metric]
                    summary_metrics[f"{candidate.name}_{mode}_{metric}_mean"] = stats["mean"]
                    summary_metrics[f"{candidate.name}_{mode}_{metric}_std"] = stats["std"]
        tracker.log_metrics(summary_metrics)
        tracker.log_artifact(str(summary_path))
    return summary
def main() -> None:
    parser = argparse.ArgumentParser(description="N5 FastText mean pooling with logistic regression")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "baselines")
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--tune-toxic-f1", action="store_true")
    parser.add_argument("--baseline-run-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.tune_toxic_f1:
        if args.baseline_run_dir is None:
            raise SystemExit("--baseline-run-dir is required with --tune-toxic-f1")
        summary = run_toxic_tuning(args.baseline_run_dir, args.train_path, args.val_path, args.output_dir, args.tracking_uri)
    else:
        summary = run_experiment(args.train_path, args.val_path, args.output_dir, args.tracking_uri)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
