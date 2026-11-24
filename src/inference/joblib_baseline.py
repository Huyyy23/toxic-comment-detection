"""Local inference adapter for the N6 binary Logistic Regression pipeline."""
from pathlib import Path
from time import perf_counter

import joblib

from src.features.preprocessing import preprocess


class JoblibBaselineAdapter:
    model_backend = "sklearn"
    threshold = 0.5
    is_calibrated = False

    def __init__(self, artifact_path: Path):
        self.pipeline = joblib.load(artifact_path)
        classes = {int(label) for label in self.pipeline.classes_}
        if classes != {0, 1} or not hasattr(self.pipeline, "predict_proba"):
            raise ValueError("Baseline artifact must expose CLEAN/TOXIC predict_proba classes")

    def predict(self, text: str) -> dict[str, object]:
        started = perf_counter()
        probabilities = self.pipeline.predict_proba([preprocess(text, variant="teencode")])[0]
        class_probabilities = {
            int(label): float(probability)
            for label, probability in zip(self.pipeline.classes_, probabilities)
        }
        toxic_probability = class_probabilities[1]
        label_id = int(toxic_probability >= self.threshold)
        named_probabilities = {
            "CLEAN": class_probabilities[0],
            "TOXIC": toxic_probability,
        }
        return {
            "label_id": label_id,
            "label_name": "TOXIC" if label_id else "CLEAN",
            "confidence": named_probabilities["TOXIC" if label_id else "CLEAN"],
            "probabilities": named_probabilities,
            "is_calibrated": self.is_calibrated,
            "threshold": self.threshold,
            "latency_ms": (perf_counter() - started) * 1000,
            "model_backend": self.model_backend,
        }

    def info(self) -> dict[str, object]:
        return {
            "model_name": "N6-LR-D seed 42 (raw validation baseline)",
            "model_backend": self.model_backend,
            "label_taxonomy": {0: "CLEAN", 1: "TOXIC"},
            "is_calibrated": self.is_calibrated,
            "calibration_method": None,
            "threshold": self.threshold,
            "artifact_version": "n6-lr-d-seed-42-raw",
            "status": "loaded",
        }
