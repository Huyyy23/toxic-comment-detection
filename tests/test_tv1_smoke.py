"""
Smoke test TV1 — kiểm tra các module không cần dữ liệu thực.

Chạy: .venv/Scripts/pytest tests/test_tv1_smoke.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import tempfile

import numpy as np
import pytest


SYNTHETIC_TRAIN = [
    {"id": "train_0", "text": "bình luận bình thường ko có vấn đề", "label": 0, "split": "train"},
    {"id": "train_1", "text": "nội dung sạch và bổ ích", "label": 0, "split": "train"},
    {"id": "train_2", "text": "mày ngu vcl không hiểu gì", "label": 1, "split": "train"},
    {"id": "train_3", "text": "con này xấu xí lắm đm", "label": 1, "split": "train"},
    {"id": "train_4", "text": "giết hết lũ chó này đi", "label": 2, "split": "train"},
    {"id": "train_5", "text": "nhóm này nên bị tiêu diệt", "label": 2, "split": "train"},
    {"id": "train_6", "text": "thích nội dung này lắm", "label": 0, "split": "train"},
    {"id": "train_7", "text": "tại sao mày ngu thế clgt", "label": 1, "split": "train"},
]
SYNTHETIC_VAL = [
    {"id": "val_0", "text": "nội dung bình thường dễ chịu", "label": 0, "split": "val"},
    {"id": "val_1", "text": "mày điên à vcl", "label": 1, "split": "val"},
    {"id": "val_2", "text": "phải tiêu diệt bọn chúng", "label": 2, "split": "val"},
]
ALL_RECORDS = SYNTHETIC_TRAIN + SYNTHETIC_VAL
MANIFEST = {r["id"]: r["split"] for r in ALL_RECORDS}


class TestPreprocessingSmoke:
    def test_all_variants(self):
        from src.features.preprocessing import preprocess, preprocess_series
        texts = [r["text"] for r in SYNTHETIC_TRAIN]
        for variant in ["raw", "basic", "teencode"]:
            results = preprocess_series(texts, variant=variant)
            assert len(results) == len(texts)
            assert all(isinstance(r, str) and r for r in results)

    def test_teencode_dict_coverage(self):
        from src.features.preprocessing import TEENCODE_DICT, preprocess
        # Một số token cơ bản phải được thay thế
        for tok, expected in [("ko", "không"), ("dc", "được"), ("mn", "mọi người")]:
            result = preprocess(tok, variant="teencode")
            assert expected in result, f"Expected '{expected}' for tok '{tok}', got '{result}'"


class TestTfidfPipelineSmoke:
    def test_fit_transform_pipeline(self):
        from src.features.preprocessing import preprocess_series
        from src.features.tfidf_pipeline import build_tfidf, fit_tfidf, transform_texts

        train_texts = preprocess_series(
            [r["text"] for r in SYNTHETIC_TRAIN], variant="basic"
        )
        val_texts = preprocess_series(
            [r["text"] for r in SYNTHETIC_VAL], variant="basic"
        )

        for mode in ["word", "char", "word+char"]:
            t = build_tfidf(mode=mode, max_features=500, min_df=1)
            fitted = fit_tfidf(t, train_texts)
            X_train = transform_texts(fitted, train_texts)
            X_val = transform_texts(fitted, val_texts)
            assert X_train.shape[0] == len(SYNTHETIC_TRAIN)
            assert X_val.shape[0] == len(SYNTHETIC_VAL)
            assert X_train.shape[1] == X_val.shape[1]  # same feature space


class TestLRPipelineSmoke:
    def _write_jsonl(self, path, records):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def test_all_variants_smoke(self):
        from src.models.lr_pipeline import run_experiment

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_path = tmpdir / "vihsd.jsonl"
            self._write_jsonl(data_path, ALL_RECORDS)

            for variant in ["A", "B", "C"]:
                result = run_experiment(
                    variant=variant,
                    data_path=data_path,
                    artifacts_dir=tmpdir / "models",
                    tracking_uri=f"file:///{(tmpdir / 'mlruns').as_posix()}",
                )
                assert "metrics" in result
                assert "macro_f1" in result["metrics"]
                # Macro-F1 phải nằm trong [0, 1]
                assert 0.0 <= result["metrics"]["macro_f1"] <= 1.0
                # Phải có predictions file
                pred_path = Path(result["artifacts_dir"]) / "val_predictions.json"
                assert pred_path.exists()
                preds = json.loads(pred_path.read_text(encoding="utf-8"))
                # Số predictions phải bằng số val records
                assert len(preds) == len(SYNTHETIC_VAL)
                # Mỗi prediction phải có đủ fields
                for p in preds:
                    assert "id" in p
                    assert "true_label" in p
                    assert "predicted_label" in p
                    assert "probabilities" in p
                    proba = p["probabilities"]
                    assert set(proba.keys()) == {"CLEAN", "OFFENSIVE", "HATE"}
                    # Tổng xác suất xấp xỉ 1
                    total = sum(proba.values())
                    assert abs(total - 1.0) < 1e-4, f"Probabilities sum={total}"

    def test_metrics_structure(self):
        from src.models.lr_pipeline import run_experiment

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_path = tmpdir / "vihsd.jsonl"
            self._write_jsonl(data_path, ALL_RECORDS)

            result = run_experiment(
                variant="A",
                data_path=data_path,
                artifacts_dir=tmpdir / "models",
                tracking_uri=f"file:///{(tmpdir / 'mlruns').as_posix()}",
            )
            m = result["metrics"]
            assert "per_class" in m
            assert set(m["per_class"].keys()) == {"CLEAN", "OFFENSIVE", "HATE"}
            for label_name, pc in m["per_class"].items():
                assert "precision" in pc and "recall" in pc and "f1" in pc
            assert "confusion_matrix" in m
            assert len(m["confusion_matrix"]) == 3  # 3x3
            assert m["model_size_bytes"] > 0

    def test_rejects_test_split(self):
        from src.models.lr_pipeline import split_data

        records = ALL_RECORDS + [{"id": "test_0", "text": "test row", "label": 0, "split": "test"}]
        with pytest.raises(ValueError, match="test"):
            split_data(records)

    def test_predictions_csv_and_load_and_predict(self):
        import pandas as pd
        from src.models.lr_pipeline import load_and_predict, run_experiment

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_path = tmpdir / "train_val.jsonl"
            self._write_jsonl(data_path, ALL_RECORDS)

            result = run_experiment(
                variant="B",
                data_path=data_path,
                artifacts_dir=tmpdir / "models",
                tracking_uri=f"file:///{(tmpdir / 'mlruns').as_posix()}",
            )
            art_dir = Path(result["artifacts_dir"])
            csv_path = art_dir / "predictions_val.csv"
            assert csv_path.exists()
            df_pred = pd.read_csv(csv_path)
            assert list(df_pred.columns) == [
                "sample_id", "y_true", "y_pred", "prob_CLEAN", "prob_OFFENSIVE", "prob_HATE"
            ]
            assert len(df_pred) == len(SYNTHETIC_VAL)

            val_texts = [r["text"] for r in SYNTHETIC_VAL]
            loaded_preds, loaded_proba = load_and_predict(art_dir, val_texts)
            assert loaded_preds == df_pred["y_pred"].tolist()
            assert loaded_proba.shape == (len(SYNTHETIC_VAL), 3)


class TestEdaUtilsSmoke:
    def test_load_jsonl_to_df(self):
        import pandas as pd
        from src.eda.eda_utils import load_jsonl_to_df

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for r in ALL_RECORDS:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            df = load_jsonl_to_df(path)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == len(ALL_RECORDS)
            assert "label_name" in df.columns

    def test_check_duplicates_no_overlap(self):
        from src.eda.eda_utils import check_duplicates
        import pandas as pd

        df = pd.DataFrame(ALL_RECORDS)
        df["label_name"] = df["label"].map({0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"})
        result = check_duplicates(df)
        assert result["cross_split_overlap"] == 0

    def test_detect_patterns(self):
        from src.eda.eda_utils import detect_patterns
        import pandas as pd

        df = pd.DataFrame([
            {"id": "1", "text": "xem tại https://example.com", "label": 0, "split": "train"},
            {"id": "2", "text": "bình thường 😂", "label": 0, "split": "train"},
        ])
        df["label_name"] = "CLEAN"
        df = detect_patterns(df)
        assert df.loc[0, "has_url"] == True
        assert df.loc[1, "has_emoji"] == True
