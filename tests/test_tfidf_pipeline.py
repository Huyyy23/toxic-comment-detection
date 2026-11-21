"""
Unit tests cho TF-IDF pipeline — TV1.
Chạy: .venv/Scripts/pytest tests/test_tfidf_pipeline.py -v
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from src.features.tfidf_pipeline import (
    build_tfidf,
    fit_tfidf,
    load_tfidf,
    save_tfidf,
    text_hash,
    transform_texts,
)


SAMPLE_TRAIN = [
    "bình luận bình thường",
    "nội dung sạch",
    "thông tin thú vị",
    "bài viết hay",
    "nội dung bổ ích",
]
SAMPLE_VAL = [
    "bình luận khác",
    "nội dung mới",
]


class TestBuildTfidf:
    def test_word_mode(self):
        t = build_tfidf(mode="word")
        assert t is not None

    def test_char_mode(self):
        t = build_tfidf(mode="char")
        assert t is not None

    def test_word_char_mode(self):
        from sklearn.pipeline import FeatureUnion
        t = build_tfidf(mode="word+char")
        assert isinstance(t, FeatureUnion)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            build_tfidf(mode="invalid")  # type: ignore


class TestFitTransform:
    def test_fit_returns_transformer(self):
        t = build_tfidf(mode="word")
        fitted = fit_tfidf(t, SAMPLE_TRAIN)
        assert fitted is not None

    def test_transform_shape(self):
        t = build_tfidf(mode="word")
        fitted = fit_tfidf(t, SAMPLE_TRAIN)
        X = transform_texts(fitted, SAMPLE_TRAIN)
        assert X.shape[0] == len(SAMPLE_TRAIN)
        assert X.shape[1] > 0

    def test_val_transform_same_features(self):
        """Val transform phải dùng vocabulary từ train (không fit lại)."""
        t = build_tfidf(mode="word")
        fitted = fit_tfidf(t, SAMPLE_TRAIN)
        X_train = transform_texts(fitted, SAMPLE_TRAIN)
        X_val = transform_texts(fitted, SAMPLE_VAL)
        # Số features phải bằng nhau
        assert X_train.shape[1] == X_val.shape[1]

    def test_word_char_shape(self):
        t = build_tfidf(mode="word+char")
        fitted = fit_tfidf(t, SAMPLE_TRAIN)
        X = transform_texts(fitted, SAMPLE_TRAIN)
        assert X.shape[0] == len(SAMPLE_TRAIN)
        assert X.shape[1] > 0

    def test_unseen_tokens_handled(self):
        """Token mới trong val không gây lỗi."""
        t = build_tfidf(mode="word", min_df=1)
        fitted = fit_tfidf(t, ["từ trong train"])
        X_val = transform_texts(fitted, ["từ hoàn toàn mới không có trong train"])
        assert X_val is not None


class TestSaveLoad:
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            t = build_tfidf(mode="word")
            fitted = fit_tfidf(t, SAMPLE_TRAIN)
            path = Path(tmpdir) / "tfidf.joblib"
            metadata = {"mode": "word", "variant": "A"}
            save_tfidf(fitted, path, metadata=metadata)

            loaded, loaded_meta = load_tfidf(path)
            assert loaded_meta["mode"] == "word"

            # Transform phải cho kết quả giống nhau
            X_orig = transform_texts(fitted, SAMPLE_VAL)
            X_load = transform_texts(loaded, SAMPLE_VAL)
            assert X_orig.shape == X_load.shape
            # Sparse matrix — compare dense
            diff = (X_orig - X_load).toarray()
            assert np.allclose(diff, 0)

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            t = build_tfidf(mode="word")
            fitted = fit_tfidf(t, SAMPLE_TRAIN)
            path = Path(tmpdir) / "subdir" / "nested" / "tfidf.joblib"
            save_tfidf(fitted, path)
            assert path.exists()


class TestTextHash:
    def test_deterministic(self):
        texts = ["a", "b", "c"]
        h1 = text_hash(texts)
        h2 = text_hash(texts)
        assert h1 == h2

    def test_different_for_different_texts(self):
        h1 = text_hash(["text one"])
        h2 = text_hash(["text two"])
        assert h1 != h2

    def test_order_matters(self):
        h1 = text_hash(["a", "b"])
        h2 = text_hash(["b", "a"])
        assert h1 != h2
