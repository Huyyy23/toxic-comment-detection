"""
TF-IDF pipeline wrapper — TV1 EXP-01.

Hỗ trợ:
  - word n-gram TF-IDF
  - char n-gram TF-IDF
  - Kết hợp word + char (FeatureUnion)
  - Fit chỉ trên train; transform val/test bằng artifact đã fit
  - Save/load artifact với joblib
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


TfidfMode = Literal["word", "char", "word+char"]


def build_tfidf(
    mode: TfidfMode = "word",
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (2, 4),
    max_features: int | None = 100_000,
    sublinear_tf: bool = True,
    min_df: int = 2,
) -> TfidfVectorizer | FeatureUnion:
    """Tạo TF-IDF transformer chưa fit.

    Args:
        mode: "word" | "char" | "word+char"
        word_ngram_range: n-gram range cho word-level.
        char_ngram_range: n-gram range cho char-level.
        max_features: Giới hạn từ vựng; None = không giới hạn.
        sublinear_tf: Dùng log(1+tf) thay tf.
        min_df: Bỏ term xuất hiện dưới min_df documents.

    Returns:
        Transformer chưa fit.
    """
    word_kwargs = dict(
        analyzer="word",
        ngram_range=word_ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
    )
    char_kwargs = dict(
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
    )

    if mode == "word":
        return TfidfVectorizer(**word_kwargs)
    elif mode == "char":
        return TfidfVectorizer(**char_kwargs)
    elif mode == "word+char":
        return FeatureUnion(
            [
                ("word", TfidfVectorizer(**word_kwargs)),
                ("char", TfidfVectorizer(**char_kwargs)),
            ]
        )
    else:
        raise ValueError(f"mode phải là 'word', 'char' hoặc 'word+char', nhận: {mode!r}")


def fit_tfidf(
    transformer: TfidfVectorizer | FeatureUnion,
    train_texts: list[str],
) -> TfidfVectorizer | FeatureUnion:
    """Fit TF-IDF **chỉ trên train**.

    Args:
        transformer: Transformer chưa fit từ build_tfidf().
        train_texts: Danh sách văn bản train đã qua preprocessing.

    Returns:
        Transformer đã fit.
    """
    return transformer.fit(train_texts)


def transform_texts(
    transformer: TfidfVectorizer | FeatureUnion,
    texts: list[str],
) -> np.ndarray:
    """Transform texts bằng transformer đã fit.

    Dùng cho cả train (sau fit) và val/test.
    """
    return transformer.transform(texts)


def save_tfidf(
    transformer: TfidfVectorizer | FeatureUnion,
    path: str | Path,
    metadata: dict | None = None,
) -> None:
    """Lưu transformer + metadata ra file .joblib.

    Args:
        transformer: Đã fit.
        path: Đường dẫn lưu (tự tạo thư mục cha nếu chưa có).
        metadata: Dict thông tin bổ sung (mode, ngram_range, v.v.).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"transformer": transformer, "metadata": metadata or {}}
    joblib.dump(artifact, path)


def load_tfidf(path: str | Path) -> tuple[TfidfVectorizer | FeatureUnion, dict]:
    """Load transformer đã fit.

    Returns:
        (transformer, metadata)
    """
    artifact = joblib.load(path)
    return artifact["transformer"], artifact["metadata"]


def text_hash(texts: list[str]) -> str:
    """SHA-256 hash của danh sách text (dùng verify split)."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
    return h.hexdigest()
