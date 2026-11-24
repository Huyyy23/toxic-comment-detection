"""Binary training, evaluation, and MLflow pipeline for TV4 CNN models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
import unicodedata
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import yaml

from src.features.preprocessing import TEENCODE_DICT, preprocess_series
from src.models.checkpoint import seed_everything
from src.models.cnn import (
    CharCNNClassifier,
    HybridWordCharClassifier,
    TextCNNClassifier,
    TriHybridWordCNNBiGRUCharClassifier,
)
from src.models.svm_binary_pipeline import (
    BINARY_LABEL_NAMES,
    _build_aggregate_error_summary,
    compute_binary_metrics,
    compute_ece,
    load_binary_train_val,
)
from src.models.training import _resolve_device
from src.tracking.mlflow_tracker import MLflowTracker

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CNN_CONFIG_PATH = ROOT / "configs" / "cnn_config.yaml"
HYBRID_ARCHITECTURES = (
    "word_char_cnn",
    "bilstm_char_cnn",
    "bigru_char_cnn",
    "word_cnn_bigru_char_cnn",
)

TEENCODE_V2_DICT: dict[str, str] = {
    **TEENCODE_DICT,
    "vlxd": "vật liệu xây dựng",
    "cmnr": "rồi",
    "xamlol": "xàm lồn",
    "xàmlol": "xàm lồn",
    "vkl": "vcl",
    "đhs": "đéo hiểu sao",
    "dhs": "đéo hiểu sao",
    "ak": "à",
    "uk": "ừ",
    "thik": "thích",
    "iu": "yêu",
    "wa": "quá",
    "qa": "quá",
    "éo": "đéo",
    "méo": "đéo",
    "chả": "chẳng",
    "đéo": "đéo",
    "vcl": "vcl",
    "vl": "vl",
    "đm": "đm",
}

_V2_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_V2_HTML_RE = re.compile(r"<[^>]+>")
_V2_ZW_RE = re.compile(r"[\u200b-\u200f\ufeff]")
_V2_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
_V2_LETTER_REP_RE = re.compile(r"([a-zà-ỹ])\1{2,}", flags=re.IGNORECASE)
_V2_CHAR_REP3_RE = re.compile(r"(.)\1{3,}")
_V2_PUNCT_SEP_RE = re.compile(r"([^\w\s<>])", flags=re.UNICODE)
_V2_WS_RE = re.compile(r"\s+")
_HARD_LEXICAL_CUES = frozenset(
    {
        "vl",
        "vcl",
        "đm",
        "dm",
        "đéo",
        "chó",
        "ngu",
        "điên",
        "khùng",
        "cút",
        "rác",
        "lồn",
        "đĩ",
    }
)
_NEGATION_CUES = frozenset(
    {"không", "chẳng", "chưa", "đừng", "chớ", "đâu", "khỏi"}
)


_V3_VI_TONE_FIXES = (
    ("òa", "oà"),
    ("óa", "oá"),
    ("ỏa", "oả"),
    ("õa", "oã"),
    ("ọa", "oạ"),
    ("òe", "oè"),
    ("óe", "oé"),
    ("ỏe", "oẻ"),
    ("õe", "oẽ"),
    ("ọe", "oẹ"),
    ("ùy", "uỳ"),
    ("úy", "uý"),
    ("ủy", "uỷ"),
    ("ũy", "uỹ"),
    ("ụy", "uỵ"),
)
_V3_OBFUSC_RE = (
    (re.compile(r"\bđ[\.\*\-_]+m\b", re.IGNORECASE), "đm"),
    (re.compile(r"\bd[\.\*\-_]+m\b", re.IGNORECASE), "đm"),
    (re.compile(r"\bv[\.\*\-_]+l\b", re.IGNORECASE), "vl"),
    (re.compile(r"\bv[\.\*\-_]+c[\.\*\-_]+l\b", re.IGNORECASE), "vcl"),
    (re.compile(r"\bc[\.\*\-_]+c\b", re.IGNORECASE), "cc"),
    (re.compile(r"\bl[\.\*\-_]+n\b", re.IGNORECASE), "lồn"),
)


def _normalize_v3_word_text(raw: str) -> str:
    s = unicodedata.normalize("NFC", str(raw))
    s = _V2_ZW_RE.sub("", s)
    s = _V2_URL_RE.sub(" <url> ", s)
    s = _V2_HTML_RE.sub(" ", s)
    s = _V2_EMOJI_RE.sub(" <emoji> ", s)
    s = s.lower()
    for old, new in _V3_VI_TONE_FIXES:
        s = s.replace(old, new)
    for pat, repl in _V3_OBFUSC_RE:
        s = pat.sub(f" {repl} ", s)
    s = _V2_LETTER_REP_RE.sub(r"\1", s)
    s = _V2_CHAR_REP3_RE.sub(r"\1\1", s)
    s = _V2_PUNCT_SEP_RE.sub(r" \1 ", s)
    tokens = [TEENCODE_V2_DICT.get(tok, tok) for tok in s.split()]
    return _V2_WS_RE.sub(" ", " ".join(tokens)).strip() or "<unk>"


def preprocess_v2_series(texts: list[str], variant: str = "basic") -> list[str]:
    """Preprocess texts supporting V1 ('raw','basic','teencode'), V2, and V3 modes."""
    if variant in ("raw", "basic", "teencode"):
        return preprocess_series(texts, variant=variant)  # type: ignore[arg-type]
    out: list[str] = []
    if variant == "v2_word_norm":
        for raw in texts:
            s = unicodedata.normalize("NFC", str(raw))
            s = _V2_ZW_RE.sub("", s)
            s = _V2_URL_RE.sub(" ", s)
            s = _V2_HTML_RE.sub(" ", s)
            s = _V2_EMOJI_RE.sub(" <emoji> ", s)
            s = s.lower()
            s = _V2_LETTER_REP_RE.sub(r"\1", s)
            s = _V2_CHAR_REP3_RE.sub(r"\1\1", s)
            s = _V2_PUNCT_SEP_RE.sub(r" \1 ", s)
            tokens = [TEENCODE_V2_DICT.get(tok, tok) for tok in s.split()]
            out.append(_V2_WS_RE.sub(" ", " ".join(tokens)).strip() or "<unk>")
        return out
    if variant == "v3_word_norm":
        return [_normalize_v3_word_text(raw) for raw in texts]
    if variant == "v3_word_pyvi":
        from pyvi import ViTokenizer

        return [
            _V2_WS_RE.sub(" ", ViTokenizer.tokenize(_normalize_v3_word_text(raw))).strip()
            or "<unk>"
            for raw in texts
        ]
    if variant == "v2_char_norm":
        for raw in texts:
            s = unicodedata.normalize("NFC", str(raw))
            s = _V2_ZW_RE.sub("", s)
            s = _V2_URL_RE.sub(" ", s)
            s = _V2_HTML_RE.sub(" ", s)
            s = _V2_CHAR_REP3_RE.sub(r"\1\1\1", s)
            out.append(_V2_WS_RE.sub(" ", s).strip() or " ")
        return out
    raise ValueError(f"Unsupported preprocessing variant: {variant!r}")


def _init_word_embedding_with_train_only_fasttext(
    model: nn.Module,
    word_vocab: SequenceVocab,
    train_word_texts: list[str],
    seed: int,
) -> None:
    """Fit gensim FastText STRICTLY on train_word_texts to initialize word embeddings."""
    from gensim.models import FastText

    emb = getattr(model, "word_embedding", None)
    if emb is None:
        emb = getattr(model, "embedding", None)
    if emb is None or not isinstance(emb, nn.Embedding):
        return
    emb_dim = int(emb.embedding_dim)
    sentences = [t.split() for t in train_word_texts if t.strip()]
    if not sentences:
        return
    ft = FastText(
        sentences=sentences,
        vector_size=emb_dim,
        window=5,
        min_count=max(1, word_vocab.min_freq),
        min_n=2,
        max_n=5,
        sg=1,
        epochs=10,
        workers=4,
        seed=seed,
    )
    weight_data = emb.weight.detach().cpu().numpy()
    assert word_vocab._token_to_id is not None
    for tok, idx in word_vocab._token_to_id.items():
        if idx < weight_data.shape[0]:
            weight_data[idx] = ft.wv[tok]
    weight_data[word_vocab.pad_id] = 0.0
    with torch.no_grad():
        emb.weight.copy_(torch.from_numpy(weight_data).to(emb.weight.device))


class SequenceVocab:
    """Train-only word or character vocabulary with deterministic token IDs."""

    pad_id = 0
    unk_id = 1

    def __init__(self, mode: str = "word", min_freq: int = 2) -> None:
        if mode not in ("word", "char"):
            raise ValueError(f"Unsupported vocab mode: {mode!r}")
        if isinstance(min_freq, bool) or not isinstance(min_freq, int) or min_freq < 1:
            raise ValueError("min_freq must be an integer >= 1")
        self.mode = mode
        self.min_freq = min_freq
        self._token_to_id: dict[str, int] | None = None

    @property
    def size(self) -> int:
        if self._token_to_id is None:
            raise RuntimeError("Vocabulary must be fitted first")
        return len(self._token_to_id) + 2

    def tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str):
            text = str(text)
        return text.split() if self.mode == "word" else list(text)

    def fit(self, texts: list[str]) -> "SequenceVocab":
        if self._token_to_id is not None:
            raise RuntimeError("SequenceVocab cannot be refitted")
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(self.tokenize(text))
        kept = sorted(tok for tok, cnt in counts.items() if cnt >= self.min_freq)
        if not kept and counts:
            kept = sorted(counts.keys())
        self._token_to_id = {tok: idx for idx, tok in enumerate(kept, start=2)}
        return self

    def encode(self, text: str, max_length: int) -> list[int]:
        if self._token_to_id is None:
            raise RuntimeError("SequenceVocab must be fitted first")
        tokens = self.tokenize(text)[:max_length]
        ids = [self._token_to_id.get(tok, self.unk_id) for tok in tokens]
        return ids if ids else [self.unk_id]

    def to_dict(self) -> dict[str, Any]:
        if self._token_to_id is None:
            raise RuntimeError("SequenceVocab must be fitted first")
        return {
            "version": 1,
            "mode": self.mode,
            "min_freq": self.min_freq,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "tokens": list(self._token_to_id.keys()),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SequenceVocab":
        vocab = cls(mode=payload["mode"], min_freq=int(payload["min_freq"]))
        vocab._token_to_id = {
            tok: idx for idx, tok in enumerate(payload["tokens"], start=2)
        }
        return vocab


class _EncodedTextDataset(Dataset):
    """In-memory dataset of pre-encoded train/val samples."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def _encode_records(
    df: pd.DataFrame,
    processed_texts: list[str],
    word_vocab: SequenceVocab | None,
    char_vocab: SequenceVocab | None,
    max_word_length: int,
    max_char_length: int,
    char_texts: list[str] | None = None,
    sample_weights: list[float] | None = None,
) -> list[dict[str, Any]]:
    c_texts = char_texts if char_texts is not None else processed_texts
    w_list = (
        sample_weights
        if sample_weights is not None
        else [1.0] * len(processed_texts)
    )
    encoded: list[dict[str, Any]] = []
    for rec_id, label, source, w_text, c_text, s_weight in zip(
        df["id"].tolist(),
        df["label"].astype(int).tolist(),
        df["source"].astype(str).tolist(),
        processed_texts,
        c_texts,
        w_list,
    ):
        row: dict[str, Any] = {
            "id": rec_id,
            "label": label,
            "source": source,
            "sample_weight": float(s_weight),
        }
        if word_vocab is not None:
            row["word_ids"] = word_vocab.encode(w_text, max_word_length)
        if char_vocab is not None:
            row["char_ids"] = char_vocab.encode(c_text, max_char_length)
        encoded.append(row)
    return encoded


def _compute_hard_example_weights(
    df: pd.DataFrame,
    word_texts: list[str],
    hard_weight: float = 1.45,
) -> list[float]:
    """Phase 4 hard-example mining weights on training data only (never changes labels)."""
    weights: list[float] = []
    for label, w_text in zip(df["label"].astype(int).tolist(), word_texts):
        toks = w_text.split()
        tok_set = set(toks)
        has_lex = bool(tok_set & _HARD_LEXICAL_CUES)
        has_neg = bool(tok_set & _NEGATION_CUES)
        is_long = len(toks) >= 32
        # Hard FP risk: CLEAN with toxic lexical cue or negation + long context
        # Hard FN risk: TOXIC without obvious lexical cue or long multi-clause comment
        is_hard = (
            (label == 0 and (has_lex or (has_neg and len(toks) >= 15)))
            or (label == 1 and (not has_lex or is_long or has_neg))
        )
        weights.append(float(hard_weight if is_hard else 1.0))
    return weights


def _augment_train_dataframe_v2(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    mode: str,
    seed: int,
) -> pd.DataFrame:
    """Phase 4 controlled train-only augmentation with strict val anti-leakage filter."""
    if mode in ("none", "", "hard_example_mining"):
        return train_df
    rng = np.random.default_rng(seed)
    val_norm = {
        " ".join(unicodedata.normalize("NFC", str(t)).lower().split())
        for t in val_df["text"].astype(str)
    }
    aug_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(train_df.to_dict("records")):
        text = str(row["text"])
        label = int(row["label"])
        words = text.split()
        if len(words) < 4:
            continue
        norm_low = unicodedata.normalize("NFC", text).lower()
        has_cue = any(cue in norm_low for cue in _HARD_LEXICAL_CUES)
        # Augment minority TOXIC rows (35% rate) + hard CLEAN rows with lexical cues (50% rate)
        if label == 1 and rng.random() < 0.35:
            drop_idx = int(rng.integers(0, len(words)))
            if words[drop_idx].lower() not in _HARD_LEXICAL_CUES:
                new_words = [w for i, w in enumerate(words) if i != drop_idx]
            else:
                new_words = list(words)
                new_words[drop_idx] = new_words[drop_idx] + new_words[drop_idx][-1]
            cand_text = " ".join(new_words)
        elif label == 0 and has_cue and rng.random() < 0.50:
            drop_idx = int(rng.integers(0, len(words)))
            new_words = [w for i, w in enumerate(words) if i != drop_idx]
            cand_text = " ".join(new_words)
        else:
            continue

        cand_norm = " ".join(
            unicodedata.normalize("NFC", cand_text).lower().split()
        )
        if not cand_norm or cand_norm in val_norm:
            continue
        aug_rows.append(
            {
                "id": f"{row['id']}_v2aug_{idx}",
                "text": cand_text,
                "label": label,
                "source": row["source"],
                "split": "train",
            }
        )
    if not aug_rows:
        return train_df
    return pd.concat([train_df, pd.DataFrame(aug_rows)], ignore_index=True)


def _pad_sequences(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    width = int(lengths.max().item())
    padded = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    for row_idx, seq in enumerate(seqs):
        padded[row_idx, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded, lengths


def _collate_cnn_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {
        "ids": [item["id"] for item in batch],
        "sources": [item["source"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "sample_weights": torch.tensor(
            [float(item.get("sample_weight", 1.0)) for item in batch],
            dtype=torch.float32,
        ),
    }
    if "word_ids" in batch[0]:
        collated["word_ids"], collated["word_lengths"] = _pad_sequences(
            [item["word_ids"] for item in batch], SequenceVocab.pad_id
        )
    if "char_ids" in batch[0]:
        collated["char_ids"], collated["char_lengths"] = _pad_sequences(
            [item["char_ids"] for item in batch], SequenceVocab.pad_id
        )
    return collated


def load_cnn_config(config_path: Path = DEFAULT_CNN_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML configuration for TV4 CNN experiments."""
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_cnn_model(
    architecture: str,
    model_cfg: dict[str, Any],
    *,
    word_vocab_size: int | None = None,
    char_vocab_size: int | None = None,
    num_classes: int = 2,
) -> nn.Module:
    """Instantiate TextCNN, CharCNN, or a hybrid Word+Char architecture."""
    if architecture == "textcnn":
        assert word_vocab_size is not None
        return TextCNNClassifier(
            vocab_size=word_vocab_size,
            embedding_dim=int(model_cfg.get("embedding_dim", 128)),
            kernel_sizes=[int(k) for k in model_cfg.get("kernel_sizes", [3, 4, 5])],
            num_filters=int(model_cfg.get("num_filters", 128)),
            dropout=float(model_cfg.get("dropout", 0.5)),
            pad_id=SequenceVocab.pad_id,
            num_classes=num_classes,
            spatial_dropout=float(model_cfg.get("spatial_dropout", 0.0)),
        )
    if architecture == "char_cnn":
        assert char_vocab_size is not None
        return CharCNNClassifier(
            vocab_size=char_vocab_size,
            embedding_dim=int(model_cfg.get("char_embedding_dim", 64)),
            kernel_sizes=[
                int(k) for k in model_cfg.get("char_kernel_sizes", [3, 4, 5, 7])
            ],
            num_filters=int(model_cfg.get("num_filters", 128)),
            dropout=float(model_cfg.get("dropout", 0.5)),
            pad_id=SequenceVocab.pad_id,
            num_classes=num_classes,
            spatial_dropout=float(model_cfg.get("spatial_dropout", 0.0)),
        )
    if architecture == "word_cnn_bigru_char_cnn":
        assert word_vocab_size is not None and char_vocab_size is not None
        return TriHybridWordCNNBiGRUCharClassifier(
            word_vocab_size=word_vocab_size,
            char_vocab_size=char_vocab_size,
            word_embedding_dim=int(model_cfg.get("embedding_dim", 128)),
            char_embedding_dim=int(model_cfg.get("char_embedding_dim", 64)),
            word_kernel_sizes=[
                int(k) for k in model_cfg.get("kernel_sizes", [2, 3, 4])
            ],
            char_kernel_sizes=[
                int(k) for k in model_cfg.get("char_kernel_sizes", [2, 3, 4, 5, 7])
            ],
            word_num_filters=int(
                model_cfg.get("word_num_filters", model_cfg.get("num_filters", 128))
            ),
            char_num_filters=int(
                model_cfg.get("char_num_filters", model_cfg.get("num_filters", 128))
            ),
            hidden_size=int(model_cfg.get("hidden_size", 128)),
            fusion_type=str(model_cfg.get("fusion_type", "concat")),
            fusion_dim=int(model_cfg.get("fusion_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.4)),
            spatial_dropout=float(model_cfg.get("spatial_dropout", 0.15)),
            word_pad_id=SequenceVocab.pad_id,
            char_pad_id=SequenceVocab.pad_id,
            num_classes=num_classes,
        )
    if architecture in HYBRID_ARCHITECTURES:
        assert word_vocab_size is not None and char_vocab_size is not None
        encoder_type = "cnn" if architecture == "word_char_cnn" else architecture.split("_")[0]
        return HybridWordCharClassifier(
            word_vocab_size=word_vocab_size,
            char_vocab_size=char_vocab_size,
            word_embedding_dim=int(model_cfg.get("embedding_dim", 128)),
            char_embedding_dim=int(model_cfg.get("char_embedding_dim", 64)),
            word_kernel_sizes=[int(k) for k in model_cfg.get("kernel_sizes", [3, 4, 5])],
            char_kernel_sizes=[
                int(k) for k in model_cfg.get("char_kernel_sizes", [3, 4, 5, 7])
            ],
            num_filters=int(model_cfg.get("num_filters", 128)),
            hidden_size=int(model_cfg.get("hidden_size", 128)),
            word_encoder_type=encoder_type,
            dropout=float(model_cfg.get("dropout", 0.5)),
            word_pad_id=SequenceVocab.pad_id,
            char_pad_id=SequenceVocab.pad_id,
            num_classes=num_classes,
            rnn_pooling=str(model_cfg.get("rnn_pooling", "last")),
            spatial_dropout=float(model_cfg.get("spatial_dropout", 0.0)),
            word_dropout=(
                float(model_cfg["word_dropout"])
                if model_cfg.get("word_dropout") is not None
                else None
            ),
            char_dropout=(
                float(model_cfg["char_dropout"])
                if model_cfg.get("char_dropout") is not None
                else None
            ),
            num_rnn_layers=int(model_cfg.get("num_rnn_layers", 1)),
            use_layernorm=bool(model_cfg.get("use_layernorm", False)),
            multi_sample_dropout=bool(model_cfg.get("multi_sample_dropout", False)),
        )
    raise ValueError(f"Unsupported CNN architecture: {architecture!r}")


def _forward_batch(model: nn.Module, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    arch = getattr(model, "architecture", "")
    if arch == "textcnn":
        return model(batch["word_ids"].to(device), batch["word_lengths"].to(device))
    if arch == "char_cnn":
        return model(batch["char_ids"].to(device), batch["char_lengths"].to(device))
    return model(
        batch["word_ids"].to(device),
        batch["word_lengths"].to(device),
        batch["char_ids"].to(device),
        batch["char_lengths"].to(device),
    )


def _compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    imbalance_method: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
    sample_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if imbalance_method == "class_weight" and class_weights is not None:
        ce = F.cross_entropy(
            logits,
            labels,
            weight=class_weights,
            reduction="none",
            label_smoothing=label_smoothing,
        )
    elif imbalance_method == "focal":
        raw_ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-raw_ce)
        ce = ((1.0 - pt) ** focal_gamma) * raw_ce
    else:
        ce = F.cross_entropy(
            logits, labels, reduction="none", label_smoothing=label_smoothing
        )
    if sample_weights is not None:
        ce = ce * sample_weights.to(device=ce.device, dtype=ce.dtype)
    return ce.mean()


def _eval_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    imbalance_method: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
    collect_outputs: bool = False,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    sources: list[str] = []
    probs_list: list[np.ndarray] = []
    ids: list[str] = []
    try:
        with torch.no_grad():
            for batch in loader:
                labels = batch["labels"].to(device)
                logits = _forward_batch(model, batch, device)
                loss = _compute_loss(
                    logits, labels,
                    imbalance_method=imbalance_method,
                    class_weights=class_weights,
                    focal_gamma=focal_gamma,
                )
                batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = batch_probs.argmax(axis=1).tolist()
                size = int(labels.shape[0])
                total_loss += float(loss.item()) * size
                total_count += size
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds)
                sources.extend(batch["sources"])
                probs_list.append(batch_probs)
                if collect_outputs:
                    ids.extend(batch["ids"])
    finally:
        model.train(was_training)

    probs = np.vstack(probs_list) if probs_list else np.zeros((0, 2), dtype=np.float32)
    metrics = compute_binary_metrics(y_true, y_pred, sources, probs[:, 1])
    metrics["loss"] = total_loss / max(1, total_count)
    if collect_outputs:
        metrics["ids"] = ids
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred
        metrics["sources"] = sources
        metrics["probs"] = probs
    return metrics


def _save_atomic_pt(path: Path, payload: dict[str, Any]) -> None:
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            tmp_path = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _build_train_loader(
    train_rows: list[dict[str, Any]],
    batch_size: int,
    seed: int,
    imbalance_method: str,
) -> DataLoader:
    dataset = _EncodedTextDataset(train_rows)
    generator = torch.Generator().manual_seed(seed)
    if imbalance_method == "sampler":
        counts = Counter(row["label"] for row in train_rows)
        weights = [1.0 / max(1, counts[row["label"]]) for row in train_rows]
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(train_rows), replacement=True, generator=generator
        )
        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, collate_fn=_collate_cnn_batch
        )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=generator, collate_fn=_collate_cnn_batch,
    )


def _needs_word_and_char(architecture: str) -> tuple[bool, bool]:
    if architecture == "textcnn":
        return True, False
    if architecture == "char_cnn":
        return False, True
    return True, True


def run_cnn_seed(
    variant: str,
    seed: int,
    full_cfg: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    data_hashes: dict[str, str],
    variant_root_dir: Path,
    *,
    tracking_uri: str | None = None,
    result_type: str = "official",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train and evaluate a single seed of a TV4 CNN variant."""
    defaults = dict(full_cfg.get("defaults", {}))
    v_cfg = dict(full_cfg["variants"][variant])
    if overrides:
        defaults.update({k: v for k, v in overrides.items() if v is not None})

    architecture = str(v_cfg["architecture"])
    preprocessing = str(v_cfg.get("preprocessing", "basic"))
    word_prep = str(v_cfg.get("word_preprocessing", preprocessing))
    char_prep = str(v_cfg.get("char_preprocessing", preprocessing))
    train_aug_mode = str(v_cfg.get("train_aug_mode", defaults.get("train_aug_mode", "none")))
    lr_scheduler_type = str(
        v_cfg.get("lr_scheduler", defaults.get("lr_scheduler", "none"))
    )
    model_cfg = dict(v_cfg.get("model", {}))

    batch_size = int(v_cfg.get("batch_size", defaults.get("batch_size", 64)))
    epochs = int(v_cfg.get("epochs", defaults.get("epochs", 15)))
    patience = int(v_cfg.get("patience", defaults.get("patience", 4)))
    lr = float(v_cfg.get("learning_rate", defaults.get("learning_rate", 1e-3)))
    grad_clip = float(v_cfg.get("grad_clip", defaults.get("grad_clip", 1.0)))
    weight_decay = float(v_cfg.get("weight_decay", defaults.get("weight_decay", 0.0)))
    imbalance_method = str(
        v_cfg.get("imbalance_method", defaults.get("imbalance_method", "none"))
    )
    if imbalance_method not in ("none", "class_weight", "focal", "sampler"):
        raise ValueError(f"Unsupported imbalance_method: {imbalance_method!r}")
    focal_gamma = float(v_cfg.get("focal_gamma", defaults.get("focal_gamma", 2.0)))
    max_word_len = int(v_cfg.get("max_word_length", defaults.get("max_word_length", 128)))
    max_char_len = int(v_cfg.get("max_char_length", defaults.get("max_char_length", 384)))
    min_word_freq = int(v_cfg.get("min_word_freq", defaults.get("min_word_freq", 2)))
    min_char_freq = int(v_cfg.get("min_char_freq", defaults.get("min_char_freq", 2)))
    use_amp = bool(v_cfg.get("use_amp", defaults.get("use_amp", False)))
    label_smoothing = float(
        v_cfg.get("label_smoothing", defaults.get("label_smoothing", 0.0))
    )
    init_fasttext = bool(
        v_cfg.get(
            "init_fasttext_train_only",
            defaults.get("init_fasttext_train_only", False),
        )
    )

    seed_dir = variant_root_dir / f"seed_{seed}"
    if (
        result_type == "official"
        and (seed_dir / "metrics.json").exists()
        and (seed_dir / "best.pt").exists()
        and (seed_dir / "val_predictions.json").exists()
        and (seed_dir / "run_id.txt").exists()
    ):
        cached_metrics = json.loads(
            (seed_dir / "metrics.json").read_text(encoding="utf-8")
        )
        cached_run_id = (seed_dir / "run_id.txt").read_text(encoding="utf-8").strip()
        print(
            f"  [RESUME-SKIP] {variant} seed_{seed} already completed | "
            f"macro_f1={cached_metrics['macro_f1']:.4f} | "
            f"f1_toxic={cached_metrics['f1_toxic']:.4f}",
            flush=True,
        )
        return {
            "variant": variant,
            "seed": seed,
            "run_id": cached_run_id,
            "seed_dir": str(seed_dir),
            "metrics": cached_metrics,
        }

    seed_everything(seed, deterministic=True)
    np.random.seed(seed)
    device = _resolve_device(str(defaults.get("device", full_cfg.get("device", "auto"))))
    amp_enabled = use_amp and device.type == "cuda"

    effective_train_df = _augment_train_dataframe_v2(
        train_df, val_df, mode=train_aug_mode, seed=seed
    )
    train_texts_raw = effective_train_df["text"].astype(str).tolist()
    val_texts_raw = val_df["text"].astype(str).tolist()
    train_word_texts = preprocess_v2_series(train_texts_raw, variant=word_prep)
    val_word_texts = preprocess_v2_series(val_texts_raw, variant=word_prep)
    train_char_texts = (
        train_word_texts
        if char_prep == word_prep
        else preprocess_v2_series(train_texts_raw, variant=char_prep)
    )
    val_char_texts = (
        val_word_texts
        if char_prep == word_prep
        else preprocess_v2_series(val_texts_raw, variant=char_prep)
    )

    need_word, need_char = _needs_word_and_char(architecture)
    word_vocab = (
        SequenceVocab("word", min_word_freq).fit(train_word_texts)
        if need_word
        else None
    )
    char_vocab = (
        SequenceVocab("char", min_char_freq).fit(train_char_texts)
        if need_char
        else None
    )

    train_sample_weights = (
        _compute_hard_example_weights(effective_train_df, train_word_texts)
        if train_aug_mode == "hard_example_mining"
        else None
    )
    train_rows = _encode_records(
        effective_train_df,
        train_word_texts,
        word_vocab,
        char_vocab,
        max_word_len,
        max_char_len,
        char_texts=train_char_texts,
        sample_weights=train_sample_weights,
    )
    val_rows = _encode_records(
        val_df,
        val_word_texts,
        word_vocab,
        char_vocab,
        max_word_len,
        max_char_len,
        char_texts=val_char_texts,
    )

    train_loader = _build_train_loader(train_rows, batch_size, seed, imbalance_method)
    val_loader = DataLoader(
        _EncodedTextDataset(val_rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_cnn_batch,
    )

    model = build_cnn_model(
        architecture,
        model_cfg,
        word_vocab_size=word_vocab.size if word_vocab is not None else None,
        char_vocab_size=char_vocab.size if char_vocab is not None else None,
        num_classes=2,
    ).to(device)
    if init_fasttext and word_vocab is not None:
        _init_word_embedding_with_train_only_fasttext(
            model=model,
            word_vocab=word_vocab,
            train_word_texts=train_word_texts,
            seed=seed,
        )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    class_weights: torch.Tensor | None = None
    if imbalance_method == "class_weight":
        train_labels_arr = train_df["label"].astype(int).tolist()
        counts = Counter(train_labels_arr)
        total = len(train_labels_arr)
        weights = [total / (2.0 * max(1, counts[c])) for c in (0, 1)]
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler: Any = None
    if lr_scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=lr * 0.08
        )
    elif lr_scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1
        )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    seed_dir = variant_root_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    vocab_payload = {
        "preprocessing": preprocessing,
        "word_preprocessing": word_prep,
        "char_preprocessing": char_prep,
        "max_word_length": max_word_len,
        "max_char_length": max_char_len,
        "word_vocab": word_vocab.to_dict() if word_vocab is not None else None,
        "char_vocab": char_vocab.to_dict() if char_vocab is not None else None,
    }
    vocab_text = json.dumps(vocab_payload, ensure_ascii=False, indent=2)
    vocab_path = seed_dir / "vocab.json"
    vocab_path.write_text(vocab_text + "\n", encoding="utf-8")
    vocab_sha256 = hashlib.sha256(vocab_path.read_bytes()).hexdigest()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    exp_name = str(full_cfg.get("experiment", "TV4-CNN-Binary"))
    tracker = MLflowTracker(experiment_name=exp_name, tracking_uri=tracking_uri)
    run_id = tracker.start_run(
        run_name=f"{variant}-seed{seed}",
        tags={
            "experiment": exp_name,
            "variant": variant,
            "seed": str(seed),
            "owner": str(full_cfg.get("owner", "TV4")),
            "architecture": architecture,
            "pretrained": "false",
            "result_type": result_type,
            "imbalance_method": imbalance_method,
            "use_augmentation": str(bool(v_cfg.get("use_augmentation", False))).lower(),
        },
    )

    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    started = time.perf_counter()

    try:
        tracker.log_environment()
        tracker.log_model_architecture(architecture=architecture, model_type="cnn_scratch")
        tracker.log_params(
            {
                "variant": variant,
                "seed": seed,
                "architecture": architecture,
                "preprocessing": preprocessing,
                "imbalance_method": imbalance_method,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "epochs": epochs,
                "patience": patience,
                "num_params": num_params,
                "word_vocab_size": word_vocab.size if word_vocab is not None else 0,
                "char_vocab_size": char_vocab.size if char_vocab is not None else 0,
                "vocab_sha256": vocab_sha256[:16],
                "train_parquet_sha256": data_hashes["train_parquet_sha256"][:16],
                "val_parquet_sha256": data_hashes["val_parquet_sha256"][:16],
                "n_train": len(train_df),
                "n_val": len(val_df),
                **{f"model_{k}": str(v) for k, v in model_cfg.items()},
            }
        )

        best_path = seed_dir / "best.pt"
        last_path = seed_dir / "last.pt"
        for epoch in range(1, epochs + 1):
            model.train()
            tr_loss = 0.0
            tr_count = 0
            tr_tp1 = 0
            tr_fp1 = 0
            tr_fn1 = 0
            tr_tn1 = 0

            for batch in train_loader:
                labels = batch["labels"].to(device)
                batch_sw = (
                    batch["sample_weights"].to(device)
                    if train_aug_mode == "hard_example_mining"
                    else None
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = _forward_batch(model, batch, device)
                    loss = _compute_loss(
                        logits, labels,
                        imbalance_method=imbalance_method,
                        class_weights=class_weights,
                        focal_gamma=focal_gamma,
                        sample_weights=batch_sw,
                        label_smoothing=label_smoothing,
                    )
                if not torch.isfinite(loss).item():
                    raise ValueError("Non-finite training loss encountered")
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip, error_if_nonfinite=True
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip, error_if_nonfinite=True
                    )
                    optimizer.step()

                size = int(labels.shape[0])
                tr_loss += float(loss.detach().item()) * size
                tr_count += size
                preds_t = logits.detach().argmax(dim=1)
                tr_tp1 += int(((labels == 1) & (preds_t == 1)).sum().item())
                tr_fp1 += int(((labels == 0) & (preds_t == 1)).sum().item())
                tr_fn1 += int(((labels == 1) & (preds_t == 0)).sum().item())
                tr_tn1 += int(((labels == 0) & (preds_t == 0)).sum().item())

            tr_f1_tox = (2.0 * tr_tp1) / max(1, 2 * tr_tp1 + tr_fp1 + tr_fn1)
            tr_f1_cln = (2.0 * tr_tn1) / max(1, 2 * tr_tn1 + tr_fn1 + tr_fp1)
            train_m = {
                "loss": tr_loss / max(1, tr_count),
                "macro_f1": 0.5 * (tr_f1_cln + tr_f1_tox),
                "f1_toxic": tr_f1_tox,
            }

            val_m = _eval_loader(
                model, val_loader, device,
                imbalance_method=imbalance_method,
                class_weights=class_weights,
                focal_gamma=focal_gamma,
            )
            if scheduler is not None:
                if lr_scheduler_type == "plateau":
                    scheduler.step(val_m["macro_f1"])
                else:
                    scheduler.step()

            history.append(
                {
                    "epoch": epoch,
                    "train": {
                        "loss": train_m["loss"],
                        "macro_f1": train_m["macro_f1"],
                        "f1_toxic": train_m["f1_toxic"],
                    },
                    "val": {
                        "loss": val_m["loss"],
                        "macro_f1": val_m["macro_f1"],
                        "f1_toxic": val_m["f1_toxic"],
                    },
                }
            )
            tracker.log_metrics(
                {
                    "train_loss": train_m["loss"],
                    "train_macro_f1": train_m["macro_f1"],
                    "train_f1_toxic": train_m["f1_toxic"],
                    "val_loss": val_m["loss"],
                    "val_macro_f1": val_m["macro_f1"],
                    "val_f1_toxic": val_m["f1_toxic"],
                },
                step=epoch,
            )

            is_best = val_m["macro_f1"] > best_macro_f1
            if is_best:
                best_macro_f1 = val_m["macro_f1"]
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1

            if is_best:
                ckpt_payload = {
                    "format_version": 1,
                    "architecture": architecture,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_macro_f1": float(best_macro_f1),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "vocab_sha256": vocab_sha256,
                }
                _save_atomic_pt(best_path, ckpt_payload)
            if bad_epochs >= patience or epoch == epochs:
                last_payload = {
                    "format_version": 1,
                    "architecture": architecture,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_macro_f1": float(best_macro_f1),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "vocab_sha256": vocab_sha256,
                }
                _save_atomic_pt(last_path, last_payload)
            if bad_epochs >= patience:
                break

        train_duration_s = time.perf_counter() - started
        peak_vram_mb = (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            if device.type == "cuda"
            else 0.0
        )

        best_ckpt = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(best_ckpt["model_state_dict"])

        t_inf = time.perf_counter()
        best_eval = _eval_loader(
            model, val_loader, device,
            imbalance_method=imbalance_method,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
            collect_outputs=True,
        )
        inference_ms = ((time.perf_counter() - t_inf) / max(1, len(val_df))) * 1000.0

        probs = best_eval.pop("probs")
        val_ids = best_eval.pop("ids")
        val_true = best_eval.pop("y_true")
        val_pred = best_eval.pop("y_pred")
        val_sources = best_eval.pop("sources")

        val_true_arr = np.asarray(val_true, dtype=int)
        prob_pos_clipped = np.clip(probs[:, 1], 1e-7, 1.0 - 1e-7)
        raw_ece = compute_ece(val_true_arr, probs[:, 1], n_bins=10)
        brier = float(brier_score_loss(val_true_arr, probs[:, 1]))
        nll = float(log_loss(val_true_arr, prob_pos_clipped, labels=[0, 1]))
        model_size_mb = round(
            (best_path.stat().st_size + vocab_path.stat().st_size) / (1024.0 * 1024.0), 4
        )
        metrics: dict[str, Any] = {
            **best_eval,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "raw_softmax_ece": round(raw_ece, 6),
            "brier_score": round(brier, 6),
            "log_loss": round(nll, 6),
            "is_calibrated": False,
            "train_time_s": round(train_duration_s, 4),
            "inference_ms_per_sample": round(inference_ms, 4),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "num_params": int(num_params),
            "model_size_mb": model_size_mb,
            "n_train": len(effective_train_df),
            "n_val": len(val_df),
            "device": str(device),
        }

        predictions_json: list[dict[str, Any]] = []
        predictions_csv: list[dict[str, Any]] = []
        for idx, (rec_id, true_lbl, pred_lbl, src) in enumerate(
            zip(val_ids, val_true, val_pred, val_sources)
        ):
            p0 = round(float(probs[idx, 0]), 6)
            p1 = round(float(probs[idx, 1]), 6)
            predictions_json.append(
                {
                    "id": rec_id,
                    "source": src,
                    "true_label": int(true_lbl),
                    "predicted_label": int(pred_lbl),
                    "probabilities": {
                        BINARY_LABEL_NAMES[0]: p0,
                        BINARY_LABEL_NAMES[1]: p1,
                    },
                    "is_calibrated": False,
                    "threshold": None,
                    "run_id": run_id,
                }
            )
            predictions_csv.append(
                {
                    "sample_id": rec_id,
                    "source": src,
                    "y_true": int(true_lbl),
                    "y_pred": int(pred_lbl),
                    "prob_CLEAN": p0,
                    "prob_TOXIC": p1,
                }
            )

        (seed_dir / "val_predictions.json").write_text(
            json.dumps(predictions_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pd.DataFrame(predictions_csv).to_csv(
            seed_dir / "predictions_val.csv", index=False, encoding="utf-8"
        )
        (seed_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        error_summary = _build_aggregate_error_summary(val_df, val_pred)
        (seed_dir / "error_summary.json").write_text(
            json.dumps(error_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        config_snapshot = {
            "experiment": exp_name,
            "variant": variant,
            "seed": seed,
            "run_id": run_id,
            "owner": full_cfg.get("owner", "TV4"),
            "description": v_cfg.get("description", ""),
            "hypothesis": str(v_cfg.get("hypothesis", "")).strip(),
            "architecture": architecture,
            "preprocessing": preprocessing,
            "word_preprocessing": word_prep,
            "char_preprocessing": char_prep,
            "train_aug_mode": train_aug_mode,
            "lr_scheduler": lr_scheduler_type,
            "use_augmentation": bool(v_cfg.get("use_augmentation", False)),
            "imbalance_method": imbalance_method,
            "focal_gamma": focal_gamma if imbalance_method == "focal" else None,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "max_word_length": max_word_len,
            "max_char_length": max_char_len,
            "min_word_freq": min_word_freq,
            "min_char_freq": min_char_freq,
            "use_amp": amp_enabled,
            "model": model_cfg,
            "word_vocab_size": word_vocab.size if word_vocab is not None else None,
            "char_vocab_size": char_vocab.size if char_vocab is not None else None,
            "vocab_sha256": vocab_sha256,
            "data_hashes": data_hashes,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "device": str(device),
        }
        (seed_dir / "config.json").write_text(
            json.dumps(config_snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (seed_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (seed_dir / "run_id.txt").write_text(run_id, encoding="utf-8")

        scalar_metrics: dict[str, float] = {
            "best_val_macro_f1": metrics["macro_f1"],
            "best_val_f1_toxic": metrics["f1_toxic"],
            "best_val_weighted_f1": metrics["weighted_f1"],
            "best_val_balanced_accuracy": metrics["balanced_accuracy"],
            "best_val_raw_ece": metrics["raw_softmax_ece"],
            "best_epoch": float(best_epoch),
            "train_time_s": metrics["train_time_s"],
            "inference_ms_per_sample": metrics["inference_ms_per_sample"],
            "peak_vram_mb": metrics["peak_vram_mb"],
            "model_size_mb": metrics["model_size_mb"],
            "num_params": float(num_params),
        }
        if metrics.get("pr_auc") is not None:
            scalar_metrics["best_val_pr_auc"] = float(metrics["pr_auc"])
        for src_name, src_m in metrics["by_source"].items():
            scalar_metrics[f"val_macro_f1_{src_name}"] = src_m["macro_f1"]
            scalar_metrics[f"val_f1_toxic_{src_name}"] = src_m["f1_toxic"]
        tracker.log_metrics(scalar_metrics)
        tracker.log_artifact(str(seed_dir))
        tracker.end_run("FINISHED")
    except BaseException:
        tracker.end_run("FAILED")
        raise

    print(
        f"  -> [{variant} seed_{seed}] val_macro_f1={metrics['macro_f1']:.4f} | "
        f"f1_toxic={metrics['f1_toxic']:.4f} | pr_auc={metrics.get('pr_auc', 0.0):.4f} | "
        f"ep={best_epoch}/{len(history)} | time={metrics['train_time_s']:.1f}s",
        flush=True,
    )
    return {
        "variant": variant,
        "seed": seed,
        "run_id": run_id,
        "seed_dir": str(seed_dir),
        "metrics": metrics,
    }


def _aggregate_cnn_seeds(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    def _mean_std(vals: list[float]) -> dict[str, float]:
        arr = np.asarray(vals, dtype=float)
        return {"mean": round(float(np.mean(arr)), 4), "std": round(float(np.std(arr)), 4)}

    sources = sorted(seed_runs[0]["metrics"]["by_source"].keys()) if seed_runs else []
    by_source = {
        src: {
            "macro_f1": _mean_std([r["metrics"]["by_source"][src]["macro_f1"] for r in seed_runs]),
            "f1_toxic": _mean_std([r["metrics"]["by_source"][src]["f1_toxic"] for r in seed_runs]),
        }
        for src in sources
    }
    pr_aucs = [
        r["metrics"]["pr_auc"] for r in seed_runs if r["metrics"].get("pr_auc") is not None
    ]
    briers = [
        r["metrics"]["brier_score"]
        for r in seed_runs
        if r["metrics"].get("brier_score") is not None
    ]
    nlls = [
        r["metrics"]["log_loss"]
        for r in seed_runs
        if r["metrics"].get("log_loss") is not None
    ]
    return {
        "n_seeds": len(seed_runs),
        "seeds": [r["seed"] for r in seed_runs],
        "macro_f1": _mean_std([r["metrics"]["macro_f1"] for r in seed_runs]),
        "f1_toxic": _mean_std([r["metrics"]["f1_toxic"] for r in seed_runs]),
        "pr_auc": _mean_std(pr_aucs) if pr_aucs else None,
        "raw_softmax_ece": _mean_std([r["metrics"]["raw_softmax_ece"] for r in seed_runs]),
        "brier_score": _mean_std(briers) if briers else None,
        "log_loss": _mean_std(nlls) if nlls else None,
        "train_time_s": _mean_std([r["metrics"]["train_time_s"] for r in seed_runs]),
        "peak_vram_mb": _mean_std([r["metrics"]["peak_vram_mb"] for r in seed_runs]),
        "model_size_mb": _mean_std([r["metrics"]["model_size_mb"] for r in seed_runs]),
        "num_params": seed_runs[0]["metrics"]["num_params"] if seed_runs else 0,
        "by_source": by_source,
    }


def run_cnn_variant(
    variant: str,
    config_path: Path = DEFAULT_CNN_CONFIG_PATH,
    dataset_dir: Path | None = None,
    artifacts_root: Path | None = None,
    tracking_uri: str | None = None,
    seeds: list[int] | None = None,
    result_type: str = "official",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all configured seeds for one CNN variant on binary train/val Parquet."""
    full_cfg = load_cnn_config(config_path)
    if variant not in full_cfg["variants"]:
        raise KeyError(f"Invalid CNN variant: {variant!r}")

    v_cfg = full_cfg["variants"][variant]
    use_aug = bool(v_cfg.get("use_augmentation", False))
    if dataset_dir is not None:
        resolved_dir = Path(dataset_dir)
    elif use_aug and full_cfg.get("n4_dataset_dir"):
        resolved_dir = ROOT / full_cfg["n4_dataset_dir"]
    else:
        resolved_dir = ROOT / full_cfg["dataset_dir"]

    train_df, val_df, data_hashes = load_binary_train_val(
        dataset_dir=resolved_dir, use_augmentation=False
    )
    resolved_seeds = seeds or [int(s) for s in full_cfg.get("seeds", [42, 43, 44])]
    resolved_tracking = tracking_uri if tracking_uri is not None else full_cfg.get("tracking_uri")
    root_out = artifacts_root or (ROOT / "models" / "cnn" / f"TV4-run-{uuid.uuid4().hex}")
    variant_dir = root_out / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    seed_runs = [
        run_cnn_seed(
            variant=variant,
            seed=s,
            full_cfg=full_cfg,
            train_df=train_df,
            val_df=val_df,
            data_hashes=data_hashes,
            variant_root_dir=variant_dir,
            tracking_uri=resolved_tracking,
            result_type=result_type,
            overrides=overrides,
        )
        for s in resolved_seeds
    ]
    summary = {
        "variant": variant,
        "architecture": v_cfg["architecture"],
        "preprocessing": v_cfg.get("preprocessing", "basic"),
        "use_augmentation": use_aug,
        "dataset_dir": str(resolved_dir),
        "variant_dir": str(variant_dir),
        "aggregate": _aggregate_cnn_seeds(seed_runs),
        "runs": seed_runs,
    }
    (variant_dir / "variant_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def load_cnn_artifacts(seed_dir: Path, device: str = "auto") -> dict[str, Any]:
    """Load saved CNN config, vocabularies, and best checkpoint for inference."""
    seed_dir = Path(seed_dir)
    config = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))
    vocab_bytes = (seed_dir / "vocab.json").read_bytes()
    vocab_sha = hashlib.sha256(vocab_bytes).hexdigest()
    vocab_payload = json.loads(vocab_bytes.decode("utf-8"))

    word_vocab = (
        SequenceVocab.from_dict(vocab_payload["word_vocab"])
        if vocab_payload.get("word_vocab") is not None
        else None
    )
    char_vocab = (
        SequenceVocab.from_dict(vocab_payload["char_vocab"])
        if vocab_payload.get("char_vocab") is not None
        else None
    )
    resolved = _resolve_device(device)
    ckpt = torch.load(seed_dir / "best.pt", map_location=resolved, weights_only=True)
    if ckpt.get("vocab_sha256") != vocab_sha:
        raise ValueError("vocab.json SHA-256 disagrees with checkpoint")

    model = build_cnn_model(
        config["architecture"],
        config["model"],
        word_vocab_size=word_vocab.size if word_vocab is not None else None,
        char_vocab_size=char_vocab.size if char_vocab is not None else None,
        num_classes=2,
    ).to(resolved)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "device": resolved,
        "config": config,
        "word_vocab": word_vocab,
        "char_vocab": char_vocab,
    }


def predict_cnn_from_artifacts(
    artifacts: dict[str, Any],
    texts: list[str],
    batch_size: int = 128,
) -> tuple[list[int], np.ndarray]:
    """Run deterministic inference from loaded CNN artifacts without refitting."""
    if not texts:
        return [], np.zeros((0, 2), dtype=np.float32)
    cfg = artifacts["config"]
    w_prep = str(cfg.get("word_preprocessing", cfg["preprocessing"]))
    c_prep = str(cfg.get("char_preprocessing", cfg["preprocessing"]))
    word_processed = preprocess_v2_series(texts, variant=w_prep)
    char_processed = (
        word_processed
        if c_prep == w_prep
        else preprocess_v2_series(texts, variant=c_prep)
    )
    word_vocab: SequenceVocab | None = artifacts["word_vocab"]
    char_vocab: SequenceVocab | None = artifacts["char_vocab"]
    max_w = int(cfg["max_word_length"])
    max_c = int(cfg["max_char_length"])

    rows = []
    for idx, (w_text, c_text) in enumerate(zip(word_processed, char_processed)):
        row: dict[str, Any] = {"id": str(idx), "label": 0, "source": "infer"}
        if word_vocab is not None:
            row["word_ids"] = word_vocab.encode(w_text, max_w)
        if char_vocab is not None:
            row["char_ids"] = char_vocab.encode(c_text, max_c)
        rows.append(row)

    loader = DataLoader(
        _EncodedTextDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_cnn_batch,
    )
    model = artifacts["model"]
    device = artifacts["device"]
    model.eval()
    probs_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = _forward_batch(model, batch, device)
            probs_parts.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.vstack(probs_parts)
    preds = probs.argmax(axis=1).astype(int).tolist()
    return preds, probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TV4 Binary CNN Training Pipeline")
    parser.add_argument(
        "--variant",
        nargs="+",
        default=["CNN-01"],
        help="One or more variants, or 'all_pdf' / 'all'",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CNN_CONFIG_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--run-group-dir", type=Path, default=None)
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_cfg = load_cnn_config(args.config)
    requested = args.variant if isinstance(args.variant, list) else [args.variant]
    if requested == ["all"]:
        variants = list(full_cfg["variants"].keys())
    elif requested == ["all_pdf"]:
        variants = ["CNN-01", "CNN-02", "CNN-03a", "CNN-03b", "CNN-04a", "CNN-04b"]
    else:
        variants = requested

    if args.run_group_dir is not None:
        run_group_dir = Path(args.run_group_dir)
        run_group_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = args.artifacts_dir or (ROOT / "models" / "cnn")
        run_group_dir = base_dir / f"TV4-run-{uuid.uuid4().hex}"
        run_group_dir.mkdir(parents=True, exist_ok=False)

    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
    }
    summaries = []
    for variant in variants:
        res = run_cnn_variant(
            variant=variant,
            config_path=args.config,
            dataset_dir=args.dataset_dir,
            artifacts_root=run_group_dir,
            tracking_uri=args.tracking_uri,
            seeds=args.seeds,
            overrides=overrides,
        )
        summaries.append(res)
        agg = res["aggregate"]
        print(
            f"[{variant}] macro_f1={agg['macro_f1']['mean']:.4f}±{agg['macro_f1']['std']:.4f}"
            f" | f1_toxic={agg['f1_toxic']['mean']:.4f}±{agg['f1_toxic']['std']:.4f}"
            f" | vram={agg['peak_vram_mb']['mean']:.1f}MB"
            f" | params={agg['num_params']}",
            flush=True,
        )

    summary_path = run_group_dir / "TV4-CNN-summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
