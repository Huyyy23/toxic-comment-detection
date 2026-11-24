"""Unit tests for TV4 V3 scratch upgrades and Transformer preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.models.cnn import HybridWordCharClassifier
from src.models.cnn_pipeline import (
    load_cnn_artifacts,
    load_cnn_config,
    predict_cnn_from_artifacts,
    preprocess_v2_series,
    run_cnn_variant,
)
from src.models.transformer_pipeline import clean_transformer_text


@pytest.mark.parametrize("pooling", ["gated_attn", "gated_attn_max", "gated_attn_max_mean"])
def test_v3_gated_attention_layernorm_and_msd(pooling: str) -> None:
    model = HybridWordCharClassifier(
        word_vocab_size=32,
        char_vocab_size=24,
        word_embedding_dim=16,
        char_embedding_dim=8,
        char_kernel_sizes=(2, 3),
        num_filters=8,
        hidden_size=12,
        word_encoder_type="bigru",
        dropout=0.3,
        rnn_pooling=pooling,
        spatial_dropout=0.15,
        use_layernorm=True,
        multi_sample_dropout=True,
    )
    model.train()
    w_ids = torch.tensor([[2, 5, 7, 4], [3, 4, 0, 0]], dtype=torch.long)
    w_len = torch.tensor([4, 2], dtype=torch.long)
    c_ids = torch.tensor([[2, 3, 4, 5, 6], [4, 5, 6, 0, 0]], dtype=torch.long)
    c_len = torch.tensor([5, 3], dtype=torch.long)
    logits = model(w_ids, w_len, c_ids, c_len)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_v3_preprocessing_obfuscation_and_pyvi() -> None:
    raw = ["đ.m v*l hòa bình quá thật sự luôn!"]
    norm = preprocess_v2_series(raw, variant="v3_word_norm")
    assert "đm" in norm[0].split()
    assert "vl" in norm[0].split()
    assert "hoà" in norm[0]

    seg = preprocess_v2_series(raw, variant="v3_word_pyvi")
    assert len(seg[0].split()) >= 3

    tr_seg = clean_transformer_text("Hòa bình thật sự tuyệt vời!", use_pyvi=True)
    assert "Hoà_bình" in tr_seg or "Hoà" in tr_seg


def test_v3_train_only_fasttext_and_label_smoothing_smoke(tmp_path: Path) -> None:
    cfg = load_cnn_config(Path("configs/cnn_v3_config.yaml"))
    assert len(cfg["variants"]) == 4

    ds_dir = tmp_path / "binary_n2_v1"
    ds_dir.mkdir(parents=True, exist_ok=True)
    tr_df = pd.DataFrame(
        {
            "id": [f"tr_{i}" for i in range(24)],
            "text": [
                f"bình luận sạch sẽ lịch sự tích cực quá {i}"
                if i % 3 != 0
                else f"đồ ngu vcl đ.m vô học cút đi {i}"
                for i in range(24)
            ],
            "label": [0 if i % 3 != 0 else 1 for i in range(24)],
            "source": ["vihsd" if i % 2 == 0 else "victsd" for i in range(24)],
            "split": ["train"] * 24,
        }
    )
    va_df = pd.DataFrame(
        {
            "id": [f"va_{i}" for i in range(12)],
            "text": [
                f"nội dung hữu ích cảm ơn bạn {i}"
                if i % 3 != 0
                else f"mày ngu v.l đm rác rưởi {i}"
                for i in range(12)
            ],
            "label": [0 if i % 3 != 0 else 1 for i in range(12)],
            "source": ["vihsd" if i % 2 == 0 else "victsd" for i in range(12)],
            "split": ["val"] * 12,
        }
    )
    tr_df.to_parquet(ds_dir / "train.parquet", index=False)
    va_df.to_parquet(ds_dir / "val.parquet", index=False)

    res = run_cnn_variant(
        variant="V3-BIGRU-GATED-FT",
        config_path=Path("configs/cnn_v3_config.yaml"),
        dataset_dir=ds_dir,
        artifacts_root=tmp_path / "out",
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        seeds=[42],
        result_type="smoke",
        overrides={"epochs": 2, "batch_size": 8, "device": "cpu"},
    )
    seed_dir = Path(res["runs"][0]["seed_dir"])
    loaded = load_cnn_artifacts(seed_dir, device="cpu")
    preds, probs = predict_cnn_from_artifacts(loaded, va_df["text"].tolist())
    assert len(preds) == len(va_df)
    assert probs.shape == (len(va_df), 2)
