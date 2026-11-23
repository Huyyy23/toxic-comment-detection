"""Unit tests for TV4 V2 architectures, preprocessing, schedulers, and hard-example mining."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.models.cnn import (
    HybridWordCharClassifier,
    TriHybridWordCNNBiGRUCharClassifier,
)
from src.models.cnn_pipeline import (
    _augment_train_dataframe_v2,
    _compute_hard_example_weights,
    load_cnn_artifacts,
    load_cnn_config,
    predict_cnn_from_artifacts,
    preprocess_v2_series,
    run_cnn_variant,
)


def test_v2_preprocessing_separates_punctuation_and_normalizes_teencode() -> None:
    raw = [
        "Đm thua vcl :)))) phim hài à :))))",
        "Cô này chủ vlxd giàu có và chuyên từ thiện luôn nhaaa bạn,bớt nghiệppp!",
    ]
    word_norm = preprocess_v2_series(raw, variant="v2_word_norm")
    assert "đm" in word_norm[0].split()
    assert "vcl" in word_norm[0].split()
    assert "vật" in word_norm[1].split()
    assert "nghiệp" in word_norm[1].split()

    char_norm = preprocess_v2_series(raw, variant="v2_char_norm")
    assert len(char_norm) == 2
    assert "))))" not in char_norm[0]


@pytest.mark.parametrize("pooling", ["last", "attn", "attn_max"])
def test_v2_hybrid_rnn_pooling_modes_and_spatial_dropout(pooling: str) -> None:
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
        spatial_dropout=0.2,
        word_dropout=0.25,
        char_dropout=0.15,
    )
    model.train()
    w_ids = torch.tensor([[2, 5, 7, 0], [3, 4, 0, 0]], dtype=torch.long)
    w_len = torch.tensor([3, 2], dtype=torch.long)
    c_ids = torch.tensor([[2, 3, 4, 5, 6], [4, 5, 6, 0, 0]], dtype=torch.long)
    c_len = torch.tensor([5, 3], dtype=torch.long)
    logits = model(w_ids, w_len, c_ids, c_len)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("fusion_type", ["concat", "gated", "attention"])
def test_v2_tri_hybrid_fusion_modes(fusion_type: str) -> None:
    model = TriHybridWordCNNBiGRUCharClassifier(
        word_vocab_size=32,
        char_vocab_size=24,
        word_embedding_dim=16,
        char_embedding_dim=8,
        word_kernel_sizes=(2, 3),
        char_kernel_sizes=(2, 3, 4),
        word_num_filters=8,
        char_num_filters=8,
        hidden_size=12,
        fusion_type=fusion_type,
        fusion_dim=16,
        dropout=0.3,
        spatial_dropout=0.15,
    )
    model.train()
    w_ids = torch.tensor([[2, 5, 7, 4], [3, 4, 8, 0]], dtype=torch.long)
    w_len = torch.tensor([4, 3], dtype=torch.long)
    c_ids = torch.tensor([[2, 3, 4, 5, 6], [4, 5, 6, 7, 0]], dtype=torch.long)
    c_len = torch.tensor([5, 4], dtype=torch.long)
    logits = model(w_ids, w_len, c_ids, c_len)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_v2_config_and_pipeline_smoke_with_reload(tmp_path: Path) -> None:
    cfg = load_cnn_config(Path("configs/cnn_v2_config.yaml"))
    assert len(cfg["variants"]) == 15

    ds_dir = tmp_path / "binary_n2_v1"
    ds_dir.mkdir(parents=True, exist_ok=True)
    tr_df = pd.DataFrame(
        {
            "id": [f"tr_{i}" for i in range(24)],
            "text": [
                f"phim này hay đc vl không hề tệ chút nào {i}"
                if i % 3 != 0
                else f"mày ngu vcl đm đồ chó cút đi {i}"
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
                f"bình luận tích cực sạch sẽ {i}"
                if i % 3 != 0
                else f"đồ ngu đm rác rưởi {i}"
                for i in range(12)
            ],
            "label": [0 if i % 3 != 0 else 1 for i in range(12)],
            "source": ["vihsd" if i % 2 == 0 else "victsd" for i in range(12)],
            "split": ["val"] * 12,
        }
    )
    tr_df.to_parquet(ds_dir / "train.parquet", index=False)
    va_df.to_parquet(ds_dir / "val.parquet", index=False)

    w_texts = preprocess_v2_series(tr_df["text"].tolist(), "v2_word_norm")
    hw = _compute_hard_example_weights(tr_df, w_texts)
    assert len(hw) == len(tr_df)
    assert any(w > 1.0 for w in hw)

    aug_df = _augment_train_dataframe_v2(tr_df, va_df, "v2_controlled_aug", seed=42)
    assert len(aug_df) >= len(tr_df)

    out_dir = tmp_path / "out"
    res = run_cnn_variant(
        variant="V2-TRI-ATTN",
        config_path=Path("configs/cnn_v2_config.yaml"),
        dataset_dir=ds_dir,
        artifacts_root=out_dir,
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        seeds=[42],
        result_type="smoke",
        overrides={"epochs": 2, "batch_size": 8, "device": "cpu"},
    )
    seed_dir = Path(res["runs"][0]["seed_dir"])
    assert "brier_score" in res["runs"][0]["metrics"]
    assert "log_loss" in res["runs"][0]["metrics"]

    loaded = load_cnn_artifacts(seed_dir, device="cpu")
    preds, probs = predict_cnn_from_artifacts(loaded, va_df["text"].tolist())
    assert len(preds) == len(va_df)
    assert probs.shape == (len(va_df), 2)
