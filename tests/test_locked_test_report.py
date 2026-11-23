"""Unit tests for TV4 Phase 6 (N10) freeze manifest and one-shot locked test evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.locked_test_report import (
    _sha256_file,
    build_n10_freeze_manifest,
    run_n10_locked_test_evaluation,
)
from src.models.cnn_pipeline import run_cnn_variant


def _setup_synthetic_n10_workspace(tmp_path: Path) -> Path:
    ds_dir = tmp_path / "data" / "processed" / "binary_n2_v1"
    ds_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(
        {
            "id": [hashlib.sha256(f"tr-{i}".encode()).hexdigest() for i in range(40)],
            "text": [
                f"bình luận tốt sạch sẽ tích cực {i}"
                if i % 4 != 0
                else f"đồ ngu vcl đm cút đi {i}"
                for i in range(40)
            ],
            "label": [0 if i % 4 != 0 else 1 for i in range(40)],
            "source": ["victsd" if i % 3 == 0 else "vihsd" for i in range(40)],
            "split": ["train"] * 40,
        }
    )
    val_df = pd.DataFrame(
        {
            "id": [hashlib.sha256(f"va-{i}".encode()).hexdigest() for i in range(24)],
            "text": [
                f"bài viết hay quá cảm ơn bạn {i}"
                if i % 4 != 0
                else f"mày ngu vcl đm rác rưởi {i}"
                for i in range(24)
            ],
            "label": [0 if i % 4 != 0 else 1 for i in range(24)],
            "source": ["victsd" if i % 3 == 0 else "vihsd" for i in range(24)],
            "split": ["val"] * 24,
        }
    )
    test_df = pd.DataFrame(
        {
            "id": [hashlib.sha256(f"te-{i}".encode()).hexdigest() for i in range(24)],
            "text": [
                f"nội dung hữu ích và rõ ràng {i}"
                if i % 4 != 0
                else f"lũ ngu vcl đm khốn nạn {i}"
                for i in range(24)
            ],
            "label": [0 if i % 4 != 0 else 1 for i in range(24)],
            "source": ["victsd" if i % 3 == 0 else "vihsd" for i in range(24)],
            "split": ["test"] * 24,
        }
    )

    train_df.to_parquet(ds_dir / "train.parquet", index=False)
    val_df.to_parquet(ds_dir / "val.parquet", index=False)
    test_df.to_parquet(ds_dir / "test.parquet", index=False)

    manifest = {
        row["id"]: {"id": row["id"], "split": row["split"]}
        for row in (
            train_df.to_dict("records")
            + val_df.to_dict("records")
            + test_df.to_dict("records")
        )
    }
    (ds_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (ds_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "test_blind": False,
                "output_sha256": {
                    "train.parquet": _sha256_file(ds_dir / "train.parquet"),
                    "val.parquet": _sha256_file(ds_dir / "val.parquet"),
                    "test.parquet": _sha256_file(ds_dir / "test.parquet"),
                },
            }
        ),
        encoding="utf-8",
    )

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "cnn_config.yaml"
    cfg_path.write_text(
        "experiment: TV4-Synth-N10\n"
        "owner: TV4\n"
        "seeds: [42, 43, 44]\n"
        f"dataset_dir: {ds_dir.as_posix()}\n"
        "device: cpu\n"
        "defaults:\n"
        "  max_word_length: 16\n"
        "  max_char_length: 48\n"
        "  batch_size: 16\n"
        "  epochs: 2\n"
        "  patience: 2\n"
        "  learning_rate: 0.01\n"
        "  min_word_freq: 1\n"
        "  min_char_freq: 1\n"
        "variants:\n"
        "  HYBRID-RNN-01:\n"
        "    description: synth bigru+char\n"
        "    hypothesis: h1\n"
        "    architecture: bigru_char_cnn\n"
        "    preprocessing: raw\n"
        "    model:\n"
        "      embedding_dim: 16\n"
        "      char_embedding_dim: 8\n"
        "      hidden_size: 16\n"
        "      char_kernel_sizes: [3]\n"
        "      num_filters: 8\n"
        "      dropout: 0.2\n"
        "  HYBRID-RNN-WD:\n"
        "    description: synth bigru+char wd\n"
        "    hypothesis: h2\n"
        "    architecture: bigru_char_cnn\n"
        "    preprocessing: raw\n"
        "    weight_decay: 0.0001\n"
        "    model:\n"
        "      embedding_dim: 16\n"
        "      char_embedding_dim: 8\n"
        "      hidden_size: 16\n"
        "      char_kernel_sizes: [3]\n"
        "      num_filters: 8\n"
        "      dropout: 0.2\n"
        "  CNN-01-WD:\n"
        "    description: synth textcnn\n"
        "    hypothesis: h3\n"
        "    architecture: textcnn\n"
        "    preprocessing: basic\n"
        "    weight_decay: 0.0001\n"
        "    model:\n"
        "      embedding_dim: 16\n"
        "      kernel_sizes: [3]\n"
        "      num_filters: 8\n"
        "      dropout: 0.2\n"
        "  CNN-04b:\n"
        "    description: synth textcnn multi\n"
        "    hypothesis: h4\n"
        "    architecture: textcnn\n"
        "    preprocessing: basic\n"
        "    weight_decay: 0.0001\n"
        "    model:\n"
        "      embedding_dim: 16\n"
        "      kernel_sizes: [3, 4]\n"
        "      num_filters: 8\n"
        "      dropout: 0.2\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "models" / "cnn" / "TV4-run-synth"
    summaries = []
    for v_name in ("HYBRID-RNN-01", "HYBRID-RNN-WD", "CNN-01-WD", "CNN-04b"):
        summaries.append(
            run_cnn_variant(
                variant=v_name,
                config_path=cfg_path,
                dataset_dir=ds_dir,
                artifacts_root=run_dir,
                tracking_uri=(tmp_path / "mlruns").as_uri(),
                seeds=[42, 43, 44],
                result_type="smoke",
                overrides={"device": "cpu"},
            )
        )
    (run_dir / "TV4-CNN-summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    return ds_dir


def test_n10_freeze_and_one_shot_locked_test_end_to_end(tmp_path: Path) -> None:
    ds_dir = _setup_synthetic_n10_workspace(tmp_path)
    out_dir = tmp_path / "artifacts" / "TV4-locked-test"
    locked_models_dir = tmp_path / "models" / "locked_test" / "TV4-N10"

    manifest = build_n10_freeze_manifest(
        root=tmp_path,
        models_dir=tmp_path / "models" / "cnn",
        baselines_dir=tmp_path / "models" / "baselines",
        out_dir=out_dir,
    )
    assert manifest["status"] == "FROZEN_BEFORE_TEST_READ"
    assert manifest["test_blind"] is False
    assert "HYBRID-RNN-01" in manifest["frozen_single_tv4"]
    assert (out_dir / "tv4_n10_freeze_manifest.json").exists()

    rep = run_n10_locked_test_evaluation(
        root=tmp_path,
        models_dir=tmp_path / "models" / "cnn",
        baselines_dir=tmp_path / "models" / "baselines",
        locked_models_dir=locked_models_dir,
        out_dir=out_dir,
        device="cpu",
        include_n6_reference=False,
    )
    assert rep["n_test_samples"] == 24
    assert len(rep["single_models"]) == 4
    assert len(rep["ensembles"]) >= 1
    assert "bootstrap_95ci_and_mcnemar" in rep
    assert "HYBRID-RNN-01" in rep["bootstrap_95ci_and_mcnemar"]["bootstrap_95ci_seed42"]
    assert (out_dir / "tv4_n10_locked_test_report.json").exists()
    for p_path in rep["plots"].values():
        assert Path(p_path).exists()
        assert Path(p_path).stat().st_size > 1000
    assert (
        locked_models_dir / "HYBRID-RNN-01" / "seed_42" / "test_predictions.json"
    ).exists()

    # Verify anti-leakage guard rejects hash mismatch on test.parquet
    (ds_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "test_blind": False,
                "output_sha256": {
                    "train.parquet": _sha256_file(ds_dir / "train.parquet"),
                    "val.parquet": _sha256_file(ds_dir / "val.parquet"),
                    "test.parquet": "tampered_hash",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_n10_locked_test_evaluation(
            root=tmp_path,
            models_dir=tmp_path / "models" / "cnn",
            baselines_dir=tmp_path / "models" / "baselines",
            locked_models_dir=locked_models_dir,
            out_dir=out_dir,
            device="cpu",
            include_n6_reference=False,
        )
