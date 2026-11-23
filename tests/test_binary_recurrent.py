import sys

import pandas as pd
import pytest
import torch

from src.models.recurrent import RecurrentClassifier
from src.models.train_recurrent_official import _load_parquet_dataset
from src.models.training import evaluate
import src.models.train_recurrent_official as runner



def test_parquet_loader_reads_only_train_and_val_and_infers_classes(tmp_path, monkeypatch):
    pd.DataFrame([
        {"id": "a", "text": "clean", "label": 0, "split": "train"},
        {"id": "b", "text": "toxic", "label": 1, "split": "train"},
    ]).to_parquet(tmp_path / "train.parquet", index=False)
    pd.DataFrame([
        {"id": "v", "text": "validation", "label": 1, "split": "val"},
    ]).to_parquet(tmp_path / "val.parquet", index=False)
    (tmp_path / "test.parquet").write_text("unreadable", encoding="utf-8")
    read_parquet = pd.read_parquet
    reads = []

    def record_read(path, *args, **kwargs):
        reads.append(path)
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", record_read)

    train, val, _, num_classes = _load_parquet_dataset(tmp_path)

    assert num_classes == 2
    assert [row["id"] for row in train] == ["a", "b"]
    assert [row["id"] for row in val] == ["v"]
    assert reads == [tmp_path / "train.parquet", tmp_path / "val.parquet"]


def test_parquet_loader_rejects_validation_label_outside_train_classes(tmp_path):
    pd.DataFrame([
        {"id": "a", "text": "clean", "label": 0},
        {"id": "b", "text": "toxic", "label": 1},
    ]).to_parquet(tmp_path / "train.parquet", index=False)
    pd.DataFrame([
        {"id": "v", "text": "validation", "label": 2},
    ]).to_parquet(tmp_path / "val.parquet", index=False)

    with pytest.raises(ValueError, match="train class range"):
        _load_parquet_dataset(tmp_path)




def test_two_class_forward_and_evaluation_without_optimizer():
    model = RecurrentClassifier(
        8, embedding_dim=4, hidden_size=3, dropout=0, num_classes=2,
    )
    batch = {
        "ids": ["clean", "toxic"],
        "input_ids": [[2, 0], [3, 4]],
        "labels": [0, 1],
        "lengths": [1, 2],
    }
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    logits = model(torch.tensor(batch["input_ids"]), torch.tensor(batch["lengths"]))
    result = evaluate(model, [batch])

    assert logits.shape == (2, 2)
    assert len(result["confusion_matrix"]) == 2
    assert all(len(row) == 2 for row in result["confusion_matrix"])
    assert len(result["per_class"]) == 2
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert all(len(row["probabilities_if_available"]) == 2 for row in result["predictions"])
    for name, value in before.items():
        assert torch.equal(value, model.state_dict()[name])

def test_cli_forwards_dataset_dir_without_training(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    received = {}

    def fake_run(config_path, architecture, smoke, output_dir, tracking_uri,
                 resume=None, dataset_dir=None):
        received["dataset_dir"] = dataset_dir
        return {
            "status": "ok",
            "device": "cpu",
            "last_validation": {"macro_f1": 0.5},
            "run_id": None,
            "artifact_dir": str(output_dir),
        }

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "train_recurrent_official", "--architecture", "lstm",
        "--output-dir", str(tmp_path / "run"), "--dataset-dir", str(dataset_dir),
    ])

    runner.main()

    assert received["dataset_dir"] == dataset_dir
