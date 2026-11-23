import hashlib

import pandas as pd
import pytest

from src.models import train_phobert_n13 as n13

@pytest.mark.parametrize("skips_step", [False, True])
def test_scheduler_advances_only_when_mixed_precision_optimizer_step_succeeds(skips_step):
    class FakeOptimizer:
        steps = 0

        def step(self):
            self.steps += 1

    class FakeScheduler:
        steps = 0

        def step(self):
            self.steps += 1

    class FakeScaler:
        def __init__(self):
            self.scale = 8.0

        def get_scale(self):
            return self.scale

        def step(self, optimizer):
            if not skips_step:
                optimizer.step()

        def update(self):
            if skips_step:
                self.scale /= 2

    optimizer, scheduler = FakeOptimizer(), FakeScheduler()
    skipped = n13._step_optimizer(FakeScaler(), optimizer, scheduler)

    assert skipped is skips_step
    assert optimizer.steps == int(not skips_step)
    assert scheduler.steps == int(not skips_step)


def test_n13_uses_shared_binary_metrics_in_clean_toxic_order():
    metrics = n13.binary_metrics([0, 0, 1, 1], [0, 1, 1, 1])

    assert metrics["macro_f1"] == pytest.approx(0.7333333333)
    assert [row["label"] for row in metrics["per_class"]] == ["CLEAN", "TOXIC"]
    assert [row["f1"] for row in metrics["per_class"]] == pytest.approx([2 / 3, 0.8])
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_n13_reads_only_train_and_validation_parquets(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "binary"
    dataset_dir.mkdir()
    for split in ("train", "val", "test"):
        (dataset_dir / f"{split}.parquet").write_bytes(split.encode())
    frames = {
        "train.parquet": pd.DataFrame([{"id": "tr-1", "text": "safe", "label": 0, "split": "train"}]),
        "val.parquet": pd.DataFrame([{"id": "va-1", "text": "toxic", "label": 1, "split": "val"}]),
    }
    reads = []

    def read_parquet(path):
        reads.append(path.name)
        return frames[path.name]

    monkeypatch.setattr(n13.pd, "read_parquet", read_parquet)
    train, val, metadata = n13.load_splits(dataset_dir)

    assert reads == ["train.parquet", "val.parquet"]
    assert [row["id"] for row in train] == ["tr-1"]
    assert [row["id"] for row in val] == ["va-1"]
    assert metadata["train_sha256"] == hashlib.sha256(b"train").hexdigest()
    assert metadata["val_sha256"] == hashlib.sha256(b"val").hexdigest()
