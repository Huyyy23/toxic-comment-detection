import hashlib
import json

import pandas as pd
import pytest

from src.data.build_binary_training_dataset import build


N2_COLUMNS = ["id", "text", "label", "source", "split"]
N4_COLUMNS = ["id", "text", "label", "split", "is_augmented", "parent_id"]


def _inputs(tmp_path, *, duplicate=None, bad_parent=False):
    n2_dir = tmp_path / "n2"
    n2_dir.mkdir()
    train = pd.DataFrame([
        {"id": "clean", "text": "Clean example", "label": 0, "source": "vihsd", "split": "train"},
        {"id": "toxic", "text": "Toxic example", "label": 1, "source": "victsd", "split": "train"},
    ], columns=N2_COLUMNS)
    val = pd.DataFrame([
        {"id": "val", "text": duplicate or "Validation example", "label": 0,
         "source": "vihsd", "split": "val"},
    ], columns=N2_COLUMNS)
    test = pd.DataFrame([
        {"id": "test", "text": "Test example", "label": 1, "source": "victsd", "split": "test"},
    ], columns=N2_COLUMNS)
    paths = {}
    for split, table in (("train", train), ("val", val), ("test", test)):
        paths[split] = n2_dir / f"{split}.parquet"
        table.to_parquet(paths[split], index=False)
    (n2_dir / "dataset_info.json").write_text(json.dumps({"seed": 42}), encoding="utf-8")
    augmentation_path = tmp_path / "augmentation.parquet"
    parent_id = "missing" if bad_parent else "toxic"
    pd.DataFrame([
        {"id": "augmented", "text": "Toxic altered example", "label": 1, "split": "train",
         "is_augmented": True, "parent_id": parent_id},
    ], columns=N4_COLUMNS).to_parquet(augmentation_path, index=False)
    return n2_dir, augmentation_path, paths


def test_build_copies_splits_and_records_provenance(tmp_path):
    n2_dir, augmentation, paths = _inputs(tmp_path)
    output = tmp_path / "binary_n4_training_v1"

    info = build(n2_dir, augmentation, output)

    train = pd.read_parquet(output / "train.parquet")
    assert list(train.columns) == N2_COLUMNS + ["is_augmented", "parent_id"]
    assert train.loc[train["id"] == "clean", "is_augmented"].item() is False
    assert pd.isna(train.loc[train["id"] == "clean", "parent_id"].item())
    augmented = train.loc[train["id"] == "augmented"].iloc[0]
    assert augmented["source"] == "victsd"
    assert augmented["parent_id"] == "toxic"
    assert (output / "val.parquet").read_bytes() == paths["val"].read_bytes()
    assert (output / "test.parquet").read_bytes() == paths["test"].read_bytes()
    metadata = json.loads((output / "dataset_info.json").read_text(encoding="utf-8"))
    assert metadata == info
    assert metadata["seed"] == 42
    assert metadata["augmentation_count"] == 1
    assert metadata["test_blind"] is False
    assert metadata["normalized_cross_split_duplicate_count"] == 0
    assert metadata["input_sha256"]
    assert all(
        hashlib.sha256((output / filename).read_bytes()).hexdigest() == digest
        for filename, digest in metadata["output_sha256"].items()
    )
    assert {"n2/train.parquet", "n2/val.parquet", "n2/test.parquet",
            "n2/dataset_info.json", "n4/augmentation.parquet"} == set(metadata["input_sha256"])
    assert metadata["split_counts"] == {"train": 3, "val": 1, "test": 1}
    assert {"split": "train", "source": "victsd", "label": 1, "count": 2} in metadata["counts"]


def test_build_refuses_existing_destination(tmp_path):
    n2_dir, augmentation, _ = _inputs(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build(n2_dir, augmentation, output)

    assert marker.read_text(encoding="utf-8") == "unchanged"


@pytest.mark.parametrize("duplicate,bad_parent", [
    ("Ｔoxic-example!!!", False),
    (None, True),
])
def test_build_rejects_normalized_leakage_and_bad_parent(tmp_path, duplicate, bad_parent):
    n2_dir, augmentation, _ = _inputs(tmp_path, duplicate=duplicate, bad_parent=bad_parent)
    with pytest.raises(ValueError):
        build(n2_dir, augmentation, tmp_path / "output")
    assert not (tmp_path / "output").exists()
