import os
import re
import unicodedata
from pathlib import Path

import pandas as pd
import pytest


DATASET = Path(__file__).resolve().parents[1] / "data/processed/binary_n2_v1"


def check_leakage(records):
    originals = {row["id"]: row for row in records if not row.get("is_augmented", False)}
    assert len(originals) == sum(not row.get("is_augmented", False) for row in records), "duplicate original ID"
    exact, normalized = {}, {}
    for row in records:
        split, text = row["split"], row["text"]
        assert split in {"train", "val", "test"}
        assert isinstance(text, str) and text.strip()
        for index, key in ((exact, text), (normalized, re.sub(r"[^\w]+", "", unicodedata.normalize("NFKC", text).casefold()))):
            if key:
                assert key not in index or index[key] == split, "cross-split text leakage"
                index[key] = split
        if row.get("is_augmented", False):
            parent = originals.get(row.get("parent_id"))
            assert parent is not None, "augmentation parent missing"
            assert split == "train" and parent["split"] == "train", "augmentation outside train"
            assert row["label"] == parent["label"] == 1, "augmentation must preserve TOXIC label"


def load_n2():
    records = []
    for split in ("train", "val", "test"):
        rows = pd.read_parquet(DATASET / f"{split}.parquet").to_dict("records")
        assert rows and all(row["split"] == split for row in rows)
        records.extend(rows)
    return records


def test_n2_split_leakage():
    records = load_n2()
    assert {row["split"] for row in records} == {"train", "val", "test"}
    check_leakage(records)


def test_augmentation_file_when_provided():
    path = os.environ.get("N3_AUGMENTATION_PARQUET")
    if not path:
        pytest.skip("N4 augmentation data not supplied")
    augmented = pd.read_parquet(path)
    assert {"id", "text", "label", "split", "is_augmented", "parent_id"} <= set(augmented.columns)
    assert len(augmented) and augmented["is_augmented"].eq(True).all()
    check_leakage(load_n2() + augmented.to_dict("records"))


def test_valid_train_augmentation():
    check_leakage([
        {"id": "p", "text": "Toxic text", "label": 1, "split": "train"},
        {"id": "a", "text": "Toxic variant", "label": 1, "split": "train", "is_augmented": True, "parent_id": "p"},
        {"id": "v", "text": "Other text", "label": 0, "split": "val"},
    ])


@pytest.mark.parametrize("text", ["Toxic text", "ＴＯＸＩＣ, text!!!"])
def test_cross_split_text_is_rejected(text):
    records = [
        {"id": "p", "text": "Toxic text", "label": 1, "split": "train"},
        {"id": "v", "text": text, "label": 1, "split": "val"},
    ]
    with pytest.raises(AssertionError, match="cross-split"):
        check_leakage(records)


@pytest.mark.parametrize("split,parent_id,text", [
    ("val", "p", "variant"),
    ("train", "v", "variant"),
    ("train", "missing", "variant"),
    ("train", "p", "ＶＡＬＩＤＡＴＩＯＮ!!!"),
    ("train", "p", "validation"),
])
def test_bad_augmentation_is_rejected(split, parent_id, text):
    records = [
        {"id": "p", "text": "toxic", "label": 1, "split": "train"},
        {"id": "v", "text": "validation", "label": 1, "split": "val"},
        {"id": "a", "text": text, "label": 1, "split": split, "is_augmented": True, "parent_id": parent_id},
    ]
    with pytest.raises(AssertionError):
        check_leakage(records)
