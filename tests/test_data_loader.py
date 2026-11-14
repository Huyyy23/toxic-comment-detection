"""Synthetic checks for the development-only JSONL adapter."""

from copy import deepcopy
import json
from types import MappingProxyType

import pytest

from src.data.loader import load_records, validate_records


def test_load_preserves_unicode_mixed_ids_order_and_text(tmp_path):
    records = [
        {"id": 0, "text": "  Tiếng Việt 🌿\nXin chào!  ", "label": 0, "split": "train"},
        {"id": "0", "text": "Khác biệt", "label": 2, "split": "val"},
        {"id": "âm", "text": "Một câu nữa", "label": 1, "split": "train"},
    ]
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records), encoding="utf-8")
    manifest = MappingProxyType({0: "train", "0": "val", "âm": "train", "locked": "test"})

    assert load_records(path, manifest=manifest) == records


def test_validate_does_not_mutate_records_or_manifest():
    records = [{"id": "one", "text": " Unchanged ", "label": 1, "split": "train", "extra": [1]}]
    manifest = {"one": "train", "unused": "test"}
    before = deepcopy((records, manifest))

    assert validate_records(records, manifest=manifest) is None
    assert (records, manifest) == before


@pytest.mark.parametrize("field", ["id", "text", "label", "split"])
def test_missing_fields_rejected(field):
    record = {"id": "one", "text": "private comment", "label": 0, "split": "train"}
    del record[field]
    with pytest.raises(ValueError, match="record 1") as error:
        validate_records([record], manifest={"one": "train"})
    assert "private comment" not in str(error.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", ""), ("id", " \t"), ("id", " one"), ("id", "one two"),
        ("id", True), ("id", False), ("id", 1.0), ("id", None), ("id", []),
        ("text", ""), ("text", " \n\t"), ("text", 1), ("text", None),
        ("label", True), ("label", False), ("label", -1), ("label", 3),
        ("label", 1.0), ("label", "1"), ("label", None),
        ("split", "test"), ("split", "validation"), ("split", "Train"),
        ("split", None), ("split", []),
    ],
)
def test_invalid_fields_rejected(field, value):
    record = {"id": "one", "text": "private comment", "label": 0, "split": "train"}
    record[field] = value
    with pytest.raises(ValueError, match="record 1") as error:
        validate_records([record], manifest={"one": "train"})
    assert "private comment" not in str(error.value)


@pytest.mark.parametrize("record", [None, [], "private comment", 3])
def test_non_object_records_rejected(record):
    with pytest.raises(ValueError, match="record 1"):
        validate_records([record], manifest={})


@pytest.mark.parametrize("manifest", [None, [], {}, {"one": "val"}, {"one": "test"}, {"one": None}])
def test_manifest_required_and_must_agree(manifest):
    records = [{"id": "one", "text": "Hello", "label": 0, "split": "train"}]
    with pytest.raises(ValueError, match="manifest"):
        validate_records(records, manifest=manifest)


@pytest.mark.parametrize("manifest", [{"0": "train"}, {False: "train"}, {0.0: "train"}])
def test_manifest_membership_requires_exact_id_type(manifest):
    records = [{"id": 0, "text": "Hello", "label": 0, "split": "train"}]
    with pytest.raises(ValueError, match="manifest"):
        validate_records(records, manifest=manifest)


def test_duplicate_id_rejected():
    records = [
        {"id": 0, "text": "First", "label": 0, "split": "train"},
        {"id": 0, "text": "Second", "label": 1, "split": "train"},
    ]
    with pytest.raises(ValueError, match="record 2.*duplicate"):
        validate_records(records, manifest={0: "train"})


def test_exact_cross_split_text_overlap_rejected_without_text_in_error():
    records = [
        {"id": "one", "text": "private comment", "label": 0, "split": "train"},
        {"id": "two", "text": "private comment", "label": 2, "split": "val"},
    ]
    with pytest.raises(ValueError, match="record 2.*overlap") as error:
        validate_records(records, manifest={"one": "train", "two": "val"})
    assert "private comment" not in str(error.value)


def test_same_split_duplicates_and_nonexact_cross_split_text_allowed():
    records = [
        {"id": 1, "text": "Hello", "label": 0, "split": "train"},
        {"id": 2, "text": "Hello", "label": 1, "split": "train"},
        {"id": 3, "text": " Hello", "label": 0, "split": "val"},
        {"id": 4, "text": "hello", "label": 0, "split": "val"},
    ]
    validate_records(records, manifest={1: "train", 2: "train", 3: "val", 4: "val"})


@pytest.mark.parametrize("line", ['{"text": "private comment",', "", "null", "[]", '"private comment"'])
def test_jsonl_invalid_line_has_path_and_line_context(tmp_path, line):
    path = tmp_path / "invalid.jsonl"
    good = {"id": 0, "text": "Good", "label": 0, "split": "train"}
    path.write_text(json.dumps(good) + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="2") as error:
        load_records(path, manifest={0: "train"})
    assert str(path) in str(error.value)
    assert "private comment" not in str(error.value)


def test_jsonl_rejects_invalid_utf8(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(b"\xff\n")
    with pytest.raises(ValueError, match="UTF-8"):
        load_records(path, manifest={})


def test_jsonl_schema_and_manifest_validation_cannot_be_bypassed(tmp_path):
    path = tmp_path / "records.jsonl"
    record = {"id": "one", "text": "Private", "label": 1, "split": "test"}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split"):
        load_records(path, manifest={"one": "test"})
    record["split"] = "train"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        load_records(path, manifest={})


def test_empty_jsonl_still_requires_manifest(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert load_records(path, manifest={}) == []
    with pytest.raises(ValueError, match="manifest"):
        load_records(path, manifest=None)
