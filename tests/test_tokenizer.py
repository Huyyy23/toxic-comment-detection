"""Synthetic-only checks for train vocabulary and development batches."""

import json

import pytest

from src.data.tokenizer import Vocabulary, collate_records


def record(text="xin chào", *, split="train", label=0, id="r1"):
    return {"id": id, "text": text, "label": label, "split": split}


def test_fit_train_only_lexical_order_and_preserved_text():
    vocabulary = Vocabulary().fit([record("á a A a"), record("xin\tchào")])
    assert vocabulary.pad_id == 0
    assert vocabulary.unk_id == 1
    assert vocabulary.encode("A a chào xin á", max_length=10) == [2, 3, 4, 5, 6]
    reordered = Vocabulary().fit([record("xin chào"), record("a á A")])
    assert reordered.encode("A a chào xin á", max_length=10) == [2, 3, 4, 5, 6]
    with pytest.raises(RuntimeError, match="fit"):
        vocabulary.fit([record("new")])


@pytest.mark.parametrize("split", ["val", "test", "validation", None])
def test_fit_rejects_nontrain_without_partial_learning(split):
    vocabulary = Vocabulary()
    with pytest.raises(ValueError):
        vocabulary.fit([record("trainword"), record("leak", split=split)])
    vocabulary.fit([record("clean")])
    assert vocabulary.encode("trainword leak clean", max_length=3) == [1, 1, 2]


@pytest.mark.parametrize("text", ["", "  \t\n", None, 7, ["word"]])
def test_invalid_text_rejected_everywhere(text):
    with pytest.raises(ValueError):
        Vocabulary().fit([record(text)])
    vocabulary = Vocabulary().fit([record()])
    with pytest.raises(ValueError):
        vocabulary.encode(text, max_length=2)
    with pytest.raises(ValueError):
        collate_records([record(text)], vocabulary, max_length=2)


def test_empty_fit_and_unfitted_operations_rejected(tmp_path):
    vocabulary = Vocabulary()
    with pytest.raises(ValueError):
        vocabulary.fit([])
    with pytest.raises(RuntimeError):
        vocabulary.encode("hello", max_length=2)
    with pytest.raises(RuntimeError):
        vocabulary.save(tmp_path / "vocab.json")
    with pytest.raises(RuntimeError):
        collate_records([record()], vocabulary, max_length=2)


@pytest.mark.parametrize("max_length", [0, -1, True, False, 1.0, "2", None])
def test_invalid_max_length_rejected(max_length):
    vocabulary = Vocabulary().fit([record()])
    with pytest.raises(ValueError):
        vocabulary.encode("xin", max_length=max_length)
    with pytest.raises(ValueError):
        collate_records([record()], vocabulary, max_length=max_length)


def test_encode_oov_never_mutates_serialized_vocabulary(tmp_path):
    vocabulary = Vocabulary().fit([record("xin chào")])
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    vocabulary.save(before)
    assert vocabulary.encode("chỉ val xin", max_length=3) == [1, 1, 3]
    vocabulary.save(after)
    assert before.read_bytes() == after.read_bytes()


def test_utf8_serialization_round_trip_and_reserved_literal_tokens(tmp_path):
    vocabulary = Vocabulary().fit([record("<PAD> <UNK> Chào chào")])
    path = tmp_path / "vocabulary.json"
    vocabulary.save(path)
    assert "Chào" in path.read_text(encoding="utf-8")
    loaded = Vocabulary.load(path)
    assert loaded.pad_id == 0
    assert loaded.unk_id == 1
    assert loaded.encode("<PAD> <UNK> Chào chào mới", max_length=5) == [2, 3, 4, 5, 1]
    with pytest.raises(RuntimeError):
        loaded.fit([record()])


@pytest.mark.parametrize("change", [
    {"version": 2}, {"version": True}, {"tokenization": "lowercase"},
    {"pad_id": 1}, {"pad_id": False}, {"unk_id": 0}, {"unk_id": True},
    {"tokens": ["a", "a"]}, {"tokens": ["b", "a"]}, {"tokens": []},
    {"tokens": [""]}, {"tokens": ["a b"]}, {"tokens": [" a"]},
    {"tokens": [3]}, {"tokens": "a"}, {"extra": "unsupported"},
])
def test_load_rejects_corrupt_artifact(tmp_path, change):
    path = tmp_path / "bad.json"
    Vocabulary().fit([record("a b")]).save(path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact.update(change)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError):
        Vocabulary.load(path)


@pytest.mark.parametrize("content", ["not json", "[]", "{}", "null", '{"version":1,"version":1}', '{"tokens":[NaN]}'])
def test_load_rejects_invalid_json_or_schema(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        Vocabulary.load(path)


def test_collate_dynamic_padding_masks_lengths_and_unchanged_ids():
    vocabulary = Vocabulary().fit([record("a b c d")])
    records = [record("a b c d", id=17, label=2), record("a", id="0017", split="val", label=1)]
    batch = collate_records(records, vocabulary, max_length=3)
    assert batch == {
        "ids": [17, "0017"],
        "input_ids": [[2, 3, 4], [2, 0, 0]],
        "labels": [2, 1],
        "lengths": [3, 1],
        "attention_mask": [[1, 1, 1], [1, 0, 0]],
    }
    assert records[0]["text"] == "a b c d"
    assert all(isinstance(value, list) for value in batch.values())


def test_size_one_batch_uses_actual_truncated_length():
    vocabulary = Vocabulary().fit([record("a b c")])
    assert collate_records([record("a b", split="val")], vocabulary, max_length=20)["input_ids"] == [[2, 3]]
    batch = collate_records([record("unknown b c", split="val")], vocabulary, max_length=1)
    assert batch["input_ids"] == [[1]]
    assert batch["lengths"] == [1]
    assert batch["attention_mask"] == [[1]]


@pytest.mark.parametrize("change", [
    {"id": True}, {"id": None}, {"id": 1.5}, {"id": []},
    {"label": True}, {"label": -1}, {"label": 3}, {"label": 1.0}, {"label": "0"},
    {"split": "test"}, {"split": "validation"}, {"split": None},
])
def test_invalid_record_fields_rejected(change):
    vocabulary = Vocabulary().fit([record()])
    invalid = record()
    invalid.update(change)
    with pytest.raises(ValueError):
        collate_records([invalid], vocabulary, max_length=3)
    with pytest.raises(ValueError):
        Vocabulary().fit([invalid])


@pytest.mark.parametrize("invalid", [None, "text", {}, {"text": "hello", "split": "train"}])
def test_missing_or_invalid_record_rejected(invalid):
    vocabulary = Vocabulary().fit([record()])
    with pytest.raises(ValueError):
        collate_records([invalid], vocabulary, max_length=3)
    with pytest.raises(ValueError):
        Vocabulary().fit([invalid])


def test_empty_batch_rejected():
    vocabulary = Vocabulary().fit([record()])
    with pytest.raises(ValueError, match="empty"):
        collate_records([], vocabulary, max_length=3)


@pytest.mark.parametrize("record_id", ["", " ", "a b", "a\tb"])
def test_invalid_string_ids_rejected(record_id):
    vocabulary = Vocabulary().fit([record()])
    with pytest.raises(ValueError):
        collate_records([record(id=record_id)], vocabulary, max_length=3)
    with pytest.raises(ValueError):
        Vocabulary().fit([record(id=record_id)])
