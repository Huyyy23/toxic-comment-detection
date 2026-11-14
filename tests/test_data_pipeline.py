"""Synthetic-only proof of the development data path; no research metrics."""
import json

from src.data.loader import load_records
from src.data.tokenizer import Vocabulary, collate_records


def test_fixture_loader_vocabulary_batch_roundtrip(tmp_path):
    records = [
        {"id": 0, "text": "bình luận tốt", "label": 0, "split": "train"},
        {"id": "t1", "text": "từ xúc phạm", "label": 1, "split": "train"},
        {"id": "t2", "text": "lời thù ghét", "label": 2, "split": "train"},
        {"id": "v1", "text": "tốt token_chỉ_validation", "label": 0, "split": "val"},
    ]
    path = tmp_path / "synthetic.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records),
                    encoding="utf-8")
    loaded = load_records(path, manifest={row["id"]: row["split"] for row in records})
    vocabulary = Vocabulary()
    vocabulary.fit([row for row in loaded if row["split"] == "train"])
    artifact = tmp_path / "vocabulary.json"
    vocabulary.save(artifact)
    restored = Vocabulary.load(artifact)
    batch = collate_records([loaded[0], loaded[-1]], restored, max_length=3)
    assert batch["ids"] == [0, "v1"]
    assert batch["labels"] == [0, 0]
    assert batch["lengths"] == [3, 2]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 1, 0]]
    assert batch["input_ids"][1] == [
        restored.encode("tốt", max_length=3)[0], restored.unk_id, restored.pad_id,
    ]
    assert restored.encode("token_chỉ_validation", max_length=3) == [restored.unk_id]
