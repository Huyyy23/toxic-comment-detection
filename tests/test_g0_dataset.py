import hashlib
import json
from pathlib import Path

from src.data.build_g0_dataset import build, grouped_split


def write_raw(raw_dir: Path, rows: list[dict]) -> None:
    raw_dir.mkdir()
    groups = {"train": [], "validation": [], "test": []}
    for index, row in enumerate(rows):
        split = ("train", "validation", "test")[index % 3]
        groups[split].append({**row, "split": "val" if split == "validation" else split})
    for filename, split in (("vihsd_train.jsonl", "train"),
                            ("vihsd_validation.jsonl", "validation"),
                            ("vihsd_test.jsonl", "test")):
        with (raw_dir / filename).open("w", encoding="utf-8") as stream:
            for row in groups[split]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_build_excludes_conflicting_texts_groups_exact_duplicates_and_is_reproducible(tmp_path):
    raw = tmp_path / "raw"
    write_raw(raw, [
        {"id": "a", "text": "same", "label": 0},
        {"id": "b", "text": "same", "label": 0},
        {"id": "c", "text": "conflict", "label": 1},
        {"id": "d", "text": "conflict", "label": 2},
        {"id": "e", "text": "other one", "label": 0},
        {"id": "f", "text": "other two", "label": 1},
        {"id": "g", "text": "other three", "label": 2},
        {"id": "h", "text": "other four", "label": 0},
        {"id": "i", "text": "other five", "label": 1},
        {"id": "j", "text": "other six", "label": 2},
    ])
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()}
    first, second = tmp_path / "first", tmp_path / "second"
    first_info = build(raw, first)
    second_info = build(raw, second)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()} == source_hashes
    train_val = [json.loads(line) for line in (first / "train_val.jsonl").read_text(encoding="utf-8").splitlines()]
    test = [json.loads(line) for line in (first / "test_locked.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = train_val + test
    assert {row["id"] for row in rows}.isdisjoint({"c", "d"})
    assert {row["id"] for row in rows} == {"a", "b", "e", "f", "g", "h", "i", "j"}
    assert first_info["conflicting_text_groups_excluded"] == 1
    assert first_info["source_revision"]
    assert "unverified" in first_info["license"]
    assert "10.1007/978-3-030-79457-6_35" in first_info["citation"]
    assert {row["split"] for row in test} == {"test"}
    assert all(row["split"] in {"train", "val"} for row in train_val)
    assignments = {row["id"]: row["split"] for row in rows}
    assert assignments["a"] == assignments["b"]
    assert not {row["text"] for row in train_val} & {row["text"] for row in test}
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == assignments
    assert not any(row["split"] == "test" for row in train_val)
    assert all(row["split"] == "test" for row in test)
    info = json.loads((first / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["processed_hashes"]["manifest.json"] == hashlib.sha256((first / "manifest.json").read_bytes()).hexdigest()
    try:
        build(raw, first)
    except FileExistsError:
        pass
    else:
        raise AssertionError("non-empty output must not be overwritten")


def test_grouped_split_is_deterministic_and_never_splits_exact_text_group():
    rows = [{"id": str(i), "text": f"group-{i // 2}", "label": i % 3} for i in range(60)]
    first = grouped_split(rows, seed=42)
    assert first == grouped_split(rows, seed=42)
    assert all(first[str(i)] == first[str(i + 1)] for i in range(0, len(rows), 2))
