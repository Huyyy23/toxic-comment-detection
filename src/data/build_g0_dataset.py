"""Build a versioned train/validation/locked-test candidate from immutable raw JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

LABELS = (0, 1, 2)
SPLITS = ("train", "val", "test")
RATIOS = (0.70, 0.15, 0.15)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw(raw_dir: Path) -> list[dict]:
    rows = []
    for filename, expected_split in (
        ("vihsd_train.jsonl", "train"),
        ("vihsd_validation.jsonl", "val"),
        ("vihsd_test.jsonl", "test"),
    ):
        with (raw_dir / filename).open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                row = json.loads(line)
                if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                        or not isinstance(row.get("text"), str) or not row["text"].strip()
                        or type(row.get("label")) is not int or row["label"] not in LABELS):
                    raise ValueError(f"{filename}:{line_no}: invalid source record")
                if row.get("split") != expected_split:
                    raise ValueError(f"{filename}:{line_no}: raw split mismatch")
                rows.append({"id": row["id"], "text": row["text"], "label": row["label"]})
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate source IDs")
    return rows


def grouped_split(rows: list[dict], seed: int = 42) -> dict[str, str]:
    by_text: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_text[row["text"]].append(row)
    groups = list(by_text.values())
    if len(groups) < len(SPLITS):
        raise ValueError("at least three exact-text groups are required")
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda group: max(Counter(row["label"] for row in group).values()), reverse=True)

    total = Counter(row["label"] for row in rows)
    target_n = [len(rows) * ratio for ratio in RATIOS]
    target_labels = [{label: total[label] * ratio for label in LABELS} for ratio in RATIOS]
    counts = [Counter() for _ in SPLITS]
    assignment = {}

    for group_index, group in enumerate(groups):
        group_labels = Counter(row["label"] for row in group)
        def cost(split_index: int) -> float:
            score = 0.0
            for candidate_index in range(len(SPLITS)):
                next_n = counts[candidate_index]["n"] + (
                    len(group) if candidate_index == split_index else 0
                )
                score += ((next_n - target_n[candidate_index]) / max(target_n[candidate_index], 1)) ** 2
                for label in LABELS:
                    next_count = counts[candidate_index][label] + (
                        group_labels[label] if candidate_index == split_index else 0
                    )
                    target = target_labels[candidate_index][label]
                    score += ((next_count - target) / max(target, 1)) ** 2
            return score
        if group_index < len(SPLITS):
            selected = group_index
        else:
            selected = min(range(len(SPLITS)), key=lambda split_index: (cost(split_index), split_index))
        for row in group:
            assignment[row["id"]] = SPLITS[selected]
        counts[selected]["n"] += len(group)
        counts[selected].update(group_labels)
    return assignment


def build(raw_dir: Path, output_dir: Path, seed: int = 42) -> dict:
    rows = read_raw(raw_dir)
    by_text: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_text[row["text"]].append(row)
    conflict_rows = [row for group in by_text.values()
                     if len({row["label"] for row in group}) > 1 for row in group]
    conflict_ids = {row["id"] for row in conflict_rows}
    candidates = [row for row in rows if row["id"] not in conflict_ids]
    assignment = grouped_split(candidates, seed)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_val_path = output_dir / "train_val.jsonl"
    test_path = output_dir / "test_locked.jsonl"
    manifest_path = output_dir / "manifest.json"
    info_path = output_dir / "dataset_info.json"
    split_rows = {split: [] for split in SPLITS}
    for row in candidates:
        split_rows[assignment[row["id"]]].append({**row, "split": assignment[row["id"]]})
    for group in split_rows.values():
        group.sort(key=lambda row: row["id"])
    for path, split_names in ((train_val_path, ("train", "val")), (test_path, ("test",))):
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for split in split_names:
                for row in split_rows[split]:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {row["id"]: assignment[row["id"]] for row in sorted(candidates, key=lambda row: row["id"])}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw_hashes = {
        filename: sha256(raw_dir / filename)
        for filename in ("vihsd_train.jsonl", "vihsd_validation.jsonl", "vihsd_test.jsonl")
    }
    conflicts_path = output_dir / "excluded_conflict_ids.json"
    conflicts_path.write_text(json.dumps(sorted(conflict_ids), indent=2) + "\n", encoding="utf-8")
    info = {
        "data_contract_version": "1.0",
        "dataset": "UIT-ViHSD",
        "source": "ura-hcmut/UIT-ViHSD",
        "source_revision": "16c4f67cf509d4f9f36ca5b63c5503c7c8830557",
        "source_revision_verified_at": "2026-11-21",
        "license": "cc-by-nc-sa-4.0 (Hugging Face mirror metadata; original UIT permission/license unverified)",
        "citation": "Luu, Son T.; Nguyen, Kiet Van; Nguyen, Ngan Luu-Thuy. A Large-scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts. Advances and Trends in Artificial Intelligence (2021). arXiv:2103.11528; DOI:10.1007/978-3-030-79457-6_35.",
        "source_urls": {
            "dataset_card": "https://huggingface.co/datasets/ura-hcmut/UIT-ViHSD",
            "dataset_api": "https://huggingface.co/api/datasets/ura-hcmut/UIT-ViHSD",
            "original_uit_datasets": "https://nlp.uit.edu.vn/datasets",
            "paper": "https://arxiv.org/abs/2103.11528",
            "doi": "https://doi.org/10.1007/978-3-030-79457-6_35",
        },
        "label_mapping": {"0": "CLEAN", "1": "OFFENSIVE", "2": "HATE"},
        "schema": {"id": "string", "text": "raw source text", "label": "integer 0..2", "split": ["train", "val", "test"]},
        "preprocessing": "none; preserve raw text. Learned transforms fit on train only.",
        "split": {"method": "seeded greedy stratified exact-text-group assignment", "seed": seed, "target_ratios": {s: r for s, r in zip(SPLITS, RATIOS)}, "counts": {s: len(split_rows[s]) for s in SPLITS}},
        "conflicting_text_groups_excluded": len({row["text"] for row in conflict_rows}),
        "excluded_conflict_record_count": len(conflict_ids),
        "excluded_conflict_ids_sha256": sha256(conflicts_path),
        "raw_hashes": raw_hashes,
        "processed_hashes": {train_val_path.name: sha256(train_val_path), test_path.name: sha256(test_path), manifest_path.name: sha256(manifest_path)},
        "test_history": "A prior re-split test was present in shared processed artifacts; replacement test is frozen for EXP-08 with historical exposure disclosed; blind status is not claimed.",
        "test_policy_approved_by_user": True,
        "training_gate": "G0_OPEN_AFTER_ACCEPTANCE; no training was run during acceptance.",
        "test_artifact": test_path.name,
        "test_access_control": "file separation only; no OS-level ACL is configured by this builder.",
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/vihsd_g0_v1"))
    args = parser.parse_args()
    info = build(args.raw_dir, args.output_dir)
    print(json.dumps(info["split"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
