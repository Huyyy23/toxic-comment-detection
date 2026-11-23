import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata

import pandas as pd


N2_COLUMNS = ["id", "text", "label", "source", "split"]
N4_COLUMNS = ["id", "text", "label", "split", "is_augmented", "parent_id"]
TRAIN_COLUMNS = N2_COLUMNS + ["is_augmented", "parent_id"]
DUPLICATE_RULE = "NFKC, casefold, remove non-word characters; no general near-duplicate detection"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(text):
    return re.sub(r"[^\w]+", "", unicodedata.normalize("NFKC", text).casefold())


def _validate_rows(rows, columns, split, *, augmentation=False):
    if list(rows.columns) != columns:
        raise ValueError(f"{split} schema mismatch")
    if rows.empty:
        raise ValueError(f"{split} must be nonempty")
    ids = set()
    for row in rows.to_dict("records"):
        if not isinstance(row["id"], str) or not row["id"].strip() or row["id"] in ids:
            raise ValueError(f"{split} IDs must be unique nonblank strings")
        ids.add(row["id"])
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"{split} text must be nonblank strings")
        if type(row["label"]) is not int or row["label"] not in (0, 1):
            raise ValueError(f"{split} labels must be integer 0 or 1")
        if row["split"] != split:
            raise ValueError(f"{split} contains rows with another split")
        if augmentation:
            if row["is_augmented"] is not True or row["label"] != 1 or row["split"] != "train":
                raise ValueError("N4 rows must be augmented toxic train rows")
            if not isinstance(row["parent_id"], str) or not row["parent_id"].strip():
                raise ValueError("N4 parent IDs must be nonblank strings")
        elif not isinstance(row["source"], str) or not row["source"].strip():
            raise ValueError(f"{split} source must be nonblank strings")
    return ids


def _validate_package(directory, n2_train, n4_rows):
    train = pd.read_parquet(directory / "train.parquet")
    val = pd.read_parquet(directory / "val.parquet")
    test = pd.read_parquet(directory / "test.parquet")
    _validate_rows(train, TRAIN_COLUMNS, "train")
    _validate_rows(val, N2_COLUMNS, "val")
    _validate_rows(test, N2_COLUMNS, "test")
    originals = {row["id"]: row for row in n2_train.to_dict("records")}
    child_counts = Counter()
    seen_ids = set()
    normalized = {}
    for split_rows, split in ((train, "train"), (val, "val"), (test, "test")):
        for row in split_rows.to_dict("records"):
            if row["id"] in seen_ids:
                raise ValueError("IDs must be unique across splits")
            key = _key(row["text"])
            if key:
                previous_split = normalized.get(key)
                if previous_split is not None and previous_split != split:
                    raise ValueError("normalized duplicate text across splits")
                normalized[key] = split
            if split == "train" and row["is_augmented"]:
                parent = originals.get(row["parent_id"])
                if parent is None or parent["label"] != 1:
                    raise ValueError("augmentation parent must be a toxic N2 train row")
                if row["source"] != parent["source"]:
                    raise ValueError("augmentation source must match its parent")
                child_counts[row["parent_id"]] += 1
            elif split == "train" and (row["is_augmented"] is not False or row["parent_id"] is not None):
                raise ValueError("original train rows must have false/null augmentation provenance")
            elif split != "train" and ("is_augmented" in row or "parent_id" in row):
                raise ValueError("augmentation fields are train-only")
    if any(count > 2 for count in child_counts.values()):
        raise ValueError("at most two augmentations are allowed per parent")
    if sum(child_counts.values()) != len(n4_rows):
        raise ValueError("staged train augmentation count does not match input")
    return train, val, test
def build(n2_dir="data/processed/binary_n2_v1",
          augmentation_path="data/processed/binary_n4_v1/augmentation.parquet",
          output_dir="data/processed/binary_n4_training_v1"):
    n2_dir, augmentation_path, output_dir = map(Path, (n2_dir, augmentation_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    n2_paths = {split: n2_dir / f"{split}.parquet" for split in ("train", "val", "test")}
    n2_info_path = n2_dir / "dataset_info.json"
    inputs_hashes = {f"n2/{path.name}": _sha256(path) for path in n2_paths.values()}
    inputs_hashes["n2/dataset_info.json"] = _sha256(n2_info_path)
    inputs_hashes["n4/augmentation.parquet"] = _sha256(augmentation_path)
    n2_tables = {split: pd.read_parquet(path) for split, path in n2_paths.items()}
    originals = {}
    for split, table in n2_tables.items():
        originals[split] = _validate_rows(table, N2_COLUMNS, split)
    all_ids = [record_id for ids in originals.values() for record_id in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("N2 IDs must be unique across splits")
    n4 = pd.read_parquet(augmentation_path)
    _validate_rows(n4, N4_COLUMNS, "train", augmentation=True)
    parent_rows = {row["id"]: row for row in n2_tables["train"].to_dict("records")}
    children = Counter()
    additions = []
    for row in n4.to_dict("records"):
        parent = parent_rows.get(row["parent_id"])
        if parent is None or parent["label"] != 1:
            raise ValueError("N4 parent must exist in N2 train with toxic label")
        children[row["parent_id"]] += 1
        additions.append({**row, "source": parent["source"]})
    if any(count > 2 for count in children.values()):
        raise ValueError("at most two augmentations are allowed per parent")
    combined = n2_tables["train"].copy()
    combined["is_augmented"] = False
    combined["parent_id"] = None
    train = pd.concat(
        [combined[TRAIN_COLUMNS], pd.DataFrame(additions, columns=TRAIN_COLUMNS)],
        ignore_index=True,
    )

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        train.to_parquet(stage / "train.parquet", index=False)
        for split in ("val", "test"):
            shutil.copyfile(n2_paths[split], stage / f"{split}.parquet")
        train, val, test = _validate_package(stage, n2_tables["train"], additions)
        counts = Counter(
            (split, row["source"], row["label"])
            for split, table in (("train", train), ("val", val), ("test", test))
            for row in table.to_dict("records")
        )
        info = {
            "seed": json.loads(n2_info_path.read_text(encoding="utf-8"))["seed"],
            "input_sha256": inputs_hashes,
            "output_sha256": {
                f"{split}.parquet": _sha256(stage / f"{split}.parquet")
                for split in ("train", "val", "test")
            },
            "split_counts": {
                split: int(len(table))
                for split, table in (("train", train), ("val", val), ("test", test))
            },
            "counts": [
                {"split": split, "source": source, "label": label, "count": count}
                for (split, source, label), count in sorted(counts.items())
            ],
            "source_names": sorted({
                row["source"]
                for table in (train, val, test)
                for row in table.to_dict("records")
            }),
            "augmentation_count": len(additions),
            "duplicate_rule": DUPLICATE_RULE,
            "normalized_cross_split_duplicate_count": 0,
            "test_blind": False,
        }
        (stage / "dataset_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        if json.loads((stage / "dataset_info.json").read_text(encoding="utf-8")) != info:
            raise ValueError("staged metadata validation failed")
        if output_dir.exists():
            raise FileExistsError(output_dir)
        os.rename(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n2-dir", type=Path, default=Path("data/processed/binary_n2_v1"))
    parser.add_argument("--augmentation", type=Path,
                        default=Path("data/processed/binary_n4_v1/augmentation.parquet"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/processed/binary_n4_training_v1"))
    args = parser.parse_args()
    print(json.dumps(build(args.n2_dir, args.augmentation, args.output_dir), indent=2))


if __name__ == "__main__":
    main()




