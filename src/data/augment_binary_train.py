"""Create deterministic, train-only TOXIC text augmentation for N2."""
import argparse
import hashlib
import math
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

from src.features.preprocessing import TEENCODE_DICT


_TEENCODE_VARIANTS = {}
for _short, _full in TEENCODE_DICT.items():
    if " " not in _short and " " not in _full and _short != _full:
        _TEENCODE_VARIANTS.setdefault(_short, set()).add(_full)
        _TEENCODE_VARIANTS.setdefault(_full, set()).add(_short)


del _short, _full


def _key(text):
    return re.sub(r"[^\w]+", "", unicodedata.normalize("NFKC", text).casefold())


def _make_lexicon(train):
    toxic, clean = Counter(), Counter()
    for row in train:
        tokens = set(re.findall(r"\w+", row["text"].casefold()))
        (toxic if row["label"] == 1 else clean).update(tokens)
    return {word for word, count in toxic.items() if count >= 3 and word not in clean}


def _candidate(text, protected, rng):
    matches = list(re.finditer(r"\w+", text))
    mutable = [m for m in matches if m.group().casefold() not in protected and len(m.group()) > 2]
    if not mutable:
        return None
    match = rng.choice(mutable)
    word = match.group()
    mode = rng.randrange(4)
    if mode == 0 and len(word) > 3:
        pos = rng.randrange(1, len(word) - 1)
        changed = word[:pos] + word[pos + 1:]
    elif mode == 1 and len(word) > 3:
        pos = rng.randrange(1, len(word) - 1)
        changed = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
    elif mode == 2:
        changed = "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")
        if changed == word:
            return None
    else:
        variants = _TEENCODE_VARIANTS.get(word.casefold())
        if not variants:
            return None
        changed = rng.choice(sorted(variants))
        if word.isupper():
            changed = changed.upper()
    if changed == word:
        return None
    return text[:match.start()] + changed + text[match.end():]


def build(dataset_dir, output, seed=42):
    dataset_dir, output = Path(dataset_dir), Path(output)
    train = pd.read_parquet(dataset_dir / "train.parquet").to_dict("records")
    val = pd.read_parquet(dataset_dir / "val.parquet")["text"].tolist()
    test = pd.read_parquet(dataset_dir / "test.parquet")["text"].tolist()
    toxic = [row for row in train if row["label"] == 1]
    if not toxic or not any(row["label"] == 0 for row in train):
        raise ValueError("N2 train must contain both binary labels")

    # Exclude validation/test overlaps without using their labels or evaluating outputs.
    seen = {_key(text) for text in [row["text"] for row in train] + val + test}
    protected = _make_lexicon(train)
    rng = random.Random(seed)
    rng.shuffle(toxic)
    target = max(0, math.ceil((0.30 * len(train) - len(toxic)) / 0.70))
    augmented = []
    per_parent = Counter()
    for parent in toxic:
        attempts = 0
        while per_parent[parent["id"]] < 2 and len(augmented) < target and attempts < 30:
            attempts += 1
            text = _candidate(parent["text"], protected, rng)
            if not text or text == parent["text"]:
                continue
            key = _key(text)
            if not key or key in seen:
                continue
            seen.add(key)
            per_parent[parent["id"]] += 1
            digest = hashlib.sha256(f"{parent['id']}:{seed}:{per_parent[parent['id']]}:{text}".encode()).hexdigest()
            augmented.append({
                "id": digest, "text": text, "label": 1, "split": "train",
                "is_augmented": True, "parent_id": parent["id"],
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(augmented, columns=["id", "text", "label", "split", "is_augmented", "parent_id"]).to_parquet(output, index=False)
    print(f"Wrote {len(augmented)} train-only rows; toxic share {((len(toxic) + len(augmented)) / (len(train) + len(augmented))):.1%}; target 30%")
    return augmented


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/processed/binary_n2_v1")
    parser.add_argument("--output", default="data/processed/binary_n4_v1/augmentation.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.dataset_dir, args.output, args.seed)


