"""Train-only whitespace vocabulary and list-based development batches.

Adapter convention, not an approved G0 data contract: records contain
id (str/int, not bool), nonblank text, label (exact int 0..2), split (train/val).
No normalization, split creation, or locked-test access occurs here.
"""

import json
from pathlib import Path
from typing import Iterable


Record = dict[str, object]


def _tokens(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a nonblank string")
    return text.split()


def _validate_max_length(max_length: int) -> None:
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("max_length must be a positive integer, not bool")


def _validate_record(record: Record, *, train_only: bool = False) -> None:
    if not isinstance(record, dict) or not {"id", "text", "label", "split"} <= record.keys():
        raise ValueError("record must contain id, text, label, and split")
    if not isinstance(record["id"], (str, int)) or isinstance(record["id"], bool):
        raise ValueError("record id must be a string or integer, not bool")
    if isinstance(record["id"], str) and (
        not record["id"] or any(character.isspace() for character in record["id"])
    ):
        raise ValueError("record string id must be nonempty and whitespace-free")
    if type(record["label"]) is not int or record["label"] not in (0, 1, 2):
        raise ValueError("record label must be an integer in 0..2, not bool")
    allowed_splits = ("train",) if train_only else ("train", "val")
    if record["split"] not in allowed_splits:
        raise ValueError(f"record split must be one of {allowed_splits}; test data is locked")
    if not isinstance(record["text"], str) or not record["text"].strip():
        raise ValueError("text must be a nonblank string")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate vocabulary artifact key: {key}")
        result[key] = value
    return result


class Vocabulary:
    """Case/diacritic-preserving str.split vocabulary, fit exactly once on train.

    Learned tokens receive IDs 2 onward in Python lexical order. Reserved IDs
    have no token spelling: literal '<PAD>'/'<UNK>' remain ordinary learned words.
    JSON artifacts store learned tokens only, alongside reserved ID metadata.
    """

    pad_id = 0
    unk_id = 1

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] | None = None

    def _require_fitted(self) -> None:
        if self._token_to_id is None:
            raise RuntimeError("Vocabulary must be fitted first")

    def fit(self, records: Iterable[Record]) -> "Vocabulary":
        """Fit train records atomically; invalid input leaves vocabulary unfitted."""
        if self._token_to_id is not None:
            raise RuntimeError("Vocabulary cannot be refitted")
        tokens = set()
        for record in records:
            _validate_record(record, train_only=True)
            tokens.update(_tokens(record["text"]))
        if not tokens:
            raise ValueError("cannot fit vocabulary on empty train records")
        self._token_to_id = {token: index for index, token in enumerate(sorted(tokens), start=2)}
        return self

    def encode(self, text: str, *, max_length: int) -> list[int]:
        """Encode without learning, truncate right, and leave padding to collate."""
        self._require_fitted()
        _validate_max_length(max_length)
        return [self._token_to_id.get(token, self.unk_id) for token in _tokens(text)[:max_length]]

    def save(self, path: str | Path) -> None:
        """Write versioned UTF-8 JSON; filesystem errors propagate to caller."""
        self._require_fitted()
        artifact = {
            "version": 1,
            "tokenization": "str.split",
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "tokens": list(self._token_to_id),
        }
        Path(path).write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load JSON only; reject unsupported or malformed vocabulary artifacts."""
        artifact = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        keys = {"version", "tokenization", "pad_id", "unk_id", "tokens"}
        if not isinstance(artifact, dict) or set(artifact) != keys:
            raise ValueError("invalid vocabulary artifact schema")
        if type(artifact["version"]) is not int or artifact["version"] != 1:
            raise ValueError("unsupported vocabulary artifact version")
        if artifact["tokenization"] != "str.split":
            raise ValueError("unsupported vocabulary tokenization policy")
        for name, expected in (("pad_id", 0), ("unk_id", 1)):
            if type(artifact[name]) is not int or artifact[name] != expected:
                raise ValueError(f"invalid reserved {name}")
        tokens = artifact["tokens"]
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("vocabulary tokens must be a nonempty list")
        if any(not isinstance(token, str) or not token or token.split() != [token] for token in tokens):
            raise ValueError("vocabulary tokens must be nonempty whitespace-free strings")
        if tokens != sorted(set(tokens)):
            raise ValueError("vocabulary tokens must be unique and lexically sorted")
        vocabulary = cls()
        vocabulary._token_to_id = {token: index for index, token in enumerate(tokens, start=2)}
        return vocabulary


def collate_records(
    records: Iterable[Record], vocabulary: Vocabulary, *, max_length: int
) -> dict[str, list]:
    """Validate train/val batch, then right-pad to longest truncated sequence.

    Output uses lists only; lengths and 1/0 masks count truncated real tokens.
    IDs retain original types and values. No record or vocabulary is modified.
    """
    vocabulary._require_fitted()
    _validate_max_length(max_length)
    ids, input_ids, labels = [], [], []
    for record in records:
        _validate_record(record)
        ids.append(record["id"])
        labels.append(record["label"])
        input_ids.append(vocabulary.encode(record["text"], max_length=max_length))
    if not input_ids:
        raise ValueError("cannot collate an empty batch")
    lengths = [len(sequence) for sequence in input_ids]
    width = max(lengths)
    attention_mask = [[1] * length + [0] * (width - length) for length in lengths]
    for sequence in input_ids:
        sequence.extend([vocabulary.pad_id] * (width - len(sequence)))
    return {
        "ids": ids,
        "input_ids": input_ids,
        "labels": labels,
        "lengths": lengths,
        "attention_mask": attention_mask,
    }
