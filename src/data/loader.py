"""Technical train/val adapter; does not approve G0 or create splits."""

from collections.abc import Iterable, Mapping
import json
from os import PathLike


def validate_records(
    records: Iterable[dict], *, manifest: Mapping[str | int, str]
) -> None:
    """Validate without mutation; manifest may include IDs outside this subset.

    IDs are integers (not bool) or nonempty strings without whitespace. Only
    exact text overlap across train/val is checked, not near duplicates or EDA.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be an ID-to-split mapping")
    if not isinstance(records, Iterable) or isinstance(records, (str, bytes, Mapping)):
        raise ValueError("records must be an iterable of dictionaries")

    # Python equates bool/float keys with ints; membership must preserve ID types.
    manifest_by_id = {(type(key), key): split for key, split in manifest.items()}
    seen_ids = set()
    text_splits = {}
    for number, record in enumerate(records, start=1):
        context = f"record {number}"
        if not isinstance(record, dict):
            raise ValueError(f"{context}: expected an object")
        if not {"id", "text", "label", "split"} <= record.keys():
            raise ValueError(f"{context}: missing required id/text/label/split field")

        record_id = record["id"]
        if type(record_id) is not int and not (
            type(record_id) is str
            and record_id
            and not any(character.isspace() for character in record_id)
        ):
            raise ValueError(f"{context}: invalid id")
        text = record["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{context}: text must be a nonblank string")
        if type(record["label"]) is not int or record["label"] not in (0, 1, 2):
            raise ValueError(f"{context}: label must be an integer in 0..2")
        split = record["split"]
        if not isinstance(split, str) or split not in ("train", "val"):
            raise ValueError(f"{context}: split must be train or val; test is locked")

        key = (type(record_id), record_id)
        if key in seen_ids:
            raise ValueError(f"{context}: duplicate id")
        if key not in manifest_by_id:
            raise ValueError(f"{context}: id missing from manifest")
        if manifest_by_id[key] != split:
            raise ValueError(f"{context}: split disagrees with manifest")
        if text in text_splits and text_splits[text] != split:
            raise ValueError(f"{context}: exact text overlap across train/val")
        seen_ids.add(key)
        text_splits[text] = split


def load_records(
    path: str | PathLike[str], *, manifest: Mapping[str | int, str]
) -> list[dict]:
    """Read UTF-8 JSONL in file order, then validate against an explicit manifest.

    Extra fields, original IDs, and original text are retained. Blank lines are
    invalid JSONL records. Validation record numbers are one-based line numbers.
    Filesystem errors remain OSError; content/contract errors become ValueError.
    """
    records = []
    try:
        with open(path, encoding="utf-8") as stream:
            for number, line in enumerate(stream, start=1):
                try:
                    records.append(json.loads(line))
                except ValueError:
                    raise ValueError(f"{path}: line {number}: malformed JSON") from None
    except UnicodeDecodeError:
        raise ValueError(f"{path}: invalid UTF-8 input") from None
    try:
        validate_records(records, manifest=manifest)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None
    return records
