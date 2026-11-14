"""Atomic, weights-only checkpoints for epoch-boundary scratch-model resume.

Vocabulary is embedded as exact UTF-8 JSON text plus its SHA256. Callers may
recreate a missing vocabulary file from ``state['vocabulary_json']``; loading
never writes it. Config/data metadata must match exactly, not merely a subset.
Batch iterator order/state is caller-owned; resume starts at completed ``epoch``.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import torch


_FORMAT_VERSION = 1
_REQUIRED_FIELDS = {
    "format_version", "model_state_dict", "optimizer_state_dict", "epoch",
    "best_epoch", "best_macro_f1", "bad_epochs", "config", "data_metadata",
    "vocabulary_path", "vocabulary_json", "vocabulary_sha256", "rng_state",
    "deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark",
}


def seed_everything(seed, *, deterministic=True):
    """Seed Python and torch CPU/CUDA; fail on unsupported deterministic ops."""
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    if type(deterministic) is not bool:
        raise ValueError("deterministic must be a bool")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def _json_mapping(value, name):
    def valid(item):
        if item is None or type(item) in (str, bool, int):
            return True
        if type(item) is float:
            return math.isfinite(item)
        if type(item) is list:
            return all(valid(child) for child in item)
        if type(item) is dict:
            return all(type(key) is str and valid(child) for key, child in item.items())
        return False

    if type(value) is not dict or not valid(value):
        raise ValueError(f"{name} must be a JSON-compatible dict with finite numbers")


def _vocabulary_json(text):
    if type(text) is not str:
        raise ValueError("vocabulary_json must be UTF-8 JSON text")
    try:
        value = json.loads(text)
        text.encode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid vocabulary JSON") from exc
    _json_mapping(value, "vocabulary")


def _validate(payload):
    if type(payload) is not dict or not _REQUIRED_FIELDS <= payload.keys():
        raise ValueError("malformed checkpoint: missing required fields")
    if type(payload["format_version"]) is not int or payload["format_version"] != _FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format_version")
    for field in ("epoch", "best_epoch", "bad_epochs"):
        if type(payload[field]) is not int or payload[field] < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    if payload["best_epoch"] > payload["epoch"] or payload["bad_epochs"] > payload["epoch"]:
        raise ValueError("best_epoch and bad_epochs cannot exceed completed epoch")
    metric = payload["best_macro_f1"]
    if type(metric) not in (int, float) or not math.isfinite(metric) or not -1 <= metric <= 1:
        raise ValueError("best_macro_f1 must be finite and in [-1, 1]")
    for field in ("config", "data_metadata"):
        _json_mapping(payload[field], field)
    for field in ("model_state_dict", "optimizer_state_dict"):
        if not isinstance(payload[field], dict):
            raise ValueError(f"invalid {field}")
    if type(payload["vocabulary_path"]) is not str or not payload["vocabulary_path"]:
        raise ValueError("invalid vocabulary_path")
    _vocabulary_json(payload["vocabulary_json"])
    digest = hashlib.sha256(payload["vocabulary_json"].encode("utf-8")).hexdigest()
    if digest != payload["vocabulary_sha256"]:
        raise ValueError("embedded vocabulary SHA256 mismatch")
    for field in ("deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark"):
        if type(payload[field]) is not bool:
            raise ValueError(f"invalid {field}")
    rng = payload["rng_state"]
    if type(rng) is not dict or set(rng) != {"python", "torch_cpu", "torch_cuda"}:
        raise ValueError("invalid rng_state")
    try:
        random.Random().setstate(rng["python"])
        torch.Generator(device="cpu").set_state(rng["torch_cpu"].cpu())
        if type(rng["torch_cuda"]) is not list:
            raise ValueError("torch_cuda RNG must be a list")
        for state in rng["torch_cuda"]:
            if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
                raise ValueError("invalid CUDA RNG tensor")
    except (TypeError, ValueError, RuntimeError, AttributeError) as exc:
        raise ValueError("invalid RNG state") from exc


def save_checkpoint(path, *, model, optimizer, epoch, best_macro_f1, bad_epochs,
                    config, vocabulary_path, data_metadata, best_epoch=0):
    """Save complete epoch state; any write failure preserves existing target."""
    vocabulary_path = Path(vocabulary_path)
    vocabulary_bytes = vocabulary_path.read_bytes()
    try:
        vocabulary_json = vocabulary_bytes.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("vocabulary must be UTF-8 JSON") from exc
    payload = {
        "format_version": _FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "bad_epochs": bad_epochs,
        "config": config,
        "data_metadata": data_metadata,
        "vocabulary_path": str(vocabulary_path.resolve()),
        "vocabulary_json": vocabulary_json,
        "vocabulary_sha256": hashlib.sha256(vocabulary_bytes).hexdigest(),
        "rng_state": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    _validate(payload)
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_checkpoint(path, *, model, optimizer, expected_config,
                    expected_data_metadata, map_location="cpu"):
    """Validate provenance first, restore model/optimizer/RNG, return full state.

    Missing original vocabulary is allowed because exact content is embedded.
    An existing vocabulary must match. CUDA RNG topology must match exactly;
    no CPU fallback silently discards CUDA resume state.
    """
    _json_mapping(expected_config, "expected_config")
    _json_mapping(expected_data_metadata, "expected_data_metadata")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    _validate(payload)
    if payload["config"] != expected_config:
        raise ValueError("checkpoint config mismatch")
    if payload["data_metadata"] != expected_data_metadata:
        raise ValueError("checkpoint data_metadata mismatch")
    vocabulary_path = Path(payload["vocabulary_path"])
    if vocabulary_path.exists():
        digest = hashlib.sha256(vocabulary_path.read_bytes()).hexdigest()
        if digest != payload["vocabulary_sha256"]:
            raise ValueError("local vocabulary SHA256 mismatch")
    rng = payload["rng_state"]
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(rng["torch_cuda"]) != cuda_count:
        raise ValueError("CUDA RNG device count mismatch; exact resume requires matching devices")
    for index, state in enumerate(rng["torch_cuda"]):
        try:
            torch.Generator(device=f"cuda:{index}").set_state(state.cpu())
        except (RuntimeError, ValueError) as exc:
            raise ValueError("invalid CUDA RNG state") from exc
    current = model.state_dict()
    saved = payload["model_state_dict"]
    if current.keys() != saved.keys() or any(
        not isinstance(saved[key], torch.Tensor) or saved[key].shape != tensor.shape
        for key, tensor in current.items()
    ):
        raise ValueError("checkpoint model state keys/shapes mismatch")
    saved_optimizer = payload["optimizer_state_dict"]
    groups = saved_optimizer.get("param_groups")
    current_groups = optimizer.state_dict()["param_groups"]
    if (not isinstance(saved_optimizer.get("state"), dict)
            or not isinstance(groups, list) or len(groups) != len(current_groups)
            or any(not isinstance(group, dict) or not isinstance(group.get("params"), list)
                   or len(group["params"]) != len(current_group["params"])
                   for group, current_group in zip(groups, current_groups))):
        raise ValueError("checkpoint optimizer parameter groups mismatch")
    model.load_state_dict(saved)
    optimizer.load_state_dict(saved_optimizer)
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch_cpu"].cpu())
    if cuda_count:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["torch_cuda"]])
    torch.use_deterministic_algorithms(payload["deterministic_algorithms"])
    torch.backends.cudnn.deterministic = payload["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = payload["cudnn_benchmark"]
    return payload
