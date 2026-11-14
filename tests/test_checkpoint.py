"""Synthetic CPU checkpoint tests; no dataset or pretrained artifacts."""

import copy
import hashlib
import json
import random

import pytest
import torch
from torch import nn

from src.models.checkpoint import load_checkpoint, save_checkpoint, seed_everything


CONFIG = {"architecture": "tiny", "seed": 23}
DATA = {"dataset_sha256": "fixture-data", "split_sha256": "fixture-split"}


def tiny_model():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Dropout(0.3), nn.Linear(4, 3))
    return model, torch.optim.Adam(model.parameters(), lr=0.02)


def step(model, optimizer):
    model.train()
    inputs = torch.randn(4, 3) * random.random()
    labels = torch.tensor([0, 1, 2, 1])
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model(inputs), labels)
    loss.backward()
    optimizer.step()
    return loss.detach()


def save(path, model, optimizer, vocabulary_path):
    save_checkpoint(
        path, model=model, optimizer=optimizer, epoch=2, best_epoch=1,
        best_macro_f1=0.7, bad_epochs=1, config=CONFIG,
        vocabulary_path=vocabulary_path, data_metadata=DATA,
    )


def load(path, model, optimizer, **overrides):
    args = {"expected_config": CONFIG, "expected_data_metadata": DATA}
    args.update(overrides)
    return load_checkpoint(path, model=model, optimizer=optimizer, **args)


@pytest.fixture
def checkpoint(tmp_path):
    seed_everything(23)
    model, optimizer = tiny_model()
    step(model, optimizer)
    vocabulary_path = tmp_path / "vocabulary.json"
    vocabulary_path.write_text(json.dumps({"tokens": ["<PAD>", "<UNK>", "chào"]}, ensure_ascii=False), encoding="utf-8")
    path = tmp_path / "checkpoint.pt"
    save(path, model, optimizer, vocabulary_path)
    return path, model, optimizer, vocabulary_path


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_resumed_training_matches_uninterrupted_model_optimizer_and_rng(checkpoint):
    path, model, optimizer, vocabulary_path = checkpoint
    expected_losses = [step(model, optimizer) for _ in range(3)]
    expected_python = random.random()
    expected_torch = torch.rand(5)
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_model = copy.deepcopy(model.state_dict())

    resumed, resumed_optimizer = tiny_model()
    state = load(path, resumed, resumed_optimizer)
    actual_losses = [step(resumed, resumed_optimizer) for _ in range(3)]
    assert_nested_equal(actual_losses, expected_losses)
    assert_nested_equal(resumed.state_dict(), expected_model)
    assert_nested_equal(resumed_optimizer.state_dict(), expected_optimizer)
    assert random.random() == expected_python
    assert torch.equal(torch.rand(5), expected_torch)
    assert state["epoch"] == 2
    assert state["best_epoch"] == 1
    assert state["best_macro_f1"] == 0.7
    assert state["bad_epochs"] == 1
    assert state["config"] == CONFIG
    assert state["data_metadata"] == DATA
    assert state["vocabulary_path"] == str(vocabulary_path)


def test_embedded_utf8_vocabulary_survives_missing_original(checkpoint):
    path, model, optimizer, vocabulary_path = checkpoint
    original = vocabulary_path.read_bytes()
    vocabulary_path.unlink()
    state = load(path, model, optimizer)
    assert state["vocabulary_json"].encode("utf-8") == original
    assert state["vocabulary_sha256"] == hashlib.sha256(original).hexdigest()
    restored_path = vocabulary_path.parent / "restored.json"
    restored_path.write_bytes(state["vocabulary_json"].encode("utf-8"))
    assert json.loads(restored_path.read_text(encoding="utf-8"))["tokens"][-1] == "chào"


@pytest.mark.parametrize("overrides", [
    {"expected_config": {**CONFIG, "seed": 99}},
    {"expected_data_metadata": {**DATA, "split_sha256": "changed"}},
])
def test_expected_metadata_mismatch_rejected_before_mutation(checkpoint, overrides):
    path, _, _, _ = checkpoint
    model, optimizer = tiny_model()
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    with pytest.raises(ValueError, match="config|data_metadata"):
        load(path, model, optimizer, **overrides)
    assert_nested_equal(model.state_dict(), before)
    assert optimizer.state == {}
    assert torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == python_rng


@pytest.mark.parametrize("corruption", ["format", "missing", "vocab_digest", "vocab_json", "rng", "epoch", "model", "optimizer"])
def test_malformed_checkpoint_rejected_before_mutation(checkpoint, corruption):
    path, _, _, _ = checkpoint
    payload = torch.load(path, weights_only=True)
    if corruption == "format":
        payload["format_version"] = 999
    elif corruption == "missing":
        del payload["optimizer_state_dict"]
    elif corruption == "vocab_digest":
        payload["vocabulary_sha256"] = "0" * 64
    elif corruption == "vocab_json":
        payload["vocabulary_json"] = "not json"
        payload["vocabulary_sha256"] = hashlib.sha256(b"not json").hexdigest()
    elif corruption == "rng":
        payload["rng_state"]["python"] = (3, (), None)
    elif corruption == "epoch":
        payload["epoch"] = -1
    elif corruption == "model":
        payload["model_state_dict"]["0.weight"] = torch.zeros(9, 9)
    else:
        payload["optimizer_state_dict"]["param_groups"] = []
    torch.save(payload, path)
    model, optimizer = tiny_model()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        load(path, model, optimizer)
    assert_nested_equal(model.state_dict(), before)
    assert optimizer.state == {}


def test_changed_local_vocabulary_rejected(checkpoint):
    path, model, optimizer, vocabulary_path = checkpoint
    vocabulary_path.write_text('{"tokens": ["changed"]}', encoding="utf-8")
    with pytest.raises(ValueError, match="vocabulary"):
        load(path, model, optimizer)


@pytest.mark.parametrize("failure", ["serialize", "replace"])
def test_atomic_save_preserves_original_and_cleans_temporary_file(checkpoint, monkeypatch, failure):
    path, model, optimizer, vocabulary_path = checkpoint
    original = path.read_bytes()
    from src.models import checkpoint as module

    def fail(*args, **kwargs):
        if failure == "serialize":
            args[1].write(b"partial write")
        raise OSError("injected write failure")

    if failure == "serialize":
        monkeypatch.setattr(module.torch, "save", fail)
    else:
        monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError, match="injected"):
        save(path, model, optimizer, vocabulary_path)
    assert path.read_bytes() == original
    assert sorted(p.name for p in path.parent.iterdir()) == ["checkpoint.pt", "vocabulary.json"]


def test_invalid_vocabulary_never_overwrites_checkpoint(checkpoint):
    path, model, optimizer, vocabulary_path = checkpoint
    original = path.read_bytes()
    vocabulary_path.write_text("not JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="vocabulary"):
        save(path, model, optimizer, vocabulary_path)
    assert path.read_bytes() == original


def test_seed_everything_repeats_python_torch_and_enables_determinism():
    seed_everything(8)
    expected = random.random(), torch.rand(3)
    seed_everything(8)
    assert random.random() == expected[0]
    assert torch.equal(torch.rand(3), expected[1])
    assert torch.are_deterministic_algorithms_enabled()


@pytest.mark.parametrize("seed", [-1, True, 2.5, "23"])
def test_seed_everything_rejects_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        seed_everything(seed)
