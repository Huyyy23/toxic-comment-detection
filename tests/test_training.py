"""CPU-only synthetic checks; no official data or training results."""

from copy import deepcopy
import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.training import evaluate, fit, train_epoch


class ToyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 3)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.tensor([
                [0.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                [0.0, 2.0, 0.0], [0.0, 0.0, 1.0],
            ]))
        self.seen_modes = []
        self.seen_grad = []

    def forward(self, input_ids, lengths):
        self.seen_modes.append(self.training)
        self.seen_grad.append(torch.is_grad_enabled())
        return self.embedding(input_ids[:, 0])


def batch(tokens=(1, 2, 3), labels=(0, 1, 2)):
    return {
        "ids": [f"row-{i}" for i in tokens],
        "input_ids": [[i] for i in tokens],
        "labels": list(labels),
        "lengths": [1] * len(tokens),
    }


def test_evaluate_weights_loss_by_examples_and_preserves_ids_and_model():
    model = ToyClassifier()
    model.train()
    before = deepcopy(model.state_dict())
    batches = [batch((1, 2), (0, 0)), batch((3,), (2,))]
    result = evaluate(model, batches)
    expected = F.cross_entropy(model.embedding.weight[1:], torch.tensor([0, 0, 2]))
    assert result["loss"] == pytest.approx(expected.item())
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 0, 0], [0, 0, 1]]
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0 + 1) / 3)
    assert result["per_class"][0] == pytest.approx({
        "precision": 1.0, "recall": 0.5, "f1": 2 / 3, "support": 2,
    })
    assert [row["id"] for row in result["predictions"]] == ["row-1", "row-2", "row-3"]
    assert [row["true_label"] for row in result["predictions"]] == [0, 0, 2]
    assert [row["predicted_label"] for row in result["predictions"]] == [0, 1, 2]
    for row in result["predictions"]:
        assert len(row["probabilities_if_available"]) == 3
        assert sum(row["probabilities_if_available"]) == pytest.approx(1.0)
    assert model.training
    assert model.seen_modes == [False, False]
    assert model.seen_grad == [False, False]
    for name, value in before.items():
        assert torch.equal(value, model.state_dict()[name])


def test_train_updates_weights_and_runs_with_gradients():
    model = ToyClassifier().eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.embedding.weight.detach().clone()
    result = train_epoch(model, [batch()], optimizer)
    assert not torch.equal(before, model.embedding.weight)
    assert model.training
    assert model.seen_modes == [True]
    assert model.seen_grad == [True]
    assert result["macro_f1"] == pytest.approx(1.0)
    assert "predictions" not in result


def test_macro_f1_includes_absent_classes():
    result = evaluate(ToyClassifier(), [batch((1,), (0,))])
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["per_class"][1] == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0,
    }


def test_fit_early_stop_callback_and_resume_counters():
    model = ToyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    states = []
    first = fit(model, [batch()], [batch()], optimizer, epochs=2, patience=2,
                on_epoch_end=states.append)
    assert [state["epoch"] for state in states] == [1, 2]
    assert len(states[0]["history"]) == 1
    assert first["best_epoch"] == 1
    assert first["bad_epochs"] == 1
    assert not first["stopped_early"]
    resumed_states = []
    resumed = fit(model, [batch()], [batch()], optimizer, epochs=8, patience=2,
                  start_epoch=2, best_epoch=1, best_macro_f1=first["best_macro_f1"],
                  bad_epochs=first["bad_epochs"], on_epoch_end=resumed_states.append)
    assert [state["epoch"] for state in resumed_states] == [3]
    assert resumed["best_epoch"] == 1
    assert resumed["bad_epochs"] == 2
    assert resumed["stopped_early"]
    assert resumed_states[0]["best_epoch"] == 1
    assert resumed_states[0]["bad_epochs"] == 2
    assert resumed_states[0]["history"][0]["epoch"] == 3
    assert resumed_states[0]["history"][0]["val"]["macro_f1"] == 1.0


def test_fit_total_target_epochs_and_callback_error_propagates():
    model = ToyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    result = fit(model, [batch()], [batch()], optimizer, epochs=2, patience=4,
                 start_epoch=2, best_epoch=1, best_macro_f1=1.0)
    assert result["history"] == []
    assert result["best_epoch"] == 1

    def fail(state):
        raise RuntimeError("callback failure")

    with pytest.raises(RuntimeError, match="callback failure"):
        fit(model, [batch()], [batch()], optimizer, epochs=2, patience=2, on_epoch_end=fail)


@pytest.mark.parametrize("batches", [[], [batch((), ())]])
def test_empty_batches_rejected(batches):
    with pytest.raises(ValueError, match="empty"):
        evaluate(ToyClassifier(), batches)


@pytest.mark.parametrize("field,value", [
    ("labels", [[0], [1], [2]]), ("labels", [0, 1]), ("labels", [0, 1, 3]),
    ("labels", [0.0, 1.0, 2.0]), ("lengths", [0, 1, 1]),
    ("lengths", [1, 1]), ("input_ids", [[1], [2]]), ("ids", ["one"]),
])
def test_invalid_batch_shapes_and_values_rejected(field, value):
    invalid = batch()
    invalid[field] = value
    with pytest.raises(ValueError):
        evaluate(ToyClassifier(), [invalid])


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_logits_rejected_and_eval_mode_restored(value):
    model = ToyClassifier().train()
    with torch.no_grad():
        model.embedding.weight[1, 0] = value
    with pytest.raises(ValueError, match="finite"):
        evaluate(model, [batch()])
    assert model.training


def test_invalid_logits_shape_rejected():
    model = ToyClassifier()
    model.embedding = nn.Embedding(4, 2)
    with pytest.raises(ValueError, match="logits"):
        evaluate(model, [batch()])


def test_auto_device_uses_cpu_without_cuda_and_explicit_cuda_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert evaluate(ToyClassifier(), [batch()], device="auto")["macro_f1"] == 1.0
    with pytest.raises(ValueError, match="CUDA"):
        evaluate(ToyClassifier(), [batch()], device="cuda")
    with pytest.raises(ValueError, match="device"):
        evaluate(ToyClassifier(), [batch()], device="mps")


@pytest.mark.parametrize("options", [
    {"patience": 0}, {"patience": 1.5}, {"epochs": -1},
    {"start_epoch": 3}, {"bad_epochs": -1}, {"best_macro_f1": float("nan")},
    {"best_epoch": 2}, {"grad_clip": 0.0},
])
def test_fit_invalid_state_rejected(options):
    model = ToyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    kwargs = {"epochs": 2, "patience": 2, **options}
    with pytest.raises(ValueError):
        fit(model, [batch()], [batch()], optimizer, **kwargs)


def test_finite_logits_with_overflowing_loss_rejected():
    model = ToyClassifier()
    with torch.no_grad():
        limit = torch.finfo(torch.float32).max
        model.embedding.weight[1].copy_(torch.tensor([-limit, limit, 0.0]))
    with pytest.raises(ValueError, match="loss must be finite"):
        evaluate(model, [batch((1,), (0,))])


def test_fit_replays_iterator_batches_and_history_is_json_serializable():
    model = ToyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    snapshots = []
    result = fit(model, iter([batch()]), iter([batch()]), optimizer,
                 epochs=3, patience=5, on_epoch_end=snapshots.append)
    assert [row["epoch"] for row in result["history"]] == [1, 2, 3]
    assert [len(state["history"]) for state in snapshots] == [1, 2, 3]
    assert json.loads(json.dumps(snapshots)) == snapshots


def test_fit_already_exhausted_patience_performs_no_more_updates():
    model = ToyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    result = fit(model, [batch()], [batch()], optimizer, epochs=9, patience=2,
                 start_epoch=4, best_epoch=2, best_macro_f1=1.0, bad_epochs=2)
    assert result["history"] == []
    assert result["stopped_early"]
    assert model.seen_modes == []


def test_evaluate_preserves_existing_gradients_and_optimizer_state():
    model = ToyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    train_epoch(model, [batch()], optimizer)
    gradient = model.embedding.weight.grad.clone()
    momentum = optimizer.state[model.embedding.weight]["momentum_buffer"].clone()
    evaluate(model, [batch()])
    assert torch.equal(model.embedding.weight.grad, gradient)
    assert torch.equal(optimizer.state[model.embedding.weight]["momentum_buffer"], momentum)
