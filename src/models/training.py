"""Three-class training and validation, without dataset or tracking side effects.

Metrics are JSON-compatible: ``per_class`` is a list ordered CLEAN/OFFENSIVE/HATE,
with precision/recall/f1/support; confusion_matrix rows are true classes, columns
predicted classes. Macro-F1 always averages all three classes, including absent
classes. Evaluation probabilities are uncalibrated softmax scores.
"""

from copy import deepcopy
import math

import torch
from torch.nn import functional as F


def _resolve_device(device):
    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError("device must be cpu, cuda, or auto") from exc
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda, or auto")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ValueError("CUDA device index is unavailable")
    return resolved


def _integer_tensor(value, name, device):
    try:
        tensor = torch.as_tensor(value, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be rectangular integers") from exc
    if tensor.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
        raise ValueError(f"{name} must contain integers")
    return tensor.to(dtype=torch.long)


def _tensorize(batch, device):
    if not batch.get("ids"):
        raise ValueError("empty batch: ids are required")
    try:
        inputs = _integer_tensor(batch["input_ids"], "input_ids", device)
        labels = _integer_tensor(batch["labels"], "labels", device)
        lengths = _integer_tensor(batch["lengths"], "lengths", device)
    except KeyError as exc:
        raise ValueError(f"missing batch field: {exc.args[0]}") from exc
    count = len(batch["ids"])
    if inputs.ndim != 2 or inputs.shape[0] != count or inputs.shape[1] == 0:
        raise ValueError("input_ids must have nonempty shape [batch, tokens]")
    if labels.shape != (count,) or ((labels < 0) | (labels > 2)).any().item():
        raise ValueError("labels must have shape [batch] with class indices 0..2")
    if lengths.shape != (count,) or ((lengths < 1) | (lengths > inputs.shape[1])).any().item():
        raise ValueError("lengths must have shape [batch] and lie within token width")
    if (inputs < 0).any().item():
        raise ValueError("input_ids must be nonnegative")
    return inputs, lengths, labels


def _metrics(loss_sum, count, confusion):
    if count == 0:
        raise ValueError("cannot process empty batches iterable")
    matrix = confusion.cpu()
    per_class = []
    for label in range(3):
        true_positive = matrix[label, label].item()
        predicted = matrix[:, label].sum().item()
        support = matrix[label, :].sum().item()
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * true_positive / (predicted + support) if predicted + support else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1": f1,
                          "support": support})
    return {
        "loss": loss_sum / count,
        "macro_f1": sum(row["f1"] for row in per_class) / 3,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def _run_epoch(model, batches, device, optimizer=None, grad_clip=1.0):
    loss_sum = 0.0
    count = 0
    confusion = torch.zeros((3, 3), dtype=torch.long, device=device)
    predictions = []
    for batch in batches:
        inputs, lengths, labels = _tensorize(batch, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, lengths)
        if logits.shape != (labels.shape[0], 3):
            raise ValueError("logits must have shape [batch, 3]")
        if not torch.isfinite(logits).all().item():
            raise ValueError("logits must be finite")
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss).item():
            raise ValueError("loss must be finite")
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
            optimizer.step()
        predicted = logits.detach().argmax(dim=1)
        confusion += torch.bincount(labels * 3 + predicted, minlength=9).reshape(3, 3)
        size = labels.shape[0]
        loss_sum += loss.detach().item() * size
        count += size
        if optimizer is None:
            probabilities = logits.softmax(dim=1).cpu().tolist()
            true_labels = labels.cpu().tolist()
            predicted_labels = predicted.cpu().tolist()
            predictions.extend({
                "id": row_id, "true_label": true_label, "predicted_label": predicted_label,
                "probabilities_if_available": probability,
            } for row_id, true_label, predicted_label, probability in
                zip(batch["ids"], true_labels, predicted_labels, probabilities))
    result = _metrics(loss_sum, count, confusion)
    if optimizer is None:
        result["predictions"] = predictions
    return result


def _validate_grad_clip(grad_clip):
    if isinstance(grad_clip, bool) or not isinstance(grad_clip, (int, float)):
        raise ValueError("grad_clip must be finite and positive")
    if not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError("grad_clip must be finite and positive")


def train_epoch(model, batches, optimizer, *, device="cpu", grad_clip=1.0):
    """Train one pass; leave model in training mode. Loss is mean per example."""
    _validate_grad_clip(grad_clip)
    resolved = _resolve_device(device)
    model.to(resolved)
    model.train()
    with torch.enable_grad():
        return _run_epoch(model, batches, resolved, optimizer, grad_clip)


def evaluate(model, batches, *, device="cpu"):
    """Evaluate without gradients; restore prior mode even when evaluation raises.

    Prediction order and IDs match input batches. Softmax is not calibration.
    """
    resolved = _resolve_device(device)
    model.to(resolved)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return _run_epoch(model, batches, resolved)
    finally:
        model.train(was_training)


def fit(model, train_batches, val_batches, optimizer, *, epochs, patience,
        device="cpu", grad_clip=1.0, start_epoch=0, best_macro_f1=-1.0,
        bad_epochs=0, on_epoch_end=None, best_epoch=0):
    """Train until total target epochs or validation Macro-F1 patience expires.

    Strict improvements reset bad_epochs; ties count as non-improvements.
    ``start_epoch`` is completed epochs, not a zero-based next epoch index.
    Pass checkpoint best_epoch on resume. History contains only this invocation's
    completed epochs as {epoch, train, val}; val excludes per-example predictions.
    Callback receives an independent JSON-compatible snapshot after each complete
    epoch: epoch, history, best_epoch, best_macro_f1, bad_epochs, stopped_early.
    Model/optimizer remain at last epoch; callback owns best-checkpoint saving.
    Exceptions, including callback failures, propagate to caller.
    """
    for name, value in (("epochs", epochs), ("patience", patience),
                        ("start_epoch", start_epoch), ("bad_epochs", bad_epochs),
                        ("best_epoch", best_epoch)):
        minimum = 1 if name == "patience" else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if start_epoch > epochs:
        raise ValueError("start_epoch cannot exceed total target epochs")
    if best_epoch > start_epoch:
        raise ValueError("best_epoch cannot exceed completed start_epoch")
    if (isinstance(best_macro_f1, bool) or not isinstance(best_macro_f1, (int, float))
            or not math.isfinite(best_macro_f1)
            or not (best_macro_f1 == -1.0 or 0.0 <= best_macro_f1 <= 1.0)):
        raise ValueError("best_macro_f1 must be -1 or finite within [0, 1]")
    _validate_grad_clip(grad_clip)
    _resolve_device(device)
    # Iterators are single-use; retain their batches for subsequent epochs.
    if iter(train_batches) is train_batches:
        train_batches = tuple(train_batches)
    if iter(val_batches) is val_batches:
        val_batches = tuple(val_batches)
    history = []
    stopped_early = bad_epochs >= patience
    for epoch in range(start_epoch + 1, epochs + 1):
        if stopped_early:
            break
        training = train_epoch(model, train_batches, optimizer, device=device, grad_clip=grad_clip)
        validation = evaluate(model, val_batches, device=device)
        validation.pop("predictions")
        history.append({"epoch": epoch, "train": training, "val": validation})
        if validation["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation["macro_f1"]
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        stopped_early = bad_epochs >= patience
        if on_epoch_end is not None:
            on_epoch_end(deepcopy({
                "epoch": epoch, "history": history, "best_epoch": best_epoch,
                "best_macro_f1": best_macro_f1, "bad_epochs": bad_epochs,
                "stopped_early": stopped_early,
            }))
    return {"history": history, "best_epoch": best_epoch, "best_macro_f1": best_macro_f1,
            "bad_epochs": bad_epochs, "stopped_early": stopped_early}
