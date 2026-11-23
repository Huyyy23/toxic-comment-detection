from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.checkpoint import save_checkpoint, seed_everything
from src.models.training import _metrics
from src.tracking.mlflow_tracker import MLflowTracker

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "phobert_n13.json"
LABEL_NAMES = ("CLEAN", "TOXIC")


def load_splits(dataset_dir: Path) -> tuple[list[dict], list[dict], dict]:
    dataset_dir = Path(dataset_dir)
    paths = {split: dataset_dir / f"{split}.parquet" for split in ("train", "val")}
    records = {split: pd.read_parquet(path).to_dict("records") for split, path in paths.items()}
    for split, rows in records.items():
        if not rows or not {"id", "text", "label"} <= rows[0].keys():
            raise ValueError(f"{split} parquet must contain id, text, label and be nonempty")
        for row in rows:
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError(f"{split} text must be nonblank")
            label = row["label"]
            if isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer)) or int(label) not in (0, 1):
                raise ValueError(f"{split} labels must be CLEAN=0 or TOXIC=1")
            if "split" in row and row["split"] not in (None, split):
                raise ValueError(f"{split} parquet contains a row from another split")
    return records["train"], records["val"], {
        "dataset_dir": str(dataset_dir),
        "train_count": len(records["train"]),
        "val_count": len(records["val"]),
        "train_sha256": hashlib.sha256(paths["train"].read_bytes()).hexdigest(),
        "val_sha256": hashlib.sha256(paths["val"].read_bytes()).hexdigest(),
    }


def binary_metrics(labels: list[int], predictions: list[int]) -> dict:
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    for label, prediction in zip(labels, predictions):
        confusion[label, prediction] += 1
    metrics = _metrics(0.0, len(labels), confusion)
    metrics.pop("loss")
    for index, name in enumerate(LABEL_NAMES):
        metrics["per_class"][index]["label"] = name
    return metrics

class EncodedComments(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int):
        from pyvi import ViTokenizer

        texts = [ViTokenizer.tokenize(row["text"]) for row in records]
        self.encodings = tokenizer(texts, max_length=max_length, truncation=True, padding=False)
        self.labels = [int(row["label"]) for row in records]
        self.ids = [row["id"] for row in records]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {key: self.encodings[key][index] for key in self.encodings} | {
            "labels": self.labels[index], "row_id": self.ids[index]
        }


def _collate(tokenizer, rows):
    labels = torch.tensor([row["labels"] for row in rows], dtype=torch.long)
    ids = [row["row_id"] for row in rows]
    features = [{key: value for key, value in row.items() if key not in ("labels", "row_id")} for row in rows]
    batch = tokenizer.pad(features, padding=True, return_tensors="pt")
    batch["labels"] = labels
    return batch, ids


def _evaluate(model, loader, device, use_fp16):
    model.eval()
    labels, predictions, rows = [], [], []
    loss_sum = 0.0
    with torch.no_grad():
        for batch, ids in loader:
            targets = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                output = model(**batch, labels=targets)
            probabilities = torch.softmax(output.logits.float(), dim=-1).cpu().tolist()
            predicted = [int(max(range(2), key=p.__getitem__)) for p in probabilities]
            labels.extend(targets.cpu().tolist())
            predictions.extend(predicted)
            loss_sum += float(output.loss) * len(ids)
            rows.extend({"id": row_id, "label": int(label), "prediction": pred,
                         "probabilities": probs}
                        for row_id, label, pred, probs in zip(ids, targets.cpu().tolist(), predicted, probabilities))
    metrics = binary_metrics(labels, predictions)
    metrics["loss"] = loss_sum / len(labels)
    metrics["predictions"] = rows
    return metrics


def _step_optimizer(scaler, optimizer, scheduler) -> bool:
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.get_scale() < previous_scale
    if not skipped:
        scheduler.step()
    return skipped


def run(config_path: Path = DEFAULT_CONFIG, *, dataset: str = "n2", seed: int | None = None) -> dict:
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if dataset not in config["dataset_dirs"]:
        raise ValueError("dataset must be n2 or n4")
    seed = config["seed"] if seed is None else seed
    seed_everything(seed)
    dataset_dir = ROOT / config["dataset_dirs"][dataset]
    train, val, data = load_splits(dataset_dir)
    if config["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])
    use_cuda = device.type == "cuda"
    use_fp16 = bool(config["fp16"] and use_cuda)

    tracking_uri = Path(config["tracking_uri"])
    if not tracking_uri.is_absolute():
        tracking_uri = ROOT / tracking_uri
    tracker = MLflowTracker("N13-phobert", tracking_uri=tracking_uri)
    run_id = tracker.start_run(
        run_name=f"N13-{dataset}-seed-{seed}",
        tags={"exp": "N13", "dataset": dataset, "seed": str(seed),
              "result_type": "official", "pretrained": "true"},
    )
    output_dir = tracker.artifact_path
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        settings = {key: value for key, value in config.items() if key not in ("dataset_dirs", "tracking_uri")}
        settings.update({"dataset": dataset, "seed": seed, "data": data,
                         "segmentation_tool": "PyVi",
                         "segmentation_version": importlib.metadata.version("pyvi"),
                         "transformers_version": transformers.__version__})
        tracker.log_params({"model_name": config["model_name"], "dataset": dataset, "seed": seed,
                            "train_count": data["train_count"], "val_count": data["val_count"],
                            "train_sha256": data["train_sha256"], "val_sha256": data["val_sha256"],
                            "max_length": config["max_length"], "epochs": config["epochs"],
                            "learning_rate": config["learning_rate"], "device": str(device),
                            "batch_size": config["batch_size"],
                            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
                            "fp16": config["fp16"], "warmup_ratio": config["warmup_ratio"],
                            "weight_decay": config["weight_decay"]})
        tracker.log_environment()
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], revision=config.get("revision"))
        tokenizer_revision = (getattr(tokenizer, "_commit_hash", None)
                              or tokenizer.init_kwargs.get("_commit_hash"))
        if not config.get("revision") and tokenizer_revision:
            config["revision"] = tokenizer_revision
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        train_data = EncodedComments(train, tokenizer, config["max_length"])
        val_data = EncodedComments(val, tokenizer, config["max_length"])
        train_loader = DataLoader(
            train_data, batch_size=config["batch_size"], shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            collate_fn=lambda rows: _collate(tokenizer, rows),
        )
        val_loader = DataLoader(
            val_data, batch_size=config["eval_batch_size"], shuffle=False,
            collate_fn=lambda rows: _collate(tokenizer, rows),
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"], revision=config.get("revision"), num_labels=2,
            id2label={0: "CLEAN", 1: "TOXIC"}, label2id={"CLEAN": 0, "TOXIC": 1},
        ).to(device)
        model_revision = getattr(model.config, "_commit_hash", None)
        if tokenizer_revision and model_revision and tokenizer_revision != model_revision:
            raise RuntimeError("PhoBERT model and tokenizer resolved to different revisions")
        revision = model_revision or tokenizer_revision or config.get("revision")
        if not revision:
            raise RuntimeError("Could not resolve an immutable PhoBERT model/tokenizer revision")
        settings.update({"revision": revision, "model_revision": model_revision,
                         "tokenizer_revision": tokenizer_revision,
                         "tokenizer_class": type(tokenizer).__name__})
        if revision and not config.get("revision"):
            config["revision"] = revision
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        settings["config"] = config
        (output_dir / "config.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tracker.log_params({"model_revision": revision or "unresolved",
                            "tokenizer_revision": tokenizer_revision or "unresolved",
                            "tokenizer_class": type(tokenizer).__name__,
                            "transformers_version": transformers.__version__,
                            "pyvi_version": settings["segmentation_version"]})
        tokenizer.save_pretrained(output_dir / "tokenizer")
        model.config.save_pretrained(output_dir / "model_config")
        vocabulary_path = output_dir / "tokenizer_vocab.json"
        vocabulary_path.write_text(json.dumps(tokenizer.get_vocab(), ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
        steps_per_epoch = math.ceil(len(train_loader) / config["gradient_accumulation_steps"])
        total_steps = steps_per_epoch * config["epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * config["warmup_ratio"]), total_steps
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        best_macro_f1, best_epoch, bad_epochs = -1.0, 0, 0
        skipped_optimizer_steps = 0
        history, started = [], time.perf_counter()
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for step, (batch, _) in enumerate(train_loader, 1):
                targets = batch.pop("labels").to(device)
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                    loss = model(**batch, labels=targets).loss
                accumulation = config["gradient_accumulation_steps"]
                group_start = ((step - 1) // accumulation) * accumulation
                group_size = min(accumulation, len(train_loader) - group_start)
                scaler.scale(loss / group_size).backward()
                loss_sum += float(loss.detach()) * len(targets)
                if step % accumulation == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                    skipped_optimizer_steps += _step_optimizer(scaler, optimizer, scheduler)
                    optimizer.zero_grad(set_to_none=True)
            validation = _evaluate(model, val_loader, device, use_fp16)
            row = {"epoch": epoch, "train_loss": loss_sum / len(train_data),
                   "val_loss": validation["loss"], "val_macro_f1": validation["macro_f1"],
                   "val_f1_clean": validation["per_class"][0]["f1"],
                   "val_f1_toxic": validation["per_class"][1]["f1"]}
            history.append(row)
            tracker.log_metrics({key: value for key, value in row.items() if key != "epoch"}, step=epoch)
            if validation["macro_f1"] > best_macro_f1:
                best_macro_f1, best_epoch, bad_epochs = validation["macro_f1"], epoch, 0
                save_checkpoint(
                    output_dir / "best.pt", model=model, optimizer=optimizer, epoch=epoch,
                    best_macro_f1=best_macro_f1, bad_epochs=bad_epochs, config=settings,
                    vocabulary_path=vocabulary_path, data_metadata=data, best_epoch=best_epoch,
                )
                (output_dir / "validation_predictions.json").write_text(
                    json.dumps(validation["predictions"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                best_metrics = {key: value for key, value in validation.items() if key != "predictions"}
            else:
                bad_epochs += 1
            if bad_epochs >= config["patience"]:
                break
        duration = time.perf_counter() - started
        result = {
            "skipped_optimizer_steps": skipped_optimizer_steps,
            "status": "OFFICIAL", "experiment": "N13", "dataset": dataset,
            "architecture": "PhoBERT", "seed": seed, "run_id": run_id,
            "best_epoch": best_epoch, "epochs_completed": len(history),
            "validation": best_metrics, "history": history,
            "training_duration_seconds": duration,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if use_cuda else None,
            "data": data, "model_name": config["model_name"], "revision": revision,
            "transformers_version": transformers.__version__,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_revision": tokenizer_revision,
        }
        (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tracker.log_metrics({"best_val_macro_f1": best_macro_f1, "training_duration_seconds": duration,
                             "skipped_optimizer_steps": skipped_optimizer_steps})
        if use_cuda:
            tracker.log_metrics({"peak_vram_gb": result["peak_vram_bytes"] / (1024 ** 3)})
        tracker.end_run()
        return result
    except BaseException:
        tracker.end_run(status="FAILED")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", choices=("n2", "n4"), default="n2")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    result = run(args.config, dataset=args.dataset, seed=args.seed)
    print(json.dumps({key: result[key] for key in
                      ("status", "dataset", "architecture", "seed", "run_id", "best_epoch", "epochs_completed", "validation", "training_duration_seconds", "peak_vram_bytes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
