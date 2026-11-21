"""Official EXP-03 entrypoint: scratch recurrent training on G0 v1 data.

Usage:
  python -m src.models.train_recurrent_official --architecture bilstm_attention
  python -m src.models.train_recurrent_official --architecture lstm --smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from src.data.loader import load_records
from src.data.tokenizer import Vocabulary, collate_records
from src.models.checkpoint import save_checkpoint, seed_everything
from src.models.recurrent import RecurrentClassifier
from src.models.training import evaluate, fit

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "recurrent_official.json"


def _load_dataset(config: dict) -> tuple[list[dict], list[dict], dict]:
    dataset_dir = ROOT / config["dataset_dir"]
    manifest_path = dataset_dir / config["manifest_file"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = load_records(dataset_dir / config["train_val_file"], manifest=manifest)
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    if not train or not val:
        raise ValueError("train/val must be nonempty")
    data_metadata = {
        "dataset_dir": config["dataset_dir"],
        "train_count": len(train),
        "val_count": len(val),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    return train, val, data_metadata


def run(config_path: Path, architecture: str, smoke: bool, output_dir: Path,
        tracking_uri: str | None) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "OFFICIAL":
        raise ValueError("config must have status OFFICIAL")
    if architecture not in config["variants"]:
        raise ValueError(f"architecture {architecture!r} not in variants")

    if smoke:
        cfg = {**config, **config.get("smoke", {})}
        cfg["epochs"] = config["smoke"]["epochs"]
        cfg["batch_size"] = config["smoke"]["batch_size"]
    else:
        cfg = config

    seed_everything(cfg["seed"])
    torch.set_num_threads(1)

    train, val, data_metadata = _load_dataset(config)
    if smoke:
        train = train[: cfg["train_size"]]
        val = val[: cfg["val_size"]]

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output_dir must be new or empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocabulary.json"
    vocabulary = Vocabulary().fit(train) if False else None
    from src.data.tokenizer import Vocabulary as _V
    vocabulary = _V().fit(train)
    vocabulary.save(vocab_path)
    vocab_size = len(json.loads(vocab_path.read_text(encoding="utf-8"))["tokens"]) + 2

    model_config = {
        "vocab_size": vocab_size,
        "embedding_dim": cfg["embedding_dim"],
        "hidden_size": cfg["hidden_size"],
        "dropout": cfg["dropout"],
        "architecture": architecture,
        "pad_id": vocabulary.pad_id,
    }
    model = RecurrentClassifier(**model_config)
    device = cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    batch_size = cfg["batch_size"]

    def batches(records):
        return [collate_records(records[i:i + batch_size], vocabulary,
                                max_length=cfg["max_length"])
                for i in range(0, len(records), batch_size)]

    train_batches = batches(train)
    val_batches = batches(val)

    checkpoint_config = {"model": model_config, "training": cfg, "data": data_metadata}
    state = {"epoch": 0, "best_macro_f1": -1.0, "bad_epochs": 0, "best_epoch": 0}

    tracker = None
    if tracking_uri is not None:
        from src.tracking.recurrent_tracking import (
            finish_tracking, log_epoch, log_run_artifacts, log_run_metadata, start_tracking,
        )
        tracker = start_tracking(tracking_uri, architecture=architecture, resume=None, official=True, smoke=True)

    def checkpoint(epoch_state):
        arguments = dict(
            model=model, optimizer=optimizer, epoch=epoch_state["epoch"],
            best_macro_f1=epoch_state["best_macro_f1"], bad_epochs=epoch_state["bad_epochs"],
            best_epoch=epoch_state["best_epoch"], config=checkpoint_config,
            vocabulary_path=vocab_path, data_metadata=data_metadata,
        )
        save_checkpoint(output_dir / "last.pt", **arguments)
        if epoch_state["best_epoch"] == epoch_state["epoch"]:
            save_checkpoint(output_dir / "best.pt", **arguments)
        if tracker is not None:
            log_epoch(tracker, epoch_state)

    try:
        if tracker is not None:
            log_run_metadata(tracker, config=cfg, model_config=model_config,
                             data_metadata=data_metadata, device=device)
        started = time.perf_counter()
        result = fit(
            model, train_batches, val_batches, optimizer,
            epochs=cfg["epochs"], patience=cfg["patience"], device=device,
            grad_clip=cfg["grad_clip"], start_epoch=state["epoch"],
            best_macro_f1=state["best_macro_f1"], bad_epochs=state["bad_epochs"],
            best_epoch=state["best_epoch"], on_epoch_end=checkpoint,
        )
        duration = time.perf_counter() - started
        validation = evaluate(model, val_batches, device=device)
        summary = {
            "status": "OFFICIAL_SMOKE" if smoke else "OFFICIAL",
            "architecture": architecture,
            "device": device,
            "training": result,
            "last_validation": validation,
            "train_count": len(train),
            "val_count": len(val),
        }
        if tracker is not None:
            finish_tracking(tracker, output_dir, summary, config=checkpoint_config,
                            data_metadata=data_metadata, duration_seconds=duration)
            log_run_artifacts(tracker, output_dir)
        (output_dir / "results.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if tracker is not None:
            tracker.end_run()
    except BaseException:
        if tracker is not None:
            tracker.end_run(status="FAILED")
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--architecture", choices=("lstm", "bilstm", "bilstm_attention"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    result = run(args.config, args.architecture, args.smoke, args.output_dir, args.tracking_uri)
    print(json.dumps({"status": result["status"], "architecture": args.architecture,
                      "device": result["device"], "macro_f1": result["last_validation"]["macro_f1"]}))


if __name__ == "__main__":
    main()
