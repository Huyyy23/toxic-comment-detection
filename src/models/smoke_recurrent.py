"""Run EXP-03 technical checks on synthetic records only, never official training.

Usage: python -m src.models.smoke_recurrent --output-dir <temporary-directory>
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from src.data.tokenizer import Vocabulary, collate_records
from src.models.checkpoint import load_checkpoint, save_checkpoint, seed_everything
from src.models.recurrent import RecurrentClassifier
from src.models.training import evaluate, fit


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "recurrent_config.json"


def run_smoke(output_dir, *, config_path=DEFAULT_CONFIG, architecture="lstm", resume=None,
              tracking_uri=None):
    """Create tiny synthetic artifacts; returned metrics are NOT research results."""
    if Path(output_dir).exists() and any(Path(output_dir).iterdir()):
        raise ValueError("output_dir must be new or empty to preserve existing artifacts")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("status") != "PROVISIONAL_SYNTHETIC_SMOKE_ONLY":
        raise ValueError("smoke config must explicitly declare synthetic-only status")
    if architecture not in config["variants"]:
        raise ValueError("architecture not listed in config variants")
    seed_everything(config["seed"])
    torch.set_num_threads(1)
    train = [
        {"id": f"synthetic-t{i}", "text": text, "label": label, "split": "train"}
        for i, (text, label) in enumerate([
            ("bình luận tốt", 0), ("lời nhẹ nhàng", 0),
            ("từ xúc phạm", 1), ("câu thô lỗ", 1),
            ("lời thù ghét", 2), ("câu kỳ thị", 2),
        ])
    ]
    val = [
        {"id": f"synthetic-v{i}", "text": text, "label": i, "split": "val"}
        for i, text in enumerate(["tốt chưa_biết", "xúc phạm", "thù ghét"])
    ]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vocab_path = output / "vocabulary.json"
    vocabulary = Vocabulary().fit(train)
    vocabulary.save(vocab_path)
    vocab_size = len(json.loads(vocab_path.read_text(encoding="utf-8"))["tokens"]) + 2
    model_config = {name: config[name] for name in ("embedding_dim", "hidden_size", "dropout")}
    model_config.update(vocab_size=vocab_size, architecture=architecture, pad_id=vocabulary.pad_id)
    model = RecurrentClassifier(**model_config)
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    data_metadata = {
        "status": "SYNTHETIC_NOT_RESEARCH",
        "dataset_split_sha256": hashlib.sha256(json.dumps(
            train + val, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    checkpoint_config = {"model": model_config, "training": config}
    state = {"epoch": 0, "best_macro_f1": -1.0, "bad_epochs": 0, "best_epoch": 0}
    if resume is not None:
        state = load_checkpoint(
            resume, model=model, optimizer=optimizer, expected_config=checkpoint_config,
            expected_data_metadata=data_metadata, map_location=device,
        )
    batch_size = config["batch_size"]
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    def batches(records):
        return [collate_records(records[i:i + batch_size], vocabulary,
                                max_length=config["max_length"])
                for i in range(0, len(records), batch_size)]

    tracker = None
    if tracking_uri is not None:
        from src.tracking.recurrent_tracking import (
            finish_tracking, log_epoch, log_run_artifacts, log_run_metadata, start_tracking,
        )
        tracker = start_tracking(tracking_uri, architecture=architecture, resume=resume)

    def checkpoint(epoch_state):
        arguments = dict(
            model=model, optimizer=optimizer, epoch=epoch_state["epoch"],
            best_macro_f1=epoch_state["best_macro_f1"], bad_epochs=epoch_state["bad_epochs"],
            best_epoch=epoch_state["best_epoch"], config=checkpoint_config,
            vocabulary_path=vocab_path, data_metadata=data_metadata,
        )
        save_checkpoint(output / "last.pt", **arguments)
        if epoch_state["best_epoch"] == epoch_state["epoch"]:
            save_checkpoint(output / "best.pt", **arguments)
        if tracker is not None:
            log_epoch(tracker, epoch_state)

    try:
        if tracker is not None:
            log_run_metadata(tracker, config=config, model_config=model_config,
                             data_metadata=data_metadata, device=device)
        started = time.perf_counter()
        result = fit(
            model, batches(train), batches(val), optimizer,
            epochs=config["epochs"], patience=config["patience"], device=device,
            grad_clip=config["grad_clip"], start_epoch=state["epoch"],
            best_macro_f1=state["best_macro_f1"], bad_epochs=state["bad_epochs"],
            best_epoch=state["best_epoch"], on_epoch_end=checkpoint,
        )
        duration = time.perf_counter() - started
        validation = evaluate(model, batches(val), device=device)
        summary = {"status": "SYNTHETIC_NOT_RESEARCH", "architecture": architecture,
                   "device": device, "training": result, "last_validation": validation}
        if tracker is not None:
            finish_tracking(tracker, output, summary, config=checkpoint_config,
                            data_metadata=data_metadata, duration_seconds=duration)
        (output / "smoke_results.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if tracker is not None:
            log_run_artifacts(tracker, output)
    except BaseException:
        if tracker is not None:
            tracker.end_run(status="FAILED")
        raise
    else:
        if tracker is not None:
            tracker.end_run()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--architecture", choices=("lstm", "bilstm", "bilstm_attention"), default="lstm")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tracking-uri", help="Local MLflow path/file URI; synthetic experiment only")
    args = parser.parse_args()
    result = run_smoke(args.output_dir, config_path=args.config,
                       architecture=args.architecture, resume=args.resume,
                       tracking_uri=args.tracking_uri)
    print(json.dumps({"status": result["status"], "architecture": args.architecture,
                      "device": result["device"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
