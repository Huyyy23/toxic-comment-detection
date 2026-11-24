"""Pretrained Transformer benchmark pipeline (PhoBERT-v2 & XLM-RoBERTa) for TV4 V3.

Strictly separated from scratch core models (AGENTS.md §6 / SPEC N13):
- Supports full-train and 5-fold stratified Train-OOF CV (leak-free calibration/stacking).
- Safe for RTX 4060 Laptop 8 GB VRAM via fp16 AMP, gradient accumulation, and max_length=160.
- Supports atomic best.pt checkpointing, MLflow logging, and automatic [RESUME-SKIP].
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyvi import ViTokenizer
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
import yaml

from src.models.checkpoint import seed_everything
from src.models.svm_binary_pipeline import (
    BINARY_LABEL_NAMES,
    _build_aggregate_error_summary,
    compute_binary_metrics,
    compute_ece,
    load_binary_train_val,
)
from src.models.training import _resolve_device
from src.tracking.mlflow_tracker import MLflowTracker

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRANSFORMER_CFG = ROOT / "configs" / "transformer_v3_config.yaml"

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_ZW_RE = re.compile(r"[\u200b-\u200f\ufeff]")
_WS_RE = re.compile(r"\s+")
_VI_TONE_FIXES = (
    ("òa", "oà"),
    ("óa", "oá"),
    ("ỏa", "oả"),
    ("õa", "oã"),
    ("ọa", "oạ"),
    ("òe", "oè"),
    ("óe", "oé"),
    ("ỏe", "oẻ"),
    ("õe", "oẽ"),
    ("ọe", "oẹ"),
    ("ùy", "uỳ"),
    ("úy", "uý"),
    ("ủy", "uỷ"),
    ("ũy", "uỹ"),
    ("ụy", "uỵ"),
)


def clean_transformer_text(text: str, use_pyvi: bool = False) -> str:
    """Minimal static normalization preserving case, punctuation, and emojis."""
    s = unicodedata.normalize("NFC", str(text))
    s = _ZW_RE.sub("", s)
    s = _URL_RE.sub(" ", s)
    s = _HTML_RE.sub(" ", s)
    for old, new in _VI_TONE_FIXES:
        s = s.replace(old, new).replace(old.upper(), new.upper())
    s = _WS_RE.sub(" ", s).strip() or " "
    if use_pyvi:
        s = ViTokenizer.tokenize(s)
    return s


class _PreTokenizedDataset(Dataset):
    """Pre-tokenized input_ids & attention_mask list for fast GPU epoch iteration."""

    def __init__(
        self,
        ids: list[str],
        sources: list[str],
        labels: list[int],
        encodings: list[dict[str, list[int]]],
    ) -> None:
        self.ids = ids
        self.sources = sources
        self.labels = labels
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "id": self.ids[idx],
            "source": self.sources[idx],
            "label": self.labels[idx],
            "input_ids": self.encodings[idx]["input_ids"],
            "attention_mask": self.encodings[idx]["attention_mask"],
        }


def _make_collate_fn(pad_token_id: int):
    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(x["input_ids"]) for x in batch)
        bsz = len(batch)
        input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        for i, item in enumerate(batch):
            seq = item["input_ids"]
            mask = item["attention_mask"]
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, : len(mask)] = torch.tensor(mask, dtype=torch.long)
        return {
            "ids": [x["id"] for x in batch],
            "sources": [x["source"] for x in batch],
            "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    return _collate


class TransformerToxicClassifier(nn.Module):
    """Pretrained backbone + Mean/CLS/Gated pooling + Multi-Sample Dropout head."""

    def __init__(
        self,
        model_name: str,
        pooling: str = "mean_cls",
        dropout: float = 0.2,
        multi_sample_dropout: bool = True,
        gradient_checkpointing: bool = False,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pooling = pooling
        self.backbone = AutoModel.from_pretrained(model_name)
        if gradient_checkpointing and hasattr(
            self.backbone, "gradient_checkpointing_enable"
        ):
            self.backbone.gradient_checkpointing_enable()
        hidden = int(self.backbone.config.hidden_size)
        out_dim = hidden * 2 if pooling == "mean_cls" else hidden
        self.norm = nn.LayerNorm(out_dim)
        self.multi_sample_dropout = bool(multi_sample_dropout)
        if self.multi_sample_dropout:
            self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(5)])
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(dropout)])
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state  # [B, T, H]
        cls_vec = seq_out[:, 0, :]
        if self.pooling == "cls":
            feat = cls_vec
        else:
            mask_f = attention_mask.unsqueeze(-1).to(dtype=seq_out.dtype)
            mean_vec = (seq_out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            feat = torch.cat((cls_vec, mean_vec), dim=1) if self.pooling == "mean_cls" else mean_vec
        feat = self.norm(feat)
        logits = torch.stack(
            [self.classifier(dp(feat)) for dp in self.dropouts], dim=0
        ).mean(dim=0)
        return logits


def _save_atomic_pt(path: Path, payload: dict[str, Any]) -> None:
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            tmp_path = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _encode_dataframe(
    df: pd.DataFrame,
    tokenizer: Any,
    max_length: int,
    use_pyvi: bool,
) -> _PreTokenizedDataset:
    raw_texts = df["text"].astype(str).tolist()
    cleaned = [clean_transformer_text(t, use_pyvi=use_pyvi) for t in raw_texts]
    encodings: list[dict[str, list[int]]] = []
    # Batch tokenize for speed
    chunk = 1024
    for start in range(0, len(cleaned), chunk):
        enc = tokenizer(
            cleaned[start : start + chunk],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        for i_ids, a_mask in zip(enc["input_ids"], enc["attention_mask"]):
            encodings.append({"input_ids": i_ids, "attention_mask": a_mask})
    return _PreTokenizedDataset(
        ids=df["id"].astype(str).tolist(),
        sources=df["source"].astype(str).tolist(),
        labels=df["label"].astype(int).tolist(),
        encodings=encodings,
    )


def _eval_transformer_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool = True,
    label_smoothing: float = 0.0,
    collect_outputs: bool = False,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    sources: list[str] = []
    probs_list: list[np.ndarray] = []
    ids: list[str] = []
    try:
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = model(input_ids, attention_mask)
                    loss = F.cross_entropy(
                        logits, labels, label_smoothing=label_smoothing
                    )
                probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                preds = probs.argmax(axis=1).tolist()
                bsz = int(labels.shape[0])
                total_loss += float(loss.item()) * bsz
                total_count += bsz
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds)
                sources.extend(batch["sources"])
                probs_list.append(probs)
                if collect_outputs:
                    ids.extend(batch["ids"])
    finally:
        model.train(was_training)

    all_probs = np.vstack(probs_list) if probs_list else np.zeros((0, 2), dtype=np.float32)
    metrics = compute_binary_metrics(y_true, y_pred, sources, all_probs[:, 1])
    metrics["loss"] = total_loss / max(1, total_count)
    if collect_outputs:
        metrics["ids"] = ids
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred
        metrics["sources"] = sources
        metrics["probs"] = all_probs
    return metrics


def _build_optimizer_param_groups(
    model: TransformerToxicClassifier,
    lr: float,
    head_lr_mult: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias", "norm.weight", "norm.bias")
    backbone_decay, backbone_nodecay, head_decay, head_nodecay = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = not name.startswith("backbone.")
        is_nd = any(nd in name for nd in no_decay)
        if is_head:
            (head_nodecay if is_nd else head_decay).append(param)
        else:
            (backbone_nodecay if is_nd else backbone_decay).append(param)
    return [
        {"params": backbone_decay, "lr": lr, "weight_decay": weight_decay},
        {"params": backbone_nodecay, "lr": lr, "weight_decay": 0.0},
        {"params": head_decay, "lr": lr * head_lr_mult, "weight_decay": weight_decay},
        {"params": head_nodecay, "lr": lr * head_lr_mult, "weight_decay": 0.0},
    ]


def run_transformer_single_split(
    variant: str,
    seed: int,
    v_cfg: dict[str, Any],
    defaults: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    data_hashes: dict[str, str],
    out_dir: Path,
    *,
    experiment_name: str = "TV4-Transformer-V3",
    tracking_uri: str | None = None,
    result_type: str = "official",
    extra_eval_df: pd.DataFrame | None = None,
    extra_eval_name: str | None = None,
    save_weights: bool = True,
) -> dict[str, Any]:
    """Train one Transformer seed/fold on (train_df, val_df) with automatic resume."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if (
        result_type == "official"
        and (out_dir / "metrics.json").exists()
        and (out_dir / "val_predictions.json").exists()
        and (out_dir / "run_id.txt").exists()
        and (extra_eval_name is None or (out_dir / f"{extra_eval_name}_predictions.json").exists())
    ):
        cached = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        cached_run_id = (out_dir / "run_id.txt").read_text(encoding="utf-8").strip()
        print(
            f"  [RESUME-SKIP] {variant} {out_dir.name} | "
            f"macro_f1={cached['macro_f1']:.4f} | f1_toxic={cached['f1_toxic']:.4f} | "
            f"victsd={cached['by_source']['victsd']['macro_f1']:.4f}",
            flush=True,
        )
        return {
            "variant": variant,
            "seed": seed,
            "run_id": cached_run_id,
            "dir": str(out_dir),
            "metrics": cached,
        }

    model_name = str(v_cfg["model_name"])
    use_pyvi = bool(v_cfg.get("use_pyvi", "phobert" in model_name.lower()))
    max_length = int(v_cfg.get("max_length", defaults.get("max_length", 160)))
    batch_size = int(v_cfg.get("batch_size", defaults.get("batch_size", 16)))
    eval_batch_size = int(v_cfg.get("eval_batch_size", defaults.get("eval_batch_size", 64)))
    grad_accum = int(v_cfg.get("grad_accum", defaults.get("grad_accum", 2)))
    epochs = int(v_cfg.get("epochs", defaults.get("epochs", 4)))
    lr = float(v_cfg.get("learning_rate", defaults.get("learning_rate", 2e-5)))
    head_lr_mult = float(v_cfg.get("head_lr_mult", defaults.get("head_lr_mult", 5.0)))
    weight_decay = float(v_cfg.get("weight_decay", defaults.get("weight_decay", 0.01)))
    warmup_ratio = float(v_cfg.get("warmup_ratio", defaults.get("warmup_ratio", 0.10)))
    dropout = float(v_cfg.get("dropout", defaults.get("dropout", 0.20)))
    pooling = str(v_cfg.get("pooling", defaults.get("pooling", "mean_cls")))
    label_smoothing = float(
        v_cfg.get("label_smoothing", defaults.get("label_smoothing", 0.03))
    )
    grad_clip = float(v_cfg.get("grad_clip", defaults.get("grad_clip", 1.0)))
    use_amp = bool(v_cfg.get("use_amp", defaults.get("use_amp", True)))
    grad_ckpt = bool(
        v_cfg.get("gradient_checkpointing", defaults.get("gradient_checkpointing", False))
    )

    seed_everything(seed, deterministic=False)
    np.random.seed(seed)
    device = _resolve_device(str(defaults.get("device", "auto")))
    amp_enabled = use_amp and device.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1)
    collate_fn = _make_collate_fn(pad_id)

    train_ds = _encode_dataframe(train_df, tokenizer, max_length, use_pyvi)
    val_ds = _encode_dataframe(val_df, tokenizer, max_length, use_pyvi)
    extra_ds = (
        _encode_dataframe(extra_eval_df, tokenizer, max_length, use_pyvi)
        if extra_eval_df is not None
        else None
    )

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    extra_loader = (
        DataLoader(
            extra_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )
        if extra_ds is not None
        else None
    )

    model = TransformerToxicClassifier(
        model_name=model_name,
        pooling=pooling,
        dropout=dropout,
        multi_sample_dropout=True,
        gradient_checkpointing=grad_ckpt,
        num_classes=2,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_groups = _build_optimizer_param_groups(model, lr, head_lr_mult, weight_decay)
    optimizer = torch.optim.AdamW(param_groups)
    steps_per_epoch = max(1, (len(train_loader) + grad_accum - 1) // grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    tracker = MLflowTracker(experiment_name=experiment_name, tracking_uri=tracking_uri)
    run_id = tracker.start_run(
        run_name=f"{variant}-{out_dir.name}",
        tags={
            "experiment": experiment_name,
            "variant": variant,
            "seed": str(seed),
            "owner": "TV4",
            "model_name": model_name,
            "pretrained": "true",
            "result_type": result_type,
        },
    )

    best_macro_f1 = -1.0
    best_epoch = 0
    best_state_cpu: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    try:
        tracker.log_params(
            {
                "variant": variant,
                "model_name": model_name,
                "use_pyvi": use_pyvi,
                "seed": seed,
                "max_length": max_length,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "epochs": epochs,
                "learning_rate": lr,
                "head_lr_mult": head_lr_mult,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "dropout": dropout,
                "pooling": pooling,
                "label_smoothing": label_smoothing,
                "num_params": num_params,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "train_sha256": data_hashes["train_parquet_sha256"][:16],
                "val_sha256": data_hashes["val_parquet_sha256"][:16],
            }
        )

        # Evaluate twice per epoch (mid-epoch + end-epoch) for sharp early stopping on 28k samples
        mid_step = max(1, len(train_loader) // 2)
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            tr_loss_sum = 0.0
            tr_count = 0
            for step_idx, batch in enumerate(train_loader, start=1):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = model(input_ids, attention_mask)
                    loss = F.cross_entropy(
                        logits, labels, label_smoothing=label_smoothing
                    )
                    scaled_loss = loss / grad_accum
                if not torch.isfinite(loss).item():
                    raise ValueError("Non-finite transformer training loss encountered")
                scaler.scale(scaled_loss).backward()
                tr_loss_sum += float(loss.detach().item()) * int(labels.shape[0])
                tr_count += int(labels.shape[0])

                if step_idx % grad_accum == 0 or step_idx == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                if epoch >= 2 and step_idx == mid_step:
                    mid_val = _eval_transformer_loader(
                        model, val_loader, device, amp_enabled=amp_enabled, label_smoothing=0.0
                    )
                    if mid_val["macro_f1"] > best_macro_f1:
                        best_macro_f1 = float(mid_val["macro_f1"])
                        best_epoch = epoch - 0.5
                        best_state_cpu = {
                            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                        }

            val_m = _eval_transformer_loader(
                model, val_loader, device, amp_enabled=amp_enabled, label_smoothing=0.0
            )
            tr_loss_avg = tr_loss_sum / max(1, tr_count)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(tr_loss_avg, 5),
                    "val_loss": round(val_m["loss"], 5),
                    "val_macro_f1": round(val_m["macro_f1"], 5),
                    "val_f1_toxic": round(val_m["f1_toxic"], 5),
                    "val_victsd_macro_f1": round(val_m["by_source"]["victsd"]["macro_f1"], 5),
                    "val_pr_auc": round(float(val_m.get("pr_auc") or 0.0), 5),
                }
            )
            tracker.log_metrics(
                {
                    "train_loss": tr_loss_avg,
                    "val_loss": val_m["loss"],
                    "val_macro_f1": val_m["macro_f1"],
                    "val_f1_toxic": val_m["f1_toxic"],
                    "val_victsd_macro_f1": val_m["by_source"]["victsd"]["macro_f1"],
                },
                step=epoch,
            )
            if val_m["macro_f1"] > best_macro_f1:
                best_macro_f1 = float(val_m["macro_f1"])
                best_epoch = epoch
                best_state_cpu = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            print(
                f"    [{variant} {out_dir.name} ep {epoch}/{epochs}] "
                f"tr_loss={tr_loss_avg:.4f} val_f1={val_m['macro_f1']:.4f} "
                f"tox={val_m['f1_toxic']:.4f} "
                f"victsd={val_m['by_source']['victsd']['macro_f1']:.4f} "
                f"best={best_macro_f1:.4f}",
                flush=True,
            )

        train_time_s = time.perf_counter() - started
        peak_vram_mb = (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            if device.type == "cuda"
            else 0.0
        )

        assert best_state_cpu is not None
        model.load_state_dict(best_state_cpu)
        best_path = out_dir / "best.pt"
        if save_weights:
            _save_atomic_pt(
                best_path,
                {
                    "format_version": 1,
                    "model_name": model_name,
                    "pooling": pooling,
                    "dropout": dropout,
                    "use_pyvi": use_pyvi,
                    "max_length": max_length,
                    "best_epoch": best_epoch,
                    "best_macro_f1": best_macro_f1,
                    "model_state_dict": best_state_cpu,
                },
            )

        t_inf = time.perf_counter()
        best_eval = _eval_transformer_loader(
            model,
            val_loader,
            device,
            amp_enabled=amp_enabled,
            label_smoothing=0.0,
            collect_outputs=True,
        )
        inference_ms = ((time.perf_counter() - t_inf) / max(1, len(val_df))) * 1000.0
        probs = best_eval.pop("probs")
        val_ids = best_eval.pop("ids")
        val_true = best_eval.pop("y_true")
        val_pred = best_eval.pop("y_pred")
        val_sources = best_eval.pop("sources")

        y_arr = np.asarray(val_true, dtype=int)
        p_pos = np.clip(probs[:, 1], 1e-7, 1.0 - 1e-7)
        raw_ece = compute_ece(y_arr, probs[:, 1], n_bins=10)
        brier = float(brier_score_loss(y_arr, probs[:, 1]))
        nll = float(log_loss(y_arr, p_pos, labels=[0, 1]))
        model_size_mb = (
            round(best_path.stat().st_size / (1024.0 * 1024.0), 4)
            if best_path.exists()
            else 0.0
        )
        metrics: dict[str, Any] = {
            **best_eval,
            "best_epoch": best_epoch,
            "raw_softmax_ece": round(raw_ece, 6),
            "brier_score": round(brier, 6),
            "log_loss": round(nll, 6),
            "is_calibrated": False,
            "train_time_s": round(train_time_s, 2),
            "inference_ms_per_sample": round(inference_ms, 4),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "num_params": int(num_params),
            "model_size_mb": model_size_mb,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "device": str(device),
        }

        def _write_preds(
            target_json: Path,
            target_csv: Path,
            rec_ids: list[str],
            rec_true: list[int],
            rec_pred: list[int],
            rec_src: list[str],
            rec_probs: np.ndarray,
        ) -> None:
            j_rows, c_rows = [], []
            for idx, (rid, yt, yp, src) in enumerate(
                zip(rec_ids, rec_true, rec_pred, rec_src)
            ):
                p0 = round(float(rec_probs[idx, 0]), 6)
                p1 = round(float(rec_probs[idx, 1]), 6)
                j_rows.append(
                    {
                        "id": rid,
                        "source": src,
                        "true_label": int(yt),
                        "predicted_label": int(yp),
                        "probabilities": {BINARY_LABEL_NAMES[0]: p0, BINARY_LABEL_NAMES[1]: p1},
                        "is_calibrated": False,
                        "threshold": None,
                        "run_id": run_id,
                    }
                )
                c_rows.append(
                    {
                        "sample_id": rid,
                        "source": src,
                        "y_true": int(yt),
                        "y_pred": int(yp),
                        "prob_CLEAN": p0,
                        "prob_TOXIC": p1,
                    }
                )
            target_json.write_text(
                json.dumps(j_rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            pd.DataFrame(c_rows).to_csv(target_csv, index=False, encoding="utf-8")

        _write_preds(
            out_dir / "val_predictions.json",
            out_dir / "predictions_val.csv",
            val_ids,
            val_true,
            val_pred,
            val_sources,
            probs,
        )

        if extra_loader is not None and extra_eval_name is not None:
            ex_eval = _eval_transformer_loader(
                model,
                extra_loader,
                device,
                amp_enabled=amp_enabled,
                label_smoothing=0.0,
                collect_outputs=True,
            )
            ex_probs = ex_eval.pop("probs")
            ex_ids = ex_eval.pop("ids")
            ex_true = ex_eval.pop("y_true")
            ex_pred = ex_eval.pop("y_pred")
            ex_src = ex_eval.pop("sources")
            _write_preds(
                out_dir / f"{extra_eval_name}_predictions.json",
                out_dir / f"predictions_{extra_eval_name}.csv",
                ex_ids,
                ex_true,
                ex_pred,
                ex_src,
                ex_probs,
            )
            metrics[f"{extra_eval_name}_macro_f1"] = round(float(ex_eval["macro_f1"]), 6)
            metrics[f"{extra_eval_name}_f1_toxic"] = round(float(ex_eval["f1_toxic"]), 6)
            metrics[f"{extra_eval_name}_victsd_macro_f1"] = round(
                float(ex_eval["by_source"]["victsd"]["macro_f1"]), 6
            )

        (out_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "error_summary.json").write_text(
            json.dumps(
                _build_aggregate_error_summary(val_df, val_pred),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (out_dir / "config.json").write_text(
            json.dumps(
                {
                    "experiment": experiment_name,
                    "variant": variant,
                    "seed": seed,
                    "run_id": run_id,
                    "pretrained": True,
                    "model_name": model_name,
                    "use_pyvi": use_pyvi,
                    "max_length": max_length,
                    "batch_size": batch_size,
                    "grad_accum": grad_accum,
                    "epochs": epochs,
                    "learning_rate": lr,
                    "head_lr_mult": head_lr_mult,
                    "weight_decay": weight_decay,
                    "warmup_ratio": warmup_ratio,
                    "dropout": dropout,
                    "pooling": pooling,
                    "label_smoothing": label_smoothing,
                    "data_hashes": data_hashes,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "run_id.txt").write_text(run_id, encoding="utf-8")

        tracker.log_metrics(
            {
                "best_val_macro_f1": metrics["macro_f1"],
                "best_val_f1_toxic": metrics["f1_toxic"],
                "best_val_pr_auc": float(metrics.get("pr_auc") or 0.0),
                "best_val_ece": metrics["raw_softmax_ece"],
                "val_victsd_macro_f1": metrics["by_source"]["victsd"]["macro_f1"],
                "val_vihsd_macro_f1": metrics["by_source"]["vihsd"]["macro_f1"],
                "peak_vram_mb": metrics["peak_vram_mb"],
                "train_time_s": metrics["train_time_s"],
            }
        )
        tracker.end_run("FINISHED")
    except BaseException:
        tracker.end_run("FAILED")
        raise
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        f"  -> [{variant} {out_dir.name}] val_f1={metrics['macro_f1']:.4f} | "
        f"f1_tox={metrics['f1_toxic']:.4f} | "
        f"victsd={metrics['by_source']['victsd']['macro_f1']:.4f} | "
        f"pr_auc={metrics.get('pr_auc', 0.0):.4f} | "
        f"ece={metrics['raw_softmax_ece']:.4f} | "
        f"vram={metrics['peak_vram_mb']:.0f}MB | time={metrics['train_time_s']:.1f}s",
        flush=True,
    )
    return {
        "variant": variant,
        "seed": seed,
        "run_id": run_id,
        "dir": str(out_dir),
        "metrics": metrics,
    }


def load_and_predict_transformer(
    ckpt_dir: Path,
    texts: list[str],
    batch_size: int = 64,
    device: str = "auto",
) -> tuple[list[int], np.ndarray]:
    """Reload a saved Transformer checkpoint and predict probabilities deterministically."""
    cfg = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    resolved = _resolve_device(device)
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=resolved, weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1)
    dummy_df = pd.DataFrame(
        {
            "id": [str(i) for i in range(len(texts))],
            "text": texts,
            "label": [0] * len(texts),
            "source": ["infer"] * len(texts),
        }
    )
    ds = _encode_dataframe(
        dummy_df, tokenizer, int(cfg["max_length"]), bool(cfg["use_pyvi"])
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=_make_collate_fn(pad_id)
    )
    model = TransformerToxicClassifier(
        model_name=cfg["model_name"],
        pooling=str(cfg["pooling"]),
        dropout=float(cfg["dropout"]),
        multi_sample_dropout=True,
        gradient_checkpointing=False,
        num_classes=2,
    ).to(resolved)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    probs_parts: list[np.ndarray] = []
    amp_enabled = resolved.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(
                    batch["input_ids"].to(resolved),
                    batch["attention_mask"].to(resolved),
                )
            probs_parts.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    del model
    if resolved.type == "cuda":
        torch.cuda.empty_cache()
    probs = np.vstack(probs_parts)
    return probs.argmax(axis=1).astype(int).tolist(), probs


def run_transformer_5fold_oof(
    variant: str,
    config_path: Path = DEFAULT_TRANSFORMER_CFG,
    out_root: Path = ROOT / "models" / "transformers_v3" / "oof_5fold",
    n_splits: int = 5,
    split_seed: int = 42,
    save_weights: bool = False,
) -> dict[str, Any]:
    """Run leak-free 5-fold stratified Train-OOF CV strictly inside train.parquet.

    Each fold early-stops ONLY on its inner oof_fold and predicts on val.parquet without
    ever seeing val.parquet labels during training or early stopping.
    """
    full_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    v_cfg = full_cfg["variants"][variant]
    defaults = full_cfg.get("defaults", {})
    train_df, val_df, data_hashes = load_binary_train_val(
        dataset_dir=ROOT / full_cfg["dataset_dir"], use_augmentation=False
    )
    strat_key = [
        f"{s}_{l}"
        for s, l in zip(train_df["source"].astype(str), train_df["label"].astype(int))
    ]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    variant_oof_dir = out_root / variant
    variant_oof_dir.mkdir(parents=True, exist_ok=True)

    oof_probs = np.zeros(len(train_df), dtype=np.float64)
    val_fold_probs: list[np.ndarray] = []

    for fold_idx, (tr_idx, oof_idx) in enumerate(skf.split(np.zeros(len(train_df)), strat_key)):
        fold_dir = variant_oof_dir / f"fold_{fold_idx}"
        tr_sub = train_df.iloc[tr_idx].reset_index(drop=True)
        oof_sub = train_df.iloc[oof_idx].reset_index(drop=True)
        run_transformer_single_split(
            variant=f"{variant}-F{fold_idx}",
            seed=split_seed + fold_idx,
            v_cfg=v_cfg,
            defaults=defaults,
            train_df=tr_sub,
            val_df=oof_sub,
            data_hashes=data_hashes,
            out_dir=fold_dir,
            experiment_name=str(full_cfg.get("experiment", "TV4-Transformer-V3")),
            extra_eval_df=val_df,
            extra_eval_name="holdout_val",
            save_weights=save_weights,
        )
        oof_j = json.loads((fold_dir / "val_predictions.json").read_text(encoding="utf-8"))
        oof_probs[oof_idx] = [r["probabilities"]["TOXIC"] for r in oof_j]
        hv_j = json.loads(
            (fold_dir / "holdout_val_predictions.json").read_text(encoding="utf-8")
        )
        val_fold_probs.append(
            np.array([r["probabilities"]["TOXIC"] for r in hv_j], dtype=np.float64)
        )

    y_train = train_df["label"].astype(int).to_numpy()
    src_train = train_df["source"].astype(str).tolist()
    oof_metrics = compute_binary_metrics(
        y_train.tolist(), (oof_probs >= 0.5).astype(int).tolist(), src_train, oof_probs
    )
    oof_metrics["ece"] = float(compute_ece(y_train, oof_probs, n_bins=10))

    val_avg_prob = np.mean(val_fold_probs, axis=0)
    y_val = val_df["label"].astype(int).to_numpy()
    src_val = val_df["source"].astype(str).tolist()
    val_metrics = compute_binary_metrics(
        y_val.tolist(), (val_avg_prob >= 0.5).astype(int).tolist(), src_val, val_avg_prob
    )
    val_metrics["ece"] = float(compute_ece(y_val, val_avg_prob, n_bins=10))

    np.save(variant_oof_dir / "train_oof_prob_toxic.npy", oof_probs)
    np.save(variant_oof_dir / "val_5fold_avg_prob_toxic.npy", val_avg_prob)
    summary = {
        "variant": variant,
        "n_splits": n_splits,
        "split_seed": split_seed,
        "oof_metrics_at_050": oof_metrics,
        "val_5fold_avg_metrics_at_050": val_metrics,
    }
    (variant_oof_dir / "oof_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    oof_vic = oof_metrics["by_source"]["victsd"]["macro_f1"]
    val_vic = val_metrics["by_source"]["victsd"]["macro_f1"]
    print(
        f"[5FOLD-OOF {variant}] Train-OOF Macro-F1={oof_metrics['macro_f1']:.4f} "
        f"(TOX={oof_metrics['f1_toxic']:.4f}, ViCTSD={oof_vic:.4f}) | "
        f"Independent Val-5Fold Macro-F1={val_metrics['macro_f1']:.4f} "
        f"(TOX={val_metrics['f1_toxic']:.4f}, ViCTSD={val_vic:.4f})",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="TV4 V3 Pretrained Transformer Benchmark")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRANSFORMER_CFG)
    parser.add_argument("--variant", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--mode", choices=["seeds", "oof5"], default="seeds")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models" / "transformers_v3")
    parser.add_argument("--save-oof-weights", action="store_true")
    args = parser.parse_args()

    full_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    defaults = full_cfg.get("defaults", {})
    train_df, val_df, data_hashes = load_binary_train_val(
        dataset_dir=ROOT / full_cfg["dataset_dir"], use_augmentation=False
    )

    for variant in args.variant:
        if args.mode == "oof5":
            run_transformer_5fold_oof(
                variant=variant,
                config_path=args.config,
                out_root=args.out_dir / "oof_5fold",
                save_weights=args.save_oof_weights,
            )
            continue
        v_cfg = full_cfg["variants"][variant]
        seeds = args.seeds or [int(s) for s in full_cfg.get("seeds", [42, 43, 44])]
        runs = []
        for s in seeds:
            runs.append(
                run_transformer_single_split(
                    variant=variant,
                    seed=s,
                    v_cfg=v_cfg,
                    defaults=defaults,
                    train_df=train_df,
                    val_df=val_df,
                    data_hashes=data_hashes,
                    out_dir=args.out_dir / variant / f"seed_{s}",
                    experiment_name=str(full_cfg.get("experiment", "TV4-Transformer-V3")),
                    save_weights=True,
                )
            )
        f1s = [r["metrics"]["macro_f1"] for r in runs]
        toxs = [r["metrics"]["f1_toxic"] for r in runs]
        vics = [r["metrics"]["by_source"]["victsd"]["macro_f1"] for r in runs]
        print(
            f"[{variant} ({len(seeds)} seeds)] Macro-F1={np.mean(f1s):.4f}±{np.std(f1s):.4f} | "
            f"F1-TOXIC={np.mean(toxs):.4f}±{np.std(toxs):.4f} | "
            f"ViCTSD={np.mean(vics):.4f}±{np.std(vics):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
