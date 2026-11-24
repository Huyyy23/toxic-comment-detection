from __future__ import annotations

import gc
import hashlib
import json
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import torch
from pyvi import ViTokenizer
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from preprocessing import SequenceVocab, preprocess, preprocess_v2

ModelId = Literal["n6_lr_d", "v3_tri_gated_ft", "v3_phobert_v2"]

MODEL_INFO: dict[ModelId, dict[str, object]] = {
    "n6_lr_d": {
        "model_name": "N6-LR-D · seed 42",
        "model_backend": "sklearn Logistic Regression",
        "artifact_version": "n6-lr-d-seed42-c5-raw",
        "validation_macro_f1": 0.7900954579623571,
        "validation_f1_toxic": 0.6419336706014614,
        "is_calibrated": False,
        "calibration_status": "Raw predict_proba; chưa calibration",
        "threshold": 0.5,
        "provenance": "N6 official, C=5, validation tie-break seed 42",
        "sha256": "2b693b43eb74ef062ab9e0b278032abd1cfd1a1f101754a1dd379a32729b81be",
    },
    "v3_tri_gated_ft": {
        "model_name": "V3-TRI-GATED-FT · seed 43",
        "model_backend": "PyTorch tri-branch CNN + BiGRU",
        "artifact_version": "v3-tri-gated-ft-retrained-2026-11-24-seed43",
        "validation_macro_f1": 0.79304572483586,
        "validation_f1_toxic": 0.6467268623024831,
        "is_calibrated": False,
        "calibration_status": "Raw softmax; chưa calibration",
        "threshold": 0.5,
        "provenance": "V3 missing-weight replacement; validation-only retrained checkpoint",
        "sha256": "97cf7e7393b613a76310d0fc69a2e6f38091d08cb54a99de4e07357cab3376d8",
    },
    "v3_phobert_v2": {
        "model_name": "V3-PHOBERT-V2 · seed 44",
        "model_backend": "PyTorch PhoBERT fine-tuned",
        "artifact_version": "v3-phobert-v2-retrained-2026-11-24-seed44",
        "validation_macro_f1": 0.8194869678484376,
        "validation_f1_toxic": 0.6943396226415094,
        "is_calibrated": False,
        "calibration_status": "Raw softmax; chưa calibration",
        "threshold": 0.5,
        "provenance": "V3 missing-weight replacement; validation-only retrained checkpoint",
        "sha256": "fd8db1d1c030abfae5e00ef0b0b825731dca5a0c99e9d3e504d4670ce6bf3b59",
    },
}

LABELS = {0: "CLEAN", 1: "TOXIC"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_pad_id(pad_id: int, vocab_size: int) -> int:
    if isinstance(pad_id, bool) or not isinstance(pad_id, int) or not 0 <= pad_id < vocab_size:
        raise ValueError("pad_id must be inside the vocabulary")
    return pad_id


def _validate_dropout(dropout: float) -> float:
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise TypeError("dropout must be numeric")
    value = float(dropout)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("dropout must be between 0 and 1")
    return value


def _validate_kernel_sizes(values: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not values:
        raise ValueError("kernel_sizes must be nonempty")
    result = tuple(_validate_positive_int(int(value), "kernel_size") for value in values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("kernel_sizes must contain integers")
    return result


def _sequence_mask(input_ids: Tensor, lengths: Tensor, vocab_size: int) -> Tensor:
    if input_ids.dtype != torch.long or lengths.dtype != torch.long:
        raise TypeError("sequence IDs and lengths must be torch.long")
    if input_ids.ndim != 2 or lengths.ndim != 1:
        raise ValueError("sequence IDs must be [batch,time] and lengths [batch]")
    batch_size, time_steps = input_ids.shape
    if batch_size == 0 or time_steps == 0 or lengths.shape[0] != batch_size:
        raise ValueError("sequence dimensions do not match")
    cpu_lengths = lengths.detach().to(device="cpu")
    if torch.any(cpu_lengths <= 0) or torch.any(cpu_lengths > time_steps):
        raise ValueError("sequence length is outside the padded sequence")
    if torch.any(input_ids < 0) or torch.any(input_ids >= vocab_size):
        raise ValueError("sequence ID is outside the vocabulary")
    positions = torch.arange(time_steps, device=input_ids.device).unsqueeze(0)
    return positions < lengths.to(input_ids.device).unsqueeze(1)


class Conv1DPoolEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, kernel_sizes: list[int], num_filters: int, pad_id: int, spatial_dropout: float) -> None:
        super().__init__()
        _validate_positive_int(vocab_size, "vocab_size")
        _validate_positive_int(embedding_dim, "embedding_dim")
        _validate_positive_int(num_filters, "num_filters")
        _validate_pad_id(pad_id, vocab_size)
        kernel_sizes = _validate_kernel_sizes(kernel_sizes)
        self.vocab_size = vocab_size
        self.spatial_dropout = _validate_dropout(spatial_dropout)
        self.out_features = num_filters * len(kernel_sizes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, kernel) for kernel in kernel_sizes])
    def forward(self, input_ids: Tensor, lengths: Tensor) -> Tensor:
        if input_ids.device != self.embedding.weight.device:
            raise ValueError("input IDs and model must be on the same device")
        mask = _sequence_mask(input_ids, lengths, self.vocab_size)
        embedded = self.embedding(input_ids) * mask.unsqueeze(-1).to(dtype=self.embedding.weight.dtype)
        channel_first = embedded.transpose(1, 2)
        if self.spatial_dropout > 0.0 and self.training:
            channel_first = F.dropout1d(channel_first, p=self.spatial_dropout, training=True)
        pooled = []
        valid_mask = mask.unsqueeze(1)
        for conv in self.convs:
            kernel = conv.kernel_size[0]
            left = (kernel - 1) // 2
            padded = F.pad(channel_first, (left, kernel - 1 - left))
            activated = F.relu(conv(padded)).masked_fill(~valid_mask, float("-inf"))
            pooled.append(activated.max(dim=2).values)
        return torch.cat(pooled, dim=1)


class MaskedSelfAttentionPool1D(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        _validate_positive_int(input_dim, "input_dim")
        self.proj = nn.Linear(input_dim, input_dim)
        self.score = nn.Linear(input_dim, 1, bias=False)

    def forward(self, states: Tensor, mask: Tensor) -> Tensor:
        energy = self.score(torch.tanh(self.proj(states))).squeeze(-1)
        weights = torch.softmax(energy.masked_fill(~mask, float("-inf")), dim=1).unsqueeze(-1)
        return torch.sum(states * weights, dim=1)


class TriHybridWordCNNBiGRUCharClassifier(nn.Module):
    def __init__(self, word_vocab_size: int, char_vocab_size: int, config: dict[str, object]) -> None:
        super().__init__()
        model_cfg = config["model"]
        self.architecture = "word_cnn_bigru_char_cnn"
        self.fusion_type = str(model_cfg["fusion_type"])
        self.num_classes = 2
        self.word_vocab_size = word_vocab_size
        self.word_cnn = Conv1DPoolEncoder(
            word_vocab_size, int(model_cfg["embedding_dim"]), list(model_cfg["kernel_sizes"]),
            int(model_cfg["word_num_filters"]), 0, float(model_cfg["spatial_dropout"]),
        )
        self.word_embedding = self.word_cnn.embedding
        self.word_rnn = nn.GRU(int(model_cfg["embedding_dim"]), int(model_cfg["hidden_size"]), batch_first=True, bidirectional=True)
        self.rnn_attn_pool = MaskedSelfAttentionPool1D(int(model_cfg["hidden_size"]) * 2)
        self.char_cnn = Conv1DPoolEncoder(
            char_vocab_size, int(model_cfg["char_embedding_dim"]), list(model_cfg["char_kernel_sizes"]),
            int(model_cfg["char_num_filters"]), 0, float(model_cfg["spatial_dropout"]),
        )
        word_cnn_dim = self.word_cnn.out_features
        word_rnn_dim = int(model_cfg["hidden_size"]) * 4
        char_cnn_dim = self.char_cnn.out_features
        fusion_dim = int(model_cfg["fusion_dim"])
        self.emb_dropout = nn.Dropout(float(model_cfg["dropout"]))
        self.dropout = nn.Dropout(float(model_cfg["dropout"]))
        if self.fusion_type != "gated":
            raise ValueError("The selected V3 model requires gated fusion")
        self.proj_wcnn = nn.Linear(word_cnn_dim, fusion_dim)
        self.proj_wrnn = nn.Linear(word_rnn_dim, fusion_dim)
        self.proj_char = nn.Linear(char_cnn_dim, fusion_dim)
        self.gate_net = nn.Linear(fusion_dim * 3, 3)
        self.classifier = nn.Linear(fusion_dim * 3, 2)

    def forward(self, word_ids: Tensor, word_lengths: Tensor, char_ids: Tensor, char_lengths: Tensor) -> Tensor:
        if word_ids.shape[0] != char_ids.shape[0]:
            raise ValueError("word and char batch sizes must match")
        if word_ids.device != self.word_embedding.weight.device:
            raise ValueError("word IDs and model must be on the same device")
        wcnn = self.word_cnn(word_ids, word_lengths)
        mask = _sequence_mask(word_ids, word_lengths, self.word_vocab_size)
        embedded = self.emb_dropout(self.word_embedding(word_ids))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, word_lengths.detach().to(device="cpu"), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.word_rnn(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=word_ids.shape[1])
        attention = self.rnn_attn_pool(unpacked, mask)
        max_pool = unpacked.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        wrnn = torch.cat((attention, max_pool), dim=1)
        char = self.char_cnn(char_ids, char_lengths)
        z1 = F.gelu(self.proj_wcnn(wcnn))
        z2 = F.gelu(self.proj_wrnn(wrnn))
        z3 = F.gelu(self.proj_char(char))
        gates = torch.sigmoid(self.gate_net(torch.cat((z1, z2, z3), dim=1)))
        fused = torch.cat((z1 * gates[:, 0:1], z2 * gates[:, 1:2], z3 * gates[:, 2:3]), dim=1)
        return self.classifier(self.dropout(fused))


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_ZW_RE = re.compile(r"[\u200b-\u200f\ufeff]")
_WS_RE = re.compile(r"\s+")
_TONE_FIXES = (
    ("òa", "oà"), ("óa", "oá"), ("ỏa", "oả"), ("õa", "oã"), ("ọa", "oạ"),
    ("òe", "oè"), ("óe", "oé"), ("ỏe", "oẻ"), ("õe", "oẽ"), ("ọe", "oẹ"),
    ("ùy", "uỳ"), ("úy", "uý"), ("ủy", "uỷ"), ("ũy", "uỹ"), ("ụy", "uỵ"),
)


def clean_transformer_text(text: str, use_pyvi: bool) -> str:
    value = unicodedata.normalize("NFC", str(text))
    value = _ZW_RE.sub("", value)
    value = _URL_RE.sub(" ", value)
    value = _HTML_RE.sub(" ", value)
    for old, new in _TONE_FIXES:
        value = value.replace(old, new).replace(old.upper(), new.upper())
    value = _WS_RE.sub(" ", value).strip() or " "
    return ViTokenizer.tokenize(value) if use_pyvi else value


class TransformerToxicClassifier(nn.Module):
    def __init__(self, config: object, pooling: str, dropout: float) -> None:
        super().__init__()
        self.pooling = pooling
        self.backbone = AutoModel.from_config(config)
        hidden = int(self.backbone.config.hidden_size)
        output_dim = hidden * 2 if pooling == "mean_cls" else hidden
        self.norm = nn.LayerNorm(output_dim)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(5)])
        self.classifier = nn.Linear(output_dim, 2)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        sequence = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = sequence[:, 0, :]
        if self.pooling == "cls":
            features = cls
        else:
            mask = attention_mask.unsqueeze(-1).to(dtype=sequence.dtype)
            mean = (sequence * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            features = torch.cat((cls, mean), dim=1) if self.pooling == "mean_cls" else mean
        features = self.norm(features)
        return torch.stack([self.classifier(dropout(features)) for dropout in self.dropouts], dim=0).mean(dim=0)


@dataclass
class ModelRuntime:
    model: nn.Module
    config: dict[str, object]
    word_vocab: SequenceVocab | None = None
    char_vocab: SequenceVocab | None = None
    tokenizer: object | None = None


class InferenceService:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._neural_lock = threading.Lock()
        self.baseline_path = models_dir / "n6_lr_d.joblib"
        self.cnn_path = models_dir / "tri_best.pt"
        self.cnn_config = json.loads((models_dir / "tri_config.json").read_text(encoding="utf-8"))
        self.cnn_vocab_path = models_dir / "tri_vocab.json"
        self.pho_path = models_dir / "phobert_best.pt"
        self.pho_config = json.loads((models_dir / "phobert_config.json").read_text(encoding="utf-8"))
        self._verify_artifact("n6_lr_d", self.baseline_path)
        self._verify_artifact("v3_tri_gated_ft", self.cnn_path)
        self._verify_artifact("v3_phobert_v2", self.pho_path)
        self.baseline = joblib.load(self.baseline_path)
        classes = {int(label) for label in self.baseline.classes_}
        if classes != {0, 1} or not hasattr(self.baseline, "predict_proba"):
            raise ValueError("N6 artifact must expose CLEAN/TOXIC predict_proba classes")
        self.cnn = self._load_cnn()
        self.phobert = self._load_phobert()

    @staticmethod
    def _verify_artifact(model_id: ModelId, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {model_id} artifact: {path}")
        if _sha256(path) != MODEL_INFO[model_id]["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {model_id} artifact")

    def _load_cnn(self) -> ModelRuntime:
        vocab_bytes = self.cnn_vocab_path.read_bytes()
        vocab_payload = json.loads(vocab_bytes)
        checkpoint = torch.load(self.cnn_path, map_location="cpu", weights_only=True)
        if checkpoint.get("vocab_sha256") != hashlib.sha256(vocab_bytes).hexdigest():
            raise ValueError("CNN vocabulary hash disagrees with checkpoint")
        word_vocab = SequenceVocab(vocab_payload["word_vocab"])
        char_vocab = SequenceVocab(vocab_payload["char_vocab"])
        if word_vocab.mode != "word" or char_vocab.mode != "char":
            raise ValueError("CNN vocab modes disagree with checkpoint")
        model = TriHybridWordCNNBiGRUCharClassifier(word_vocab.size, char_vocab.size, self.cnn_config)
        model.load_state_dict(checkpoint["model_state_dict"], assign=True)
        model.eval()
        del checkpoint
        return ModelRuntime(model, self.cnn_config, word_vocab, char_vocab)

    def _load_phobert(self) -> ModelRuntime:
        cfg = self.pho_config
        revision = str(cfg["model_revision"])
        name = str(cfg["model_name"])
        config = AutoConfig.from_pretrained(name, revision=revision)
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        checkpoint = torch.load(self.pho_path, map_location="cpu", weights_only=True)
        model = TransformerToxicClassifier(config, str(cfg["pooling"]), float(cfg["dropout"]))
        model.load_state_dict(checkpoint["model_state_dict"], assign=True)
        model.eval()
        del checkpoint
        gc.collect()
        return ModelRuntime(model, cfg, tokenizer=tokenizer)

    def model_info(self, model_id: ModelId) -> dict[str, object]:
        return {
            **MODEL_INFO[model_id],
            "model_id": model_id,
            "label_taxonomy": LABELS,
            "calibration_method": None,
            "status": "loaded",
        }

    def predict_baseline(self, text: str) -> dict[str, object]:
        started = time.perf_counter()
        probabilities = self.baseline.predict_proba([preprocess(text, "teencode")])[0]
        by_class = {int(label): float(probability) for label, probability in zip(self.baseline.classes_, probabilities)}
        toxic = by_class[1]
        label_id = int(toxic >= 0.5)
        return self._result("n6_lr_d", label_id, by_class[0], toxic, (time.perf_counter() - started) * 1000)

    def predict_neural(self, model_ids: list[ModelId], text: str) -> list[dict[str, object]]:
        if any(model_id not in ("v3_tri_gated_ft", "v3_phobert_v2") for model_id in model_ids):
            raise ValueError("GPU inference accepts only neural model IDs")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = []
        with self._neural_lock:
            for model_id in model_ids:
                started = time.perf_counter()
                runtime = self.cnn if model_id == "v3_tri_gated_ft" else self.phobert
                runtime.model.to(device)
                try:
                    with torch.inference_mode():
                        if model_id == "v3_tri_gated_ft":
                            probabilities = self._predict_cnn(runtime, text, device)
                        else:
                            probabilities = self._predict_phobert(runtime, text, device)
                    label_id = int(probabilities[1] > probabilities[0])
                    results.append(self._result(model_id, label_id, float(probabilities[0]), float(probabilities[1]), (time.perf_counter() - started) * 1000))
                finally:
                    runtime.model.to("cpu")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
        return results

    @staticmethod
    def _predict_cnn(runtime: ModelRuntime, text: str, device: torch.device) -> np.ndarray:
        cfg = runtime.config
        word_variant = str(cfg.get("word_preprocessing", cfg["preprocessing"]))
        char_variant = str(cfg.get("char_preprocessing", cfg["preprocessing"]))
        word_text = preprocess_v2([text], word_variant)[0]
        char_text = word_text if char_variant == word_variant else preprocess_v2([text], char_variant)[0]
        word_ids = runtime.word_vocab.encode(word_text, int(cfg["max_word_length"]))
        char_ids = runtime.char_vocab.encode(char_text, int(cfg["max_char_length"]))
        word_tensor = torch.tensor([word_ids], dtype=torch.long, device=device)
        char_tensor = torch.tensor([char_ids], dtype=torch.long, device=device)
        logits = runtime.model(
            word_tensor,
            torch.tensor([len(word_ids)], dtype=torch.long, device=device),
            char_tensor,
            torch.tensor([len(char_ids)], dtype=torch.long, device=device),
        )
        return torch.softmax(logits, dim=1)[0].float().cpu().numpy()

    @staticmethod
    def _predict_phobert(runtime: ModelRuntime, text: str, device: torch.device) -> np.ndarray:
        cfg = runtime.config
        cleaned = clean_transformer_text(text, bool(cfg["use_pyvi"]))
        encoded = runtime.tokenizer(
            cleaned,
            truncation=True,
            max_length=int(cfg["max_length"]),
            padding=False,
            return_tensors="pt",
        )
        inputs = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = runtime.model(inputs, mask)
        return torch.softmax(logits.float(), dim=1)[0].cpu().numpy()

    @staticmethod
    def _result(model_id: ModelId, label_id: int, clean: float, toxic: float, latency_ms: float) -> dict[str, object]:
        info = MODEL_INFO[model_id]
        return {
            "model_id": model_id,
            "model_name": info["model_name"],
            "label_id": label_id,
            "label_name": LABELS[label_id],
            "confidence": toxic if label_id else clean,
            "probabilities": {"CLEAN": clean, "TOXIC": toxic},
            "is_calibrated": False,
            "calibration_status": info["calibration_status"],
            "threshold": 0.5,
            "latency_ms": max(0.0, latency_ms),
            "model_backend": info["model_backend"],
            "validation_macro_f1": info["validation_macro_f1"],
            "validation_f1_toxic": info["validation_f1_toxic"],
        }

    def predict(self, model_id: ModelId, text: str) -> dict[str, object]:
        if model_id == "n6_lr_d":
            return self.predict_baseline(text)
        return self.predict_neural([model_id], text)[0]

    def compare(self, text: str) -> list[dict[str, object]]:
        baseline = self.predict_baseline(text)
        neural = self.predict_neural(["v3_tri_gated_ft", "v3_phobert_v2"], text)
        return [baseline, *neural]
