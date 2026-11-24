"""Random-initialized TextCNN, Char-CNN, and hybrid CNN classifiers."""

from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_pad_id(pad_id: int, vocab_size: int, name: str = "pad_id") -> int:
    if isinstance(pad_id, bool) or not isinstance(pad_id, int) or not 0 <= pad_id < vocab_size:
        raise ValueError(f"{name} must be an integer inside [0, {vocab_size})")
    return pad_id


def _validate_dropout(dropout: float) -> float:
    if isinstance(dropout, bool) or not isinstance(dropout, Real):
        raise TypeError("dropout must be a real number")
    value = float(dropout)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("dropout must be finite and between 0 and 1")
    return value


def _validate_kernel_sizes(kernel_sizes: Sequence[int]) -> tuple[int, ...]:
    if isinstance(kernel_sizes, (str, bytes)) or not kernel_sizes:
        raise ValueError("kernel_sizes must be a nonempty sequence of positive integers")
    validated = tuple(_validate_positive_int(int(k), "kernel_size") for k in kernel_sizes)
    for raw in kernel_sizes:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError("kernel_sizes must contain integers only")
    return validated


def _validate_seq_inputs(
    input_ids: Tensor,
    lengths: Tensor,
    vocab_size: int,
    name: str = "input_ids",
) -> tuple[Tensor, Tensor]:
    if not isinstance(input_ids, Tensor) or not isinstance(lengths, Tensor):
        raise TypeError(f"{name} and lengths must be tensors")
    if input_ids.dtype != torch.long or lengths.dtype != torch.long:
        raise TypeError(f"{name} and lengths must have torch.long dtype")
    if input_ids.ndim != 2 or lengths.ndim != 1:
        raise ValueError(f"{name} must be [batch, time] and lengths [batch]")
    batch_size, time_steps = input_ids.shape
    if batch_size == 0 or time_steps == 0:
        raise ValueError(f"{name} must have nonempty batch and time dimensions")
    if lengths.shape[0] != batch_size:
        raise ValueError("lengths must contain one entry per example")
    cpu_lengths = lengths.detach().to(device="cpu")
    if torch.any(cpu_lengths <= 0) or torch.any(cpu_lengths > time_steps):
        raise ValueError("lengths must be between 1 and the padded sequence length")
    if torch.any(input_ids < 0) or torch.any(input_ids >= vocab_size):
        raise ValueError(f"{name} must be inside the vocabulary")
    positions = torch.arange(time_steps, device=input_ids.device).unsqueeze(0)
    mask = positions < lengths.to(input_ids.device).unsqueeze(1)
    return mask, lengths


class Conv1DPoolEncoder(nn.Module):
    """Multi-kernel 1D Conv + ReLU + masked global max-pooling encoder."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        kernel_sizes: Sequence[int] = (3, 4, 5),
        num_filters: int = 128,
        pad_id: int = 0,
        spatial_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_positive_int(vocab_size, "vocab_size")
        _validate_positive_int(embedding_dim, "embedding_dim")
        _validate_positive_int(num_filters, "num_filters")
        _validate_pad_id(pad_id, vocab_size)
        kernels = _validate_kernel_sizes(kernel_sizes)
        sp_drop = _validate_dropout(spatial_dropout)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernels
        self.num_filters = num_filters
        self.pad_id = pad_id
        self.spatial_dropout = sp_drop
        self.out_features = num_filters * len(kernels)

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, k) for k in kernels]
        )

    def forward(self, input_ids: Tensor, lengths: Tensor) -> Tensor:
        """Encode [batch, time] sequences ignoring right padding."""
        if input_ids.device != self.embedding.weight.device:
            raise ValueError("input_ids and model must be on the same device")
        mask, _ = _validate_seq_inputs(input_ids, lengths, self.vocab_size)
        embedded = self.embedding(input_ids)
        embedded = embedded * mask.unsqueeze(-1).to(dtype=embedded.dtype)
        channel_first = embedded.transpose(1, 2)  # [batch, emb_dim, time]
        if self.spatial_dropout > 0.0 and self.training:
            channel_first = F.dropout1d(
                channel_first, p=self.spatial_dropout, training=True
            )

        pooled_outputs: list[Tensor] = []
        valid_mask = mask.unsqueeze(1)  # [batch, 1, time]
        for conv in self.convs:
            k = conv.kernel_size[0]
            pad_left = (k - 1) // 2
            pad_right = (k - 1) - pad_left
            padded = F.pad(channel_first, (pad_left, pad_right))
            activated = F.relu(conv(padded))
            activated = activated.masked_fill(~valid_mask, float("-inf"))
            pooled_outputs.append(activated.max(dim=2).values)
        return torch.cat(pooled_outputs, dim=1)


class MaskedSelfAttentionPool1D(nn.Module):
    """Additive self-attention pooling over variable-length sequence states."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        _validate_positive_int(input_dim, "input_dim")
        self.proj = nn.Linear(input_dim, input_dim)
        self.score = nn.Linear(input_dim, 1, bias=False)

    def forward(self, states: Tensor, mask: Tensor) -> Tensor:
        """Pool [batch, time, dim] states using boolean [batch, time] valid mask."""
        energy = self.score(torch.tanh(self.proj(states))).squeeze(-1)
        energy = energy.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(energy, dim=1).unsqueeze(-1)
        return torch.sum(states * weights, dim=1)


class MaskedGatedAttentionPool1D(nn.Module):
    """Gated attention pooling (Ilse et al. 2018): w^T (tanh(Vh) * sigmoid(Uh))."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        _validate_positive_int(input_dim, "input_dim")
        self.proj_v = nn.Linear(input_dim, input_dim)
        self.proj_u = nn.Linear(input_dim, input_dim)
        self.score = nn.Linear(input_dim, 1, bias=False)

    def forward(self, states: Tensor, mask: Tensor) -> Tensor:
        """Pool [batch, time, dim] states with gated context suppression."""
        gated = torch.tanh(self.proj_v(states)) * torch.sigmoid(self.proj_u(states))
        energy = self.score(gated).squeeze(-1)
        energy = energy.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(energy, dim=1).unsqueeze(-1)
        return torch.sum(states * weights, dim=1)


class TextCNNClassifier(nn.Module):
    """Word-level TextCNN classifier (single or multi-kernel, random-init)."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        kernel_sizes: Sequence[int] = (3, 4, 5),
        num_filters: int = 128,
        dropout: float = 0.5,
        pad_id: int = 0,
        num_classes: int = 2,
        spatial_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_positive_int(num_classes, "num_classes")
        drop = _validate_dropout(dropout)
        self.architecture = "textcnn"
        self.num_classes = num_classes
        self.encoder = Conv1DPoolEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            kernel_sizes=kernel_sizes,
            num_filters=num_filters,
            pad_id=pad_id,
            spatial_dropout=spatial_dropout,
        )
        self.embedding = self.encoder.embedding
        self.kernel_sizes = self.encoder.kernel_sizes
        self.dropout = nn.Dropout(drop)
        self.classifier = nn.Linear(self.encoder.out_features, num_classes)

    def forward(self, input_ids: Tensor, lengths: Tensor) -> Tensor:
        features = self.encoder(input_ids, lengths)
        return self.classifier(self.dropout(features))


class CharCNNClassifier(nn.Module):
    """Character-level multi-kernel 1D CNN classifier (random-init)."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        kernel_sizes: Sequence[int] = (3, 4, 5, 7),
        num_filters: int = 128,
        dropout: float = 0.5,
        pad_id: int = 0,
        num_classes: int = 2,
        spatial_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_positive_int(num_classes, "num_classes")
        drop = _validate_dropout(dropout)
        self.architecture = "char_cnn"
        self.num_classes = num_classes
        self.encoder = Conv1DPoolEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            kernel_sizes=kernel_sizes,
            num_filters=num_filters,
            pad_id=pad_id,
            spatial_dropout=spatial_dropout,
        )
        self.embedding = self.encoder.embedding
        self.kernel_sizes = self.encoder.kernel_sizes
        self.dropout = nn.Dropout(drop)
        self.classifier = nn.Linear(self.encoder.out_features, num_classes)

    def forward(self, char_ids: Tensor, char_lengths: Tensor) -> Tensor:
        features = self.encoder(char_ids, char_lengths)
        return self.classifier(self.dropout(features))


class HybridWordCharClassifier(nn.Module):
    """Hybrid classifier combining a word branch (TextCNN/BiGRU/BiLSTM) and Char-CNN."""

    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int,
        word_embedding_dim: int = 128,
        char_embedding_dim: int = 64,
        word_kernel_sizes: Sequence[int] = (3, 4, 5),
        char_kernel_sizes: Sequence[int] = (3, 4, 5, 7),
        num_filters: int = 128,
        hidden_size: int = 128,
        word_encoder_type: str = "cnn",
        dropout: float = 0.5,
        word_pad_id: int = 0,
        char_pad_id: int = 0,
        num_classes: int = 2,
        rnn_pooling: str = "last",
        spatial_dropout: float = 0.0,
        word_dropout: float | None = None,
        char_dropout: float | None = None,
        num_rnn_layers: int = 1,
        use_layernorm: bool = False,
        multi_sample_dropout: bool = False,
    ) -> None:
        super().__init__()
        _validate_positive_int(num_classes, "num_classes")
        _validate_positive_int(word_vocab_size, "word_vocab_size")
        _validate_positive_int(word_embedding_dim, "word_embedding_dim")
        _validate_pad_id(word_pad_id, word_vocab_size, "word_pad_id")
        if word_encoder_type not in ("cnn", "bilstm", "bigru"):
            raise ValueError("word_encoder_type must be cnn, bilstm, or bigru")
        valid_poolings = (
            "last",
            "attn",
            "attn_max",
            "gated_attn",
            "gated_attn_max",
            "gated_attn_max_mean",
        )
        if rnn_pooling not in valid_poolings:
            raise ValueError(f"rnn_pooling must be one of {valid_poolings}")
        drop = _validate_dropout(dropout)
        w_drop = _validate_dropout(drop if word_dropout is None else word_dropout)
        c_drop = _validate_dropout(0.0 if char_dropout is None else char_dropout)

        self.word_encoder_type = word_encoder_type
        self.rnn_pooling = rnn_pooling
        self.use_layernorm = bool(use_layernorm)
        self.multi_sample_dropout = bool(multi_sample_dropout)
        self.architecture = (
            "word_char_cnn" if word_encoder_type == "cnn" else f"{word_encoder_type}_char_cnn"
        )
        self.num_classes = num_classes
        self.word_vocab_size = word_vocab_size

        self.attn_pool: nn.Module | None = None
        if word_encoder_type == "cnn":
            self.word_cnn = Conv1DPoolEncoder(
                vocab_size=word_vocab_size,
                embedding_dim=word_embedding_dim,
                kernel_sizes=word_kernel_sizes,
                num_filters=num_filters,
                pad_id=word_pad_id,
                spatial_dropout=spatial_dropout,
            )
            self.word_embedding = self.word_cnn.embedding
            word_out_dim = self.word_cnn.out_features
        else:
            _validate_positive_int(hidden_size, "hidden_size")
            _validate_positive_int(num_rnn_layers, "num_rnn_layers")
            self.word_embedding = nn.Embedding(
                word_vocab_size, word_embedding_dim, padding_idx=word_pad_id
            )
            rnn_cls = nn.GRU if word_encoder_type == "bigru" else nn.LSTM
            rnn_inter_drop = drop * 0.5 if num_rnn_layers > 1 else 0.0
            self.word_rnn = rnn_cls(
                word_embedding_dim,
                hidden_size,
                num_layers=num_rnn_layers,
                dropout=rnn_inter_drop,
                batch_first=True,
                bidirectional=True,
            )
            if rnn_pooling in ("attn", "attn_max"):
                self.attn_pool = MaskedSelfAttentionPool1D(hidden_size * 2)
            elif rnn_pooling in ("gated_attn", "gated_attn_max", "gated_attn_max_mean"):
                self.attn_pool = MaskedGatedAttentionPool1D(hidden_size * 2)
            if rnn_pooling == "gated_attn_max_mean":
                word_out_dim = hidden_size * 6
            elif rnn_pooling in ("attn_max", "gated_attn_max"):
                word_out_dim = hidden_size * 4
            else:
                word_out_dim = hidden_size * 2

        self.char_cnn = Conv1DPoolEncoder(
            vocab_size=char_vocab_size,
            embedding_dim=char_embedding_dim,
            kernel_sizes=char_kernel_sizes,
            num_filters=num_filters,
            pad_id=char_pad_id,
            spatial_dropout=spatial_dropout,
        )
        self.word_drop = nn.Dropout(w_drop)
        self.char_drop = nn.Dropout(c_drop)
        self.dropout = nn.Dropout(drop)
        fused_dim = word_out_dim + self.char_cnn.out_features
        self.fusion_norm = nn.LayerNorm(fused_dim) if self.use_layernorm else None
        self.ms_dropouts = (
            nn.ModuleList([nn.Dropout(drop) for _ in range(5)])
            if self.multi_sample_dropout
            else None
        )
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(
        self,
        word_ids: Tensor,
        word_lengths: Tensor,
        char_ids: Tensor,
        char_lengths: Tensor,
    ) -> Tensor:
        if word_ids.shape[0] != char_ids.shape[0]:
            raise ValueError("word_ids and char_ids must have matching batch size")
        if word_ids.device != self.word_embedding.weight.device:
            raise ValueError("word_ids and model must be on the same device")

        if self.word_encoder_type == "cnn":
            word_feat = self.word_cnn(word_ids, word_lengths)
        else:
            mask, _ = _validate_seq_inputs(
                word_ids, word_lengths, self.word_vocab_size, "word_ids"
            )
            embedded = self.word_drop(self.word_embedding(word_ids))
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded,
                word_lengths.detach().to(device="cpu"),
                batch_first=True,
                enforce_sorted=False,
            )
            if self.word_encoder_type == "bilstm":
                packed_out, (hidden, _) = self.word_rnn(packed)
            else:
                packed_out, hidden = self.word_rnn(packed)
            if self.rnn_pooling == "last":
                word_feat = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                assert self.attn_pool is not None
                unpacked, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_out, batch_first=True, total_length=word_ids.shape[1]
                )
                attn_vec = self.attn_pool(unpacked, mask)
                if self.rnn_pooling in ("attn", "gated_attn"):
                    word_feat = attn_vec
                else:
                    masked_states = unpacked.masked_fill(
                        ~mask.unsqueeze(-1), float("-inf")
                    )
                    max_vec = masked_states.max(dim=1).values
                    if self.rnn_pooling == "gated_attn_max_mean":
                        mask_f = mask.unsqueeze(-1).to(dtype=unpacked.dtype)
                        mean_vec = (unpacked * mask_f).sum(dim=1) / mask_f.sum(
                            dim=1
                        ).clamp(min=1.0)
                        word_feat = torch.cat((attn_vec, max_vec, mean_vec), dim=1)
                    else:
                        word_feat = torch.cat((attn_vec, max_vec), dim=1)

        char_feat = self.char_drop(self.char_cnn(char_ids, char_lengths))
        fused = torch.cat((word_feat, char_feat), dim=1)
        if self.fusion_norm is not None:
            fused = self.fusion_norm(fused)
        if self.ms_dropouts is not None:
            return torch.stack(
                [self.classifier(dp(fused)) for dp in self.ms_dropouts], dim=0
            ).mean(dim=0)
        return self.classifier(self.dropout(fused))


class TriHybridWordCNNBiGRUCharClassifier(nn.Module):
    """Phase 3D tri-branch hybrid: Word-CNN + BiGRU (Attn+Max) + Char-CNN with fusion."""

    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int,
        word_embedding_dim: int = 128,
        char_embedding_dim: int = 64,
        word_kernel_sizes: Sequence[int] = (2, 3, 4),
        char_kernel_sizes: Sequence[int] = (2, 3, 4, 5, 7),
        word_num_filters: int = 128,
        char_num_filters: int = 128,
        hidden_size: int = 128,
        fusion_type: str = "concat",
        fusion_dim: int = 256,
        dropout: float = 0.4,
        spatial_dropout: float = 0.15,
        word_pad_id: int = 0,
        char_pad_id: int = 0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        _validate_positive_int(num_classes, "num_classes")
        _validate_positive_int(word_vocab_size, "word_vocab_size")
        _validate_positive_int(char_vocab_size, "char_vocab_size")
        if fusion_type not in ("concat", "gated", "attention"):
            raise ValueError("fusion_type must be concat, gated, or attention")
        drop = _validate_dropout(dropout)
        sp_drop = _validate_dropout(spatial_dropout)

        self.architecture = "word_cnn_bigru_char_cnn"
        self.fusion_type = fusion_type
        self.num_classes = num_classes
        self.word_vocab_size = word_vocab_size

        # Shared word embedding feeding both local Word-CNN and sequential BiGRU
        self.word_cnn = Conv1DPoolEncoder(
            vocab_size=word_vocab_size,
            embedding_dim=word_embedding_dim,
            kernel_sizes=word_kernel_sizes,
            num_filters=word_num_filters,
            pad_id=word_pad_id,
            spatial_dropout=sp_drop,
        )
        self.word_embedding = self.word_cnn.embedding
        self.word_rnn = nn.GRU(
            word_embedding_dim, hidden_size, batch_first=True, bidirectional=True
        )
        self.rnn_attn_pool = MaskedSelfAttentionPool1D(hidden_size * 2)
        self.char_cnn = Conv1DPoolEncoder(
            vocab_size=char_vocab_size,
            embedding_dim=char_embedding_dim,
            kernel_sizes=char_kernel_sizes,
            num_filters=char_num_filters,
            pad_id=char_pad_id,
            spatial_dropout=sp_drop,
        )

        w_cnn_dim = self.word_cnn.out_features
        w_rnn_dim = hidden_size * 4
        c_cnn_dim = self.char_cnn.out_features
        total_concat_dim = w_cnn_dim + w_rnn_dim + c_cnn_dim

        self.emb_dropout = nn.Dropout(drop)
        self.dropout = nn.Dropout(drop)

        if fusion_type == "concat":
            cls_in_dim = total_concat_dim
        elif fusion_type == "gated":
            self.proj_wcnn = nn.Linear(w_cnn_dim, fusion_dim)
            self.proj_wrnn = nn.Linear(w_rnn_dim, fusion_dim)
            self.proj_char = nn.Linear(c_cnn_dim, fusion_dim)
            self.gate_net = nn.Linear(fusion_dim * 3, 3)
            cls_in_dim = fusion_dim * 3
        else:  # attention fusion across the 3 branch modality vectors
            self.proj_wcnn = nn.Linear(w_cnn_dim, fusion_dim)
            self.proj_wrnn = nn.Linear(w_rnn_dim, fusion_dim)
            self.proj_char = nn.Linear(c_cnn_dim, fusion_dim)
            self.branch_attn = nn.Linear(fusion_dim, 1, bias=False)
            cls_in_dim = fusion_dim * 4

        self.classifier = nn.Linear(cls_in_dim, num_classes)

    def forward(
        self,
        word_ids: Tensor,
        word_lengths: Tensor,
        char_ids: Tensor,
        char_lengths: Tensor,
    ) -> Tensor:
        if word_ids.shape[0] != char_ids.shape[0]:
            raise ValueError("word_ids and char_ids must have matching batch size")
        if word_ids.device != self.word_embedding.weight.device:
            raise ValueError("word_ids and model must be on the same device")

        wcnn_feat = self.word_cnn(word_ids, word_lengths)
        mask, _ = _validate_seq_inputs(
            word_ids, word_lengths, self.word_vocab_size, "word_ids"
        )
        embedded = self.emb_dropout(self.word_embedding(word_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            word_lengths.detach().to(device="cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.word_rnn(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=word_ids.shape[1]
        )
        attn_vec = self.rnn_attn_pool(unpacked, mask)
        masked_states = unpacked.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_vec = masked_states.max(dim=1).values
        wrnn_feat = torch.cat((attn_vec, max_vec), dim=1)
        char_feat = self.char_cnn(char_ids, char_lengths)

        if self.fusion_type == "concat":
            fused = torch.cat((wcnn_feat, wrnn_feat, char_feat), dim=1)
        elif self.fusion_type == "gated":
            z1 = F.gelu(self.proj_wcnn(wcnn_feat))
            z2 = F.gelu(self.proj_wrnn(wrnn_feat))
            z3 = F.gelu(self.proj_char(char_feat))
            gates = torch.sigmoid(self.gate_net(torch.cat((z1, z2, z3), dim=1)))
            fused = torch.cat(
                (
                    z1 * gates[:, 0:1],
                    z2 * gates[:, 1:2],
                    z3 * gates[:, 2:3],
                ),
                dim=1,
            )
        else:
            z1 = F.gelu(self.proj_wcnn(wcnn_feat))
            z2 = F.gelu(self.proj_wrnn(wrnn_feat))
            z3 = F.gelu(self.proj_char(char_feat))
            stack = torch.stack((z1, z2, z3), dim=1)  # [B, 3, D]
            weights = torch.softmax(
                self.branch_attn(torch.tanh(stack)).squeeze(-1), dim=1
            ).unsqueeze(-1)
            context = torch.sum(stack * weights, dim=1)
            fused = torch.cat((z1, z2, z3, context), dim=1)

        return self.classifier(self.dropout(fused))
