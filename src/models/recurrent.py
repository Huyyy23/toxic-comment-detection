"""Random-initialized recurrent classifiers for CLEAN, OFFENSIVE, HATE logits."""

import math
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RecurrentClassifier(nn.Module):
    """Single-layer LSTM variants with padding-independent sequence features.

    Inputs are right-padded LongTensors [batch, time] and positive sequence
    lengths [batch]. Lengths, not token values, define valid positions. Output
    is unnormalized logits [batch, 3]; dropout is disabled by ``eval()``.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        dropout: float = 0.3,
        architecture: str = "lstm",
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("vocab_size", vocab_size),
            ("embedding_dim", embedding_dim),
            ("hidden_size", hidden_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(pad_id, bool) or not isinstance(pad_id, int):
            raise TypeError("pad_id must be an integer")
        if not 0 <= pad_id < vocab_size:
            raise ValueError("pad_id must be inside the vocabulary")
        if isinstance(dropout, bool) or not isinstance(dropout, Real):
            raise TypeError("dropout must be a real number")
        if not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError("dropout must be finite and between 0 and 1")
        if not isinstance(architecture, str) or architecture not in (
            "lstm", "bilstm", "bilstm_attention",
        ):
            raise ValueError("architecture must be lstm, bilstm, or bilstm_attention")

        self.architecture = architecture
        bidirectional = architecture != "lstm"
        feature_size = hidden_size * (2 if bidirectional else 1)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(float(dropout))
        self.recurrent = nn.LSTM(
            embedding_dim, hidden_size, batch_first=True,
            bidirectional=bidirectional,
        )
        if architecture == "bilstm_attention":
            self.attention = nn.Linear(feature_size, 1, bias=False)
        self.classifier = nn.Linear(feature_size, 3)

    def forward(self, input_ids: Tensor, lengths: Tensor) -> Tensor:
        """Return three logits per example; ignore all positions past lengths."""
        if not isinstance(input_ids, Tensor) or not isinstance(lengths, Tensor):
            raise TypeError("input_ids and lengths must be tensors")
        if input_ids.dtype != torch.long or lengths.dtype != torch.long:
            raise TypeError("input_ids and lengths must have torch.long dtype")
        if input_ids.ndim != 2 or lengths.ndim != 1:
            raise ValueError("input_ids must be [batch, time] and lengths [batch]")
        batch_size, time_steps = input_ids.shape
        if batch_size == 0 or time_steps == 0:
            raise ValueError("input_ids must have nonempty batch and time dimensions")
        if lengths.shape[0] != batch_size:
            raise ValueError("lengths must contain one entry per example")
        cpu_lengths = lengths.detach().to(device="cpu")
        if torch.any(cpu_lengths <= 0) or torch.any(cpu_lengths > time_steps):
            raise ValueError("lengths must be between 1 and the padded sequence length")
        if input_ids.device != self.embedding.weight.device:
            raise ValueError("input_ids and model must be on the same device")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.embedding.num_embeddings):
            raise ValueError("input_ids must be inside the vocabulary")

        embedded = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(
            embedded, cpu_lengths, batch_first=True, enforce_sorted=False,
        )
        encoded, (hidden, _) = self.recurrent(packed)
        if self.architecture == "bilstm_attention":
            outputs, _ = pad_packed_sequence(encoded, batch_first=True)
            positions = torch.arange(outputs.shape[1], device=outputs.device)
            mask = positions.unsqueeze(0) < lengths.to(outputs.device).unsqueeze(1)
            scores = self.attention(outputs).squeeze(-1).masked_fill(~mask, -torch.inf)
            weights = torch.softmax(scores, dim=1)
            representation = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)
        elif self.architecture == "bilstm":
            representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            representation = hidden[-1]
        return self.classifier(self.dropout(representation))
