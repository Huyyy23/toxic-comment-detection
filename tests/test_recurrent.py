"""CPU fixture coverage for scratch recurrent classifiers; no dataset required."""

import pytest
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.recurrent import RecurrentClassifier


ARCHITECTURES = ("lstm", "bilstm", "bilstm_attention")


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_forward_backward_updates_parameters(architecture):
    torch.manual_seed(17)
    model = RecurrentClassifier(
        12, embedding_dim=5, hidden_size=4, dropout=0.2,
        architecture=architecture,
    )
    inputs = torch.tensor([[2, 3, 0], [4, 5, 6], [7, 0, 0]])
    lengths = torch.tensor([2, 3, 1])
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    logits = model(inputs, lengths)
    assert logits.shape == (3, 3)
    assert torch.isfinite(logits).all()
    nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    optimizer.step()

    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all()
        assert not torch.equal(before[name], parameter), name
    assert model.embedding.weight.requires_grad
    assert model.embedding.padding_idx == 0
    assert torch.count_nonzero(model.embedding.weight[0]) == 0


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_unsorted_batch_matches_individual_sequences_and_ignores_right_padding(architecture):
    torch.manual_seed(23)
    model = RecurrentClassifier(
        16, embedding_dim=6, hidden_size=5, dropout=0.3,
        architecture=architecture,
    ).eval()
    inputs = torch.tensor([[2, 3, 0, 0], [4, 5, 6, 7], [8, 0, 0, 0]])
    lengths = torch.tensor([2, 4, 1])
    with torch.no_grad():
        batched = model(inputs, lengths)
        individual = torch.cat([
            model(inputs[index:index + 1, :length], torch.tensor([length]))
            for index, length in enumerate(lengths.tolist())
        ])
        extended = model(nn.functional.pad(inputs, (0, 5)), lengths)
        changed_padding = inputs.clone()
        mask = torch.arange(inputs.shape[1]).unsqueeze(0) >= lengths.unsqueeze(1)
        changed_padding[mask] = 15
        ignored_tokens = model(changed_padding, lengths)
        reordered = model(inputs.flip(0), lengths.flip(0)).flip(0)
    torch.testing.assert_close(batched, individual, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(batched, extended)
    torch.testing.assert_close(batched, ignored_tokens)
    torch.testing.assert_close(batched, reordered)


def test_attention_masks_padding_before_softmax():
    torch.manual_seed(31)
    model = RecurrentClassifier(
        10, embedding_dim=4, hidden_size=3, dropout=0,
        architecture="bilstm_attention",
    ).eval()
    inputs = torch.tensor([[2, 0, 0, 0], [3, 4, 5, 6]])
    lengths = torch.tensor([1, 4])
    with torch.no_grad():
        model.attention.weight.zero_()
        packed = pack_padded_sequence(
            model.embedding(inputs), lengths, batch_first=True, enforce_sorted=False,
        )
        encoded, _ = model.recurrent(packed)
        outputs, _ = pad_packed_sequence(encoded, batch_first=True)
        means = torch.stack([outputs[0, :1].mean(0), outputs[1, :4].mean(0)])
        expected = model.classifier(means)
        actual = model(inputs, lengths)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("architecture", ("lstm", "bilstm"))
def test_plain_variants_use_final_forward_and_backward_hidden_states(architecture):
    torch.manual_seed(41)
    model = RecurrentClassifier(
        10, embedding_dim=4, hidden_size=3, dropout=0,
        architecture=architecture,
    ).eval()
    inputs = torch.tensor([[2, 0, 0], [3, 4, 5]])
    lengths = torch.tensor([1, 3])
    with torch.no_grad():
        packed = pack_padded_sequence(
            model.embedding(inputs), lengths, batch_first=True, enforce_sorted=False,
        )
        _, (hidden, _) = model.recurrent(packed)
        representation = hidden[-1] if architecture == "lstm" else torch.cat(
            (hidden[-2], hidden[-1]), dim=1,
        )
        expected = model.classifier(representation)
        actual = model(inputs, lengths)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_single_layer_dropout_is_active_only_during_training(architecture):
    torch.manual_seed(43)
    model = RecurrentClassifier(
        12, embedding_dim=12, hidden_size=8, dropout=0.5,
        architecture=architecture,
    )
    inputs = torch.tensor([[2, 3, 4], [5, 6, 7]])
    lengths = torch.tensor([3, 3])
    assert not torch.equal(model(inputs, lengths), model(inputs, lengths))
    model.eval()
    torch.testing.assert_close(model(inputs, lengths), model(inputs, lengths))


def test_custom_padding_id_stays_zero_after_update():
    model = RecurrentClassifier(8, embedding_dim=4, hidden_size=3, pad_id=7)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    nn.functional.cross_entropy(
        model(torch.tensor([[2, 7], [3, 4]]), torch.tensor([1, 2])),
        torch.tensor([0, 2]),
    ).backward()
    optimizer.step()
    assert model.embedding.padding_idx == 7
    assert torch.count_nonzero(model.embedding.weight[7]) == 0


@pytest.mark.parametrize("config", [
    {"vocab_size": 0}, {"vocab_size": -1}, {"vocab_size": True},
    {"vocab_size": 4.5}, {"embedding_dim": 0}, {"embedding_dim": False},
    {"hidden_size": -2}, {"hidden_size": 2.5},
    {"dropout": -0.1}, {"dropout": 1.1}, {"dropout": float("nan")},
    {"dropout": float("inf")}, {"dropout": "0.2"}, {"dropout": True},
    {"architecture": "gru"}, {"architecture": None},
    {"pad_id": -1}, {"pad_id": 10}, {"pad_id": True}, {"pad_id": 0.0},
])
def test_invalid_configuration_rejected(config):
    arguments = {"vocab_size": 10, "embedding_dim": 4, "hidden_size": 3}
    arguments.update(config)
    with pytest.raises((TypeError, ValueError)):
        RecurrentClassifier(**arguments)


@pytest.mark.parametrize("inputs,lengths", [
    ([[2, 3]], torch.tensor([2])),
    (torch.tensor([[2, 3]]), [2]),
    (torch.tensor([2, 3]), torch.tensor([2])),
    (torch.ones(1, 2, 1, dtype=torch.long), torch.tensor([2])),
    (torch.empty(0, 2, dtype=torch.long), torch.empty(0, dtype=torch.long)),
    (torch.empty(1, 0, dtype=torch.long), torch.tensor([0])),
    (torch.tensor([[2.0, 3.0]]), torch.tensor([2])),
    (torch.tensor([[2, 3]], dtype=torch.int32), torch.tensor([2])),
    (torch.tensor([[2, 3]]), torch.tensor([2.0])),
    (torch.tensor([[2, 3]]), torch.tensor([[2]])),
    (torch.tensor([[2, 3]]), torch.tensor([1, 2])),
    (torch.tensor([[2, 3]]), torch.tensor([0])),
    (torch.tensor([[2, 3]]), torch.tensor([-1])),
    (torch.tensor([[2, 3]]), torch.tensor([3])),
    (torch.tensor([[-1, 3]]), torch.tensor([2])),
    (torch.tensor([[2, 10]]), torch.tensor([2])),
])
def test_invalid_batch_rejected(inputs, lengths):
    model = RecurrentClassifier(10, embedding_dim=4, hidden_size=3)
    with pytest.raises((TypeError, ValueError)):
        model(inputs, lengths)
