import warnings

import numpy as np
import pytest
import torch

from src.models.train_recurrent_official import _copy_fasttext_vectors


class KeyedVectors:
    vector_size = 3
    key_to_index = {"apple": 0, "pear": 1}

    def __getitem__(self, token):
        vector = np.asarray({"apple": [1, 2, 3], "pear": [4, 5, 6]}[token], dtype=np.float32)
        vector.setflags(write=False)
        return vector


def test_fasttext_vectors_align_to_learned_ids_and_preserve_reserved_rows():
    embedding = torch.nn.Embedding(5, 3, padding_idx=0)
    with torch.no_grad():
        embedding.weight[0].fill_(9)
        embedding.weight[1].fill_(8)
        embedding.weight[2:].fill_(7)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        copied = _copy_fasttext_vectors(embedding, ["apple", "missing", "pear"], KeyedVectors())

    assert copied == 2
    assert torch.equal(embedding.weight[0], torch.zeros(3))
    assert torch.equal(embedding.weight[1], torch.full((3,), 8.0))
    assert torch.equal(embedding.weight[2], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(embedding.weight[3], torch.full((3,), 7.0))
    assert torch.equal(embedding.weight[4], torch.tensor([4.0, 5.0, 6.0]))


def test_fasttext_dimensions_must_match_embedding():
    embedding = torch.nn.Embedding(3, 4, padding_idx=0)
    with pytest.raises(ValueError, match="vector size"):
        _copy_fasttext_vectors(embedding, ["apple"], KeyedVectors())
