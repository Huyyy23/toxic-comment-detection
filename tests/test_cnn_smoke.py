"""Unit and fixture smoke tests for TV4 TextCNN, Char-CNN, and 2-class compatibility."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import torch
from mlflow.tracking import MlflowClient
from torch import nn

from src.evaluation.recurrent_report import load_report, render_report
from src.models.checkpoint import seed_everything
from src.models.cnn import (
    CharCNNClassifier,
    HybridWordCharClassifier,
    TextCNNClassifier,
)
from src.models.cnn_pipeline import (
    SequenceVocab,
    load_cnn_artifacts,
    predict_cnn_from_artifacts,
    run_cnn_variant,
)
from src.tracking.mlflow_tracker import MLflowTracker
from src.tracking.recurrent_tracking import log_epoch


def _row(rid: str, text: str, label: int, source: str, split: str) -> dict:
    return {"id": rid, "text": text, "label": label, "source": source, "split": split}


TRAIN_ROWS = [
    _row("tr0", "bài viết rất hay và bổ ích", 0, "vihsd", "train"),
    _row("tr1", "cảm ơn chia sẻ thông tin chuẩn", 0, "victsd", "train"),
    _row("tr2", "nội dung sạch sẽ dễ hiểu", 0, "vihsd", "train"),
    _row("tr3", "rất thích bài viết này ạ", 0, "victsd", "train"),
    _row("tr4", "https://example.com", 0, "vihsd", "train"),
    _row("tr5", "đồ ngu vcl cút đi lũ óc chó", 1, "vihsd", "train"),
    _row("tr6", "mày điên khùng nói nhảm đm", 1, "victsd", "train"),
    _row("tr7", "giết hết lũ rác rưởi này", 1, "vihsd", "train"),
    _row("tr8", "con này xấu xí vô duyên clgt", 1, "victsd", "train"),
    _row("tr9", "bọn súc vật ngu ngốc cút đi", 1, "vihsd", "train"),
]

VAL_ROWS = [
    _row("va0", "bài viết hay dễ hiểu valonlyword", 0, "vihsd", "val"),
    _row("va1", "cảm ơn bạn đã chia sẻ", 0, "victsd", "val"),
    _row("va2", "mày ngu quá vcl cút đi", 1, "vihsd", "val"),
    _row("va3", "bọn rác rưởi đáng ghét", 1, "victsd", "val"),
]


def _write_fixture(tmp_path: Path, val_rows: list[dict] = VAL_ROWS) -> Path:
    ds_dir = tmp_path / "binary_n2"
    ds_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(TRAIN_ROWS).to_parquet(ds_dir / "train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(ds_dir / "val.parquet", index=False)
    manifest = {r["id"]: {"id": r["id"], "split": r["split"]} for r in TRAIN_ROWS + val_rows}
    (ds_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ds_dir


@pytest.mark.parametrize("cls,kernels", [
    (TextCNNClassifier, (3,)),
    (TextCNNClassifier, (3, 4, 5)),
    (CharCNNClassifier, (3, 4, 5, 7)),
])
def test_cnn_forward_backward_and_short_sequence(cls, kernels):
    torch.manual_seed(17)
    model = cls(vocab_size=16, embedding_dim=8, kernel_sizes=kernels, num_filters=6, dropout=0.2)
    inputs = torch.tensor([[2, 3, 0, 0], [4, 5, 6, 7], [8, 0, 0, 0]])
    lengths = torch.tensor([2, 4, 1])
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    logits = model(inputs, lengths)
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()
    nn.functional.cross_entropy(logits, torch.tensor([0, 1, 1])).backward()
    for param in model.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()
    optimizer.step()

    for name, param in model.named_parameters():
        assert not torch.equal(before[name], param), name
    assert torch.count_nonzero(model.embedding.weight[0]) == 0


@pytest.mark.parametrize("cls,kernels", [
    (TextCNNClassifier, (3, 4, 5)),
    (CharCNNClassifier, (3, 4, 5, 7)),
])
def test_cnn_ignores_right_padding_and_batch_order(cls, kernels):
    torch.manual_seed(29)
    model = cls(
        vocab_size=16, embedding_dim=6, kernel_sizes=kernels, num_filters=4, dropout=0.3
    ).eval()
    inputs = torch.tensor([[2, 3, 0, 0], [4, 5, 6, 7], [8, 0, 0, 0]])
    lengths = torch.tensor([2, 4, 1])
    with torch.no_grad():
        batched = model(inputs, lengths)
        individual = torch.cat([
            model(inputs[i:i + 1, :length], torch.tensor([length]))
            for i, length in enumerate(lengths.tolist())
        ])
        extended = model(nn.functional.pad(inputs, (0, 6)), lengths)
        corrupted = inputs.clone()
        mask = torch.arange(inputs.shape[1]).unsqueeze(0) >= lengths.unsqueeze(1)
        corrupted[mask] = 15
        ignored = model(corrupted, lengths)
        reordered = model(inputs.flip(0), lengths.flip(0)).flip(0)

    torch.testing.assert_close(batched, individual, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(batched, extended)
    torch.testing.assert_close(batched, ignored)
    torch.testing.assert_close(batched, reordered)


@pytest.mark.parametrize("encoder_type", ["cnn", "bilstm", "bigru"])
def test_hybrid_word_char_forward_backward(encoder_type):
    torch.manual_seed(11)
    model = HybridWordCharClassifier(
        word_vocab_size=12,
        char_vocab_size=15,
        word_embedding_dim=6,
        char_embedding_dim=4,
        word_kernel_sizes=(3, 4),
        char_kernel_sizes=(3, 5),
        num_filters=4,
        hidden_size=4,
        word_encoder_type=encoder_type,
        dropout=0.2,
    )
    w_ids = torch.tensor([[2, 3, 0], [4, 5, 6]])
    w_len = torch.tensor([2, 3])
    c_ids = torch.tensor([[2, 3, 4, 0], [5, 6, 7, 8]])
    c_len = torch.tensor([3, 4])
    logits = model(w_ids, w_len, c_ids, c_len)
    assert logits.shape == (2, 2)
    nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    for param in model.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_deterministic_cnn_forward_backward():
    seed_everything(42, deterministic=True)
    model = HybridWordCharClassifier(
        word_vocab_size=12, char_vocab_size=16, word_embedding_dim=6, char_embedding_dim=4,
        word_kernel_sizes=(3, 4, 5), char_kernel_sizes=(3, 4, 5, 7),
        num_filters=4, hidden_size=4, word_encoder_type="bigru",
    ).cuda()
    logits = model(
        torch.tensor([[2, 3, 4]], device="cuda"), torch.tensor([3], device="cuda"),
        torch.tensor([[2, 3, 4, 5]], device="cuda"), torch.tensor([4], device="cuda"),
    )
    nn.functional.cross_entropy(logits, torch.tensor([1], device="cuda")).backward()
    assert torch.isfinite(logits).all()


def test_sequence_vocab_handles_empty_preprocessed_text():
    vocab = SequenceVocab("word", min_freq=1).fit(["xin chào"])
    assert vocab.encode("", max_length=10) == [vocab.unk_id]


@pytest.mark.parametrize("variant", [
    "CNN-01",
    "CNN-02",
    "CHAR-01",
    "HYBRID-01",
    "HYBRID-RNN-01",
    "CNN-IMB-CW",
    "CNN-IMB-FOCAL",
    "CNN-IMB-SAMPLER",
])
def test_cnn_pipeline_smoke_train_vocab_reload_and_mlflow(tmp_path, variant):
    ds_dir = _write_fixture(tmp_path)
    tracking_uri = (tmp_path / "mlruns").as_uri()
    res = run_cnn_variant(
        variant=variant,
        dataset_dir=ds_dir,
        artifacts_root=tmp_path / "out",
        tracking_uri=tracking_uri,
        seeds=[42, 43],
        result_type="smoke",
        overrides={
            "epochs": 2,
            "batch_size": 4,
            "patience": 2,
            "min_word_freq": 1,
            "min_char_freq": 1,
            "device": "cpu",
        },
    )
    agg = res["aggregate"]
    assert agg["n_seeds"] == 2
    assert set(agg["by_source"].keys()) == {"vihsd", "victsd"}
    assert 0.0 <= agg["macro_f1"]["mean"] <= 1.0

    seed_dir = Path(res["runs"][0]["seed_dir"])
    vocab_data = json.loads((seed_dir / "vocab.json").read_text(encoding="utf-8"))
    if vocab_data["word_vocab"] is not None:
        assert "valonlyword" not in vocab_data["word_vocab"]["tokens"]

    err_text = (seed_dir / "error_summary.json").read_text(encoding="utf-8")
    assert "va0" not in err_text and "valonlyword" not in err_text

    preds_json = json.loads((seed_dir / "val_predictions.json").read_text(encoding="utf-8"))
    preds_csv = pd.read_csv(seed_dir / "predictions_val.csv")
    assert len(preds_json) == len(VAL_ROWS) == len(preds_csv)
    for row in preds_json:
        assert set(row["probabilities"].keys()) == {"CLEAN", "TOXIC"}
        assert abs(sum(row["probabilities"].values()) - 1.0) < 1e-4
        assert row["is_calibrated"] is False

    loaded = load_cnn_artifacts(seed_dir, device="cpu")
    re_preds, re_probs = predict_cnn_from_artifacts(loaded, [r["text"] for r in VAL_ROWS])
    assert re_preds == [row["predicted_label"] for row in preds_json]
    assert re_probs.shape == (len(VAL_ROWS), 2)

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(res["runs"][0]["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.tags["pretrained"] == "false"


def test_cnn_pipeline_rejects_test_split(tmp_path):
    bad_val = VAL_ROWS + [
        {"id": "te0", "text": "test leak", "label": 0, "source": "vihsd", "split": "test"}
    ]
    ds_dir = _write_fixture(tmp_path, val_rows=bad_val)
    with pytest.raises(ValueError, match="split='val'"):
        run_cnn_variant(
            variant="CNN-01",
            dataset_dir=ds_dir,
            artifacts_root=tmp_path / "out",
            tracking_uri=(tmp_path / "mlruns").as_uri(),
            seeds=[42],
            overrides={"epochs": 1, "device": "cpu"},
        )


def test_binary_tracking_and_report_compatibility(tmp_path):
    store = (tmp_path / "mlruns").as_uri()
    tracker = MLflowTracker(experiment_name="binary-compat", tracking_uri=store)
    metrics = {
        "loss": 0.4,
        "macro_f1": 1.0,
        "per_class": [
            {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
            {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
        ],
        "confusion_matrix": [[1, 0], [0, 1]],
    }
    with tracker.run(run_name="binary-epoch") as active:
        run_id = active.run_id
        log_epoch(
            active,
            {"epoch": 1, "history": [{"train": deepcopy(metrics), "val": deepcopy(metrics)}]},
        )
    ml_run = MlflowClient(tracking_uri=store).get_run(run_id)
    assert "val_TOXIC_f1" in ml_run.data.metrics
    assert "val_OFFENSIVE_f1" not in ml_run.data.metrics

    report_payload = {
        "status": "OFFICIAL",
        "architecture": "textcnn",
        "run_id": "bin-run",
        "training": {
            "history": [{"epoch": 1, "train": deepcopy(metrics), "val": deepcopy(metrics)}],
            "best_epoch": 1,
            "best_macro_f1": 1.0,
        },
        "last_validation": {
            **deepcopy(metrics),
            "predictions": [
                {
                    "id": "0",
                    "true_label": 0,
                    "predicted_label": 0,
                    "probabilities_if_available": [1.0, 0.0],
                },
                {
                    "id": "1",
                    "true_label": 1,
                    "predicted_label": 1,
                    "probabilities_if_available": [0.0, 1.0],
                },
            ],
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    loaded = load_report(report_path)
    figs = render_report(loaded, tmp_path / "figs")
    assert set(figs) == {"loss_curves", "macro_f1", "confusion_matrix"}
