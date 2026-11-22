"""CPU synthetic-only integration of group-one data and scratch training."""

import json
import pytest
from mlflow.tracking import MlflowClient
import torch

from src.models import train_recurrent_official as official
from src.models.smoke_recurrent import run_smoke


def test_official_runner_evaluates_validation_best_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(official, "_load_dataset", lambda _config: (
        [{"id": "train-1", "text": "hello there", "label": 0, "split": "train"}],
        [{"id": "val-1", "text": "hello again", "label": 1, "split": "val"}],
        {"train_val_sha256": "data-hash", "manifest_sha256": "manifest-hash"},
    ))

    def fake_fit(model, *_args, on_epoch_end, **_kwargs):
        scores = [{"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 1}] * 3
        values = {"loss": 0.5, "macro_f1": 0.5, "per_class": scores}
        history = [{"train": values, "val": values}]
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(1)
        on_epoch_end({
            "epoch": 1, "best_macro_f1": 0.5, "bad_epochs": 0, "best_epoch": 1,
            "history": history,
        })
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(2)
        on_epoch_end({
            "epoch": 2, "best_macro_f1": 0.5, "bad_epochs": 1, "best_epoch": 1,
            "history": history,
        })
        return {"best_epoch": 1, "best_macro_f1": 0.5}

    def fake_evaluate(model, _batches, *, device):
        assert all(torch.all(parameter == 1) for parameter in model.parameters())
        return {"macro_f1": 0.5, "predictions": []}

    monkeypatch.setattr(official, "fit", fake_fit)
    monkeypatch.setattr(official, "evaluate", fake_evaluate)
    store = (tmp_path / "mlruns").as_uri()
    output = tmp_path / "unused-output-dir"
    result = official.run(official.DEFAULT_CONFIG, "lstm", False, output, store)

    assert result["prediction_checkpoint"] == "best"
    assert result["training"]["best_epoch"] == 1
    assert result["last_validation"]["macro_f1"] == 0.5
    assert not output.exists()
    client = MlflowClient(tracking_uri=store)
    run = client.get_run(result["run_id"])
    assert run.info.status == "FINISHED"
    assert {item.path for item in client.list_artifacts(result["run_id"])} == {
        "best.pt", "last.pt", "checkpoint-epoch-1.pt", "config.json", "metrics.json",
        "validation_predictions.json",
    }

def test_official_resume_allows_increasing_only_epoch_target(tmp_path, monkeypatch):
    monkeypatch.setattr(official, "_load_dataset", lambda _config: (
        [{"id": "train-1", "text": "hello there", "label": 0, "split": "train"}],
        [{"id": "val-1", "text": "hello again", "label": 1, "split": "val"}],
        {"train_val_sha256": "data-hash", "manifest_sha256": "manifest-hash"},
    ))
    calls = []

    def fake_fit(model, *_args, epochs, start_epoch, on_epoch_end, **_kwargs):
        calls.append((epochs, start_epoch))
        epoch = start_epoch + 1
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(epoch)
        on_epoch_end({
            "epoch": epoch, "best_macro_f1": 0.5 + start_epoch / 10,
            "bad_epochs": 0, "best_epoch": epoch,
        })
        return {"best_epoch": epoch, "best_macro_f1": 0.5 + start_epoch / 10}

    monkeypatch.setattr(official, "fit", fake_fit)
    monkeypatch.setattr(official, "evaluate",
                        lambda *_args, **_kwargs: {"macro_f1": 0.5, "predictions": []})
    output = tmp_path / "run"
    official.run(official.DEFAULT_CONFIG, "lstm", False, output, None)
    config = json.loads(official.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["epochs"] = 30
    config_path = tmp_path / "resume_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    resumed_output = tmp_path / "resumed-run"
    resumed = official.run(config_path, "lstm", False, resumed_output, None,
                           resume=output / "last.pt")

    assert calls == [(10, 0), (30, 1)]
    assert (output / "last.pt").is_file()
    assert {p.name for p in output.iterdir()} == {
        "best.pt", "last.pt", "config.json", "metrics.json", "validation_predictions.json",
    }
    assert {p.name for p in resumed_output.iterdir()} == {
        "best.pt", "last.pt", "config.json", "metrics.json", "validation_predictions.json",
    }
    assert resumed["training"]["best_epoch"] == 2


@pytest.mark.parametrize("architecture", ["lstm", "bilstm", "bilstm_attention"])
def test_synthetic_pipeline_all_architectures(tmp_path, architecture):
    result = run_smoke(tmp_path / architecture, architecture=architecture)
    assert result["status"] == "SYNTHETIC_NOT_RESEARCH"
    output = tmp_path / architecture
    assert (output / "best.pt").is_file()
    checkpoint = torch.load(output / "last.pt", weights_only=True)
    assert checkpoint["epoch"] >= 1
    predictions = result["last_validation"]["predictions"]
    assert [row["id"] for row in predictions] == ["synthetic-v0", "synthetic-v1", "synthetic-v2"]
    assert all(len(row["probabilities_if_available"]) == 3 for row in predictions)


def test_smoke_resume_preserves_completed_epoch(tmp_path):
    first = run_smoke(tmp_path / "first")
    resumed = run_smoke(tmp_path / "resumed", resume=tmp_path / "first" / "last.pt")
    assert resumed["last_validation"] == first["last_validation"]
    assert resumed["training"]["history"] == []


def test_smoke_rejects_nonempty_output_directory(tmp_path):
    artifact = tmp_path / "existing.txt"
    artifact.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        run_smoke(tmp_path)
    assert artifact.read_text(encoding="utf-8") == "keep"
