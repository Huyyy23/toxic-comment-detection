"""Real local MLflow integration; all data and metrics are synthetic-only."""
import json

import mlflow
import pytest
from mlflow import MlflowClient

from src.models.smoke_recurrent import run_smoke


def test_real_tracking_persists_epoch_history_provenance_and_predictions(tmp_path):
    store = (tmp_path / "mlruns").as_uri()
    result = run_smoke(tmp_path / "output", architecture="bilstm_attention", tracking_uri=store)
    client = MlflowClient(tracking_uri=store)
    run = client.get_run(result["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.tags["result_type"] == "synthetic"
    assert run.data.tags["research_result"] == "false"
    assert run.data.params["architecture"] == "bilstm_attention"
    assert run.data.params["seed"] == "42"
    assert run.data.params["dataset_split_sha256"] == result["data_metadata"]["dataset_split_sha256"]
    assert run.data.tags["pretrained"] == "false"
    metrics = client.get_metric_history(result["run_id"], "val_macro_f1")
    assert [metric.step for metric in metrics] == [row["epoch"] for row in result["training"]["history"]]
    assert [metric.value for metric in metrics] == [row["val"]["macro_f1"] for row in result["training"]["history"]]
    artifacts = {item.path for item in client.list_artifacts(result["run_id"])}
    assert {"last.pt", "best.pt", "vocabulary.json", "smoke_results.json", "config.json",
            "data_metadata.json", "validation_predictions.json"} <= artifacts
    downloaded = client.download_artifacts(result["run_id"], "validation_predictions.json")
    predictions = json.loads(open(downloaded, encoding="utf-8").read())
    assert [row["id"] for row in predictions] == ["synthetic-v0", "synthetic-v1", "synthetic-v2"]
    assert all(row["run_id"] == result["run_id"] for row in predictions)
    assert mlflow.active_run() is None


def test_tracking_failure_marks_run_failed_and_propagates(tmp_path, monkeypatch):
    from src.models import smoke_recurrent

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic intentional failure")

    monkeypatch.setattr(smoke_recurrent, "fit", fail)
    store = (tmp_path / "mlruns").as_uri()
    with pytest.raises(RuntimeError, match="intentional failure"):
        run_smoke(tmp_path / "output", tracking_uri=store)
    client = MlflowClient(tracking_uri=store)
    experiment = client.get_experiment_by_name("EXP-03-synthetic-smoke")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1 and runs[0].info.status == "FAILED"
    assert mlflow.active_run() is None


@pytest.mark.parametrize("uri", ["https://example.invalid/mlflow", "file://server/share"])
def test_smoke_tracking_rejects_remote_store(tmp_path, uri):
    with pytest.raises(ValueError, match="local path or file URI"):
        run_smoke(tmp_path / "output", tracking_uri=uri)
