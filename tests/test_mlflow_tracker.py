"""Real MLflow 2 file-store regression tests; no tracking server required."""

from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from src.tracking import mlflow_tracker as tracking
from src.tracking.mlflow_tracker import MLflowTracker


@pytest.fixture(autouse=True)
def restore_mlflow_state(monkeypatch):
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    monkeypatch.setattr(tracking, "_default_tracker", None)
    yield
    if mlflow.active_run() is not None:
        mlflow.end_run(status="KILLED")
    mlflow.set_tracking_uri(previous_uri)


@pytest.mark.parametrize("file_uri", [False, True])
def test_local_run_params_metric_history_artifacts_and_status(tmp_path, file_uri):
    store = tmp_path / "tracking store"
    uri = store.as_uri() if file_uri else str(store)
    tracker = MLflowTracker("real-local", uri)
    assert tracker.tracking_uri == store.as_uri()
    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    artifact = tmp_path / "summary.json"
    artifact.write_text('{"source": "test"}', encoding="utf-8")
    directory = tmp_path / "report"
    directory.mkdir()
    (directory / "nested").mkdir()
    (directory / "nested" / "scores.txt").write_text("0.5", encoding="utf-8")

    with tracker.run(run_name="local-run", tags={"result_type": "synthetic"}):
        run_id = tracker.run_id
        tracker.log_params({"seed": 7, "architecture": "lstm"})
        tracker.log_metrics({"val_macro_f1": 0.25}, step=1)
        tracker.log_metrics({"val_macro_f1": 0.5}, step=2)
        tracker.log_artifact(str(artifact), "metadata")
        tracker.log_artifact(str(directory), "reports")
        assert tracker.log_training_duration() >= 0

    result = client.get_run(run_id)
    assert result.info.experiment_id == tracker.experiment_id
    assert result.info.status == "FINISHED"
    assert result.data.params == {"seed": "7", "architecture": "lstm"}
    assert result.data.tags["result_type"] == "synthetic"
    history = client.get_metric_history(run_id, "val_macro_f1")
    assert [(metric.step, metric.value) for metric in history] == [(1, 0.25), (2, 0.5)]
    downloaded = client.download_artifacts(run_id, "metadata/summary.json")
    assert Path(downloaded).read_text(encoding="utf-8") == artifact.read_text(encoding="utf-8")
    downloaded = client.download_artifacts(run_id, "reports/nested/scores.txt")
    assert Path(downloaded).read_text(encoding="utf-8") == "0.5"
    assert tracker.run_id is None
    assert mlflow.active_run() is None


def test_exception_propagates_and_marks_run_failed(tmp_path):
    tracker = MLflowTracker("failed-local", str(tmp_path / "mlruns"))
    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    with pytest.raises(ValueError, match="training failed"):
        with tracker.run():
            run_id = tracker.run_id
            tracker.log_metrics({"loss": 1.0}, step=0)
            raise ValueError("training failed")
    assert client.get_run(run_id).info.status == "FAILED"
    assert tracker.run_id is None
    assert mlflow.active_run() is None


def test_explicit_experiment_prevents_global_experiment_contamination(tmp_path):
    uri = str(tmp_path / "mlruns")
    first = MLflowTracker("first", uri)
    second = MLflowTracker("second", uri)
    client = MlflowClient(tracking_uri=first.tracking_uri)
    mlflow.set_tracking_uri(first.tracking_uri)
    mlflow.set_experiment("second")
    for tracker in (first, second, first):
        with tracker.run():
            run_id = tracker.run_id
            tracker.log_params({"owner": tracker.experiment_id})
        result = client.get_run(run_id)
        assert result.info.experiment_id == tracker.experiment_id
        assert result.data.params["owner"] == tracker.experiment_id
    assert len(client.search_runs([first.experiment_id])) == 2
    assert len(client.search_runs([second.experiment_id])) == 1


def test_two_stores_restore_uri_and_reject_overlapping_runs(tmp_path):
    first = MLflowTracker("first", str(tmp_path / "first"))
    with first.run():
        first_id = first.run_id
        second = MLflowTracker("second", str(tmp_path / "second"))
        with pytest.raises(RuntimeError, match="Another MLflow run"):
            second.start_run()
        assert mlflow.active_run().info.run_id == first_id
        first.log_params({"owner": "first"})
        mlflow.set_tracking_uri(second.tracking_uri)
        first.log_metrics({"score": 0.4}, step=1)
        mlflow.set_tracking_uri(second.tracking_uri)
    with second.run():
        second_id = second.run_id
        mlflow.set_tracking_uri(first.tracking_uri)
        second.log_params({"owner": "second"})
        mlflow.set_tracking_uri(first.tracking_uri)
        second.log_metrics({"score": 0.7}, step=2)
    for tracker, run_id, owner, score in (
        (first, first_id, "first", 0.4), (second, second_id, "second", 0.7)
    ):
        client = MlflowClient(tracking_uri=tracker.tracking_uri)
        result = client.get_run(run_id)
        assert result.info.status == "FINISHED"
        assert result.data.params == {"owner": owner}
        assert result.data.metrics == {"score": score}
        assert [run.info.run_id for run in client.search_runs([tracker.experiment_id])] == [run_id]


def test_wrappers_preserve_foreign_run_and_accept_failed_status(tmp_path, monkeypatch):
    tracker = MLflowTracker("wrappers", str(tmp_path / "mlruns"))
    monkeypatch.setattr(tracking, "_default_tracker", tracker)
    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    mlflow.set_tracking_uri(tracker.tracking_uri)
    with mlflow.start_run(experiment_id=tracker.experiment_id) as foreign:
        with pytest.raises(RuntimeError, match="Another MLflow run"):
            tracking.start_run()
        tracking.end_run()
        assert mlflow.active_run().info.run_id == foreign.info.run_id
        with pytest.raises(RuntimeError, match="No active run"):
            tracking.log_metrics({"wrong_owner": 1.0})
    run_id = tracking.start_run(tags={"result_type": "synthetic"})
    tracking.log_params({"seed": 7})
    tracking.log_metrics({"loss": 0.2}, step=3)
    artifact = tmp_path / "note.txt"
    artifact.write_text("wrapper artifact", encoding="utf-8")
    tracking.log_artifact(str(artifact))
    tracking.end_run(status="FAILED")
    assert client.get_run(run_id).info.status == "FAILED"
    assert client.get_run(run_id).data.params["seed"] == "7"
    assert [item.path for item in client.list_artifacts(run_id)] == ["note.txt"]
