"""Report artifacts stay faithful to measured validation results."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.evaluation.recurrent_report import compare_reports, load_report, render_report


@pytest.fixture
def report():
    metrics = {
        "loss": 0.7,
        "macro_f1": 1.0,
        "per_class": [
            {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}
            for _ in range(3)
        ],
        "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    return {
        "status": "SYNTHETIC_NOT_RESEARCH",
        "architecture": "lstm",
        "device": "cpu",
        "run_id": "fixture-run",
        "training": {
            "history": [{"epoch": 1, "train": deepcopy(metrics), "val": deepcopy(metrics)}],
            "best_epoch": 1,
            "best_macro_f1": 1.0,
        },
        "last_validation": {
            **deepcopy(metrics),
            "predictions": [
                {"id": f"val-{i}", "true_label": i, "predicted_label": i,
                 "probabilities_if_available": [float(j == i) for j in range(3)]}
                for i in range(3)
            ],
        },
    }


def save_report(tmp_path, report, name="smoke_results.json"):
    path = tmp_path / name
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_load_accepts_file_and_artifact_directory(tmp_path, report):
    path = save_report(tmp_path, report)
    assert load_report(path) == report
    assert load_report(tmp_path) == report


@pytest.mark.parametrize("identifiers", [[0, 1, 2], [1, "1", 0]])
def test_load_preserves_integer_and_mixed_ids(tmp_path, report, identifiers):
    for prediction, identifier in zip(report["last_validation"]["predictions"], identifiers):
        prediction["id"] = identifier
    loaded = load_report(save_report(tmp_path, report))
    assert [row["id"] for row in loaded["last_validation"]["predictions"]] == identifiers


@pytest.mark.parametrize("identifier", [True, False, 1.0, None, "", " ", "a b"])
def test_load_rejects_invalid_prediction_ids(tmp_path, report, identifier):
    report["last_validation"]["predictions"][0]["id"] = identifier
    with pytest.raises(ValueError, match="Prediction IDs"):
        load_report(save_report(tmp_path, report))


@pytest.mark.parametrize("identifier", [0, "same"])
def test_load_rejects_duplicate_typed_ids(tmp_path, report, identifier):
    for prediction in report["last_validation"]["predictions"][:2]:
        prediction["id"] = identifier
    with pytest.raises(ValueError, match="unique"):
        load_report(save_report(tmp_path, report))


@pytest.mark.parametrize("corruption", ["status", "nan", "matrix", "probabilities", "predictions", "epoch", "support", "run_id"])
def test_load_rejects_corrupt_evidence(tmp_path, report, corruption):
    if corruption == "status":
        report.pop("status")
    elif corruption == "nan":
        report["training"]["history"][0]["val"]["loss"] = float("nan")
    elif corruption == "matrix":
        report["last_validation"]["confusion_matrix"] = [[1]]
    elif corruption == "probabilities":
        report["last_validation"]["predictions"][0]["probabilities_if_available"] = [0.5] * 3
    elif corruption == "predictions":
        report["last_validation"]["predictions"][0]["predicted_label"] = 2
    elif corruption == "epoch":
        report["training"]["history"][0]["epoch"] = 0
    elif corruption == "run_id":
        report["last_validation"]["predictions"][0]["run_id"] = "different-run"
    else:
        report["last_validation"]["per_class"][0]["support"] = 5
    with pytest.raises(ValueError):
        load_report(save_report(tmp_path, report))


def test_load_rejects_invalid_json(tmp_path):
    path = tmp_path / "smoke_results.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        load_report(path)


def test_render_writes_real_pngs_and_closes_figures(tmp_path, report):
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    figures = render_report(report, tmp_path / "figures")
    assert set(figures) == {"loss_curves", "macro_f1", "confusion_matrix"}
    for name, path in figures.items():
        assert Path(path).name == f"{name}.png"
        assert Path(path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plt.get_fignums() == before


@pytest.mark.parametrize("missing", [True, False])
def test_no_epochs_means_no_plots(tmp_path, report, capsys, missing):
    if missing:
        report["training"].pop("history")
    else:
        report["training"]["history"] = []
    parsed = load_report(save_report(tmp_path, report))
    assert render_report(parsed, tmp_path / "figures") == {}
    assert "No epochs" in capsys.readouterr().out
    assert not (tmp_path / "figures").exists()


def test_comparison_preserves_status_and_distinguishes_last_from_best(tmp_path, report):
    report["last_validation"]["loss"] = 0.9
    path = save_report(tmp_path, report)
    official = deepcopy(report)
    official.update(status="OFFICIAL", architecture="bilstm", run_id="official-run")
    official_path = save_report(tmp_path, official, "official.json")
    rows = compare_reports([path, official_path])
    assert rows == [
        {"architecture": "lstm", "run_id": "fixture-run", "status": "SYNTHETIC_NOT_RESEARCH",
         "val_loss": 0.9, "val_macro_f1": 1.0, "best_epoch": 1, "best_macro_f1": 1.0},
        {"architecture": "bilstm", "run_id": "official-run", "status": "OFFICIAL",
         "val_loss": 0.9, "val_macro_f1": 1.0, "best_epoch": 1, "best_macro_f1": 1.0},
    ]


def test_comparison_does_not_invent_missing_metrics(tmp_path):
    report = {"status": "SYNTHETIC_NOT_RESEARCH", "architecture": "lstm", "training": {}}
    row = compare_reports([save_report(tmp_path, report)])[0]
    assert row["val_macro_f1"] is None
    assert row["best_macro_f1"] is None
    assert row["run_id"] is None
