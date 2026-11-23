import pytest

import json
from pathlib import Path

import pandas as pd
from gensim.models import FastText
from mlflow import MlflowClient

from src.models.n5_fasttext import run_experiment


def test_n5_single_seed_outputs_and_train_only_vocabulary(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    pd.DataFrame(
        [
            {"id": "tr0", "text": "clean sunny day", "label": 0, "source": "synthetic", "split": "train"},
            {"id": "tr1", "text": "kind bright sunny", "label": 0, "source": "synthetic", "split": "train"},
            {"id": "tr2", "text": "toxic angry words", "label": 1, "source": "synthetic", "split": "train"},
            {"id": "tr3", "text": "angry harsh words", "label": 1, "source": "synthetic", "split": "train"},
        ]
    ).to_parquet(train_path, index=False)
    pd.DataFrame(
        [
            {"id": "va0", "text": "clean sunny day", "label": 0, "source": "synthetic", "split": "val"},
            {"id": "va1", "text": "toxic angry words", "label": 1, "source": "synthetic", "split": "val"},
            {"id": "va2", "text": "validationonly sunny", "label": 0, "source": "synthetic", "split": "val"},
            {"id": "va3", "text": "validationonly angry", "label": 1, "source": "synthetic", "split": "val"},
        ]
    ).to_parquet(val_path, index=False)

    tracking_uri = (tmp_path / "mlruns").as_uri()
    result = run_experiment(train_path, val_path, tmp_path / "output", tracking_uri, seeds=(42,))
    seed_dir = Path(result["seeds"][0]["seed_dir"])
    assert {"config.json", "dataset_hashes.json", "fasttext.model", "logistic_regression.joblib",
            "val_predictions.csv", "metrics.json", "run_id.txt"} <= {path.name for path in seed_dir.iterdir()}
    assert {"seed-42"} <= {path.name for path in seed_dir.parent.iterdir()}
    config = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))
    assert config["seed"] == 42 and config["vector_size"] == 100 and config["epochs"] == 20
    model = FastText.load(str(seed_dir / "fasttext.model"))
    assert "validationonly" not in model.wv.key_to_index
    assert model.wv["validationonly"].shape == (100,)
    predictions = pd.read_csv(seed_dir / "val_predictions.csv")
    assert predictions["id"].tolist() == ["va0", "va1", "va2", "va3"]
    assert predictions["predicted_label"].isin([0, 1]).all()
    metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {"macro_f1", "f1_clean", "f1_toxic", "training_duration_seconds"}
    assert (seed_dir.parent / "summary.json").is_file()
    run_id = (seed_dir / "run_id.txt").read_text(encoding="utf-8").strip()
    client = MlflowClient(tracking_uri=tracking_uri)
    tracked = client.get_run(run_id)
    assert tracked.info.status == "FINISHED"
    assert tracked.data.metrics["macro_f1"] == metrics["macro_f1"]

def test_n5_rejects_rows_without_a_train_split(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    pd.DataFrame(
        [
            {"id": "tr0", "text": "clean words", "label": 0, "split": None},
            {"id": "tr1", "text": "toxic words", "label": 1, "split": "train"},
        ]
    ).to_parquet(train_path, index=False)
    pd.DataFrame(
        [
            {"id": "va0", "text": "clean words", "label": 0, "split": "val"},
            {"id": "va1", "text": "toxic words", "label": 1, "split": "val"},
        ]
    ).to_parquet(val_path, index=False)

    with pytest.raises(ValueError, match="expected only split"):
        run_experiment(train_path, val_path, tmp_path / "output", seeds=(42,))
