"""
Smoke tests cho TV1 N6 Binary Logistic Regression pipeline trên fixture Parquet nhị phân giả lập.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.lr_binary_pipeline import (
    load_lr_binary_artifacts,
    predict_lr_binary_from_artifacts,
    run_lr_binary_variant,
)
from tests.test_svm_binary_smoke import VAL_ROWS, _write_binary_fixture


class TestLrBinarySmoke:
    def test_variants_multi_seed_and_source_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ds_dir, aug_path = _write_binary_fixture(tmp)
            tracking_uri = (tmp / "mlruns").as_uri()

            for variant, prep_mode in [("A", "raw"), ("B", "basic"), ("C", "teencode"), ("D", "teencode")]:
                res = run_lr_binary_variant(
                    variant=variant,
                    dataset_dir=ds_dir,
                    augmentation_path=aug_path,
                    artifacts_root=tmp / "out",
                    tracking_uri=tracking_uri,
                    seeds=[42, 43, 44],
                    calibrate=False,
                    result_type="smoke",
                )
                agg = res["aggregate"]
                assert agg["n_seeds"] == 3
                assert 0.0 <= agg["macro_f1"]["mean"] <= 1.0
                assert set(agg["by_source"].keys()) == {"vihsd", "victsd"}

                first_seed_dir = Path(res["runs"][0]["seed_dir"])
                preds = json.loads((first_seed_dir / "val_predictions.json").read_text(encoding="utf-8"))
                assert len(preds) == len(VAL_ROWS)
                for p in preds:
                    assert abs((p["probabilities"]["CLEAN"] + p["probabilities"]["TOXIC"]) - 1.0) < 1e-6
                    assert p["is_calibrated"] is False
                    assert p["source"] in {"vihsd", "victsd"}

                err_text = (first_seed_dir / "error_summary.json").read_text(encoding="utf-8")
                assert "va0" not in err_text and "mày ngu quá" not in err_text

                loaded = load_lr_binary_artifacts(first_seed_dir)
                re_preds = predict_lr_binary_from_artifacts(
                    loaded, [r["text"] for r in VAL_ROWS], preprocessing_mode=prep_mode
                )
                assert re_preds == [p["predicted_label"] for p in preds]

    def test_calibrated_threshold_and_ece_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ds_dir, aug_path = _write_binary_fixture(tmp)
            res = run_lr_binary_variant(
                variant="C",
                dataset_dir=ds_dir,
                augmentation_path=aug_path,
                artifacts_root=tmp / "out",
                tracking_uri=(tmp / "mlruns").as_uri(),
                seeds=[42],
                calibrate=True,
                result_type="smoke",
            )
            cal_diag = res["runs"][0]["calibration_diagnostics"]
            assert cal_diag is not None
            assert 0.0 <= cal_diag["ece"] <= 1.0
            assert 0.10 <= cal_diag["val_selected_threshold"]["threshold"] <= 0.90
            seed_dir = Path(res["runs"][0]["seed_dir"])
            preds = json.loads((seed_dir / "val_predictions.json").read_text(encoding="utf-8"))
            assert all(
                p["predicted_label"] == int(p["probabilities"]["TOXIC"] >= 0.5)
                for p in preds
            )
            loaded = load_lr_binary_artifacts(seed_dir)
            re_preds = predict_lr_binary_from_artifacts(
                loaded, [r["text"] for r in VAL_ROWS], preprocessing_mode="teencode"
            )
            assert re_preds == [p["predicted_label"] for p in preds]
            assert res["runs"][0]["fallback_size_mb"] <= 15.0
