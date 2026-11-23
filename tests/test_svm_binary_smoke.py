"""
Smoke tests cho TV2 N6 Binary SVM pipeline trên fixture Parquet nhị phân giả lập.

Chạy: .venv/Scripts/python.exe -m pytest tests/test_svm_binary_smoke.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from src.models.svm_binary_pipeline import (
    load_binary_svm_artifacts,
    predict_binary_from_artifacts,
    run_binary_variant,
)

TRAIN_ROWS = [
    {"id": "tr0", "text": "bình luận tích cực rất hay và bổ ích", "label": 0, "source": "vihsd", "split": "train"},
    {"id": "tr1", "text": "cảm ơn bài viết chia sẻ rất chi tiết", "label": 0, "source": "victsd", "split": "train"},
    {"id": "tr2", "text": "nội dung bình thường sạch sẽ dễ hiểu", "label": 0, "source": "vihsd", "split": "train"},
    {"id": "tr3", "text": "thông tin chuẩn xác quá tuyệt vời", "label": 0, "source": "victsd", "split": "train"},
    {"id": "tr4", "text": "thích bài viết này rất nhiều ạ", "label": 0, "source": "vihsd", "split": "train"},
    {"id": "tr5", "text": "đồ ngu vcl cút đi lũ óc chó", "label": 1, "source": "vihsd", "split": "train"},
    {"id": "tr6", "text": "mày điên khùng nói nhảm đm", "label": 1, "source": "victsd", "split": "train"},
    {"id": "tr7", "text": "giết hết lũ rác rưởi phản động", "label": 1, "source": "vihsd", "split": "train"},
    {"id": "tr8", "text": "con này xấu xí vô duyên clgt", "label": 1, "source": "victsd", "split": "train"},
    {"id": "tr9", "text": "tiêu diệt sạch bọn súc vật này", "label": 1, "source": "vihsd", "split": "train"},
]

VAL_ROWS = [
    {"id": "va0", "text": "bài viết hay và rất dễ hiểu", "label": 0, "source": "vihsd", "split": "val"},
    {"id": "va1", "text": "cảm ơn bạn đã chia sẻ nội dung", "label": 0, "source": "victsd", "split": "val"},
    {"id": "va2", "text": "mày ngu quá vcl cút đi", "label": 1, "source": "vihsd", "split": "val"},
    {"id": "va3", "text": "bọn rác rưởi đáng bị tiêu diệt", "label": 1, "source": "victsd", "split": "val"},
]

AUG_ROWS = [
    {"id": "aug0", "text": "do ngu vcl cut di lu oc cho", "label": 1, "split": "train", "is_augmented": True, "parent_id": "tr5"},
    {"id": "aug1", "text": "giet het lu rac ruoi phan dong", "label": 1, "split": "train", "is_augmented": True, "parent_id": "tr7"},
]


def _write_binary_fixture(
    tmp: Path,
    train_rows: list[dict] = TRAIN_ROWS,
    val_rows: list[dict] = VAL_ROWS,
    aug_rows: list[dict] = AUG_ROWS,
) -> tuple[Path, Path]:
    ds_dir = tmp / "binary_n2"
    ds_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(ds_dir / "train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(ds_dir / "val.parquet", index=False)
    manifest = {r["id"]: {"id": r["id"], "split": r["split"]} for r in train_rows + val_rows}
    (ds_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    aug_path = tmp / "augmentation.parquet"
    pd.DataFrame(aug_rows).to_parquet(aug_path, index=False)
    return ds_dir, aug_path


class TestSvmBinarySmoke:
    def test_variants_multi_seed_and_source_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ds_dir, aug_path = _write_binary_fixture(tmp)
            tracking_uri = (tmp / "mlruns").as_uri()

            for variant in ["A", "B", "C", "D"]:
                res = run_binary_variant(
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
                    assert p["probabilities"] is None
                    assert p["is_calibrated"] is False
                    assert p["source"] in {"vihsd", "victsd"}

                # Bảo vệ riêng tư: error_summary không lưu text gốc hay ID bản ghi
                err_text = (first_seed_dir / "error_summary.json").read_text(encoding="utf-8")
                assert "va0" not in err_text and "mày ngu quá" not in err_text

                # Reload artifact dự đoán lại khớp 100%
                loaded = load_binary_svm_artifacts(first_seed_dir)
                re_preds = predict_binary_from_artifacts(loaded, [r["text"] for r in VAL_ROWS])
                assert re_preds == [p["predicted_label"] for p in preds]

    def test_calibrated_threshold_and_ece_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ds_dir, aug_path = _write_binary_fixture(tmp)
            res = run_binary_variant(
                variant="C",
                dataset_dir=ds_dir,
                augmentation_path=aug_path,
                artifacts_root=tmp / "out",
                tracking_uri=(tmp / "mlruns").as_uri(),
                seeds=[42],
                calibrate=True,
                tune=True,
                result_type="smoke",
            )
            seed_dir = Path(res["runs"][0]["seed_dir"])
            metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
            assert "calibration_diagnostics" in metrics
            assert 0.0 <= metrics["calibration_diagnostics"]["ece"] <= 1.0
            assert metrics["within_15mb_fallback_limit"] is True

            preds = json.loads((seed_dir / "val_predictions.json").read_text(encoding="utf-8"))
            for p in preds:
                assert p["is_calibrated"] is True
                assert set(p["probabilities"].keys()) == {"CLEAN", "TOXIC"}
                assert abs(sum(p["probabilities"].values()) - 1.0) < 1e-4

            loaded = load_binary_svm_artifacts(seed_dir)
            re_preds = predict_binary_from_artifacts(loaded, [r["text"] for r in VAL_ROWS])
            assert re_preds == [p["predicted_label"] for p in preds]

    def test_rejects_test_split_or_val_overlap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bad_val = VAL_ROWS + [
                {"id": "te0", "text": "dữ liệu test lọt vào", "label": 0, "source": "vihsd", "split": "test"}
            ]
            ds_dir, aug_path = _write_binary_fixture(tmp, val_rows=bad_val)
            with pytest.raises(ValueError, match="split='val'"):
                run_binary_variant(
                    variant="A",
                    dataset_dir=ds_dir,
                    augmentation_path=aug_path,
                    artifacts_root=tmp / "out",
                    tracking_uri=(tmp / "mlruns").as_uri(),
                    seeds=[42],
                    result_type="smoke",
                )
