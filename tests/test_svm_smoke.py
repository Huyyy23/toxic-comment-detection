"""
Smoke tests cho TV2 EXP-02 (SVM pipeline) trên dữ liệu synthetic theo Data Contract v1.

Chạy: pytest tests/test_svm_smoke.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.models import svm_pipeline
from src.evaluation.svm_report import build_error_handover
from src.models.svm_pipeline import (
    load_svm_artifacts,
    predict_from_artifacts,
    run_experiment,
)

SYNTHETIC_TRAIN = [
    {"id": "tr_0", "text": "bình luận bình thường rất hữu ích", "label": 0, "split": "train"},
    {"id": "tr_1", "text": "nội dung sạch sẽ tích cực", "label": 0, "split": "train"},
    {"id": "tr_2", "text": "cảm ơn bạn đã chia sẻ thông tin", "label": 0, "split": "train"},
    {"id": "tr_3", "text": "bài viết hay quá trời", "label": 0, "split": "train"},
    {"id": "tr_4", "text": "mày ngu vcl ko biết gì", "label": 1, "split": "train"},
    {"id": "tr_5", "text": "đồ điên khùng nói nhảm đm", "label": 1, "split": "train"},
    {"id": "tr_6", "text": "con này xấu xí vô duyên quá", "label": 1, "split": "train"},
    {"id": "tr_7", "text": "óc chó thật sự clgt", "label": 1, "split": "train"},
    {"id": "tr_8", "text": "giết hết lũ phản động này đi", "label": 2, "split": "train"},
    {"id": "tr_9", "text": "bọn này đáng bị tiêu diệt sạch", "label": 2, "split": "train"},
    {"id": "tr_10", "text": "tẩy chay đuổi cổ lũ rác rưởi", "label": 2, "split": "train"},
    {"id": "tr_11", "text": "căm thù lũ súc vật vùng này", "label": 2, "split": "train"},
]

SYNTHETIC_VAL = [
    {"id": "va_0", "text": "bài viết bình thường dễ hiểu", "label": 0, "split": "val"},
    {"id": "va_1", "text": "mày ngu thế vcl", "label": 1, "split": "val"},
    {"id": "va_2", "text": "tiêu diệt hết bọn chúng đi", "label": 2, "split": "val"},
]

ALL_RECORDS = SYNTHETIC_TRAIN + SYNTHETIC_VAL
MANIFEST = {r["id"]: r["split"] for r in ALL_RECORDS}


def _prepare_fixture(tmpdir: Path, records: list[dict] = ALL_RECORDS) -> tuple[Path, Path]:
    data_path = tmpdir / "train_val.jsonl"
    manifest_path = tmpdir / "manifest.json"
    with open(data_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    manifest = {r["id"]: r["split"] for r in records}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return data_path, manifest_path


class TestSvmPipelineSmoke:
    def test_variants_abc_uncalibrated_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_path, manifest_path = _prepare_fixture(tmp)
            tracking_uri = (tmp / "mlruns").as_uri()

            for variant in ["A", "B", "C"]:
                res = run_experiment(
                    variant=variant,
                    data_path=data_path,
                    manifest_path=manifest_path,
                    artifacts_dir=tmp / "models",
                    tracking_uri=tracking_uri,
                    calibrate=False,
                    result_type="smoke",
                )
                assert 0.0 <= res["metrics"]["macro_f1"] <= 1.0
                assert res["run_id"]

                variant_dir = Path(res["artifacts_dir"])
                assert (variant_dir / "svm_model.joblib").exists()
                assert (variant_dir / "tfidf.joblib").exists()
                assert (variant_dir / "config.json").exists()
                assert (variant_dir / "metrics.json").exists()

                preds = json.loads((variant_dir / "val_predictions.json").read_text(encoding="utf-8"))
                assert len(preds) == len(SYNTHETIC_VAL)
                assert [p["id"] for p in preds] == [r["id"] for r in SYNTHETIC_VAL]
                for p in preds:
                    assert p["run_id"] == res["run_id"]
                    assert p["predicted_label"] in (0, 1, 2)
                    # Không được biến decision_function thành xác suất
                    assert p["probabilities"] is None

                # Kiểm tra reload artifact cho ra kết quả dự đoán nhất quán
                loaded = load_svm_artifacts(variant_dir)
                re_preds = predict_from_artifacts(loaded, [r["text"] for r in SYNTHETIC_VAL])
                assert re_preds == [p["predicted_label"] for p in preds]

    def test_calibrated_probabilities_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_path, manifest_path = _prepare_fixture(tmp)
            res = run_experiment(
                variant="A",
                data_path=data_path,
                manifest_path=manifest_path,
                artifacts_dir=tmp / "models",
                tracking_uri=(tmp / "mlruns").as_uri(),
                calibrate=True,
                result_type="smoke",
            )
            preds = json.loads(
                (Path(res["artifacts_dir"]) / "val_predictions.json").read_text(encoding="utf-8")
            )
            for p in preds:
                proba = p["probabilities"]
                assert isinstance(proba, dict)
                assert set(proba.keys()) == {"CLEAN", "OFFENSIVE", "HATE"}
                assert abs(sum(proba.values()) - 1.0) < 1e-4

    def test_tuning_mode_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_path, manifest_path = _prepare_fixture(tmp)
            res = run_experiment(
                variant="C",
                data_path=data_path,
                manifest_path=manifest_path,
                artifacts_dir=tmp / "models",
                tracking_uri=(tmp / "mlruns").as_uri(),
                tune=True,
                result_type="smoke",
            )
            tuning_path = Path(res["artifacts_dir"]) / "tuning_results.json"
            assert tuning_path.exists()
            history = json.loads(tuning_path.read_text(encoding="utf-8"))
            assert len(history) > 1

    def test_rejects_test_split_in_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bad_records = ALL_RECORDS + [
                {"id": "te_0", "text": "dữ liệu test bị rò rỉ", "label": 0, "split": "test"}
            ]
            data_path, manifest_path = _prepare_fixture(tmp, bad_records)
            with pytest.raises(ValueError, match="split must be train or val"):
                run_experiment(
                    variant="A",
                    data_path=data_path,
                    manifest_path=manifest_path,
                    artifacts_dir=tmp / "models",
                    tracking_uri=(tmp / "mlruns").as_uri(),
                    result_type="smoke",
                )
    def test_repeated_runs_preserve_distinct_outputs_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_path, manifest_path = _prepare_fixture(tmp)
            kwargs = {
                "variant": "A",
                "data_path": data_path,
                "manifest_path": manifest_path,
                "artifacts_dir": tmp / "models",
                "tracking_uri": (tmp / "mlruns").as_uri(),
                "calibrate": False,
                "result_type": "smoke",
            }

            first = run_experiment(**kwargs)
            first_dir = Path(first["artifacts_dir"])
            first_run_id = (first_dir / "run_id.txt").read_text(encoding="utf-8")
            second = run_experiment(**kwargs)
            second_dir = Path(second["artifacts_dir"])

            assert first_dir != second_dir
            assert (first_dir / "run_id.txt").read_text(encoding="utf-8") == first_run_id
            assert first_run_id != second["run_id"]
            for key in ("macro_f1", "weighted_f1", "balanced_accuracy", "per_class", "confusion_matrix"):
                assert first["metrics"][key] == second["metrics"][key]

    def test_error_handover_excludes_comment_text_and_record_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variant_dir = Path(tmpdir) / "EXP-02-C"
            variant_dir.mkdir()
            private_text = "unique comment text that must not be serialized"
            record_id = "source-record-987"
            (variant_dir / "val_predictions.json").write_text(
                json.dumps(
                    [
                        {
                            "id": record_id,
                            "true_label": 1,
                            "predicted_label": 2,
                            "run_id": "run-123",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            handover = build_error_handover(
                variant_dir,
                [{"id": record_id, "text": private_text, "label": 1, "split": "val"}],
            )
            serialized = json.dumps(handover, ensure_ascii=False)

            assert private_text not in serialized
            assert record_id not in serialized
            assert "representative_samples_by_pair" not in handover
            assert handover["n_errors_total"] == 1
            assert handover["confusion_pair_counts"] == {"OFFENSIVE -> HATE": 1}
            assert handover["pattern_tag_counts_among_errors"]["offensive_hate_boundary"] == 1

    def test_cli_all_runs_use_unique_output_groups(self, monkeypatch, tmp_path):
        args = SimpleNamespace(
            variant="all",
            config=svm_pipeline.DEFAULT_CONFIG_PATH,
            data_path=None,
            manifest_path=None,
            artifacts_dir=tmp_path,
            tracking_uri=None,
            calibrate=False,
            tune=False,
            output_name=None,
            C=None,
        )
        monkeypatch.setattr(svm_pipeline, "parse_args", lambda: args)

        def fake_run_experiment(**kwargs):
            output_dir = Path(kwargs["artifacts_dir"]) / kwargs["artifact_name"]
            output_dir.mkdir(parents=True)
            return {
                "artifacts_dir": str(output_dir),
                "run_id": output_dir.name,
                "metrics": {
                    "macro_f1": 0.5,
                    "train_time_s": 0.1,
                    "model_size_mb": 1.0,
                },
            }

        monkeypatch.setattr(svm_pipeline, "run_experiment", fake_run_experiment)
        svm_pipeline.main()
        first_group = next(tmp_path.glob("EXP-02-run-*"))
        first_summary = (first_group / "EXP-02-summary.json").read_bytes()

        svm_pipeline.main()
        groups = list(tmp_path.glob("EXP-02-run-*"))

        assert len(groups) == 2
        assert (first_group / "EXP-02-summary.json").read_bytes() == first_summary
        assert all(
            len(json.loads((group / "EXP-02-summary.json").read_text(encoding="utf-8"))) == 3
            for group in groups
        )
