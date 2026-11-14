"""CPU synthetic-only integration of group-one data and scratch training."""

import pytest
import torch

from src.models.smoke_recurrent import run_smoke


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
