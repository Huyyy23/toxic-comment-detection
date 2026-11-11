"""Smoke test for development environment and project scaffolding."""
import os
import sys
import pytest


def test_python_version():
    """Verify that Python version is at least 3.10."""
    assert sys.version_info >= (3, 10), "Python 3.10 or higher is required."


def test_project_structure():
    """Verify that all core project directories exist."""
    required_dirs = [
        "src",
        "src/data",
        "src/models",
        "src/evaluation",
        "src/inference",
        "src/tools",
        "backend",
        "backend/api",
        "tests",
        "configs",
        "data/raw",
        "data/processed",
        "models/baselines",
        "models/transformer_best",
        "models/exported",
        "docs",
    ]
    for rel_path in required_dirs:
        assert os.path.exists(rel_path), f"Required directory missing: {rel_path}"


def test_core_package_import():
    """Verify that src package can be imported."""
    import src
    assert src.__version__ == "1.0.0"


def test_configs_exist():
    """Verify that YAML configuration files exist and are readable."""
    baseline_cfg = os.path.join("configs", "baseline_config.yaml")
    transformer_cfg = os.path.join("configs", "transformer_config.yaml")
    assert os.path.exists(baseline_cfg), f"Missing config: {baseline_cfg}"
    assert os.path.exists(transformer_cfg), f"Missing config: {transformer_cfg}"

def test_schemas_and_probability_contract():
    """Verify that backend schemas enforce genuine calibrated probability contract."""
    pytest.importorskip("pydantic")
    from backend.schemas import PredictionResponse
    # Valid case
    res = PredictionResponse(
        text="Chào bạn",
        cleaned_text="Chào bạn",
        label_id=0,
        label_name="CLEAN",
        confidence=0.92,
        probabilities={"CLEAN": 0.92, "OFFENSIVE": 0.05, "HATE": 0.03},
        is_calibrated=True,
        latency_ms=12.5,
        model_backend="sklearn",
    )
    assert res.label_name == "CLEAN"
    assert res.is_calibrated is True

    # Invalid case: sum does not approximate 1.0 (fake confidence check)
    with pytest.raises(ValueError):
        PredictionResponse(
            text="Fake",
            cleaned_text="Fake",
            label_id=0,
            label_name="CLEAN",
            confidence=0.5,
            probabilities={"CLEAN": 0.5, "OFFENSIVE": 0.1, "HATE": 0.1}, # sum = 0.7 != 1.0
            is_calibrated=True,
            latency_ms=10.0,
            model_backend="sklearn",
        )
