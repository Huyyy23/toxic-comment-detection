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
        "frontend",
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
