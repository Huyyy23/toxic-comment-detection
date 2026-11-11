"""Tool for downloading, verifying, and versioning model artifacts for production Docker runtime."""
import os
import sys
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def ensure_model_artifacts(
    hub_repo_id: Optional[str] = None,
    revision: str = "main",
    target_dir: str = "models/exported",
    fallback_dir: str = "models/baselines",
) -> dict:
    """Ensure that valid model artifacts exist for the production container.
    
    Order of precedence:
    1. Local exported artifacts already present in target_dir.
    2. Download from Hugging Face Model Hub using pinned revision.
    3. Fallback to bundled traditional baseline model in fallback_dir.
    """
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(fallback_dir, exist_ok=True)

    status = {
        "active_backend": None,
        "artifact_path": None,
        "is_fallback": False,
        "version": revision,
        "message": "",
    }

    # 1. Check local transformer/onnx artifact in target_dir
    onnx_file = os.path.join(target_dir, "model.onnx")
    pytorch_file = os.path.join(target_dir, "pytorch_model.bin")
    safetensors_file = os.path.join(target_dir, "model.safetensors")

    if os.path.exists(onnx_file):
        status["active_backend"] = "onnx"
        status["artifact_path"] = onnx_file
        status["message"] = "Found local ONNX artifact."
        logger.info(status["message"])
        return status

    if os.path.exists(pytorch_file) or os.path.exists(safetensors_file):
        status["active_backend"] = "pytorch"
        status["artifact_path"] = pytorch_file if os.path.exists(pytorch_file) else safetensors_file
        status["message"] = "Found local PyTorch/Safetensors artifact."
        logger.info(status["message"])
        return status

    # 2. Attempt Hugging Face Hub download if hub_repo_id is configured
    if hub_repo_id:
        try:
            from huggingface_hub import snapshot_download
            logger.info("Downloading pinned model artifacts from Hub: %s (revision: %s)", hub_repo_id, revision)
            downloaded_dir = snapshot_download(
                repo_id=hub_repo_id,
                revision=revision,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
            )
            status["active_backend"] = "pytorch"
            status["artifact_path"] = downloaded_dir
            status["message"] = f"Successfully downloaded artifacts from {hub_repo_id}@{revision}"
            logger.info(status["message"])
            return status
        except Exception as e:
            logger.warning("Failed to download from HF Hub (%s): %s", hub_repo_id, str(e))

    # 3. Fallback to bundled traditional baseline model
    baseline_file = os.path.join(fallback_dir, "baseline_model.joblib")
    if os.path.exists(baseline_file):
        status["active_backend"] = "sklearn"
        status["artifact_path"] = baseline_file
        status["is_fallback"] = True
        status["message"] = "Using bundled Baseline TF-IDF model as reliable fallback."
        logger.warning(status["message"])
        return status

    status["message"] = (
        "No local artifacts found, Hub download unavailable, and no bundled baseline found. "
        "Container requires either trained weights in models/ or an active internet connection."
    )
    logger.error(status["message"])
    return status


if __name__ == "__main__":
    hub_repo = os.environ.get("HF_MODEL_REPO_ID")
    hub_rev = os.environ.get("HF_MODEL_REVISION", "v1.0.0")
    result = ensure_model_artifacts(hub_repo_id=hub_repo, revision=hub_rev)
    print(f"Artifact status: {result['message']}")
    if not result["artifact_path"]:
        sys.exit(1)
