"""Pydantic schemas and strict probability contract for Toxic Comment Detection API."""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Schema for single comment classification request."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Vietnamese comment text to classify",
        example="Bài viết này rất hữu ích, cảm ơn bạn!",
    )


class PredictionResponse(BaseModel):
    """Schema for single comment classification response with calibrated probability contract."""
    text: str = Field(..., description="Original input text")
    cleaned_text: str = Field(..., description="Preprocessed text (Unicode NFC, cleaned teencode)")
    label_id: int = Field(..., ge=0, le=2, description="Class ID: 0 (CLEAN), 1 (OFFENSIVE), 2 (HATE)")
    label_name: str = Field(..., description="Human-readable label name: CLEAN | OFFENSIVE | HATE")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated probability of the winning class (range 0.0 - 1.0)",
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description=(
            "Calibrated probabilities for all 3 classes (summing to ~1.0). "
            "For LogisticRegression and Transformer, derived from softmax/predict_proba. "
            "For LinearSVC, derived via CalibratedClassifierCV (Platt scaling). "
            "Never mocked or uncalibrated raw decision function scores."
        ),
    )
    is_calibrated: bool = Field(
        default=True,
        description="Indicates whether probabilities are genuinely calibrated",
    )
    latency_ms: float = Field(..., ge=0.0, description="Inference latency in milliseconds")
    model_backend: str = Field(..., description="Inference backend used: sklearn | pytorch | onnx")

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities_sum(cls, v: Dict[str, float]) -> Dict[str, float]:
        required_keys = {"CLEAN", "OFFENSIVE", "HATE"}
        if not required_keys.issubset(v.keys()):
            raise ValueError(f"Probabilities must contain keys: {required_keys}")
        total = sum(v[k] for k in required_keys)
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"Sum of probabilities must be approximately 1.0, got {total}")
        return v


class BatchPredictionRequest(BaseModel):
    """Schema for batch comment analysis."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of comment texts (maximum 500 items per batch)",
    )


class BatchPredictionResponse(BaseModel):
    """Schema for batch comment analysis results."""
    total_processed: int
    summary_counts: Dict[str, int]
    results: List[PredictionResponse]
    total_latency_ms: float


class ModelInfoResponse(BaseModel):
    """Schema for model metadata and operational card."""
    model_name: str
    model_backend: str
    label_taxonomy: Dict[int, str]
    is_calibrated: bool
    calibration_method: Optional[str] = None
    artifact_version: Optional[str] = None
    status: str


class HealthResponse(BaseModel):
    """Schema for container health check."""
    status: str = Field(..., example="healthy")
    model_loaded: bool
    model_backend: str
    artifact_status: str
    device: str
