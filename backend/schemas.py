"""Pydantic schemas for the binary CLEAN/TOXIC inference API."""
from typing import Dict, Literal

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value


class PredictionResponse(BaseModel):
    text: str
    label_id: Literal[0, 1]
    label_name: Literal["CLEAN", "TOXIC"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    is_calibrated: bool
    threshold: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float = Field(..., ge=0.0)
    model_backend: str

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, value: Dict[str, float]) -> Dict[str, float]:
        if set(value) != {"CLEAN", "TOXIC"}:
            raise ValueError("probabilities must contain exactly CLEAN and TOXIC")
        if any(not 0.0 <= probability <= 1.0 for probability in value.values()):
            raise ValueError("probabilities must be between 0 and 1")
        if abs(sum(value.values()) - 1.0) > 0.05:
            raise ValueError("probabilities must sum approximately to 1.0")
        return value


class ModelInfoResponse(BaseModel):
    model_name: str
    model_backend: str
    label_taxonomy: Dict[int, str]
    is_calibrated: bool
    calibration_method: str | None = None
    threshold: float = Field(..., ge=0.0, le=1.0)
    artifact_version: str | None = None
    status: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_backend: str
    artifact_status: str
    device: str
