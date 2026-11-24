"""Local FastAPI inference application."""
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import HealthResponse, ModelInfoResponse, PredictionRequest, PredictionResponse
from src.inference.adapter import InferenceAdapter
from src.inference.joblib_baseline import JoblibBaselineAdapter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    ROOT
    / "models/runtime/n11/EXP-N6-LR-D/seed_42/fallback_pipeline.joblib"
)


def create_app(adapter: InferenceAdapter | None = None) -> FastAPI:
    if adapter is None:
        model_path = Path(os.getenv("N11_MODEL_PATH", DEFAULT_MODEL_PATH))
        if model_path.is_file():
            adapter = JoblibBaselineAdapter(model_path)

    app = FastAPI(title="Vietnamese Toxic Comment Detection API", version="1.0.0")
    app.state.inference_adapter = adapter
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        current_adapter = app.state.inference_adapter
        if current_adapter is None:
            raise HTTPException(status_code=503, detail="Inference model is not loaded")
        return PredictionResponse(text=request.text, **current_adapter.predict(request.text))

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        current_adapter = app.state.inference_adapter
        return HealthResponse(
            status="healthy" if current_adapter else "unavailable",
            model_loaded=current_adapter is not None,
            model_backend=current_adapter.model_backend if current_adapter else "none",
            artifact_status="loaded" if current_adapter else "missing",
            device="cpu",
        )

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        current_adapter = app.state.inference_adapter
        if current_adapter is None:
            raise HTTPException(status_code=503, detail="Inference model is not loaded")
        return ModelInfoResponse(**current_adapter.info())

    return app


app = create_app()
