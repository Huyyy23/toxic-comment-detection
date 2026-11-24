from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import gradio as gr
import spaces
import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from model_runtime import InferenceService, ModelId, MODEL_INFO

ROOT = Path(__file__).resolve().parent
SERVICE = InferenceService(ROOT / "models")


class TextPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1000)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value


class PredictionRequest(TextPayload):
    model_id: ModelId = "n6_lr_d"


class ModelSelection(BaseModel):
    model_id: ModelId

class PredictionResponse(BaseModel):
    text: str
    model_id: ModelId
    model_name: str
    label_id: Literal[0, 1]
    label_name: Literal["CLEAN", "TOXIC"]
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    is_calibrated: bool
    calibration_status: str
    threshold: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    model_backend: str
    validation_macro_f1: float = Field(ge=0.0, le=1.0)
    validation_f1_toxic: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_probability_contract(self) -> PredictionResponse:
        if set(self.probabilities) != {"CLEAN", "TOXIC"}:
            raise ValueError("probabilities must contain exactly CLEAN and TOXIC")
        if any(not 0.0 <= value <= 1.0 for value in self.probabilities.values()):
            raise ValueError("probabilities must be between 0 and 1")
        if abs(sum(self.probabilities.values()) - 1.0) > 0.05:
            raise ValueError("probabilities must sum approximately to 1.0")
        if self.confidence != self.probabilities[self.label_name]:
            raise ValueError("confidence must match the predicted label probability")
        return self


class ModelInfoResponse(BaseModel):
    model_id: ModelId
    model_name: str
    model_backend: str
    label_taxonomy: dict[int, str]
    is_calibrated: bool
    calibration_status: str
    calibration_method: str | None
    threshold: float
    validation_macro_f1: float
    validation_f1_toxic: float
    artifact_version: str
    provenance: str
    status: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_backend: str
    artifact_status: str
    device: str


@spaces.GPU
def run_neural_models(model_ids: list[ModelId], text: str) -> list[dict[str, object]]:
    return SERVICE.predict_neural(model_ids, text)


def predict_one(model_id: ModelId, text: str) -> dict[str, object]:
    if model_id == "n6_lr_d":
        return SERVICE.predict_baseline(text)
    return run_neural_models([model_id], text)[0]


def compare_all(text: str) -> list[dict[str, object]]:
    baseline = SERVICE.predict_baseline(text)
    return [baseline, *run_neural_models(["v3_tri_gated_ft", "v3_phobert_v2"], text)]


api = FastAPI(title="Vietnamese Toxic Comment Detection API", version="2.0.0")


@api.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    return PredictionResponse(text=request.text, **predict_one(request.model_id, request.text))


@api.post("/compare", response_model=list[PredictionResponse])
def compare(request: TextPayload) -> list[PredictionResponse]:
    return [PredictionResponse(text=request.text, **result) for result in compare_all(request.text)]


@api.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_backend="multi-model",
        artifact_status="loaded",
        device="gpu-on-demand",
    )


@api.get("/model-info", response_model=ModelInfoResponse)
def model_info(model_id: ModelId = Query(default="n6_lr_d")) -> ModelInfoResponse:
    return ModelInfoResponse(**SERVICE.model_info(model_id))


@api.get("/models", response_model=list[ModelInfoResponse])
def models() -> list[ModelInfoResponse]:
    return [ModelInfoResponse(**SERVICE.model_info(model_id)) for model_id in MODEL_INFO]


CSS = """
:root { --n12-ink: #10243a; --n12-muted: #607387; --n12-accent: #087e8b; --n12-border: #dce5ec; }
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
.n12-hero { padding: 1.5rem 1.6rem; border: 1px solid var(--n12-border); border-radius: 18px; background: linear-gradient(120deg, #f1f7fb 0%, #f8fbfc 72%); }
.n12-hero h1 { color: var(--n12-ink); margin: 0 0 .45rem; letter-spacing: -.03em; }
.n12-hero p { color: var(--n12-muted); margin: .3rem 0; }
.n12-kicker { color: var(--n12-accent) !important; font-size: .78rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
.n12-section { color: var(--n12-ink); }
.n12-footnote { color: var(--n12-muted); font-size: .9rem; }
footer { display: none !important; }
"""


def model_card(model_id: str) -> str:
    selected = ModelSelection(model_id=model_id).model_id
    info = SERVICE.model_info(selected)
    return (
        f"**{info['model_name']}**  \n"
        f"{info['model_backend']} · Macro-F1 validation {info['validation_macro_f1']:.4f} · "
        f"F1 TOXIC validation {info['validation_f1_toxic']:.4f}  \n"
        f"Calibration: **{info['calibration_status']}**.  \n"
        f"{info['provenance']}"
    )


def ui_predict(text: str, model_id: str):
    request = PredictionRequest(text=text, model_id=model_id)
    result = predict_one(request.model_id, request.text)
    probabilities = result["probabilities"]
    detail = (
        f"**Nhãn:** {result['label_name']} · **Độ tin cậy thô:** {result['confidence']:.1%}  \n"
        f"**Thời gian suy luận:** {result['latency_ms']:.2f} ms · **Hiệu chuẩn:** {result['calibration_status']}  \n"
        f"**Mô hình:** {result['model_name']} · **Validation Macro-F1:** {result['validation_macro_f1']:.4f}"
    )
    return probabilities, detail


def ui_compare(text: str):
    request = TextPayload(text=text)
    results = compare_all(request.text)
    rows = [
        [
            result["model_name"],
            result["label_name"],
            result["probabilities"]["CLEAN"],
            result["probabilities"]["TOXIC"],
            result["latency_ms"],
            result["validation_macro_f1"],
            result["calibration_status"],
        ]
        for result in results
    ]
    return rows


with gr.Blocks(
    title="Phân tích bình luận tiếng Việt",
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="blue", neutral_hue="slate"),
    css=CSS,
) as demo:
    gr.HTML(
        """<header class="n12-hero"><p class="n12-kicker">N12 · DEMO NGHIÊN CỨU</p>
        <h1>Phân tích bình luận tiếng Việt</h1>
        <p>So sánh mô hình truyền thống, deep learning tự huấn luyện và PhoBERT trên cùng đầu vào.</p></header>"""
    )
    gr.Markdown(
        "**Lưu ý:** Xác suất của cả ba mô hình chưa được calibration. Kết quả phục vụ demo học thuật, không thay thế quyết định kiểm duyệt của con người.",
        elem_classes="n12-footnote",
    )
    with gr.Tabs():
        with gr.Tab("Một mô hình"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="Bình luận cần phân tích",
                        placeholder="Nhập bình luận tiếng Việt, tối đa 1.000 ký tự",
                        lines=5,
                        max_lines=8,
                        max_length=1000,
                    )
                    with gr.Row():
                        model_choice = gr.Dropdown(
                            choices=[(str(info["model_name"]), model_id) for model_id, info in MODEL_INFO.items()],
                            value="n6_lr_d",
                            label="Mô hình",
                        )
                        predict_button = gr.Button("Phân tích", variant="primary")
                    model_details = gr.Markdown(model_card("n6_lr_d"))
                with gr.Column(scale=2):
                    label_output = gr.Label(label="Kết quả · CLEAN / TOXIC", num_top_classes=2)
                    result_details = gr.Markdown()
            predict_button.click(ui_predict, inputs=[text_input, model_choice], outputs=[label_output, result_details])
            text_input.submit(ui_predict, inputs=[text_input, model_choice], outputs=[label_output, result_details])
            model_choice.change(model_card, inputs=model_choice, outputs=model_details)
        with gr.Tab("So sánh ba mô hình"):
            compare_input = gr.Textbox(
                label="Bình luận cần so sánh",
                placeholder="Nhập một bình luận để chạy cả ba mô hình",
                lines=4,
                max_lines=8,
                max_length=1000,
            )
            compare_button = gr.Button("Chạy cả ba mô hình", variant="primary")
            compare_output = gr.Dataframe(
                headers=["Mô hình", "Nhãn", "P(CLEAN)", "P(TOXIC)", "ms", "Val Macro-F1", "Calibration"],
                datatype=["str", "str", "number", "number", "number", "number", "str"],
                interactive=False,
                label="Kết quả đối chiếu",
            )
            compare_button.click(ui_compare, inputs=compare_input, outputs=compare_output)
            compare_input.submit(ui_compare, inputs=compare_input, outputs=compare_output)
    gr.Markdown(
        "Nhãn nhị phân: **CLEAN = 0**, **TOXIC = 1**. Thời gian gồm preprocessing và inference tại môi trường hiện hành.",
        elem_classes="n12-footnote",
    )


demo.queue(default_concurrency_limit=1, max_size=20)
app = gr.mount_gradio_app(api, demo, path="/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
