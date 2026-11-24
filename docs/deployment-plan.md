# Kế hoạch triển khai N12 trên Hugging Face Gradio ZeroGPU

## Trạng thái

Bundle inference riêng tại `deployment/hf-gradio/` đã được tạo và kiểm thử local. Chưa tạo Space, chưa upload, chưa deploy. Public launch vẫn bị chặn bởi eligibility tài khoản, quyền phân phối artifacts, bằng chứng N10 hiện có và kiểm chứng remote cho FastAPI REST trên ZeroGPU.

## Kiến trúc và contract

- Gradio là UI tiếng Việt tại `/`; FastAPI cùng process phục vụ `/predict`, `/compare`, `/health`, `/model-info`, `/models` và Swagger tại `/docs`.
- Uvicorn bind port 7860. Metadata không dùng `app_port` vì khóa đó dành cho Docker SDK.
- N6 chạy trên CPU. CNN và PhoBERT dùng `@spaces.GPU` khi hosted, với khóa inference chung để serialize model GPU.
- API trả `CLEAN=0`, `TOXIC=1`, probabilities sau calibration nếu model vượt activation gate, latency và status theo từng model. Chỉ TRI đang áp dụng Temperature Scaling; N6 và PhoBERT trả xác suất gốc vì không đạt gate.

## Candidate và artifact

| Model | Candidate | Validation Macro-F1 | F1 TOXIC | Calibration | Ghi chú |
|---|---|---:|---:|---|---|
| N6-LR-D | Official C=5, seed 42 | 0.7901 | 0.6419 | Platt fit, không áp dụng | Seed 42/43/44 tie; chọn seed thấp nhất. |
| V3-TRI-GATED-FT | V3 replacement, seed 43 | 0.7930 | 0.6467 | Temperature `T=0.871645`, áp dụng | Historical weight thiếu; chỉ dùng checkpoint validation-only hiện có. |
| V3-PHOBERT-V2 | V3 replacement, seed 44 | 0.8195 | 0.6943 | Temperature fit, không áp dụng | Historical weight thiếu; ECE holdout tăng. |

Đường dẫn gốc, bản sao và SHA-256 của ba checkpoint cùng validation provenance được ghi tại `deployment/hf-gradio/README.md`. Bundle kiểm tra SHA lúc khởi động và xác minh vocabulary với checkpoint TRI. Bản sao deployment giữ nguyên SHA của artifact gốc. Không mở lại N10, N15 hoặc test.

## Kiểm thử local đã có bằng chứng

- `smoke_test.py` passed with synthetic text on a CPU-only local profile with GPUs hidden. It covered REST, Swagger endpoint, all three model IDs, `/compare`, labels, probabilities, latency and calibration status.
- The browser verified single-model inference with TRI calibration applied, three-model comparison and the Swagger UI at `/docs`.
- A GPU-visible local run failed while returning PhoBERT to CPU because memory allocation failed. CPU-only smoke passed afterward; neither result establishes hosted behavior.

## Tối ưu có giới hạn

- Checkpoint PhoBERT đã chứa backbone fine-tuned; runtime tải config/tokenizer ở revision cố định thay vì tải thêm base-model weights.
- Ba checkpoint cần thiết và CNN vocabulary được copy vào bundle riêng. Không có dataset, predictions, logs, notebook hoặc training artifacts.
- Models neural ở CPU khi idle; lần lượt chuyển một model sang device khi inference rồi trả về CPU. Không lượng tử hóa hay đổi kiến trúc để tránh suy giảm chất lượng.
- Free ZeroGPU có quota tối đa 5 GPU-phút/ngày theo tài liệu HF; thời gian demo kéo dài có thể chạm quota. Chưa đo hiệu năng hosted.

## Rào cản trước public deploy

1. Tài khoản chưa xác minh được: thiếu `HF_TOKEN`, email/account age/good standing/quota đều chưa biết.
2. Quyền public của model checkpoints, PhoBERT tokenizer/config và dữ liệu nguồn chưa được xác minh độc lập.
3. Bằng chứng sign-off N10 hiện có chưa được đối chiếu; không mở lại N10 để bổ sung.
4. Local FastAPI REST hoạt động, nhưng chưa chứng minh direct REST requests được cấp ZeroGPU trên Space.
5. Free ZeroGPU private-Space eligibility chưa xác định. Nếu test yêu cầu public Space, dừng cho tới khi rights được xác nhận.

## Upload sau khi các gate đóng

1. Tạo Space mới, SDK Gradio, Python 3.12, hardware ZeroGPU Free. Không chọn Docker hoặc hardware trả phí.
2. Nếu private ZeroGPU được hỗ trợ, upload vào Space private và kiểm tra `/`, `/docs`, `/health`, `/predict`, `/compare`. Nếu không, chưa upload.
3. Tại `deployment/hf-gradio/`, chạy `hf auth login` rồi `hf upload <username>/<space-name> . --repo-type space`.
4. Chỉ upload nội dung của bundle này. Không upload `data/`, model directory gốc, notebooks, logs, `artifacts/` hay toàn repo.

Nguồn nền tảng: [Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview), [ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu), [Spaces config](https://huggingface.co/docs/hub/en/spaces-config-reference), [Gradio mount](https://www.gradio.app/docs/gradio/mount_gradio_app).
