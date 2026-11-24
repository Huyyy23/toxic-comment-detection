---
title: Vietnamese Toxic Comment Model Comparison
short_description: Demo nghiên cứu so sánh ba mô hình phân loại bình luận tiếng Việt.
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
python_version: 3.12
tags:
  - text-classification
  - vietnamese
  - zero-gpu
---

# N12 · Gradio ZeroGPU

**Trạng thái:** bundle local, chưa được phép public. Không upload trước khi xác minh quyền phân phối model/tokenizer và chốt eligibility tài khoản. Bộ này không chứa dataset, validation predictions, training logs, notebooks hay experiment artifacts.

FastAPI giữ các endpoint `/predict`, `/compare`, `/health`, `/model-info`, `/models` và Swagger tại `/docs`. Gradio là giao diện chính tại `/`. Cả hai gọi chung một inference registry.

## Mô hình và provenance

| Model | Checkpoint | Validation Macro-F1 | F1 TOXIC | Calibration |
|---|---|---:|---:|---|
| N6-LR-D | Official C=5, seed 42; seed 42/43/44 tied, chọn seed thấp nhất | 0.7901 | 0.6419 | Raw `predict_proba`, chưa calibration |
| V3-TRI-GATED-FT | V3 missing-weight replacement, retrained validation-only, seed 43 | 0.7930 | 0.6467 | Raw softmax, chưa calibration |
| V3-PHOBERT-V2 | V3 missing-weight replacement, retrained validation-only, seed 44 | 0.8195 | 0.6943 | Raw softmax, chưa calibration |

### Artifact mapping and SHA-256

- N6 source: `models/baselines/N6-LR-run-ea238c630a2f41b4b5be6b58edf4cdcd/EXP-N6-LR-D/seed_42/fallback_pipeline.joblib`; bundle: `models/n6_lr_d.joblib`; SHA-256: `2b693b43eb74ef062ab9e0b278032abd1cfd1a1f101754a1dd379a32729b81be`.
- TRI source: `models/retrained/v3/retrained_20261124/cnn/V3-TRI-GATED-FT/seed_43/best.pt`; bundle: `models/tri_best.pt`; SHA-256: `97cf7e7393b613a76310d0fc69a2e6f38091d08cb54a99de4e07357cab3376d8`.
- PhoBERT source: `models/retrained/v3/retrained_20261124/transformers/V3-PHOBERT-V2/seed_44/best.pt`; bundle: `models/phobert_best.pt`; SHA-256: `fd8db1d1c030abfae5e00ef0b0b825731dca5a0c99e9d3e504d4670ce6bf3b59`.
- `models/tri_vocab.json` is the matching frozen inference vocabulary; startup verifies its SHA-256 against the TRI checkpoint.

Metrics chỉ dùng validation trên cùng split N2. Không có test data hoặc test predictions trong bundle. Hai model V3 là checkpoint thay thế vì checkpoint lịch sử không còn; không ghi nhãn chúng là artifact lịch sử. Mọi xác suất đều chưa calibration.

PhoBERT fine-tuned checkpoint chứa backbone weights. Runtime chỉ tải `config` và tokenizer của revision `86cd7fd4c148980922ac11a2cf5e257f2ba639e1`; không tải thêm bộ pretrained weights 540 MB. PhoBERT tokenizer dùng PyVi. N6 dùng teencode; CNN dùng `v3_word_norm` và `v2_char_norm` cùng vocabulary đã đóng băng.

## Chạy local

Từ thư mục này, dùng Python 3.12 trong virtual environment riêng:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

`app.py` starts Uvicorn on port 7860; `app_port` is intentionally absent because that metadata key is Docker-only. Mở `http://127.0.0.1:7860` cho Gradio và `http://127.0.0.1:7860/docs` cho Swagger. Smoke test REST synthetic cho cả ba model và `/compare` đã pass; Swagger UI, luồng chọn model và so sánh ba model đã được mở trong browser local.

```bash
python smoke_test.py
```

PhoBERT cần kết nối Hub khi khởi động để lấy tokenizer/config. Inference không tải dữ liệu huấn luyện và không ghi bình luận vào file.

## Hugging Face Space

- Free ZeroGPU chỉ phù hợp khi tài khoản đủ điều kiện. HF yêu cầu email xác minh, tài khoản tối thiểu 30 ngày và good standing; tài khoản Free có tối đa 5 GPU-phút/ngày và 2 ZeroGPU Spaces. ZeroGPU dành cho Gradio SDK và yêu cầu PyTorch 2.8 trở lên. Quyền tạo Space, quota và khả năng gọi FastAPI custom routes trên ZeroGPU chưa được xác minh cho tài khoản này. Local smoke test không chứng minh hành vi hosted.

Trước khi public, hoàn tất các gate sau:

1. Xác minh tài khoản HF và quota ZeroGPU trong tài khoản của bạn.
2. Xác nhận quyền public của N6-LR-D, hai checkpoint V3 replacement, tokenizer/config PhoBERT và các thư viện liên quan. Quyền phân phối dataset nguồn chưa được xác minh trong repo.
3. Thu thập sign-off N10 đã tồn tại mà không mở lại N10 hoặc test set.
4. Tạo Space riêng tư để smoke test nếu tài khoản hỗ trợ ZeroGPU riêng tư. Nếu ZeroGPU buộc Space public, dừng cho tới khi các quyền ở trên được xác nhận.
5. Chọn SDK **Gradio**, Python 3.12, và hardware **ZeroGPU Free**. Không chọn Docker, GPU trả phí, hoặc bật nâng cấp phần cứng.
6. Chỉ sau khi các gate được đóng, đăng nhập CLI bằng write token cất trong credential store, rồi upload đúng thư mục này:

```bash
hf auth login
hf upload <username>/<space-name> . --repo-type space
```

Không upload thư mục `data/`, `models/` gốc, notebooks, logs, `artifacts/` hoặc repo đồ án đầy đủ. Bundle này chỉ chứa ba checkpoint đã chọn cùng config/vocabulary inference tối thiểu.

## API contract

`POST /predict` nhận `{"text":"..."}`; mặc định dùng N6-LR-D. Có thể gửi `model_id` là `n6_lr_d`, `v3_tri_gated_ft` hoặc `v3_phobert_v2`. `POST /compare` chạy cả ba. Mỗi kết quả trả CLEAN/TOXIC, xác suất raw, latency, validation metadata và calibration status. Nhãn giữ `CLEAN=0`, `TOXIC=1`.

Hugging Face mô tả ZeroGPU cho Gradio; route FastAPI custom được giữ theo yêu cầu REST nhưng chưa có bằng chứng hosted allocation. Hãy gọi `/predict` và `/compare` trực tiếp trên Space sau khi tạo private test Space trước khi dựa vào API.
