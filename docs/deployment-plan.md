# Kế hoạch Triển khai & Ràng buộc Tài nguyên (Deployment Plan)

Trạng thái: **DRAFT — PENDING USER REVIEW**  
Ngày cập nhật: 2026-11-11  
Dự án: **Vietnamese Toxic Comment Detection**  
Kiến trúc: **ML-first, API-minimal** (FastAPI REST API, không frontend)

---

## 1. Kiến trúc Triển khai (System Architecture)

Chỉ bao gồm khối **Production Inference** tối giản:

```
+-----------------------------------------------------------------------+
|                          KHÁCH HÀNG / NGƯỜI DÙNG                      |
|           (Gọi API qua Swagger UI tại /docs, hoặc curl / HTTP client) |
+-----------------------------------+-----------------------------------+
                                    | HTTP
                                    v
+-----------------------------------------------------------------------+
|                     NỀN TẢNG TRIỂN KHAI (đề xuất)                     |
|  - Hugging Face Spaces (CPU Basic, 2 vCPU, 16 GB RAM)                 |
|  - Thay thế: Render, Railway, hoặc chạy FastAPI trực tiếp local       |
|  - Docker: **không bắt buộc**, chỉ dùng nếu hosting yêu cầu container |
+-----------------------------------+-----------------------------------+
                                    | HTTP
                                    v
+-----------------------------------------------------------------------+
|                  FASTAPI APPLICATION (Production Inference)           |
|                                                                       |
|  Endpoints:                                                           |
|    POST /predict   → văn bản → nhãn + xác suất hiệu chuẩn + latency   |
|    GET  /health    → model_loaded, backend, artifact, device          |
|    GET  /model-info → metadata, label taxonomy, calibration, version  |
|                                                                       |
|  Swagger UI: /docs                                                    |
+-----------------------------------+-----------------------------------+
                                    | In-memory call
                                    v
+-----------------------------------------------------------------------+
|               INFERENCE PIPELINE (.py, không phải Notebook)           |
|  - Best model hoặc ensemble đã chọn (từ M5)                           |
|  - Tokenizer / preprocessing artifacts                                |
|  - Label mapping (0:CLEAN, 1:OFFENSIVE, 2:HATE)                       |
|  - Probability contract (tổng ≈ 1.0, LinearSVC calibrated)            |
+-----------------------------------+-----------------------------------+
                                    | File load
                                    v
+-----------------------------------------------------------------------+
|           MODEL ARTIFACTS (models/exported/, models/baselines/)       |
|  - Tùy chọn: Hugging Face Model Hub (pinned revision)                 |
|  - Fallback: Baseline .joblib (<15MB) đóng gói sẵn                    |
+-----------------------------------------------------------------------+
```

**Không đưa vào production**: Notebook, training dataset, MLflow server, Optuna, training processes, unused checkpoints, frontend (React/Vite).

---

## 2. Ba Endpoint (API schema)

| Endpoint | Phương thức | Mô tả |
|---|---|---|
| `POST /predict` | POST | Nhận `text`, trả về `label_id`, `label_name`, `confidence`, `probabilities` (3 lớp, sum ≈ 1.0), `is_calibrated`, `latency_ms`, `model_backend`. |
| `GET /health` | GET | Trả về `status`, `model_loaded`, `model_backend`, `artifact_status`, `device`. |
| `GET /model-info` | GET | Trả về `model_name`, `model_backend`, `label_taxonomy`, `is_calibrated`, `calibration_method`, `artifact_version`, `status`. |

Schema chi tiết tại `backend/schemas.py` (Pydantic, probability contract).

---

## 3. Docker — tùy chọn, không bắt buộc

- `Dockerfile` và `docker-compose.yml` **không bắt buộc** trong phiên này.
- Giữ lại chỉ nếu nền tảng hosting yêu cầu container (ví dụ Hugging Face Spaces Docker SDK, Render Docker).
- FastAPI có thể chạy trực tiếp local qua Uvicorn mà không cần Docker.
- Kiến trúc API không đổi giữa trường hợp chạy local hay chạy trong container.

---

## 4. Xác minh Điều kiện Triển khai Hugging Face (M0 Status)

- **Trạng thái**: `docs/hf_eligibility_report.md` ghi nhận **`[ATTENTION / PENDING TOKEN]`** (chưa có `HF_TOKEN`, chưa xác minh quyền tạo Docker Space).
- **Quy tắc**: Không đánh dấu triển khai thành công khi chưa có bằng chứng thực tế về tài khoản và quyền tạo Space.
- **Phương án dự phòng**: Nếu HF yêu cầu thẻ tín dụng hoặc bị chặn, chuyển sang Render Free Docker, Railway, hoặc chạy container Docker local. Giữ nguyên kiến trúc FastAPI để porting dễ dàng.

---

## 5. Quản lý Tài nguyên (dựa trên đo đạc thực tế)

| Tham số | Ngưỡng dự kiến | Giải pháp kiểm soát |
|---|---|---|
| Dung lượng artifacts | PyTorch ~500 MB hoặc ONNX ~150 MB | Version hóa HF Model Hub, pinned tag/revision. |
| RAM | Quota 16 GB (HF Spaces CPU Basic) | Memory profiling thực tế khi nạp model; fallback sang baseline nếu vượt hạn mức. |
| Thời gian khởi động | Cần đo lường thực tế | Nạp model một lần qua FastAPI lifespan; không train/download nặng lúc runtime nếu đã có local cache. |
| Độ trễ suy luận | Cần benchmark thực tế trên CPU | Hỗ trợ đa backend; chỉ bật ONNX nếu benchmark chứng minh giảm latency ≥ 25%. |
| Chi phí hạ tầng | 0 VNĐ (miễn phí) | HF Spaces CPU Basic, hoặc Render/Railway miễn phí, hoặc local Docker. |

---

## 6. Thực hiện trong phiên này

- Đã xóa thư mục `frontend/` (React + Vite).
- Đã thiết lập MLflow local tracking tại `src/tracking/`.
- Docker không bắt buộc (chỉ giữ nếu hosting yêu cầu container).
- Ensemble/Stacking không bắt buộc (chỉ nếu có bằng chứng thực nghiệm).
- Không tạo `Dockerfile`, không code FastAPI `main.py` thực tế, không triển khai public deployment.
