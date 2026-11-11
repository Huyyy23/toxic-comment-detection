# Kế hoạch Triển khai Ứng dụng Web & Docker Container (Deployment Plan)

Trạng thái: **APPROVED & SYNCHRONIZED**  
Ngày cập nhật: 2026-11-11  
Dự án: **Vietnamese Toxic Comment Detection System**

---

## 1. KIẾN TRÚC TRIỂN KHAI (Deployment Architecture)

Ứng dụng được đóng gói thành một **Docker Container độc lập**, kết hợp **React (Frontend)** và **FastAPI (Backend)** chạy trên cùng một port `7860` (cổng tiêu chuẩn của Hugging Face Spaces):

```
+-----------------------------------------------------------------------+
|                            USER BROWSER                               |
|        (Giao diện React hiện đại: Single / Batch / Model Card)        |
+-----------------------------------+-----------------------------------+
                                    | HTTPS / Port 7860
                                    v
+-----------------------------------------------------------------------+
|                 HUGGING FACE SPACES (Docker SDK)                      |
|                  2 vCPU | 16 GB RAM (Dự kiến) | 50 GB Disk            |
|                                                                       |
|   +---------------------------------------------------------------+   |
|   |                  FastAPI Backend Server (Port 7860)           |   |
|   |  - API Endpoints (backend/schemas.py contract):               |   |
|   |      * POST /api/predict (Phân tích 1 câu, xác suất chuẩn)    |   |
|   |      * POST /api/predict-batch (Phân tích tệp CSV <= 500 dòng)|   |
|   |      * GET  /api/model-info (Thông tin model & calibration)   |   |
|   |      * GET  /api/health (Kiểm tra model_loaded & artifact)    |   |
|   |  - Static Files Mount:                                        |   |
|   |      * Mount thư mục /dist (React Build) phục vụ Single Page  |   |
|   +-------------------------------+-------------------------------+   |
|                                   | In-memory call                    |
|                                   v                                   |
|   +---------------------------------------------------------------+   |
|   |               Multi-Backend Inference Engine                  |   |
|   |  - src/inference/predictor.py (Singleton instance)            |   |
|   |  - Hỗ trợ: SklearnBackend (Calibrated) / PyTorch / ONNX       |   |
|   +-------------------------------+-------------------------------+   |
|                                   |                                   |
|                                   v                                   |
|   +---------------------------------------------------------------+   |
|   |         Model Artifact Delivery (src/tools/download_artifacts)|   |
|   |  - 1. Kiểm tra local models/exported/                         |   |
|   |  - 2. Tải từ HF Model Hub theo pinned revision (v1.0.0)       |   |
|   |  - 3. Fallback sang Baseline model joblib đóng gói sẵn        |   |
|   +---------------------------------------------------------------+   |
+-----------------------------------------------------------------------+
```

---

## 2. QUY TRÌNH XÁC MINH HUGGING FACE DOCKER SPACE ELIGIBILITY (M0)

- **Trạng thái Hiện tại**: **`[ATTENTION / PENDING TOKEN]`**.
  - Script `src/tools/check_hf_eligibility.py` đã chạy và ghi nhận rằng môi trường chưa được cấp `HF_TOKEN`.
  - **Không đánh dấu xác minh deployment thành công** khi chưa có bằng chứng thực tế về tài khoản và quyền tạo Docker Space.
- **Quy trình khi chuẩn bị Deploy (M6)**:
  1. Người dùng cung cấp `HF_TOKEN` (quyền write).
  2. Chạy lại `check_hf_eligibility.py` để xác nhận tài khoản không bị yêu cầu thẻ tín dụng hoặc hạn chế.
  3. Nếu tài khoản bị chặn xác thực thẻ tín dụng: ghi nhận blocker vào `docs/hf_eligibility_report.md` và kích hoạt phương án dự phòng (Render Free Docker hoặc chạy container local), **không tự ý đổi nền tảng**.

---

## 3. QUY TRÌNH GIAO NHẬN & VERSION HÓA MODEL ARTIFACTS TRONG DOCKER

Để đảm bảo container trên production thực sự có model và không crash vì thiếu file:
1. **Version hóa trên Hugging Face Model Hub**:
   - Model Transformer sau khi huấn luyện được đẩy lên repo weights riêng với tag cố định (ví dụ: `v1.0.0`).
2. **Kịch bản Khởi động (Startup Protocol)**:
   - Script `src/tools/download_artifacts.py` chạy lúc khởi động container (hoặc build-time).
   - Nếu tìm thấy file trong `models/exported/`: Khởi tạo `PyTorchBackend` hoặc `ONNXBackend`.
   - Nếu chưa có: Thực hiện `snapshot_download(repo_id=..., revision="v1.0.0")`.
   - Nếu mạng lỗi hoặc chưa có weights: Tự động fallback sang `models/baselines/baseline_model.joblib` đã được build trực tiếp vào Docker image.
3. **Giám sát Sức khỏe Container (`/api/health`)**:
   - Trả về JSON:
     ```json
     {
       "status": "healthy",
       "model_loaded": true,
       "model_backend": "pytorch",
       "artifact_status": "loaded_local",
       "device": "cpu"
     }
     ```

---

## 4. CHIẾN LƯỢC QUẢN LÝ TÀI NGUYÊN (Dựa trên Đo đạc Thực tế)

1. **Bộ nhớ RAM**:
   - Hugging Face Spaces cung cấp quota 16GB RAM cho CPU Basic.
   - Cần thực hiện **memory profiling** thực tế khi nạp model vào RAM ở giai đoạn smoke test, không đưa ra cam kết chủ quan.
2. **Thời gian Khởi động (Cold Start)**:
   - Sử dụng Lifespan context của FastAPI (`@asynccontextmanager`) để nạp model một lần duy nhất lúc server start.
   - Thời gian khởi động cần được đo lường thực tế trong bài kiểm thử smoke test.
3. **Cơ chế Fallback Mô hình**:
   - Nếu mô hình Transformer gặp sự cố (OOM hoặc thiếu file), `Predictor` tự động fallback sang `SklearnBackend` (<15MB) và gắn cờ `"model_backend": "sklearn_fallback"` để người dùng web nhận biết, đảm bảo ứng dụng không bao giờ bị trắng trang.
