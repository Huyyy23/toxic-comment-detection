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
|                  2 vCPU | 16 GB RAM | 50 GB Disk                      |
|                                                                       |
|   +---------------------------------------------------------------+   |
|   |                  FastAPI Backend Server (Port 7860)           |   |
|   |  - API Endpoints:                                             |   |
|   |      * POST /api/predict (Phân tích 1 câu)                    |   |
|   |      * POST /api/predict-batch (Phân tích tệp CSV)            |   |
|   |      * GET  /api/model-info (Thông tin & metrics mô hình)     |   |
|   |      * GET  /api/health (Health check)                        |   |
|   |  - Static Files Mount:                                        |   |
|   |      * Mount thư mục /dist (React Build) phục vụ Single Page  |   |
|   +-------------------------------+-------------------------------+   |
|                                   | In-memory call                    |
|                                   v                                   |
|   +---------------------------------------------------------------+   |
|   |               Multi-Backend Inference Engine                  |   |
|   |  - src/inference/predictor.py (Singleton instance)            |   |
|   |  - Hỗ trợ: SklearnBackend (Baseline) / PyTorchBackend / ONNX  |   |
|   +-------------------------------+-------------------------------+   |
|                                   |                                   |
|                                   v                                   |
|   +---------------------------------------------------------------+   |
|   |                   Model Checkpoint & Configs                  |   |
|   |  - models/exported/ (PyTorch weights hoặc ONNX)               |   |
|   |  - models/baselines/ (Fallback joblib model)                  |   |
|   +---------------------------------------------------------------+   |
+-----------------------------------------------------------------------+
```

---

## 2. QUY TRÌNH XÁC MINH HUGGING FACE DOCKER SPACE ELIGIBILITY (M0)

Để tránh rủi ro chuẩn bị toàn bộ code nhưng tài khoản không thể deploy Docker Space:

### Checklist Xác minh Điều kiện tại M0:
1. **Kiểm tra Tài khoản Hugging Face**:
   - Tài khoản đã được tạo và kích hoạt email.
   - Kiểm tra xem tài khoản có bị giới hạn (flag/suspension) hay không.
2. **Kiểm tra Quyền tạo Docker Space**:
   - Thử nghiệm tạo một Blank Docker Space qua Web UI hoặc Python API (`huggingface_hub.HfApi().create_repo(..., repo_type="space", space_sdk="docker")`).
   - Xác nhận Hugging Face có yêu cầu thêm thẻ tín dụng (Credit Card verification) cho Docker Space hay cho phép tạo miễn phí trực tiếp.
3. **Xử lý nếu Không Đủ Điều kiện (Blocker & Fallback Strategy)**:
   - **Ghi nhận Blocker**: Báo cáo rõ tài khoản bị vướng xác thực thẻ hoặc chính sách mới của HF.
   - **Phương án dự phòng 1**: Chuyển cấu hình Space SDK sang `static` + FastAPI backend host trên Render Free Docker Web Service (hoặc Fly.io).
   - **Phương án dự phòng 2**: Chạy Docker container cục bộ (`docker compose up`) để hội đồng chấm điểm kiểm tra trực tiếp.
   - *Nguyên tắc*: Không tự ý chuyển nền tảng khi chưa thông báo và nhận phê duyệt từ người dùng.

---

## 3. THIẾT KẾ MULTI-STAGE DOCKERFILE

Để tối ưu hóa kích thước image và tốc độ khởi động:

```dockerfile
# STAGE 1: Build React Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# STAGE 2: Python Backend & Static Serving
FROM python:3.10-slim AS runner
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

# Cài đặt thư viện hệ thống tối thiểu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Python production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code backend và ML engine
COPY backend/ ./backend/
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/exported/ ./models/exported/

# Copy React build artifacts từ Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Tạo user không có quyền root (theo khuyến nghị của HF Spaces)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## 4. CHIẾN LƯỢC QUẢN LÝ TÀI NGUYÊN & COLD START

1. **Bộ nhớ RAM**:
   - Hugging Face Spaces cung cấp 16GB RAM cho CPU Basic.
   - Mô hình Transformer PyTorch FP32 khi nạp chiếm ~1.5GB RAM.
   - Mô hình ONNX INT8 khi nạp chiếm ~500MB RAM.
   - Hệ thống vận hành an toàn tuyệt đối, không lo nguy cơ OOM.
2. **Khởi động Nhanh (Fast Cold-Start)**:
   - Sử dụng cơ chế Lazy Load hoặc Async Lifespan Event của FastAPI (`@asynccontextmanager`) để khởi tạo model ngay khi server boot, trả về 200 OK cho `/api/health` trong vòng 10 giây.
3. **Cơ chế Fallback Mô hình**:
   - Nếu mô hình Transformer gặp sự cố lúc nạp, `Predictor` tự động fallback sang `SklearnBackend` (TF-IDF Baseline < 15MB) và gắn cờ `"model_backend": "sklearn_fallback"` trong response để giao diện hiển thị cảnh báo nhẹ.
