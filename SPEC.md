# Vietnamese Toxic Comment Detection System — Specification (SPEC.md)

Tài liệu đặc tả chi tiết và chuẩn mực: **[`docs/spec.md`](docs/spec.md)**  
Tài liệu quyết định kiến trúc: **[`docs/decisions.md`](docs/decisions.md)**  
Kế hoạch triển khai & phân rã: **[`tasks/plan.md`](tasks/plan.md)** và **[`tasks/todo.md`](tasks/todo.md)**

---

## Tóm tắt Quyết định Đã Phê duyệt (Approved Decisions)

1. **Dataset**: Tải cả `tarudesu/ViCTSD` và `ura-hcmut/UIT-ViHSD` cho EDA. Ưu tiên `UIT-ViHSD` làm tập huấn luyện chính thức (chốt sau EDA).
2. **Label Taxonomy**: Phân loại **3 lớp chuẩn**:
   - `0: CLEAN`
   - `1: OFFENSIVE`
   - `2: HATE`
   *(Tuyệt đối không tự động chuyển về Binary).*
3. **Mô hình Hóa**: Huấn luyện và đối sánh thực nghiệm khoa học giữa **Traditional ML** (TF-IDF + Linear Classifier) và **Transformer tiếng Việt** (`PhoBERT` / `ViSoBERT`) trên **cùng một tập test 3 lớp duy nhất**.
4. **Môi trường Huấn luyện**: Local GPU **NVIDIA GeForce RTX 4060 8GB** (Google Colab/Kaggle là backup).
5. **Inference Engine**: Interface `BasePredictor` hỗ trợ đa backend (`SklearnBackend`, `PyTorchBackend`, `ONNXBackend` tùy chọn). Chỉ convert ONNX khi benchmark chứng minh được lợi ích thực tế.
6. **Ứng dụng Web & Hosting**: Kiến trúc **Docker container kết hợp React (Frontend) + FastAPI (Backend)**, mục tiêu triển khai công khai trên **Hugging Face Spaces** (Docker SDK, port 7860). Bước xác minh điều kiện triển khai được thực hiện ngay tại Milestone 0.
