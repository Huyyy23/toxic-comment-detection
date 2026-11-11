# Báo cáo Quyết định Kiến trúc & Phân tích Đánh đổi (Architecture Decisions & Trade-offs)

Trạng thái: **APPROVED & SYNCHRONIZED**  
Ngày cập nhật: 2026-11-11  
Dự án: **Vietnamese Toxic Comment Detection System**

---

## ADR-01: Lựa chọn Giao diện & Kiến trúc Web (Docker với React + FastAPI)

### 1. Bối cảnh & Phê duyệt
Hệ thống cần phục vụ cả người dùng cuối và hội đồng nghiệm thu với giao diện chuyên nghiệp, phản hồi mượt mà, đồng thời có kiến trúc module chuẩn mực của một ứng dụng phần mềm thực tế.
- **Quyết định đã phê duyệt**: Triển khai theo mô hình **Tách biệt Frontend (React) và Backend (FastAPI)**, đóng gói chung trong **Docker container** chạy trên **Hugging Face Spaces**.

### 2. So sánh & Đánh giá

| Tiêu chí | Phương án: Docker (React + FastAPI) |
| :--- | :--- |
| **Kiến trúc** | Tách biệt hoàn toàn: Backend cung cấp REST API (`/api/predict`, `/api/predict-batch`, `/api/model-info`), Frontend React xử lý giao diện người dùng và tương tác trực quan. |
| **Trải nghiệm UI/UX** | Linh hoạt tối đa: Thiết kế giao diện hiện đại, responsive, biểu đồ tương tác, hiển thị trực quan các trạng thái loading/error/empty mà không bị giới hạn bởi widget layout của Streamlit. |
| **Đóng gói & Độc lập** | Multi-stage Dockerfile: Stage 1 build React (Vite/Node), Stage 2 nạp Python FastAPI và phục vụ file tĩnh của React. Đảm bảo tính nhất quán giữa môi trường local và cloud. |
| **Khả năng mở rộng** | Rất cao: API chuẩn RESTful có thể tái sử dụng cho extension trình duyệt, mobile app hoặc bot kiểm duyệt nội dung mạng xã hội. |

---

## ADR-02: Phân loại Nhãn & Mô hình Hóa (3-Class Taxonomy: CLEAN / OFFENSIVE / HATE)

### 1. Quyết định
- **Tập dữ liệu ưu tiên chính**: **UIT-ViHSD** (Hate Speech Detection) với không gian nhãn **3 lớp chuẩn**:
  - `0: CLEAN` (Lành mạnh / Không độc hại)
  - `1: OFFENSIVE` (Xúc phạm, thô tục nhưng chưa đến mức thù ghét cực đoan)
  - `2: HATE` (Thù ghét, kỳ thị, công kích định danh)
- **Quy tắc bắt buộc**: **Tuyệt đối KHÔNG tự động gộp nhãn về Binary**. Phải giữ nguyên 3 lớp để phản ánh đúng mức độ nghiêm trọng của ngôn ngữ mạng và tính khoa học của dataset.
- Cả hai nhóm mô hình (Traditional ML và Transformer) **bắt buộc dùng chung tập Test và bảng ánh xạ nhãn đồng nhất** để đảm bảo tính so sánh công bằng.

---

## ADR-03: Môi trường Huấn luyện Transformer (Local RTX 4060 8GB vs Cloud)

### 1. Quyết định
- **Môi trường huấn luyện chính (Primary)**: **Local GPU NVIDIA GeForce RTX 4060 8GB VRAM**.
  - Tối ưu hóa bộ nhớ VRAM: Sử dụng Mixed Precision (`fp16` hoặc `bf16`), Gradient Accumulation (steps = 2 hoặc 4) với Batch Size 16.
  - Ngân sách VRAM: Giữ mức tiêu thụ < 7.0 GB VRAM để đảm bảo hệ thống không bị crash.
- **Môi trường dự phòng (Fallback)**: Google Colab / Kaggle với GPU T4 miễn phí (chỉ kích hoạt nếu môi trường local gặp xung đột CUDA driver hoặc thiếu bộ nhớ).

---

## ADR-04: Kiến trúc Inference Engine Đa Backend (Multi-backend Interface)

### 1. Quyết định
- **Không ép buộc ONNX từ đầu**. Xây dựng interface `BasePredictor` trừu tượng với các backend cắm rút linh hoạt:
  - `SklearnBackend`: Cho mô hình truyền thống (TF-IDF + Logistic Regression / LinearSVC).
  - `PyTorchBackend`: Cho mô hình Transformer (`PhoBERT` / `ViSoBERT`) chạy trực tiếp trên PyTorch.
  - `ONNXBackend` (Tùy chọn nâng cao): Chỉ chuyển đổi và kích hoạt khi bài đo benchmark chứng minh được lợi ích thực tế về độ trễ (latency reduction) và tiết kiệm RAM trên CPU.
- Sau khi hoàn thành huấn luyện Transformer, phải có bước kiểm thử tích hợp (Integration Test) kiểm tra cả hai nhóm mô hình trước khi deploy.

---

## ADR-05: Nền tảng Triển khai & Kiểm tra Điều kiện Tiên quyết (Hugging Face Docker Space)

### 1. Quyết định
- **Mục tiêu**: Triển khai Docker Space công khai trên Hugging Face Spaces (miễn phí, cấp 2 vCPU, 16GB RAM).
- **Quy tắc an toàn**: Đưa bước xác minh điều kiện triển khai (Docker Space Eligibility Check) lên ngay **Milestone 0**:
  - Kiểm tra tài khoản Hugging Face có thể tạo Docker Space miễn phí hay bị yêu cầu thẻ/xác minh danh tính.
  - Nếu gặp blocker tài nguyên, ghi nhận vào `docs/plans/deployment.md` và kích hoạt phương án dự phòng (Render/Railway hoặc chuyển sang cấu hình Spaces hỗ trợ), không tự ý đổi nền tảng mà không báo cáo.
