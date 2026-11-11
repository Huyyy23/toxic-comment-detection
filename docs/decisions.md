# Báo cáo Quyết định Kiến trúc & Phân tích Đánh đổi (Architecture Decisions & Trade-offs)

Trạng thái: **APPROVED & SYNCHRONIZED**  
Ngày cập nhật: 2026-11-11  
Dự án: **Vietnamese Toxic Comment Detection System**

---

## ADR-01: Lựa chọn Giao diện & Kiến trúc Web (Docker với React + FastAPI)

- **Quyết định**: Triển khai theo mô hình tách biệt Frontend (React) và Backend (FastAPI), đóng gói chung trong Multi-stage Docker container chạy trên Hugging Face Spaces.
- **Lý do**: Đảm bảo phân tách trách nhiệm (Separation of Concerns), cung cấp REST API chuẩn mực (`backend/schemas.py`), cho phép giao diện người dùng hiển thị trực quan các phân bố xác suất và biểu đồ mà không bị phụ thuộc vào widget của Streamlit.

---

## ADR-02: Phân loại Nhãn & Giao thức Đánh giá Khoa học

- **Không gian Nhãn**: **3 lớp CLEAN (0) / OFFENSIVE (1) / HATE (2)**. Tuyệt đối không tự động gộp về Binary.
- **Nguyên tắc Khóa Tập Test (Locked Test Set Protocol)**:
  - **Tập Train (70%)**: Huấn luyện trọng số mô hình.
  - **Tập Validation (15%)**: Điều chỉnh siêu tham số (hyperparameters), early stopping, và lựa chọn mô hình (model selection).
  - **Tập Test (15%)**: **Bị khóa (Locked/Sealed)** trong suốt quá trình phát triển. Chỉ mở ra đúng một lần để đối sánh khách quan giữa Best Baseline và Best Transformer. Tuyệt đối không dùng kết quả test set để tối ưu ngược lại mô hình.
- **Tiêu chí So sánh Khách quan**: Không đặt giả định hay yêu cầu Transformer bắt buộc phải vượt Baseline; mục tiêu khoa học là ghi nhận trung thực toàn bộ chỉ số (Macro-F1, Per-class F1, Latency, Resource footprint) và phân tích các đánh đổi (trade-offs).

---

## ADR-03: Hợp đồng Xác suất & Hiệu chuẩn Mô hình (Probability Calibration)

- **Vấn đề**: `LogisticRegression` tự nhiên hỗ trợ `.predict_proba()`. Tuy nhiên, `LinearSVC` chỉ sinh ra khoảng cách siêu phẳng (uncalibrated margins qua `decision_function()`), không phải xác suất và không cộng lại bằng 1.0.
- **Quyết định**:
  - Không tự viết hàm softmax tùy tiện trên decision scores để giả tạo confidence.
  - Khi sử dụng LinearSVC, **bắt buộc bọc bằng `CalibratedClassifierCV(estimator=LinearSVC(...), cv=5, method='sigmoid')`** (Platt scaling) trên tập huấn luyện để tạo ra xác suất hiệu chuẩn thực sự.
  - Hợp đồng API bắt buộc: $\sum P(\text{class}) \approx 1.0$, `confidence` là xác suất của lớp chiến thắng.

---

## ADR-04: Quản lý Siêu tham số Chưa kiểm chứng (`[PROVISIONAL]`)

- **Quyết định**: Mọi giá trị tham số cấu hình hiện tại trong `configs/` (như `max_length = 160`, `batch_size = 16`, `learning_rate = 2e-5`, `VRAM budget < 7.0GB`, TF-IDF `max_features = 10000`) đều được đánh dấu là **`[PROVISIONAL / TẠM THỜI]`**.
- Các giá trị này chỉ được quyết định chính thức sau khi có kết quả phân tích phân vị 95% độ dài chuỗi ở EDA (M1) và kiểm tra bộ nhớ VRAM thực tế trên GPU RTX 4060.

---

## ADR-05: Chiến lược Tách biệt Kiểm thử Tích hợp (Decoupled Integration Tests)

- **Quyết định**: Tách kiểm thử tích hợp thành 2 bước độc lập:
  1. `T4.2: Baseline Integration Test`: Kiểm thử tích hợp Baseline model với `BasePredictor` và API sớm, unblock việc phát triển Web React + FastAPI (M5) ngay lập tức.
  2. `T4.3: Transformer Integration Test`: Sau khi huấn luyện Transformer trên RTX 4060 (M3), kiểm thử tích hợp checkpoint vào `PyTorchBackend` và so sánh trực tiếp với Baseline.
  3. `T4.4: Conditional ONNX Benchmark`: Đo đạc benchmark CPU PyTorch vs ONNX INT8, chỉ chuyển đổi khi chứng minh được giảm latency $\ge 25\%$.

---

## ADR-06: Quy trình Giao nhận & Version hóa Artifacts cho Production Docker

- **Bối cảnh**: Không thể lưu checkpoint nặng hàng trăm MB trên Git repository thông thường. Nếu container khởi động mà không có model weights thì ứng dụng sẽ bị crash.
- **Quyết định**:
  - Xây dựng công cụ `src/tools/download_artifacts.py` kiểm tra artifact theo thứ tự: Local -> Tải từ Hugging Face Hub (pinned revision `v1.0.0`) -> Fallback sang Baseline model `.joblib` đóng gói sẵn (<15MB).
  - Đảm bảo container trên Hugging Face Spaces **luôn có model hoạt động**, không bị sập container ngay cả khi chưa tải xong Transformer.
  - Bỏ các cam kết chủ quan chưa có benchmark như "an toàn tuyệt đối về RAM" hay "health check dưới 10 giây". Thay bằng cơ chế kiểm tra trạng thái thực tế và ghi nhận log đo đạc tài nguyên.
