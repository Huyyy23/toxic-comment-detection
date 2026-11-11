# Danh sách Nhiệm vụ Dự án (Project Task List)
## Vietnamese Toxic Comment Detection (3-Class: React + FastAPI in Docker)

Kế hoạch tham chiếu: [`tasks/plan.md`](plan.md) và [`docs/plans/task-breakdown.md`](../docs/plans/task-breakdown.md)

---

### Milestone 0: Cấu trúc Dự án, Khung Kiểm chuẩn & Xác minh HF
- [x] **Task T0.1**: Khởi tạo cấu trúc thư mục chuẩn và các file cấu hình môi trường (`requirements.txt`, `requirements-dev.txt`, frontend `package.json`, `.gitignore`, `configs/`).
- [x] **Task T0.2**: Thiết lập khung kiểm thử tự động `pytest` và cấu hình linting (`flake8`) cho backend kèm bài test mẫu.
- [x] **Task T0.3**: Xác minh điều kiện triển khai Hugging Face Docker Space (kiểm tra token/account, xác minh quyền tạo Docker Space, xuất `docs/hf_eligibility_report.md`).

#### 🏁 Checkpoint 0: Khung Dự án & Xác minh Triển khai
- [x] Môi trường ảo và dependencies cơ bản được liệt kê rõ ràng.
- [x] Lệnh `pytest tests/` thực thi thành công không lỗi cú pháp (4/4 passed).
- [x] Báo cáo điều kiện triển khai Hugging Face Docker Space hoàn tất (`docs/hf_eligibility_report.md`).

---

### Milestone 1: Pipeline Dữ liệu & Báo cáo EDA 3 Lớp
- [ ] **Task T1.1**: Viết script `src/data/download.py` tải raw data từ Hugging Face Hub (`ViCTSD`, `UIT-ViHSD`) về `data/raw/`.
- [ ] **Task T1.2**: Phân tích khám phá dữ liệu (EDA), thống kê phân bố 3 lớp (`CLEAN`, `OFFENSIVE`, `HATE`), độ dài câu (95th percentile) và xuất `docs/eda_report.md`.
- [ ] **Task T1.3**: Xây dựng module tiền xử lý văn bản tiếng Việt `src/data/preprocessor.py` (chuẩn hóa Unicode NFC, khoảng trắng, teencode dictionary).
- [ ] **Task T1.4**: Xây dựng module chia dữ liệu `src/data/splitter.py` (Stratified split 3 lớp 70-15-15, kiểm tra assert chống rò rỉ train/test, lưu `data/processed/test.parquet`).

#### 🏁 Checkpoint 1: Dữ liệu Sạch & Sẵn sàng Huấn luyện
- [ ] Báo cáo EDA hoàn tất, xác định tỉ lệ 3 lớp và `max_length`.
- [ ] Tập test chuẩn 3 lớp duy nhất được tạo ra tại `data/processed/test.parquet`, không có rò rỉ dữ liệu.
- [ ] Mọi unit tests tiền xử lý đều pass.

---

### Milestone 2: Mô hình Cơ sở Truyền thống (Traditional ML 3-Class)
- [ ] **Task T2.1**: Xây dựng module tính toán độ đo đa chiều 3 lớp `src/evaluation/metrics.py` (Macro-F1, Per-class F1, Confusion Matrix).
- [ ] **Task T2.2**: Huấn luyện mô hình Baseline `src/models/baseline.py` (TF-IDF word/char + Logistic Regression / LinearSVC) trên 3 lớp và lưu artifact.
- [ ] **Task T2.3**: Đánh giá Baseline trên tập Test chuẩn 3 lớp, xuất kết quả vào `docs/experiments/baseline_results.md`.

#### 🏁 Checkpoint 2: Hoàn tất Baseline 3 Lớp
- [ ] Baseline model 3 lớp được huấn luyện và lưu trữ tại `models/baselines/`.
- [ ] Đã có chỉ số Macro-F1 sàn trên tập test chuẩn 3 lớp.

---

### Milestone 3: Thực nghiệm Transformer trên GPU Local RTX 4060
*(Có thể chạy song song với M4/M5)*
- [ ] **Task T3.1**: Xây dựng pipeline fine-tuning Transformer tiếng Việt `src/models/transformer.py` (`uitnlp/visobert` / `PhoBERT`) tối ưu VRAM RTX 4060 (Mixed precision `fp16`, gradient accumulation, class-weighted loss).
- [ ] **Task T3.2**: Huấn luyện trên RTX 4060, lưu checkpoint tốt nhất, đánh giá đối sánh trên **cùng tập test 3 lớp duy nhất của M2**.
- [ ] **Task T3.3**: Phân tích lỗi trên tối thiểu 30 trường hợp dự đoán sai, lập bảng so sánh đối đầu tại `docs/experiments/model_comparison.md`.

#### 🏁 Checkpoint 3: Hoàn tất Thực nghiệm Học máy Đối sánh
- [ ] Cả 2 nhóm mô hình được đánh giá công bằng trên cùng 1 tập test 3 lớp.
- [ ] Báo cáo khoa học và phân tích lỗi hoàn chỉnh.

---

## Milestone 4: Inference Engine Đa Backend & Kiểm thử Tích hợp
- [ ] **Task T4.1**: Xây dựng interface suy luận đa backend `src/inference/predictor.py` (`BasePredictor`, `SklearnBackend`, `PyTorchBackend`).
- [ ] **Task T4.2**: Xây dựng test suite kiểm thử tích hợp cả hai nhóm mô hình qua một interface thống nhất `tests/test_inference_backends.py`.
- [ ] **Task T4.3**: Viết script benchmark đo đạc PyTorch CPU vs ONNX INT8 (`src/models/export_onnx.py`), chỉ kích hoạt ONNX khi chứng minh được lợi ích thực tế.

#### 🏁 Checkpoint 4: Core Engine Được Kiểm thử Tích hợp Toàn diện
- [ ] Cả hai nhóm mô hình được tích hợp mượt mà qua một interface duy nhất.
- [ ] Test tích hợp pass 100%.

---

### Milestone 5: Xây dựng Ứng dụng Web (React + FastAPI)
- [ ] **Task T5.1**: Xây dựng FastAPI backend `backend/main.py` với các endpoints `/api/predict`, `/api/predict-batch`, `/api/model-info`, `/api/health`.
- [ ] **Task T5.2**: Xây dựng giao diện React cho phân tích bình luận đơn lẻ (Badge màu 3 lớp, thanh phân bố xác suất).
- [ ] **Task T5.3**: Xây dựng giao diện React cho phân tích tệp CSV hàng loạt và trang Model Card hiển thị báo cáo khoa học.
- [ ] **Task T5.4**: Kiểm thử End-to-End Web & API cục bộ (`tests/test_api.py`, `npm run build` xuất `dist/`).

#### 🏁 Checkpoint 5: Web App Hoàn chỉnh trên Môi trường Local
- [ ] Frontend React và Backend FastAPI kết nối hoàn hảo, phục vụ cả phân tích đơn lẻ và hàng loạt.

---

### Milestone 6: Dockerization, Hugging Face Deployment & Nghiệm thu
- [ ] **Task T6.1**: Xây dựng Multi-stage Dockerfile, cấu hình `docker-compose.yml`, kiểm thử container cục bộ trên port 7860.
- [ ] **Task T6.2**: Triển khai lên Hugging Face Spaces (Docker SDK) và chạy Live Smoke Test.
- [ ] **Task T6.3**: Hoàn thiện tài liệu nghiệm thu, hướng dẫn sử dụng và `README.md`.

#### 🏁 Checkpoint 6: Nghiệm thu Toàn diện Đồ án
- [ ] Ứng dụng live công khai 24/7 trên Hugging Face Spaces.
- [ ] Đạt 100% tiêu chí nghiệm thu AC-ML và AC-WEB.
