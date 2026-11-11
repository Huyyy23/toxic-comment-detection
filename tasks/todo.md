# Danh sách Nhiệm vụ Dự án (Project Task List)
## Vietnamese Toxic Comment Detection (3-Class: React + FastAPI in Docker)

Kế hoạch tham chiếu: [`tasks/plan.md`](plan.md) và [`docs/plans/task-breakdown.md`](../docs/plans/task-breakdown.md)

---

### Milestone 0: Cấu trúc Dự án, Khung Kiểm chuẩn & Xác minh HF
- [x] **Task T0.1**: Khởi tạo cấu trúc thư mục chuẩn và các file cấu hình môi trường (`requirements.txt`, `requirements-dev.txt`, frontend `package.json`, `.gitignore`, `configs/`).
- [x] **Task T0.2**: Thiết lập khung kiểm thử tự động `pytest` và cấu hình linting (`flake8`) cho backend kèm bài test mẫu.
- [ ] **Task T0.3**: Xác minh điều kiện triển khai Hugging Face Docker Space (Script `src/tools/check_hf_eligibility.py` đã tạo; báo cáo ghi nhận `ATTENTION / PENDING TOKEN` do chưa có token; chưa đánh dấu hoàn tất verification).

#### 🏁 Checkpoint 0: Khung Dự án & Ghi nhận Trạng thái Triển khai
- [x] Cấu trúc code Python và React sẵn sàng.
- [x] Lệnh `pytest tests/` thực thi thành công không lỗi cú pháp (4 passed, 1 skipped).
- [ ] Báo cáo điều kiện triển khai Hugging Face Docker Space: **PENDING TOKEN / CHƯA XÁC MINH** (Ghi nhận blocker, chờ cung cấp token ở M6).

---

### Milestone 1: Pipeline Dữ liệu, EDA & Phân chia 3 Lớp
- [ ] **Task T1.1**: Viết script `src/data/download.py` tải raw data từ Hugging Face Hub (`ViCTSD`, `UIT-ViHSD`) về `data/raw/`.
- [ ] **Task T1.2**: Phân tích khám phá dữ liệu (EDA), thống kê phân bố 3 lớp (`CLEAN`, `OFFENSIVE`, `HATE`), độ dài câu (95th percentile để chốt `max_length`), xuất `docs/eda_report.md`.
- [ ] **Task T1.3**: Xây dựng module tiền xử lý văn bản tiếng Việt `src/data/preprocessor.py` (chuẩn hóa Unicode NFC, khoảng trắng, teencode dictionary).
- [ ] **Task T1.4**: Xây dựng module chia dữ liệu `src/data/splitter.py` (Stratified split 3 lớp 70/15/15, lưu `train.parquet`, `val.parquet`, và `test.parquet` [LOCKED], assert chống rò rỉ train/test).

#### 🏁 Checkpoint 1: Dữ liệu Sạch & Tập Test Đã Khóa
- [ ] Báo cáo EDA hoàn tất, xác định tỉ lệ 3 lớp và chốt thông số `max_length`.
- [ ] Tập test chuẩn 3 lớp duy nhất được tạo ra tại `data/processed/test.parquet`, bị khóa và không có rò rỉ dữ liệu.

---

### Milestone 2: Mô hình Cơ sở Truyền thống (Validation Tuned)
- [ ] **Task T2.1**: Xây dựng module tính toán độ đo đa chiều 3 lớp `src/evaluation/metrics.py` (Macro-F1, Per-class F1, Confusion Matrix).
- [ ] **Task T2.2**: Huấn luyện mô hình Baseline `src/models/baseline.py` (TF-IDF word/char + Logistic Regression / LinearSVC) trên tập Train, bọc LinearSVC bằng `CalibratedClassifierCV` (Platt scaling).
- [ ] **Task T2.3**: Đánh giá, tinh chỉnh siêu tham số và lựa chọn mô hình Baseline tốt nhất trên **Validation set (`val.parquet`)**, lưu `models/baselines/baseline_model.joblib`. Xuất `docs/experiments/baseline_val_results.md`.

#### 🏁 Checkpoint 2: Hoàn tất Baseline Trên Validation Set
- [ ] Best Baseline model 3 lớp được lựa chọn trên Validation set và lưu trữ tại `models/baselines/`.
- [ ] Đạt chuẩn hợp đồng xác suất hiệu chuẩn ($\sum P \approx 1.0$).

---

### Milestone 3: Thực nghiệm Transformer trên GPU Local RTX 4060
*(Có thể chạy song song với M4/M5)*
- [ ] **Task T3.1**: Xây dựng pipeline fine-tuning Transformer tiếng Việt `src/models/transformer.py` (`uitnlp/visobert` / `PhoBERT`) tối ưu VRAM RTX 4060 (Mixed precision `fp16`, gradient accumulation, class-weighted loss).
- [ ] **Task T3.2**: Huấn luyện trên RTX 4060 (Train set), lưu checkpoint tốt nhất theo **Validation Macro-F1** (Early stopping trên tập Validation). Xuất `docs/experiments/transformer_val_results.md`.
- [ ] **Task T3.3**: Phân tích lỗi trên tập Validation (ít nhất 30 mẫu sai), lập báo cáo định tính tại `docs/experiments/error_analysis_val.md`.

#### 🏁 Checkpoint 3: Hoàn tất Thực nghiệm Transformer trên Validation Set
- [ ] Best Transformer checkpoint được lựa chọn hoàn toàn dựa trên tập Validation.
- [ ] Hoàn thành báo cáo phân tích lỗi định tính.

---

### Milestone 4: Inference Engine Đa Backend & Kiểm thử Tích hợp Tách biệt
- [ ] **Task T4.1**: Xây dựng interface suy luận đa backend `src/inference/predictor.py` (`BasePredictor`, `SklearnBackend`, `PyTorchBackend`), tuân thủ schema `backend/schemas.py`.
- [ ] **Task T4.2**: Xây dựng test suite **Baseline Integration Test** `tests/test_baseline_integration.py` (kiểm thử tích hợp Baseline sớm, unblock Web MVP).
- [ ] **Task T4.3**: Xây dựng test suite **Transformer Integration Test** `tests/test_transformer_integration.py` (kiểm thử tích hợp Transformer checkpoint, đo latency và memory).
- [ ] **Task T4.4**: Viết script benchmark đo đạc PyTorch CPU vs ONNX INT8 (`src/models/export_onnx.py`), chỉ kích hoạt ONNX khi chứng minh được lợi ích thực tế ($\ge 25\%$ latency reduction).

#### 🏁 Checkpoint 4: Core Engine Được Kiểm thử Tích hợp Toàn diện
- [ ] Cả hai nhóm mô hình được kiểm thử tích hợp riêng biệt qua interface thống nhất.
- [ ] Hợp đồng xác suất được kiểm chứng tự động 100%.

---

### Milestone 5: Xây dựng Ứng dụng Web (React + FastAPI)
- [ ] **Task T5.1**: Xây dựng FastAPI backend `backend/main.py` với các endpoints `/api/predict`, `/api/predict-batch`, `/api/model-info`, `/api/health`.
- [ ] **Task T5.2**: Xây dựng giao diện React cho phân tích bình luận đơn lẻ (Badge màu 3 lớp, thanh phân bố xác suất).
- [ ] **Task T5.3**: Xây dựng giao diện React cho phân tích tệp CSV hàng loạt và trang Model Card hiển thị báo cáo khoa học.
- [ ] **Task T5.4**: Kiểm thử End-to-End Web & API cục bộ (`tests/test_api.py`, `npm run build` xuất `dist/`).

#### 🏁 Checkpoint 5: Web App Hoàn chỉnh trên Môi trường Local
- [ ] Frontend React và Backend FastAPI kết nối hoàn hảo, phục vụ cả phân tích đơn lẻ và hàng loạt.

---

### Milestone 6: Đánh giá Tập Test Đã Khóa, Docker & Deployment
- [ ] **Task T6.1**: **Đánh giá đối kháng trên tập Test đã khóa** (`src/evaluation/evaluate_locked_test.py` - mở tập test đúng 1 lần, báo cáo trung thực toàn bộ chỉ số vào `docs/experiments/final_test_benchmark.md`).
- [ ] **Task T6.2**: Xây dựng Multi-stage Dockerfile, tích hợp công cụ tải artifact `src/tools/download_artifacts.py` (có fallback), cấu hình `docker-compose.yml`, kiểm thử container cục bộ trên port 7860.
- [ ] **Task T6.3**: Triển khai lên Hugging Face Spaces (Docker SDK) và chạy Live Smoke Test (cung cấp `HF_TOKEN`; fallback Render/Local nếu vướng thẻ tín dụng).
- [ ] **Task T6.4**: Hoàn thiện tài liệu nghiệm thu, hướng dẫn sử dụng và `README.md`.

#### 🏁 Checkpoint 6: Nghiệm thu Toàn diện Đồ án
- [ ] Ứng dụng live công khai 24/7 trên Hugging Face Spaces (hoặc phương án dự phòng được phê duyệt).
- [ ] Báo cáo đối sánh khách quan trên tập test đã khóa, hoàn thành 100% tiêu chí nghiệm thu.
