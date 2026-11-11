# Chi tiết Phân rã Công việc (Task Breakdown & Specifications)
## Vietnamese Toxic Comment Detection (3-Class: React + FastAPI in Docker)

Trạng thái: **APPROVED & SYNCHRONIZED**  
Ngày cập nhật: 2026-11-11  
Dự án: **Vietnamese Toxic Comment Detection System**

---

## TỔNG QUAN PHÂN RÃ CÔNG VIỆC

- **Tổng số Milestone**: 7 (`M0` đến `M6`)
- **Tổng số Task**: 24 tasks nguyên tử (XS: 5, S: 15, M: 4, L/XL: 0)
- **Quy tắc an toàn**: Mỗi task độc lập, có acceptance criteria rõ ràng, file list cụ thể và lệnh kiểm chứng.

---

## MILESTONE 0: CẤU TRÚC DỰ ÁN, KHUNG KIỂM CHUẨN & XÁC MINH HF (M0)

### Task T0.1: Khởi tạo Cấu trúc Thư mục & Môi trường Phát triển Đa nền tảng
- **Mô tả**: Thiết lập khung cấu trúc thư mục dự án cho cả Python (Backend/ML) và Node.js (React Frontend), cấu hình file dependencies (`requirements.txt`, `requirements-dev.txt`, frontend `package.json`), cấu hình `.gitignore` và các thư mục config/data/models.
- **Trạng thái**: **Đã hoàn thành (Done)**.
- **Files**: `requirements.txt`, `requirements-dev.txt`, `frontend/package.json`, `.gitignore`, `configs/`.

### Task T0.2: Thiết lập Khung Kiểm thử Tự động & Linting
- **Mô tả**: Thiết lập cấu hình `pytest`, `flake8` cho Python và cấu hình kiểm thử/linting cho Frontend React, kèm 1 test case mẫu (smoke test) trong `tests/test_smoke.py`.
- **Trạng thái**: **Đã hoàn thành (Done - 4 passed, 1 skipped)**.
- **Files**: `setup.cfg`, `tests/__init__.py`, `tests/test_smoke.py`.

### Task T0.3: Xác minh Điều kiện Triển khai Hugging Face Docker Space (HF Eligibility Check)
- **Mô tả**: Viết script kiểm tra xem tài khoản Hugging Face của người dùng có đủ điều kiện tạo và chạy Docker Space miễn phí hay không (kiểm tra token, quyền hạn, ràng buộc thẻ tín dụng). Ghi nhận kết quả vào `docs/hf_eligibility_report.md`.
- **Trạng thái**: **Bị chặn / Chờ token (`[BLOCKED / PENDING HF TOKEN]`)**. Script `src/tools/check_hf_eligibility.py` đã chạy và ghi nhận trạng thái `ATTENTION / PENDING TOKEN` do chưa cấu hình `HF_TOKEN`. Không đánh dấu hoàn tất deployment verification khi chưa có token.
- **Files**: `src/tools/check_hf_eligibility.py`, `docs/hf_eligibility_report.md`.

---

### 🏁 Checkpoint 0: Khung Dự án & Ghi nhận Trạng thái Triển khai
- [x] Cấu trúc code Python và React sẵn sàng.
- [x] Khung kiểm thử tự động `pytest` pass.
- [ ] Tình trạng Hugging Face Docker Space: **PENDING TOKEN / CHƯA XÁC MINH** (Ghi nhận blocker, sẽ xác minh lại khi người dùng cung cấp token ở M6).

---

## MILESTONE 1: PIPELINE DỮ LIỆU & BÁO CÁO EDA 3 LỚP (M1)

### Task T1.1: Tải & Lưu trữ Nguyên bản Cả 2 Dataset từ Hugging Face
- **Mô tả**: Viết script `src/data/download.py` tải `ViCTSD` (`tarudesu/ViCTSD`) và `UIT-ViHSD` (`ura-hcmut/UIT-ViHSD`) về `data/raw/`.
- **Phụ thuộc**: `M0`.
- **Files**: `src/data/download.py`, `data/raw/`.

### Task T1.2: Phân tích Khám phá Dữ liệu Đối chiếu (EDA Report)
- **Mô tả**: Thống kê số lượng mẫu, phân bố 3 lớp của `UIT-ViHSD` (`CLEAN`, `OFFENSIVE`, `HATE`), phân bố độ dài từ/ký tự (xác định ngưỡng 95th percentile để chốt `max_length` từ mức PROVISIONAL), phân tích teencode/slang. Xuất báo cáo ra `docs/eda_report.md`.
- **Phụ thuộc**: `T1.1`.
- **Files**: `notebooks/01_dataset_eda.ipynb` (hoặc `src/data/eda.py`), `docs/eda_report.md`.

### Task T1.3: Module Tiền Xử lý Văn bản Tiếng Việt 3 Lớp
- **Mô tả**: Xây dựng `src/data/preprocessor.py` chuẩn hóa Unicode NFC, xử lý khoảng trắng, regex khử ký tự điều khiển ẩn, và chuẩn hóa teencode cơ bản qua từ điển tra cứu. Viết unit tests `tests/test_preprocessing.py`.
- **Phụ thuộc**: `T1.2`.
- **Files**: `src/data/preprocessor.py`, `src/data/teencode_dict.json`, `tests/test_preprocessing.py`.

### Task T1.4: Phân chia Tập Dữ liệu 3 Lớp & Khóa Tập Test
- **Mô tả**: Xây dựng `src/data/splitter.py` chuẩn hóa nhãn 3 lớp (`0: CLEAN`, `1: OFFENSIVE`, `2: HATE`), chia tập Train (70%), Validation (15%), và Test (15% - **LOCKED**). Assert `set(train_texts) ∩ set(test_texts) == 0`. Lưu vào `data/processed/` (`train.parquet`, `val.parquet`, `test.parquet`).
- **Phụ thuộc**: `T1.3`.
- **Files**: `src/data/splitter.py`, `tests/test_leakage.py`, `data/processed/*.parquet`.

---

### 🏁 Checkpoint 1: Dữ liệu Sạch & Tập Test Đã Khóa
- [ ] Báo cáo EDA hoàn tất, xác định tỉ lệ 3 lớp và chốt thông số `max_length`.
- [ ] Tập test chuẩn 3 lớp duy nhất được tạo ra tại `data/processed/test.parquet`, bị khóa và không có rò rỉ dữ liệu.

---

## MILESTONE 2: MÔ HÌNH CƠ SỞ TRUYỀN THỐNG (TRADITIONAL ML 3-CLASS) (M2)

### Task T2.1: Xây dựng Module Đánh giá Đa chiều 3 Lớp (Metrics Module)
- **Mô tả**: Xây dựng `src/evaluation/metrics.py` tính toán: Macro-F1, Per-class Precision/Recall/F1 cho 3 lớp (`CLEAN`, `OFFENSIVE`, `HATE`), Confusion Matrix, và Latency.
- **Phụ thuộc**: `T1.4`.
- **Files**: `src/evaluation/metrics.py`, `tests/test_metrics.py`.

### Task T2.2: Huấn luyện Mô hình Cơ sở & Hiệu chuẩn Xác suất (Calibration)
- **Mô tả**: Xây dựng `src/models/baseline.py` trích xuất đặc trưng TF-IDF word n-grams (1, 2) + char n-grams (2, 4) trên tập Train. Huấn luyện Logistic Regression và LinearSVC; bọc LinearSVC bằng `CalibratedClassifierCV(cv=5, method='sigmoid')` trên tập Train để đảm bảo hợp đồng xác suất hiệu chuẩn.
- **Phụ thuộc**: `T2.1`.
- **Files**: `src/models/baseline.py`, `configs/baseline_config.yaml`.

### Task T2.3: Tinh chỉnh Siêu tham số & Chọn Baseline Tốt nhất trên Validation Set
- **Mô tả**: Đánh giá các ứng viên Baseline trên tập **Validation (`val.parquet`)**. Chọn ra mô hình Baseline tốt nhất theo Validation Macro-F1. Lưu artifact vào `models/baselines/baseline_model.joblib`. Xuất báo cáo vào `docs/experiments/baseline_val_results.md`. **Tuyệt đối không đụng vào Test set**.
- **Phụ thuộc**: `T2.2`.
- **Files**: `src/evaluation/evaluate_baseline_val.py`, `docs/experiments/baseline_val_results.md`.

---

### 🏁 Checkpoint 2: Hoàn tất Baseline Trên Validation Set
- [ ] Best Baseline model 3 lớp được lựa chọn trên Validation set và lưu trữ tại `models/baselines/`.
- [ ] Đạt chuẩn hợp đồng xác suất hiệu chuẩn ($\sum P \approx 1.0$).

---

## MILESTONE 3: THỰC NGHIỆM TRANSFORMER TRÊN GPU LOCAL RTX 4060 (M3)
*(Có thể chạy song song với M4/M5)*

### Task T3.1: Xây dựng Pipeline Fine-tuning Transformer Tối ưu VRAM (RTX 4060)
- **Mô tả**: Xây dựng `src/models/transformer.py` sử dụng PyTorch và Hugging Face Trainer để fine-tune `uitnlp/visobert` (hoặc `PhoBERT`) trên 3 lớp, cấu hình Mixed Precision `fp16`, gradient accumulation, class-weighted cross-entropy loss, VRAM budget < 7.0GB.
- **Phụ thuộc**: `T2.1`.
- **Files**: `src/models/transformer.py`, `configs/transformer_config.yaml`, `src/models/dataset_torch.py`.

### Task T3.2: Huấn luyện trên RTX 4060 & Chọn Checkpoint Tốt nhất trên Validation Set
- **Mô tả**: Huấn luyện mô hình Transformer trên local GPU RTX 4060 (tập Train). Lưu checkpoint tốt nhất dựa trên **Validation Macro-F1** (Early stopping sau 2 epochs không cải thiện trên tập Validation). Xuất báo cáo vào `docs/experiments/transformer_val_results.md`. **Tuyệt đối không đụng vào Test set**.
- **Phụ thuộc**: `T3.1`.
- **Files**: `models/transformer_best/`, `docs/experiments/transformer_val_results.md`.

### Task T3.3: Phân tích Lỗi trên Tập Validation & Báo cáo Định tính
- **Mô tả**: Trích xuất tối thiểu 30 trường hợp dự đoán sai của mô hình Transformer trên tập Validation. Phân tích nguyên nhân ngôn ngữ (mỉa mai, ngữ cảnh thô tục không ác ý, teencode phức tạp). Xuất báo cáo phân tích lỗi vào `docs/experiments/error_analysis_val.md`.
- **Phụ thuộc**: `T3.2`.
- **Files**: `src/evaluation/error_analysis.py`, `docs/experiments/error_analysis_val.md`.

---

### 🏁 Checkpoint 3: Hoàn tất Thực nghiệm Transformer trên Validation Set
- [ ] Best Transformer checkpoint được lựa chọn hoàn toàn dựa trên tập Validation.
- [ ] Hoàn thành báo cáo phân tích lỗi định tính.

---

## MILESTONE 4: INFERENCE ENGINE ĐA BACKEND & KIỂM THỬ TÍCH HỢP TÁCH BIỆT (M4)

### Task T4.1: Xây dựng Interface Suy luận Đa Backend (`BasePredictor`)
- **Mô tả**: Xây dựng `src/inference/predictor.py` định nghĩa `BasePredictor` và các backend: `SklearnBackend` (nạp baseline joblib với calibrated probabilities) và `PyTorchBackend` (nạp Transformer weights). Tuân thủ schema `backend/schemas.py`.
- **Phụ thuộc**: `T1.3`, `T2.2`.
- **Files**: `src/inference/predictor.py`, `src/inference/backends.py`.

### Task T4.2: Kiểm thử Tích hợp Baseline Riêng biệt (Baseline Integration Test)
- **Mô tả**: Xây dựng test suite `tests/test_baseline_integration.py` kiểm tra tích hợp Baseline với `BasePredictor` và REST API schema. Xác nhận xác suất hiệu chuẩn hợp lệ, latency < 50ms, và unblock phát triển Web MVP ngay lập tức.
- **Phụ thuộc**: `T4.1`, `T2.3`.
- **Files**: `tests/test_baseline_integration.py`.

### Task T4.3: Kiểm thử Tích hợp Transformer Riêng biệt (Transformer Integration Test)
- **Mô tả**: Xây dựng test suite `tests/test_transformer_integration.py` kiểm tra tích hợp checkpoint Transformer vào `PyTorchBackend`. Xác nhận output tuân thủ schema `PredictionResponse`, đo đạc memory footprint và so sánh latency với Baseline trên CPU.
- **Phụ thuộc**: `T4.1`, `T3.2`.
- **Files**: `tests/test_transformer_integration.py`.

### Task T4.4: Benchmark Tùy chọn ONNX & Quantization (Conditional ONNX)
- **Mô tả**: Viết script `src/models/export_onnx.py`. Đo lường so sánh giữa PyTorch CPU và ONNX INT8 (độ trễ và RAM). Chỉ kích hoạt `ONNXBackend` nếu benchmark chứng minh giảm độ trễ $\ge 25\%$ mà không làm giảm F1 (>0.5%). Ghi nhận kết quả vào `docs/experiments/onnx_benchmark.md`.
- **Phụ thuộc**: `T4.3`.
- **Files**: `src/models/export_onnx.py`, `docs/experiments/onnx_benchmark.md`.

---

### 🏁 Checkpoint 4: Core Engine Được Kiểm thử Tích hợp Toàn diện
- [ ] Cả 2 họ mô hình vượt qua các bài kiểm thử tích hợp riêng biệt.
- [ ] Hợp đồng xác suất được kiểm chứng tự động 100%.

---

## MILESTONE 5: XÂY DỰNG ỨNG DỤNG WEB (REACT + FASTAPI) (M5)

### Task T5.1: Xây dựng Backend FastAPI & REST Endpoints
- **Mô tả**: Xây dựng ứng dụng FastAPI tại `backend/main.py` với các router:
  - `POST /api/predict`: Nhận text, trả về 3 nhãn + calibrated probabilities + latency.
  - `POST /api/predict-batch`: Nhận file CSV/TXT, trả về kết quả batch.
  - `GET /api/model-info`: Trả về thông tin mô hình và metrics.
  - `GET /api/health`: Health check (trạng thái model_loaded, backend, artifact).
- **Phụ thuộc**: `T4.2` (Chạy được ngay với Baseline).
- **Files**: `backend/main.py`, `backend/api/routes_predict.py`, `backend/api/routes_meta.py`.

### Task T5.2: Xây dựng Giao diện React: Phân tích Bình luận Đơn lẻ (Single Comment UI)
- **Mô tả**: Xây dựng component React cho phép nhập bình luận tiếng Việt, gọi API `/api/predict`, hiển thị Badge màu theo 3 lớp (`CLEAN` xanh, `OFFENSIVE` vàng cam, `HATE` đỏ), các thanh phân bố xác suất cho cả 3 nhãn và thời gian xử lý.
- **Phụ thuộc**: `T5.1`.
- **Files**: `frontend/src/App.jsx`, `frontend/src/components/SingleComment.jsx`, `frontend/src/services/api.js`.

### Task T5.3: Xây dựng Giao diện React: Phân tích Tệp Hàng loạt & Model Card
- **Mô tả**: Xây dựng component upload CSV (giới hạn 500 dòng), thanh tiến trình, biểu đồ phân bố 3 lớp, nút download kết quả CSV; cùng tab Model Card hiển thị bảng đối sánh khoa học và confusion matrix.
- **Phụ thuộc**: `T5.2`.
- **Files**: `frontend/src/components/BatchUpload.jsx`, `frontend/src/components/ModelCard.jsx`, `frontend/src/components/Charts.jsx`.

### Task T5.4: Kiểm thử End-to-End Web & API Cục bộ
- **Mô tả**: Viết test suite `tests/test_api.py` sử dụng `fastapi.testclient.TestClient` kiểm tra toàn bộ API endpoints; build thử frontend React (`npm run build`) và kiểm tra FastAPI phục vụ file tĩnh.
- **Phụ thuộc**: `T5.3`.
- **Files**: `tests/test_api.py`.

---

### 🏁 Checkpoint 5: Web App Hoàn chỉnh trên Môi trường Local
- [ ] Frontend React và Backend FastAPI kết nối hoàn hảo, phục vụ phân tích đơn lẻ và batch.

---

## MILESTONE 6: ĐÁNH GIÁ TẬP TEST ĐÃ KHÓA, DOCKER & DEPLOYMENT (M6)

### Task T6.1: Đánh giá Đối kháng trên Tập Test Đã Khóa (Locked Test Final Evaluation)
- **Mô tả**: Mở tập Test đã khóa (`data/processed/test.parquet`) **đúng một lần duy nhất**. Đánh giá đối kháng khách quan giữa Best Baseline và Best Transformer. Xuất ma trận nhầm lẫn và bảng so sánh toàn diện vào `docs/experiments/final_test_benchmark.md`. Báo cáo trung thực toàn bộ chỉ số bất kể Transformer cao hơn, tương đương hay thấp hơn Baseline.
- **Phụ thuộc**: `T2.3`, `T3.2`.
- **Files**: `src/evaluation/evaluate_locked_test.py`, `docs/experiments/final_test_benchmark.md`.

### Task T6.2: Xây dựng Multi-stage Dockerfile & Tích hợp Tải Artifacts
- **Mô tả**: Viết `Dockerfile` đa tầng (Stage 1 build React -> Stage 2 copy vào FastAPI slim image), tích hợp công cụ `src/tools/download_artifacts.py` để nạp model (hoặc fallback sang baseline joblib tích hợp sẵn). Cấu hình `docker-compose.yml`, build và kiểm thử container cục bộ trên port 7860.
- **Phụ thuộc**: `T5.4`, `T6.1`.
- **Files**: `Dockerfile`, `docker-compose.yml`, `src/tools/download_artifacts.py`.

### Task T6.3: Triển khai Lên Hugging Face Spaces & Chạy Live Smoke Test
- **Mô tả**: Cung cấp `HF_TOKEN`, chạy lại script xác thực `check_hf_eligibility.py`. Đẩy mã nguồn lên Hugging Face Spaces (Docker SDK), theo dõi log container, chạy bài kiểm thử khói (Smoke Test) trên link public 24/7. Nếu vướng rào cản thẻ tín dụng, kích hoạt phương án dự phòng Render/Local Docker.
- **Phụ thuộc**: `T6.2`.
- **Files**: `README.md` (HF Spaces metadata), `docs/live_smoke_test_report.md`.

### Task T6.4: Hoàn thiện Tài liệu Nghiệm thu, README & Hướng dẫn Sử dụng
- **Mô tả**: Tổng kết báo cáo nghiệm thu đối chiếu với toàn bộ tiêu chí AC-ML và AC-WEB; hoàn thiện `README.md` với video/ảnh chụp minh chứng, link public và hướng dẫn chạy cục bộ.
- **Phụ thuộc**: `T6.3`.
- **Files**: `README.md`, `docs/acceptance_report.md`.

---

### 🏁 Checkpoint 6: Nghiệm thu Toàn diện Đồ án
- [ ] Ứng dụng live công khai 24/7 trên Hugging Face Spaces (hoặc phương án dự phòng được phê duyệt).
- [ ] Báo cáo đối sánh trung thực trên tập test đã khóa, hoàn thành 100% tiêu chí nghiệm thu.
