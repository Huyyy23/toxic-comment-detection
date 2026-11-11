# Chi tiết Phân rã Công việc (Task Breakdown & Specifications)
## Vietnamese Toxic Comment Detection (3-Class: React + FastAPI in Docker)

Trạng thái: **APPROVED & SYNCHRONIZED**  
Ngày cập nhật: 2026-11-11  
Dự án: **Vietnamese Toxic Comment Detection System**

---

## TỔNG QUAN PHÂN RÃ CÔNG VIỆC

- **Tổng số Milestone**: 7 (`M0` đến `M6`)
- **Tổng số Task**: 22 tasks nguyên tử (XS: 5, S: 13, M: 4, L/XL: 0)
- **Quy tắc an toàn**: Mỗi task độc lập, có acceptance criteria rõ ràng, file list cụ thể và lệnh kiểm chứng (verification command).

---

## MILESTONE 0: CẤU TRÚC DỰ ÁN, KHUNG KIỂM CHUẨN & XÁC MINH HF (M0)

### Task T0.1: Khởi tạo Cấu trúc Thư mục & Môi trường Phát triển Đa nền tảng
- **Mô tả**: Thiết lập khung cấu trúc thư mục dự án cho cả Python (Backend/ML) và Node.js (React Frontend), cấu hình file dependencies (`requirements.txt`, `requirements-dev.txt`, frontend `package.json`), cấu hình `.gitignore` và các thư mục config/data/models.
- **Phụ thuộc**: Không.
- **Ước lượng**: **S** (3 files).
- **Files tác động**:
  - `requirements.txt`
  - `requirements-dev.txt`
  - `frontend/package.json`
  - `.gitignore`
- **Tiêu chí nghiệm thu**:
  - [ ] Thư mục `src/`, `backend/`, `frontend/`, `tests/`, `configs/`, `data/`, `models/`, `docs/` được tạo đầy đủ.
  - [ ] `.gitignore` chặn các file checkpoint lớn (`.bin`, `.pth`, `.joblib`, `models/transformer_best/`), dữ liệu thô (`data/raw/`) và `node_modules/`.
- **Quy trình kiểm chứng**:
  - Kiểm tra cấu trúc thư mục bằng lệnh danh sách file.

---

### Task T0.2: Thiết lập Khung Kiểm thử Tự động & Linting
- **Mô tả**: Thiết lập cấu hình `pytest`, `flake8` cho Python và cấu hình kiểm thử/linting cho Frontend React, kèm 1 test case mẫu (smoke test) trong `tests/test_smoke.py`.
- **Phụ thuộc**: `T0.1`.
- **Ước lượng**: **S** (3 files).
- **Files tác động**:
  - `setup.cfg`
  - `tests/__init__.py`
  - `tests/test_smoke.py`
- **Tiêu chí nghiệm thu**:
  - [ ] Chạy `pytest tests/ -v` phát hiện và pass bài test mẫu.
  - [ ] Lệnh kiểm tra định dạng `flake8 src/ backend/ tests/` thực thi thành công không lỗi cú pháp.
- **Quy trình kiểm chứng**:
  - Chạy: `pytest tests/ -v` (kết quả: 1 passed).
  - Chạy: `flake8 src/ backend/ tests/ --max-line-length=100`.

---

### Task T0.3: Xác minh Điều kiện Triển khai Hugging Face Docker Space (HF Eligibility Check)
- **Mô tả**: Viết script kiểm tra và thực hiện xác minh xem tài khoản Hugging Face của người dùng có đủ điều kiện tạo và chạy Docker Space miễn phí hay không (kiểm tra token, quyền hạn, ràng buộc thẻ tín dụng). Ghi nhận kết quả vào `docs/hf_eligibility_report.md`.
- **Phụ thuộc**: `T0.1`.
- **Ước lượng**: **XS** (2 files).
- **Files tác động**:
  - `src/tools/check_hf_eligibility.py`
  - `docs/hf_eligibility_report.md`
- **Tiêu chí nghiệm thu**:
  - [ ] Báo cáo `docs/hf_eligibility_report.md` ghi nhận rõ tình trạng: Đủ điều kiện (Eligible) hay Bị chặn (Blocked).
  - [ ] Nếu bị chặn, ghi rõ nguyên nhân và phương án dự phòng (Render Free Docker / Local Container), không tự ý đổi nền tảng.
- **Quy trình kiểm chứng**:
  - Chạy: `python -m src.tools.check_hf_eligibility`.
  - Đọc nội dung báo cáo tại `docs/hf_eligibility_report.md`.

---

### 🏁 Checkpoint 0: Khung Dự án & Xác minh Triển khai Hoàn tất
- [ ] Cấu trúc code Python và React sẵn sàng.
- [ ] Khung kiểm thử tự động `pytest` pass.
- [ ] Tình trạng Hugging Face Docker Space được ghi nhận minh bạch.

---

## MILESTONE 1: PIPELINE DỮ LIỆU & BÁO CÁO EDA 3 LỚP (M1)

### Task T1.1: Tải & Lưu trữ Nguyên bản Cả 2 Dataset từ Hugging Face
- **Mô tả**: Viết script `src/data/download.py` tải `ViCTSD` (`tarudesu/ViCTSD`) và `UIT-ViHSD` (`ura-hcmut/UIT-ViHSD`) về `data/raw/`.
- **Phụ thuộc**: `M0`.
- **Ước lượng**: **S** (2 files).
- **Files tác động**: `src/data/download.py`, `data/raw/`

### Task T1.2: Phân tích Khám phá Dữ liệu Đối chiếu (EDA Report)
- **Mô tả**: Thống kê số lượng mẫu, phân bố nhãn của cả 2 dataset, kiểm tra phân bố 3 lớp của `UIT-ViHSD` (`CLEAN`, `OFFENSIVE`, `HATE`), phân bố độ dài từ/ký tự (xác định ngưỡng 95th percentile cho `max_length`), phân tích teencode/slang. Xuất báo cáo ra `docs/eda_report.md`.
- **Phụ thuộc**: `T1.1`.
- **Ước lượng**: **M** (2 files).
- **Files tác động**: `notebooks/01_dataset_eda.ipynb` (hoặc `src/data/eda.py`), `docs/eda_report.md`.

### Task T1.3: Module Tiền Xử lý Văn bản Tiếng Việt 3 Lớp
- **Mô tả**: Xây dựng `src/data/preprocessor.py` chuẩn hóa Unicode NFC, xử lý khoảng trắng, regex khử ký tự điều khiển ẩn, và chuẩn hóa teencode cơ bản qua từ điển tra cứu. Viết unit tests kiểm thử các trường hợp biên.
- **Phụ thuộc**: `T1.2`.
- **Ước lượng**: **M** (3 files).
- **Files tác động**: `src/data/preprocessor.py`, `src/data/teencode_dict.json`, `tests/test_preprocessing.py`.

### Task T1.4: Phân chia Tập Dữ liệu 3 Lớp & Chống Rò rỉ Dữ liệu
- **Mô tả**: Xây dựng `src/data/splitter.py` chuẩn hóa nhãn 3 lớp (`0: CLEAN`, `1: OFFENSIVE`, `2: HATE`), chia tập Train/Val/Test phân tầng (Stratified) hoặc theo split chuẩn của UIT-ViHSD. Kiểm tra assert `set(train_texts) ∩ set(test_texts) == 0`. Lưu vào `data/processed/`.
- **Phụ thuộc**: `T1.3`.
- **Ước lượng**: **S** (2 files).
- **Files tác động**: `src/data/splitter.py`, `tests/test_leakage.py`, `data/processed/*.parquet`.

---

### 🏁 Checkpoint 1: Dữ liệu Sạch & Sẵn sàng Huấn luyện
- [ ] Báo cáo EDA hoàn tất, xác định tỉ lệ 3 lớp và `max_length`.
- [ ] Tập test chuẩn 3 lớp duy nhất được tạo ra tại `data/processed/test.parquet`, không có rò rỉ dữ liệu.

---

## MILESTONE 2: MÔ HÌNH CƠ SỞ TRUYỀN THỐNG (TRADITIONAL ML 3-CLASS) (M2)

### Task T2.1: Xây dựng Module Đánh giá Đa chiều 3 Lớp (Metrics Module)
- **Mô tả**: Xây dựng `src/evaluation/metrics.py` tính toán: Macro-F1, Per-class Precision/Recall/F1 cho 3 lớp (`CLEAN`, `OFFENSIVE`, `HATE`), Confusion Matrix, và Inference Latency.
- **Phụ thuộc**: `T1.4`.
- **Ước lượng**: **S** (2 files).
- **Files tác động**: `src/evaluation/metrics.py`, `tests/test_metrics.py`.

### Task T2.2: Huấn luyện Mô hình Cơ sở (TF-IDF + Linear Classifier)
- **Mô tả**: Xây dựng `src/models/baseline.py` trích xuất đặc trưng TF-IDF word n-grams (1, 2) + char n-grams (2, 4), huấn luyện Logistic Regression / LinearSVC có `class_weight='balanced'`. Lưu artifact vào `models/baselines/`.
- **Phụ thuộc**: `T2.1`.
- **Ước lượng**: **S** (2 files).
- **Files tác động**: `src/models/baseline.py`, `configs/baseline_config.yaml`.

### Task T2.3: Đánh giá Baseline trên Tập Test Chuẩn 3 Lớp
- **Mô tả**: Đánh giá mô hình Baseline trên tập `test.parquet`. Xuất bảng chỉ số chi tiết và confusion matrix vào `docs/experiments/baseline_results.md`.
- **Phụ thuộc**: `T2.2`.
- **Ước lượng**: **XS** (2 files).
- **Files tác động**: `src/evaluation/evaluate_baseline.py`, `docs/experiments/baseline_results.md`.

---

### 🏁 Checkpoint 2: Hoàn tất Baseline 3 Lớp
- [ ] Baseline artifact hoạt động được, có số liệu Macro-F1 sàn trên tập test 3 lớp.

---

## MILESTONE 3: THỰC NGHIỆM TRANSFORMER TRÊN GPU LOCAL RTX 4060 (M3)
*(Có thể chạy song song với M4/M5)*

### Task T3.1: Xây dựng Pipeline Fine-tuning Transformer Tối ưu VRAM (RTX 4060)
- **Mô tả**: Xây dựng `src/models/transformer.py` sử dụng PyTorch và Hugging Face Trainer để fine-tune `uitnlp/visobert` (hoặc `PhoBERT`) trên 3 lớp, cấu hình Mixed Precision `fp16`, gradient accumulation, class-weighted cross-entropy loss, VRAM budget < 7.0GB.
- **Phụ thuộc**: `T2.1`.
- **Ước lượng**: **M** (3 files).
- **Files tác động**: `src/models/transformer.py`, `configs/transformer_config.yaml`, `src/models/dataset_torch.py`.

### Task T3.2: Huấn luyện trên RTX 4060 & Đánh giá trên Cùng Tập Test 3 Lớp
- **Mô tả**: Huấn luyện mô hình Transformer trên local GPU RTX 4060 (dự phòng Colab/Kaggle nếu có lỗi driver). Lưu checkpoint tốt nhất theo Validation Macro-F1. Đánh giá trên **cùng tập `test.parquet` chuẩn của M2**.
- **Phụ thuộc**: `T3.1`.
- **Ước lượng**: **M** (2 files).
- **Files tác động**: `src/evaluation/evaluate_transformer.py`, `models/transformer_best/`, `docs/experiments/transformer_results.md`.

### Task T3.3: Phân tích Lỗi & Lập Báo cáo Đối sánh Khoa học
- **Mô tả**: Trích xuất tối thiểu 30 trường hợp dự đoán sai của mô hình Transformer trên tập Test. Phân tích nguyên nhân ngữ nghĩa (mỉa mai, ngữ cảnh thô tục không ác ý, teencode phức tạp). Lập bảng so sánh đối đầu giữa Baseline và Transformer tại `docs/experiments/model_comparison.md`.
- **Phụ thuộc**: `T3.2`, `T2.3`.
- **Ước lượng**: **S** (2 files).
- **Files tác động**: `src/evaluation/error_analysis.py`, `docs/experiments/model_comparison.md`.

---

### 🏁 Checkpoint 3: Hoàn tất Thực nghiệm Học máy Đối sánh
- [ ] Cả Baseline và Transformer đều được đánh giá trung thực trên cùng 1 tập test 3 lớp.
- [ ] Báo cáo phân tích lỗi hoàn chỉnh.

---

## MILESTONE 4: INFERENCE ENGINE ĐA BACKEND & KIỂM THỬ TÍCH HỢP (M4)

### Task T4.1: Xây dựng Interface Suy luận Đa Backend (`BasePredictor`)
- **Mô tả**: Xây dựng `src/inference/predictor.py` định nghĩa lớp cơ sở trừu tượng `BasePredictor` và các implementation: `SklearnBackend` (nạp baseline .joblib) và `PyTorchBackend` (nạp weights PyTorch). Trả về dataclass `PredictionResult` chứa nhãn 3 lớp và phân bố xác suất.
- **Phụ thuộc**: `T1.3`, `T2.2`.
- **Ước lượng**: **S** (2 files).
- **Files tác động**: `src/inference/predictor.py`, `src/inference/backends.py`.

### Task T4.2: Kiểm thử Tích hợp Cả Hai Nhóm Mô hình (Integration Testing Task)
- **Mô tả**: Xây dựng test suite `tests/test_inference_backends.py` kiểm tra suy luận trên cả `SklearnBackend` và `PyTorchBackend` (sau khi có checkpoint T3.2). Đảm bảo cả hai backend đều cho ra output tương thích 100% với REST API, xử lý chuỗi rỗng và chuỗi dài.
- **Phụ thuộc**: `T4.1`, `T3.2` (khi M3 xong) hoặc mock checkpoint.
- **Ước lượng**: **S** (1-2 files).
- **Files tác động**: `tests/test_inference_backends.py`.
- **Tiêu chí nghiệm thu**:
  - [ ] Cả 2 backend đều pass các test cases với cùng format `PredictionResult`.
  - [ ] Đo đạc latency thực tế trên CPU cho cả 2 backend.
- **Quy trình kiểm chứng**:
  - Chạy: `pytest tests/test_inference_backends.py -v`.

### Task T4.3: Benchmark Tùy chọn ONNX & Quantization (Conditional ONNX)
- **Mô tả**: Viết script `src/models/export_onnx.py`. Đo lường so sánh hiệu năng giữa PyTorch CPU và ONNX INT8 (độ trễ và RAM). Chỉ kích hoạt `ONNXBackend` nếu benchmark chứng minh giảm độ trễ ≥ 25% mà không làm giảm F1. Ghi nhận kết quả vào `docs/experiments/onnx_benchmark.md`.
- **Phụ thuộc**: `T4.2`.
- **Ước lượng**: **S** (2 files).
- **Files tác động**: `src/models/export_onnx.py`, `docs/experiments/onnx_benchmark.md`.

---

### 🏁 Checkpoint 4: Core Engine Được Kiểm thử Tích hợp Toàn diện
- [ ] Cả hai nhóm mô hình được tích hợp mượt mà qua một interface duy nhất.
- [ ] Test tích hợp pass 100%.

---

## MILESTONE 5: XÂY DỰNG ỨNG DỤNG WEB (REACT + FASTAPI) (M5)

### Task T5.1: Xây dựng Backend FastAPI & REST Endpoints
- **Mô tả**: Xây dựng ứng dụng FastAPI tại `backend/main.py` với các router:
  - `POST /api/predict`: Nhận text, trả về 3 nhãn + probabilities + latency.
  - `POST /api/predict-batch`: Nhận file CSV/TXT, trả về kết quả batch.
  - `GET /api/model-info`: Trả về thông tin mô hình và metrics.
  - `GET /api/health`: Health check.
- **Phụ thuộc**: `T4.1`.
- **Ước lượng**: **S** (3 files).
- **Files tác động**: `backend/main.py`, `backend/schemas.py`, `backend/api/routes_predict.py`, `backend/api/routes_meta.py`.

### Task T5.2: Xây dựng Giao diện React: Phân tích Bình luận Đơn lẻ (Single Comment UI)
- **Mô tả**: Xây dựng component React cho phép nhập bình luận tiếng Việt, gọi API `/api/predict`, hiển thị Badge màu theo 3 lớp (`CLEAN` xanh, `OFFENSIVE` vàng cam, `HATE` đỏ), các thanh phân bố xác suất cho cả 3 nhãn và thời gian xử lý.
- **Phụ thuộc**: `T5.1`.
- **Ước lượng**: **S** (3 files).
- **Files tác động**: `frontend/src/App.jsx`, `frontend/src/components/SingleComment.jsx`, `frontend/src/services/api.js`.

### Task T5.3: Xây dựng Giao diện React: Phân tích Tệp Hàng loạt & Model Card (Batch & Model Card UI)
- **Mô tả**: Xây dựng component upload CSV (giới hạn 500 dòng), thanh tiến trình, biểu đồ phân bố 3 lớp, nút download kết quả CSV; cùng tab Model Card hiển thị bảng đối sánh khoa học và confusion matrix.
- **Phụ thuộc**: `T5.2`.
- **Ước lượng**: **M** (3 files).
- **Files tác động**: `frontend/src/components/BatchUpload.jsx`, `frontend/src/components/ModelCard.jsx`, `frontend/src/components/Charts.jsx`.

### Task T5.4: Kiểm thử End-to-End Web & API Cục bộ
- **Mô tả**: Viết test suite `tests/test_api.py` sử dụng `fastapi.testclient.TestClient` kiểm tra toàn bộ API endpoints với cả 2 nhóm mô hình; build thử frontend React (`npm run build`) và kiểm tra FastAPI phục vụ file tĩnh.
- **Phụ thuộc**: `T5.3`.
- **Ước lượng**: **S** (2 files).
- **Files tác động**: `tests/test_api.py`.
- **Tiêu chí nghiệm thu**:
  - [ ] 100% API tests pass.
  - [ ] `npm run build` xuất thư mục `frontend/dist/` thành công không lỗi cú pháp.
- **Quy trình kiểm chứng**:
  - Chạy: `pytest tests/test_api.py -v`.
  - Chạy: `cd frontend && npm run build && cd ..`.

---

### 🏁 Checkpoint 5: Web App Hoàn chỉnh trên Môi trường Local
- [ ] Frontend React và Backend FastAPI kết nối hoàn hảo, phục vụ cả phân tích đơn lẻ và hàng loạt.

---

## MILESTONE 6: DOCKERIZATION, HUGGING FACE DEPLOYMENT & NGHIỆM THU (M6)

### Task T6.1: Xây dựng Multi-stage Dockerfile & Chạy Thử Container Cục bộ
- **Mô tả**: Viết `Dockerfile` đa tầng (Stage 1 build React -> Stage 2 copy vào FastAPI slim image), cấu hình `docker-compose.yml`, build và chạy thử nghiệm container trên port 7860.
- **Phụ thuộc**: `T5.4`.
- **Ước lượng**: **S** (2 files).
- **Files tác động**: `Dockerfile`, `docker-compose.yml`.
- **Tiêu chí nghiệm thu**:
  - [ ] Container build thành công và chạy ổn định tại port 7860.
  - [ ] Truy cập `http://localhost:7860` render đúng giao diện React và gọi được API.
- **Quy trình kiểm chứng**:
  - Chạy: `docker build -t toxic-detector:test . && docker run -d -p 7860:7860 toxic-detector:test`.

### Task T6.2: Triển khai Lên Hugging Face Spaces & Chạy Live Smoke Test
- **Mô tả**: Đẩy mã nguồn lên Hugging Face Spaces (Docker SDK), theo dõi container build log, chạy bài kiểm thử khói (Smoke Test) trực tiếp trên link public 24/7.
- **Phụ thuộc**: `T6.1`, `T0.3`.
- **Ước lượng**: **S** (2 files).
- **Files tác động**: `README.md` (HF Spaces metadata), `docs/live_smoke_test_report.md`.
- **Tiêu chí nghiệm thu**:
  - [ ] Container running trên Hugging Face Spaces.
  - [ ] Thử nghiệm 3 câu mẫu (Clean, Offensive, Hate) trên link public cho kết quả chính xác trong < 500ms.
- **Quy trình kiểm chứng**:
  - Đọc log và kiểm tra link live public.

### Task T6.3: Hoàn thiện Tài liệu Nghiệm thu, README & Hướng dẫn Sử dụng
- **Mô tả**: Tổng kết báo cáo nghiệm thu đối chiếu với toàn bộ tiêu chí AC-ML và AC-WEB; hoàn thiện `README.md` với video/ảnh chụp minh chứng, link public và hướng dẫn chạy cục bộ.
- **Phụ thuộc**: `T6.2`.
- **Ước lượng**: **S** (2 files).
- **Files tác động**: `README.md`, `docs/acceptance_report.md`.
- **Tiêu chí nghiệm thu**:
  - [ ] Đạt 100% tiêu chí nghiệm thu đề ra trong SPEC.
- **Quy trình kiểm chứng**:
  - Rà soát toàn bộ tài liệu và chạy kiểm tra tổng thể.

---

### 🏁 Checkpoint 6: Nghiệm thu Toàn diện Đồ án
- [ ] Ứng dụng live công khai 24/7 trên Hugging Face Spaces.
- [ ] Đạt điểm tối đa về tính khoa học và tính ứng dụng thực tế.
