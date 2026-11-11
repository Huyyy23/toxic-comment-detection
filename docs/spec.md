# ĐẶC TẢ HỆ THỐNG: NHẬN DẠNG BÌNH LUẬN ĐỘC HẠI TIẾNG VIỆT
## (Vietnamese Toxic Comment Detection System - Specification)

- **Mã tài liệu**: `SPEC-TOXIC-COMMENT-2026`
- **Phiên bản**: `2.1.0-REVISED`
- **Trạng thái**: `APPROVED & SYNCHRONIZED`
- **Tác giả**: Senior ML Engineer / NLP Researcher / Software Architect
- **Ngày cập nhật**: 2026-11-11
- **Phân loại**: Đồ án tốt nghiệp / Đồ án cuối kỳ môn Xử lý Ngôn ngữ Tự nhiên & Ứng dụng Web

---

## 1. MỤC TIÊU & PHẠM VI DỰ ÁN (Objective & Scope)

### 1.1. Bối cảnh & Mục tiêu
- **Mục tiêu**: Xây dựng hệ thống Machine Learning hoàn chỉnh từ khâu nghiên cứu dữ liệu tiếng Việt, đối sánh khoa học trung thực giữa mô hình truyền thống (Traditional ML) và mô hình học sâu Transformer trên cùng một tập test chuẩn 3 lớp, đóng gói inference engine đa backend, và triển khai ứng dụng Web hiện đại (React frontend + FastAPI backend trong Docker container) lên Hugging Face Spaces.
- **Đối tượng người dùng**:
  1. *Giảng viên & Hội đồng chấm điểm*: Cần kiểm chứng phương pháp khoa học, số liệu thực nghiệm trung thực trên cùng một tập test chuẩn, độ trễ và khả năng demo trực tiếp.
  2. *Người dùng & Kiểm duyệt viên (Content Moderators)*: Cần giao diện web mượt mà, phản hồi nhanh để kiểm tra câu đơn lẻ hoặc tải file CSV hàng loạt.

### 1.2. Phân loại Trạng thái Thông tin (Information Ledger)
- **`[Confirmed]`**: Yêu cầu đã được phê duyệt:
  - Dataset: Tải cả `ViCTSD` và `UIT-ViHSD` cho EDA; ưu tiên `UIT-ViHSD` làm tập chính.
  - Nhãn phân loại: **3 lớp CLEAN (0) / OFFENSIVE (1) / HATE (2)**. Không tự gộp về Binary.
  - Mô hình: Triển khai cả Traditional ML (TF-IDF + Linear Classifier) và Transformer (PhoBERT/ViSoBERT).
  - Web & Deploy: **Docker container với React + FastAPI**, triển khai trên **Hugging Face Spaces**.
  - Phần cứng huấn luyện: **Local GPU NVIDIA RTX 4060 8GB** (Google Colab/Kaggle là backup).
  - Inference: Hỗ trợ đa backend (`SklearnBackend`, `PyTorchBackend`, `ONNXBackend` tùy chọn).
  - Giao thức Đánh giá: **Validation set dùng cho tuning và model selection; Test set được KHÓA (LOCKED) chỉ dùng đánh giá cuối cùng**.
  - Hợp đồng Xác suất: Xác suất phải được hiệu chuẩn (calibrated), sum $\approx 1.0$, không giả mạo confidence.
- **`[Provisional]`**: Các siêu tham số chưa qua kiểm chứng thực tế:
  - `max_length = 160`, `batch_size = 16`, `learning_rate = 2e-5`, `VRAM budget < 7.0GB`, TF-IDF `max_features = 10000`. (Sẽ chốt sau EDA và hardware profiling).
- **`[Blocked / Pending Token]`**: Xác minh tài khoản Hugging Face Docker Space đang chờ cấu hình `HF_TOKEN`.

---

## 2. GIAO THỨC ĐÁNH GIÁ KHOA HỌC & KHÓA TẬP TEST (Evaluation Protocol)

### 2.1. Phân định Vai trò 3 Tập Dữ liệu
1. **Tập Huấn luyện (Train Set - 70%)**:
   - Dùng để tối ưu hóa trọng số mô hình (weights) và huấn luyện bộ trích xuất đặc trưng TF-IDF.
2. **Tập Thẩm định (Validation Set - 15%)**:
   - Dùng để điều chỉnh siêu tham số (Hyperparameter Tuning): thử nghiệm các giá trị $C$, learning rate, batch size.
   - Dùng cho Early Stopping trong quá trình fine-tuning Transformer.
   - Dùng để **Lựa chọn Mô hình (Model Selection)**: chọn ra mô hình tốt nhất trong từng họ mô hình (Best Baseline và Best Transformer).
   - Dùng để hiệu chuẩn xác suất (Probability Calibration) nếu cần.
3. **Tập Kiểm thử Độc lập (Locked Test Set - 15%)**:
   - **BỊ KHÓA HOÀN TOÀN** trong suốt quá trình phát triển và tinh chỉnh.
   - Chỉ được mở ra **ĐÚNG MỘT LẦN** ở giai đoạn đánh giá cuối cùng để đo đạc và lập bảng so sánh đối đầu khoa học.
   - **Quy tắc tuyệt đối**: Tuyệt đối không dùng kết quả trên tập Test để quay lại điều chỉnh tham số, chọn mô hình hay sửa đổi pipeline.

---

## 3. HỢP ĐỒNG API XÁC SUẤT & THIẾT KẾ ĐA BACKEND

### 3.1. Hợp đồng Xác suất Nghiêm ngặt (Probability Contract)
- Mọi mô hình khi trả về kết quả dự đoán qua API đều phải tuân thủ schema:
  - `label_id`: int (0, 1, 2)
  - `label_name`: str ("CLEAN", "OFFENSIVE", "HATE")
  - `confidence`: float trong đoạn $[0.0, 1.0]$, tương ứng với xác suất của nhãn được dự đoán.
  - `probabilities`: dictionary chứa đủ 3 nhãn, đảm bảo $\sum P(\text{label}) \approx 1.0$ (dung sai $\pm 0.05$).
  - `is_calibrated`: bool, xác nhận xác suất là giá trị hiệu chuẩn thực sự.
- **Cơ chế cho từng mô hình**:
  - `Logistic Regression`: Sử dụng `.predict_proba()` chính thống (Softmax/Sigmoid).
  - `LinearSVC`: Bản chất hàm `decision_function()` chỉ trả về khoảng cách siêu phẳng (uncalibrated margins). Khi sử dụng LinearSVC, **bắt buộc phải bọc qua `CalibratedClassifierCV(estimator=LinearSVC(...), cv=5, method='sigmoid')`** (Platt Scaling) để sinh xác suất hiệu chuẩn thực sự. Tuyệt đối không dùng các hàm softmax tự chế trên khoảng cách chưa chuẩn hóa để giả tạo confidence.
  - `Transformer (PyTorch / ONNX)`: Sử dụng Softmax trên logits đầu ra để tính xác suất.

---

## 4. QUY TRÌNH GIAO NHẬN & VERSION HÓA MODEL ARTIFACTS CHO PRODUCTION DOCKER

Để đảm bảo container trên Hugging Face Spaces luôn có model hoạt động và không bao giờ crash vì thiếu file:

1. **Thứ tự Ưu tiên Nạp Model (Artifact Precedence)**:
   - *Ưu tiên 1*: Kiểm tra file artifact cục bộ tại `models/exported/` (`model.onnx` hoặc `pytorch_model.bin`).
   - *Ưu tiên 2*: Nếu chưa có, script `src/tools/download_artifacts.py` tải artifact được version hóa từ Hugging Face Model Hub (dựa trên commit hash hoặc git tag cố định như `v1.0.0`).
   - *Ưu tiên 3 (Dự phòng tin cậy)*: Nếu mạng lỗi hoặc chưa kịp tải Transformer, hệ thống tự động fallback nạp mô hình truyền thống đóng gói sẵn tại `models/baselines/baseline_model.joblib` (<15MB, được build trực tiếp vào Docker image).
2. **Kiểm tra Sức khỏe Container (`/api/health`)**:
   - Endpoint `/api/health` kiểm tra trạng thái thực tế của model: trả về `model_loaded: true` cùng tên backend đang phục vụ (`pytorch`, `onnx`, hoặc `sklearn_fallback`).

---

## 5. BẢN ĐỒ KHẢ NĂNG & THỨ TỰ THỰC HIỆN ĐIỀU CHỈNH (Capability Map)

| Module ID | Tên Khả năng | Trách nhiệm | Phụ thuộc |
| :--- | :--- | :--- | :--- |
| `mod-setup` | **Setup & HF Verification** | Khởi tạo cấu trúc thư mục, testing scaffold, ghi nhận trạng thái HF Docker Space. | — |
| `mod-data` | **Data Pipeline & EDA** | Tải raw data (`ViCTSD`, `UIT-ViHSD`), EDA, tiền xử lý Unicode/teencode, chuẩn hóa 3 lớp, tạo train/val/locked-test. | `mod-setup` |
| `mod-baseline` | **Traditional ML (Val Tuned)** | Huấn luyện và tinh chỉnh TF-IDF + Classifier trên Train/Val; áp dụng calibration cho LinearSVC. | `mod-data` |
| `mod-base-int` | **Baseline Integration Test** | Tích hợp Baseline vào `BasePredictor`, kiểm thử tích hợp sớm, unblock Web MVP. | `mod-baseline` |
| `mod-web` | **Web App (React + FastAPI)** | Xây dựng REST API và giao diện React (Single, Batch CSV, Model Card). | `mod-base-int` |
| `mod-trans` | **Transformer on RTX 4060** | Fine-tune Transformer trên Train, chọn checkpoint tốt nhất theo Validation Macro-F1. | `mod-data` |
| `mod-trans-int`| **Transformer Integration Test**| Kiểm thử tích hợp Transformer checkpoint vào `PyTorchBackend`, so sánh latency/memory. | `mod-trans` & `mod-base-int` |
| `mod-onnx` | **Conditional ONNX Benchmark** | Đo đạc benchmark PyTorch vs ONNX trên CPU; chỉ kích hoạt khi có lợi ích thực tế. | `mod-trans-int` |
| `mod-test-eval`| **Locked Test Evaluation** | Mở tập Test đã khóa đúng 1 lần; đối sánh khách quan và lập báo cáo trung thực. | `mod-baseline` & `mod-trans` |
| `mod-deploy` | **Docker & HF Deployment** | Multi-stage Docker, nạp artifact có version hóa, deploy lên HF Space và smoke test. | `mod-web` & `mod-test-eval` |

---

## 6. CẤU TRÚC THƯ MỤC DỰ ÁN

```
Detecting Toxic Comment/
├── backend/                    # FastAPI Backend
│   ├── api/routes_predict.py, routes_meta.py
│   ├── main.py
│   └── schemas.py              # Pydantic schemas với strict probability contract
├── frontend/                   # React Frontend (Vite)
├── configs/
│   ├── baseline_config.yaml    # Hyperparameters TF-IDF & Calibration [PROVISIONAL]
│   └── transformer_config.yaml # Hyperparameters Transformer RTX 4060 [PROVISIONAL]
├── data/
│   ├── raw/                    # Dữ liệu gốc
│   └── processed/              # train.parquet, val.parquet, test.parquet (LOCKED)
├── docs/                       # Tài liệu kỹ thuật & Báo cáo
│   ├── spec.md, decisions.md, dataset-requirements.md, deployment-plan.md, hf_eligibility_report.md
│   ├── plans/                  # implementation-plan.md, task-breakdown.md, ml-experiments.md
│   └── experiments/            # Báo cáo thực nghiệm validation và locked test
├── models/
│   ├── baselines/              # Model joblib truyền thống có calibration
│   ├── transformer_best/       # Checkpoints transformer PyTorch
│   └── exported/               # Artifact phục vụ production
├── src/                        # Core ML Package
│   ├── data/                   # download, preprocess, split
│   ├── models/                 # baseline train, transformer train, export_onnx
│   ├── evaluation/             # metrics 3 lớp, error analysis, locked test evaluation
│   ├── inference/              # BasePredictor, SklearnBackend, PyTorchBackend, ONNXBackend
│   └── tools/                  # check_hf_eligibility, download_artifacts
├── tests/
│   ├── test_smoke.py           # Scaffold & schema smoke tests
│   ├── test_preprocessing.py   # Test làm sạch tiếng Việt
│   ├── test_leakage.py         # Test chống rò rỉ dữ liệu
│   ├── test_baseline_integration.py    # Test tích hợp Baseline riêng biệt
│   ├── test_transformer_integration.py # Test tích hợp Transformer riêng biệt
│   └── test_api.py             # Test REST API endpoints
├── Dockerfile                  # Multi-stage Dockerfile
├── requirements.txt, requirements-dev.txt
├── setup.cfg
└── SPEC.md
```

---

## 7. TIÊU CHÍ NGHIỆM THU DỰ ÁN (Acceptance Criteria)

- [ ] **AC-ML-01**: Cả hai nhóm mô hình (Traditional ML và Transformer tiếng Việt) được huấn luyện trên tập Train và lựa chọn siêu tham số tối ưu hoàn toàn trên tập **Validation**.
- [ ] **AC-ML-02**: Đảm bảo hợp đồng xác suất nghiêm ngặt: toàn bộ các lớp trả về xác suất hiệu chuẩn (calibrated probabilities), tổng $\approx 1.0$, không sử dụng confidence giả tạo.
- [ ] **AC-ML-03**: **Đánh giá đối kháng khách quan**: Tiến hành đánh giá cả hai mô hình trên cùng một tập **Test đã khóa (Locked Test Set)**; báo cáo trung thực toàn bộ chỉ số (Macro-F1, Per-class F1, Latency, Model Size) và phân tích nguyên nhân/đánh đổi (trade-offs) bất kể Transformer cao hơn, tương đương hay thấp hơn Baseline.
- [ ] **AC-ML-04**: Đảm bảo 100% không rò rỉ dữ liệu giữa Train/Val/Test (được chứng minh bằng test tự động).
- [ ] **AC-WEB-01**: Ứng dụng có backend FastAPI và frontend React hoạt động mượt mà, hỗ trợ phân tích đơn lẻ và tải file CSV hàng loạt (≤ 500 dòng).
- [ ] **AC-WEB-02**: Có trang Model Card hiển thị báo cáo khoa học trực tiếp.
- [ ] **AC-WEB-03**: Có quy trình version hóa và tải artifact rõ ràng; container Docker luôn nạp được model (hoặc fallback model) khi khởi động trong môi trường production.
