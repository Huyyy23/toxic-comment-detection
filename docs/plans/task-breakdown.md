# Chi tiết Phân rã Công việc (Task Breakdown & Specifications)
## Vietnamese Toxic Comment Detection (3-Class: ML-first, API-minimal)

Trạng thái: **DRAFT — PENDING USER REVIEW**  
Ngày cập nhật: 2026-11-11  
Dự án: **Vietnamese Toxic Comment Detection**  
Kế hoạch tham chiếu: [`docs/plans/implementation-plan.md`](implementation-plan.md) và [`docs/plans/ml-experiments.md`](ml-experiments.md)

---

## TỔNG QUAN PHÂN RÃ CÔNG VIỆC

- **Tổng số Milestone**: 6 (`M0` đến `M6`)
- **Phương pháp**: Hướng 2 (baseline → Transformer → Ensemble nếu có bằng chứng)
- **Kiến trúc**: ML-first, API-minimal (FastAPI REST API tối giản, không frontend)
- **Quy tắc an toàn**: Mỗi task độc lập, có acceptance criteria rõ ràng, file list cụ thể và lệnh kiểm chứng dự kiến.
- **GHI RỖNG**: Phiên này Chỉ lập kế hoạch tài liệu. Không viết code, không xóa source code, không chạy training/deploy. Việc xóa `frontend/` sẽ được thực hiện ở phiên implementation sau.

---

## MILESTONE 0: AUDIT & ARCHITECTURE SIMPLIFICATION (M0)

Mục tiêu: Xác định giữ/bỏ thành phần, thiết kế cấu trúc ML-first, lập kế hoạch loại bỏ an toàn frontend và thành phần dư thừa. Không xóa code trong phiên này.

### Task T0.1: Audit Repository & Gap Analysis
- **Mục tiêu**: Xác thực trạng thái thực tế của repository (code, tài liệu, dataset, tests, config), đối chiếu với yêu cầu giảng viên, phân loại thành phần (GIỮ / CẢI TIẾN / BỔ SUNG / LOẠI BỎ).
- **Phạm vi**: Toàn bộ repo; tập trung vào `SPEC.md`, `docs/`, `src/`, `backend/`, `frontend/`, `tests/`, `configs/`, `data/`, `models/`.
- **Files dự kiến tác động**: Không sửa code; chỉ ghi nhận vào tài liệu (`docs/open-questions.md`, báo cáo audit).
- **Dependency**: Không.
- **Inputs**: Repository hiện tại.
- **Outputs**: Báo cáo kết quả audit (danh sách giữ/cải tiến/bổ sung/loại bỏ).
- **Acceptance Criteria**: Đã liệt kê đầy đủ thành phần, phân loại rõ ràng, không bỏ sót yêu cầu giảng viên.
- **Validation/Test Commands**: Không chạy code; review tay.
- **Risk**: Bỏ sót thành phần quan trọng.
- **Độ phức tạp**: S.

### Task T0.2: Thiết kế Kiến trúc ML-first & API-minimal
- **Mục tiêu**: Xác định hai khối chính (Research & Experimentation, Production Inference), ba endpoint FastAPI, chính sách Notebook, MLflow local tracking, Ensemble conditional strategy.
- **Phạm vi**: `docs/decisions.md`, `docs/deployment-plan.md`, `docs/spec.md`, `SPEC.md`.
- **Files dự kiến tác động**: Tài liệu thiết kế (đã cập nhật trong phiên này).
- **Dependency**: T0.1.
- **Inputs**: Kết quả audit T0.1, yêu cầu giảng viên.
- **Outputs**: Kiến trúc xác định, danh sách endpoint, chính sách Notebook.
- **Acceptance Criteria**: Đủ 20 mục SPEC (§14), phân biệt bắt buộc vs mở rộng, không overengineering.
- **Validation/Test Commands**: Review tài liệu đối chiếu §14 SPEC.
- **Risk**: Thiếu yêu cầu giảng viên.
- **Độ phức tạp**: M.

### Task T0.3: Lập Kế hoạch Loại bỏ An toàn Frontend & Thành phần Dư thừa
- **Mục tiêu**: Lập kế hoạch (không thực hiện xóa) loại bỏ `frontend/` (React/Vite), giữ `Dockerfile`/`docker-compose.yml` chỉ nếu hosting yêu cầu, loại bỏ auth/JWT, user DB, dashboard, CI/CD phức tạp. Nêu dependency và tests cần chạy trước khi xóa.
- **Phạm vi**: `docs/open-questions.md` (Đã đóng DD-08, DD-09), `docs/implementation-plan.md`.
- **Files dự kiến tác động**: Tài liệu kế hoạch.
- **Dependency**: T0.1, T0.2.
- **Inputs**: Danh sách thành phần dư thừa từ T0.1.
- **Outputs**: Kế hoạch loại bỏ, danh sách dependency, tests cần chạy trước khi xóa.
- **Acceptance Criteria**: Có phương án dự phòng deployment, biết clearly những gì sẽ bị ảnh hưởng khi xóa frontend.
- **Validation/Test Commands**: Review.
- **Risk**: Xóa sớm làm hỏng API nếu chưa tách rõ inference engine.
- **Độ phức tạp**: S.

#### 🏁 Checkpoint 0: Kiến trúc Xác định & Kế hoạch Loại bỏ
- [ ] Báo cáo audit hoàn tất.
- [ ] Kiến trúc ML-first, API-minimal được ghi nhận.
- [ ] Kế hoạch loại bỏ frontend/thành phần dư thừa đã nêu.

---

## MILESTONE 1: DATA & EDA (M1)

Mục tiêu: Tải dataset, kiểm tra chất lượng, EDA, chia train/val/locked-test. Liên quan EXP-001.

### Task T1.1: Tạo Script Tải Dataset & Lưu Nguyên bản
- **Mục tiêu**: Viết `src/data/download.py` tải `UIT-ViHSD` (tập chính) và `ViCTSD` (so sánh EDA) về `data/raw/`.
- **Phạm vi**: `src/data/download.py`, `data/raw/`.
- **Dependency**: M0.
- **Inputs**: URL dataset từ `link dataset.txt`.
- **Outputs**: File dữ liệu gốc trong `data/raw/`, script download.
- **Acceptance Criteria**: Script chạy được, tải thành công cả 2 dataset, không sửa đổi raw data.
- **Validation/Test Commands**: Chạy script download, kiểm tra file tồn tại (chỉ trong phiên implementation sau, không chạy trong phiên này).
- **Risk**: HF token, mạng, giới hạn tải.
- **Độ phức tạp**: S.

### Task T1.2: EDA & Xác minh Dataset (EXP-001)
- **Mục tiêu**: Thống kê số lượng mẫu, phân bố 3 lớp, độ dài chuỗi (p95), teencode/slang, near-duplicate, overlap split. Xuất `docs/eda_report.md`.
- **Phạm vi**: EDA notebook hoặc script, `docs/eda_report.md`.
- **Dependency**: T1.1.
- **Inputs**: `data/raw/`.
- **Outputs**: `docs/eda_report.md`, chốt `max_length`, quyết định class weighting.
- **Acceptance Criteria**: Có thống kê phân bố nhãn, p95 độ dài, kiểm tra leakage, quyết định dataset chính.
- **Validation/Test Commands**: Review báo cáo EDA.
- **Risk**: Dataset cũ không tổng quát ngôn ngữ mới.
- **Độ phức tạp**: S.

### Task T1.3: Xây dựng Module Tiền xử lý Tiếng Việt (EXP-002 phần dữ liệu)
- **Mục tiêu**: Viết `src/data/preprocessor.py` (Unicode NFC, khoảng trắng, teencode dictionary), `src/data/splitter.py` (stratified 70/15/15, locked test, assert không overlap).
- **Phạm vi**: `src/data/preprocessor.py`, `src/data/splitter.py`, `data/processed/*.parquet`.
- **Dependency**: T1.2.
- **Inputs**: `data/raw/`, kết quả EDA.
- **Outputs**: `data/processed/train.parquet`, `val.parquet`, `test.parquet` (LOCKED).
- **Acceptance Criteria**: Tập test bị khóa, `set(train) ∩ set(test) == 0`, `set(train) ∩ set(val) == 0`, stratified 3 lớp.
- **Validation/Test Commands**: `tests/test_leakage.py` (dự kiến).
- **Risk**: Lỗi split làm rò rỉ dữ liệu.
- **Độ phức tạp**: M.

#### 🏁 Checkpoint 1: Dữ liệu Sạch & Tập Test Đã Khóa
- [ ] Báo cáo EDA hoàn tất, chốt `max_length`.
- [ ] Tập test chuẩn 3 lớp duy nhất tại `data/processed/test.parquet`, bị khóa, không rò rỉ.

---

## MILESTONE 2: BASELINE MODELS (TRADITIONAL ML) (M2)

Mục tiêu: Huấn luyện và tuning Logistic Regression + LinearSVC calibrated trên Validation. Liên quan EXP-002, EXP-003.

### Task T2.1: Xây dựng Module Đánh giá Đa chiều 3 Lớp
- **Mục tiêu**: Viết `src/evaluation/metrics.py` (Macro-F1, Per-class Precision/Recall/F1, Confusion Matrix, Latency).
- **Phạm vi**: `src/evaluation/metrics.py`.
- **Dependency**: T1.3.
- **Inputs**: Train/val predictions.
- **Outputs**: Module metrics, `tests/test_metrics.py` (dự kiến).
- **Acceptance Criteria**: Tính đúng Macro-F1, per-class F1, confusion matrix cho 3 lớp.
- **Validation/Test Commands**: Chạy unit test metrics (phiên sau).
- **Risk**: Sai công thức metric.
- **Độ phức tạp**: S.

### Task T2.2: Huấn luyện Baseline TF-IDF + Logistic Regression & LinearSVC Calibrated
- **Mục tiêu**: Viết `src/models/baseline.py` (TF-IDF word/char, LR `predict_proba`, LinearSVC bọc `CalibratedClassifierCV(cv=5, method='sigmoid')`).
- **Phạm vi**: `src/models/baseline.py`, `configs/baseline_config.yaml`.
- **Dependency**: T2.1.
- **Inputs**: `data/processed/train.parquet`.
- **Outputs**: Model artifacts (.joblib), calibration artifacts.
- **Acceptance Criteria**: Hợp đồng xác suất (sum ≈ 1.0), LinearSVC dùng CalibratedClassifierCV.
- **Validation/Test Commands**: Kiểm tra xác suất tổng ≈ 1.0 (phiên sau).
- **Risk**: LinearSVC decision score bị hiểu nhầm là xác suất.
- **Độ phức tạp**: M.

### Task T2.3: Tuning Hyperparameters & Chọn Baseline Tốt nhất trên Validation (EXP-003)
- **Mục tiêu**: Grid search `C`, `max_features`, n-grams trên Validation. Chốt Best Baseline theo Validation Macro-F1. Lưu `models/baselines/baseline_model.joblib`.
- **Phạm vi**: `src/evaluation/evaluate_baseline_val.py`, `docs/experiments/exp-003-baseline-val-results.md`.
- **Dependency**: T2.2.
- **Inputs**: Train/val, configs.
- **Outputs**: Best baseline artifact, báo cáo kết quả val.
- **Acceptance Criteria**: Chọn được Best Baseline, không đụng tập test, MLflow local log run.
- **Validation/Test Commands**: Chạy evaluation trên val (phiên sau).
- **Risk**: Tuning nhầm lên test (data leakage).
- **Độ phức tạp**: M.

#### 🏁 Checkpoint 2: Hoàn tất Baseline Trên Validation Set
- [ ] Best Baseline 3 lớp được lựa chọn trên Validation, lưu tại `models/baselines/`.
- [ ] Hợp đồng xác suất hiệu chuẩn đạt chuẩn.

---

## MILESTONE 3: ADVANCED TRANSFORMER TRAINING (M3)

Mục tiêu: Fine-tune PhoBERT và ViSoBERT trên GPU. Liên quan EXP-004, EXP-005. Có thể chạy song song một phần với M4 sau khi có checkpoint.

### Task T3.1: Xây dựng Pipeline Fine-tuning Transformer Tối ưu VRAM
- **Mục tiêu**: Viết `src/models/transformer.py` (PyTorch/HF Trainer, fp16, gradient accumulation, class-weighted loss, early stopping).
- **Phạm vi**: `src/models/transformer.py`, `configs/transformer_config.yaml`, `src/models/dataset_torch.py` (dự kiến).
- **Dependency**: T2.1 (metrics), T1.3 (preprocessing).
- **Inputs**: Train/val processed.
- **Outputs**: Pipeline code, config [PROVISIONAL].
- **Acceptance Criteria**: Pipeline chạy trên CPU/GPU, hỗ trợ mixed precision, early stopping theo Validation Macro-F1.
- **Validation/Test Commands**: Dry-run trên CPU nhỏ (phiên sau).
- **Risk**: OOM trên RTX 4060 8GB.
- **Độ phức tạp**: L.

### Task T3.2: Fine-tune PhoBERT & Chọn Checkpoint Tốt nhất trên Validation (EXP-004)
- **Mục tiêu**: Huấn luyện `vinai/phobert-base-v2` trên Train, chọn checkpoint theo Validation Macro-F1. Grid search lr, batch, max_length, weight_decay.
- **Phạm vi**: `models/transformer_best/phobert/`, `docs/experiments/exp-004-phobert-val-results.md`.
- **Dependency**: T3.1.
- **Inputs**: Train/val, config.
- **Outputs**: Checkpoint tốt nhất, báo cáo val.
- **Acceptance Criteria**: Chọn checkpoint tốt nhất theo Val Macro-F1, MLflow local log run, không đụng test.
- **Validation/Test Commands**: Chạy training trên GPU (phiên sau).
- **Risk**: OOM, kết quả kém baseline.
- **Độ phức tạp**: L (GPU).

### Task T3.3: Fine-tune ViSoBERT & Chọn Checkpoint Tốt nhất trên Validation (EXP-005)
- **Mục tiêu**: Huấn luyện `uitnlp/visobert` trên Train, cùng không gian hyperparameters với PhoBERT để so sánh công bằng.
- **Phạm vi**: `models/transformer_best/visobert/`, `docs/experiments/exp-005-visobert-val-results.md`.
- **Dependency**: T3.1.
- **Inputs**: Train/val, config.
- **Outputs**: Checkpoint tốt nhất, báo cáo val.
- **Acceptance Criteria**: Chọn Best Transformer (PhoBERT hoặc ViSoBERT) theo Val Macro-F1.
- **Validation/Test Commands**: Chạy training trên GPU (phiên sau).
- **Risk**: OOM, ViSoBERT không phải lúc nào cũng tốt hơn PhoBERT.
- **Độ phức tạp**: L (GPU).

#### 🏁 Checkpoint 3: Hoàn tất Thực nghiệm Transformer trên Validation Set
- [ ] Best Transformer checkpoint được lựa chọn hoàn toàn dựa trên Validation.
- [ ] So sánh Best Transformer vs Best Baseline trên Validation.

---

## MILESTONE 4: ENSEMBLE & ABLATION STUDY (M4)

Mục tiêu: Phân tích lỗi, ablation, ensemble/stacking nếu có giá trị. Liên quan EXP-006, EXP-007, EXP-008.

### Task T4.1: Error Analysis trên Validation (EXP-006)
- **Mục tiêu**: Dự đoán trên Validation với Best Transformer, trích xuất ít nhất 30 mẫu sai, phân loại lỗi (teencode, slang, emoji, ngữ cảnh, OOD).
- **Phạm vi**: `src/evaluation/error_analysis.py`, `docs/experiments/exp-006-error-analysis-val.md`.
- **Dependency**: T3.2/T3.3 (Best Transformer).
- **Inputs**: Validation predictions.
- **Outputs**: Danh sách mẫu lỗi, báo cáo phân tích.
- **Acceptance Criteria**: Có nhóm lỗi chủ đạo, liên kết với giả thuyết vòng tiếp theo.
- **Validation/Test Commands**: Review báo cáo.
- **Risk**: Mẫu lỗi không đại diện.
- **Độ phức tạp**: S.

### Task T4.2: Ablation Study (EXP-008)
- **Mục tiêu**: Tắt từng thành phần (preprocessing, char n-grams, class weighting, hyperparameter tuning) đo tác động đến Macro-F1.
- **Phạm vi**: `docs/experiments/exp-008-ablation-study.md`.
- **Dependency**: T2.3, T3.2/T3.3.
- **Inputs**: Best model.
- **Outputs**: Báo cáo ablation, quyết định giữ/bỏ thành phần.
- **Acceptance Criteria**: Có số liệu so sánh trước/sau cho từng thành phần.
- **Validation/Test Commands**: Chạy vài lần train lại nhỏ (phiên sau).
- **Risk**: Kết quả ablation không rõ ràng.
- **Độ phức tạp**: S.

### Task T4.3: Ensemble / Stacking nếu có giá trị (EXP-007 — conditional, không bắt buộc)
- **Mục tiêu**: Thử Soft Voting / Weighted Soft Voting / Stacking (out-of-fold) kết hợp Best Baseline + Best Transformer. Chỉ triển khai nếu cải thiện ≥ 1% Macro-F1 so với Best Transformer đơn.
- **Phạm vi**: `docs/experiments/exp-007-ensemble-results.md`.
- **Dependency**: T2.3, T3.2/T3.3.
- **Inputs**: Best baseline, best transformer predictions.
- **Outputs**: Báo cáo ensemble, quyết định final model.
- **Acceptance Criteria**: Nếu có ensemble, meta-learner không học từ test; so sánh với mô hình đơn tốt nhất.
- **Validation/Test Commands**: Chạy inference ensemble (phiên sau).
- **Risk**: Stacking quá tốn tài nguyên, không mang lại lợi ích.
- **Độ phức tạp**: M.

#### 🏁 Checkpoint 4: Ensemble & Ablation Hoàn tất
- [ ] Báo cáo error analysis và ablation hoàn tất.
- [ ] Quyết định final model (đơn hoặc ensemble).

---

## MILESTONE 5: EVALUATION & RESEARCH DOCUMENTATION (M5)

Mục tiêu: Đánh giá cuối cùng trên Locked Test, tổng hợp báo cáo nghiên cứu, Notebook. Liên quan EXP-009.

### Task T5.1: Đánh giá Locked Test Final (EXP-009)
- **Mục tiêu**: Mở `test.parquet` đúng một lần, chạy Best Baseline, Best Transformer, Best Ensemble (nếu có). Báo cáo trung tính Macro-F1, per-class F1, confusion matrix, latency, size.
- **Phạm vi**: `src/evaluation/evaluate_locked_test.py`, `docs/experiments/exp-009-final-test-benchmark.md`.
- **Dependency**: T4.3 (final model đã chọn).
- **Inputs**: `data/processed/test.parquet` (LOCKED).
- **Outputs**: Báo cáo đánh giá cuối cùng, bảng so sánh đối đầu.
- **Acceptance Criteria**: Test chỉ mở một lần, báo cáo trung tính bất kỳ mô hình nào cao hơn, MLflow local log run.
- **Validation/Test Commands**: Chạy inference trên test (phiên sau).
- **Risk**: Lạm dụng test set để chọn mô hình.
- **Độ phức tạp**: S.

### Task T5.2: Tổng hợp Notebook Báo cáo Nghiên cứu
- **Mục tiêu**: Thiết kế nội dung 6 notebook (`01_eda` … `06_error_analysis`), chỉ import module `.py`, đọc kết quả MLflow, trình bày biểu đồ, confusion matrix, MLflow Run ID, kết luận.
- **Phạm vi**: `notebooks/` (thiết kế nội dung, không tạo file `.ipynb` trong phiên này).
- **Dependency**: T1.2, T2.3, T3.2/T3.3, T4.1, T5.1.
- **Inputs**: Kết quả experiments, MLflow runs.
- **Outputs**: Cấu trúc notebook, mẫu nội dung Markdown.
- **Acceptance Criteria**: Mỗi notebook có giải thích, biểu đồ, bảng, confusion matrix, MLflow Run ID, kết luận; không tạo số liệu giả; không nhúng checkpoint.
- **Validation/Test Commands**: Review thiết kế.
- **Risk**: Notebook thành phần production (vi phạm chính sách).
- **Độ phức tạp**: M.

#### 🏁 Checkpoint 5: Đánh giá & Tài liệu Nghiên cứu Hoàn tất
- [ ] Báo cáo locked test hoàn tất, trung tính.
- [ ] Thiết kế notebook nghiên cứu hoàn tất.

---

## MILESTONE 6: REST API DEPLOYMENT (API-MINIMAL) (M6)

Mục tiêu: Triển khai FastAPI REST API tối giản (3 endpoint), kiểm thử inference, đặc tả deployment. Không tạo Dockerfile/FastAPI code trong phiên này (chỉ đặc tả).

### Task T6.1: Xây dựng Inference Pipeline (.py)
- **Mục tiêu**: Viết `src/inference/predictor.py` (load best model/ensemble, tokenizer, preprocessing, label mapping, probability contract).
- **Phạm vi**: `src/inference/predictor.py`.
- **Dependency**: T5.1 (final model đã chọn).
- **Inputs**: Model artifacts, preprocessing artifacts.
- **Outputs**: Inference pipeline code.
- **Acceptance Criteria**: Trả về đúng schema `PredictionResponse`, tổng xác suất ≈ 1.0.
- **Validation/Test Commands**: Chạy inference thử (phiên sau).
- **Risk**: Model artifact thiếu, pipeline không nhất quán training.
- **Độ phức tạp**: M.

### Task T6.2: Xây dựng FastAPI API Tối giản (3 endpoint)
- **Mục tiêu**: Viết `api/main.py` (hoặc `backend/main.py`) với `POST /predict`, `GET /health`, `GET /model-info`; Swagger UI tại `/docs`.
- **Phạm vi**: `api/main.py` / `backend/main.py`, `backend/schemas.py` (đã có).
- **Dependency**: T6.1.
- **Inputs**: Inference pipeline, schemas.
- **Outputs**: FastAPI app.
- **Acceptance Criteria**: 3 endpoint hoạt động, Swagger UI truy cập được, validation input qua Pydantic.
- **Validation/Test Commands**: `tests/test_api.py` (dự kiến, phiên sau).
- **Risk**: Schema không khớp inference output.
- **Độ phức tạp**: M.

### Task T6.3: Kiểm thử Inference & API
- **Mục tiêu**: Viết `tests/test_api.py` kiểm tra endpoints, xác suất hiệu chuẩn, error handling.
- **Phạm vi**: `tests/test_api.py`.
- **Dependency**: T6.2.
- **Inputs**: FastAPI app.
- **Outputs**: Test suite.
- **Acceptance Criteria**: Test pass cho predict/health/model-info, kiểm tra hợp đồng xác suất.
- **Validation/Test Commands**: `pytest tests/test_api.py` (phiên sau).
- **Risk**: Test không覆盖 edge cases.
- **Độ phức tạp**: S.

### Task T6.4: Đặc tả Deployment & Kế hoạch Public
- **Mục tiêu**: Ghi rõ nền tảng triển khai (HF Spaces / Render / Railway / local Docker), chiến lược nạp artifact, fallback baseline, phương án dự phòng nếu HF token thiếu. Docker chỉ giữ nếu hosting yêu cầu.
- **Phạm vi**: `docs/deployment-plan.md`, `docs/hf_eligibility_report.md`, `docs/open-questions.md`.
- **Dependency**: T6.2, T6.3.
- **Inputs**: Kết quả kiểm thử API.
- **Outputs**: Tài liệu deployment, danh sách rủi ro.
- **Acceptance Criteria**: Có phương án dự phòng, biết rõ ràng constraint hosting.
- **Validation/Test Commands**: Review.
- **Risk**: HF token thiếu, yêu cầu thẻ tín dụng.
- **Độ phức tạp**: S.

#### 🏁 Checkpoint 6: REST API Hoàn tất & Đặc tả Deployment
- [ ] Inference pipeline và FastAPI API hoạt động.
- [ ] Kiểm thử inference pass.
- [ ] Tài liệu deployment và phương án dự phòng hoàn tất.

---

## MA TRẬN RỦI RO & PHƯƠNG ÁN XỬ LÝ

| Mã | Rủi ro | Mức độ | Giảm thiểu |
|---|---|---|---|
| R1 | HF Token chưa có / yêu cầu thẻ tín dụng | Cao | Phương án dự phòng: Render, Railway, local Docker. |
| R2 | Mất cân bằng 3 lớp (HATE ít mẫu) | Cao | Class-weighted loss, Macro-F1, stratified sampling. |
| R3 | Transformer OOM trên GPU 8GB | Trung bình | Giảm `max_length`, `batch_size`; fp16; Colab/Kaggle backup. |
| R4 | Dùng Test set để tuning (data leakage) | Chí mạng | Khóa test; chọn mô hình trên Validation; mở test một lần. |
| R5 | Dataset cũ không tổng quát ngôn ngữ mới | Trung bình | Đánh giá cross-dataset, ghi nhận giới hạn. |
| R6 | Ensemble không cải thiện ≥ 1% Macro-F1 | Trung bình | Giữ mô hình đơn tốt nhất; Soft Voting làm thay thế. |

## NHỮNG VIỆC KHÔNG LÀM TRONG PHIÊN NÀY (XÁC NHẬN)

- Không viết code Python.
- Không sửa training pipeline.
- Không refactor source code.
- Không xóa React/frontend (chỉ lập kế hoạch xóa ở phiên sau).
- Không cài đặt package.
- Không thay đổi môi trường.
- Không download dataset.
- Không xử lý dữ liệu.
- Không chạy training.
- Không chạy inference.
- Không chạy experiment.
- Không tạo Notebook thực thi.
- Không tạo API implementation.
- Không tạo Dockerfile.
- Không public deployment.
- Không git commit/push nếu chưa được yêu cầu.

Chỉ tạo/chỉnh sửa file tài liệu Markdown: SPEC.md, docs/spec.md, docs/plans/*.md, docs/decisions.md, docs/deployment-plan.md, docs/open-questions.md, docs/hf_eligibility_report.md, docs/dataset-requirements.md.
