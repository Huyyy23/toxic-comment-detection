# Kế hoạch Thực nghiệm Học máy (ML Experiments Plan)
## Vietnamese Toxic Comment Detection (3-Class Classification)

**Mã tài liệu**: `PLAN-ML-EXP-2026`  
**Phiên bản**: `3.0.0-ML-FIRST`  
**Trạng thái**: `DRAFT — PENDING USER REVIEW`  
**Ngày cập nhật**: 2026-11-11  
**Dự án**: Vietnamese Toxic Comment Detection  

---

## 1. NGUYÊN TẮC THỰC NGHIỆM KHOA HỌC

### 1.1. Vai trò các phân vùng dữ liệu (bắt buộc)

| Tập | Tỷ lệ | Vai trò |
|---|---|---|
| **Train** | 70% | Huấn luyện trọng số mô hình; TF-IDF vectorizer chỉ `fit` trên Train. |
| **Validation** | 15% | **DUY NHẤT** được dùng để tuning hyperparameters, early stopping, calibration, và model selection. |
| **Locked Test** | 15% | **Bị khóa hoàn toàn** trong toàn bộ quá trình nghiên cứu. Chỉ mở **đúng một lần** ở bước đánh giá cuối cùng (T6.1). Tuyệt đối không dùng kết quả Test để tối ưu ngược lại mô hình, chọn tham số hoặc sửa pipeline. |

### 1.2. Không gian nhãn (3 lớp, không đổi trừ khi có bằng chứng)

- `0: CLEAN` — lành mạnh / không độc hại.
- `1: OFFENSIVE` — xúc phạm, thô tục, chửi thề.
- `2: HATE` — thù ghét, kỳ thị, công kích định danh nghiêm trọng.
- Tuyệt đối không chuyển về nhị phân (Binary) trừ khi có lý do kỹ thuật và bằng chứng rõ ràng, ghi nhận trong SPEC.

### 1.3. Chống rò rỉ thông tin (Leakage Prevention)

- TF-IDF vectorizer chỉ `fit` trên tập Train.
- Kiểm tra giao thoa văn bản: `set(train_texts) ∩ set(test_texts) == 0` và `set(train_texts) ∩ set(val_texts) == 0`.
- Deduplicate (và near-duplicate khi có công cụ) trước khi split.
- Hyperparameter tuning chỉ trên Validation, không đụng Test.
- Ensemble out-of-fold predictions hợp lệ; meta-learner không học từ Test set.

### 1.4. Hợp đồng xác suất (Probability Contract)

- Logistic Regression: dùng `.predict_proba()` chính thống.
- LinearSVC: bắt buộc `CalibratedClassifierCV(estimator=LinearSVC(...), cv=5, method='sigmoid')` (Platt scaling). Không tự viết softmax trên decision scores.
- Transformer: Softmax trên logits.
- Tổng xác suất 3 lớp $\approx 1.0$ (dung sai $\pm 0.05$).

### 1.5. Metric chính và bổ trợ

- **Metric chính**: Macro-F1 (dùng để chọn mô hình).
- Metric bổ trợ: Weighted-F1, Per-class Precision/Recall/F1 (CLEAN / OFFENSIVE / HATE), Confusion Matrix, Balanced Accuracy, Training Time, Inference Latency (ms), Model Size (MB).
- Không dùng accuracy đơn lẻ làm căn cứ đánh giá chính.

### 1.6. Phân biệt loại kết quả

Mọi kết quả trong tài liệu phải được gán nhãn rõ ràng một trong các loại sau:

- **Đã đo (Measured)**: số liệu thực sau khi chạy training/evaluation.
- **Dự kiến (Expected)**: ước lượng, chưa chạy.
- **Giả thuyết (Hypothesis)**: giả định cần kiểm chứng.
- **Mục tiêu (Target)**: ngưỡng mong muốn đạt được.
- **Chưa xác minh (Unverified)**: chưa có bằng chứng.

**Không bịa metric, thời gian huấn luyện hoặc hiệu quả cải thiện.**

---

## 2. HƯỚNG NGHIÊN CỨU ĐƯỢC ĐỀ XUẤT

**Hướng 2** (Xây dựng mô hình A → đánh giá hạn chế → cải thiện thành mô hình B) được chọn làm phương pháp chính.

Lý do: phù hợp mục tiêu chứng minh quá trình cải tiến, khuyến khích error analysis, có thể kết hợp traditional ML → Transformer → Ensemble trong một câu chuyện nghiên cứu xuyên suốt báo cáo.

Hướng 1 (thiết kế N mô hình, so sánh, chọn tốt nhất) được giữ lại như **khía cạnh đối sánh khách quan trên tập test đã khóa**, không làm hai quy trình training độc lập.

---

## 3. CHUỖI THÍ NGHIỆM (Experiment Chain)

Mỗi experiment có: ID, câu hỏi nghiên cứu, giả thuyết, baseline, phương pháp thay đổi, dataset/split, cấu hình培训, metric, yêu cầu tracking, artifacts dự kiến, tiêu chí quyết định, dependencies, chi phí tính toán tương đối.

### EXP-001: EDA & Data Validation (bắt buộc)

| Trường | Nội dung |
|---|---|
| **ID** | EXP-001 |
| **Câu hỏi nghiên cứu** | Dataset có chất lượng đủ để huấn luyện 3 lớp không? Phân bố nhãn, độ dài chuỗi, teencode, near-duplicate, leakage giữa split như thế nào? |
| **Giả thuyết** | UIT-ViHSD có 3 nhãn CLEAN/OFFENSIVE/HATE phù hợp bài toán; có thể mất cân bằng lớp HATE; có near-duplicate giữa train/test cần loại bỏ. |
| **Baseline** | Không có (thí nghiệm dữ liệu). |
| **Thay đổi** | Tải 2 dataset (UIT-ViHSD, ViCTSD), inspect schema, thống kê phân bố, độ dài (p95), teencode/slang, near-duplicate, kiểm tra overlap split. |
| **Dataset/Split** | UIT-ViHSD (tập chính), ViCTSD (so sánh EDA). Chưa split; split được thực hiện sau EDA. |
| **Cấu hình培训** | Không áp dụng. |
| **Metric** | Số lượng mẫu, phân bố nhãn %, độ dài p95, tỷ lệ near-duplicate, tỷ lệ overlap train/test. |
| **Tracking** | MLflow run (tag: `exp=EXP-001`, `stage=data_eda`); lưu `docs/eda_report.md`. |
| **Artifacts** | `docs/eda_report.md`, thống kê dữ liệu, `max_length` đề xuất. |
| **Tiêu chí quyết định** | Chốt dataset chính, chốt `max_length` (dựa trên p95 độ dài), xác định cần class weighting hay không. |
| **Dependencies** | Tải dataset từ Hugging Face Hub (chưa tải trong phiên này). |
| **Chi phí** | XS (không train mô hình). |

### EXP-002: Preprocessing Ablation (bắt buộc)

| Trường | Nội dung |
|---|---|
| **ID** | EXP-002 |
| **Câu hỏi nghiên cứu** | Tiền xử lý tiếng Việt (Unicode NFC, teencode normalization, cleaning) có ảnh hưởng đáng kể đến hiệu năng traditional ML không? |
| **Giả thuyết** | Chuẩn hóa teencode và Unicode giúp Logistic Regression/LinearSVC cải thiện Macro-F1 trên Validation, đặc biệt lớp OFFENSIVE/HATE. |
| **Baseline** | TF-IDF + Logistic Regression không preprocessing (raw text). |
| **Thay đổi** | So sánh: (a) raw text; (b) basic cleaning (Unicode NFC, whitespace); (c) basic + teencode dictionary. |
| **Dataset/Split** | Train/Validation từ EXP-001. |
| **Cấu hình培训** | TF-IDF word(1,2)+char(2,4), `max_features=10000` [PROVISIONAL], LR `class_weight='balanced'`. |
| **Metric** | Validation Macro-F1, Per-class F1. |
| **Tracking** | MLflow runs (tag: `exp=EXP-002`, `variant=raw|basic|teencode`). |
| **Artifacts** | Vectorizer artifacts, validation metrics, `docs/experiments/exp-002-preprocessing-ablation.md`. |
| **Tiêu chí quyết định** | Nếu biến thể có preprocessing cải thiện Macro-F1 ≥ 1% (tuyệt đối) trên Validation → áp dụng preprocessing cho tất cả thí nghiệm sau. |
| **Dependencies** | EXP-001. |
| **Chi phí** | S (nhiều lần train sklearn nhỏ). |

### EXP-003: Baseline Traditional ML (bắt buộc)

| Trường | Nội dung |
|---|---|
| **ID** | EXP-003 |
| **Câu hỏi nghiên cứu** | Mô hình truyền thống đạt Macro-F1 bao nhiêu trên 3 lớp? Logistic Regression vs LinearSVC (calibrated) nào tốt hơn? |
| **Giả thuyết** | Logistic Regression và LinearSVC calibrated đạt mức Macro-F1 chấp nhận được làm baseline; LinearSVC có thể ưu thế trên dữ liệu sparse cao. |
| **Baseline** | Không có (là baseline đầu tiên). |
| **Thay đổi** | Huấn luyện: (a) TF-IDF + Logistic Regression; (b) TF-IDF + LinearSVC bọc `CalibratedClassifierCV(cv=5, method='sigmoid')`. Thử nghiệm `C ∈ {0.1, 1.0, 10.0}`, `max_features`, n-grams. |
| **Dataset/Split** | Train/Validation từ EXP-001. |
| **Cấu hình培训** | Xem `configs/baseline_config.yaml` ([PROVISIONAL], sẽ chốt sau EDA). |
| **Metric** | Validation Macro-F1, Per-class F1, Confusion Matrix, Inference Latency, Model Size. |
| **Tracking** | MLflow experiments (tag: `exp=EXP-003`, `model=lr|linear_svc`). Lưu hyperparameters, metrics, artifacts `.joblib`. |
| **Artifacts** | `models/baselines/baseline_model.joblib`, `docs/experiments/exp-003-baseline-val-results.md`. |
| **Tiêu chí quyết định** | Chốt Best Baseline theo Validation Macro-F1 cao nhất. Best Baseline là đối thủ cho các experiment sau. |
| **Dependencies** | EXP-001, EXP-002 (preprocessing đã chốt). |
| **Chi phí** | S (train sklearn nhỏ, nhanh). |

### EXP-004: Transformer Fine-tuning — PhoBERT (bắt buộc)

| Trường | Nội dung |
|---|---|
| **ID** | EXP-004 |
| **Câu hỏi nghiên cứu** | PhoBERT fine-tuned trên dữ liệu tiếng Việt có cải thiện Macro-F1 so với Best Baseline không? |
| **Giả thuyết** | Biểu diễn ngữ cảnh của PhoBERT xử lý tốt các trường hợp phụ thuộc ý nghĩa toàn câu, nhưng có thể kém trên teencode/slang mạng xã hội. |
| **Baseline** | Best Baseline từ EXP-003. |
| **Thay đổi** | Fine-tune `vinai/phobert-base-v2` trên 3 lớp. Thử nghiệm `learning_rate ∈ {1e-5, 2e-5, 3e-5}`, `batch_size`, `max_length`, `weight_decay`, scheduler, early stopping, class-weighted loss. |
| **Dataset/Split** | Train/Validation từ EXP-001. |
| **Cấu hình培训** | Xem `configs/transformer_config.yaml` ([PROVISIONAL]: `lr=2e-5`, `batch=16`, `max_length=160`, `fp16`, early stopping patience=2, `metric_for_best_model=eval_macro_f1`). |
| **Metric** | Validation Macro-F1, Per-class F1, Training Time, VRAM usage, Inference Latency, Model Size. |
| **Tracking** | MLflow run (tag: `exp=EXP-004`, `model=phobert`). Lưu hyperparameters, checkpoint, metrics, training duration, environment metadata. |
| **Artifacts** | `models/transformer_best/phobert/`, `docs/experiments/exp-004-phobert-val-results.md`. |
| **Tiêu chí quyết định** | Chốt checkpoint tốt nhất theo Validation Macro-F1. So sánh với Best Baseline (EXP-003) trên Validation. |
| **Dependencies** | EXP-001, EXP-003. GPU/CUDA (RTX 4060 8GB hoặc Colab/Kaggle T4 backup). |
| **Chi phí** | M/L (fine-tune Transformer trên GPU). |

### EXP-005: Transformer Fine-tuning — ViSoBERT (bắt buộc)

| Trường | Nội dung |
|---|---|
| **ID** | EXP-005 |
| **Câu hỏi nghiên cứu** | ViSoBERT (đomain mạng xã hội) có cải thiện Macro-F1 so với PhoBERT và Best Baseline không, đặc biệt trên teencode/slang? |
| **Giả thuyết** | Mô hình pre-trained trên ngôn ngữ mạng xã hội có thể xử lý tốt hơn teencode, viết tắt, slang, nhưng cần xác minh bằng thực nghiệm. |
| **Baseline** | Best Baseline (EXP-003) và PhoBERT (EXP-004). |
| **Thay đổi** | Fine-tune `uitnlp/visobert` trên 3 lớp. Cùng không gian hyperparameters với EXP-004 để so sánh công bằng. |
| **Dataset/Split** | Train/Validation từ EXP-001. |
| **Cấu hình培训** | Xem `configs/transformer_config.yaml` ([PROVISIONAL]). |
| **Metric** | Validation Macro-F1, Per-class F1, Training Time, VRAM usage, Inference Latency, Model Size. |
| **Tracking** | MLflow run (tag: `exp=EXP-005`, `model=visobert`). |
| **Artifacts** | `models/transformer_best/visobert/`, `docs/experiments/exp-005-visobert-val-results.md`. |
| **Tiêu chí quyết định** | Chốt Best Transformer theo Validation Macro-F1 cao nhất (PhoBERT hoặc ViSoBERT). |
| **Dependencies** | EXP-001, EXP-003. GPU/CUDA. |
| **Chi phí** | M/L (fine-tune Transformer trên GPU). |

### EXP-006: Error Analysis on Validation (bắt buộc)

| Trường | Nội dung |
|---|---|
| **ID** | EXP-006 |
| **Câu hỏi nghiên cứu** | Best Transformer sai lỗi ở các mẫu nào? Những mẫu sai có đặc điểm ngôn ngữ chung (teencode, slang, emoji, ngữ cảnh, lỗi chính tả)? |
| **Giả thuyết** | Các lỗi tập trung vào OFFENSIVE/HATE nhầm lẫn với CLEAN do teencode/slang hoặc phụ thuộc ngữ cảnh. |
| **Baseline** | Không có (phân tích lỗi). |
| **Thay đổi** | Dự đoán trên tập Validation với Best Transformer, trích xuất ít nhất 30 mẫu sai, phân loại lỗi theo nhóm: teencode, slang, emoji, viết tắt, lỗi chính tả, ngữ cảnh, out-of-distribution. |
| **Dataset/Split** | Validation từ EXP-001. |
| **Cấu hình培训** | Không áp dụng. |
| **Metric** | Số mẫu sai theo từng nhóm lỗi, confusion matrix per error group. |
| **Tracking** | MLflow run (tag: `exp=EXP-006`, `stage=error_analysis`). |
| **Artifacts** | `docs/experiments/exp-006-error-analysis-val.md`, danh sách mẫu lỗi. |
| **Tiêu chí quyết định** | Xác định nhóm lỗi chủ đạo → đưa vào giả thuyết cho EXP-007 (Ensemble/Stacking) hoặc vòng cải thiện tiếp theo. |
| **Dependencies** | EXP-004 hoặc EXP-005 (Best Transformer). |
| **Chi phí** | XS (inference trên Validation). |

### EXP-007: Ensemble / Stacking (conditional — chỉ nếu có giá trị thực nghiệm)

| Trường | Nội dung |
|---|---|
| **ID** | EXP-007 |
| **Câu hỏi nghiên cứu** | Kết hợp Best Baseline và Best Transformer qua Soft Voting / Weighted Soft Voting / Stacking có cải thiện Macro-F1 so với mô hình đơn tốt nhất không? |
| **Giả thuyết** | Best Baseline (TF-IDF) và Best Transformer (contextual) có lỗi bổ sung cho nhau → ensemble cải thiện tổng thể. |
| **Baseline** | Best Transformer đơn lẻ (từ EXP-004/EXP-005). |
| **Thay đổi** | (a) Soft Voting (xác suất trung bình); (b) Weighted Soft Voting (trọng số theo Validation Macro-F1); (c) Stacking với out-of-fold predictions, meta-learner Logistic Regression (chỉ nếu tài nguyên cho phép). |
| **Dataset/Split** | Train/Validation từ EXP-001. Stacking dùng out-of-fold trên Train để train meta-learner, đánh giá trên Validation. |
| **Cấu hình培训** | Không retrain base models; chỉ kết hợp inference. |
| **Metric** | Validation Macro-F1, Per-class F1, Inference Latency, Model Size, Error diversity (khớp lỗi giữa base models). |
| **Tracking** | MLflow run (tag: `exp=EXP-007`, `method=soft_voting|weighted_voting|stacking`). |
| **Artifacts** | `docs/experiments/exp-007-ensemble-results.md`. |
| **Tiêu chí quyết định** | Nếu ensemble cải thiện Validation Macro-F1 ≥ 1% (tuyệt đối) so với Best Transformer đơn → chọn ensemble làm final model. Nếu không, giữ Best Transformer đơn. Nếu Stacking quá tốn tài nguyên, thay bằng Soft Voting. |
| **Dependencies** | EXP-003, EXP-004/EXP-005. |
| **Chi phí** | S/M (inference ensemble, Stacking cần out-of-fold train). |

### EXP-008: Ablation Study (bắt buộc, ít nhất một thành phần)

| Trường | Nội dung |
|---|---|
| **ID** | EXP-008 |
| **Câu hỏi nghiên cứu** | Từng thành phần (preprocessing, char n-grams, class weighting, hyperparameter tuning, transformer architecture) có tác động bao nhiêu đến Macro-F1? |
| **Giả thuyết** | Class weighting và preprocessing có tác động lớn nhất trên dữ liệu mất cân bằng; char n-grams giúp xử lý teencode. |
| **Baseline** | Best model cuối cùng (từ EXP-003/EXP-005/EXP-007). |
| **Thay đổi** | Loại bỏ/tắt từng thành phần một: (a) không preprocessing; (b) không char n-grams; (c) không class weighting; (d) không hyperparameter tuning (dùng mặc định). |
| **Dataset/Split** | Train/Validation từ EXP-001. |
| **Cấu hình培训** | Giống best model, tắt từng thành phần. |
| **Metric** | Validation Macro-F1, Per-class F1, sự chênh lệch so với best model. |
| **Tracking** | MLflow runs (tag: `exp=EXP-008`, `ablation=...`). |
| **Artifacts** | `docs/experiments/exp-008-ablation-study.md`. |
| **Tiêu chí quyết định** | Xác định thành phần nào cần giữ, thành phần nào có thể loại bỏ để đơn giản hóa pipeline. |
| **Dependencies** | EXP-003, EXP-005 (best model). |
| **Chi phí** | S (vài lần train lại nhỏ). |

### EXP-009: Locked Test Final Evaluation (bắt buộc)

| Trường | Nội dung |
|---|---|
| **ID** | EXP-009 |
| **Câu hỏi nghiên cứu** | Best model cuối cùng (sau tất cả các vòng cải tiến) đạt Macro-F1 bao nhiêu trên tập Test đã khóa? So sánh với Best Baseline như thế nào? |
| **Giả thuyết** | Không khẳng định trước; báo cáo trung tính bất kể mô hình nào cao hơn. |
| **Baseline** | Best Baseline (EXP-003). |
| **Thay đổi** | Chạy inference trên `test.parquet` (mở đúng một lần) với: Best Baseline, Best Transformer đơn, Best Ensemble (nếu có). |
| **Dataset/Split** | Locked Test (15%) — mở đúng một lần. |
| **Cấu hình培训** | Không training; chỉ inference. |
| **Metric** | Test Macro-F1, Per-class F1, Confusion Matrix, Inference Latency, Model Size. Báo cáo trade-offs (độ chính xác vs tài nguyên). |
| **Tracking** | MLflow run (tag: `exp=EXP-009`, `stage=locked_test_final`). |
| **Artifacts** | `docs/experiments/exp-009-final-test-benchmark.md`, bảng so sánh đối đầu. |
| **Tiêu chí quyết định** | Chốt final model cho API inference (theo kết quả Test + trade-off tài nguyên). |
| **Dependencies** | Tất cả experiment trước. |
| **Chi phí** | XS (inference trên Test). |

---

## 4. PHÂN LOẠI THÍ NGHIỆM

| Loại | Danh sách |
|---|---|
| **Bắt buộc** | EXP-001, EXP-002, EXP-003, EXP-004, EXP-005, EXP-006, EXP-008, EXP-009 |
| **Có điều kiện** | EXP-007 (chỉ nếu Base models có lỗi bổ sung và tài nguyên cho phép) |
| **Mở rộng** | TF-IDF + XGBoost, BiLSTM, TextCNN (chỉ thêm nếu có giả thuyết rõ ràng) |

---

## 5. BẢNG SO SÁNH ĐỐI DẬU (Template — chưa có số liệu)

| Model | Backend | Val Macro-F1 | Test Macro-F1 | Clean F1 | Offensive F1 | Hate F1 | Latency (ms) | Size (MB) | Status |
|---|---|---|---|---|---|---|---|---|---|
| Best Baseline (LR/LinearSVC) | Sklearn | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Chưa chạy |
| PhoBERT | PyTorch | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Chưa chạy |
| ViSoBERT | PyTorch | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Chưa chạy |
| Best Ensemble (nếu có) | Sklearn+PyTorch | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Có điều kiện |

*TBD = chưa có số liệu thực nghiệm. Không bịa số liệu.*

---

## 6. YÊU CẦU TRACKING & REPRODUCIBILITY

- MLflow là công cụ tracking chính. Mỗi experiment run lưu: Experiment ID, Run ID, dataset version/hash, split config, model architecture, hyperparameters, random seed, training duration, train/val metrics, environment metadata, checkpoint/artifacts, git commit (nếu có).
- Random seed cố định (42) và được ghi log.
- Không hard-code đường dẫn; cấu hình qua YAML/CLI.
- Tự nhận diện CPU/GPU; tùy chỉnh batch size, workers, mixed precision.
- Tổng hợp MLflow runs từ nhiều máy nếu cần (hỗ trợ nhóm sử dụng nhiều máy).
- Không dùng accuracy đơn lẻ làm căn cứ đánh giá chính.
