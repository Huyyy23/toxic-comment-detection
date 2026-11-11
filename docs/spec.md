# ĐẶC TẢ HỆ THỐNG: NHẬN DẠNG BÌNH LUẬN ĐỘC HẠI TIẾNG VIỆT
## (Vietnamese Toxic Comment Detection System - Specification)

- **Mã tài liệu**: `SPEC-TOXIC-COMMENT-2026`
- **Phiên bản**: `3.1.0-ML-FIRST`
- **Trạng thái**: `DRAFT — PENDING USER REVIEW`
- **Tác giả**: Senior ML Engineer / NLP Researcher / Software Architect
- **Ngày cập nhật**: 2026-11-11
- **Phân loại**: Đồ án cuối kỳ môn Machine Learning (hướng nghiên cứu, không phải phát triển full-stack)

---

## 1. MỤC TIÊU & PHẠM VI DỰ ÁN (Objective & Scope)

### 1.1. Bối cảnh & Mục tiêu

- **Mục tiêu**: Xây dựng một đồ án nghiên cứu Machine Learning có độ cao, tập trung vào quá trình xây dựng, huấn luyện, so sánh và cải thiện nhiều mô hình phân loại bình luận tiếng Việt (3 lớp: CLEAN / OFFENSIVE / HATE). Có bằng chứng thực nghiệm, ghi log đầy đủ thí nghiệm, phân tích lỗi, nghiên cứu giải pháp (ablation), và triển khai công khai bằng **FastAPI REST API tối giản**.
- **Không phải**: Xây dựng ứng dụng web kiểm duyệt bình luận đầy đủ tính năng. **Không phát triển frontend riêng** (React/Vite sẽ được xóa hoàn toàn).
- **Đối tượng người dùng**: Giảng viên & hội đồng chấm điểm (cần kiểm chứng phương pháp khoa học, số liệu thực nghiệm trung tính), sinh viên thực hiện đồ án.

### 1.2. Nguyên tắc kiến trúc

| Nguyên tắc | Ý nghĩa |
|---|---|
| **ML-first** | Phần quan trọng nhất là thí nghiệm ML: EDA, preprocessing, training, tuning, evaluation, error analysis, ablation, ensemble. |
| **API-minimal** | Chỉ cần FastAPI với 3 endpoint (`POST /predict`, `GET /health`, `GET /model-info`). Swagger UI tại `/docs` dùng để kiểm thử và demo. Không frontend riêng. |
| **Reproducibility-first** | Mọi huấn luyện chính thức được ghi log qua **MLflow local**: config, dataset version, seed, hyperparameters, metrics, thời lượng, artifacts, Experiment ID, git commit. |
| **No unnecessary complexity** | Không React, không Vite, không CSS/UI framework, không auth/JWT, không user database, không CI/CD phức tạp, không distributed training trừ khi thực sự cần. Docker không bắt buộc (giữ nếu hosting yêu cầu). |

---

## 2. ĐỊNH NGHĨA BÀI TOÁN (Problem Definition)

- **Loại bài toán**: Phân loại văn bản (text classification), multi-class 3 nhãn.
- **Taxonomy (bắt buộc giữ nguyên)**:
  - `0: CLEAN`: Lành mạnh / không độc hại.
  - `1: OFFENSIVE`: Xúc phạm, thô tục, chửi thề.
  - `2: HATE`: Thù ghét, kỳ thị, công kích định danh nghiêm trọng.
- **Không tự động gộp về nhị phân (Binary)** trừ khi có lý do kỹ thuật và bằng chứng rõ ràng, được ghi nhận trong SPEC.
- **Metric chính**: Macro-F1 (để chọn mô hình). Không dùng accuracy đơn lẻ làm căn cứ đánh giá chính.
- **Metric bổ trợ**: Weighted-F1, Per-class Precision/Recall/F1 cho 3 lớp (`CLEAN`, `OFFENSIVE`, `HATE`), Confusion Matrix, Balanced Accuracy, Training Time, Inference Latency, Model Size.

---

## 3. YÊU CẦU GIẢNG VIÊN (Course Requirements)

### 3.1. End-to-end Machine Learning (bắt buộc)

1. Thu thập hoặc tiếp nhận dữ liệu.
2. Khám phá dữ liệu (EDA).
3. Làm sạch và tiền xử lý.
4. Feature Engineering.
5. Huấn luyện mô hình.
6. Tối ưu hyperparameters.
7. Đánh giá và so sánh mô hình.
8. Lựa chọn mô hình tốt nhất.
9. Triển khai công khai (REST API).
10. Kiểm thử inference.

### 3.2. Độ khó cao (bắt buộc)

- Nhiều mô hình hoặc kiến trúc khác nhau.
- Quá trình huấn luyện, tuning và đánh giá thực chất.
- Ensemble/Stacking khi khả fi (không bắt buộc; chỉ nếu có bằng chứng thực nghiệm).
- So sánh hiệu quả của các cải tiến.
- Phân tích những thử nghiệm thất bại.
- Thảo luận độ phức tạp và chi phí tính toán.

### 3.3. Ghi log toàn bộ thí nghiệm (bắt buộc)

Mọi lần huấn luyện chính thức lưu: cấu hình, dataset version, model architecture, random seed, hyperparameters, metrics, training duration, model artifacts, experiment ID, git commit (nếu có). Ưu tiên **MLflow local** làm công cụ tracking chính.

### 3.4. Dữ liệu đáng tin cậy (bắt buộc)

- Xuất xứ rõ ràng, có đơn vị phát hành / công trình nghiên cứu xác thực.
- Có thông tin giấy phép sử dụng.
- Có phương pháp gán nhãn đáng tin cậy.
- Có tính phù hợp với tiếng Việt mạng xã hội hiện tại.
- Nếu dùng dataset cũ: phải giải thích phạm vi sử dụng và đánh giá khả năng tổng quát hóa.

### 3.5. Triển khai công khai (bắt buộc)

- Sử dụng **FastAPI REST API**.
- Không bắt buộc frontend riêng.
- Swagger UI mặc định của FastAPI (`/docs`) dùng để kiểm thử và demo.

---

## 4. PHƯƠNG PHÁP NGHIÊN CỨU: HƯỚNG 1 VÀ HƯỚNG 2 (Research Methodology)

### 4.1. Hướng 1 — Thiết kế N mô hình, so sánh và chọn mô hình tốt nhất

Thiết kế và huấn luyện nhiều mô hình độc lập để giải quyết cùng một bài toán, đánh giá theo quy trình thống nhất, sau đó so sánh để chọn mô hình phù hợp nhất.

Ví dụ: TF-IDF + Logistic Regression, TF-IDF + Linear SVM, PhoBERT, ViSoBERT, Ensemble.

**Đặc điểm cốt lõi**: Nhiều mô hình được xây dựng và so sánh trong cùng một quy trình thực nghiệm.

### 4.2. Hướng 2 — Xây dựng mô hình A, đánh giá hạn chế, cải thiện thành mô hình B

Bắt đầu với mô hình ban đầu, đánh giá để xác định hạn chế, đưa ra giả thuyết cải tiến cụ thể, thiết kế mô hình mới / thay đổi pipeline, huấn luyện lại và đo lường tác động. Có thể lặp lại nhiều vòng.

Ví dụ chuỗi:
- Vòng 1 (A): TF-IDF + Logistic Regression — mốc hiệu năng ban đầu.
- Vòng 2 (B): Transformer (PhoBERT / ViSoBERT) — giả thuyết: biểu diễn ngữ cảnh xử lý tốt hơn các trường hợp phụ thuộc ý nghĩa toàn câu.
- Vòng 3 (C): Ensemble / Stacking — giả thuyết: các mô hình có lỗi bổ sung cho nhau cải thiện kết quả tổng thể. **Không bắt buộc**, chỉ nếu có bằng chứng thực nghiệm.

**Yêu cầu đặc biệt**: Không được khẳng định trước kết quả (B chắc chắn tốt hơn A, v.v.). Các vòng tiếp theo phải dựa trên kết quả đã quan sát. Nếu chưa thực hiện training, chỉ ghi nhận giả thuyết và thí nghiệm dự kiến.

**Kết quả bắt buộc của mỗi vòng**: Vấn đề xác định, bằng chứng thực nghiệm, giả thuyết cải tiến, phương pháp thực hiện, kết quả trước/sau, phân tích sự khác biệt, kết luận, quyết định tiếp theo.

### 4.3. Hướng nghiên cứu được đề xuất

**Hướng 2 được chọn làm phương pháp nghiên cứu chính.**

Lý do:
- Phù hợp mục tiêu chứng minh quá trình cải tiến (giảng viên ưu tiên).
- Khuyến khích error analysis trực tiếp liên kết với giả thuyết vòng tiếp theo.
- Có thể kết hợp traditional ML, Transformer và Ensemble trong một câu chuyện nghiên cứu xuyên suốt.
- Dễ trình bày trong báo cáo đồ án: baseline → hạn chế → cải thiện → kết luận.
- Hướng 1 (so sánh N mô hình chọn tốt nhất) giữ lại như một khía cạnh đối sánh khách quan trên tập test đã khóa, không làm hai quy trình training độc lập.

**Cách kiểm chứng đáp ứng yêu cầu giảng viên**: Mỗi vòng thí nghiệm trong Experiment Plan (`docs/plans/ml-experiments.md`) có mục tiêu, giả thuyết, baseline, phương pháp, metric và tiêu chí quyết định rõ ràng. Không có số liệu bịa đặt; kết quả chỉ được ghi sau khi chạy thực tế.

---

## 5. CHIẾN LƯỢC DỮ LIỆU (Dataset Strategy)

### 5.1. Nguồn đề xuất

Hai nguồn tham chiếu từ `link dataset.txt`:
1. `https://huggingface.co/datasets/tarudesu/ViCTSD` (Vietnamese Constructive and Toxic Speech Detection).
2. `https://huggingface.co/datasets/ura-hcmut/UIT-ViHSD` (UIT Vietnamese Hate Speech Detection).

Trạng thái hiện tại: **[Assumption]** — link đã cung cấp, chưa tải local, chưa inspect raw schema, chưa có thống kê số lượng mẫu / phân bố nhãn.

**Đề xuất**: Ưu tiên `UIT-ViHSD` làm tập chính (3 nhãn CLEAN / OFFENSIVE / HATE khớp taxonomy), `ViCTSD` dùng EDA so sánh hoặc cross-dataset evaluation (conditional). Chốt chính thức sau EDA (M1).

### 5.2. Tiêu chí chọn dữ liệu (bắt buộc)

- Có xuất xứ rõ ràng, đơn vị phát hành / công trình xác thực.
- Có giấy phép sử dụng.
- Có phương pháp gán nhãn đáng tin cậy.
- Phù hợp với tiếng Việt mạng xã hội hiện tại.
- Nếu dùng dataset cũ, phải giải thích phạm vi sử dụng và đánh giá khả năng tổng quát hóa.

### 5.3. Quy định chung (không bịa dữ liệu)

- Không bịa nguồn dữ liệu hoặc giấy phép chưa được xác minh.
- Không thu thập dữ liệu trong phiên này.
- Không tự động gộp các dataset có định nghĩa nhãn khác nhau mà không có lý do kỹ thuật.

---

## 6. KIẾN TRÚC PIPELINE ML (ML Pipeline Architecture)

```
Raw Data (data/raw/)
    │
    ▼
Download & Ingestion (src/data/download.py)
    │
    ▼
Preprocessing (src/data/preprocessor.py)  [Unicode NFC, teencode, cleaning]
    │
    ▼
Split 70/15/15 Stratified (src/data/splitter.py)  [train / val / LOCKED test]
    │
    ▼
┌─────────────────┬──────────────────┬─────────────────────┐
│ Traditional ML  │ Transformer      │ Ensemble / Stacking │
│ TF-IDF + LR     │ PhoBERT / ViSoBERT│ Soft Voting /     │
│ TF-IDF + LinearSVC│ fine-tuning      │ Stacking (conditional)│
│ (calibrated)    │                  │                   │
└─────────────────┴──────────────────┴─────────────────────┘
    │                       │                    │
    ▼                       ▼                    ▼
Evaluation on Validation set (tuning & model selection)
    │
    ▼
Locked Test Set (mở đúng 1 lần, đánh giá cuối cùng)
    │
    ▼
MLflow Local Tracking (mỗi run: config, seed, hyperparams, metrics, artifacts)
```

- Tất cả logic dùng lại được nằm trong file Python `.py`.
- Notebook chỉ phục vụ nghiên cứu (xem mục 13).
- Không hard-code đường dẫn; cấu hình qua YAML/CLI.
- Tự nhận diện CPU/GPU khi có thể; tùy chỉnh batch size, workers, mixed precision.

---

## 7. CÁC MÔ HÌNH ỨNG VIÊN (Model Candidates)

### 7.1. Traditional Machine Learning (bắt buộc)

| Mã | Mô hình | Đặc trưng | Ghi chú |
|---|---|---|---|
| M-A1 | Logistic Regression | TF-IDF word (1,2) + char (2,4) | Baseline A, `predict_proba` tự nhiên |
| M-A2 | LinearSVC (calibrated) | TF-IDF word (1,2) + char (2,4) | `CalibratedClassifierCV(cv=5, method='sigmoid')` |

Thử nghiệm: word/char n-grams, `max_features`, regularization `C`, class weighting, preprocessing variants.

### 7.2. Deep Learning / Transformer (ưu tiên)

| Mã | Mô hình | Ghi chú |
|---|---|---|
| M-B1 | PhoBERT (`vinai/phobert-base-v2`) | Ngữ cảnh tiếng Việt |
| M-B2 | ViSoBERT (`uitnlp/visobert`) | Đomain mạng xã hội, teencode/slang |

Thiết kế thí nghiệm: learning rate, batch size, epoch, max sequence length, weight decay, scheduler, early stopping, gradient accumulation, mixed precision, class imbalance handling.

### 7.3. Mô hình mở rộng (conditional)

- TF-IDF + XGBoost.
- BiLSTM.
- TextCNN.

Chỉ thêm nếu có giả thuyết nghiên cứu và giá trị thực nghiệm rõ ràng. Không thêm chỉ để tăng số lượng.

### 7.4. Ensemble Learning (không bắt buộc)

| Mã | Phương pháp | Điều kiện |
|---|---|---|
| M-C1 | Soft Voting | Cần xác suất có ý nghĩa; đánh giá calibration. |
| M-C2 | Weighted Soft Voting | Trọng số theo validation performance của base model. |
| M-C3 | Stacking | Out-of-fold predictions hợp lệ; meta-learner không học từ test set. |

Lưu ý: Không xem decision score của LinearSVC là xác suất. So sánh ensemble với mô hình đơn tốt nhất. Nếu Stacking quá tốn tài nguyên, có phương án thay thế (Soft Voting). Nếu không cải thiện ≥ 1% Macro-F1, giữ mô hình đơn tốt nhất.

---

## 8. EXPERIMENT TRACKING (MLflow Local)

- Công cụ tracking chính: **MLflow local** (file store).
- Hyperparameter optimization: **Optuna** (tùy chọn).
- Mỗi run lưu: Experiment ID, Run ID, dataset version/hash, split config, model architecture, hyperparameters, random seed, training duration, train/val metrics, environment metadata, checkpoint/artifacts, git commit (nếu có).
- Metric chính: Macro-F1. Không dùng accuracy đơn lẻ làm căn cứ đánh giá.
- Phân biệt rõ: kết quả đã đo vs kết quả dự kiến vs giả thuyết vs mục tiêu vs chưa xác minh. Không bịa metric, thời gian train hoặc hiệu quả cải thiện.

---

## 9. GIÁO THỨC ĐÁNH GIÁ (Evaluation Protocol)

| Tập | Tỷ lệ | Vai trò |
|---|---|---|
| Train | 70% | Huấn luyện trọng số, fit TF-IDF chỉ trên Train. |
| Validation | 15% | Tuning hyperparameters, early stopping, model selection, calibration. |
| Locked Test | 15% | **Bị khóa hoàn toàn**; chỉ mở đúng 1 lần đánh giá cuối cùng. Không dùng để chọn mô hình hoặc tuning. |

- Chống rò rỉ: `set(train_texts) ∩ set(test_texts) == 0`; deduplicate trước khi split; TF-IDF chỉ fit trên Train.
- Đánh giá đối kháng khách quan trên cùng một tập test chuẩn, báo cáo trung tính bất kể mô hình nào cao hơn.

---

## 10. PHÒNG CHỐNG DỮ LIỆU ROÀ RIÊ (Data Leakage Prevention)

- TF-IDF vectorizer chỉ `fit` trên tập Train.
- Kiểm tra giao thoa văn bản giữa các split.
- Hyperparameter tuning chỉ trên Validation, không đụng Test.
- Ensemble out-of-fold predictions hợp lệ; meta-learner không học từ Test set.
- Locked test set: chỉ mở một lần tại EXP-009.

---

## 11. PHÂN TÍCH LỖI (Error Analysis)

- Nhầm lẫn CLEAN / OFFENSIVE / HATE.
- False Positive / False Negative.
- Phân phối lỗi theo lớp.
- Teencode, tiếng lóng, emoji, viết tắt, lỗi chính tả.
- Bình luận phụ thuộc ngữ cảnh.
- Dữ liệu ngoài phân phối huấn luyện (out-of-distribution).
- Nếu chọn Hướng 2, Error Analysis liên kết trực tiếp với giả thuyết cải tiến của vòng tiếp theo.
- Tổng hợp kết quả trong Notebook (`06_error_analysis.ipynb`).

---

## 12. CHÍNH SÁCH JUPYTER NOTEBOOK (Notebook Policy)

- Notebook dùng để lưu EDA, trình bày phương pháp, trực quan hóa dữ liệu, phân tích kết quả training, so sánh mô hình, error analysis, ablation study, kết luận thực nghiệm.
- **Không dùng Notebook làm thành phần production.**
- Tất cả logic dùng lại được phải nằm trong file Python `.py`.
- Notebook chỉ import module Python, đọc kết quả thực nghiệm và trình bày phân tích.
- Cấu trúc đề xuất:
  - `01_eda.ipynb`
  - `02_preprocessing.ipynb`
  - `03_baseline_comparison.ipynb`
  - `04_transformer_analysis.ipynb`
  - `05_ensemble_comparison.ipynb`
  - `06_error_analysis.ipynb`
- Mỗi Notebook: giải thích Markdown, biểu đồ, bảng kết quả, confusion matrix, training curves (khi phù hợp), MLflow Run ID, kết luận nghiên cứu.
- Không huấn luyện lại mô hình chỉ để hiển thị báo cáo. Không tạo số liệu giả. Không nhúng checkpoint hoặc dữ liệu nhạy cảm vào Notebook.
- Phiên này chỉ thiết kế nội dung và cấu trúc Notebook, không tạo file `.ipynb`.

---

## 13. TRIỂN KHAI FASTAPI (Production Inference)

### 13.1. Kiến trúc

- Chỉ bao gồm: FastAPI application, inference pipeline, best model/ensemble được chọn, tokenizer, preprocessing artifacts, label mapping, runtime dependencies.
- **Không đưa vào production**: Notebook, training dataset, MLflow server, Optuna, training processes, unused checkpoints, frontend.

### 13.2. Ba endpoint

| Endpoint | Phương法 | Mô tả |
|---|---|---|
| `POST /predict` | POST | Nhận văn bản, trả về nhãn + xác suất hiệu chuẩn + latency. |
| `GET /health` | GET | Kiểm tra model_loaded, backend, artifact status, device. |
| `GET /model-info` | GET | Trả về metadata mô hình, label taxonomy, calibration method, artifact version. |

- Swagger UI tại `/docs`.
- Input validation qua Pydantic schemas.
- Error handling, model readiness, inference consistency.

### 13.3. Nền tảng triển khai đề xuất

- Ưu tiên: Hugging Face Spaces (nếu đáp ứng điều kiện).
- Thay thế: Render, Railway, hoặc chạy container Docker local.
- Docker không bắt buộc; chỉ giữ nếu hosting yêu cầu container.
- Không tự tạo Dockerfile hay code FastAPI trong phiên này (chỉ đặc tả trong tài liệu).

---

## 14. HỖ TRỢ NHIỀU MÁY (Multi-machine)

- Không ràng buộc theo một CPU, GPU, RAM hoặc hệ điều hành cụ thể.
- Cấu hình training qua YAML/CLI.
- Tự nhận diện CPU/GPU khi có thể.
- Tùy chỉnh batch size, workers, mixed precision.
- Random seed cố định và được ghi log.
- Resume checkpoint khi phù hợp.
- Không hard-code đường dẫn.
- Chia thí nghiệm độc lập cho nhiều thành viên.
- Tổng hợp MLflow local runs hoặc kết quả thí nghiệm từ nhiều máy.
- Ghi nhận môi trường để đánh giá sự khác biệt về tốc độ train.
- Không yêu cầu distributed training nếu thí nghiệm có thể thực hiện độc lập.

---

## 15. KHÔNG TRONG PHẠM VI (Out-of-scope Features)

- **React, Vite, frontend components, CSS/UI framework — đã xóa hoàn toàn thư mục `frontend/` trong phiên này.**
- Dashboard, user authentication, JWT, user database.
- CRUD không liên quan ML.
- Full-stack infrastructure dư thừa.
- CI/CD quá phức tạp.
- Dockerfile bắt buộc (tùy chọn, giữ nếu hosting yêu cầu).
- Distributed training bắt buộc.
- Public deployment thực tế (chỉ đặc tả, không triển khai trong phiên này).

---

## 16. RỦI RO (Risks)

| Mã | Rủi ro | Mức độ | Giảm thiểu |
|---|---|---|---|
| R1 | Hugging Face Space chưa có token / yêu cầu thẻ tín dụng | Cao | Phương án dự phòng: Render / Railway / local container. |
| R2 | Mất cân bằng 3 lớp (Lớp HATE chiếm tỉ lệ nhỏ nhất) | Cao | Áp dụng Class-weighted Cross-Entropy Loss; đo đạc Macro-F1 và Per-class F1 trên tập Validation; không dùng Accuracy đơn lẻ. |
| R3 | Huấn luyện Transformer trên GPU RTX 4060 bị tràn VRAM (OOM) | Trung bình | Đánh dấu `max_length = 160`, `batch_size = 16` là PROVISIONAL; chạy profiling bước đầu; nếu VRAM vượt 7GB thì giảm `max_length = 128` hoặc chuyển sang Colab/Kaggle T4. |
| R4 | Rò rỉ dữ liệu hoặc lạm dụng tập Test để tuning mô hình | Chí mạng | Khóa hoàn toàn `test.parquet`; toàn bộ việc chọn mô hình diễn ra trên Validation set; chỉ unseal test set một lần duy nhất tại EXP-009. |
| R5 | Dataset cũ không tổng quát hóa ngôn ngữ mạng xã hội mới | Trung bình | Đánh giá cross-dataset, ghi nhận giới hạn trong báo cáo. |
| R6 | Ensemble/Stacking quá tốn tài nguyên, không mang lại lợi ích | Trung bình | Có phương án thay thế: Soft Voting; nếu không cải thiện ≥ 1% Macro-F1, giữ mô hình đơn tốt nhất. |

---

## 17. TIÊU CHÍ NGHIỆM THU (Acceptance Criteria)

| Mã | Tiêu chí |
|---|---|
| AC-01 | Có ít nhất 2 mô hình traditional ML (LR, LinearSVC calibrated) được huấn luyện và tuning trên Validation. |
| AC-02 | Có ít nhất 1 mô hình Transformer (PhoBERT hoặc ViSoBERT) được fine-tune và tuning trên Validation. |
| AC-03 | Có thí nghiệm Ensemble/Stacking (ít nhất Soft Voting) so sánh với mô hình đơn tốt nhất (conditional). |
| AC-04 | Có ít nhất 1 ablation study (ví dụ: preprocessing, class weighting, char n-grams). |
| AC-05 | Có error analysis trực quan trong Notebook, liên kết với giả thuyết cải tiến. |
| AC-06 | Mọi run training chính thức được ghi log trên MLflow local (config, seed, hyperparams, metrics, artifacts). |
| AC-07 | Locked test set chỉ được mở đúng 1 lần để đánh giá cuối cùng; báo cáo trung tính. |
| AC-08 | Hợp đồng xác suất: tổng xác suất 3 lớp ≈ 1.0, LinearSVC dùng `CalibratedClassifierCV`. |
| AC-09 | FastAPI có 3 endpoint (`/predict`, `/health`, `/model-info`), Swagger UI hoạt động. |
| AC-10 | Không có số liệu bịa đặt; phân biệt rõ kết quả đã đo vs dự kiến. |
| AC-11 | Tài liệu SPEC, Experiment Plan, Implementation Plan nhất quán, không mâu thuẫn. |
| AC-12 | Quyết định xóa hoàn toàn `frontend/` đã được thực hiện trong phiên này. |
