# ML Experiments Plan — Scratch-first v4.0

**Ngày:** 2026-11-11 · **Status:** PLAN ONLY — no experiment executed.

## Protocol (áp dụng cho tất cả)
- 3 nhãn: CLEAN=0, OFFENSIVE=1, HATE=2; dataset ưu tiên UIT-ViHSD (phải kiểm chứng). Shared Data Contract v1 do TV3 chốt tại Gate G0.
- Official split nếu phù hợp; nếu không stratified 70/15/15. Validation để tune và chọn; locked test chỉ dùng final batch. Fit vocabulary/TF-IDF/calibrator chỉ trong train hoặc train-fold; dedup/group leakage checks.
- **Primary:** Macro-F1 trên Validation. Secondary: per-class F1, confusion, training time, latency, size; calibration diagnostics nếu xuất xác suất.
- Bốn nhóm mô hình chính **random-init/fit-from-scratch**. PhoBERT pretrained **chỉ benchmark tùy chọn**. Mỗi TV ≥3 cấu hình mang giả thuyết rõ ràng. Log MLflow run IDs, seeds, configs, data hash, environment, predictions và artifacts.
- Không bịa chỉ số; không khẳng định bất kỳ thay đổi nào cải thiện khi chưa có dữ liệu. Mọi kết quả ghi trạng thái PENDING đến khi chạy thật.

## Thứ tự và chủ sở hữu
| Experiment | Owner | Đặc điểm | Dependency | Kết quả |
|---|---|---|---|---|
| EXP-00 | TV3 (lead), TV1 review | Dataset audit + locked split + Data Contract v1 | Dataset nguồn | manifest, mapping, leakage check |
| EXP-01 | TV1 | Logistic Regression: raw/basic/teencode, word/char TF-IDF, class weight | G0 | ≥3 run + EDA/ablation |
| EXP-02 | TV2 | LinearSVC: C, feature selection, TF-IDF n-grams, optional RBF | G0 | ≥3 run + SVM comparison |
| EXP-03 | TV3 | Scratch LSTM → BiLSTM → BiLSTM+Attention | G0 | ≥3 run + architecture ablation |
| EXP-04 | TV4 | Scratch single-kernel → multi-kernel TextCNN | G0 | ≥3 run + CNN ablation |
| EXP-05 | TV3 chủ trì, mỗi TV góp dữ liệu | Error analysis cross-model trên validation | EXP-01…04 | phân nhóm lỗi, hypotheses |
| EXP-06 | TV4 chủ trì | Ensemble (conditional), soft/weighted voting / OOF stacking nếu hợp lệ | val predictions của ≥2 models | compare best single |
| EXP-07 | TV3 | **Optional** pretrained PhoBERT fine-tuning benchmark | EXP-03 đã đạt nghiệm thu, G0 | transfer-learning comparison |
| EXP-08 | TV1 điều phối; TV3 duyệt | One-shot locked-test final comparison của các candidates đã khóa | EXP-01…04, selection frozen | final benchmark |
| EXP-09 | Cả nhóm | Research notebooks, reproduction, FastAPI demo plan | Final candidates | report, artifacts, demo |

## EXP-00: G0 — data governance
1. Xác nhận source/revision/license/schema/class labels, official split nếu có.
2. Kiểm tra trùng lặp, near-duplicate, lỗi nhãn, phân bố lớp và leakage theo thread nếu metadata có.
3. Khóa manifest `id→split` + seed + hash. Xuất schema `id,text,label,split` thống nhất.
4. Không cung cấp locked test cho các experiment trước final. Nếu phát hiện lỗi nghiêm trọng, version data contract và chạy lại experiments bị ảnh hưởng.
**Acceptance:** có manifest và xác minh overlap; TV2–TV4 có thể code skeleton bằng fixture trước G0, không lấy benchmark từ fixture.

## EXP-01: TV1 — LR & preprocessing ablation
**Question:** preprocessing tiếng Việt và word/char n-grams tác động thế nào đến Macro-F1?
**Variants (minimum):** (A) raw + word TF-IDF + LR; (B) basic cleanup + word+char TF-IDF; (C) teencode normalization + word+char + class weighting. Tuning `C` giữ các thay đổi được kiểm soát và ghi lại. `TF-IDF.fit` chỉ dùng train.
**Output:** EDA notebook, metrics, confusion matrix, LR pipeline artifacts, val predictions và ablation conclusions.

## EXP-02: TV2 — SVM & feature engineering
**Question:** LinearSVC có tốt hơn/khác LR ở đâu, feature selection và C ảnh hưởng gì?
**Variants:** (A) LinearSVC word TF-IDF; (B) word+char; (C) feature selection + tune C/class_weight. Optional (D) `SVC(kernel="rbf")` khi memory/time hợp lý, ghi chi phí.
**Output:** SVM configs, models, grid results, val predictions, ablation. Nếu calibration cho API: cross-validation train-only; raw decision score KHÔNG là xác suất.

## EXP-03: TV3 — scratch recurrent architectures
**Question:** bidirectionality và attention có cải thiện trên dữ liệu tiếng Việt không?
**Variants:** (A) LSTM; (B) BiLSTM; (C) BiLSTM + attention; riêng tune embedding dimension, sequence length, dropout/hidden size nếu có thời gian.
**Constraints:** random initialized embeddings, vocabulary train-only, early stopping on validation, fp16 nếu hỗ trợ, gradient clipping, checkpoints kèm config.
**Output:** architecture diagrams, loss curves, metrics, checkpoints, val predictions, ablation và error categories. Thí nghiệm nhóm trưởng **không dùng pretrained weights**.

## EXP-04: TV4 — scratch CNN architectures
**Question:** single kernel vs multi-kernel CNN và regularization có khác nhau?
**Variants:** (A) single kernel; (B) multiple kernels; (C) tuned filters/dropout/pooling; embeddings random init. Chọn trên val Macro-F1.
**Output:** code/models, curves, confusion matrix, val predictions, ablation.

## EXP-05: error analysis
So sánh ≥30 mẫu validation sai có chủ đích và tỷ lệ lỗi theo CLEAN/OFFENSIVE/HATE; xem teencode, emoji, slang, phủ định, ngữ cảnh, near duplicates. TV3 tổ chức, TV1/2/4 đóng góp errors và giải thích. Kết luận phải dẫn tới giả thuyết kiểm chứng, không chỉ liệt kê mẫu.

## EXP-06: ensemble (conditional)
TV4 kiểm tra error diversity trên predictions validation của LR/SVM/BiLSTM/TextCNN. Có căn cứ mới thử weighted/soft voting với xác suất có ý nghĩa; nếu stacking, tạo out-of-fold predictions **trên train folds** chứ không fit meta-learner trực tiếp trên val/test. Compare best single và overhead. Nếu không có lợi ích hoặc xác suất không đáng tin: ghi kết quả và dừng.

## EXP-07: optional PhoBERT pretrained benchmark — TV3
Sau khi scratch BiLSTM đã có báo cáo, tải pretrained base `vinai/phobert-base-v2`, fine-tune classifier trên train set chung; không tải mô hình toxic đã fine-tune từ trước. Log base version, head initialization, time, VRAM, val Macro-F1, inference latency. So sánh với **4 mô hình scratch**, minh bạch sự khác biệt pretrained vs train from scratch. Nếu thiếu GPU/thời gian: không chặn final.

## EXP-08: locked-test final
Chỉ khi mô hình/threshold/ensemble/calibration đã freeze bằng validation. TV1 chạy script đánh giá duy nhất theo danh sách candidate và TV3 giám sát; ghi per-class F1, Macro-F1, confusion matrix, latency, model size. Không tuning ngược từ test.

## EXP-09: nghiên cứu, demo và tái lập
Cả nhóm báo cáo thí nghiệm chính của mình, notebook chỉ làm phân tích, không nhúng training logic độc quyền; API-minimal dùng best model đã chọn. Deployment có điều kiện; không triển khai trong lần sửa PLAN/SPEC này.
