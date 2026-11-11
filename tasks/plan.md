# Team Execution Plan — Vietnamese Toxic Comment Detection

**Version 4.0.0 · 2026-11-11 · PLAN ONLY**

## Quyết định cuối cùng
- Đồ án Machine Learning **scratch-first** trên 3 nhãn CLEAN/OFFENSIVE/HATE.
- 4 mô hình chính train từ đầu: **Logistic Regression (TV1), LinearSVC (TV2), LSTM→BiLSTM→Attention (TV3/lead), TextCNN (TV4)**.
- **TV3 tùy chọn fine-tune pretrained PhoBERT** làm đối chứng, công bố rõ là pretrained benchmark, không thay thế model scratch.
- Ensemble do TV4 dẫn dắt **chỉ khi** các dự đoán validation chứng minh việc kết hợp có triển vọng.
- Không frontend React; FastAPI `POST /predict`, `GET /health`, `GET /model-info` làm demo tối giản.

## Workflow song song
1. **G0 (TV3):** kiểm tra dataset, label mapping, dedup/leakage, chốt official split hoặc 70/15/15 + seed/hash/manifest. TV1 làm EDA; TV2–TV4 code skeleton trước khi G0 xong.
2. **G1:** 4 người training song song trên **cùng** Train/Validation; mỗi người ≥3 cấu hình thực nghiệm + MLflow runs, checkpoints, predictions.
3. **G2:** TV1/TV2/TV4 ablation, TV3 error analysis; TV4 ensemble conditional; TV3 PhoBERT optional sau khi scratch BiLSTM xong. Freeze models theo Validation Macro-F1.
4. **G3:** TV1 tổ chức đánh giá final locked test, TV3 review; cả nhóm báo cáo/notebook; tích hợp FastAPI minimal nếu đủ điều kiện.

## Lộ trình đề xuất 4 tuần
| Tuần | TV1 | TV2 | TV3 (lead) | TV4 |
|---|---|---|---|---|
| 1 | EDA + LR setup | SVM setup | G0 + LSTM setup | CNN setup |
| 2 | LR experiments | SVM experiments | LSTM/BiLSTM experiments | CNN experiments |
| 3 | LR ablation | SVM ablation | Error analysis + optional PhoBERT | CNN ablation + optional ensemble |
| 4 | Final test coordination | SVM report | Integrate/technical review | CNN report + API support |

## Quy tắc nghiệm thu
Mỗi người nộp scripts, ≥3 official variants, MLflow runs và seed/dataset hash, val Macro-F1/per-class scores, checkpoint, predictions IDs, ablation + error analysis, notebook/report. **Không dùng test để tune**. Cùng một GPU thì LSTM/TextCNN/PhoBERT chia lịch train dù code phát triển song song.

Chi tiết: [SPEC](../SPEC.md) · [Full spec](../docs/spec.md) · [Experiments](../docs/plans/ml-experiments.md) · [Tasks](../docs/plans/task-breakdown.md) · [TODO](todo.md).

**Đợt này chỉ sửa SPEC/PLAN; không code, train, download, sửa config, hay deploy.**
