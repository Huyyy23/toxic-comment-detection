# Implementation Plan — 4 thành viên làm song song

**Version:** 4.0.0-SCRATCH-FIRST · **Ngày:** 2026-11-11 · **Trạng thái:** PLAN ONLY.

## Mục tiêu
Bốn thành viên huấn luyện độc lập Logistic Regression, LinearSVC, LSTM/BiLSTM+Attention và TextCNN **từ đầu**. Nhóm trưởng (TV3) chốt Data Contract và làm PhoBERT pretrained **đối chứng mở rộng**. FastAPI minimal chỉ là demo sau nghiên cứu.

## Chốt trách nhiệm
| Member | Làm chính | Thực nghiệm | Bàn giao |
|---|---|---|---|
| TV1 | EDA, preprocessing, Logistic Regression | EXP-01 | data analysis, LR, ablation, val predictions |
| TV2 | TF-IDF, LinearSVC, tuning/feature selection | EXP-02 | SVM, ablation, val predictions |
| TV3 — Lead | Gate G0, LSTM→BiLSTM→Attention, error analysis; optional PhoBERT | EXP-00,03,05,07 | split manifest, DL artifacts, comparative report |
| TV4 | TextCNN variants, conditional ensemble | EXP-04,06 | CNN, val predictions, ensemble assessment |

## Phụ thuộc tối thiểu: Data Contract Gate G0
- **TV3 chịu trách nhiệm**, không phải chờ EDA chuyên sâu của TV1; phải xác thực labels/schema, dữ liệu trùng lặp, leakage và phù hợp của official split trước khi khóa.
- Dataset ưu tiên UIT-ViHSD, ViCTSD chỉ là nguồn OOD/cross-dataset conditional. Nếu tự chia: stratified 70/15/15, seed lưu `split_manifest`; official split phù hợp được ưu tiên.
- Cố định cùng IDs/split, nhãn, version, metrics (Macro-F1), output predictions contract; test locked.
- Trước G0: TV1 EDA; TV2–TV4 **code skeleton/smoke test với fixture**. Sau G0: cả bốn train chính thức **song song**.

## Lịch 4 tuần (gợi ý, tùy deadline)
| Tuần | TV1 | TV2 | TV3 (Lead) | TV4 | Gate |
|---|---|---|---|---|---|
| 1 | EDA + LR skeleton | SVM skeleton | G0 manifest + LSTM skeleton | CNN skeleton | G0 |
| 2 | LR 3 variants | LinearSVC 3 variants | LSTM/BiLSTM/Attention 3 variants | CNN 3 variants | G1: 4 official runs |
| 3 | LR error review | SVM ablation | Error analysis; optional PhoBERT | CNN ablation; conditional ensemble | G2: best candidates frozen by val |
| 4 | Coordinate locked test | Reproduction + SVM report | Review + overall research report | Ensemble/CNN report; API support | G3: final test, demo, documentation |

GPU resource: nếu cùng RTX 4060 8GB, TV3 và TV4 **phát triển song song nhưng xếp lịch GPU**, không chạy đồng thời hai DL jobs lớn. PhoBERT chỉ chạy sau khi scratch deliverables đạt chuẩn.

## Branch/contract
- `feature/data-logreg` (TV1); `feature/svm` (TV2); `feature/bilstm` (TV3); `feature/textcnn` (TV4). PR review trước merge.
- Shared `id,text,label,split` data contract; prediction output `id,true_label,predicted_label,probabilities_if_available,run_id`.
- Mỗi EXP có hypothesis, ≥3 meaningful variants, MLflow logs, metrics, plots, artifact, ablation + report. Checkpoints nặng lưu ngoài git, version trong metadata.
- Không sửa split do một cá nhân tự ý. Contract changes cần nhóm trưởng phê duyệt và chạy lại các kết quả liên quan.

## Các điểm chặn rõ ràng
**G0:** split/schema/hash/seed/leakage checks đã chốt → official train mở.  
**G1:** 4 mô hình scratch có official validation runs và artifacts.  
**G2:** tuning, error analysis, ablation, optional ensemble/PhoBERT hoàn tất hoặc đánh dấu skipped; freeze final candidates.  
**G3:** one-shot locked test, notebook và FastAPI minimal demo (khi triển khai sau).

## Chưa làm trong đợt cập nhật này
Không sửa `src/`, `backend/`, `configs/`, `tests/`, `frontend/`; không download dataset, train, install packages, deploy, hay xóa file code. Đây chỉ là kế hoạch triển khai tương lai. Các tài liệu code/history cũ ngoài nhóm PLAN/SPEC có thể chưa đồng bộ; source of truth: SPEC v4 và docs/plans v4.
