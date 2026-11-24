# N8 — Calibration và chọn threshold trên validation

**Trạng thái:** hoàn tất trên 90 official validation runs đã có probability outputs. Không train/fine-tune model mới; không truy cập test hoặc locked-test artifacts.

**Validation contract:** CLEAN=0, TOXIC=1; 6.048 mẫu, 6.048 ID duy nhất. File SHA-256 `4c33430a520818078aef6bcc34b3e2573873445dc7040c5729313e11bb60197a`; SHA-256 ID đã sort `7f454a9926bf0a616c54426513b98874b944ea06921c927cd336c2203152f01d`. Tất cả 90 output probability đã đối chiếu đủ ID, nhãn nhị phân và xác suất trong [0,1].

**Protocol:** Platt scaling trên log-odds của P(TOXIC), fit trên validation; ECE dùng 10 bins; chọn threshold sau calibration bằng Macro-F1 trên grid 0,10–0,90 bước 0,01. Số trong bảng là trung bình đều theo run, không phải pooled prediction.

| Nhóm candidate | Runs | ECE trước → sau N8 | Macro-F1 tại threshold chọn | F1 TOXIC tại threshold chọn |
|---|---:|---:|---:|---:|
| N6 Logistic Regression (A/B/C/D và D-calibrated) | 15 | 0,041081 → 0,006735 | 0,790240 | 0,642468 |
| N6 LinearSVC (C-calibrated) | 3 | 0,011115 → 0,006498 | 0,794711 | 0,656155 |
| N5 FastText | 3 | 0,253967 → 0,016612 | 0,663301 | 0,440310 |
| N7 recurrent (LSTM/BiLSTM/Attention) | 9 | 0,271023 → 0,018409 | 0,726951 | 0,543441 |
| TV4 CNN/Char-CNN/hybrid | 54 | 0,067678 → 0,010079 | 0,765865 | 0,601874 |
| PhoBERT N2 baseline | 3 | 0,077536 → 0,018093 | 0,802938 | 0,667442 |
| PhoBERT N4 augmentation | 3 | 0,093917 → 0,017166 | 0,792395 | 0,649573 |

Threshold, Platt parameters, ECE per-run, calibrated probabilities và input lineage nằm trong các báo cáo/thư mục `artifacts/N8/{baselines,fasttext,recurrent,cnn,phobert}/`.

**Giới hạn:** calibrator và threshold được fit/chọn và đánh giá trên cùng validation set; các metrics là in-sample, có bias lạc quan, không phải ước lượng tổng quát hóa. ECE trước của SVM là ECE từ probability artifact đã sigmoid-calibrated trên train folds; raw LinearSVC margins không được coi là xác suất. Các N6 SVM variant chưa calibrated bị loại khỏi phần N8 vì predictions không có probabilities và run dirs không lưu model weights/joblib để tái tạo inference mà không train. PhoBERT loại 3 run SUPERSEDED và 1 run INVALID.
