# Báo cáo thí nghiệm N5: cải thiện F1 TOXIC

**Ngày:** 2026-11-23  
**Trạng thái:** Đã chạy đủ ba seed và xác minh artifacts.

## Mục tiêu và phạm vi

Thí nghiệm so sánh threshold của Logistic Regression có trọng số cân bằng, Logistic Regression không trọng số và Linear SVM trên cùng FastText mean-pooled embeddings của N5. Mục tiêu là cải thiện F1 TOXIC, đồng thời theo dõi Macro-F1 và F1 CLEAN.

Chỉ dùng N2 `train.parquet` để fit mô hình và N2 `val.parquet` để chọn threshold, đánh giá. Không dùng N4 hoặc test set. Không sửa dữ liệu N2 hay artifacts baseline N5.

| Tập dữ liệu | SHA-256 |
|---|---|
| N2 train | `097b32bea731bf1a4771a8485155d339a96027d2819fe8a38aa2c9d2e2f32a73` |
| N2 validation | `4c33430a520818078aef6bcc34b3e2573873445dc7040c5729313e11bb60197a` |

## Thiết kế thí nghiệm

Mỗi seed dùng FastText model đã lưu trong N5, được huấn luyện chỉ trên N2 train. Giữ nguyên mean pooling và chạy các ứng viên sau:

- `lr_balanced`: Logistic Regression N5 hiện có, `C=1`, `lbfgs`, `max_iter=1000`, `class_weight="balanced"`.
- `lr_unweighted`: Logistic Regression cùng cấu hình nhưng `class_weight=None`.
- `linear_svc`: Linear SVM, `C=1`, `max_iter=2000`, `class_weight="balanced"`.

Chọn threshold bằng F1 TOXIC trên validation. Khi nhiều threshold đồng hạng, chọn threshold cao hơn. LR dùng score xác suất; Linear SVM dùng decision margin, không phải xác suất. Threshold được chọn riêng cho mỗi seed.

## Kết quả tổng hợp

Các giá trị là mean ± population std trên ba seed. Baseline là N5 Logistic Regression balanced ở threshold 0,5.

| Cấu hình | Threshold theo seed 42 / 123 / 2026 | Macro-F1 | F1 CLEAN | F1 TOXIC |
|---|---:|---:|---:|---:|
| N5 baseline, LR balanced | 0,5000 / 0,5000 / 0,5000 | 0,622218 ± 0,001276 | 0,808546 ± 0,001488 | 0,435891 ± 0,001425 |
| LR balanced, threshold tuned | 0,6002 / 0,6031 / 0,6042 | **0,661476 ± 0,001131** | **0,870965 ± 0,000701** | 0,451987 ± 0,002208 |
| LR không trọng số, threshold tuned | 0,2115 / 0,2254 / 0,1953 | 0,658345 ± 0,004976 | 0,862556 ± 0,008657 | **0,454134 ± 0,001391** |
| Linear SVM, threshold tuned | 0,1662 / 0,1650 / 0,1336 | 0,661226 ± 0,002384 | 0,869859 ± 0,003850 | 0,452592 ± 0,001317 |

Theo mục tiêu F1 TOXIC, LR không trọng số với threshold tuned là ứng viên tốt nhất. F1 TOXIC tăng 0,018243 tuyệt đối so với baseline, tương đương 4,18% tương đối. Mục tiêu 0,55 chưa đạt. Nếu chọn theo Macro-F1 như metric chính trong kế hoạch dự án, LR balanced tuned có Macro-F1 cao nhất.

LR không trọng số ở threshold mặc định 0,5 chỉ đạt F1 TOXIC mean 0,275039. Threshold tuning tăng F1 TOXIC lên 0,454134. Linear SVM ở threshold mặc định đạt F1 TOXIC mean 0,437399; sau tuning đạt 0,452592.

## Phân tích lỗi

Baseline có precision TOXIC mean 0,317951 và recall 0,692946. LR không trọng số tuned đạt precision 0,377226 và recall 0,572614. Threshold làm giảm false positive nhưng cũng tăng false negative.

Các confusion counts dưới đây được cộng qua ba lần dự đoán trên cùng validation set. Thứ tự là TN / FP / FN / TP.

| Cấu hình | TN / FP / FN / TP |
|---|---:|
| Baseline N5 | 10953 / 4299 / 888 / 2004 |
| LR không trọng số tuned | 12506 / 2746 / 1236 / 1656 |

Kết quả cho thấy baseline nhận diện được nhiều TOXIC hơn nhưng gán nhầm nhiều CLEAN thành TOXIC. Threshold tuned giảm FP và cải thiện precision, đổi lại recall giảm. Các threshold được chọn và chấm trên cùng validation set nên các F1 tuned có thể lạc quan.

## Kết quả theo seed cho cấu hình F1 TOXIC tốt nhất

| Seed | Threshold | Macro-F1 | F1 CLEAN | F1 TOXIC | Precision TOXIC | Recall TOXIC |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0,211455 | 0,658732 | 0,863899 | 0,453565 | 0,377762 | 0,567427 |
| 123 | 0,225401 | 0,664236 | 0,872423 | 0,456049 | 0,392804 | 0,543568 |
| 2026 | 0,195318 | 0,652066 | 0,851346 | 0,452786 | 0,361111 | 0,606846 |

## Artifacts và MLflow

Artifacts theo seed, gồm config, dataset hashes, validation scores, metrics và classifier artifacts, nằm tại:

`models/baselines/N5-toxic-tuning-run-d15f90a55f944793b61a49dfcdb9360b/`

MLflow experiment là `N5-toxic-F1-tuning`. Ba seed run và summary run đều có trạng thái `FINISHED`.

| Run | MLflow run ID |
|---|---|
| Seed 42 | `162e9fbc6d534029938806c695f95e6b` |
| Seed 123 | `851d3cb5b1f04d1aa951d1e7a1889354` |
| Seed 2026 | `a24fe70decbb4006ad91c248d37ba45f` |
| Summary | `86516c6788154b5f9a8628332b43e4c8` |

Thời gian chạy thành công là 27,54 giây. Một lần chạy ban đầu dừng ở seed 42 do dùng `log_metric`, không có trong `MLflowTracker`. Sau khi đổi sang `log_metrics`, toàn bộ ba seed chạy thành công. Run lỗi `72a5820ee1d34fc19e189b671e2c2d31` vẫn được giữ trong MLflow history với trạng thái `FAILED`; output cục bộ dở dang đã được xóa và run này không nằm trong kết quả báo cáo.

## Xác minh

- `pytest tests/test_n5_fasttext.py tests/test_n5_toxic_tuning.py -q`: 5 passed. Test all-negative tổng hợp phát một cảnh báo sklearn dự kiến.
- Đối chiếu lại metrics từ validation predictions: các giá trị khớp metrics JSON và MLflow.
- Cả ba seed và summary run có trạng thái `FINISHED`.
- SHA-256 N2 train/validation khớp với config của từng seed.
- Fingerprints của artifacts baseline N5 không đổi.
- Không chạy full test suite và không đánh giá trên locked test set.
