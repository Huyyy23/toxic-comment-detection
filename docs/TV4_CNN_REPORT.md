# Báo cáo Thực nghiệm TV4 (EXP-04 / N7): Word TextCNN, Char-CNN & Hybrid BiGRU/BiLSTM + Char-CNN

- **Thành viên phụ trách:** TV4 (TheAnh)
- **Ngày hoàn thành Phần 0–Phần 6:** 2026-11-24
- **Trạng thái:** `EXPERIMENTALLY VALIDATED` đầy đủ từ **Validation (`val.parquet`, `n = 6,048`)** cho **54 runs đơn lẻ (Phần 2–4)** và **10 cấu hình Conditional Ensemble (Phần 5 / `EXP-06`)** đến **One-Shot Locked Test (`test.parquet`, `n = 6,048`, Phần 6 / `N10`)**.
- **Phạm vi bài toán:** Phân loại bình luận độc hại tiếng Việt nhị phân (`CLEAN = 0`, `TOXIC = 1`) theo `SPEC.md` v5.0.0 và `docs/decisions.md` (D1–D4).
- **Thiết bị thực nghiệm:** `NVIDIA GeForce RTX 4060 Laptop GPU` (8 GB VRAM, CUDA, PyTorch `2.6.0+cu124`), 3 seeds cố định `[42, 43, 44]` cho toàn bộ **18 biến thể × 3 seeds = 54 runs**.

---

## 1. Dữ liệu, Chống rò rỉ (Anti-Leakage) và Truy vết Artifacts

Toàn bộ 54 runs tuân thủ nghiêm ngặt quy tắc chống rò rỉ dữ liệu trong `AGENTS.md` §5:

1. **Tập dữ liệu chuẩn (`binary_n2_v1`):**
   - `data/processed/binary_n2_v1/train.parquet`: `28,231` mẫu (`84.07% CLEAN`, `15.93% TOXIC`), SHA-256 `097b32bea731bf1a4771a8485155d339a96027d2819fe8a38aa2c9d2e2f32a73`.
   - `data/processed/binary_n2_v1/val.parquet`: `6,048` mẫu (`5,084 CLEAN`, `964 TOXIC`; gồm `4,559` mẫu `vihsd` và `1,489` mẫu `victsd`), SHA-256 `4c33430a520818078aef6bcc34b3e2573873445dc7040c5729313e11bb60197a`.
2. **Tập dữ liệu tăng cường train-only (`binary_n4_training_v1`, dùng cho biến thể đối chứng `CNN-AUG-N4`):**
   - `data/processed/binary_n4_training_v1/train.parquet`: `33,906` mẫu (`70.0% CLEAN`, `30.0% TOXIC`, thêm `5,675` mẫu `TOXIC` chỉ trên train), SHA-256 `f57fbf0a55de6b08c8cba1cd366c3895bfed495f2817069f892ba27b5786542e`.
   - `val.parquet`: giữ nguyên `6,048` mẫu gốc (`4c33430a520818078aef6bcc34b3e2573873445dc7040c5729313e11bb60197a`).
3. **Kỷ luật fit train-only & kiểm chứng độc lập:**
   - Từ điển mức từ (`SequenceVocab(mode="word", min_freq=2)`) và mức ký tự (`SequenceVocab(mode="char", min_freq=2)`), trọng số lớp (`class_weight`) và bộ lấy mẫu (`WeightedRandomSampler`) chỉ được fit trên `train.parquet` của từng run.
   - `test.parquet` hoàn toàn không được đọc hoặc truy cập trong `src/models/cnn_pipeline.py` và `src/evaluation/cnn_report.py`.
   - Kiểm chứng tự động (`python -m src.evaluation.cnn_report --verify-reload --device cuda`): **54/54 runs** khớp 100% SHA-256 dataset, trạng thái MLflow `FINISHED` (experiment `TV4-CNN-Binary`), tổng xác suất `P(CLEAN) + P(TOXIC) = 1.0`, và nạp lại checkpoint `best.pt` khớp dự đoán **`6,048 / 6,048` mẫu** trên mọi run (`all_integrity_checks_passed = true`).

---

## 2. Bảng Tổng hợp 18 Biến thể × 3 Seeds trên Tập Validation (`n = 6,048`)

> **Quy tắc xếp hạng (`docs/decisions.md`):** Xếp hạng mô hình chỉ dựa trên **Uncalibrated Validation Macro-F1** (ngưỡng mặc định `argmax` / `0.50`). Các số liệu xác suất thô từ lớp `softmax` được ghi rõ là `Raw ECE` (`is_calibrated: false`).

| Hạng | Biến thể | Kiến trúc | Tiền xử lý | Cấu hình chính / Thay đổi | Val Macro-F1 (mean ± std) | Val F1-TOXIC (mean ± std) | Weighted-F1 | Balanced Acc | PR-AUC | Raw ECE | ViCTSD Macro-F1 (F1-TOXIC) | ViHSD Macro-F1 (F1-TOXIC) | Params (Size) | Train Time (s) | Latency (ms/mẫu) | Peak VRAM (MB) | Best Seed (Val Macro-F1) |
|---:|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|:---:|
| 1 | **`HYBRID-RNN-01`** | `bigru_char_cnn` | `raw` | BiGRU (`h=128`) + Char-CNN `[3,4,5,7]`, `filters=128`, `drop=0.5` | **0.7733 ± 0.0072** | 0.6149 ± 0.0130 | 0.8812 | 0.7614 | 0.6673 | 0.0635 | **0.6542** (**0.3766**) | 0.7961 (0.6606) | 1,964,226 (22.69 MB) | 207.2 | 0.2232 | 230.8 | `s42` (**0.7806**) |
| 2 | **`HYBRID-RNN-WD`** | `bigru_char_cnn` | `raw` | `HYBRID-RNN-01` + `weight_decay=1e-4` | **0.7731 ± 0.0052** | **0.6155 ± 0.0103** | 0.8804 | **0.7641** | 0.6652 | 0.0653 | 0.6420 (0.3552) | **0.7982** (**0.6651**) | 1,964,226 (22.69 MB) | 188.3 | 0.1609 | 208.2 | `s42` (0.7791) |
| 3 | **`CNN-01-WD`** | `textcnn` | `basic` | Word TextCNN `[3]` + `weight_decay=1e-4` | **0.7705 ± 0.0027** | 0.6058 ± 0.0056 | **0.8826** | 0.7461 | **0.6751** | 0.0494 | 0.6211 (0.3044) | 0.7951 (0.6560) | 1,307,906 (15.12 MB) | **39.1** | 0.0502 | 128.3 | `s42` (0.7734) |
| 4 | **`CNN-01`** | `textcnn` | `basic` | Baseline Word TextCNN single-kernel `[3]`, `filters=128`, `drop=0.5` | 0.7702 ± 0.0041 | 0.6065 ± 0.0091 | 0.8817 | 0.7495 | 0.6734 | 0.0550 | 0.6285 (0.3195) | 0.7937 (0.6548) | 1,307,906 (15.12 MB) | 58.6 | 0.0644 | **105.8** | `s42` (0.7740) |
| 5 | **`CNN-04b`** | `textcnn` | `basic` | Word TextCNN `[3,4,5]` + `weight_decay=1e-4` | 0.7688 ± 0.0045 | 0.6028 ± 0.0112 | 0.8818 | 0.7448 | 0.6561 | 0.0769 | 0.6342 (0.3285) | 0.7907 (0.6486) | 1,456,130 (16.82 MB) | 53.7 | 0.0546 | 129.8 | `s42` (0.7739) |
| 6 | **`CNN-03a`** | `textcnn` | `basic` | Word TextCNN `[3,4,5]`, giảm `num_filters=64` | 0.7679 ± 0.0055 | 0.6018 ± 0.0081 | 0.8810 | 0.7450 | 0.6661 | 0.0488 | 0.6242 (0.3118) | 0.7921 (0.6513) | 1,357,250 (15.69 MB) | 61.0 | 0.0642 | 119.9 | `s43` (0.7714) |
| 7 | **`CNN-03b`** | `textcnn` | `basic` | Word TextCNN `[3,4,5]`, tăng `num_filters=256` | 0.7676 ± 0.0017 | 0.6013 ± 0.0040 | 0.8810 | 0.7445 | 0.6527 | 0.0890 | 0.6277 (0.3159) | 0.7901 (0.6481) | 1,653,890 (19.09 MB) | 73.2 | 0.0621 | 145.9 | `s44` (0.7687) |
| 8 | **`HYBRID-BILSTM-01`** | `bilstm_char_cnn` | `raw` | BiLSTM (`h=128`) + Char-CNN `[3,4,5,7]`, `filters=128`, `drop=0.5` | 0.7665 ± 0.0027 | 0.6060 ± 0.0030 | 0.8759 | 0.7623 | 0.6543 | 0.0806 | 0.6387 (0.3518) | 0.7914 (0.6553) | 2,030,274 (23.45 MB) | 225.8 | 0.2213 | 230.6 | `s44` (0.7690) |
| 9 | **`CNN-02`** | `textcnn` | `basic` | Multi-kernel Word TextCNN `[3,4,5]`, `filters=128`, `drop=0.5` | 0.7664 ± 0.0048 | 0.5989 ± 0.0075 | 0.8805 | 0.7427 | 0.6573 | 0.0778 | 0.6379 (0.3361) | 0.7877 (0.6435) | 1,456,130 (16.82 MB) | 84.2 | 0.0771 | 128.5 | `s42` (0.7714) |
| 10 | **`CNN-IMB-CW`** | `textcnn` | `basic` | `CNN-02` + `class_weight` nghịch đảo tần suất train | 0.7650 ± 0.0019 | 0.5999 ± 0.0028 | 0.8774 | 0.7508 | 0.6575 | 0.0827 | 0.6357 (0.3353) | 0.7863 (0.6447) | 1,456,130 (16.82 MB) | 64.9 | 0.0617 | 128.5 | `s42` (0.7668) |
| 11 | **`CNN-AUG-N4`** | `textcnn` | `basic` | `CNN-02` huấn luyện trên `binary_n4_training_v1` (30% TOXIC) | 0.7649 ± 0.0054 | 0.6034 ± 0.0131 | 0.8748 | 0.7616 | 0.6526 | 0.0789 | 0.6442 (0.3567) | 0.7858 (0.6472) | 1,763,714 (20.38 MB) | 75.5 | 0.0708 | 132.0 | `s42` (0.7695) |
| 12 | **`CNN-IMB-FOCAL`** | `textcnn` | `basic` | `CNN-02` + Focal Loss (`gamma=2.0`) | 0.7631 ± 0.0035 | 0.5973 ± 0.0130 | 0.8761 | 0.7513 | 0.6494 | 0.0668 | 0.6494 (0.3657) | 0.7837 (0.6399) | 1,456,130 (16.82 MB) | 69.3 | 0.0536 | 128.5 | `s42` (0.7670) |
| 13 | **`HYBRID-01`** | `word_char_cnn` | `raw` | Word-CNN `[3,4,5]` + Char-CNN `[3,4,5,7]`, `filters=128` | 0.7604 ± 0.0063 | 0.5916 ± 0.0133 | 0.8754 | 0.7454 | 0.6545 | 0.0856 | 0.6362 (0.3402) | 0.7830 (0.6378) | 1,963,330 (22.68 MB) | 108.6 | 0.0859 | 216.4 | `s42` (0.7663) |
| 14 | **`CNN-IMB-SAMPLER`** | `textcnn` | `basic` | `CNN-02` + `WeightedRandomSampler` cân bằng batch train | 0.7593 ± 0.0039 | 0.5939 ± 0.0047 | 0.8720 | 0.7552 | 0.6537 | 0.0844 | 0.6318 (0.3339) | 0.7814 (0.6398) | 1,456,130 (16.82 MB) | 46.6 | **0.0460** | 130.0 | `s42` (0.7637) |
| 15 | **`CNN-04a`** | `textcnn` | `basic` | `CNN-02` giảm `dropout=0.3` | 0.7576 ± 0.0096 | 0.5869 ± 0.0150 | 0.8739 | 0.7426 | 0.6431 | 0.0806 | 0.6276 (0.3234) | 0.7805 (0.6339) | 1,456,130 (16.82 MB) | 55.7 | 0.0468 | 129.8 | `s42` (0.7654) |
| 16 | **`CHAR-02`** | `char_cnn` | `raw` | Char-CNN `[3,4,5,7]`, tăng `num_filters=256`, `drop=0.5` | 0.7550 ± 0.0040 | 0.5817 ± 0.0105 | 0.8730 | 0.7388 | 0.6316 | 0.0460 | 0.6183 (0.3097) | 0.7812 (0.6337) | 345,666 (3.98 MB) | 110.9 | 0.1047 | 243.1 | `s44` (0.7573) |
| 17 | **`CHAR-03`** | `char_cnn` | `raw` | Char-CNN `[3,4,5,7]`, giảm `dropout=0.3`, `filters=128` | 0.7525 ± 0.0050 | 0.5787 ± 0.0123 | 0.8708 | 0.7397 | 0.6279 | 0.0483 | 0.6134 (0.3020) | 0.7791 (0.6314) | **188,482** (**2.18 MB**) | 69.1 | 0.0698 | 170.5 | `s42` (0.7554) |
| 18 | **`CHAR-01`** | `char_cnn` | `raw` | Baseline Char-CNN `[3,4,5,7]`, `filters=128`, `drop=0.5` | 0.7501 ± 0.0030 | 0.5696 ± 0.0020 | 0.8730 | 0.7248 | 0.6287 | **0.0387** | 0.6247 (0.3157) | 0.7730 (0.6165) | **188,482** (**2.18 MB**) | 69.6 | 0.0661 | 171.4 | `s43` (0.7525) |

---

## 3. Phân tích Cải tiến Có Căn cứ theo Hướng 2 (`AGENTS.md` §4 & §11)

Mọi vòng thí nghiệm trong Phần 2 và Phần 3 đều chỉ thay đổi **một yếu tố mỗi lần** và được ghi theo cấu trúc **Baseline → Vấn đề → Giả thuyết → Thay đổi → Kết quả → Kết luận**:

### Vòng 1 — Kích thước cửa sổ tích chập mức từ: Single-kernel `[3]` (`CNN-01`) vs Multi-kernel `[3, 4, 5]` (`CNN-02`)
- **Baseline:** `CNN-01` (`textcnn`, `kernel_sizes=[3]`, `num_filters=128`, `dropout=0.5`) đạt Val Macro-F1 `0.7702 ± 0.0041`, F1-TOXIC `0.6065 ± 0.0091`, PR-AUC `0.6734`, Raw ECE `0.0550`.
- **Vấn đề:** `CNN-01` có hiệu năng rất tốt trên `ViHSD` (`0.7937`) nhưng thấp hơn trên các bình luận tin tức dài của `ViCTSD` (Macro-F1 `0.6285`, F1-TOXIC `0.3195`).
- **Giả thuyết:** Mở rộng sang đa cửa sổ `[3, 4, 5]` giúp nắm bắt các cụm phủ định/mỉa mai dài 4–5 từ trên `ViCTSD`.
- **Thay đổi:** Chỉ đổi `kernel_sizes` từ `[3]` sang `[3, 4, 5]` (`CNN-02`, số tham số tăng từ `1.31M` lên `1.46M`).
- **Kết quả:** `CNN-02` cải thiện rõ trên `ViCTSD` (Macro-F1 tăng từ `0.6285` lên `0.6379`, F1-TOXIC tăng từ `0.3195` lên `0.3361`), nhưng giảm trên `ViHSD` (`0.7937` → `0.7877`) và tăng overfit/ECE (`0.0550` → `0.0778`), khiến tổng Val Macro-F1 giảm nhẹ xuống `0.7664 ± 0.0048`.
- **Kết luận:** **Bác bỏ một phần giả thuyết tổng thể, chấp nhận trên miền câu dài (`ViCTSD`).** Multi-kernel `[3,4,5]` có lợi cho ngữ cảnh dài `ViCTSD` nhưng dễ overfit trên `ViHSD` nếu không thêm L2 regularization.

### Vòng 2 — Dung lượng bộ lọc (Filter Tuning): `64` (`CNN-03a`) vs `128` (`CNN-02`) vs `256` (`CNN-03b`)
- **Baseline:** `CNN-02` (`filters=128`, Macro-F1 `0.7664`, Raw ECE `0.0778`).
- **Vấn đề:** Cần xác định `CNN-02` đang dư thừa dung lượng (overfit) hay thiếu số lượng bộ lọc đặc trưng.
- **Giả thuyết:** Giảm xuống `64` filters (`CNN-03a`) giúp giảm overfit và hạ ECE; tăng lên `256` filters (`CNN-03b`) tăng độ phủ n-gram nhưng có nguy cơ tăng độ tự tin quá mức.
- **Thay đổi:** Chỉ đổi `num_filters` thành `64` (`CNN-03a`) và `256` (`CNN-03b`).
- **Kết quả:** `CNN-03a` (`64` filters) tăng tổng Val Macro-F1 lên `0.7679 ± 0.0055`, PR-AUC tăng lên `0.6661` và giảm mạnh Raw ECE từ `0.0778` xuống `0.0488`. Ngược lại, `CNN-03b` (`256` filters) đạt Macro-F1 `0.7676 ± 0.0017` nhưng đẩy Raw ECE lên `0.0890`.
- **Kết luận:** **Chấp nhận giả thuyết thu gọn (`CNN-03a`).** Với embedding khởi tạo ngẫu nhiên trên `28.2k` mẫu train, `64–128` filters cân bằng tốt hơn `256` filters cả về Macro-F1 lẫn độ hiệu chuẩn xác suất.

### Vòng 3 — Điều chuẩn (Regularization): `dropout=0.3` (`CNN-04a`) vs `weight_decay=1e-4` (`CNN-04b`, `CNN-01-WD`)
- **Baseline:** `CNN-02` (`dropout=0.5, wd=0.0`, Macro-F1 `0.7664`) và `CNN-01` (`dropout=0.5, wd=0.0`, Macro-F1 `0.7702`).
- **Vấn đề:** Đường cong huấn luyện cho thấy khoảng cách lớn giữa Train Macro-F1 (`> 0.90` sau epoch 8) và Val Macro-F1 (`~0.77`).
- **Giả thuyết:** Giảm dropout xuống `0.3` (`CNN-04a`) sẽ làm nặng thêm overfitting, trong khi giữ `dropout=0.5` và thêm L2 `weight_decay=1e-4` (`CNN-04b`, `CNN-01-WD`) sẽ kiểm soát độ lớn trọng số embedding/linear và nâng Val Macro-F1.
- **Thay đổi:** Thử `dropout=0.3` (`CNN-04a`), `weight_decay=1e-4` trên `[3,4,5]` (`CNN-04b`) và `weight_decay=1e-4` trên `[3]` (`CNN-01-WD`).
- **Kết quả:**
  - `CNN-04a` (`dropout=0.3`) tụt sâu xuống Macro-F1 `0.7576 ± 0.0096` (F1-TOXIC `0.5869`).
  - `CNN-04b` (`[3,4,5] + wd=1e-4`) tăng lên `0.7688 ± 0.0045` (`+0.0024` so với `CNN-02`).
  - `CNN-01-WD` (`[3] + wd=1e-4`) đạt **`0.7705 ± 0.0027`**, PR-AUC cao nhất toàn bộ 18 biến thể (**`0.6751`**), Weighted-F1 cao nhất (**`0.8826`**), giảm phương sai giữa các seed và rút thời gian train trung bình xuống `39.1s`.
- **Kết luận:** **Chấp nhận hoàn toàn giả thuyết L2 regularization.** `CNN-01-WD` trở thành mô hình **Word TextCNN thuần tốt nhất** của TV4.

### Vòng 4 — Xử lý mất cân bằng lớp D3 & Tăng cường dữ liệu N4 (`CNN-IMB-CW`, `CNN-IMB-FOCAL`, `CNN-IMB-SAMPLER`, `CNN-AUG-N4`)
- **Baseline:** `CNN-02` (CE chuẩn trên `binary_n2_v1`, Macro-F1 `0.7664`, Balanced Acc `0.7427`, ViCTSD Macro-F1 `0.6379`, ViCTSD F1-TOXIC `0.3361`).
- **Vấn đề:** Tỷ lệ lớp `TOXIC` chỉ chiếm `15.93%` (`11.08%` trên `ViCTSD`), khiến recall lớp `TOXIC` trên `ViCTSD` chỉ đạt `~28–34%` ở ngưỡng mặc định `0.50`.
- **Giả thuyết:** Áp dụng 3 cơ chế D3 (`class_weight`, `focal gamma=2.0`, `WeightedRandomSampler`) hoặc tập train tăng cường `binary_n4_training_v1` (`30% TOXIC`) sẽ nâng Balanced Accuracy và F1-TOXIC trên miền khó `ViCTSD`.
- **Thay đổi:** Lần lượt bật `imbalance_method = class_weight` (`CNN-IMB-CW`), `focal` (`CNN-IMB-FOCAL`), `sampler` (`CNN-IMB-SAMPLER`) và `use_augmentation = true` (`CNN-AUG-N4`).
- **Kết quả:**
  - Cả 4 phương pháp đều **tăng Balanced Accuracy tổng thể** từ `0.7427` lên `0.7508` (`CW`), `0.7513` (`FOCAL`), `0.7552` (`SAMPLER`) và **`0.7616` (`AUG-N4`)**.
  - Đặc biệt trên miền khó `ViCTSD`, `CNN-IMB-FOCAL` nâng Macro-F1 từ `0.6379` lên **`0.6494`** (`F1-TOXIC` từ `0.3361` lên **`0.3657`**), và `CNN-AUG-N4` nâng `ViCTSD` Macro-F1 lên **`0.6442`** (`F1-TOXIC = 0.3567`, đồng thời tăng tổng `F1-TOXIC` lên `0.6034`).
  - Tuy nhiên, ở ngưỡng cứng `0.50`, việc đẩy recall `TOXIC` làm tăng False Positives (`CLEAN -> TOXIC`) trên `ViHSD`, khiến tổng Uncalibrated Val Macro-F1 đạt `0.7650` (`CW`), `0.7649` (`AUG-N4`), `0.7631` (`FOCAL`) và `0.7593` (`SAMPLER`).
- **Kết luận:** **Ghi nhận đánh đổi Precision–Recall rõ ràng.** Thay vì ép trọng số loss làm lệch phân phối xác suất toàn cục, kết quả Phần 4 chỉ ra rằng giữ hàm loss CE chuẩn kết hợp **Platt Calibration + điều chỉnh ngưỡng** giải quyết bài toán mất cân bằng hiệu quả hơn mà không làm giảm chất lượng xếp hạng (`PR-AUC`).

### Vòng 5 — Đặc trưng Hình thái Ký tự (`CHAR-01`, `CHAR-02`, `CHAR-03`) theo `SPEC.md` N7
- **Baseline:** `CHAR-01` (`char_cnn`, `kernels=[3,4,5,7]`, `filters=128`, `dropout=0.5`, văn bản `raw` NFC) chỉ dùng `188,482` tham số (`2.18 MB`), đạt Val Macro-F1 `0.7501 ± 0.0030`, F1-TOXIC `0.5696`, Raw ECE thấp nhất (`0.0387`).
- **Vấn đề:** `CHAR-01` có dung lượng rất nhỏ (`188k` params, bằng `1/7` Word TextCNN) nên đang underfit trên các mẫu từ ghép dài.
- **Giả thuyết:** Tăng `num_filters=256` (`CHAR-02`) hoặc giảm `dropout=0.3` (`CHAR-03`) giúp tăng năng lực biểu diễn ký tự mà không cần tách từ.
- **Thay đổi:** Chỉ đổi `num_filters=256` (`CHAR-02`) hoặc `dropout=0.3` (`CHAR-03`).
- **Kết quả:** Cả hai hướng đều cải thiện rõ rệt so với `CHAR-01`: `CHAR-03` tăng lên `0.7525 ± 0.0050` (F1-TOXIC `0.5787`), và `CHAR-02` đạt **`0.7550 ± 0.0040`** (F1-TOXIC **`0.5817`**, `+0.0121` so với `CHAR-01`, chỉ tốn `345,666` params / `3.98 MB`).
- **Kết luận:** **Chấp nhận giả thuyết.** Dù đứng độc lập thấp hơn Word TextCNN khoảng `~1.5%` Macro-F1 do thiếu ngữ nghĩa mức từ, nhánh `Char-CNN` rất gọn nhẹ và cung cấp tín hiệu hình thái ký tự bổ trợ quan trọng khi ghép vào kiến trúc Hybrid.

### Vòng 6 — Kiến trúc Lai (Hybrid Word/BiLSTM/BiGRU + Char-CNN) theo `TV4.md` N7
- **Baseline:** `CNN-02` (Word-CNN `[3,4,5]`, `0.7664`) và `CHAR-01` (Char-CNN `[3,4,5,7]`, `0.7501`).
- **Vấn đề:** Word-CNN thuần nhạy cảm với từ ngoài từ điển (`<unk>`, teencode, viết tắt, lách luật như `đ.m`, `v*l`) và mất thứ tự phụ thuộc xa; Char-CNN thuần thiếu ngữ nghĩa câu dài trên `ViCTSD`.
- **Giả thuyết:** Kết hợp biểu diễn tuần tự mức từ (**BiGRU** hoặc **BiLSTM** với masked average + max pooling) cùng nhánh **Char-CNN** `[3,4,5,7]` trên văn bản `raw` sẽ đồng thời cải thiện cả `ViHSD` (nhiễu ký tự/teencode) lẫn `ViCTSD` (phụ thuộc tuần tự câu dài).
- **Thay đổi:** So sánh đối chứng 4 cấu hình Hybrid:
  1. `HYBRID-01` (`Word-CNN [3,4,5] + Char-CNN [3,4,5,7]`)
  2. `HYBRID-BILSTM-01` (`BiLSTM + Char-CNN`)
  3. `HYBRID-RNN-01` (`BiGRU + Char-CNN`)
  4. `HYBRID-RNN-WD` (`BiGRU + Char-CNN + weight_decay=1e-4`)
- **Kết quả:**
  - `HYBRID-01` (`Word-CNN + Char-CNN`) chỉ đạt `0.7604 ± 0.0063` do cả hai nhánh tích chập cục bộ cùng overfit nhanh từ sau epoch 3 (xem biểu đồ `learning_curves.png`).
  - Thay nhánh từ bằng bộ mã hóa hồi quy hai chiều cải thiện vượt bậc: `HYBRID-BILSTM-01` đạt `0.7665 ± 0.0027` (`F1-TOXIC = 0.6060`, Balanced Acc `0.7623`).
  - **`HYBRID-RNN-01` (`BiGRU + Char-CNN`) dẫn đầu toàn bộ 18 biến thể** với Val Macro-F1 **`0.7733 ± 0.0072`** (best seed 42 đạt **`0.7806`**, F1-TOXIC **`0.6287`**), đồng thời đạt Macro-F1 cao nhất trên cả **`ViCTSD` (`0.6542`, F1-TOXIC `0.3766`)** lẫn `ViHSD` (`0.7961`).
  - **`HYBRID-RNN-WD` (`BiGRU + Char-CNN + wd=1e-4`)** đạt Val Macro-F1 **`0.7731 ± 0.0052`**, giảm độ lệch chuẩn giữa 3 seeds từ `0.0072` xuống `0.0052`, đạt **F1-TOXIC trung bình cao nhất (`0.6155 ± 0.0103`)**, **Balanced Accuracy cao nhất (`0.7641`)** và **`ViHSD` Macro-F1 cao nhất (`0.7982`)**, với peak VRAM chỉ `208.2 MB` (`<< 7 GB`).
- **Kết luận:** **Chấp nhận hoàn toàn giả thuyết N7.** `BiGRU + Char-CNN` (`HYBRID-RNN-01` / `HYBRID-RNN-WD`) vượt qua cả `BiLSTM + Char-CNN` (`+0.0068` Macro-F1, ít hơn `66k` tham số và nhanh hơn `~18–37s/run`) lẫn mọi biến thể Word TextCNN/Char-CNN đơn lẻ.

---

## 4. Phần 4A — Chẩn đoán Calibration (Temperature vs Platt Scaling) & Ngưỡng Quyết định

> **Cảnh báo minh bạch học thuật (`in_sample_validation_only = true`):** Toàn bộ phép hiệu chuẩn (`Temperature Scaling`, `Platt Scaling`) và quét ngưỡng dưới đây được fit và đánh giá chẩn đoán trên `val.parquet` (`n = 6,048`), tuyệt đối **không** chạm `test.parquet` và **không** dùng để thay đổi bảng xếp hạng chọn mô hình ở Mục 2.

| Biến thể (Best Seed) | Uncal Val Macro-F1 (`@0.50`) | Uncal Val F1-TOXIC (`@0.50`) | Raw Softmax ECE | Raw Log-Loss | Temp Scaling $T$ | Temp Scaling ECE | Temp Log-Loss | Platt Scaling ECE | Platt Log-Loss | Ngưỡng tối ưu Platt (Val In-Sample) | Macro-F1 sau Platt + Thr (Val In-Sample) | F1-TOXIC sau Platt + Thr (Val In-Sample) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`HYBRID-RNN-01` (`s42`)** | **0.7806** | **0.6287** | 0.0636 | 0.3564 | 1.774 | 0.0475 | 0.3032 | **0.0078** | **0.2866** | `0.35` | **0.7822** | **0.6303** |
| **`HYBRID-RNN-WD` (`s42`)** | 0.7791 | 0.6271 | 0.0644 | 0.3547 | 1.784 | 0.0512 | 0.3037 | **0.0109** | **0.2864** | `0.34` | 0.7796 | 0.6268 |
| **`CNN-01` (`s42`)** | 0.7740 | 0.6150 | 0.0665 | 0.3698 | 1.875 | 0.0444 | 0.3029 | **0.0125** | 0.2877 | `0.34` | 0.7749 | 0.6162 |
| **`CNN-04b` (`s42`)** | 0.7739 | 0.6155 | 0.0730 | 0.4272 | 2.290 | 0.0477 | 0.3145 | **0.0096** | 0.2966 | `0.33` | 0.7756 | 0.6185 |
| **`CNN-01-WD` (`s42`)** | 0.7734 | 0.6113 | 0.0546 | 0.3372 | 1.626 | 0.0318 | 0.2953 | **0.0064** | **0.2856** | `0.33` | 0.7745 | 0.6169 |
| **`CNN-02` (`s42`)** | 0.7714 | 0.6064 | 0.0782 | 0.5061 | 2.872 | 0.0354 | 0.3147 | **0.0085** | 0.2981 | `0.37` | 0.7717 | 0.6068 |
| **`CNN-03a` (`s43`)** | 0.7714 | 0.6077 | 0.0501 | 0.3338 | 1.602 | 0.0338 | 0.2985 | 0.0130 | 0.2886 | `0.33` | 0.7724 | 0.6130 |
| **`CNN-AUG-N4` (`s42`)** | 0.7695 | 0.6133 | 0.0760 | 0.4090 | 1.983 | 0.0683 | 0.3333 | 0.0177 | 0.2998 | `0.32` | 0.7707 | 0.6115 |
| **`HYBRID-BILSTM-01` (`s44`)** | 0.7690 | 0.6088 | 0.0722 | 0.3914 | 2.056 | 0.0480 | 0.3137 | **0.0101** | 0.2950 | `0.34` | 0.7714 | 0.6124 |
| **`CNN-03b` (`s44`)** | 0.7687 | 0.6021 | 0.0933 | 0.7282 | 4.480 | 0.0323 | 0.3210 | 0.0145 | 0.3030 | `0.30` | 0.7715 | 0.6131 |
| **`CNN-IMB-FOCAL` (`s42`)** | 0.7670 | 0.6120 | 0.0784 | 0.3452 | 1.149 | 0.0907 | 0.3408 | **0.0080** | 0.2960 | `0.28` | 0.7706 | 0.6147 |
| **`CNN-IMB-CW` (`s42`)** | 0.7668 | 0.6017 | 0.0713 | 0.4041 | 2.142 | 0.0438 | 0.3175 | **0.0060** | 0.2951 | `0.34` | 0.7681 | 0.6041 |
| **`HYBRID-01` (`s42`)** | 0.7663 | 0.6036 | 0.0966 | 0.6594 | 3.947 | 0.0594 | 0.3372 | 0.0115 | 0.3016 | `0.32` | 0.7669 | 0.6042 |
| **`CNN-04a` (`s42`)** | 0.7654 | 0.6024 | 0.0916 | 0.5624 | 3.326 | 0.0514 | 0.3335 | **0.0076** | 0.3029 | `0.35` | 0.7666 | 0.6017 |
| **`CNN-IMB-SAMPLER` (`s42`)** | 0.7637 | 0.5986 | 0.0831 | 0.4713 | 2.701 | 0.0523 | 0.3282 | **0.0058** | 0.2995 | `0.35` | 0.7651 | 0.5983 |
| **`CHAR-02` (`s44`)** | 0.7573 | 0.5884 | 0.0519 | 0.3194 | 1.165 | 0.0534 | 0.3156 | 0.0112 | 0.3075 | `0.30` | 0.7608 | 0.5975 |
| **`CHAR-03` (`s42`)** | 0.7554 | 0.5897 | 0.0738 | 0.3578 | 1.592 | 0.0716 | 0.3303 | **0.0087** | 0.3114 | `0.30` | 0.7559 | 0.5890 |
| **`CHAR-01` (`s43`)** | 0.7525 | 0.5704 | **0.0256** | 0.3237 | 1.254 | **0.0171** | 0.3168 | **0.0095** | 0.3144 | `0.32` | 0.7596 | 0.5918 |

**Phát hiện kỹ thuật cốt lõi từ Calibration:**
1. **Hiện tượng Overconfidence của Softmax thô:** Toàn bộ các mô hình CNN/Hybrid đều có nhiệt độ tối ưu $T \in [1.15, 4.48] > 1.0$, chứng tỏ logit thô tại checkpoint có Val Macro-F1 cao nhất bị sắc (overconfident) ở vùng xác suất `0.60–0.95` (ví dụ trên `HYBRID-RNN-01 s42`, bin `[0.80, 0.90]` có độ tự tin trung bình `0.8547` nhưng tỷ lệ thực nghiệm `TOXIC` chỉ đạt `0.5036`).
2. **Vì sao Platt Scaling vượt trội Temperature Scaling trên dữ liệu mất cân bằng (`15.93% TOXIC`):** Temperature Scaling chỉ co giãn độ dốc ($\text{logit}/T$) mà giữ nguyên điểm cắt `0.0` (`p = 0.50`), nên ECE chỉ giảm từ `0.0636` xuống `0.0475`. Ngược lại, **Platt Scaling** hiệu chỉnh đồng thời cả độ dốc (`coef = 0.4655`) lẫn độ lệch prior (`intercept = -0.6816`), kéo ECE từ **`0.0636` xuống chỉ còn `0.0078`** trên `HYBRID-RNN-01` (và xuống **`0.0064`** trên `CNN-01-WD`), cung cấp xác suất hậu nghiệm đáng tin cậy để kết hợp Soft/Weighted Voting trong Phần 5.
3. **Dịch chuyển ngưỡng quyết định sau Platt Scaling:** Khi xác suất đã được hiệu chuẩn đúng theo tỷ lệ lớp hiếm (`~15.9%`), ngưỡng tối ưu Macro-F1 tự nhiên dịch về vùng `0.30–0.35` (tương đồng hoàn toàn với ngưỡng `0.34` của `N6-C` trong `docs/TV2_N6_REPORT.md`). Tại ngưỡng `0.35` (chẩn đoán in-sample trên Val), `HYBRID-RNN-01 (s42)` đạt `Macro-F1 = 0.7822`, `F1-TOXIC = 0.6303`.

---

## 5. Phần 4B — Phân tích Lỗi Validation Ẩn danh (`HYBRID-RNN-01` seed 42 vs `CNN-01` seed 42)

### 5.1. Thống kê tổng hợp lỗi trên `val.parquet` (`n = 6,048`)
- **Tổng số lỗi của `HYBRID-RNN-01 (s42)`:** `691 / 6,048` mẫu (tỷ lệ lỗi `11.43%`):
  - **False Negatives (`TOXIC -> CLEAN`):** `379` mẫu (chiếm `54.85%` tổng lỗi; giảm `34` lỗi FN so với `CNN-01 s42` có `413` lỗi FN). Trong đó có `274` mẫu lỗi FN tự tin cao (`P(TOXIC) <= 0.20`).
  - **False Positives (`CLEAN -> TOXIC`):** `312` mẫu (chiếm `45.15%` tổng lỗi), trong đó có `160` mẫu lỗi FP tự tin cao (`P(TOXIC) >= 0.80`).
- **Phân bố lỗi theo nguồn dữ liệu:**
  - `vihsd`: `498 / 4,559` mẫu lỗi (`10.92%`), Confusion Matrix: `TN = 3,564`, `FP = 254`, `FN = 244`, `TP = 497` (`Macro-F1 = 0.8063`, `F1-TOXIC = 0.6783`).
  - `victsd`: `193 / 1,489` mẫu lỗi (`12.96%`), Confusion Matrix: `TN = 1,208`, `FP = 58`, `FN = 135`, `TP = 88` (`Macro-F1 = 0.6555`, `F1-TOXIC = 0.3714`).
- **Tỷ lệ lỗi theo hiện tượng ngôn ngữ (`error_rate_by_tag` trên `HYBRID-RNN-01 s42`):**

| Nhóm hiện tượng ngôn ngữ (`tag`) | Tổng số mẫu trên Val | Số mẫu lỗi (`HYBRID-RNN-01 s42`) | Tỷ lệ lỗi (`HYBRID-RNN-01`) | Tỷ lệ lỗi tham chiếu (`CNN-01 s42`) | Nhận xét chuyên môn |
|:---|---:|---:|---:|---:|:---|
| `long_text_ge_35_words` ($\ge 35$ từ) | 638 | 114 | **17.87%** | 16.46% | Nhóm có tỷ lệ lỗi cao nhất: bình luận dài chứa nhiều mệnh đề chuyển ý, trích dẫn hoặc mỉa mai gián tiếp. |
| `contains_url_or_mention` | 23 | 4 | **17.39%** | 13.04% | Cỡ mẫu nhỏ (`23` mẫu), thường đi kèm tag tên người hoặc link tranh luận. |
| `contains_negation` (từ phủ định) | 1,640 | 258 | **15.73%** | 14.39% | Các cấu trúc phủ định kép hoặc khuyên can có chứa từ nhạy cảm (`"đừng nói ngu..."`, `"không phải..."`). |
| `contains_teencode` (viết tắt/lóng) | 1,156 | 179 | **15.48%** | 14.62% | Từ lóng cảm thán (`vcl`, `vl`, `đm`, `ko`) xuất hiện ở cả câu khen hài hước (`CLEAN`) lẫn câu chửi (`TOXIC`). |
| `obfuscated_or_repeated_chars` | 1,031 | 145 | **14.06%** | 13.19% | Ký tự lặp kéo dài (`thậttt`, `nghiệppp`) hoặc viết không dấu/né bộ lọc. |
| `context_or_lexical_nuance` | 1,836 | 180 | **9.80%** | 10.89% | `HYBRID-RNN-01` giảm hơn `1.09%` tỷ lệ lỗi so với `CNN-01` nhờ BiGRU nắm bắt ngữ cảnh tuần tự. |
| `contains_emoji_or_punct` | 762 | 71 | **9.32%** | 9.97% | Biểu tượng cảm xúc cười (`😂🤣`, `:)))`) đi kèm lời châm biếm hoặc đùa cợt. |
| `short_text_le_5_words` ($\le 5$ từ) | 1,438 | 108 | **7.51%** | 8.07% | `HYBRID-RNN-01` xử lý câu cực ngắn tốt hơn `CNN-01` (`7.51%` vs `8.07%`) nhờ nhánh Char-CNN. |

- **Tỷ lệ lỗi theo độ dài bình luận (`error_rate_by_length_bucket`):**
  - `le_5_words` ($\le 5$ từ): `108 / 1,438` lỗi (**`7.51%`**)
  - `6_to_15_words` ($6\text{–}15$ từ): `258 / 2,700` lỗi (**`9.56%`**)
  - `16_to_30_words` ($16\text{–}30$ từ): `187 / 1,127` lỗi (**`16.59%`**)
  - `gt_30_words` ($> 30$ từ): `138 / 783` lỗi (**`17.62%`**)
  $\Rightarrow$ Tỷ lệ lỗi tăng đơn điệu theo độ dài câu (`7.51%` → `17.62%`), giải thích vì sao các mô hình ngữ cảnh sâu (như Transformer PhoBERT/ViSoBERT hoặc Ensemble kết hợp TF-IDF char/word n-gram của TV2 + BiGRU-CharCNN của TV4) là hướng bổ trợ tự nhiên cho các câu dài trên `15` từ.

### 5.2. Danh sách 32 Mẫu Lỗi Validation Đại diện Đã Che Thông tin Định danh (`>= 30` mẫu)

> **Quy tắc bảo vệ quyền riêng tư (`AGENTS.md` §5 & `docs/research/EXP-02-comment-provenance.md`):** Chỉ lưu `12` ký tự đầu của SHA-256 `sample_id`, che toàn bộ `@handle`, `<URL>`, `<EMAIL>`, chuỗi số `<NUM>`, tên riêng `<NAME>`, và cắt gọn tối đa `84` ký tự. Bản đầy đủ 40 mẫu ẩn danh được lưu tại `artifacts/TV4-cnn-analysis/tv4_error_analysis_val.json`.

| # | `sample_id[:12]` | Nguồn | Loại lỗi (`True -> Pred`) | `P(TOXIC)` | Số từ | Nhãn hiện tượng (`tags`) | Đoạn trích đã che PII (`redacted_snippet`) | Nguyên nhân lỗi chính |
|---:|:---|:---:|:---:|---:|---:|:---|:---|:---|
| 1 | `c85bb3e9846e` | `vihsd` | `TOXIC -> CLEAN` | 0.0000 | 5 | `short_text_le_5_words`, `contains_emoji_or_punct` | `ANH NGHỊ TRẺ VIỆT CỘNG😂🤣😅😄🤣😂` | Châm biếm chính trị viết hoa kèm chuỗi emoji cười làm lệch trọng số về `CLEAN`. |
| 2 | `8ff99af4a2c9` | `vihsd` | `TOXIC -> CLEAN` | 0.0001 | 9 | `contains_teencode`, `contains_emoji_or_punct`, `obfuscated_or_repeated_chars` | `<NAME> vãi cả đc lên fan cứng. 🤣🤣🤣` | Từ cảm thán `"vãi cả"` đi cùng `"fan cứng"` và emoji cười bị mô hình xem là bình luận vui. |
| 3 | `d314176ee848` | `vihsd` | `TOXIC -> CLEAN` | 0.0001 | 16 | `context_or_lexical_nuance` | `Dương vật của team :)) à em nhầm linh vật ớ lại nhầm linh hồn :))` | Chơi chữ thô tục giả vờ nói nhầm (`"linh vật"`) kèm mặt cười `:))`. |
| 4 | `c7bb23448a7b` | `vihsd` | `TOXIC -> CLEAN` | 0.4797 | 9 | `obfuscated_or_repeated_chars` | `<NAME> mấy anh trên núi tởm thậttt` | Mẫu ranh giới sát ngưỡng `0.50` (`P(TOXIC)=0.4797`), kì thị vùng miền kèm kéo dài ký tự `"thậttt"`. |
| 5 | `ad47a4cdb074` | `vihsd` | `TOXIC -> CLEAN` | 0.4791 | 8 | `context_or_lexical_nuance` | `Cuộc sống thú vị hơn khi bạn xamlol` | Từ lóng viết liền không dấu `"xamlol"` đứng cuối câu mở đầu tích cực (`P(TOXIC)=0.4791`). |
| 6 | `0c28763f7794` | `victsd` | `TOXIC -> CLEAN` | 0.0001 | 12 | `context_or_lexical_nuance` | `Kế hoạch lãi 10tr đồng nhưng khi thực hiện thành 140 tỷ` | Mỉa mai thời sự/kinh tế không chứa từ chửi trực diện (nhãn gốc `ViCTSD` gán `TOXIC`). |
| 7 | `0f3eeb826f45` | `victsd` | `TOXIC -> CLEAN` | 0.0002 | 42 | `contains_negation`, `long_text_ge_35_words` | `Hoá ra từ "bigcityboy" từ đây mà ra. Tôi vào xem thấy MV đẹp, đầu tư. Nhưng ca từ...` | Câu dài 42 từ mở đầu khen ngợi, phần chê trách nằm ở mệnh đề cuối sau từ chuyển ý `"Nhưng"`. |
| 8 | `061f3ad0214b` | `victsd` | `TOXIC -> CLEAN` | 0.0003 | 26 | `contains_negation` | `Sao mình thấy bây giờ <NAME> hơn lúc <NAME> hậu 10 năm trước nhỉ? Không biết có n...` | Bình luận ngoại hình mang tính chất nghi vấn/so sánh gián tiếp trên `ViCTSD`. |
| 9 | `5ed0440002c3` | `victsd` | `TOXIC -> CLEAN` | 0.4737 | 125 | `long_text_ge_35_words`, `contains_emoji_or_punct`, `obfuscated_or_repeated_chars` | `Đây mới là lớp 1, mà <NAME> dụng như kiểu cho sinh viên cao đẳng, đại học!!!"Chún...` | Bình luận rất dài (`125` từ, bị cắt tại `max_word_length=128`), điểm xác suất sát ngưỡng (`0.4737`). |
| 10 | `6fc2c0e5f2fc` | `victsd` | `TOXIC -> CLEAN` | 0.4507 | 3 | `short_text_le_5_words` | `Hôi pen à?` | Câu lóng bóng đá cực ngắn (`3` từ), sát ngưỡng quyết định (`0.4507`). |
| 11 | `8874d016a7a7` | `victsd` | `TOXIC -> CLEAN` | 0.0071 | 21 | `contains_negation` | `Thiệt không thể hiểu nổi, dịch đang bùng phát , mà mọi người đều không đeo khẩu t...` | Phê phán hành vi xã hội thiếu ý thức, không dùng từ tục nhưng được gán nhãn tiêu cực trong `ViCTSD`. |
| 12 | `a001d3c62e6e` | `victsd` | `TOXIC -> CLEAN` | 0.0644 | 4 | `short_text_le_5_words` | `Tóm lại là nhạt.` | Nhận xét chê ngắn gọn (`4` từ) không có từ khóa độc hại mạnh. |
| 13 | `25f94f9a9aa3` | `victsd` | `TOXIC -> CLEAN` | 0.0686 | 67 | `contains_teencode`, `contains_negation`, `long_text_ge_35_words` | `Thật ko thể tin được: nhỏ con hơn <USER> vậy mà thi đấu 731 trận ghi 634 bàn, 7 lần...` | Bình luận tranh luận bóng đá dài `67` từ, phần công kích nằm ở cuối đoạn văn. |
| 14 | `f3364f7c23dc` | `victsd` | `TOXIC -> CLEAN` | 0.0509 | 29 | `context_or_lexical_nuance` | `Nghe các bạn online phán mất mệt..hông ai sát đánh bắt sát sinh lấy gì cho mấy bạ...` | Phản biện mỉa mai cộng đồng mạng bằng phương ngữ (`"hông ai"`). |
| 15 | `6f3de4888f0d` | `vihsd` | `TOXIC -> CLEAN` | 0.3451 | 20 | `contains_teencode`, `contains_negation` | `Tui thấy mâ<NAME> chống cộng chă<NAME> nào còn tỉnh táo. Hâm hết ko giốn...` | Unicode tổ hợp (dấu rời) làm phân mảnh token mức từ; tại ngưỡng Platt `0.35` mẫu này tiến sát ranh giới. |
| 16 | `685dad248242` | `vihsd` | `TOXIC -> CLEAN` | 0.0188 | 2 | `short_text_le_5_words`, `obfuscated_or_repeated_chars` | `đắng =)))` | Bình luận `2` từ quá ngắn, thiếu ngữ cảnh để phân biệt cảm thán với mỉa mai. |
| 17 | `428ea70aa1e5` | `vihsd` | `TOXIC -> CLEAN` | 0.0045 | 6 | `context_or_lexical_nuance` | `Trấn thành đẹp trai đụ hariwon` | Mở đầu bằng cụm khen tích cực `"đẹp trai"` làm lu mờ từ tục đơn lẻ đi kèm tên riêng không viết hoa. |
| 18 | `ba4b24b925d3` | `vihsd` | `TOXIC -> CLEAN` | 0.1339 | 15 | `obfuscated_or_repeated_chars` | `CON QUI CAI CHUYEN DI LUA GAT... MAY GAT CHI MINH HIEU ... SAO LON` | Toàn bộ câu viết HOA không dấu (`"SAO LON"`, `"CON QUI CAI"`), từ vựng mức từ rơi vào `<unk>`. |
| 19 | `1debabf8e9a3` | `vihsd` | `TOXIC -> CLEAN` | 0.0608 | 34 | `obfuscated_or_repeated_chars` | `<NAME> ơi hiện có kênh con <USER> chửi cả DÂN TỘC VIỆT NAM NGU VÌ ĂN MẮM THÚI , EM L...` | Bình luận thuật lại/tố cáo lời chửi của người khác (`"chửi cả... NGU"`). |
| 20 | `1baec0dca32f` | `vihsd` | `TOXIC -> CLEAN` | 0.2222 | 24 | `contains_teencode`, `contains_negation` | `Thấy cái đó là xỉu vậy giờ bà cũng có cái <NAME> cuối xuống nhìn có bị xỉu k, nói...` | Công kích ẩn dụ dài 24 từ kèm viết tắt `"k"` mà không dùng từ chửi trực tiếp. |
| 21 | `5f719f6c7e1e` | `vihsd` | `CLEAN -> TOXIC` | 0.9999 | 10 | `contains_teencode`, `obfuscated_or_repeated_chars` | `Đm thua vcl :)))) phim hài à :)))) <NAME> vcl` | **False Positive điển hình do teencode:** người dùng dùng `"Đm"`, `"vcl"` để khen phim hài hước nhưng mô hình bắt mạnh lexical cue. |
| 22 | `4a9160598574` | `vihsd` | `CLEAN -> TOXIC` | 0.9983 | 17 | `context_or_lexical_nuance` | `vay toàn lảnh <NAME> hết cmnr đến lúc hoàn nợ thì đè dân ra đón thuế` | Từ lóng `"cmnr"` + giọng điệu bức xúc chính sách khiến mô hình dự đoán `TOXIC` dù nhãn gốc là `CLEAN`. |
| 23 | `0faac4718bb3` | `vihsd` | `CLEAN -> TOXIC` | 0.9983 | 38 | `contains_teencode`, `contains_negation`, `long_text_ge_35_words` | `Tao đéo pha<NAME> viên nhé, nhưg côg ty tao có khoảg chu<NAME> viên, họ phả...` | Đại từ `"Tao"` + từ phủ định mạnh `"đéo"` dùng trong câu giải thích thông thường (nhãn gốc `CLEAN`). |
| 24 | `32fd35f1c4f1` | `vihsd` | `CLEAN -> TOXIC` | 0.5005 | 5 | `short_text_le_5_words` | `<NAME> nghe ói :)` | Mẫu nằm đúng trên ngưỡng quyết định (`P(TOXIC)=0.5005`), chứa từ `"ói"` kèm mặt cười `:)`. |
| 25 | `9704bb96c95b` | `vihsd` | `CLEAN -> TOXIC` | 0.5005 | 18 | `contains_teencode` | `<NAME> :3 méo tin dc page này có cả cmt của TDQ` | Mẫu sát ngưỡng `0.5005`: từ lóng `"méo tin dc"` biểu thị sự ngạc nhiên tích cực. |
| 26 | `175c9c814491` | `victsd` | `CLEAN -> TOXIC` | 0.9991 | 54 | `contains_negation`, `long_text_ge_35_words` | `Chó không chỉ là vật nuôi mà còn là người bạn thân thiết của gia đình. Tôi đang n...` | **False Positive do từ khóa `"Chó"`:** bài viết bảo vệ động vật nhưng từ `"Chó"` có tương quan cao với nhãn `TOXIC` trong tập train. |
| 27 | `cd34a51831a5` | `victsd` | `CLEAN -> TOXIC` | 0.9961 | 38 | `contains_negation`, `long_text_ge_35_words`, `contains_emoji_or_punct`, `obfuscated_or_repeated_chars` | `ham cho vay quá ,mà thủ tục lỏng lẻo dễ qua mặt nên mất tiền vay không <NAME> là...` | Nhận xét phê bình thủ tục cho vay chứa nhiều tính từ tiêu cực (`"lỏng lẻo"`, `"mất tiền"`). |
| 28 | `8090d35728d8` | `victsd` | `CLEAN -> TOXIC` | 0.9946 | 7 | `context_or_lexical_nuance` | `Ác nhân ắt sẽ có ác báo!` | Câu thành ngữ/nhân quả chứa từ `"Ác nhân"`, `"ác báo"` bị kích hoạt nhầm thành lời nguyền rủa. |
| 29 | `2dbef82259a6` | `victsd` | `CLEAN -> TOXIC` | 0.5050 | 14 | `obfuscated_or_repeated_chars` | `Khoảnh khắc ấy, mặt con cá đực to thẫn thờ thấy tội... :') :P` | Cụm `"mặt con cá đực"` bị nhiễu bởi mẫu n-gram `"mặt con..."`, xác suất sát ngưỡng (`0.5050`). |
| 30 | `7e6d2e6243ad` | `victsd` | `CLEAN -> TOXIC` | 0.5124 | 27 | `context_or_lexical_nuance` | `Ca ngộ <NAME> tiên này đã xảy ra từ lâu rồi mà sao giờ vận còn người ăn nhỉ. TV,...` | Bình luận tin tức ngộ độc thực phẩm, xác suất ngay sát ngưỡng (`0.5124`). |
| 31 | `a179bc25dc14` | `vihsd` | `CLEAN -> TOXIC` | 0.8145 | 11 | `contains_teencode` | `Phim đc vl mà thấy chê nhiều nhở. Thót tim vl.` | Dùng từ cảm thán `"đc vl"`, `"Thót tim vl"` để khen bộ phim hay (lexical false positive). |
| 32 | `e4432885f8bd` | `vihsd` | `CLEAN -> TOXIC` | 0.9186 | 17 | `contains_teencode`, `obfuscated_or_repeated_chars` | `Cô này chủ vlxd giàu có và chuyên từ thiện luôn nhaaa bạn,bớt nói cho bớt nghiệppp` | Viết tắt `"vlxd"` (vật liệu xây dựng) + từ `"nghiệppp"` bị mô hình nhầm với từ lóng xúc phạm. |

---

## 6. Biểu đồ Trực quan và Notebook Báo cáo

5 biểu đồ phân tích độ phân giải cao (`160 DPI`) đã được sinh tự động từ `src/evaluation/cnn_report.py` và lưu tại `artifacts/TV4-cnn-analysis/`:

1. `artifacts/TV4-cnn-analysis/ablation_macro_f1.png`: Xếp hạng Validation Macro-F1 (`mean ± std` 3 seeds) và F1-TOXIC của toàn bộ 18 biến thể.
2. `artifacts/TV4-cnn-analysis/learning_curves.png`: Đường cong Train/Val Loss và Macro-F1 theo epoch của 5 kiến trúc đại diện (`HYBRID-RNN-01`, `CNN-01-WD`, `HYBRID-BILSTM-01`, `HYBRID-01`, `CHAR-02`).
3. `artifacts/TV4-cnn-analysis/confusion_matrices.png`: Ma trận nhầm lẫn 2×2 (số lượng & phần trăm theo hàng) trên `val.parquet` (`n = 6,048`) của 4 kiến trúc dẫn đầu.
4. `artifacts/TV4-cnn-analysis/calibration_reliability.png`: Biểu đồ độ tin cậy (Reliability Diagram) so sánh `Raw Softmax` vs `Temperature Scaling` vs `Platt Scaling` và đường cong quét ngưỡng (`Threshold Sweep`).
5. `artifacts/TV4-cnn-analysis/source_comparison.png`: So sánh khả năng tổng quát hóa theo nguồn `ViHSD` (`n = 4,559`) và `ViCTSD` (`n = 1,489`).

Notebook thực nghiệm **`notebooks/EXP-04-cnn-analysis.ipynb`** đã được tạo và thực thi toàn bộ các cell từ trên xuống dưới (kiểm chứng tương đương `Restart & Run All` qua 2 lượt chạy độc lập), nhúng trực tiếp bảng số liệu và 5 biểu đồ trên mà không chứa bất kỳ logic huấn luyện trùng lặp hay bình luận thô chưa che PII nào.

---

## 7. Phần 5 (`EXP-06`) — Kiểm định Độ Đa dạng Lỗi (Error Diversity) & Conditional Ensemble

Thực thi qua module [src/evaluation/ensemble_report.py](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/src/evaluation/ensemble_report.py) và kiểm thử bằng [tests/test_ensemble_report.py](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/tests/test_ensemble_report.py). Toàn bộ kết quả và biểu đồ được lưu tại `artifacts/TV4-ensemble-analysis/` và `models/ensemble/TV4-EXP06/`.

### 7.1. Kiểm kê Nguồn Dự đoán Validation (`n = 6,048`, khớp 100% `sample_id` và SHA-256)
- **TV4 (`EXP-04 / N7`):** 7 kiến trúc đại diện × 3 seeds (`HYBRID-RNN-01`, `HYBRID-RNN-WD`, `CNN-01-WD`, `CNN-01`, `CNN-04b`, `HYBRID-BILSTM-01`, `CHAR-02`).
- **TV2 (`EXP-02 / N6`):**
  - `EXP-N6-C-calibrated` (`models/baselines/N6-run-faffb8025fbc41fb8d9b463468bdd5f3/EXP-N6-C-calibrated/seed_42..44`): TF-IDF word `(1,2)` + char_wb `(3,5)` + SelectKBest `75k` + LinearSVC + 3-fold Train-CV Sigmoid Calibration (`is_calibrated: true`, ECE `0.0111`). Val Macro-F1 ở ngưỡng mặc định `@0.50` đạt `0.7836` (`F1-TOXIC = 0.6274`); ở ngưỡng in-sample `@0.34` đạt `0.7954` (`F1-TOXIC = 0.6575`).
  - `EXP-N6-C-uncal` (`models/baselines/N6-run-931168a6f09241c6a8c943fd99f26c7f/EXP-N6-C/seed_42..44`): chỉ có nhãn cứng từ `decision_function` (`Macro-F1 = 0.7892`, `F1-TOXIC = 0.6502`, `probabilities = null`). Theo đúng `AGENTS.md` §6, `blend_validated_probabilities()` từ chối mọi mô hình không có xác suất hợp lệ.
- **Trạng thái TV1 & TV3:** `models/baselines/EXP-01-A/B/C` của TV1 thuộc lịch sử G0 3 lớp (`vihsd.jsonl`, `n = 2,672`); `N5` FastText của TV3 (`docs/N5_TOXIC_TUNING_REPORT.md`) đạt Val Macro-F1 `0.6222` (tuned `0.6615`), thấp hơn `> 11%` so với `N6-C` và `TV4` nên không đưa vào tổ hợp.

### 7.2. Ma trận Độ Đa dạng Lỗi Đôi một (Pairwise Error Diversity trên `val.parquet`, Seed 42)

| Cặp Mô hình ($A$ vs $B$) | Nhóm | Bất đồng dự đoán (`disagreement_rate`) | Chỉ $A$ đúng (`only_a_correct`) | Chỉ $B$ đúng (`only_b_correct`) | Tổng mẫu bổ trợ (`A+B`) | Cùng sai (`double_fault_rate`) | Cohen's $\kappa$ | Tương quan $P(\text{TOXIC})$ ($r$) | Oracle Macro-F1 | Oracle F1-TOXIC |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `HYBRID-RNN-01` vs `CNN-01-WD` | TV4 + TV4 | **8.20%** (496) | 237 | 259 | **496** | 7.14% (432) | 0.6530 | 0.7830 | **0.8543** | **0.7503** |
| `HYBRID-RNN-01` vs `CNN-04b` | TV4 + TV4 | **8.75%** (529) | 267 | 262 | **529** | 7.09% (429) | 0.6455 | 0.7585 | **0.8584** | **0.7583** |
| `HYBRID-RNN-01` vs `CHAR-02` | TV4 + TV4 | **8.22%** (497) | 277 | 220 | **497** | 7.79% (471) | 0.6672 | 0.7994 | **0.8448** | 0.7352 |
| `CNN-01-WD` vs `CHAR-02` | TV4 + TV4 | **9.51%** (575) | 327 | 248 | **575** | **6.96%** (421) | 0.5871 | 0.7262 | **0.8563** | **0.7531** |
| `CNN-01-WD` vs `HYBRID-BILSTM-01` | TV4 + TV4 | **8.70%** (526) | 299 | 227 | **526** | 7.31% (442) | 0.6362 | 0.7615 | **0.8516** | 0.7460 |
| `HYBRID-RNN-01` vs `EXP-N6-C-calibrated` | TV4 + TV2 | **7.36%** (445) | 190 | 255 | **445** | 7.21% (436) | 0.6823 | 0.8305 | **0.8535** | **0.7491** |
| `HYBRID-RNN-WD` vs `EXP-N6-C-calibrated` | TV4 + TV2 | **7.64%** (462) | 192 | 270 | **462** | 7.18% (434) | 0.6749 | 0.8301 | **0.8545** | **0.7509** |
| `CNN-01-WD` vs `EXP-N6-C-calibrated` | TV4 + TV2 | **6.13%** (371) | 164 | 207 | **371** | 7.64% (462) | 0.7132 | 0.8575 | **0.8403** | 0.7250 |
| `CHAR-02` vs `EXP-N6-C-calibrated` | TV4 + TV2 | **8.20%** (496) | 187 | 309 | **496** | 7.26% (439) | 0.6362 | 0.8047 | **0.8500** | 0.7422 |
| `HYBRID-RNN-01` vs `EXP-N6-C-uncal` | TV4 + TV2 | **9.62%** (582) | 306 | 276 | **582** | **6.86%** (415) | 0.6512 | — | **0.8693** | **0.7791** |

$\Rightarrow$ **Bằng chứng Error Diversity rất mạnh:** Giữa các họ kiến trúc khác nhau (`BiGRU+Char-CNN`, `Word TextCNN`, `Pure Char-CNN`, và `TF-IDF LinearSVC`), tỷ lệ bất đồng dự đoán đạt **`6.13%–9.62%` (`371–582` mẫu trên `6,048` mẫu Val)**, hệ số tương quan xác suất vừa phải (`r = 0.726–0.831`), và **Oracle Macro-F1 vọt lên `0.8535–0.8693`**, chứng minh các mô hình mắc lỗi ở các vùng dữ liệu khác nhau và đủ điều kiện tiên quyết để ghép Ensemble.

### 7.3. Bảng So sánh 10 Cấu hình Ensemble theo 3 Protocols trên 3 Seeds (`[42, 43, 44]`)

| Cấu hình / Mô hình | Phạm vi | Protocol đánh giá | Val Macro-F1 (mean ± std) | Val F1-TOXIC (mean ± std) | PR-AUC | ECE | ViCTSD Macro-F1 (F1-TOXIC) | ViHSD Macro-F1 (F1-TOXIC) | $\Delta$ vs Best TV4 (`0.7733`) | $\Delta$ vs `N6-C-cal @0.50` (`0.7836`) | $\Delta$ vs `N6-C uncal` (`0.7892`) | Tổng Size (MB) | Tổng Latency (ms/mẫu) | Đạt cổng ADR `+1.0%`? |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|
| `[Single] HYBRID-RNN-01` | TV4 | Default `@0.50` | `0.7733 ± 0.0072` | `0.6149 ± 0.0130` | 0.6673 | 0.0635 | 0.6542 (0.3766) | 0.7961 (0.6606) | `+0.0000` | `-0.0103` | `-0.0159` | 22.69 | 0.2232 | Baseline TV4 |
| `[Single] CNN-01-WD` | TV4 | Default `@0.50` | `0.7705 ± 0.0027` | `0.6058 ± 0.0056` | 0.6751 | 0.0494 | 0.6211 (0.3044) | 0.7951 (0.6560) | `-0.0028` | `-0.0131` | `-0.0187` | 15.12 | 0.0502 | Baseline TV4 |
| `[Single] EXP-N6-C-calibrated` | TV2 | Train-CV `@0.50` | `0.7836 ± 0.0000` | `0.6274 ± 0.0000` | 0.7243 | 0.0111 | 0.6527 (0.3621) | 0.8055 (0.6722) | `+0.0103` | `+0.0000` | `-0.0056` | 10.40 | 0.0042 | Baseline TV2 |
| `[Single] EXP-N6-C-uncal` | TV2 | Margin `@0.00` | `0.7892 ± 0.0000` | `0.6502 ± 0.0000` | — | — | 0.6879 (0.4471) | 0.8091 (0.6903) | `+0.0159` | `+0.0056` | `+0.0000` | 7.35 | 0.0005 | Baseline TV2 |
| **`ENS-TV4-INTRA-3`** | **Nội bộ TV4** | **Fixed `@0.50` (0 tuning)** | **`0.7834 ± 0.0058`** | `0.6299 ± 0.0098` | 0.7028 | 0.0378 | 0.6515 (0.3654) | 0.8069 (0.6771) | **`+0.0102`** | `-0.0002` | `-0.0057` | 60.51 | 0.4344 | **ĐẠT (`+1.02%`)** |
| **`ENS-TV4-INTRA-4`** | **Nội bộ TV4** | **Fixed `@0.50` (0 tuning)** | **`0.7855 ± 0.0053`** | `0.6313 ± 0.0105` | 0.7073 | 0.0350 | 0.6413 (0.3407) | 0.8092 (0.6794) | **`+0.0123`** | `+0.0019` | `-0.0036` | 77.33 | 0.4890 | **ĐẠT (`+1.23%`)** |
| **`ENS-TV4-DIVERSE-6`** | **Nội bộ TV4** | **Fixed `@0.50` (0 tuning)** | **`0.7859 ± 0.0078`** | `0.6329 ± 0.0146` | 0.7098 | 0.0296 | 0.6520 (0.3623) | 0.8082 (0.6785) | **`+0.0126`** | `+0.0022` | `-0.0033` | 104.76 | 0.8150 | **ĐẠT (`+1.26%`)** |
| **`ENS-TV4-OOF-PLATT`** | **Nội bộ TV4** | **5-Fold OOF Platt + OOF Thr** | **`0.7900 ± 0.0051`** | **`0.6447 ± 0.0097`** | **0.7104** | **0.0271** | 0.6504 (0.3676) | **0.8153** (**0.6946**) | **`+0.0167`** | **`+0.0063`** | **`+0.0008`** | 104.76 | 0.8150 | **ĐẠT (`+1.67%`)** |
| `ENS-CROSS-50-50` | TV2 + TV4 | Fixed `@0.50` (0 tuning) | `0.7884 ± 0.0048` | `0.6378 ± 0.0082` | 0.7199 | 0.0190 | 0.6593 (0.3780) | 0.8109 (0.6833) | `+0.0152` | `+0.0048` | `-0.0007` | 33.10 | 0.2274 | `+0.48%` vs N6-cal |
| **`ENS-CROSS-WEIGHTED-050`** | **TV2 + TV4** | **Fixed `@0.50` (0 tuning)** | **`0.7933 ± 0.0035`** | `0.6442 ± 0.0066` | 0.7301 | 0.0130 | 0.6440 (0.3449) | 0.8178 (0.6933) | **`+0.0200`** | **`+0.0097`** | `+0.0042` | 70.91 | 0.4386 | **`~+1.0%` vs N6-cal** |
| **`ENS-CROSS-DIVERSE-050`** | **TV2 + TV4** | **Fixed `@0.50` (0 tuning)** | **`0.7928 ± 0.0022`** | `0.6432 ± 0.0045` | 0.7313 | **0.0113** | 0.6421 (0.3407) | 0.8172 (0.6923) | **`+0.0196`** | `+0.0092` | `+0.0037` | 91.71 | 0.5979 | `+0.92%` vs N6-cal |
| **`ENS-CROSS-OOF-PLATT`** | **TV2 + TV4** | **5-Fold OOF Platt + OOF Thr** | **`0.7989 ± 0.0003`** | **`0.6615 ± 0.0017`** | **0.7327** | 0.0223 | **0.6719** (**0.4078**) | **0.8217** (**0.7069**) | **`+0.0257`** | **`+0.0153`** | **`+0.0098`** | 91.71 | 0.5979 | **ĐẠT (`+1.53%` / `~+1.0%`)** |
| **`ENS-CROSS-OOF-STACKING-LR`** | **TV2 + TV4** | **Chẩn đoán 5-Fold Val-OOF LR** | **`0.8002 ± 0.0019`** | **`0.6637 ± 0.0057`** | **0.7341** | **0.0072** | **0.6795** (**0.4209**) | 0.8213 (0.7066) | **`+0.0269`** | **`+0.0165`** | **`+0.0110`** | 52.20 | 0.3824 | **ĐẠT (`+1.10%` vs N6-C)** |
| `ENS-CROSS-IN-SAMPLE-DIAG` | TV2 + TV4 | Chẩn đoán In-Sample Thr | `0.8021 ± 0.0012` | `0.6654 ± 0.0020` | 0.7313 | 0.0113 | 0.6756 (0.4130) | **0.8247** (**0.7104**) | `+0.0288` | `+0.0184` | `+0.0129` | 91.71 | 0.5979 | Chẩn đoán (`+0.67%` vs `0.7954`) |

### 7.4. Quyết định Kỹ thuật theo ADR (`docs/todos/TV4.md` P2 & `AGENTS.md` §6)

1. **Trong phạm vi nội bộ TV4 (`EXP-04 / N7`):**
   - Ngay ở ngưỡng mặc định `0.50` (không chạm bất kỳ tham số hiệu chuẩn hay ngưỡng nào trên validation), cả 3 tổ hợp **`ENS-TV4-INTRA-3` (`0.7834`, `+1.02%`)**, **`ENS-TV4-INTRA-4` (`0.7855`, `+1.23%`)** và **`ENS-TV4-DIVERSE-6` (`0.7859`, `+1.26%`, best seed 42 đạt `0.7946`)** đều **vượt ngưỡng ADR `>= +1.0%` Validation Macro-F1** so với mô hình đơn tốt nhất của TV4 (`HYBRID-RNN-01`, `0.7733`), đồng thời hạ ECE từ `0.0635` xuống `0.0296` và tăng PR-AUC từ `0.6673` lên `0.7098`.
   - Khi áp dụng **5-fold Out-of-Fold Platt Calibration (`ENS-TV4-OOF-PLATT`)**, tổ hợp thuần neural của TV4 đạt **`0.7900 ± 0.0051`** (`+1.67%` so với best single TV4 và vượt cả `EXP-N6-C` `0.7892` của TV2).
2. **Trong phạm vi Liên thành viên (`TV2 EXP-N6-C-calibrated + TV4 Neural Models`):**
   - Kết hợp xác suất đã hiệu chuẩn Train-CV của `EXP-N6-C-calibrated` với các nhánh neural của TV4 theo **5-fold Out-of-Fold Platt Voting (`ENS-CROSS-OOF-PLATT`)** đạt **`0.7989 ± 0.0003`** (phương sai cực thấp giữa 3 seeds: `0.7987–0.7993`), `F1-TOXIC = 0.6615 ± 0.0017`, `PR-AUC = 0.7327`:
     - Tăng **`+0.0257` (`+2.57%`)** so với mô hình đơn tốt nhất của TV4 (`0.7733`).
     - Tăng **`+0.0153` (`+1.53% >= 1.0%`)** so với `EXP-N6-C-calibrated @0.50` (`0.7836`).
     - Tăng **`+0.0098` (`~+1.0%`)** so với `EXP-N6-C` uncalibrated (`0.7892`), và đạt **`0.8002 ± 0.0019` (`+1.10%` so với `0.7892`)** với chẩn đoán Val-OOF Logistic Meta-Learner (`ENS-CROSS-OOF-STACKING-LR`, ECE `0.0072`).
3. **Quyết định về Stacking OOF trên Train (`AGENTS.md` §6):**
   - Để huấn luyện Stacking meta-learner chính thức không rò rỉ dữ liệu, toàn bộ các mô hình thành phần phải sinh **Out-of-Fold predictions trên `train.parquet`**. Với 6 kiến trúc neural × 3 seeds × 5 folds, việc này đòi hỏi `90` lượt huấn luyện GPU mới (`~2.5` giờ GPU), trong khi chẩn đoán 5-fold OOF cho thấy `ENS-CROSS-OOF-STACKING-LR` (`0.8002`) chỉ chênh `+0.0013` so với `ENS-CROSS-OOF-PLATT` (`0.7989`).
   - Do đó, tuân thủ `AGENTS.md` §6 (*"Cân nhắc chi phí OOF... nếu không khả thi, ưu tiên Voting hợp lệ"*), TV4 **chọn Calibrated Weighted Probability Voting (`ENS-TV4-DIVERSE-6` / `ENS-CROSS-OOF-PLATT`) và ghi rõ quyết định skip Train-OOF Stacking trong production**.

---

## 8. Đánh giá Cuối Một Đợt trên Tập Locked Test (`N10` / `EXP-08` — Phần 6)

### 8.1. Giao thức Đóng băng Trước Khi Đọc Test & Minh bạch Hóa Lịch sử Tập Test (`test_blind: false`)

1. **Manifest Đóng băng (`artifacts/TV4-locked-test/tv4_n10_freeze_manifest.json`):**
   - Được khởi tạo và ghi xuống đĩa với `status = "FROZEN_BEFORE_TEST_READ"` **trước khi** `src/evaluation/locked_test_report.py` mở `data/processed/binary_n2_v1/test.parquet`.
   - Khóa cứng 7 kiến trúc mô hình đơn của TV4 × 3 seeds (`[42, 43, 44]` = 21 checkpoints), đường dẫn `val_predictions.json`, trọng số bộ hiệu chuẩn Platt đã fit trên validation (`coef`, `intercept`, `val_platt_opt_threshold`), mô hình tham chiếu `EXP-N6-C-calibrated` của TV2 (fit lại từ `train.parquet` đạt `val_repro_max_prob_diff = 0.0` tuyệt đối trên `6,048 / 6,048` mẫu validation), và 5 cấu hình ensemble đã chọn ở Phần 5.
   - **Không có bất kỳ bước chọn mô hình, thay đổi kiến trúc, tinh chỉnh trọng số hay tối ưu ngưỡng quyết định nào được phép thực hiện sau khi đọc `test.parquet`.**
2. **Thông tin tập `test.parquet` và công bố `test_blind: false`:**
   - Đường dẫn: `data/processed/binary_n2_v1/test.parquet` (`n = 6,048` mẫu: `5,084 CLEAN` (`84.06%`), `964 TOXIC` (`15.94%`); theo nguồn: `4,558 vihsd` (`75.36%`), `1,490 victsd` (`24.64%`)).
   - SHA-256: `d200be4a7ff2f2e725ba8e6c2c0ab835afd73612b9a5b2f14b528cb5c577be4e`.
   - **Công bố minh bạch (`test_blind: false`):** Tuân thủ `docs/decisions.md` (D7) và `AGENTS.md` §5, báo cáo ghi nhận rõ `test_blind: false` vì tập test của repository từng bị chạm ở một số script lịch sử trước mốc tái cấu trúc nhị phân `SPEC v5.0.0`. Tuy nhiên, **toàn bộ 54 runs của TV4 (Phần 2–3), phân tích lỗi & hiệu chuẩn (Phần 4), và thiết kế ensemble (Phần 5) đều hoàn toàn mù với `test.parquet`** và chỉ truy cập `test.parquet` đúng một lần duy nhất trong Phần 6 (`N10`).

---

### 8.2. Kết quả Mô hình Đơn Đóng băng trên Locked Test (`test.parquet`, 3 seeds `42, 43, 44`)

| Mô hình đơn (Frozen) | Kiến trúc & Tiền xử lý | Val Macro-F1 (`@0.50`) | Test Macro-F1 (`@0.50`, Mean ± Std) | Test F1-TOXIC (`@0.50`, Mean ± Std) | Test PR-AUC (`@0.50`) | Test Raw ECE (`@0.50`) | Test Macro-F1 (Frozen Val-Platt) | Test Platt ECE | Test ViCTSD / ViHSD Macro-F1 (`@0.50`) | $\Delta(\text{Test} - \text{Val})$ |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CNN-01`** | Word TextCNN `[3]` (`compression`) | `0.7702` | **`0.7867 ± 0.0049`** | `0.6350 ± 0.0105` | **`0.6961`** | `0.0452` | **`0.7865 ± 0.0050`** | `0.0146` | `0.6519` / **`0.8105`** | **`+0.0166`** |
| **`HYBRID-RNN-01`** | BiGRU + Char-CNN `[3,4,5,7]` (`none`) | **`0.7733`** | **`0.7855 ± 0.0082`** | **`0.6352 ± 0.0134`** | `0.6889` | `0.0535` | **`0.7854 ± 0.0067`** | `0.0112` | **`0.6696`** / `0.8070` | `+0.0122` |
| **`CNN-01-WD`** | Word TextCNN `[3]` + `wd=1e-4` (`compression`) | `0.7705` | `0.7839 ± 0.0071` | `0.6290 ± 0.0133` | `0.6945` | `0.0391` | `0.7851 ± 0.0087` | `0.0139` | `0.6486` / `0.8077` | `+0.0135` |
| **`HYBRID-RNN-WD`** | BiGRU + Char-CNN + `wd=1e-4` (`none`) | `0.7731` | `0.7808 ± 0.0026` | `0.6278 ± 0.0040` | `0.6932` | `0.0584` | `0.7805 ± 0.0031` | **`0.0100`** | `0.6536` / `0.8052` | `+0.0077` |
| **`HYBRID-BILSTM-01`** | BiLSTM + Char-CNN `[3,4,5,7]` (`none`) | `0.7665` | `0.7772 ± 0.0021` | `0.6235 ± 0.0033` | `0.6805` | `0.0740` | `0.7775 ± 0.0022` | `0.0115` | `0.6461` / `0.8032` | `+0.0107` |
| **`CNN-04b`** | Word TextCNN `[3,4,5]` + `wd=1e-4` (`compression`) | `0.7688` | `0.7754 ± 0.0071` | `0.6140 ± 0.0155` | `0.6770` | `0.0699` | `0.7757 ± 0.0077` | `0.0131` | `0.6357` / `0.7998` | `+0.0067` |
| **`CHAR-02`** | Pure Char-CNN `[3,4,5,7]` `filters=256` (`none`) | `0.7550` | `0.7579 ± 0.0092` | `0.5861 ± 0.0187` | `0.6497` | `0.0382` | `0.7590 ± 0.0081` | `0.0125` | `0.6234` / `0.7831` | `+0.0029` |
| **`EXP-N6-C-calibrated` (`@0.50`)** | TF-IDF Word+Char LinearSVC + Calibrated (TV2 Ref) | `0.7836` | `0.7989` | `0.6535` | `0.7381` | `0.0103` | `0.7989` | `0.0103` | `0.6776` / `0.8195` | `+0.0153` |
| **`EXP-N6-C-calibrated` (`@0.34`)** | TF-IDF Word+Char LinearSVC (`thr=0.34` đóng băng từ Val) | `0.7954` | **`0.8048`** | **`0.6728`** | `0.7381` | `0.0103` | **`0.8048`** | `0.0103` | **`0.7101`** / **`0.8236`** | `+0.0094` |

---

### 8.3. Kết quả Các Cấu hình Ensemble Đóng băng trên Locked Test (`test.parquet`)

| Cấu hình Ensemble (Frozen) | Phạm vi & Giao thức Đóng băng từ Validation | Val Macro-F1 | Test Macro-F1 (Mean ± Std) | Test F1-TOXIC (Mean ± Std) | Test Weighted-F1 | Test Balanced Acc | Test PR-AUC | Test ECE | Test ViCTSD Macro-F1 (`F1-TOXIC`) | Test ViHSD Macro-F1 (`F1-TOXIC`) | $\Delta(\text{Test} - \text{Val})$ |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ENS-TV4-INTRA-4`** | Nội bộ TV4 (4 họ: `0.35 HYBRID-RNN-01 + 0.30 CNN-01-WD + 0.20 CNN-04b + 0.15 CHAR-02`, raw `@0.50`) | `0.7855` | `0.7991 ± 0.0028` | `0.6548 ± 0.0059` | `0.8974` | `0.7715` | `0.7326` | `0.0265` | `0.6591` (`0.3747`) | `0.8228` (`0.7023`) | `+0.0136` |
| **`ENS-TV4-DIVERSE-6`** | Nội bộ TV4 (6 mô hình neural đa dạng, raw `@0.50`) | `0.7859` | `0.7981 ± 0.0024` | `0.6538 ± 0.0048` | `0.8964` | `0.7732` | `0.7361` | `0.0228` | `0.6635` (`0.3842`) | `0.8211` (`0.7000`) | `+0.0122` |
| **`ENS-TV4-FROZEN-PLATT`** | Nội bộ TV4 (6 mô hình, Frozen Val-Platt + Frozen Val Thr `[0.45, 0.37, 0.42]`) | `0.7900` | **`0.8029 ± 0.0025`** | **`0.6660 ± 0.0065`** | `0.8962` | **`0.7926`** | **`0.7373`** | `0.0342` | **`0.6727` (`0.4074`)** | **`0.8265` (`0.7126`)** | `+0.0129` |
| **`ENS-CROSS-DIVERSE-050`** | Liên thành viên TV2 + TV4 (`0.48 N6 + 0.52 TV4`, raw `@0.50`) | `0.7928` | `0.8029 ± 0.0009` | `0.6606 ± 0.0013` | `0.8998` | `0.7721` | `0.7571` | **`0.0073`** | `0.6699` (`0.3934`) | `0.8252` (`0.7056`) | `+0.0101` |
| **`ENS-CROSS-FROZEN-PLATT`** | Liên thành viên TV2 + TV4 (`0.48 N6 + 0.52 TV4`, Frozen Val-Platt + Frozen Val Thr `[0.38, 0.36, 0.38]`) | **`0.7989`** | **`0.8128 ± 0.0019`** | **`0.6834 ± 0.0027`** | **`0.9009`** | **`0.8050`** | **`0.7572`** | `0.0282` | **`0.6930` (`0.4444`)** | **`0.8342` (`0.7261`)** | `+0.0139` |

---

### 8.4. Phân tích Khả năng Tổng quát hóa Validation $\to$ Locked Test

1. **Không xảy ra Overfitting lên Validation ($\Delta(\text{Test} - \text{Val}) > 0$ trên toàn bộ 13 dòng ứng viên):**
   - Cả 7 mô hình đơn của TV4, mô hình tham chiếu `EXP-N6-C-calibrated` của TV2 và 5 cấu hình ensemble đều có Test Macro-F1 cao hơn Validation Macro-F1 từ **`+0.0029` đến `+0.0166`**, giữ nguyên thứ tự xếp hạng tương đối giữa các nhóm mô hình (`Pure Char-CNN < Word TextCNN / Hybrid BiGRU+Char-CNN < Intra-TV4 Ensemble < Cross-Member Ensemble`).
   - Bộ hiệu chuẩn Platt đóng băng từ validation (`Frozen Val-Platt`) tiếp tục giảm mạnh sai số hiệu chuẩn ECE trên tập Locked Test của tất cả mô hình đơn neural từ `0.0382–0.0740` xuống chỉ còn **`0.0100–0.0146`**.
2. **Hiệu quả của Ensemble Nội bộ TV4 trên Locked Test:**
   - Ngay ở ngưỡng chuẩn `0.50`, `ENS-TV4-INTRA-4` đạt Test Macro-F1 **`0.7991 ± 0.0028`** (`F1-TOXIC = 0.6548`), vượt mô hình đơn tốt nhất của TV4 trên test (`CNN-01` `0.7867`, `HYBRID-RNN-01` `0.7855`) tới **`+1.24%` – `+1.36%` Macro-F1** và ngang bằng với baseline `EXP-N6-C-calibrated @0.50` của TV2 (`0.7989`).
   - Khi áp dụng bộ hiệu chuẩn Platt và ngưỡng đã đóng băng từ validation, `ENS-TV4-FROZEN-PLATT` đạt Test Macro-F1 **`0.8029 ± 0.0025`** (`max = 0.8058` ở `seed_44`, `F1-TOXIC = 0.6660`, `Balanced Acc = 0.7926`).
3. **Hiệu quả của Ensemble Liên thành viên TV2 + TV4 trên Locked Test:**
   - Ở ngưỡng chuẩn `0.50`, `ENS-CROSS-DIVERSE-050` đạt Test Macro-F1 **`0.8029 ± 0.0009`**, **`PR-AUC = 0.7571`** (`+1.90%` PR-AUC so với `EXP-N6-C-calibrated`) và sai số hiệu chuẩn cực thấp **`Test ECE = 0.0073`**.
   - Với bộ hiệu chuẩn Platt và ngưỡng đóng băng từ validation, **`ENS-CROSS-FROZEN-PLATT`** đạt kết quả cao nhất toàn chiến dịch:
     - **Test Macro-F1 = `0.8128 ± 0.0019`** (`min = 0.8113`, `max = 0.8149` ở `seed_44`).
     - **Test F1-TOXIC = `0.6834 ± 0.0027`**, **Test Weighted-F1 = `0.9009`**, **Test Balanced Accuracy = `0.8050`**, **Test PR-AUC = `0.7572`**.
     - Vượt qua cả `EXP-N6-C-calibrated @0.50` (**`+1.39%` Macro-F1**, **`+2.99%` F1-TOXIC**) lẫn `EXP-N6-C-calibrated @0.34` (**`+0.80%` Macro-F1**, **`+1.06%` F1-TOXIC**, và **`+1.06%` Macro-F1 trên miền chính `ViHSD`: `0.8342` vs `0.8236`**).
4. **Biểu đồ & Artifacts Bàn giao của Phần 6 (`N10`):**
   - Manifest đóng băng trước khi đọc test: `artifacts/TV4-locked-test/tv4_n10_freeze_manifest.json`
   - Báo cáo JSON chi tiết toàn bộ metrics, Bootstrap 95% CI, McNemar & confusion matrices trên Locked Test: `artifacts/TV4-locked-test/tv4_n10_locked_test_report.json`
   - Dự đoán từng mẫu (`test_predictions.json` & `predictions_test.csv`) cho các mô hình và ensemble: `models/locked_test/TV4-N10/`
   - Bộ 3 biểu đồ Locked Test:
     - `artifacts/TV4-locked-test/locked_test_val_vs_test.png` (Đối chiếu Validation vs Locked Test Macro-F1 & khoảng cách tổng quát hóa $\Delta$)
     - `artifacts/TV4-locked-test/locked_test_confusion_matrices.png` (Ma trận nhầm lẫn trên `test.parquet` của `HYBRID-RNN-01`, `EXP-N6-C-calibrated @0.34`, `ENS-TV4-FROZEN-PLATT`, và `ENS-CROSS-FROZEN-PLATT`)
     - `artifacts/TV4-locked-test/locked_test_source_breakdown.png` (Phân rã Macro-F1, F1-TOXIC và PR-AUC theo nguồn `ViCTSD` vs `ViHSD` trên `test.parquet`)

---

### 8.5. Khoảng Tin cậy Bootstrap 95% (`n = 1,000`) & Kiểm định Ý nghĩa Thống kê McNemar (D4 / N10)

Theo đúng yêu cầu của `tasks/plan.md` (D4) và `docs/plans/task-breakdown.md` (`N10`), báo cáo tính khoảng tin cậy Bootstrap 95% (`1,000` lần lấy mẫu có hoàn lại trên `6,048` mẫu `test.parquet`, `seed = 42`) và kiểm định McNemar có hiệu chỉnh liên tục ($\chi^2_{\text{cc}}$):

1. **Khoảng tin cậy Bootstrap 95% trên `test.parquet` (`seed_42`):**
   - **`ENS-CROSS-FROZEN-PLATT`:** Macro-F1 = **`0.8113 [0.7977, 0.8247]`**, F1-TOXIC = **`0.6804 [0.6561, 0.7036]`**
   - **`ENS-CROSS-DIVERSE-050`:** Macro-F1 = **`0.8028 [0.7892, 0.8180]`**, F1-TOXIC = **`0.6612 [0.6367, 0.6879]`**
   - **`ENS-TV4-FROZEN-PLATT`:** Macro-F1 = **`0.8015 [0.7875, 0.8155]`**, F1-TOXIC = **`0.6642 [0.6396, 0.6881]`**
   - **`ENS-TV4-INTRA-4`:** Macro-F1 = **`0.8015 [0.7869, 0.8160]`**, F1-TOXIC = **`0.6606 [0.6341, 0.6857]`**
   - **`EXP-N6-C-calibrated @0.34` (TV2):** Macro-F1 = `0.8048 [0.7907, 0.8190]`, F1-TOXIC = `0.6728 [0.6484, 0.6972]`
   - **`EXP-N6-C-calibrated @0.50` (TV2):** Macro-F1 = `0.7989 [0.7838, 0.8142]`, F1-TOXIC = `0.6535 [0.6264, 0.6806]`
   - **`CNN-01` (TV4):** Macro-F1 = `0.7915 [0.7771, 0.8053]`, F1-TOXIC = `0.6453 [0.6204, 0.6699]`
   - **`CNN-01-WD` (TV4):** Macro-F1 = `0.7908 [0.7762, 0.8056]`, F1-TOXIC = `0.6411 [0.6151, 0.6671]`
   - **`HYBRID-RNN-01` (TV4):** Macro-F1 = `0.7761 [0.7607, 0.7909]`, F1-TOXIC = `0.6201 [0.5919, 0.6467]`

2. **Kiểm định McNemar Đôi một trên `test.parquet` (`seed_42`):**

| Cặp đối chiếu ($A$ vs $B$) | Chỉ $A$ đoán đúng ($b$) | Chỉ $B$ đoán đúng ($c$) | Chênh lệch mẫu đúng ($b - c$) | Thống kê $\chi^2_{\text{cc}}$ | $p\text{-value}$ | Ý nghĩa thống kê ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ENS-TV4-INTRA-4` vs `HYBRID-RNN-01`** | `181` | `81` | **`+100`** | `37.4084` | **`< 0.000001`** | **Có ý nghĩa thống kê rất mạnh** |
| **`ENS-TV4-FROZEN-PLATT` vs `HYBRID-RNN-01`** | `153` | `83` | **`+70`** | `20.1737` | **`0.000007`** | **Có ý nghĩa thống kê rất mạnh** |
| **`ENS-CROSS-DIVERSE-050` vs `HYBRID-RNN-01`** | `195` | `76` | **`+119`** | `51.3801` | **`< 0.000001`** | **Có ý nghĩa thống kê rất mạnh** |
| **`ENS-CROSS-FROZEN-PLATT` vs `HYBRID-RNN-01`** | `214` | `109` | **`+105`** | `33.4861` | **`< 0.000001`** | **Có ý nghĩa thống kê rất mạnh** |
| **`ENS-CROSS-FROZEN-PLATT` vs `EXP-N6-C-calibrated @0.34`** | `123` | `74` | **`+49`** | `11.6954` | **`0.000627`** | **Có ý nghĩa thống kê ($p < 0.001$)** |


