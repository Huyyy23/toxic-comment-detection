# Báo cáo Audit Toàn diện Phase 0–Phase 2 (TV4 Experiment Research & Retraining V2)

- **Thành viên phụ trách:** TV4 (TheAnh)
- **Ngày thực hiện Audit:** 2026-11-24
- **Phạm vi:** Kiểm tra toàn bộ repository, chất lượng dữ liệu, chống rò rỉ dữ liệu (Anti-Leakage), khả năng tái lập các baseline V1 (`54 runs`) và xác định các điểm nghẽn kỹ thuật có bằng chứng thực nghiệm trước khi triển khai chiến dịch **Retraining V2**.

---

## 1. Phase 0 — Audit Hiện trạng Repository & Các Điểm Nghẽn Kỹ thuật Của V1

### 1.1. Những phần đã triển khai đúng và được bảo toàn nguyên vẹn
1. **Không rò rỉ dữ liệu giữa các tập (`train.parquet`, `val.parquet`, `test.parquet`):**
   - Toàn bộ 54 runs V1 (`models/cnn/TV4-run-*/`) chỉ fit `SequenceVocab`, `class_weight` và `WeightedRandomSampler` trên `train.parquet`.
   - Kiểm chứng độc lập `verify_all_runs_integrity()` xác nhận **54/54 runs** khớp 100% SHA-256 dataset, trạng thái MLflow `FINISHED`, tổng xác suất `= 1.0`, và nạp lại checkpoint `best.pt` khớp dự đoán **`6,048 / 6,048` mẫu** trên `val.parquet`.
2. **Bảo toàn 100% Artifacts V1 (Phase 2 — Preserve Baselines):**
   - Toàn bộ thư mục `models/cnn/TV4-run-*/`, `artifacts/TV4-cnn-analysis/`, `artifacts/TV4-ensemble-analysis/`, `artifacts/TV4-locked-test/`, `models/ensemble/TV4-EXP06/`, `models/locked_test/TV4-N10/` và báo cáo `docs/TV4_CNN_REPORT.md` được giữ nguyên tuyệt đối, không xóa và không ghi đè.
   - Chiến dịch V2 lưu vào không gian độc lập: `configs/cnn_v2_config.yaml`, `models/cnn_v2/`, `artifacts/TV4-v2-research/`, và báo cáo mới `docs/TV4_RETRAINING_V2_REPORT.md`.

### 1.2. Phát hiện 4 Điểm Nghẽn Kỹ thuật Trong Code V1 (Có Bằng Chứng Thực Nghiệm)

| # | Điểm nghẽn trong V1 | Vị trí code V1 | Bằng chứng định lượng từ dữ liệu & lỗi V1 | Giải pháp nâng cấp trong V2 |
|---:|:---|:---|:---|:---|
| **1** | **BiGRU/BiLSTM chỉ lấy trạng thái cuối (`hidden[-2], hidden[-1]`) thay vì Attention + Max Pooling trên toàn chuỗi** | `src/models/cnn.py` dòng `281–285` | Tỷ lệ lỗi của `HYBRID-RNN-01` tăng vọt từ `7.51%` (câu $\le 5$ từ) lên **`17.87%` (câu dài $\ge 35$ từ)**, làm giảm Macro-F1 trên `ViCTSD` (`0.6542` so với `0.7961` trên `ViHSD`) vì trạng thái cuối quên các mệnh đề độc hại ở giữa câu. | Giải nén toàn bộ chuỗi ẩn $H = [h_1, \dots, h_T]$ và bổ sung **Masked Self-Attention Pooling (`attn`)** và **Attention + Masked Global Max Pooling (`attn_max`)**. |
| **2** | **Chưa tách dấu câu khỏi biên từ & dùng chung `preprocessing="raw"` cho cả nhánh Word lẫn nhánh Char trong Hybrid** | `src/models/cnn_pipeline.py` dòng `74, 423–428` | Khi dùng `text.split()` trên `raw`, từ điển train phình lên **`31,084` từ** (`basic`: `25,909` từ) do dấu câu dính vào từ (`"ngu,"`, `"ngu!"`, `"chó."`, `"đm?"`), khiến **tỷ lệ từ ngoài từ điển (OOV) trên Validation lên tới `31.65%–32.35%`**! Khi tách dấu câu + chuẩn hóa ký tự lặp về 1 + mở rộng từ điển teencode (`v2_word_norm`), số từ duy nhất giảm còn **`13,096` từ** và **tỷ lệ OOV trên Validation giảm sâu xuống `18.56%`** (`-13.1%` OOV). | Tách riêng `word_preprocessing: v2_word_norm` cho nhánh từ (TextCNN/BiGRU) và `char_preprocessing: v2_char_norm` (hoặc `raw`) cho nhánh ký tự (Char-CNN). |
| **3** | **Thiếu `SpatialDropout1D` (Channel Embedding Dropout), Branch-Specific Dropout và Learning Rate Scheduler** | `src/models/cnn.py` dòng `107`; `src/models/cnn_pipeline.py` dòng `462` | Learning curves V1 cho thấy các mô hình đạt đỉnh ở epoch `3–5` rồi overfit mạnh (Train Macro-F1 `> 0.92`, Raw ECE tăng lên `0.063–0.089`) do `lr = 1e-3` cố định suốt 15 epochs và mô hình học vẹt từng từ đơn lẻ (`"chó"`, `"vl"`, `"đm"` gây False Positive). | Bổ sung **`SpatialDropout1D` (`Dropout1d` trên kênh embedding)**, **dropout riêng cho từng nhánh (`word_dropout`, `char_dropout`)**, và **`CosineAnnealingLR` / `ReduceLROnPlateau`**. |
| **4** | **Hiệu chuẩn xác suất (Calibration) & Stacking trong V1 mới dừng ở `Val-OOF` (chưa có `Train-OOF` trên `train.parquet`)** | `src/evaluation/ensemble_report.py` | Trong V1, bộ hiệu chuẩn Platt của mô hình neural được fit trên `val.parquet` (hoặc 5-fold Val-OOF) và bỏ qua Train-OOF Stacking do chi phí. | Triển khai **5-Fold Stratified `Train-OOF` trên `train.parquet` (`n = 28,231`)** cho các ứng viên hạt giống V2 + TV2 `EXP-N6-C`, cho phép fit **Platt, Temperature, Isotonic Regression và Logistic Regression Stacking Meta-Learner hoàn toàn từ `Train-OOF`** mà không chạm nhãn `val.parquet`. |

---

## 2. Phase 1 — Kiểm định Chất lượng Dữ liệu, Anti-Leakage & Phân tầng Holdout

1. **Kiểm tra phân phối nhãn, nguồn và chuẩn hóa Unicode (`data/processed/binary_n2_v1/`):**
   - `train.parquet`: `28,231` mẫu (`23,734 CLEAN = 84.07%`, `4,497 TOXIC = 15.93%`; nguồn: `21,277 vihsd`, `6,954 victsd`). Phát hiện **`708` dòng chưa ở chuẩn Unicode `NFC`** và `144` dòng viết IN HOA dày đặc.
   - `val.parquet`: `6,048` mẫu (`5,084 CLEAN = 84.06%`, `964 TOXIC = 15.94%`; nguồn: `4,559 vihsd`, `1,489 victsd`). Phát hiện **`143` dòng chưa ở chuẩn `NFC`**.
   - `test.parquet`: `6,048` mẫu (`5,084 CLEAN = 84.06%`, `964 TOXIC = 15.94%`; nguồn: `4,558 vihsd`, `1,490 victsd`). Phát hiện **`144` dòng chưa ở chuẩn `NFC`**.
2. **Kiểm tra trùng lặp giữa các tập:**
   - Số mẫu trùng lặp văn bản sau khi chuẩn hóa `NFC + lowercase + whitespace` giữa `train–val`, `train–test` và `val–test`: **`0` mẫu (`0.00%`)**.
   - Phát hiện `247` lượt từ khóa nhạy cảm/teencode (`ko, k, dc, đc, vl, vcl, đm, dm, ngu, chó`) bị dính trực tiếp dấu câu trong `train.parquet`, làm lệch hoàn toàn phép tra cứu `TEENCODE_DICT` cũ.
3. **Chiến lược 4 Tầng Đánh giá Chống Rò rỉ & Minh bạch Hóa `test_blind = false`:**
   - Vì `test.parquet` của repository có `test_blind = false` (từng có lịch sử truy cập ở các nhánh cũ và đã được đánh giá ở mốc V1 `N10`), báo cáo V2 **không gọi `test.parquet` là blind holdout mới** mà phân tách rạch ròi **4 tầng đánh giá độc lập**:
     1. **Tầng 1 — `Train-OOF` (`n = 28,231`, 5-Fold Stratified CV chỉ trên `train.parquet`):** Nguồn dữ liệu duy nhất để fit bộ hiệu chuẩn xác suất (`Platt`, `Temperature`, `Isotonic`), chọn ngưỡng quyết định (`threshold`) và huấn luyện `LogisticRegression` Stacking Meta-Learner.
     2. **Tầng 2 — `Validation Full` (`val.parquet`, `n = 6,048`) & `Validation Dev Split` (`val_dev`, `n = 3,024`):** Dùng để early stopping và sàng lọc kiến trúc ở Phase 3–6 (đồng thời giữ tương thích 100% với bảng xếp hạng V1).
     3. **Tầng 3 — `Locked V2 Validation Holdout` (`val_holdout_v2`, `n = 3,024`):** Nửa phân tầng độc lập của `val.parquet` (chia cố định theo `(source, label)` với `seed = 42` trước khi chạy V2), **không được dùng để chọn trọng số ensemble hay chọn ngưỡng**, đóng vai trò holdout kiểm chứng độc lập cho các quyết định chọn mô hình của V2.
     4. **Tầng 4 — `Historical Benchmark Test` (`test.parquet`, `n = 6,048`, `test_blind = false`):** Chỉ mở đúng 1 lần ở Phase 9 sau khi đã đóng băng manifest V2 (`tv4_v2_freeze_manifest.json`) để đối chiếu công bằng với V1.

---

## 3. Phase 2 — Bảo toàn và Xác nhận Tái lập 7 Baseline V1

Trước khi chạy các thí nghiệm V2, toàn bộ 7 baseline cốt lõi từ V1 đã được xác minh tính toàn vẹn (`all_integrity_checks_passed = True`, khớp 100% dự đoán khi nạp lại checkpoint):

| Mã Baseline V1 | Loại mô hình | Val Macro-F1 (`@0.50`, 3 seeds) | Val F1-TOXIC (`@0.50`) | Val PR-AUC | Val Raw/Platt ECE | Historical Test Macro-F1 (`N10`) | Historical Test F1-TOXIC (`N10`) | Đường dẫn Artifacts gốc được bảo toàn |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`HYBRID-RNN-01`** | BiGRU (`last`) + Char-CNN (`raw`) | `0.7733 ± 0.0072` | `0.6149 ± 0.0130` | `0.6673` | `0.0635` / `0.0078` | `0.7855 ± 0.0082` | `0.6352 ± 0.0134` | `models/cnn/TV4-run-46dfa77e45994e48a27d7822e4ccf42f/HYBRID-RNN-01/` |
| **`HYBRID-RNN-WD`** | BiGRU (`last`) + Char-CNN + `wd=1e-4` | `0.7731 ± 0.0052` | `0.6155 ± 0.0103` | `0.6652` | `0.0653` / `0.0109` | `0.7808 ± 0.0026` | `0.6278 ± 0.0040` | `models/cnn/TV4-run-46dfa77e45994e48a27d7822e4ccf42f/HYBRID-RNN-WD/` |
| **`CNN-01-WD`** | Word TextCNN `[3]` + `wd=1e-4` (`basic`) | `0.7705 ± 0.0027` | `0.6058 ± 0.0056` | `0.6751` | `0.0494` / `0.0064` | `0.7839 ± 0.0071` | `0.6290 ± 0.0133` | `models/cnn/TV4-run-0a40921400b646aa916d560517a4bebf/CNN-01-WD/` |
| **`CNN-01`** | Word TextCNN `[3]` (`basic`) | `0.7702 ± 0.0041` | `0.6065 ± 0.0091` | `0.6734` | `0.0550` / `0.0125` | `0.7867 ± 0.0049` | `0.6350 ± 0.0105` | `models/cnn/TV4-run-c97812df7da7416395530962c87a7664/CNN-01/` |
| **`CHAR-02`** | Pure Char-CNN `[3,4,5,7]` `filters=256` | `0.7550 ± 0.0040` | `0.5817 ± 0.0105` | `0.6316` | `0.0460` / `0.0112` | `0.7579 ± 0.0092` | `0.5861 ± 0.0187` | `models/cnn/TV4-run-beab1e7c49a6401e902710849eacb8bf/CHAR-02/` |
| **`ENS-TV4-INTRA-4`** | Weighted Voting 4 họ TV4 (`@0.50`) | `0.7855 ± 0.0053` | `0.6313 ± 0.0105` | `0.7073` | `0.0350` | `0.7991 ± 0.0028` | `0.6548 ± 0.0059` | `models/ensemble/TV4-EXP06/ENS-TV4-INTRA-4/` |
| **`ENS-TV4-FROZEN-PLATT`** | Calibrated Voting 6 mô hình TV4 | `0.7900 ± 0.0051` | `0.6447 ± 0.0097` | `0.7104` | `0.0271` | `0.8029 ± 0.0025` | `0.6660 ± 0.0065` | `models/ensemble/TV4-EXP06/ENS-TV4-OOF-PLATT/` |
