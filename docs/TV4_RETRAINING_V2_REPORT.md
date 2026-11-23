# Báo cáo Nghiên cứu Thực nghiệm & Huấn luyện Lại V2 (TV4 — `EXP-04-V2`)

- **Thành viên phụ trách:** TV4 (TheAnh)
- **Ngày hoàn thành:** 2026-11-24
- **Trạng thái:** `EXPERIMENTALLY VALIDATED` (Tổng cộng **54 runs V1 + 24 runs V2 = 78 official GPU runs** trên `NVIDIA GeForce RTX 4060 Laptop GPU`).
- **Tài liệu liên kết:**
  - Báo cáo Audit Phase 0–2: [docs/TV4_V2_PHASE0_AUDIT.md](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/docs/TV4_V2_PHASE0_AUDIT.md)
  - Báo cáo gốc V1 (54 runs): [docs/TV4_CNN_REPORT.md](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/docs/TV4_CNN_REPORT.md)
  - Cấu hình V2: [configs/cnn_v2_config.yaml](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/configs/cnn_v2_config.yaml)
  - Code kiến trúc & pipeline V2: [src/models/cnn.py](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/src/models/cnn.py), [src/models/cnn_pipeline.py](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/src/models/cnn_pipeline.py), [tests/test_cnn_v2.py](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/tests/test_cnn_v2.py)
  - Checkpoints & Logs V2: `models/cnn_v2/TV4-run-4e7ab33dcccb446a9cca550579e71213/`
  - Tổng hợp đánh giá V2 JSON: [artifacts/TV4-v2-research/tv4_v2_evaluation_summary.json](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/artifacts/TV4-v2-research/tv4_v2_evaluation_summary.json)

---

## 1. Tóm tắt Phát hiện Audit (Phase 0–2) & Bảo toàn 100% Baselines V1

Qua quá trình audit toàn diện trong [docs/TV4_V2_PHASE0_AUDIT.md](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/docs/TV4_V2_PHASE0_AUDIT.md), toàn bộ 54 runs V1 (`models/cnn/TV4-run-*/`) được giữ nguyên tuyệt đối và phát hiện 3 nút cổ chai kỹ thuật cốt lõi trong V1:

1. **Rò rỉ OOV do dấu câu dính vào từ và dùng chung `preprocessing="raw"` cho cả hai nhánh của Hybrid:**
   - Trong V1, `HYBRID-RNN-01` dùng `preprocessing="raw"` cho cả nhánh từ (BiGRU) lẫn nhánh ký tự (Char-CNN), khiến từ điển train phình lên **`31,084` từ** và **tỷ lệ từ ngoài từ điển (OOV) trên Validation lên tới `31.65%`** (`247` lượt từ khóa nhạy cảm như `ko, k, dc, đc, vl, vcl, đm, ngu, chó` bị dính dấu câu).
   - Khi tách riêng `word_preprocessing: v2_word_norm` (chuẩn hóa `NFC`, xóa ký tự ẩn zero-width, tách dấu câu khỏi biên từ, thu gọn ký tự lặp chữ cái về 1, mở rộng từ điển `TEENCODE_V2_DICT`) và `char_preprocessing: v2_char_norm` (giữ chữ hoa/thường và dấu câu cho nhánh ký tự), từ điển mức từ giảm xuống **`9,041` token sạch** (`min_freq=2`) và **tỷ lệ OOV trên Validation giảm sâu từ `31.65%` xuống `18.56%`** (`-13.1%` OOV).
2. **Nút cổ chai Pooling của BiGRU trên bình luận dài (`ViCTSD`):**
   - Trong V1, `HybridWordCharClassifier` chỉ lấy trạng thái ẩn cuối cùng (`hidden[-2], hidden[-1]`), dẫn đến tỷ lệ lỗi lên tới `17.87%` ở các bình luận dài $\ge 35$ từ.
   - Trong V2, `src/models/cnn.py` bổ sung **`MaskedSelfAttentionPool1D` (`rnn_pooling: "attn"`)** và **`Attention + Masked Global Max Pooling` (`rnn_pooling: "attn_max"`)** trên toàn bộ chuỗi trạng thái ẩn $H = [h_1, \dots, h_T]$, đồng thời bổ sung kiến trúc 3 nhánh **`TriHybridWordCNNBiGRUCharClassifier`** (`Word-CNN + BiGRU + Char-CNN`) hỗ trợ 3 cơ chế kết hợp (`concat`, `gated`, `attention`).
3. **Điều chuẩn mức kênh (`SpatialDropout1D`) & Lịch giảm tốc độ học (`CosineAnnealingLR`):**
   - Bổ sung `SpatialDropout1D` (`F.dropout1d` ngay sau lớp embedding), dropout riêng từng nhánh (`word_dropout`, `char_dropout`), `CosineAnnealingLR`, và ghi đầy đủ `brier_score` cùng `log_loss` vào `metrics.json`.

---

## 2. Kết quả Thực nghiệm V2 Đối chứng với Baselines V1 (3 Seeds `[42, 43, 44]`)

> **Phân biệt rõ các tập đánh giá:**
> - **Validation Full (`val.parquet`, `n = 6,048`)**: Tập validation chuẩn để chọn checkpoint (`early stopping`) và đối chiếu trực tiếp với 54 runs V1.
> - **Locked V2 Validation Holdout (`val_holdout_v2`, `n = 3,024`)**: Nửa phân tầng độc lập của `val.parquet` (chia cố định theo `(source, label)` với `seed = 42`), **hoàn toàn không được dùng để fit bộ hiệu chuẩn Platt hay chọn ngưỡng** (bộ hiệu chuẩn và ngưỡng chỉ được fit trên nửa `val_dev` `n = 3,024`).
> - **Historical Benchmark Test (`test.parquet`, `n = 6,048`, `test_blind = false`)**: Tập test lịch sử của repository, chỉ dùng để đánh giá đối chứng một lần với kết quả `N10` của V1.

| Biến thể / Mô hình | Nhóm & Thay đổi kỹ thuật chính | Số tham số | Val Macro-F1 (`@0.50`, Mean ± Std) | Val F1-TOXIC (`@0.50`) | Val PR-AUC | Val Brier / Log-Loss | Historical Test Macro-F1 (`@0.50`) | Historical Test F1-TOXIC (`@0.50`) | Historical Test PR-AUC | Historical Test Macro-F1 (Frozen Platt) | Historical Test Platt ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`[V1] HYBRID-RNN-01`** | Baseline V1 BiGRU (`last`) + Char-CNN (`raw`) | `1,964,226` | `0.7733 ± 0.0072` | `0.6149` | `0.6673` | — / `0.3564` | `0.7855 ± 0.0082` | `0.6352` | `0.6889` | `0.7854` | `0.0112` |
| **`[V1] HYBRID-RNN-WD`** | Baseline V1 BiGRU (`last`) + Char-CNN + `wd=1e-4` | `1,964,226` | `0.7731 ± 0.0052` | `0.6155` | `0.6652` | — / `0.3547` | `0.7808 ± 0.0026` | `0.6278` | `0.6932` | `0.7805` | `0.0100` |
| **`[V1] CNN-01-WD`** | Baseline V1 Word TextCNN `[3]` + `wd=1e-4` (`basic`) | `1,307,906` | `0.7705 ± 0.0027` | `0.6058` | `0.6751` | — / `0.3372` | `0.7839 ± 0.0071` | `0.6290` | `0.6945` | `0.7851` | `0.0139` |
| **`[V1] CHAR-02`** | Baseline V1 Pure Char-CNN `[3,4,5,7]` `filters=256` | `345,666` | `0.7550 ± 0.0040` | `0.5817` | `0.6316` | — / `0.3194` | `0.7579 ± 0.0092` | `0.5861` | `0.6497` | `0.7590` | `0.0125` |
| **`V2-BIGRU-PREP`** | BiGRU (`last`) + Char-CNN + **`v2_word_norm` / `v2_char_norm` + `CosineLR`** | **`1,243,138`** | **`0.7839 ± 0.0029`** | `0.6321` | `0.6899` | `0.0856` / `0.3104` | `0.7884 ± 0.0038` | `0.6394` | `0.7040` | `0.7898` | **`0.0079`** |
| **`V2-BIGRU-ATTN`** | BiGRU (**`Self-Attention Pooling`**) + Char-CNN + `SpatialDropout=0.15` | `1,309,186` | **`0.7854 ± 0.0029`** | **`0.6360`** | `0.6951` | `0.0880` / `0.3276` | **`0.7943 ± 0.0017`** | **`0.6507`** | `0.7129` | `0.7937` | `0.0091` |
| **`V2-BIGRU-ATTN-MAX`** | BiGRU (**`Attention + Max Pooling`**) + Char-CNN `[2,3,4,5,7]` | `1,326,466` | **`0.7855 ± 0.0019`** | `0.6333` | **`0.6994`** | **`0.0849`** / **`0.3143`** | `0.7925 ± 0.0026` | `0.6453` | **`0.7148`** | **`0.7958`** | `0.0108` |
| **`V2-BIGRU-H192`** | BiGRU (`h=192`, `attn_max`, `filters=160`, `lr=5e-4`) | `1,876,354` | `0.7729 ± 0.0094` | `0.6118` | `0.6726` | `0.0889` / `0.3352` | `0.7830 ± 0.0053` | `0.6294` | `0.7002` | `0.7846` | `0.0122` |
| **`V2-WCNN-K234`** | Word TextCNN **`kernels=[2,3,4]`** + `v2_word_norm` + `SpatialDropout=0.15` | **`1,004,802`** | **`0.7708 ± 0.0039`** | `0.6042` | **`0.6856`** | `0.0861` / `0.2969` | `0.7782 ± 0.0052` | `0.6170` | **`0.7064`** | `0.7850` | `0.0097` |
| **`V2-WCNN-K2345-F192`** | Word TextCNN `kernels=[2,3,4,5]`, `filters=192`, `SpatialDropout=0.20` | `1,502,626` | `0.7654 ± 0.0056` | `0.5931` | `0.6787` | `0.0878` / `0.3041` | `0.7682 ± 0.0085` | `0.5976` | `0.6947` | `0.7830` | `0.0087` |
| **`V2-WCNN-K3457-F256`** | Word TextCNN `kernels=[3,4,5,7]`, `filters=256`, `wd=2e-4` | `1,851,554` | `0.7513 ± 0.0045` | `0.5671` | `0.6611` | `0.0908` / `0.3125` | `0.7580 ± 0.0091` | `0.5789` | `0.6758` | `0.7756` | `0.0100` |
| **`V2-CHAR-K23457-F192`** | Pure Char-CNN `kernels=[2,3,4,5,7]`, `emb=96`, `filters=192` | `436,610` | `0.7187 ± 0.0450` | `0.5106` | `0.5873` | `0.1045` / `0.3510` | `0.7249 ± 0.0405` | `0.5212` | `0.6160` | `0.7386` | `0.0101` |

---

## 3. Phân tích Khoa học Theo Từng Giả thuyết Cải tiến (Hướng 2)

1. **Hiệu quả đột phá của Tách tiền xử lý `v2_word_norm` + `v2_char_norm` và `CosineAnnealingLR` (`V2-BIGRU-PREP`):**
   - Chỉ riêng việc sửa lỗi dấu câu dính vào từ (`OOV 31.65% -> 18.56%`) và bật lịch giảm tốc độ học `CosineAnnealingLR` đã nâng một mô hình đơn từ `0.7733 ± 0.0072` (`HYBRID-RNN-01`) lên **`0.7839 ± 0.0029`** trên Validation (**`+1.06%` Macro-F1**, **`+1.72%` F1-TOXIC**, **`+2.26%` PR-AUC**), giảm hơn một nửa độ lệch chuẩn giữa các seed (`0.0072` $\to$ `0.0029`) và cắt giảm **37% số tham số** (`1.96M` $\to$ `1.24M`).
2. **Hiệu quả của `MaskedSelfAttentionPool1D` và `Attention + Max Pooling` (`V2-BIGRU-ATTN` & `V2-BIGRU-ATTN-MAX`):**
   - Thay trạng thái cuối `last` bằng `MaskedSelfAttentionPool1D` (`V2-BIGRU-ATTN`) nâng tiếp Val Macro-F1 lên **`0.7854 ± 0.0029`** (`F1-TOXIC = 0.6360`), cải thiện trực tiếp miền câu dài **`ViCTSD` từ `0.6520` lên `0.6609`** (`+0.89%`) và đạt **`0.7943 ± 0.0017`** (`F1-TOXIC = 0.6507`) ngay tại ngưỡng mặc định `@0.50` trên tập Historical Test.
   - Kết hợp `Attention + Masked Global Max Pooling` cùng cửa sổ ký tự `[2,3,4,5,7]` (`V2-BIGRU-ATTN-MAX`) đạt **Val Macro-F1 cao nhất cho mô hình đơn (`0.7855 ± 0.0019`)**, **Val PR-AUC cao nhất (`0.6994`, `+3.21%` so với V1)**, và trên Historical Test đạt **PR-AUC `0.7148` (`+2.59%` so với V1)** cùng **Platt-calibrated Test Macro-F1 `0.7958` (`+1.04%` so với V1)**.
   - Đáng chú ý, chỉ một mô hình đơn `V2-BIGRU-ATTN-MAX` (`0.7855 ± 0.0019`, `15.3 MB`) đã đạt ngang điểm với cả tổ hợp 4 mô hình của V1 (`ENS-TV4-INTRA-4` `0.7855 ± 0.0053`, `77.3 MB`) nhưng nhỏ gọn hơn **5 lần**!
3. **Bằng chứng thực nghiệm từ các thí nghiệm âm tính (Negative Results — Tăng dung lượng quá mức):**
   - Tăng kích thước ẩn BiGRU lên `h=192` với `lr=5e-4` (`V2-BIGRU-H192`: `0.7729 ± 0.0094`) hoặc mở rộng Word TextCNN lên `[2,3,4,5] filters=192` (`0.7654`) và `[3,4,5,7] filters=256` (`0.7513`) đều làm giảm hiệu năng tổng quát hóa so với cấu hình gọn `h=128` / `kernels=[2,3,4] filters=128` (`V2-WCNN-K234`, đạt Val `0.7708`, best seed `0.7763`, **Test PR-AUC `0.7064`**, **Test Raw ECE chỉ `0.0206`**). Điều này khẳng định quy luật nhất quán trên tập dữ liệu `28.2k` mẫu khởi tạo ngẫu nhiên: **chất lượng tiền xử lý biên từ và cơ chế pooling quan trọng hơn nhiều so với việc phình to số chiều ẩn/số bộ lọc**.

---

## 4. Kết quả Tổ hợp V2 (`V2-ENS-TV4-INTRA`) Trên Cả 3 Tầng Đánh giá

Tổ hợp nội bộ 4 mô hình V2 (`0.35 V2-BIGRU-ATTN-MAX + 0.30 V2-BIGRU-ATTN + 0.20 V2-BIGRU-PREP + 0.15 V2-WCNN-K234`) được đánh giá tách biệt trên:
1. **Validation Full (`val.parquet`, `n = 6,048`, `@0.50` không tinh chỉnh ngưỡng):**
   - **Val Macro-F1 = `0.7921 ± 0.0013`** (tăng **`+0.66%`** so với `ENS-TV4-INTRA-4` của V1 `0.7855 ± 0.0053`, và vượt cả `ENS-TV4-OOF-PLATT` `0.7900` của V1 ngay ở ngưỡng mặc định `0.50`!).
2. **Kiểm chứng Không Rò rỉ trên `Locked V2 Validation Holdout` (`val_holdout_v2`, `n = 3,024`):**
   - Khi chỉ fit bộ hiệu chuẩn Platt và chọn ngưỡng trên nửa `val_dev` (`n = 3,024`, đạt `0.8044`), rồi áp dụng đóng băng sang nửa `val_holdout_v2` (`n = 3,024` chưa từng nhìn thấy), `V2-ENS-TV4-INTRA` tăng từ `0.7795` (`@0.50`) lên **`0.7850`** (`+0.55%` Macro-F1 trên blind holdout).
3. **Đánh giá Đối chứng trên `Historical Benchmark Test` (`test.parquet`, `n = 6,048`, `test_blind = false`):**
   - Ở ngưỡng cố định `@0.50`: đạt **Test Macro-F1 = `0.7988 ± 0.0026`**, `Test F1-TOXIC = 0.6559`, **`Test PR-AUC = 0.7344`**, `Test ECE = 0.0273`.
   - Khi áp dụng bộ hiệu chuẩn Platt + ngưỡng đã fit chỉ trên `val_dev` (`n = 3,024`): đạt **Test Macro-F1 = `0.8022 ± 0.0002`** (`Test F1-TOXIC = 0.6646`, độ lệch chuẩn giữa 3 seeds gần như bằng 0: `± 0.0002`).

---

## 5. Quyết định Chọn Mô hình Cuối cùng của TV4 (Phase 8–10)

| Mục đích sử dụng | Mô hình / Cấu hình được chọn | Val Macro-F1 (3 seeds) | Historical Test Macro-F1 | Historical Test F1-TOXIC | Historical Test PR-AUC | Platt Test ECE | Kích thước | Lý do chọn |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Mô hình Đơn Tốt nhất (Single Production Neural Model)** | **`V2-BIGRU-ATTN-MAX`** (`BiGRU Attn+Max + Char-CNN [2,3,4,5,7]`, `v2_word_norm` + `v2_char_norm`) | **`0.7855 ± 0.0019`** | **`0.7925`** (`@0.50`) / **`0.7958`** (Platt) | **`0.6453`** (`0.6507` ở `V2-BIGRU-ATTN`) | **`0.7148`** | **`0.0108`** | **`~15.3 MB`** | Vượt mô hình đơn tốt nhất V1 (`HYBRID-RNN-01`) **`+1.22%` Val Macro-F1**, **`+2.59%` Test PR-AUC**, giảm 37% tham số và nhỏ hơn 5 lần so với ensemble V1. |
| **2. Tổ hợp Thuần Neural Nội bộ TV4** | **`V2-ENS-TV4-INTRA`** (`V2-BIGRU-ATTN-MAX + V2-BIGRU-ATTN + V2-BIGRU-PREP + V2-WCNN-K234`) | **`0.7921 ± 0.0013`** | **`0.7988`** (`@0.50`) / **`0.8022 ± 0.0002`** (Platt) | **`0.6646`** | **`0.7344`** | **`0.0105`** | `~56.2 MB` | Đạt `0.7921` trên Val ngay ở `@0.50` với độ lệch chuẩn cực thấp `0.0013`. |
| **3. Tổ hợp Liên thành viên Toàn Nhóm (TV2 + TV4)** | **`ENS-CROSS-FROZEN-PLATT`** (`TV2 EXP-N6-C-calibrated + TV4 Neural`) | **`0.7989 ± 0.0003`** | **`0.8128 ± 0.0019`** | **`0.6834 ± 0.0027`** | **`0.7572`** | `0.0282` (`0.0073` ở `@0.50`) | `~91.7 MB` | Kết hợp tín hiệu TF-IDF tuyến tính diện rộng (`75k` features) của TV2 và tín hiệu biểu diễn tuần tự/hình thái học của TV4, đạt điểm tổng quát hóa cao nhất toàn dự án. |
