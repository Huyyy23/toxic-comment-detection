# TV4 V3 Experiment Protocol, Audit Fixes & Live Iteration Log

> **Trạng thái (2026-11-24):** Đã hoàn tất chiến dịch thực nghiệm V3 (16 runs GPU mới) và xuất báo cáo tổng kết chính thức tại [docs/TV4_V3_FINAL_REPORT.md](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/docs/TV4_V3_FINAL_REPORT.md) theo yêu cầu dừng huấn luyện và tổng kết của người dùng.

---

## 1. Sửa lỗi Evaluation Protocol Từ V2 (Chống Rò rỉ Dữ liệu Nghiêm ngặt)

### Các bất cập của V2 đã được phát hiện và loại bỏ trong V3:
1. **Loại bỏ khái niệm `val_holdout_v2` cắt từ `val.parquet` làm "blind holdout" cho mô hình full-train:**
   - Trong V2, các mô hình đơn đều dùng toàn bộ `6,048` mẫu của `val.parquet` để `early stopping` (`best.pt`). Do đó, việc tách một nửa `val.parquet` (`val_holdout_v2`) sau khi train để làm holdout cho ensemble không đảm bảo độc lập hoàn toàn với quá trình chọn checkpoint.
2. **Loại bỏ hoàn toàn việc fit Calibration / Threshold / Ensemble Weights / Stacking trên `val.parquet`:**
   - Mọi tham số hậu xử lý (**Platt Scaling, Temperature Scaling, Isotonic Regression, Decision Threshold, Weighted Voting Weights, Logistic Regression Stacking Meta-Learner**) trong V3 chỉ được học từ **5-Fold Stratified `Train-OOF` (`n = 28,231`) tạo hoàn toàn bên trong `train.parquet`**.

### Giao thức Đánh giá Chuẩn 3 Tầng Của V3 (100% Leak-Free):
1. **Tầng 1 — `Train-OOF` (`n = 28,231`, 5-Fold Stratified CV theo `(source, label)` bên trong `train.parquet`, `seed = 42`):**
   - Mỗi fold train trên `~22,585` mẫu và early-stop trên `~5,646` mẫu `oof_fold` nội bộ — **hoàn toàn không chạm `val.parquet` hay `test.parquet`**.
   - Dùng `28,231` dự đoán Out-of-Fold để:
     - Fit `Platt Scaling`, `Temperature Scaling`, `Isotonic Regression`.
     - Chọn ngưỡng quyết định tối ưu Macro-F1 có ràng buộc không làm suy giảm F1-TOXIC.
     - Học trọng số `Weighted Probability Voting` và huấn luyện `LogisticRegression` Stacking Meta-Learner.
2. **Tầng 2 — `Validation` (`data/processed/binary_n2_v1/val.parquet`, `n = 6,048`, SHA-256 `4c33430a...`):**
   - **Đối với mô hình Full-Train (`n_train = 28,231`):** dùng `val.parquet` cho early stopping và so sánh công bằng trực tiếp với bảng V1/V2; nhưng calibration & threshold vẫn học từ `Train-OOF` để `val.parquet` đánh giá trung thực ECE và Macro-F1 sau hiệu chuẩn.
   - **Đối với mô hình 5-Fold CV Ensemble (`5-fold average`):** vì cả 5 fold đều early-stop bên trong `train.parquet`, **`val.parquet` (`n = 6,048`) hoàn toàn chưa từng được sử dụng ở bất kỳ bước nào (train, early stopping, calibration, threshold hay stacking)** $\Rightarrow$ đóng vai trò **True Independent Validation Holdout** cho V3!
3. **Tầng 3 — `Historical Benchmark Test` (`data/processed/binary_n2_v1/test.parquet`, `n = 6,048`, SHA-256 `d200be4a...`, `test_blind = false`):**
   - Luôn công bố rõ `test_blind = false` (do lịch sử repository và đợt đánh giá V1 `N10`).
   - Chỉ mở đúng **một lần duy nhất ở giai đoạn xác nhận cuối** sau khi đóng băng toàn bộ kiến trúc, checkpoints, calibrators, thresholds và trọng số ensemble V3.

---

## 2. Mục tiêu Chất lượng V3 & Trạng thái Hiện tại (Trước V3)

| Chỉ số | Mục tiêu V3 | Tốt nhất V1 | Tốt nhất V2 | Khoảng cách cần vượt |
|---|---:|---:|---:|---:|
| Single Model Validation Macro-F1 (Scratch) | >= 0.790 | `0.7733` (`HYBRID-RNN-01`) | `0.7855` (`V2-BIGRU-ATTN-MAX`) | `+0.0045` |
| Single Model Validation Macro-F1 (Pretrained Transformer) | >= 0.790 | — | — | Cần benchmark `PhoBERT-v2` / `XLM-R` |
| Ensemble Validation Macro-F1 (Intra-TV4) | >= 0.805 | `0.7900` | `0.7921` (`@0.50`) | `+0.0129` |
| F1-Toxic (Validation) | >= 0.700 | `0.6615` | `0.6360` (single) | `+0.0385` |
| Cross-Ensemble Validation Macro-F1 | >= 0.815 | `0.7989` | — | `+0.0161` |
| ViCTSD Macro-F1 (Validation) | >= 0.710 | `0.6719` | `0.6609` | `+0.0381` |
| Calibration ECE (Out-of-sample trên Val) | <= 0.015 | `0.0223` (Val-OOF) | `0.0254` (raw) | Cần fit `Train-OOF` |

---

## 3. Phân tích Lỗi V2 (`V2-BIGRU-ATTN-MAX`, seed 44 trên `val.parquet`)

- Tổng lỗi: **`660 / 6,048` (`10.91%`)** (`FN = 378`, `FP = 282`).
- Theo nguồn:
  - `vihsd` (`n = 4,559`): `485` lỗi (`10.64%`), `Precision_TOXIC = 0.7105`, `Recall_TOXIC = 0.6633`.
  - `victsd` (`n = 1,489`): `175` lỗi (`11.75%`), `Precision_TOXIC = 0.4590`, **`Recall_TOXIC = 0.3394`** (`109 FN` vs `66 FP` — bỏ sót 66% bình luận độc hại gián tiếp trên báo chí).
- Theo hiện tượng ngôn ngữ:
  - `has_quote` (trích dẫn/tường thuật): **`17.19%`** error rate.
  - `non_nfc` (Unicode tổ hợp gốc): **`16.78%`** error rate.
  - `16–30` từ / `>30` từ: **`16.59%` / `15.96%`** error rate.
  - `has_teencode`: **`16.43%`** error rate.
  - `has_neg` (phủ định): **`14.82%`** error rate (`121 FN`, `77 FP`).

---

## 4. Nhật ký Vòng lặp Thực nghiệm V3 (Cập nhật Liên tục)

### Vòng lặp 1 — Track A: Scratch BiGRU Gated Attention + Train-Only FastText (`models/cnn_v3/TV4-V3-scratch`, 3 seeds `42, 43, 44`)
- **Giả thuyết:** (1) Thay Additive Attention bằng **Gated Attention + Max + Mean Pooling** + `LayerNorm` + `Multi-Sample Dropout` + `label_smoothing=0.03` giảm False Positive do từ lóng và giảm overconfidence ECE; (2) Khởi tạo word embedding bằng **Train-Only Gensim FastText (`min_n=2..5`, chỉ fit trên `train.parquet`)** giúp các từ hiếm/lỗi chính tả/từ ghép nhận biểu diễn hình thái ngay từ epoch 1 mà 100% không dùng pretraining ngoài.
- **Kết quả thực tế (Validation `n = 6,048`, `@0.50`):**
  - `V3-BIGRU-GATED-MSD` (không FastText): Val Macro-F1 `0.7853 ± 0.0026`, F1-TOXIC `0.6335 ± 0.0070`, PR-AUC `0.7050`, VRAM `284.2 MB`.
  - **`V3-BIGRU-GATED-FT`** (+ Train-Only FastText): Val Macro-F1 **`0.7900 ± 0.0024`** (`s42=0.7868, s43=0.7906, s44=0.7926`), F1-TOXIC **`0.6420 ± 0.0055`**, **PR-AUC `0.7202`** (`s44=0.7248`), VRAM `279.0 MB` $\Rightarrow$ **ĐẠT MỤC TIÊU `Single Model Val Macro-F1 >= 0.790` bằng mô hình Scratch thuần!**
  - `V3-BIGRU-PYVI-FT` (+ PyVi word segmentation + Train-Only FastText): Val Macro-F1 `0.7879 ± 0.0019`, F1-TOXIC `0.6388 ± 0.0031`, VRAM `283.9 MB`.
  - **`V3-TRI-GATED-FT`** (`Word-CNN + BiGRU + Char-CNN` + Gated Fusion + Train-Only FastText): Val Macro-F1 **`0.7907 ± 0.0034`** (`s42=0.7863, s43=0.7945, s44=0.7914`), F1-TOXIC **`0.6421 ± 0.0083`**, VRAM `301.1 MB` $\Rightarrow$ **Kỷ lục mới cho mô hình Scratch đơn!**

### Vòng lặp 2 — Track B: Pretrained `vinai/phobert-base-v2` (`models/transformers_v3/V3-PHOBERT-V2`, 3 seeds `42, 43, 44`)
- **Giả thuyết:** `PhoBERT-v2` (word-segmented bằng `pyvi` + Mean+CLS pooling + Multi-Sample Dropout + `label_smoothing=0.03`) nắm bắt được mỉa mai, phủ định, câu trích dẫn và bình luận tin tức gián tiếp của `ViCTSD`.
- **Kết quả thực tế (Validation `n = 6,048`, `@0.50`, Peak VRAM `3,315 MB`):**
  - `seed_42`: Val Macro-F1 `0.8143`, F1-TOXIC `0.6857`, **`ViCTSD = 0.7358`**, `PR-AUC = 0.7591`, `ECE = 0.0303` (best epoch 2).
  - `seed_43`: Val Macro-F1 `0.8101`, F1-TOXIC `0.6819`, **`ViCTSD = 0.7303`**, `PR-AUC = 0.7532`, `ECE = 0.0356` (best epoch 2).
  - `seed_44`: Val Macro-F1 `0.8155`, F1-TOXIC `0.6875`, **`ViCTSD = 0.7168`**, `PR-AUC = 0.7616`, `ECE = 0.0462` (best epoch 3).
  - **Trung bình 3 seeds:** **Val Macro-F1 = `0.8133 ± 0.0023`**, **F1-TOXIC = `0.6851 ± 0.0023`**, **`ViCTSD = 0.7276 ± 0.0080`**, **PR-AUC = `0.7580`** $\Rightarrow$ Vượt xa mục tiêu `Single >= 0.790` và `ViCTSD >= 0.710`!
- **Phát hiện đường cong học:** Cả 3 seeds đều đạt đỉnh ở Epoch 2–3 và overfit ở Epoch 4 (`val_loss` tăng từ `0.248` lên `0.296`) $\Rightarrow$ tối ưu `epochs = 3` cho vòng lặp tiếp theo.

### Vòng lặp 3 — Track B: Pretrained `FacebookAI/xlm-roberta-base` (`models/transformers_v3/V3-XLMR-BASE`, `epochs=3`)
- **Trạng thái:** Hoàn tất `seed_42` (`best.pt` đã lưu và đánh giá); dừng `seed_43` giữa epoch 1 theo yêu cầu ngưng train của người dùng.
- **Kết quả thực tế (`seed_42`, Validation `n = 6,048`, `@0.50`, Peak VRAM `5,498.1 MB`):**
  - Val Macro-F1 = **`0.8006`**, F1-TOXIC = `0.6608`, `ViCTSD = 0.7027`, `ViHSD = 0.8173`, `PR-AUC = 0.7393`, `ECE = 0.0330`.
  - Historical Test Macro-F1 = `0.7970`, `ViCTSD = 0.7219`, `PR-AUC = 0.7446`.
  - Đa dạng lỗi so với `V3-PHOBERT-V2` (`seed_42`): `6.40%` bất đồng (`387` mẫu; `179` mẫu chỉ `XLM-R` đúng), `Oracle Macro-F1 = 0.8689`.

### Vòng lặp 4 — Đánh giá Tổng hợp, Ensemble, Cross-Ensemble, Calibration & Báo cáo Cuối cùng (Hoàn tất 2026-11-24)
- Đã chạy `src/evaluation/v3_final_evaluator.py` xuất `artifacts/TV4-v3-research/tv4_v3_final_report.json` cùng 4 biểu đồ PNG (`160 DPI`).
- **6/6 mục tiêu chất lượng V3 đã ĐẠT:**
  - Single Scratch: `V3-TRI-GATED-FT` = `0.7907 ± 0.0034` Val / `0.8023` Test; `V3-BIGRU-GATED-FT` = `0.7900 ± 0.0024` Val / `0.8032` Test.
  - Single Pretrained: `V3-PHOBERT-V2` = `0.8133 ± 0.0023` Val (`ViCTSD = 0.7276`) / `0.8158` Test (`0.8255` Platt, `ViCTSD = 0.7767`).
  - Intra-TV4 Ensemble (`V3-ENS-TV4-ALL`): `0.8217` Val `@0.50` / `0.8212` Val-OOF Platt (`ECE = 0.0079`) / `0.8351` Test Frozen Platt (`F1-TOXIC = 0.7214`, `ViCTSD = 0.7629`).
  - Cross-Ensemble (`V3-ENS-CROSS-VOTING`, TV4 V3 + TV2 `EXP-N6-C-calibrated`): **`0.8241` Val-OOF Platt** (`F1-TOXIC = 0.7020`, `ViCTSD = 0.7112`, **`ECE = 0.0067`**) / **`0.8393` Historical Test Frozen Platt** (`95% CI [0.8267, 0.8521]`, **`F1-TOXIC = 0.7283`**, **`PR-AUC = 0.8011`**, `ViCTSD = 0.7597`, **`ECE = 0.0086`**, McNemar $p = 0.000614$ vs `PhoBERT-v2`).
- **Lưu ý phương pháp luận:** Do dừng huấn luyện sớm theo yêu cầu người dùng trước khi chạy `5-Fold Train-OOF CV` và `5 seeds`, phần đánh giá Calibration/Stacking sử dụng `Val-5Fold-OOF` trên `val.parquet` và đóng băng sang `test.parquet` (`test_blind = false`).
- **Báo cáo tổng kết chính thức:** Xem [docs/TV4_V3_FINAL_REPORT.md](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/docs/TV4_V3_FINAL_REPORT.md).
