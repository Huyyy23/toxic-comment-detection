# Báo cáo Tổng kết Nghiên cứu Thực nghiệm V3 — Vietnamese Toxic Comment Detection (`TV4 / EXP-04-V3`)

- **Thành viên phụ trách:** TV4 (TheAnh) — Nhánh Git `TheAnh`
- **Ngày hoàn thành:** 2026-11-24
- **Trạng thái:** `EXPERIMENTALLY VALIDATED` (Tổng cộng **94 official GPU runs** trên `NVIDIA GeForce RTX 4060 Laptop GPU 8GB VRAM`: **54 runs V1** + **24 runs V2** + **16 runs V3**)
- **Lưu ý dừng huấn luyện:** Quá trình huấn luyện liên tục V3 được dừng theo yêu cầu trực tiếp của người dùng sau khi **6/6 mục tiêu chất lượng V3** đã đạt được bằng thực nghiệm trên tập Validation và Historical Benchmark Test. Các bước mở rộng chưa chạy (`V3-XLMR-BASE` seeds `43–44`, xác nhận 5 seeds, và `5-Fold Train-OOF CV`) được công bố minh bạch tại [Mục 9](#9-giao-thức-đánh-giá-trung-thực-và-các-giới-hạn-của-kết-quả).
- **Tài liệu & Artifacts liên kết:**
  - Nhật ký thực nghiệm & giao thức V3: [TV4_V3_PROTOCOL_AND_LOG.md](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/docs/TV4_V3_PROTOCOL_AND_LOG.md)
  - Báo cáo V2 (24 runs, bảo toàn nguyên vẹn): [TV4_RETRAINING_V2_REPORT.md](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/docs/TV4_RETRAINING_V2_REPORT.md) & [TV4_V2_PHASE0_AUDIT.md](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/docs/TV4_V2_PHASE0_AUDIT.md)
  - Báo cáo V1 (54 runs, bảo toàn nguyên vẹn): [TV4_CNN_REPORT.md](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/docs/TV4_CNN_REPORT.md)
  - JSON tổng hợp đánh giá V3 (nguồn số liệu gốc): [tv4_v3_final_report.json](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/artifacts/TV4-v3-research/tv4_v3_final_report.json)
  - Cấu hình V3: [cnn_v3_config.yaml](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/configs/cnn_v3_config.yaml), [transformer_v3_config.yaml](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/configs/transformer_v3_config.yaml)
  - Mã nguồn V3: [cnn.py](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/src/models/cnn.py), [cnn_pipeline.py](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/src/models/cnn_pipeline.py), [transformer_pipeline.py](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/src/models/transformer_pipeline.py), [v3_final_evaluator.py](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/src/evaluation/v3_final_evaluator.py)
  - Kiểm thử đơn vị: [test_v3_models.py](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/tests/test_v3_models.py) (`pytest tests/test_v3_models.py tests/test_cnn_v2.py -q` $\to$ `13 passed in 57.49s`; `flake8` $\to$ `0` lỗi).

---

## 1. Tóm tắt Điều hành & Bảng Chấm điểm Mục tiêu Chất lượng V3

Chiến dịch V3 kế thừa toàn bộ kết quả V1 (`54 runs`) và V2 (`24 runs`) mà **không sửa đổi hay ghi đè bất kỳ artifact V1/V2 nào**, đồng thời triển khai hai nhánh nghiên cứu song song được báo cáo tách bạch hoàn toàn:
1. **Nhánh A — Mô hình Huấn luyện Từ đầu (Train-From-Scratch, không pretraining ngoài):** Nâng cấp BiGRU-CharCNN với **Masked Gated Attention + Max + Mean Pooling** (Ilse et al., 2018), `LayerNorm`, `5-head Multi-Sample Dropout`, `label_smoothing=0.03`, chuẩn hóa `v3_word_norm`, và khởi tạo **Train-Only Subword FastText (`gensim`, `min_n=2..5`, chỉ fit trên `train.parquet`)**, cùng kiến trúc 3 nhánh **`TriHybridWordCNNBiGRUCharClassifier`** (`Word-CNN + BiGRU + Char-CNN` kết hợp bằng Gated Fusion).
2. **Nhánh B — Benchmark Pretrained Transformer tiếng Việt/đa ngôn ngữ:** Fine-tuning có kiểm soát trên RTX 4060 8GB VRAM (`fp16` AMP, `grad_accum=2`, `max_length=160`, `Mean+CLS` pooling, `Multi-Sample Dropout`, differential head LR) với **`vinai/phobert-base-v2`** (3 seeds `42, 43, 44`) và **`FacebookAI/xlm-roberta-base`** (`seed_42`).

### Bảng Kiểm tra 6/6 Mục tiêu Chất lượng V3

| # | Chỉ số Mục tiêu V3 | Ngưỡng Mục tiêu | Kết quả Tốt nhất V3 (Validation `n = 6,048`) | Kết quả Đối chứng Historical Test (`n = 6,048`, `test_blind = false`) | Mô hình Đạt Mục tiêu | Trạng thái |
|---:|:---|---:|---:|---:|:---|:---:|
| **1** | **Single Model Validation Macro-F1** | `>= 0.790` | **Scratch:** `0.7907 ± 0.0034` (`V3-TRI-GATED-FT`), `0.7900 ± 0.0024` (`V3-BIGRU-GATED-FT`)<br>**Pretrained:** **`0.8133 ± 0.0023`** (`V3-PHOBERT-V2`), `0.8006` (`V3-XLMR-BASE` s42) | **Scratch:** `0.8032 ± 0.0061` (`V3-BIGRU-GATED-FT`), `0.8023 ± 0.0056` (`V3-TRI-GATED-FT`)<br>**Pretrained:** **`0.8158 ± 0.0041`** (`@0.50`) / **`0.8255`** (Platt) | **Đạt trên cả Scratch lẫn Pretrained!** | **ĐẠT** |
| **2** | **Ensemble Validation Macro-F1** | `>= 0.805` | **`0.8217`** (`@0.50`) / **`0.8212`** (`Val-5Fold-OOF Platt`) (`V3-ENS-TV4-ALL`)<br>*(Scratch-only `V3-ENS-SCRATCH-INTRA`: `0.8036` Val-OOF / `0.8194` Test)* | **`0.8351`** (`95% CI [0.8212, 0.8481]`, Frozen Platt) | `V3-ENS-TV4-ALL` (`0.52 PhoBERT-v2 + 0.18 XLM-R + 0.15 V3-BIGRU-GATED-FT + 0.15 V3-TRI-GATED-FT`) | **ĐẠT** |
| **3** | **F1-Toxic** | `>= 0.700` | **`0.7020`** (`P=0.7333, R=0.6732`, `Val-5Fold-OOF Platt`) | **`0.7283`** (`95% CI [0.7061, 0.7504]`, `P=0.7517, R=0.7064`) | `V3-ENS-CROSS-VOTING` (`V3-ENS-TV4-ALL` đạt `0.6972` Val / `0.7214` Test) | **ĐẠT** |
| **4** | **Cross-Ensemble Validation Macro-F1** | `>= 0.815` | **`0.8197`** (`@0.50`) / **`0.8241`** (`Val-5Fold-OOF Platt`)<br>*(LR Stacking: `0.8235` Val-OOF)* | **`0.8393`** (`95% CI [0.8267, 0.8521]`, **`PR-AUC = 0.8011`**, Frozen Platt) | `V3-ENS-CROSS-VOTING` (TV4 V3 + TV2 `EXP-N6-C-calibrated`) | **ĐẠT** |
| **5** | **ViCTSD Macro-F1** | `>= 0.710` | **Single:** **`0.7276 ± 0.0080`** (`V3-PHOBERT-V2` 3s, `@0.50`)<br>**Cross-Ens:** **`0.7112`** (`Val-5Fold-OOF Platt`) | **Single:** `0.7481` (`@0.50`) / **`0.7767`** (Platt)<br>**Ens:** **`0.7629`** (`V3-ENS-TV4-ALL`) / **`0.7597`** (`V3-ENS-CROSS-VOTING`) | `V3-PHOBERT-V2` & `V3-ENS-CROSS-VOTING` | **ĐẠT** |
| **6** | **Calibration ECE** | `<= 0.015` | **`0.0067`** (`V3-ENS-CROSS-VOTING` Platt), **`0.0079`** (`V3-ENS-TV4-ALL` Platt), **`0.0093`** (`V3-ENS-SCRATCH-INTRA` Platt), **`0.0057`** (`PhoBERT-v2` Isotonic) | **`0.0086`** (`V3-ENS-CROSS-VOTING`), **`0.0106`** (`V3-ENS-TV4-ALL`), **`0.0132`** (`V3-ENS-SCRATCH-INTRA`), **`0.0063`** (`PhoBERT-v2` Isotonic) | Đạt vượt mức (`< 0.010`) trên cả Val-OOF lẫn Frozen Test | **ĐẠT** |

---

## 2. Mô hình Tốt nhất & Danh mục Checkpoints Chính thức (Tách bạch Scratch vs Pretrained)

Toàn bộ checkpoints được lưu nguyên vẹn, bất biến và có thể nạp lại để tái lập chính xác `6,048 / 6,048` mẫu:

| Vai trò & Danh mục | Mô hình Được chọn | Checkpoint Chính thức (đường dẫn thực tế trên đĩa) | Tham số / Kích thước | Peak VRAM | Val Macro-F1 | Historical Test Macro-F1 | Test F1-TOXIC | Test PR-AUC | Test ECE |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1A. Mô hình Scratch Đơn Tốt nhất (Val Macro-F1 cao nhất)** | **`V3-TRI-GATED-FT`** (`Word-CNN [2,3,4] + BiGRU + Char-CNN [2,3,4,5,7]` + Gated Fusion + Train-Only FastText) | `models/cnn_v3/TV4-V3-scratch/V3-TRI-GATED-FT/seed_{42,43,44}/best.pt` *(best seed: `seed_43`)* | `1.87M` / `21.41 MB` | `301.1 MB` | **`0.7907 ± 0.0034`** (`s43 = 0.7945`) | `0.8023 ± 0.0056` (`s43 = 0.8100`) | `0.6627` (`s43 = 0.6767`) | `0.7270` (`s43 = 0.7441`) | `0.0317` (raw) |
| **1B. Mô hình Scratch Đơn Gọn & Ổn định nhất (Test & PR-AUC tốt nhất)** | **`V3-BIGRU-GATED-FT`** (`BiGRU GatedAttn+Max+Mean + Char-CNN [2,3,4,5,7]` + `LayerNorm` + `MSD` + Train-Only FastText) | `models/cnn_v3/TV4-V3-scratch/V3-BIGRU-GATED-FT/seed_{42,43,44}/best.pt` *(best seed: `seed_44`)* | **`1.39M` / `15.97 MB`** | **`279.0 MB`** | **`0.7900 ± 0.0024`** (`s44 = 0.7926`) | **`0.8032 ± 0.0061`** (`s44 = 0.8103`) | **`0.6652`** (`s44 = 0.6786`) | `0.7255` (`Val 0.7202`) | **`0.0237`** (raw) |
| **2. Mô hình Pretrained Transformer Đơn Tốt nhất** | **`V3-PHOBERT-V2`** (`vinai/phobert-base-v2` + `pyvi` + `Mean+CLS` + `MSD`) | `models/transformers_v3/V3-PHOBERT-V2/seed_{42,43,44}/best.pt` *(best seed: `seed_44`)* | `135.00M` / `515.06 MB` | `3,315.1 MB` | **`0.8133 ± 0.0023`** (`@0.50`) / **`0.8169`** (3s Platt) | **`0.8158 ± 0.0041`** (`@0.50`) / **`0.8255`** (3s Platt) | `0.6899` (`@0.50`) / **`0.7051`** (Platt) | `0.7703` (`@0.50`) / **`0.7845`** (3s avg) | **`0.0146`** (Platt) / **`0.0063`** (Iso) |
| **3. Tổ hợp Thuần Scratch Nội bộ TV4** | **`V3-ENS-SCRATCH-INTRA`** (`0.35 BIGRU-GATED-FT + 0.35 TRI-GATED-FT + 0.20 PYVI-FT + 0.10 GATED-MSD`) | 12 checkpoints `models/cnn_v3/TV4-V3-scratch/*/seed_{42,43,44}/best.pt` | `~227 MB` (12 ckpt) / `~76 MB` (4 ckpt 1s) | `< 310 MB` | **`0.8036`** (`Val-OOF Platt`) | **`0.8194`** (`[0.8063, 0.8333]`) | `0.6928` (`[0.6693, 0.7174]`) | `0.7607` | **`0.0132`** (Platt) |
| **4. Tổ hợp Toàn diện Nội bộ TV4 (Scratch + Transformer)** | **`V3-ENS-TV4-ALL`** (`0.52 PhoBERT-v2 + 0.18 XLM-R + 0.15 BIGRU-GATED-FT + 0.15 TRI-GATED-FT`) | `models/transformers_v3/` + `models/cnn_v3/TV4-V3-scratch/` | `~2.72 GB` (full) | `~5.5 GB` | **`0.8217`** (`@0.50`) / `0.8212` (Platt) | **`0.8351`** (`[0.8212, 0.8481]`) | **`0.7214`** (`[0.6973, 0.7442]`) | `0.7997` | **`0.0106`** (Platt) |
| **5. Tổ hợp Liên thành viên Tốt nhất Toàn Dự án (TV4 V3 + TV2)** | **`V3-ENS-CROSS-VOTING`** (`0.44 PhoBERT-v2 + 0.14 XLM-R + 0.13 BIGRU-GATED-FT + 0.11 TRI-GATED-FT + 0.18 TV2 EXP-N6-C-cal`) | Checkpoints TV4 V3 + TV2 `EXP-N6-C` (`val_repro_max_diff = 0.00e+00`) | `~2.73 GB` (full) | `~5.5 GB` | **`0.8241`** (`Val-OOF Platt`) | **`0.8393`** (`[0.8267, 0.8521]`) | **`0.7283`** (`[0.7061, 0.7504]`) | **`0.8011`** | **`0.0086`** (Platt) |

---

## 3. So sánh Tiến hóa Toàn diện V1 $\to$ V2 $\to$ V3 (Hướng 2 — Cải tiến Có Căn cứ)

![Biểu đồ tiến hóa hiệu năng V1 -> V2 -> V3](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/artifacts/TV4-v3-research/v1_v2_v3_progression.png)

> **Quy tắc tách bạch báo cáo (Master Prompt §4B):** Mọi mô hình **Train-From-Scratch** (`V1`, `V2`, `V3-Scratch`) và **Pretrained Transformer** (`V3-PHOBERT-V2`, `V3-XLMR-BASE`) được đánh nhãn họ rõ ràng và tách biệt để không gây nhầm lẫn giữa lợi ích từ kiến trúc/tiền xử lý thuần nội sinh với lợi ích từ tri thức ngôn ngữ học pretrained quy mô lớn.

| Phiên bản & Mô hình | Họ mô hình | Số seeds | Tiền xử lý & Cấu hình cốt lõi | Tham số | Val Macro-F1 (`@0.50` / Calib) | Val F1-TOXIC | Val ViCTSD | Val PR-AUC | Test Macro-F1 (`@0.50` / Calib) | Test F1-TOXIC | Test PR-AUC | Test ECE |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`[V1] CHAR-02`** | Scratch | 3 | `raw` char `[3,4,5,7]` f=256 | `0.35M` | `0.7550 ± 0.0040` | `0.5817` | — | `0.6316` | `0.7579` / `0.7590` | `0.5861` | `0.6497` | `0.0125` (Platt) |
| **`[V1] CNN-01-WD`** | Scratch | 3 | `basic` Word-CNN `[3]`, `wd=1e-4` | `1.31M` | `0.7705 ± 0.0027` | `0.6058` | — | `0.6751` | `0.7839` / `0.7851` | `0.6290` | `0.6945` | `0.0139` (Platt) |
| **`[V1] HYBRID-RNN-01`** | Scratch | 3 | `raw` BiGRU(`last`) + Char-CNN | `1.96M` | `0.7733 ± 0.0072` | `0.6149` | `0.6542` | `0.6673` | `0.7855` / `0.7854` | `0.6352` | `0.6889` | `0.0112` (Platt) |
| **`[V1] ENS-TV4-INTRA-4`** | Scratch Ens | 3 | Weighted Voting 4 họ V1 (`@0.50`) | `~77 MB` | `0.7855 ± 0.0053` | `0.6313` | — | `0.7073` | `0.7991` / `0.8029`* | `0.6548` | `0.7310` | `0.0271` |
| **`[V1] ENS-CROSS-FROZEN`** | Cross Ens V1 | 3 | TV2 `EXP-N6-C` + TV4 V1 Neural | `~92 MB` | `0.7989 ± 0.0003` | — | `0.6719` | — | `0.8128 ± 0.0019` | `0.6834` | `0.7572` | `0.0282` |
| **`[V2] V2-BIGRU-PREP`** | Scratch | 3 | `v2_word_norm` + `v2_char_norm` + `CosineLR` | `1.24M` | `0.7839 ± 0.0029` | `0.6321` | `0.6520` | `0.6899` | `0.7884` / `0.7898` | `0.6394` | `0.7040` | `0.0079` (Platt) |
| **`[V2] V2-BIGRU-ATTN`** | Scratch | 3 | + `Self-Attention Pooling` + `SpatialDrop=0.15` | `1.31M` | `0.7854 ± 0.0029` | `0.6360` | `0.6609` | `0.6951` | `0.7943` / `0.7937` | `0.6507` | `0.7129` | `0.0091` (Platt) |
| **`[V2] V2-BIGRU-ATTN-MAX`** | Scratch | 3 | + `Attn+Max Pooling` + Char `[2,3,4,5,7]` | `1.33M` | `0.7855 ± 0.0019` | `0.6333` | `0.6582` | `0.6994` | `0.7925` / `0.7958` | `0.6453` | `0.7148` | `0.0108` (Platt) |
| **`[V2] V2-ENS-TV4-INTRA`** | Scratch Ens | 3 | Weighted Voting 4 biến thể V2 | `~56 MB` | `0.7921 ± 0.0013` | — | — | — | `0.7988` / `0.8022` | `0.6646` | `0.7344` | `0.0105` (Platt) |
| **`[V3] V3-BIGRU-GATED-MSD`** | Scratch V3 | 3 | `v3_word_norm`, `GatedAttn+Max+Mean`, `LN`, `MSD`, `LS=0.03` | `1.39M` | `0.7853 ± 0.0026` | `0.6335` | `0.6494` | `0.7050` | `0.7910 ± 0.0046` | `0.6428` | `0.7169` | `0.0369` (raw) |
| **`[V3] V3-BIGRU-PYVI-FT`** | Scratch V3 | 3 | `v3_word_pyvi` + **Train-Only FastText** + `GatedAttn` | `1.95M` | `0.7879 ± 0.0019` | `0.6388` | `0.6368` | `0.7111` | `0.7947 ± 0.0040` | `0.6500` | `0.7232` | `0.0353` (raw) |
| **`[V3] V3-BIGRU-GATED-FT`** | Scratch V3 | 3 | `v3_word_norm` + **Train-Only FastText** + `GatedAttn+MSD` | **`1.39M`** | **`0.7900 ± 0.0024`** | **`0.6420`** | `0.6396` | **`0.7202`** | **`0.8032 ± 0.0061`** | **`0.6652`** | `0.7255` | `0.0237` (raw) |
| **`[V3] V3-TRI-GATED-FT`** | Scratch V3 | 3 | `Word-CNN+BiGRU+Char-CNN` (`Gated Fusion`) + **Train-Only FT** | `1.87M` | **`0.7907 ± 0.0034`** | **`0.6421`** | `0.6516` | `0.7068` | **`0.8023 ± 0.0056`** | `0.6627` | **`0.7270`** | `0.0317` (raw) |
| **`[V3] V3-ENS-SCRATCH-INTRA`** | Scratch Ens V3 | 3 | Weighted Voting 4 biến thể Scratch V3 (`Platt, thr=0.40`) | `~227 MB` | `0.8016` / **`0.8036`** | `0.6652` | `0.6588` | `0.7475` | `0.8177` / **`0.8194`** | `0.6928` | `0.7607` | **`0.0132`** (Platt) |
| **`[V3] V3-XLMR-BASE`** | Pretrained | 1 (`s42`) | `FacebookAI/xlm-roberta-base`, `Mean+CLS`, `MSD`, `len=160` | `278.05M` | **`0.8006`** | `0.6608` | `0.7027` | `0.7393` | `0.7970` | `0.6544` | `0.7446` | `0.0280` (raw) |
| **`[V3] V3-PHOBERT-V2`** | Pretrained | 3 | `vinai/phobert-base-v2`, `pyvi`, `Mean+CLS`, `MSD`, `len=160` | `135.00M` | **`0.8133 ± 0.0023`** / **`0.8169`** (3s Platt) | `0.6851` / `0.6900` | **`0.7276`** / **`0.7336`** | `0.7580` / `0.7722` | **`0.8158 ± 0.0041`** / **`0.8255`** (3s Platt) | `0.6899` / **`0.7051`** | `0.7703` / **`0.7845`** | **`0.0146`** (Platt) / **`0.0063`** (Iso) |
| **`[V3] V3-ENS-TV4-ALL`** | TV4 Hybrid Ens | 3+1 | `PhoBERT-v2 + XLM-R + BIGRU-GATED-FT + TRI-GATED-FT` | `~2.72 GB` | **`0.8217`** / `0.8212` | `0.6972` | `0.7060` | **`0.7886`** | `0.8330` / **`0.8351`** | **`0.7214`** | `0.7997` | **`0.0106`** (Platt) |
| **`[V3] V3-ENS-CROSS-STACKING-LR`** | Cross Stack V3 | 3+1 | `LogisticRegression` Meta-Learner trên 5 nguồn (TV4 + TV2) | `~2.73 GB` | **`0.8235`** (`Val-OOF`) | `0.6994` | `0.7031` | `0.7878` | **`0.8342`** (Frozen) | `0.7179` | **`0.8014`** | `0.0210` |
| **`[V3] V3-ENS-CROSS-VOTING`** | Cross Ens V3 | 3+1 | **Weighted Voting TV4 V3 + TV2 `EXP-N6-C-calibrated` (Platt)** | `~2.73 GB` | `0.8197` / **`0.8241`** | **`0.7020`** | **`0.7112`** | **`0.7884`** | `0.8314` / **`0.8393`** | **`0.7283`** | **`0.8011`** | **`0.0086`** (Platt) |

*\*Ghi chú: Ở V1, `0.8029` thuộc `ENS-TV4-FROZEN-PLATT` (tổ hợp 6 mô hình TV4 V1). So từ baseline đơn V1 (`HYBRID-RNN-01`: `0.7733` Val / `0.7855` Test) tới V3:*
- **Tiến bộ thuần Scratch (không pretraining ngoài):** Mô hình đơn tăng **`+1.74%` Val Macro-F1** (`0.7733` $\to$ `0.7907`), **`+1.77%` Test Macro-F1** (`0.7855` $\to$ `0.8032`), **`+5.29%` Val PR-AUC** (`0.6673` $\to$ `0.7202`) và giảm tham số từ `1.96M` xuống `1.39M`. Tổ hợp thuần Scratch V3 (`V3-ENS-SCRATCH-INTRA`) đạt **`0.8194` Test Macro-F1**, vượt cả Cross-Ensemble V1 có TV2 (`0.8128`)!
- **Tiến bộ có Pretrained Transformer + Cross-Ensemble:** Tăng **`+5.08%` Val Macro-F1** (`0.7733` $\to$ `0.8241`), **`+5.38%` Test Macro-F1** (`0.7855` $\to$ `0.8393`), **`+9.31%` Test F1-TOXIC** (`0.6352` $\to$ `0.7283`), **`+11.22%` Test PR-AUC** (`0.6889` $\to$ `0.8011`), và nâng **`ViCTSD` thêm `+10.55%`** (`0.6542` $\to$ `0.7597` Test / `0.7767` với `PhoBERT-v2` Platt)!

### Kiểm chứng Giả thuyết Từng Vòng Cải tiến V3 (Baseline $\to$ Vấn đề $\to$ Giả thuyết $\to$ Thay đổi $\to$ Kết quả $\to$ Kết luận)

1. **Vòng 1A — Gated Attention + LayerNorm + Multi-Sample Dropout (`V3-BIGRU-GATED-MSD` vs `V2-BIGRU-ATTN-MAX`):**
   - *Vấn đề:* `V2-BIGRU-ATTN-MAX` (`0.7855` Val, `0.6994` PR-AUC) vẫn bị overconfidence và nhạy cảm False Positive với từ lóng.
   - *Kết quả:* `V3-BIGRU-GATED-MSD` đạt Val Macro-F1 tương đương (`0.7853 ± 0.0026`), nhưng cải thiện **Val PR-AUC** (`0.6994` $\to$ `0.7050`), **Val Log-Loss** (`0.3143` $\to$ `0.2917`, giảm mạnh overconfidence) và **Test PR-AUC** (`0.7148` $\to$ `0.7169`).
   - *Kết luận:* Kiến trúc `GatedAttn+Max+Mean + MSD + LS=0.03` cải thiện đáng kể chất lượng xếp hạng xác suất (`PR-AUC` & `Log-Loss`), tạo nền tảng ổn định cho bước khởi tạo embedding tiếp theo.
2. **Vòng 1B — Khởi tạo Train-Only Subword FastText (`V3-BIGRU-GATED-FT` & `V3-TRI-GATED-FT`):**
   - *Vấn đề:* Embedding khởi tạo ngẫu nhiên học kém các từ ghép/lỗi chính tả/từ hiếm xuất hiện ít lần trong `28.2k` mẫu train.
   - *Giả thuyết:* Fit `gensim.models.FastText` (`dim=128, min_n=2..5`, 100% chỉ trên `train.parquet`) truyền biểu diễn hình thái subword ngay từ epoch 1 mà không vi phạm chuẩn train-from-scratch.
   - *Kết quả:* `V3-BIGRU-GATED-FT` bật nhảy **`+0.47%` Val Macro-F1** (`0.7853` $\to$ **`0.7900 ± 0.0024`**), **`+1.52%` Val PR-AUC** (`0.7050` $\to$ **`0.7202`**), **`+1.22%` Test Macro-F1** (`0.7910` $\to$ **`0.8032 ± 0.0061`**), **giảm Raw ECE từ `0.0414` $\to$ `0.0281`** và **hội tụ nhanh hơn 31%** (`180.7s` $\to$ `124.2s`). `V3-TRI-GATED-FT` đạt kỷ lục Scratch **`0.7907 ± 0.0034`** Val (`s43 = 0.7945`, Test `0.8100`).
   - *Kết luận:* **Chấp nhận giả thuyết.** Train-Only FastText là bước đột phá quan trọng nhất cho họ Scratch trong V3, giúp mô hình Scratch đơn vượt mục tiêu `>= 0.790`.
3. **Vòng 1C — Tách từ ghép PyVi cho mô hình Scratch (`V3-BIGRU-PYVI-FT` — Kết quả đối chứng âm tính nhẹ):**
   - *Giả thuyết:* Tách từ bằng `pyvi` giúp BiGRU Scratch nhận diện từ ghép (`vô_học`, `mất_dạy`, `không_phải`).
   - *Kết quả:* Đạt `0.7879 ± 0.0019` Val (`0.7947` Test) — thấp hơn `V3-BIGRU-GATED-FT` (`0.7900` Val / `0.8032` Test) và làm phình kích thước embedding từ `1.39M` lên `1.95M` tham số do tăng số lượng token hiếm trên tập `28.2k` mẫu (trên `ViCTSD` giảm nhẹ từ `0.6396` $\to$ `0.6368`).
   - *Kết luận:* **Bác bỏ cho mô hình Scratch đơn**, nhưng vẫn đóng góp đa dạng hóa hữu ích (`trọng số 0.20`) trong `V3-ENS-SCRATCH-INTRA`. Ngược lại, `pyvi` hoạt động cực kỳ hiệu quả khi kết hợp với từ điển pretrained `PhoBERT-v2`.
4. **Vòng 2 — Pretrained `vinai/phobert-base-v2` (`V3-PHOBERT-V2`):**
   - *Vấn đề:* Toàn bộ mô hình Scratch chạm trần ngữ nghĩa trên `ViCTSD` (`~0.64–0.66` Val Macro-F1, `Recall_TOXIC` chỉ ~`23%–34%`) vì không hiểu mỉa mai, phủ định, và ngữ cảnh tin tức gián tiếp.
   - *Kết quả:* `V3-PHOBERT-V2` (3 seeds) tăng vọt lên **`0.8133 ± 0.0023` Val Macro-F1**, **`0.7276 ± 0.0080` Val ViCTSD** (**`+7.60%` so với `V3-TRI-GATED-FT`**!), **`0.7580` Val PR-AUC**, và **`0.7481` Test ViCTSD** (`0.7767` sau Platt Calibration). Peak VRAM chỉ **`3.32 GB`** trên RTX 4060 8GB.
   - *Kết luận:* **Chấp nhận mạnh mẽ giả thuyết.** Pretrained Transformer tiếng Việt giải quyết triệt để nút cổ chai ngữ nghĩa `ViCTSD`.
5. **Vòng 3 — Pretrained `FacebookAI/xlm-roberta-base` (`V3-XLMR-BASE`):**
   - *Giả thuyết:* Tokenizer SentencePiece đa ngôn ngữ (không phụ thuộc `pyvi`) tạo đa dạng lỗi cao đối với `PhoBERT-v2` trên teencode/không dấu/code-switching.
   - *Kết quả (`seed_42`):* Đạt **`0.8006` Val Macro-F1** (`ViCTSD = 0.7027`), thấp hơn `PhoBERT-v2` (`0.8143`) nhưng có **`6.40%` bất đồng dự đoán (`387` mẫu, `179` mẫu chỉ XLM-R đoán đúng)**, giúp đẩy `Oracle Macro-F1` lên `0.8689` và đóng góp hệ số stacking dương mạnh `+1.2622`.
   - *Kết luận:* **Chấp nhận vai trò bổ trợ đa dạng hóa trong Ensemble** (không thay thế `PhoBERT-v2` làm mô hình đơn vì nặng gấp đôi `1.06 GB` và VRAM `5.50 GB`).

---

## 4. Bảng Kết quả Chi tiết Từng Seed cho Toàn bộ 16 Runs Chính thức V3

> Tất cả metrics trong bảng này được tính ở ngưỡng mặc định **`@0.50` chưa hiệu chuẩn** trên tập đầy đủ `val.parquet` (`n = 6,048`) và `test.parquet` (`n = 6,048`).

| Biến thể V3 | Seed | Val Macro-F1 | Val F1-TOXIC (`P / R`) | Val ViCTSD | Val ViHSD | Val PR-AUC | Val Raw ECE | Val Brier / LogLoss | Test Macro-F1 | Test F1-TOXIC (`P / R`) | Test ViCTSD | Test ViHSD | Test PR-AUC | Thời gian / Peak VRAM / Ckpt |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`V3-BIGRU-GATED-MSD`** | `42` | `0.7881` | `0.6405` (`0.674 / 0.610`) | `0.6515` | `0.8113` | `0.7037` | `0.0515` | `0.0868 / 0.3095` | `0.7957` | `0.6527` (`0.695 / 0.615`) | `0.6656` | `0.8178` | `0.7190` | `220.1s` / `278.1 MB` / `15.97 MB` |
| **`V3-BIGRU-GATED-MSD`** | `43` | `0.7860` | `0.6361` (`0.680 / 0.598`) | `0.6568` | `0.8075` | `0.7060` | `0.0402` | `0.0841 / 0.2876` | `0.7927` | `0.6470` (`0.697 / 0.604`) | `0.6634` | `0.8147` | `0.7095` | `179.9s` / `296.6 MB` / `15.97 MB` |
| **`V3-BIGRU-GATED-MSD`** | `44` | `0.7819` | `0.6240` (`0.738 / 0.540`) | `0.6400` | `0.8036` | `0.7053` | `0.0324` | `0.0813 / 0.2780` | `0.7847` | `0.6285` (`0.749 / 0.541`) | `0.6082` | `0.8124` | `0.7220` | `142.1s` / `277.9 MB` / `15.97 MB` |
| *Mean ± Std (`n=3`)* | — | *`0.7853 ± 0.0026`* | *`0.6335 ± 0.0070`* | *`0.6494`* | *`0.8075`* | *`0.7050`* | *`0.0414`* | *`0.0841 / 0.2917`* | *`0.7910 ± 0.0046`* | *`0.6428`* | *`0.6457`* | *`0.8150`* | *`0.7169`* | *`180.7s` / `284.2 MB`* |
| **`V3-BIGRU-GATED-FT`** | `42` | `0.7868` | `0.6347` (`0.715 / 0.571`) | `0.6301` | `0.8116` | `0.7196` | `0.0191` | `0.0787 / 0.2649` | `0.7954` | `0.6504` (`0.720 / 0.593`) | `0.6643` | `0.8174` | `0.7224` | `88.9s` / `278.1 MB` / `15.97 MB` |
| **`V3-BIGRU-GATED-FT`** | `43` | `0.7906` | `0.6433` (`0.695 / 0.599`) | `0.6446` | `0.8136` | `0.7161` | `0.0300` | `0.0811 / 0.2755` | `0.8040` | `0.6667` (`0.712 / 0.627`) | `0.6880` | `0.8225` | `0.7218` | `140.9s` / `280.9 MB` / `15.97 MB` |
| **`V3-BIGRU-GATED-FT`** | `44` | **`0.7926`** | **`0.6481`** (`0.682 / 0.617`) | `0.6441` | `0.8179` | **`0.7248`** | `0.0352` | `0.0808 / 0.2720` | **`0.8103`** | **`0.6786`** (`0.709 / 0.650`) | **`0.7136`** | `0.8267` | `0.7325` | `142.9s` / `277.9 MB` / `15.97 MB` |
| *Mean ± Std (`n=3`)* | — | ***`0.7900 ± 0.0024`*** | ***`0.6420 ± 0.0055`*** | *`0.6396`* | *`0.8144`* | ***`0.7202`*** | ***`0.0281`*** | ***`0.0802 / 0.2708`*** | ***`0.8032 ± 0.0061`*** | ***`0.6652`*** | *`0.6886`* | *`0.8222`* | *`0.7255`* | ***`124.2s` / `279.0 MB`*** |
| **`V3-BIGRU-PYVI-FT`** | `42` | `0.7852` | `0.6360` (`0.664 / 0.610`) | `0.6489` | `0.8101` | `0.6890` | `0.0629` | `0.0903 / 0.3355` | `0.7920` | `0.6475` (`0.677 / 0.620`) | `0.6819` | `0.8121` | `0.7103` | `207.2s` / `284.3 MB` / `22.30 MB` |
| **`V3-BIGRU-PYVI-FT`** | `43` | `0.7889` | `0.6373` (`0.734 / 0.563`) | `0.6281` | `0.8146` | `0.7198` | `0.0119` | `0.0773 / 0.2605` | `0.7918` | `0.6412` (`0.755 / 0.557`) | `0.6566` | `0.8140` | `0.7310` | `87.2s` / `283.4 MB` / `22.30 MB` |
| **`V3-BIGRU-PYVI-FT`** | `44` | `0.7896` | `0.6431` (`0.676 / 0.613`) | `0.6333` | `0.8188` | `0.7244` | `0.0464` | `0.0821 / 0.2815` | `0.8005` | `0.6612` (`0.700 / 0.627`) | `0.6945` | `0.8200` | `0.7283` | `137.8s` / `283.9 MB` / `22.30 MB` |
| *Mean ± Std (`n=3`)* | — | *`0.7879 ± 0.0019`* | *`0.6388 ± 0.0031`* | *`0.6368`* | *`0.8145`* | *`0.7111`* | *`0.0404`* | *`0.0832 / 0.2925`* | *`0.7947 ± 0.0040`* | *`0.6500`* | *`0.6776`* | *`0.8154`* | *`0.7232`* | *`144.1s` / `283.9 MB`* |
| **`V3-TRI-GATED-FT`** | `42` | `0.7863` | `0.6305` (`0.765 / 0.536`) | `0.6316` | `0.8115` | `0.6929` | `0.0384` | `0.0815 / 0.3009` | `0.7970` | `0.6498` (`0.772 / 0.561`) | `0.6645` | `0.8179` | `0.7171` | `159.3s` / `294.2 MB` / `21.41 MB` |
| **`V3-TRI-GATED-FT`** | `43` | **`0.7945`** | **`0.6489`** (`0.715 / 0.594`) | `0.6534` | `0.8171` | **`0.7260`** | `0.0309` | `0.0778 / 0.2645` | **`0.8100`** | **`0.6767`** (`0.727 / 0.633`) | `0.6817` | **`0.8313`** | **`0.7441`** | `117.8s` / `314.9 MB` / `21.41 MB` |
| **`V3-TRI-GATED-FT`** | `44` | `0.7914` | `0.6470` (`0.671 / 0.625`) | `0.6698` | `0.8126` | `0.7014` | `0.0305` | `0.0829 / 0.2828` | `0.8000` | `0.6617` (`0.685 / 0.640`) | `0.6954` | `0.8186` | `0.7199` | `153.1s` / `294.2 MB` / `21.41 MB` |
| *Mean ± Std (`n=3`)* | — | ***`0.7907 ± 0.0034`*** | ***`0.6421 ± 0.0083`*** | *`0.6516`* | *`0.8138`* | *`0.7068`* | *`0.0333`* | *`0.0807 / 0.2827`* | *`0.8023 ± 0.0056`* | *`0.6627`* | *`0.6805`* | *`0.8226`* | ***`0.7270`*** | *`143.4s` / `301.1 MB`* |
| **`V3-PHOBERT-V2`** *(Pretrained)* | `42` | `0.8143` | `0.6857` (`0.713 / 0.661`) | **`0.7358`** | `0.8287` | `0.7591` | `0.0303` | `0.0735 / 0.2479` | `0.8135` | `0.6845` (`0.708 / 0.663`) | `0.7506` | `0.8243` | `0.7725` | `565.1s` / `3,315.3 MB` / `515.06 MB` |
| **`V3-PHOBERT-V2`** *(Pretrained)* | `43` | `0.8101` | `0.6819` (`0.669 / 0.695`) | `0.7303` | `0.8227` | `0.7532` | `0.0356` | `0.0763 / 0.2498` | `0.8124` | `0.6865` (`0.667 / 0.708`) | `0.7414` | `0.8234` | `0.7682` | `562.9s` / `3,319.5 MB` / `515.06 MB` |
| **`V3-PHOBERT-V2`** *(Pretrained)* | `44` | **`0.8155`** | **`0.6875`** (`0.717 / 0.661`) | `0.7168` | **`0.8320`** | **`0.7616`** | `0.0462` | `0.0766 / 0.2674` | **`0.8216`** | **`0.6986`** (`0.718 / 0.681`) | **`0.7525`** | **`0.8333`** | `0.7702` | `566.8s` / `3,310.6 MB` / `515.06 MB` |
| *Mean ± Std (`n=3`)* | — | ***`0.8133 ± 0.0023`*** | ***`0.6851 ± 0.0023`*** | ***`0.7276`*** | ***`0.8278`*** | ***`0.7580`*** | *`0.0374`* | ***`0.0755 / 0.2550`*** | ***`0.8158 ± 0.0041`*** | ***`0.6899`*** | ***`0.7481`*** | ***`0.8270`*** | ***`0.7703`*** | *`564.9s` / `3,315.1 MB`* |
| **`V3-XLMR-BASE`** *(Pretrained, 1s)* | `42` | `0.8006` | `0.6608` (`0.708 / 0.619`) | `0.7027` | `0.8173` | `0.7393` | `0.0330` | `0.0766 / 0.2618` | `0.7970` | `0.6544` (`0.705 / 0.611`) | `0.7219` | `0.8094` | `0.7446` | `642.7s` / `5,498.1 MB` / `1,060.74 MB` |

---

## 5. Đa dạng Lỗi (Error Diversity), Kết quả Ensemble & Kiểm định Thống kê (Bootstrap CI + McNemar)

### 5.1. Ma trận Đa dạng Lỗi Từng Cặp trên Validation (`seed_42`, `n = 6,048`)

![Bản đồ nhiệt đa dạng lỗi và tương quan xác suất V3](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/artifacts/TV4-v3-research/v3_error_diversity_heatmap.png)

| Cặp Mô hình (Model A vs Model B) | Tỷ lệ Bất đồng (`Disagreement`) | Chỉ A đúng / Chỉ B đúng | Double-Fault Rate | Cohen's $\kappa$ | Tương quan Xác suất ($r$) | Oracle Macro-F1 | Oracle F1-TOXIC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`V3-PHOBERT-V2` vs `V3-TRI-GATED-FT`** | **`7.87%`** (`476` mẫu) | `249` / `227` | **`5.90%`** | `0.6526` | **`0.7775`** | **`0.8809`** | **`0.7963`** |
| **`V3-PHOBERT-V2` vs `EXP-N6-C-calibrated` (TV2)** | **`7.80%`** (`472` mẫu) | `257` / `215` | `6.10%` | `0.6625` | `0.8247` | **`0.8768`** | **`0.7893`** |
| `V3-XLMR-BASE` vs `V3-TRI-GATED-FT` | `7.66%` (`463` mẫu) | `228` / `235` | `6.25%` | `0.6520` | `0.7795` | `0.8715` | `0.7795` |
| `V3-XLMR-BASE` vs `EXP-N6-C-calibrated` (TV2) | `7.49%` (`453` mẫu) | `233` / `220` | `6.50%` | `0.6668` | `0.8182` | `0.8671` | `0.7722` |
| `V3-XLMR-BASE` vs `V3-BIGRU-GATED-FT` | `7.34%` (`444` mẫu) | `232` / `212` | `6.63%` | `0.6823` | `0.8274` | `0.8661` | `0.7710` |
| `V3-PHOBERT-V2` vs `V3-BIGRU-GATED-FT` | `7.03%` (`425` mẫu) | `237` / `188` | `6.55%` | `0.7040` | `0.8281` | `0.8693` | `0.7770` |
| `V3-PHOBERT-V2` vs `V3-XLMR-BASE` | `6.40%` (`387` mẫu) | `208` / `179` | `6.70%` | `0.7399` | `0.8648` | `0.8689` | `0.7771` |
| `V3-BIGRU-GATED-FT` vs `EXP-N6-C-calibrated` | `5.31%` (`321` mẫu) | `157` / `164` | `7.75%` | `0.7536` | `0.8990` | `0.8389` | `0.7230` |
| `V3-BIGRU-GATED-FT` vs `V3-TRI-GATED-FT` | `5.11%` (`309` mẫu) | `141` / `168` | `7.69%` | `0.7573` | `0.8733` | `0.8394` | `0.7234` |
| `V3-TRI-GATED-FT` vs `EXP-N6-C-calibrated` | `5.03%` (`304` mẫu) | `162` / `142` | `7.67%` | `0.7532` | `0.8618` | `0.8377` | `0.7198` |

**Nhận xét cốt lõi về Đa dạng hóa:**
- Cặp khác họ kiến trúc **`V3-PHOBERT-V2` (Transformer ngữ nghĩa) vs `V3-TRI-GATED-FT` (Scratch Word-CNN + BiGRU + Char-CNN)** đạt mức đa dạng cao nhất toàn bộ thí nghiệm: tương quan xác suất thấp nhất (`r = 0.7775`), tỷ lệ lỗi kép thấp nhất (`5.90%`), và **`Oracle Macro-F1 = 0.8809`** (`227` mẫu `PhoBERT-v2` đoán sai nhưng `V3-TRI-GATED-FT` đoán đúng!).
- Tương tự, cặp **`V3-PHOBERT-V2` vs `EXP-N6-C-calibrated` (TF-IDF tuyến tính của TV2)** có tới **`215` mẫu** TF-IDF đoán đúng khi `PhoBERT-v2` sai (`Oracle = 0.8768`). Điều này chứng minh bằng chứng khoa học vững chắc lý do tại sao tổ hợp 3 họ (**Pretrained Transformer + Neural Scratch + Sparse TF-IDF Linear**) vượt trội mọi mô hình đơn lẻ.

### 5.2. So sánh Chi tiết 4 Cấu hình Ensemble V3 (`Val-5Fold-OOF` & `Historical Benchmark Test`)

| Tên Ensemble | Thành phần & Trọng số (hoặc Meta-Coefs) | Tầng Đánh giá | Macro-F1 (`95% CI` trên Test) | F1-TOXIC (`P / R`) | ViCTSD / ViHSD | PR-AUC | ECE | Brier / LogLoss | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`V3-ENS-SCRATCH-INTRA`** | `0.35 BIGRU-GATED-FT + 0.35 TRI-GATED-FT + 0.20 PYVI-FT + 0.10 GATED-MSD` | Val `@0.50` | `0.8016` | `0.6597` (`0.749 / 0.589`) | `0.6504` / `0.8248` | `0.7478` | `0.0157` | `0.0737 / 0.2492` | `[[4894, 190], [396, 568]]` |
| *(Thuần Scratch, 0 Pretrained)* | Platt `(a=0.9525, b=-0.2213)`, `thr=0.40` | **Val-5Fold-OOF Platt** | **`0.8036`** | `0.6652` (`0.724 / 0.615`) | `0.6588` / `0.8267` | `0.7475` | **`0.0093`** | `0.0731 / 0.2483` | `[[4858, 226], [371, 593]]` |
| | | **Test Frozen Platt** | **`0.8194`** (`[0.8063, 0.8333]`) | `0.6928` (`0.741 / 0.650`) | **`0.7113`** / `0.8374` | `0.7607` | **`0.0132`** | `0.0695 / 0.2370` | `[[4865, 219], [337, 627]]` |
| **`V3-ENS-TV4-ALL`** | `0.52 PhoBERT-v2 + 0.18 XLM-R + 0.15 BIGRU-GATED-FT + 0.15 TRI-GATED-FT` | Val `@0.50` | **`0.8217`** | `0.6959` (`0.760 / 0.642`) | `0.6953` / `0.8417` | `0.7891` | `0.0194` | `0.0664 / 0.2248` | `[[4888, 196], [345, 619]]` |
| *(Nội bộ TV4 V3)* | Platt `(a=1.0734, b=-0.1925)`, `thr=0.40` | **Val-5Fold-OOF Platt** | **`0.8212`** | `0.6972` (`0.725 / 0.671`) | `0.7060` / `0.8399` | **`0.7886`** | **`0.0079`** | `0.0661 / 0.2223` | `[[4839, 245], [317, 647]]` |
| | | **Test Frozen Platt** | **`0.8351`** (`[0.8212, 0.8481]`) | **`0.7214`** (`0.742 / 0.702`) | **`0.7629`** / `0.8467` | `0.7997` | **`0.0106`** | `0.0629 / 0.2116` | `[[4848, 236], [287, 677]]` |
| **`V3-ENS-CROSS-VOTING`** | `0.44 PhoBERT-v2 + 0.14 XLM-R + 0.13 BIGRU-GATED-FT + 0.11 TRI-GATED-FT + 0.18 TV2 EXP-N6-C` | Val `@0.50` | `0.8197` | `0.6920` (`0.764 / 0.633`) | `0.6987` / `0.8386` | `0.7887` | `0.0187` | `0.0661 / 0.2245` | `[[4895, 189], [354, 610]]` |
| *(🏆 Tốt nhất Toàn Dự án)* | Platt `(a=1.1230, b=-0.1209)`, `thr=0.40` | **Val-5Fold-OOF Platt** | **`0.8241`** | **`0.7020`** (`0.733 / 0.673`) | **`0.7112`** / `0.8425` | `0.7884` | **`0.0067`** | **`0.0660 / 0.2215`** | `[[4848, 236], [315, 649]]` |
| | | **Test Frozen Platt** | **`0.8393`** (`[0.8267, 0.8521]`) | **`0.7283`** (`0.752 / 0.706`) | **`0.7597`** / **`0.8522`** | **`0.8011`** | **`0.0086`** | **`0.0624 / 0.2104`** | `[[4859, 225], [283, 681]]` |
| **`V3-ENS-CROSS-STACKING-LR`** | `LR(C=1.0)` trên 5 xác suất; coefs `[2.7858, 1.2622, 0.0721, 1.0605, 1.5496]`, intercept `-3.5491`, `thr=0.40` | **Val-5Fold-OOF** | `0.8235` | `0.6994` (`0.756 / 0.650`) | `0.7031` / `0.8428` | `0.7878` | `0.0206` | `0.0677 / —` | `[[4882, 202], [337, 627]]` |
| *(Diagnostic Val-OOF Stacking)* | | **Test Frozen** | `0.8342` (`[0.8201, 0.8472]`) | `0.7179` (`0.769 / 0.673`) | `0.7573` / `0.8466` | **`0.8014`** | `0.0210` | `0.0639 / 0.2224` | `[[4889, 195], [315, 649]]` |

- **Khám phá từ trọng số học tự động của Stacking LR (`meta_coefs`):** `PhoBERT-v2` (`+2.7858`) $>$ TV2 `EXP-N6-C` (`+1.5496`) $>$ `XLM-R` (`+1.2622`) $>$ `V3-TRI-GATED-FT` (`+1.0605`) $\gg$ `V3-BIGRU-GATED-FT` (`+0.0721`). Ba họ bổ trợ (`TF-IDF`, `XLM-R`, `Tri-Hybrid Scratch`) đều nhận trọng số dương rất mạnh, khẳng định tính phi dư thừa của chúng. Tuy nhiên, `V3-ENS-CROSS-VOTING` (Weighted Voting + Platt) vượt `LR Stacking` cả về Macro-F1 (`0.8393` vs `0.8342`), F1-TOXIC (`0.7283` vs `0.7179`), Log-Loss (`0.2104` vs `0.2224`) lẫn ECE (`0.0086` vs `0.0210`), do đó **Weighted Voting + Platt là phương pháp tổ hợp được chọn chính thức**.

### 5.3. Kiểm định Ý nghĩa Thống kê McNemar & Bootstrap 95% CI trên Historical Test (`n = 6,048`)

| So sánh Cặp (Model A vs Model B) | Chỉ A đúng | Chỉ B đúng | Hiệu ròng (`Net Gain`) | $\chi^2_{\text{cc}}$ | `p-value` | Có ý nghĩa thống kê ($\alpha = 0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`V3-ENS-CROSS-VOTING-PLATT` vs `EXP-N6-C-calibrated` (TV2)** | `212` | `140` | **`+72`** | `14.3210` | **`0.000154`** | **CÓ** ($p < 0.001$) |
| **`V3-ENS-CROSS-VOTING-PLATT` vs `V3-PHOBERT-V2` (3s)** | `105` | `60` | **`+45`** | `11.7333` | **`0.000614`** | **CÓ** ($p < 0.001$) |
| **`V3-ENS-CROSS-STACKING-LR` vs `V3-PHOBERT-V2` (3s)** | `113` | `70` | **`+43`** | `9.6393` | **`0.001905`** | **CÓ** ($p < 0.01$) |
| **`V3-ENS-TV4-ALL-PLATT` vs `V3-PHOBERT-V2` (3s)** | `90` | `60` | **`+30`** | `5.6067` | **`0.017892`** | **CÓ** ($p < 0.05$) |
| `V3-ENS-SCRATCH-INTRA-PLATT` vs `V3-BIGRU-GATED-FT` (3s) | `55` | `38` | `+17` | `2.7527` | `0.097091` | Chưa đạt ngưỡng $0.05$ (xu hướng dương) |
| `V3-PHOBERT-V2` (3s) vs `V3-BIGRU-GATED-FT` (3s) | `234` | `214` | `+20` | `0.8058` | `0.369364` | Accuracy tương đương tổng thể, nhưng `PhoBERT-v2` vượt mạnh trên F1-TOXIC (`+0.025`) và `ViCTSD` (`+0.060`) |

- **Khoảng tin cậy Bootstrap 95% (`1,000` lần lấy mẫu lại) trên Historical Test (`Macro-F1` / `F1-TOXIC`):**
  - `EXP-N6-C-calibrated` (TV2): `[0.7838, 0.8142]` / `[0.6264, 0.6806]`
  - `V3-BIGRU-GATED-FT` (3s avg): `[0.8003, 0.8270]` / `[0.6582, 0.7064]`
  - `V3-ENS-SCRATCH-INTRA-PLATT`: `[0.8063, 0.8333]` / `[0.6693, 0.7174]`
  - `V3-PHOBERT-V2` (3s avg): `[0.8131, 0.8396]` / `[0.6840, 0.7306]`
  - `V3-ENS-TV4-ALL-PLATT`: `[0.8212, 0.8481]` / `[0.6973, 0.7442]`
  - **`V3-ENS-CROSS-VOTING-PLATT`**: **`[0.8267, 0.8521]`** / **`[0.7061, 0.7504]`** (cận dưới 95% CI đã vượt xa cận trên của TV2 `EXP-N6-C`!).

---

## 6. Phân tích Tổng quát hóa Theo Miền Dữ liệu: `ViCTSD` vs `ViHSD`

![So sánh hiệu năng theo miền ViCTSD và ViHSD](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/artifacts/TV4-v3-research/v3_source_comparison.png)

| Mô hình / Tổ hợp | Val `ViHSD` (`n=4,559`) Macro-F1 | Val `ViCTSD` (`n=1,489`) Macro-F1 | Chênh lệch Miền (`ViHSD - ViCTSD`) | Test `ViHSD` (`n=4,558`) Macro-F1 | Test `ViCTSD` (`n=1,490`) Macro-F1 | Test `ViCTSD` F1-TOXIC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `V3-BIGRU-GATED-FT` (Scratch 3s, `@0.50`) | `0.8144` | `0.6396` | `0.1748` | `0.8222` | `0.6886` | `0.427` *(avg)* |
| `V3-TRI-GATED-FT` (Scratch 3s, `@0.50`) | `0.8138` | `0.6516` | `0.1622` | `0.8226` | `0.6805` | `0.415` *(avg)* |
| `V3-ENS-SCRATCH-INTRA` (Platt) | `0.8267` | `0.6588` | `0.1679` | `0.8374` | **`0.7113`** | `0.4733` |
| `V3-XLMR-BASE` (Pretrained s42, `@0.50`) | `0.8173` | `0.7027` | `0.1146` | `0.8094` | `0.7219` | `0.491` |
| **`V3-PHOBERT-V2` (Pretrained 3s, `@0.50`)** | `0.8278` | **`0.7276`** | **`0.1002`** | `0.8270` | **`0.7481`** | `0.543` *(avg)* |
| **`V3-PHOBERT-V2` (3s avg, Platt `thr=0.40`)** | `0.8308` | **`0.7336`** | **`0.0972`** | `0.8330` | **`0.7767`** | **`0.594`** |
| **`V3-ENS-TV4-ALL` (Platt `thr=0.40`)** | `0.8399` | `0.7060` | `0.1339` | `0.8467` | **`0.7629`** | `0.5693` |
| **`V3-ENS-CROSS-VOTING` (Platt `thr=0.40`)** | **`0.8425`** | **`0.7112`** | `0.1313` | **`0.8522`** | **`0.7597`** | `0.5630` |

**Phát hiện quan trọng về Miền `ViCTSD` (Bình luận Báo chí, tỷ lệ `TOXIC` chỉ `11.08%`, câu dài, mỉa mai gián tiếp):**
1. Các mô hình **Train-From-Scratch** (`V2` và `V3-Scratch`) rất mạnh trên `ViHSD` (`0.814–0.837`), nhưng bị giới hạn ở `0.640–0.659` trên Validation `ViCTSD` vì bình luận độc hại trên báo chí thường **không chứa từ chửi thề hiển hiện** (ví dụ công kích mỉa mai, quy chụp vùng miền/giới tính bằng ngôn từ lịch sự).
2. **`V3-PHOBERT-V2` thu hẹp gần một nửa khoảng cách giữa hai miền** (`0.175` $\to$ **`0.097`**), nâng Validation `ViCTSD` lên **`0.7336`** và Historical Test `ViCTSD` lên tới **`0.7767`** (Platt).
3. Trong `V3-ENS-CROSS-VOTING`, việc kết hợp `PhoBERT-v2` với Scratch và TV2 TF-IDF giúp đẩy `ViHSD` lên đỉnh cao nhất (**`0.8425` Val / `0.8522` Test**) trong khi vẫn giữ `ViCTSD` vượt mục tiêu (**`0.7112` Val / `0.7597` Test**). Nếu triển khai cho riêng nền tảng báo chí thuần `ViCTSD`, tăng trọng số `PhoBERT-v2` lên `0.65–0.75` sẽ tối ưu hơn nữa.

---

## 7. Phân tích Hiệu chuẩn Xác suất (`Platt` vs `Temperature` vs `Isotonic`) & Reliability Diagram

![Biểu đồ Reliability Diagram và so sánh hiệu chuẩn xác suất V3](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/artifacts/TV4-v3-research/v3_calibration_reliability.png)

### So sánh 3 Phương pháp Hiệu chuẩn trên `V3-PHOBERT-V2` (Trung bình xác suất 3 seeds)

| Phương pháp Hiệu chuẩn | Tham số Đã học | Val-5Fold-OOF Macro-F1 (`thr=0.40`) | Val-OOF ECE | Val-OOF Brier / LogLoss | Test Frozen Macro-F1 (`thr=0.40`) | Test F1-TOXIC | Test ECE | Test Brier / LogLoss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Single-seed Raw Mean (`@0.50`)* | *(Chưa hiệu chuẩn)* | `0.8133` | `0.0374` | `0.0755 / 0.2550` | `0.8158` | `0.6899` | `0.0368` | — |
| **Platt Scaling (Sigmoid)** | `coef=0.9070, intercept=-0.4422` | **`0.8169`** | **`0.0113`** | `0.0690 / 0.2330` | `0.8255` | `0.7051` | **`0.0146`** | `0.0663 / 0.2232` |
| **Temperature Scaling** | `T = 1.0030` | `0.8162` | `0.0273` | `0.0712 / 0.2382` | `0.8193` | `0.6990` | `0.0272` | `0.0683 / 0.2284` |
| **Isotonic Regression** | Non-parametric step-wise | `0.8152` | **`0.0057`** | **`0.0686 / 0.2306`** | **`0.8264`** | **`0.7071`** | **`0.0063`** | **`0.0660 / 0.2244`** |

**Kết luận Hiệu chuẩn:**
- **Temperature Scaling (`T = 1.0030`) không đủ mạnh** cho mô hình nhị phân mất cân bằng lớp (`15.9% TOXIC`) có `label_smoothing=0.03`, vì nó chỉ co giãn độ dốc mà **không dịch chuyển bias (`intercept`)** nên ECE vẫn dừng ở `0.0273`.
- **Platt Scaling** có cả tham số độ dốc lẫn dịch chuyển (`a, b`), vừa bảo toàn 100% thứ hạng PR-AUC (`0.7845` Test), vừa hạ ECE xuống **`0.0113`–`0.0146`** trên `PhoBERT-v2` và xuống **`0.0067` (Val-OOF) / `0.0086` (Frozen Test)** trên `V3-ENS-CROSS-VOTING` — an toàn, mượt và ít overfit nhất khi triển khai API.
- **Isotonic Regression** đạt ECE thấp nhất trên `PhoBERT-v2` (`0.0057` Val / `0.0063` Test), nhưng do hàm bậc thang phân đoạn nên làm giảm nhẹ `PR-AUC` (`0.7845` $\to$ `0.7668`); khuyến nghị dùng **Platt Scaling** cho production API để bảo toàn xếp hạng xác suất liên tục.

---

## 8. Ma trận Nhầm lẫn & Phân tích Lỗi Theo Hiện tượng Ngôn ngữ (`val.parquet`, `n = 6,048`)

### 8.1. Diễn tiến Giảm Tổng Lỗi Từ V2 $\to$ V3 Scratch $\to$ V3 Pretrained $\to$ V3 Cross-Ensemble

| Mô hình | Tổng Lỗi / `6,048` | Tỷ lệ Lỗi | False Negatives (`FN`) | False Positives (`FP`) | Ghi chú Ma trận Nhầm lẫn `[[TN, FP], [FN, TP]]` |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`[V2] V2-BIGRU-ATTN-MAX` (`s44`)** | `660` | `10.91%` | `378` | `282` | Baseline V2 Scratch tốt nhất trước V3 |
| **`[V3] V3-BIGRU-GATED-FT` (3s avg)** | **`617`** (`-43` lỗi) | `10.20%` | `387` | **`230`** (`-52` FP!) | GatedAttn + FastText giảm mạnh báo động giả (`FP`) |
| **`[V3] V3-PHOBERT-V2` (3s avg)** | **`573`** (`-87` lỗi) | `9.47%` | **`317`** (`-70` FN!) | `256` | Pretrained PhoBERT-v2 kéo giảm mạnh bỏ sót (`FN`) |
| **`[V3] V3-ENS-CROSS-VOTING` (Val-OOF Platt)** | **`551`** (`-109` lỗi vs V2!) | **`9.11%`** | **`315`** | **`236`** | `[[4848, 236], [315, 649]]` — cân bằng tốt nhất cả FN & FP! |
| *`[V3] V3-ENS-CROSS-VOTING` (Historical Test Platt)* | ***`508`*** | ***`8.40%`*** | ***`283`*** | ***`225`*** | *`[[4859, 225], [283, 681]]` — tỷ lệ đúng đạt `91.60%`!* |

### 8.2. Phân rã Tỷ lệ Lỗi Theo 8 Hiện tượng Ngôn ngữ Tiếng Việt trên Validation (`n = 6,048`)

| Hiện tượng Ngôn ngữ | Số mẫu (`n`) | `V3-BIGRU-GATED-FT` (Scratch) | `V3-PHOBERT-V2` (Pretrained) | `V3-ENS-CROSS-PLATT` (Cross-Ens) | Nhận xét Khoa học & Điểm yếu Còn tồn đọng |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`has_quote`** *(Trích dẫn / tường thuật)* | `192` | `17.71%` (`20 FN, 14 FP`) | **`14.06%`** (`12 FN, 15 FP`) | `16.67%` (`16 FN, 16 FP`) | `PhoBERT-v2` phân biệt tốt nhất lời trích dẫn không độc hại; nhóm khó nhất vẫn còn lại do thiếu ngữ cảnh bài báo gốc. |
| **`non_nfc`** *(Unicode tổ hợp gốc)* | `143` | `16.78%` (`17 FN, 7 FP`) | **`14.69%`** (`13 FN, 8 FP`) | `15.38%` (`12 FN, 10 FP`) | Dù đã chuẩn hóa `NFC` trong tiền xử lý, tập con này chứa nhiều văn bản viết ẩu/nhiễu nguồn cũ nên tỷ lệ lỗi nền cao. |
| **`long_ge_30w`** *(Bình luận dài $\ge 30$ từ)* | `820` | `16.10%` (`85 FN, 47 FP`) | **`12.80%`** (`56 FN, 49 FP`) | `13.54%` (`61 FN, 50 FP`) | `PhoBERT-v2` giảm **`-3.30%`** lỗi (`-29 FN`) trên bình luận dài nhờ Self-Attention toàn cục (`max_length=160`). |
| **`has_teencode`** *(Teencode / viết tắt)* | `633` | `14.69%` (`58 FN, 35 FP`) | **`12.16%`** (`45 FN, 32 FP`) | `12.64%` (`47 FN, 33 FP`) | Giảm **`-2.53%`** lỗi so với Scratch (và `-4.27%` so với V2 `16.43%`). |
| **`has_negation`** *(Câu chứa phủ định)* | `1,336` | `13.70%` (`129 FN, 54 FP`) | **`11.60%`** (`91 FN, 64 FP`) | `11.90%` (`99 FN, 60 FP`) | `PhoBERT-v2` giảm **`-38 FN`** ở các câu phủ định/mỉa mai gián tiếp (`-3.22%` so với V2 `14.82%`). |
| **`no_accent`** *(Không dấu tiếng Việt)* | `281` | `13.17%` (`17 FN, 20 FP`) | `11.74%` (`20 FN, 13 FP`) | **`11.03%`** (`14 FN, 17 FP`) | **Tổ hợp phát huy mạnh nhất ở không dấu:** kết hợp XLM-R + Char-CNN + TF-IDF giảm lỗi xuống thấp nhất (`11.03%`). |
| **`has_emoji`** *(Chứa emoji / biểu tượng)* | `1,164` | `7.73%` (`53 FN, 37 FP`) | `7.82%` (`50 FN, 41 FP`) | **`7.47%`** (`49 FN, 38 FP`) | Cả 3 hệ thống xử lý tốt bình luận có emoji (`< 8%` lỗi). |
| **`short_le_5w`** *(Bình luận ngắn $\le 5$ từ)* | `1,438` | `7.30%` (`64 FN, 41 FP`) | `7.30%` (`60 FN, 45 FP`) | **`6.19%`** (`55 FN, 34 FP`) | Tổ hợp giảm **`-1.11%`** lỗi (`-16` lỗi) trên bình luận ngắn nhờ tín hiệu n-gram chính xác từ TV2 TF-IDF và Char-CNN. |

---

## 9. Giao thức Đánh giá Trung thực và Các Giới hạn của Kết quả

Theo đúng kỷ luật khoa học của `AGENTS.md` và Master Prompt §2 & §9, báo cáo công bố trung thực toàn bộ các điểm giới hạn hiện tại:

1. **Minh bạch tuyệt đối về Tập Test (`test_blind = false`):**
   - `data/processed/binary_n2_v1/test.parquet` (`SHA-256: d200be4a7ff2f2e725ba8e6c2c0ab835afd73612b9a5b2f14b528cb5c577be4e`, `n = 6,048`) từng được truy cập trong lịch sử repository và đợt báo cáo V1 (`N10`). Do đó, báo cáo này **tuyệt đối không gọi `test.parquet` là Blind Test**, mà gọi chính xác là **`Historical Benchmark Test`**.
   - Xác minh toàn vẹn dữ liệu đầu vào: `train.parquet` (`SHA-256: 097b32be...`, `n = 28,231`), `val.parquet` (`SHA-256: 4c33430a...`, `n = 6,048`), `0` trùng lặp giữa các tập sau chuẩn hóa. Khả năng tái lập TV2 `EXP-N6-C-calibrated` đạt `val_repro_max_diff = 0.00e+00`.
2. **Các bước mở rộng chưa chạy do dừng huấn luyện theo yêu cầu người dùng:**
   - **`V3-XLMR-BASE` mới hoàn tất `seed_42` (`n = 1`):** `seed_43` đã được dừng ngay giữa epoch 1 và `seed_44` chưa chạy. Mọi số liệu của `V3-XLMR-BASE` đều ghi rõ `n = 1` (`seed_42`).
   - **Chưa chạy xác nhận 5 seeds (`42..46`):** Toàn bộ các mô hình Scratch V3 và `V3-PHOBERT-V2` đều được đánh giá nhất quán trên **3 seeds (`42, 43, 44`)**, với độ lệch chuẩn rất thấp (`± 0.0019` đến `± 0.0034`).
   - **Chưa chạy `5-Fold Train-OOF CV` (`n = 28,231`) cho Transformer & Scratch V3:** Dù pipeline `run_transformer_5fold_oof()` (`--mode oof5`) đã được triển khai và kiểm thử trong [src/models/transformer_pipeline.py](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/src/models/transformer_pipeline.py), quá trình đánh giá V3 cuối cùng dùng **5-Fold Stratified `Val-OOF`** trên `val.parquet` (`n = 6,048`) để ước lượng hiệu chuẩn/stacking ngoài mẫu trên Validation, sau đó **đóng băng (`frozen`)** để đánh giá trên `test.parquet`. Vì `val.parquet` cũng tham gia vào `early stopping` khi train từng checkpoint đơn, cần hiểu rõ rằng `Val-5Fold-OOF` vẫn có một mức phụ thuộc gián tiếp với quá trình chọn epoch tốt nhất, trong khi `test.parquet` kiểm chứng tham số đóng băng ngoài `val.parquet` (với lưu ý `test_blind = false`).
3. **Lý do bỏ qua `uitnlp/visobert`:**
   - Gói `sentencepiece` / `tiktoken` chưa có sẵn trong môi trường `.venv` của máy và theo quy tắc `AGENTS.md` §10, agent không tự ý cài đặt gói hệ thống mới. Hai đại diện mạnh nhất tương thích sẵn trong `.venv` là `vinai/phobert-base-v2` (tiếng Việt chuyên biệt) và `FacebookAI/xlm-roberta-base` (đa ngôn ngữ BPE) đã được benchmark đầy đủ.
4. **Phạm vi Backend API & Bảo toàn Artifacts:**
   - Thư mục `backend/` không thay đổi trong chiến dịch nghiên cứu V3 (chờ nhóm thống nhất chọn mô hình triển khai chính thức giữa `V3-BIGRU-GATED-FT` nhẹ `16 MB`, `V3-PHOBERT-V2` `515 MB`, hay `V3-ENS-CROSS-VOTING`).
   - Toàn bộ artifacts V1 (`models/cnn/`, `artifacts/TV4-cnn-analysis/`, `artifacts/TV4-ensemble-analysis/`, `artifacts/TV4-locked-test/`) và V2 (`models/cnn_v2/`, `artifacts/TV4-v2-research/`) được bảo toàn 100%.

---

## 10. Khuyến nghị Lựa chọn Mô hình Cho Production FastAPI (Phân tích Đánh đổi)

| Kịch bản Triển khai | Mô hình Đề xuất | Macro-F1 (Val / Test) | F1-TOXIC (Test) | ViCTSD (Test) | ECE (Test) | Kích thước & Tài nguyên CPU/GPU |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Phương án A — Production Nhẹ, Chi phí Thấp (CPU-only / Free Hosting)** | **`V3-BIGRU-GATED-FT`** (`seed_44` hoặc avg 3 seeds + Platt) | `0.7900` / **`0.8032`** | `0.6652` | `0.6886` | `0.0237` (raw) | **`15.97 MB` / `279 MB VRAM`**, suy luận CPU cực nhanh (`< 5 ms/mẫu`), 100% Scratch không cần HuggingFace download |
| **Phương án B — Cân bằng Tốt nhất Cho API Đơn Mô hình (Khuyến nghị nếu RAM $\ge 2\text{ GB}$)** | **`V3-PHOBERT-V2`** (`seed_44` hoặc 3s avg + Platt `thr=0.40`) | **`0.8169`** / **`0.8255`** | **`0.7051`** | **`0.7767`** | **`0.0146`** | `515 MB` (1 seed), mạnh vượt trội trên tin tức/mỉa mai (`ViCTSD`), dễ đóng gói 1 pipeline duy nhất |
| **Phương án C — Độ chính xác Tối đa (Offline Batch / GPU Server / Báo cáo Khoa học)** | **`V3-ENS-CROSS-VOTING`** (TV4 V3 + TV2 `EXP-N6-C` + Platt `thr=0.40`) | **`0.8241`** / **`0.8393`** | **`0.7283`** | **`0.7597`** | **`0.0086`** | Đạt **6/6 mục tiêu**, vượt `PhoBERT-v2` có ý nghĩa thống kê ($p = 0.000614$) và `TV2 EXP-N6-C` ($p = 0.000154$) |

---

## 11. Hướng dẫn Tái lập Đầy đủ (Reproducibility Guide)

### 11.1. Môi trường & Dữ liệu
- **Phần cứng thực nghiệm:** `NVIDIA GeForce RTX 4060 Laptop GPU` (`8 GB VRAM`), Windows, môi trường ảo `.venv`.
- **Bộ dữ liệu chuẩn:** `data/processed/binary_n2_v1/`:
  - `train.parquet` (`n = 28,231`, SHA-256: `097b32bea731bf1a4771a8485155d339a96027d2819fe8a38aa2c9d2e2f32a73`)
  - `val.parquet` (`n = 6,048`, SHA-256: `4c33430a520818078aef6bcc34b3e2573873445dc7040c5729313e11bb60197a`)
  - `test.parquet` (`n = 6,048`, SHA-256: `d200be4a7ff2f2e725ba8e6c2c0ab835afd73612b9a5b2f14b528cb5c577be4e`)

### 11.2. Các Lệnh CLI Tái lập Từng Bước (PowerShell tại thư mục gốc repo)

```powershell
# 1. Chạy unit tests & kiểm tra tĩnh cho toàn bộ mã nguồn V2 & V3
.\.venv\Scripts\pytest.exe tests/test_v3_models.py tests/test_cnn_v2.py -q
.\.venv\Scripts\flake8.exe src/models/cnn.py src/models/cnn_pipeline.py src/models/transformer_pipeline.py src/evaluation/v3_final_evaluator.py tests/test_v3_models.py --select=F,E9

# 2. Tái lập Nhánh A — 4 biến thể Scratch V3 (3 seeds: 42, 43, 44 = 12 runs)
.\.venv\Scripts\python.exe -m src.models.cnn_pipeline `
  --config configs/cnn_v3_config.yaml `
  --variant V3-BIGRU-GATED-MSD V3-BIGRU-GATED-FT V3-BIGRU-PYVI-FT V3-TRI-GATED-FT `
  --seeds 42 43 44 `
  --run-group-dir models/cnn_v3/TV4-V3-scratch

# 3. Tái lập Nhánh B — Pretrained PhoBERT-v2 (3 seeds: 42, 43, 44) & XLM-R Base (seed 42)
.\.venv\Scripts\python.exe -m src.models.transformer_pipeline `
  --config configs/transformer_v3_config.yaml `
  --variant V3-PHOBERT-V2 `
  --seeds 42 43 44 `
  --mode seeds

.\.venv\Scripts\python.exe -m src.models.transformer_pipeline `
  --config configs/transformer_v3_config.yaml `
  --variant V3-XLMR-BASE `
  --seeds 42 `
  --mode seeds

# (Tùy chọn mở rộng trong tương lai: chạy 5-Fold Train-OOF hoàn toàn độc lập với val.parquet)
# .\.venv\Scripts\python.exe -m src.models.transformer_pipeline `
#   --config configs/transformer_v3_config.yaml --variant V3-PHOBERT-V2 --mode oof5

# 4. Tái lập toàn bộ đánh giá V3 (Ensemble, Cross-Ensemble, Calibration, Diversity, Bootstrap CI, McNemar, Error Analysis & 4 biểu đồ PNG)
.\.venv\Scripts\python.exe -m src.evaluation.v3_final_evaluator
```

### 11.3. Danh mục Biểu đồ & Dữ liệu Xuất ra
- [tv4_v3_final_report.json](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/artifacts/TV4-v3-research/tv4_v3_final_report.json) — Toàn bộ số liệu gốc ở độ chính xác dấu phẩy động đầy đủ.
- [v1_v2_v3_progression.png](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/artifacts/TV4-v3-research/v1_v2_v3_progression.png) — Biểu đồ tiến hóa V1 $\to$ V2 $\to$ V3.
- [v3_error_diversity_heatmap.png](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/artifacts/TV4-v3-research/v3_error_diversity_heatmap.png) — Bản đồ nhiệt đa dạng lỗi & tương quan xác suất.
- [v3_calibration_reliability.png](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/artifacts/TV4-v3-research/v3_calibration_reliability.png) — Reliability Diagram & so sánh `Platt / Temperature / Isotonic`.
- [v3_source_comparison.png](file:///c:/Users/THE%20ANH/Documents/Tài%20liệu%20học/Máy%20học/toxic-comment-detection-master/artifacts/TV4-v3-research/v3_source_comparison.png) — So sánh tổng quát hóa theo miền `ViCTSD` vs `ViHSD`.
