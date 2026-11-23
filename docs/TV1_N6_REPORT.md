# Báo cáo Thực nghiệm & Bàn giao TV1 (Tam) — N6 Binary Logistic Regression

**Ngày:** 2026-11-24 · **SPEC:** v5.0.0 · **Trạng thái:** `EXPERIMENTALLY VALIDATED`

## 1. Dữ liệu & Giao thức
- **Dataset chính (`binary_n2_v1`):** `train.parquet` (`28,231` mẫu: `23,734` CLEAN / `4,497` TOXIC = `15.9%`), `val.parquet` (`6,048` mẫu: `4,559` ViHSD + `1,489` ViCTSD). `test.parquet` bị khóa hoàn toàn (chỉ mở ở N10).
- **Augmentation (`binary_n4_v1`):** `5,675` mẫu `TOXIC` train-only (đưa `train` lên `33,906` mẫu, `30.0%` TOXIC) dùng cho ablation `N6-LR-D`.
- **Seeds:** `[42, 43, 44]` (3 seeds mỗi biến thể).
- **Metric chọn mô hình:** Raw Validation **Macro-F1**; metrics bổ sung: **F1 TOXIC**, **PR-AUC**, **Weighted-F1**, metrics theo nguồn (`vihsd` vs `victsd`), raw ECE, latency và kích thước `.joblib` (`< 15 MB`).

## 2. Kết quả Thực nghiệm N6 (Validation `6,048` mẫu)

| Biến thể | Mô hình & Thiết lập chính | Macro-F1 (mean±std) | F1 TOXIC (mean±std) | Weighted-F1 | PR-AUC | Raw ECE | ViHSD Macro-F1 | ViCTSD Macro-F1 | Size (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **N6-LR-A** | `raw` + `word(1-2)` + `LR(C=1.0)` | `0.7128 ± 0.0000` | `0.4905 ± 0.0000` | `0.8643` | `0.6807` | `0.0431` | `0.7369` | `0.5533` | `0.985` |
| **N6-LR-B** | `basic` + `word+char` + `LR(C=5.0)` | `0.7874 ± 0.0000` | `0.6327 ± 0.0000` | **`0.8928`** | **`0.7283`** | **`0.0104`** | `0.8093` | `0.6565` | `1.549` |
| **N6-LR-C** | `teencode` + `word+char` + `LR(C=5.0, balanced)` | `0.7871 ± 0.0000` | **`0.6479 ± 0.0000`** | `0.8819` | `0.7231` | `0.0792` | `0.8090` | **`0.6721`** | `2.190` |
| **N6-LR-D** *(Best raw)* | `N6-LR-C` + `binary_n4_v1` aug (`33,906` train, `C=5.0, balanced`) | **`0.7901 ± 0.0000`** | `0.6419 ± 0.0000` | `0.8910` | `0.7059` | `0.0289` | **`0.8127`** | `0.6464` | `3.194` |

### Kết quả Calibration & Threshold Diagnostics cho `N6-LR-D`
- **Run:** `N6-LR-D-calibrated` (`CalibratedClassifierCV(cv=5, method="sigmoid")` fit trên train CV, 3 seeds).
- **Diagnostics (in-sample trên validation, KHÔNG dùng để xếp hạng model):**
  - `ECE`: `0.0438` | `Log-Loss`: `0.2949` | `Brier`: `0.0816` | `PR-AUC`: `0.7036`
  - Tại ngưỡng mặc định `0.50`: `Macro-F1 = 0.7738`
  - Tại ngưỡng tối ưu trên validation (`threshold = 0.37`): `Macro-F1 = 0.7903`, `F1 TOXIC = 0.6429`
- **Dung lượng fallback `.joblib`:** **`3.194 MB`** (raw) / **`7.854 MB`** (calibrated) — đều đạt yêu cầu `< 15 MB`.

## 3. Phân tích Ablation & Lỗi (`N6-LR-D`)
1. **Ảnh hưởng của `basic` + `word+char TF-IDF` (`A -> B`):** Macro-F1 tăng mạnh `+0.0746` (`0.7128 -> 0.7874`), F1 TOXIC tăng `+0.1422` (`0.4905 -> 0.6327`), đồng thời đạt `raw_ece = 0.0104` và `PR-AUC = 0.7283` tốt nhất.
2. **Ảnh hưởng của `teencode` + `class_weight="balanced"` (`B -> C`):** Đẩy `F1 TOXIC` lên cao nhất (`0.6479`, tăng `+0.0152`) và cải thiện rõ trên nguồn `ViCTSD` (`0.6565 -> 0.6721`). Tuy nhiên `raw_ece` tăng lên `0.0792` do `class_weight="balanced"` làm dịch xác suất dự đoán.
3. **Ảnh hưởng của `binary_n4_v1` augmentation (`C -> D`):** Bổ sung `5,675` mẫu TOXIC tăng cường vào train giúp Macro-F1 đạt đỉnh **`0.7901`** (tăng ở `ViHSD`: `0.8090 -> 0.8127`, giảm nhẹ ở `ViCTSD`: `0.6721 -> 0.6464`).
4. **Phân tích lỗi ẩn danh trên Validation (`N6-LR-D`, `637 / 6,048` mẫu, tỷ lệ lỗi `10.53%`):**
   - Cặp nhầm lẫn: `TOXIC -> CLEAN`: **393** | `CLEAN -> TOXIC`: **244**.
   - Theo nguồn: `vihsd`: **478** (`10.48%`) | `victsd`: **159** (`10.68%`).
   - Pattern heuristic trong các ca lỗi: `contains_teencode` (`257`), `contains_negation` (`196`), `context_or_lexical_nuance` (`187`), `short_text_le_5_words` (`116`), `contains_emoji_or_punct` (`64`), `contains_url_or_mention` (`2`).

## 4. Artifacts Bàn giao cho Huy (`N8`, `N11`, `N15`)
- **N6-LR-A, B, C, D (raw, 3 seeds `[42, 43, 44]`):** `models/baselines/N6-LR-run-0ede1adcfaa84d56803375b97621ac4f/` (`EXP-N6-LR-A`, `EXP-N6-LR-B`, `EXP-N6-LR-C`, `EXP-N6-LR-D` best raw, `N6-LR-summary.json`)
- **N6-LR-D-calibrated (`cv=5`, `sigmoid`, `7.854 MB < 15 MB`):** `models/baselines/N6-LR-run-1533b39d1efa4b70b4179f884d1143f0/EXP-N6-LR-D-calibrated/`
- **Code & Config & Tests:** `src/models/lr_binary_pipeline.py`, `configs/lr_binary_config.yaml`, `tests/test_lr_binary_smoke.py` (`2 passed`).
