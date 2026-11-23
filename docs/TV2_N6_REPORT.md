# Báo cáo Thực nghiệm & Bàn giao TV2 (Vinh) — N6 Binary LinearSVC

**Ngày:** 2026-11-23 · **SPEC:** v5.0.0 · **Trạng thái:** `EXPERIMENTALLY VALIDATED`

## 1. Dữ liệu & Giao thức
- **Dataset chính (`binary_n2_v1`):** `train.parquet` (`28,231` mẫu: `23,734` CLEAN / `4,497` TOXIC = `15.9%`), `val.parquet` (`6,048` mẫu: `4,559` ViHSD + `1,489` ViCTSD). `test.parquet` bị khóa hoàn toàn (chỉ mở ở N10).
- **Augmentation (`binary_n4_v1`):** `5,675` mẫu `TOXIC` train-only (đưa `train` lên `33,906` mẫu, `30.0%` TOXIC) chỉ dùng cho ablation N6-D.
- **Seeds:** `[42, 43, 44]` (3 seeds mỗi biến thể).
- **Metric chọn mô hình:** Raw Validation **Macro-F1**; metrics bổ sung: **F1 TOXIC**, **PR-AUC**, **Weighted-F1**, metrics theo nguồn (`vihsd` vs `victsd`), ECE, latency, và kích thước `.joblib` (`< 15 MB`).

## 2. Kết quả Thực nghiệm N6 (Validation `6,048` mẫu)

| Biến thể | Mô hình & Thiết lập chính | Macro-F1 (mean±std) | F1 TOXIC (mean±std) | Weighted-F1 | PR-AUC | ViHSD Macro-F1 | ViCTSD Macro-F1 | Size (MB) |
|---|---|---|---|---|---|---|---|---|
| **N6-A** | `raw` + `word(1-2)` + `LinearSVC(C=1.0)` | `0.7768 ± 0.0000` | `0.6145 ± 0.0000` | `0.8875` | `0.7018` | `0.7958` | `0.6578` | `2.339` |
| **N6-B** | `basic` + `word+char` + `LinearSVC(C=1.0)` | `0.7875 ± 0.0000` | `0.6351 ± 0.0000` | **`0.8912`** | `0.7180` | **`0.8104`** | `0.6516` | `3.715` |
| **N6-C** *(Best raw)* | `teencode` + `word+char` + `chi2(100k)` + `C=0.5, balanced` | **`0.7892 ± 0.0000`** | **`0.6502 ± 0.0000`** | `0.8838` | `0.7217` | `0.8091` | **`0.6879`** | `7.349` |
| **N6-D (bal)** | `N6-C` + `binary_n4_v1` augmentation (`33,906` train) | `0.7833 ± 0.0000` | `0.6294 ± 0.0000` | `0.8881` | `0.7120` | `0.8062` | `0.6428` | `11.025` |
| **N6-D (none)** | `N6-C (class_weight=None)` + `binary_n4_v1` augmentation | `0.7741 ± 0.0000` | `0.6063 ± 0.0000` | `0.8884` | `0.7133` | `0.8002` | `0.5964` | `11.025` |

### Kết quả Calibration & Threshold cho Artifact Inference (sau khi chọn N6-C)
- **Run:** `N6-C-calibrated` (`CalibratedClassifierCV(cv=5, method="sigmoid")` fit trên train CV, 3 seeds).
- **Diagnostics (in-sample trên validation, KHÔNG dùng để xếp hạng model):**
  - `ECE`: **`0.0111`** | `Log-Loss`: `0.2559` | `Brier`: `0.0763` | `PR-AUC`: **`0.7243`**
  - Tại ngưỡng mặc định `0.50`: `Macro-F1 = 0.7836`, `F1 TOXIC = 0.6274`, `Weighted-F1 = 0.8901`
  - Tại ngưỡng tối ưu trên validation (`threshold = 0.34`): `Macro-F1 = 0.7954`, `F1 TOXIC = 0.6575`
- **Dung lượng fallback `.joblib`:** **`10.403 MB`** (đạt yêu cầu `< 15 MB`).

## 3. Phân tích Ablation & Lỗi (`N6-C`)
1. **Ảnh hưởng của `word+char` (`A -> B`):** Macro-F1 tăng `+0.0107` (`0.7768 -> 0.7875`), Weighted-F1 đạt `0.8912`.
2. **Ảnh hưởng của `chi2` + `class_weight="balanced"` (`B -> C`):** Kéo `F1 TOXIC` tăng `+0.0151` (`0.6351 -> 0.6502`) và cải thiện rõ rệt trên nguồn `ViCTSD` (`0.6516 -> 0.6879`, `TOXIC F1` tăng từ `0.3614 -> 0.4471`).
3. **Ảnh hưởng của `binary_n4_v1` augmentation (`C -> D`):** Thêm 5,675 mẫu TOXIC biến thể ký tự/bỏ dấu làm `LinearSVC` giảm nhẹ (`0.7892 -> 0.7833` khi có `balanced`, hoặc `0.7741` khi `None`), đặc biệt giảm ở `ViCTSD`. Giữ phương án **`N6-C` không augmentation với `class_weight="balanced"`** làm phương án xử lý imbalance tốt nhất cho SVM.
4. **Kiểm tra mở rộng không giới hạn từ điển (`max_features=null, min_df=1`):** Thử nghiệm kiểm chứng (36 tổ hợp `k`, `C`, `class_weight`) đạt đỉnh tại `k=150,000 / 652,571` chiều, `C=0.7, balanced`: `Macro-F1 = 0.7994`, `F1 TOXIC = 0.6598`, `Weighted-F1 = 0.8946`, nhưng kích thước `.joblib` tăng lên **`28.319 MB`** (vượt ngưỡng `< 15 MB`), nên không thay thế `N6-C` chuẩn (`7.349–10.403 MB`).
4. **Chênh lệch theo nguồn (`vihsd` vs `victsd`):** `ViHSD` đạt Macro-F1 `0.8091` (`TOXIC F1 = 0.6903`), trong khi `ViCTSD` chỉ đạt `0.6879` (`TOXIC F1 = 0.4471`) do bình luận ViCTSD dài và độc hại theo ngữ cảnh/quan điểm nhiều hơn là từ khóa thô tục.
5. **Tổng hợp lỗi ẩn danh tính trên Validation (`N6-C`, `721 / 6,048` mẫu, tỷ lệ lỗi `11.92%`):**
   - Cặp nhầm lẫn: `CLEAN -> TOXIC`: **427** | `TOXIC -> CLEAN`: **294**.
   - Theo nguồn: `vihsd`: **533** (`11.69%`) | `victsd`: **188** (`12.63%`).
   - Pattern heuristic trong các ca lỗi: `contains_teencode` (`277`), `context_or_lexical_nuance` (`228`), `contains_negation` (`222`), `short_text_le_5_words` (`123`), `contains_emoji_or_punct` (`79`), `contains_url_or_mention` (`3`).

## 4. Thư mục Artifacts Bàn giao cho Huy (`N8`, `N11`, `N15`) (`max_iter = 5000`)
- **N6-A, B, C, D (`balanced`, 3 seeds):** `models/baselines/N6-run-931168a6f09241c6a8c943fd99f26c7f/` (`EXP-N6-A`, `EXP-N6-B`, `EXP-N6-C` best raw, `EXP-N6-D`, `N6-SVM-summary.json`)
- **N6-C-calibrated (Calibration + threshold `0.34`, `10.403 MB < 15 MB`):** `models/baselines/N6-run-faffb8025fbc41fb8d9b463468bdd5f3/EXP-N6-C-calibrated/`
- **N6-D (`class_weight=None`, 3 seeds):** `models/baselines/N6-run-ff683d828c7244ee9124b396bf2988d4/EXP-N6-D/`
