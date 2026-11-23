# Báo cáo Phân tích & Bàn giao TV2 — EXP-02 (LinearSVC)

> **Lưu ý (2026-11-23):** Báo cáo 3 lớp (`CLEAN/OFFENSIVE/HATE`) dưới đây thuộc giai đoạn G0 cũ và đã được thay thế bởi báo cáo nhị phân hiện hành (SPEC v5.0.0) tại [`docs/TV2_N6_REPORT.md`](TV2_N6_REPORT.md).

**Dataset G0:** `data/processed/vihsd_g0_v1/train_val.jsonl` (`n_train=23002`, `n_val=4929`)
**Dataset SHA-256:** `c17d920f788ff982775405d7031373dc2cd5525cbba5db6ed4e4f85e70159164`
**Manifest SHA-256:** `1a3364ae6dcdc50e4f55e4bda665c5b4065c2835d2d98db808e60f4932c2edb2`

## 1. So sánh hiệu năng & chi phí các biến thể EXP-02

| Variant | Cấu hình chính | Val Macro-F1 | Weighted-F1 | Balanced Acc | CLEAN F1 | OFFENSIVE F1 | HATE F1 | Features | Train Time | Latency (ms/sample) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **EXP-02-A** | raw + word(1-2) + LinearSVC(C=1.0) | `0.6378` | `0.8716` | `0.5850` | `0.9462` | `0.4185` | `0.5486` | `35436`/`35436` | `0.447s` | `0.0007` | `1.999` |
| **EXP-02-B** | basic + word+char + LinearSVC(C=1.0) | `0.6615` | `0.8800` | `0.6145` | `0.9498` | `0.4592` | `0.5755` | `64732`/`64732` | `2.184s` | `0.0024` | `3.520` |
| **EXP-02-C** | teencode + word+char + chi2(50k) + LinearSVC(C=0.1, balanced) | `0.6845` | `0.8865` | `0.6517` | `0.9512` | `0.5000` | `0.6022` | `50000`/`88044` | `1.000s` | `0.0012` | `5.365` |
| **EXP-02-C-calibrated** | EXP-02-C + CalibratedClassifierCV(cv=5, sigmoid) | `0.6486` | `0.8724` | `0.5880` | `0.9437` | `0.4363` | `0.5659` | `50000`/`88044` | `4.777s` | `0.0192` | `9.945` |
| **EXP-02-D** | teencode + word+char + chi2(10k) + SVC(kernel=rbf) | `0.6608` | `0.8740` | `0.6440` | `0.9427` | `0.4757` | `0.5639` | `10000`/`41399` | `538.020s` | `10.7627` | `12.922` |

## 2. Phân tích Ablation & Đánh đổi (`EXP-02-C`, 40 cấu hình)

- **Cấu hình tối ưu (chọn theo Val Macro-F1):** `k=50000`, `C=0.1`, `class_weight=balanced` $\rightarrow$ `val_macro_f1=0.6845` (`train_time_s=1.000s`).

### 2.1. Tác động của `class_weight`
| `class_weight` | Số cấu hình | Best Macro-F1 | Mean Macro-F1 | Mean Train Time |
|---|---|---|---|---|
| `None` | 20 | `0.6651` | `0.6391` | `2.995s` |
| `balanced` | 20 | `0.6845` | `0.6648` | `4.949s` |

### 2.2. Tác động của tham số điều chuẩn `C`
| `C` | Số cấu hình | Best Macro-F1 | Mean Macro-F1 | Mean Train Time |
|---|---|---|---|---|
| `0.1` | 8 | `0.6845` | `0.6408` | `1.329s` |
| `0.5` | 8 | `0.6744` | `0.6604` | `2.088s` |
| `1.0` | 8 | `0.6664` | `0.6591` | `3.082s` |
| `2.0` | 8 | `0.6600` | `0.6523` | `4.624s` |
| `5.0` | 8 | `0.6556` | `0.6472` | `8.738s` |

### 2.3. Tác động của chọn lọc đặc trưng `SelectKBest(chi2)` (`k`)
| `k` | Số cấu hình | Best Macro-F1 | Mean Macro-F1 | Mean Train Time |
|---|---|---|---|---|
| `20000` | 10 | `0.6811` | `0.6457` | `3.474s` |
| `50000` | 10 | `0.6845` | `0.6515` | `3.595s` |
| `100000` | 10 | `0.6820` | `0.6553` | `4.490s` |
| `all(88044)` | 10 | `0.6820` | `0.6553` | `4.331s` |

### 2.4. Giới hạn thiết kế của LinearSVC
- Mô hình tuyến tính trên bag-of-n-grams (TF-IDF) không mô hình hóa được thứ tự từ dài, đảo ngữ phức tạp hay ngữ cảnh hội thoại ngoài cửa sổ 1–3 từ.
- `LinearSVC` chỉ tạo lề phân tách (`decision_function`), không sinh xác suất tự nhiên; muốn dùng cho soft voting/API bắt buộc phải bọc `CalibratedClassifierCV` trên train folds.
- `class_weight='balanced'` cải thiện mạnh recall cho `OFFENSIVE`/`HATE` nhưng đánh đổi nhẹ precision của `CLEAN` (nhiều bình luận sạch có từ nhạy cảm bị đẩy sang `OFFENSIVE`/`HATE`).

## 3. Kiểm chứng Run Lineage & Reproducibility

| Variant | MLflow Run ID | Status | Val Count | Reload Exact Match | Calibrated | Extra Calibration Metrics |
|---|---|---|---|---|---|---|
| **EXP-02-A** | `435376e5ce864397a2ca954f2a0e57cb` | `FINISHED` | `4929` | `4929/4929` | `False` | - |
| **EXP-02-B** | `8816655917444836b9fb1723b9bd9c80` | `FINISHED` | `4929` | `4929/4929` | `False` | - |
| **EXP-02-C** | `e58e837ef7d04a72942788751711e6e3` | `FINISHED` | `4929` | `4929/4929` | `False` | - |
| **EXP-02-C-calibrated** | `afb9812abaf2445e98450502d7eded55` | `FINISHED` | `4929` | `4929/4929` | `True` | log_loss=0.3194, brier_macro=0.0547 |
| **EXP-02-D** | `a595fa7e704f45a5a89e139d8ccf6b1c` | `FINISHED` | `4929` | `4929/4929` | `False` | - |

## 4. Bàn giao Phân tích Lỗi cho EXP-05 (TV3) & Ensemble cho EXP-06 (TV4)

- **Tổng lỗi của `EXP-02-C` trên Validation:** `525 / 4929` (`error_rate=0.1065`).
- **Phân bố cặp nhầm lẫn:**
  - `HATE -> CLEAN`: **172** mẫu
  - `OFFENSIVE -> CLEAN`: **123** mẫu
  - `CLEAN -> HATE`: **67** mẫu
  - `HATE -> OFFENSIVE`: **65** mẫu
  - `OFFENSIVE -> HATE`: **50** mẫu
  - `CLEAN -> OFFENSIVE`: **48** mẫu
- **Nhóm đặc trưng xuất hiện trong các ca lỗi (heuristic):**
  - `contains_teencode`: **223** lần
  - `context_or_lexical_nuance`: **151** lần
  - `short_text_le_5_words`: **124** lần
  - `offensive_hate_boundary`: **115** lần
  - `contains_negation`: **109** lần
  - `contains_emoji_or_punct`: **72** lần
  - `contains_url_or_mention`: **8** lần

### Metadata bàn giao TV4 (EXP-06)
- **Hard predictions (LinearSVC tốt nhất):** `models/baselines/EXP-02-C/val_predictions.json` (`Macro-F1=0.6845`).
- **Calibrated probabilities (Platt scaling `cv=5` chỉ trên train):** `models/baselines/EXP-02-C-calibrated/val_predictions.json` (`Macro-F1=0.6486`, `log_loss=0.3194`, `brier_macro=0.0547`).
- **File lỗi đại diện cho TV3 (EXP-05):** `models/baselines/EXP-02-report/svm_error_analysis_val.json`.
