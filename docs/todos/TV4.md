# TODO — TV4: TheAnh — TextCNN, tuning kiến trúc, Ensemble/Stacking (EXP-04/06)

## Kế hoạch hiện hành D1–D4

Các mục G0 bên dưới là lịch sử ba lớp; giữ trạng thái đã ghi và làm lại phần cần thiết trên split nhị phân mới.

- [x] **N1:** Duyệt spec, decisions, plan và checklist cùng nhóm. (`EXPERIMENTALLY VALIDATED` 2026-11-23 — chốt bài toán nhị phân D1–D4 `CLEAN=0, TOXIC=1` trên `binary_n2_v1` & `binary_n4_training_v1`, gộp Word TextCNN + Char-CNN/Hybrid N7).
- [x] **N7:** Sau N4 và N5, phối hợp Huy thử BiGRU/BiLSTM với Char-CNN. Chạy ba seed, chọn checkpoint theo val Macro-F1, báo F1 TOXIC và VRAM mục tiêu dưới 7 GB. (`EXPERIMENTALLY VALIDATED` 2026-11-24 — `HYBRID-RNN-01` BiGRU+Char-CNN đạt Val Macro-F1 `0.7733 ± 0.0072`, F1-TOXIC `0.6149 ± 0.0130`, best seed 42 `0.7806`, VRAM `230.8 MB << 7 GB`; `HYBRID-RNN-WD` đạt `0.7731 ± 0.0052`, F1-TOXIC `0.6155`; `HYBRID-BILSTM-01` BiLSTM+Char-CNN đạt `0.7665 ± 0.0027`, VRAM `230.6 MB`).
- [x] **N7:** Thực hiện ablation Char-CNN trên validation; chỉ thay một yếu tố mỗi lần. Lưu config, hash, checkpoint, MLflow run ID, metrics và predictions theo lớp, nguồn. (`EXPERIMENTALLY VALIDATED` 2026-11-24 — 18 biến thể × 3 seeds = 54 runs trong `models/cnn/TV4-run-*/` & MLflow `TV4-CNN-Binary`, gồm Char-CNN `CHAR-01` `0.7501`, `CHAR-02` `filters=256` `0.7550`, `CHAR-03` `drop=0.3` `0.7525`, cùng các nhánh Word TextCNN `CNN-01..04b`, `CNN-01-WD` `0.7705`, D3 imbalance `CW/FOCAL/SAMPLER` và `CNN-AUG-N4`).
- [x] Bàn giao validation predictions và lỗi CNN cho Huy; không gọi raw scores là xác suất calibrated. (`EXPERIMENTALLY VALIDATED` 2026-11-24 — bàn giao `val_predictions.json` với `is_calibrated: false`, báo cáo chẩn đoán Temperature/Platt ECE & 40 mẫu lỗi ẩn danh trong `artifacts/TV4-cnn-analysis/`, `notebooks/EXP-04-cnn-analysis.ipynb` và `docs/TV4_CNN_REPORT.md`).
- [x] **N10:** Cùng nhóm đánh giá danh sách đã freeze trong một đợt locked test; không tuning theo test. (`EXPERIMENTALLY VALIDATED` 2026-11-24 — đóng băng manifest `artifacts/TV4-locked-test/tv4_n10_freeze_manifest.json` trước khi đọc `test.parquet` (`n = 6,048`, SHA-256 `d200be4a7ff2f2e725ba8e6c2c0ab835afd73612b9a5b2f14b528cb5c577be4e`, công bố rõ `test_blind: false`); mô hình đơn TV4 đạt Test Macro-F1 `0.7855 ± 0.0082` (`HYBRID-RNN-01`), `0.7867 ± 0.0049` (`CNN-01`), `0.7839 ± 0.0071` (`CNN-01-WD`); ensemble nội bộ TV4 đạt `0.7991 ± 0.0028` (`ENS-TV4-INTRA-4 @0.50`) và `0.8029 ± 0.0025` (`ENS-TV4-FROZEN-PLATT`); ensemble liên thành viên TV2+TV4 đạt `0.8029 ± 0.0009` (`ENS-CROSS-DIVERSE-050`, `ECE = 0.0073`) và **`0.8128 ± 0.0019`** (`ENS-CROSS-FROZEN-PLATT`, `F1-TOXIC = 0.6834`, `PR-AUC = 0.7572`), vượt mô hình đơn tốt nhất của cả hai thành viên).

  **Phạm vi đã đối soát:** đây là batch lịch sử TV4 (7 singles, 3 ensemble TV4-only) và một TV2 reference dùng thêm trong 2 cross-member ensembles; không chứng minh đủ coverage/sign-off toàn nhóm. Không suy rộng checkbox này thành N10 team-wide hoàn tất; xem `artifacts/TV4-locked-test/tv4_n10_reconciliation.md`. Không mở lại locked test.

## Lịch sử G0 trước D1–D4


**Snapshot audit (lịch sử trước khi triển khai):** tại thời điểm đầu kỳ chưa thấy implementation TextCNN, config, test, run metrics, validation predictions hay checkpoints riêng cho TV4. **Cập nhật nghiệm thu 2026-11-24 (`EXPERIMENTALLY VALIDATED`):** Toàn bộ các mục `P0`, `P1`, `P2` bên dưới đã được hoàn thành, kiểm thử tự động (`28/28` tests của TV4 pass, `373/373` tests toàn repo pass) và xác nhận thực nghiệm với **18 biến thể × 3 seeds = 54 GPU runs** trên split nhị phân chuẩn `data/processed/binary_n2_v1/` & `binary_n4_training_v1/`.

**Quy ước chung:** dùng [Data Contract v1](../DATA_CONTRACT_v1.md) (và bản nâng cấp nhị phân D1–D4 `data/processed/binary_n2_v1/`), embedding random-init; vocabulary fit train-only; chọn theo validation Macro-F1; locked test chỉ EXP-08 / N10. Softmax/score của CNN không tự động là probability đã calibration.

## P0 — nghiệm thu EXP-04

- [x] **P0 — Chốt input/config của TextCNN trên G0 / D1–D4.** (`EXPERIMENTALLY VALIDATED` 2026-11-24 — `configs/cnn_config.yaml` v2.0.0 định nghĩa đầy đủ 18 biến thể × 3 seeds `[42, 43, 44]`, random-init embedding `128D/64D`, `min_word_freq=2`, `min_char_freq=2`, chỉ đọc `train.parquet` & `val.parquet`, ghi SHA-256 manifest, không chạm locked test).
  - **Đầu ra:** config versioned cho dataset/split, vocabulary, embedding, kernel/filter/dropout, seed và output path.
  - **Nghiệm thu:** chỉ dùng train/val; ghi manifest/hash; embedding random-init; không đọc locked test.
  - **Dependencies:** G0 / N2 OPEN; không cần đợi EDA TV1.

- [x] **P0 — Tạo TextCNN single-kernel và train/evaluate entrypoint.** (`EXPERIMENTALLY VALIDATED` 2026-11-24 — `src/models/cnn.py` và `src/models/cnn_pipeline.py` hỗ trợ 5 họ kiến trúc `textcnn`, `char_cnn`, `word_char_cnn`, `bilstm_char_cnn`, `bigru_char_cnn`; được kiểm thử bằng 20 bài smoke tests trong `tests/test_cnn_smoke.py` với `result_type="smoke"` tách biệt khỏi official runs).
  - **Đầu ra:** model/runner cho baseline single-kernel, validation metrics và prediction artifact.
  - **Nghiệm thu:** fixture smoke kiểm tra forward/train/eval, logits, checkpoint; smoke không tính là official run.
  - **Dependencies:** config P0; có thể tham khảo recurrent utilities nhưng model thuộc TV4.

- [x] **P0 — Chạy EXP-04-A: single-kernel TextCNN baseline.** (`EXPERIMENTALLY VALIDATED` 2026-11-24 — `CNN-01` `kernel_sizes=[3], filters=128, drop=0.5` chạy 3 seeds `42, 43, 44` tại `models/cnn/TV4-run-c97812df7da7416395530962c87a7664/CNN-01/`, đạt Val Macro-F1 `0.7702 ± 0.0041`, F1-TOXIC `0.6065 ± 0.0091`, `15.12 MB`, `58.6s`, VRAM `105.8 MB`, reload `best.pt` khớp `6,048/6,048` mẫu).
  - **Đầu ra:** config, checkpoint, history, metrics, validation predictions, MLflow run ID.
  - **Nghiệm thu:** split/hash; Macro-F1/per-class/confusion, seed/compute/time/size log; artifact load lại được.
  - **Dependencies:** entrypoint P0.

- [x] **P0 — Thêm multi-kernel TextCNN variant.** (`EXPERIMENTALLY VALIDATED` 2026-11-24 — `CNN-02` `kernel_sizes=[3, 4, 5], filters=128, drop=0.5` chạy 3 seeds tại `models/cnn/TV4-run-c97812df7da7416395530962c87a7664/CNN-02/`, đạt Val Macro-F1 `0.7664 ± 0.0048`, cải thiện `ViCTSD` từ `0.6285` lên `0.6379` nhưng thấp hơn `CNN-01` trên `ViHSD` do overfit khi chưa có L2 weight decay).
  - **Đầu ra:** config/model cho nhiều kernel và run metrics/predictions.
  - **Nghiệm thu:** thay đổi kiến trúc được ghi rõ; cùng split; không mặc định multi-kernel tốt hơn.
  - **Dependencies:** single-kernel baseline và runner.

- [x] **P0 — Chạy biến thể thứ ba với một thay đổi có giả thuyết.** (`EXPERIMENTALLY VALIDATED` 2026-11-24 — Đã chạy 16 biến thể tiếp theo × 3 seeds, mỗi lần chỉ thay 1 yếu tố: `CNN-03a` `filters=64` `0.7679`, `CNN-03b` `filters=256` `0.7676`, `CNN-04a` `drop=0.3` `0.7576`, `CNN-04b` `[3,4,5]+wd=1e-4` `0.7688`, `CNN-01-WD` `[3]+wd=1e-4` `0.7705`, D3 imbalance `CW/FOCAL/SAMPLER`, `CNN-AUG-N4`, `CHAR-01..03`, `HYBRID-01`, `HYBRID-BILSTM-01` `0.7665`, `HYBRID-RNN-01` `0.7733`, `HYBRID-RNN-WD` `0.7731`).
  - **Đầu ra:** run thay đổi filters/dropout/pooling/class weighting so với baseline; config, checkpoint, metrics và predictions.
  - **Nghiệm thu:** ≥3 biến thể có ý nghĩa tổng cộng; chỉ thay đổi khai báo; cùng data hash; Macro-F1/per-class/confusion lưu đủ.
  - **Dependencies:** baseline và multi-kernel runs; validation.

## P1 — ablation và bàn giao

- [x] **P1 — Tổng hợp ablation TextCNN.** (`EXPERIMENTALLY VALIDATED` 2026-11-24 — Bảng tổng hợp 18 biến thể × 3 seeds, 6 vòng phân tích Hướng 2 và 5 biểu đồ PNG tại `docs/TV4_CNN_REPORT.md`, `artifacts/TV4-cnn-analysis/tv4_phase4_report.json` và `notebooks/EXP-04-cnn-analysis.ipynb`).
  - **Đầu ra:** bảng so sánh kernel/filter/dropout/pooling, chi phí và kết luận.
  - **Nghiệm thu:** kết luận chỉ từ validation; nêu biến thể không cải thiện/giới hạn; configs và run IDs dẫn nguồn được.
  - **Dependencies:** ba official runs.

- [x] **P1 — Kiểm tra predictions/artifacts để ghép cross-model.** (`EXPERIMENTALLY VALIDATED` 2026-11-24 — Kiểm chứng tự động 54/54 runs bằng `verify_all_runs_integrity()` trong `src/evaluation/cnn_report.py` & `tests/test_cnn_report.py`: `all_integrity_checks_passed = True`, khớp 100% `6,048` IDs của `val.parquet`, không có dòng test, ghi rõ `is_calibrated: false` và `probability_source: "uncalibrated_softmax"`).
  - **Đầu ra:** validation predictions có ID, true/predicted labels, run ID và hash; model/checkpoint có thể load.
  - **Nghiệm thu:** ID khớp validation; không có test rows; score không bị gọi là calibrated probability nếu chưa hiệu chuẩn.
  - **Dependencies:** official runs; TV3 có thể nhận mỗi variant khi sẵn sàng.

- [x] **P1 — Gửi error examples và kết luận CNN cho TV3 EXP-05.** (`EXPERIMENTALLY VALIDATED` 2026-11-24 — Xuất 40 mẫu lỗi validation đã che thông tin cá nhân PII `<USER>`, `<URL>`, `<EMAIL>`, `<NUM>`, `<NAME>` tại `artifacts/TV4-cnn-analysis/tv4_error_analysis_val.json` kèm bảng phân tích 32 mẫu đại diện và thống kê lỗi theo 8 nhóm hiện tượng ngôn ngữ & 4 khoảng độ dài câu trong `docs/TV4_CNN_REPORT.md` Mục 5).
  - **Đầu ra:** các lỗi validation đại diện, nhóm lỗi và nhận xét CNN bắt/nhầm lớp nào.
  - **Nghiệm thu:** cùng IDs với các nhánh khác; chỉ validation; không thay thế phân tích cross-model của TV3.
  - **Dependencies:** validation predictions; các model khác có thể bàn giao sau.

## P2 — Ensemble/Stacking có điều kiện

- [x] **P2 — Đo error diversity trước khi chọn ensemble.** (`EXPERIMENTALLY VALIDATED` 2026-11-24 — Đã đo ma trận bất đồng dự đoán `disagreement_rate`, `double_fault_rate`, `cohen_kappa`, `prob_pearson_corr` và `oracle_macro_f1` trên `6,048` mẫu `val.parquet` giữa 7 kiến trúc TV4 và `EXP-N6-C` của TV2 tại `artifacts/TV4-ensemble-analysis/tv4_phase5_ensemble_report.json` & `error_diversity_heatmap.png`. Bất đồng giữa `HYBRID-RNN-01`/`CNN-01-WD`/`CHAR-02`/`EXP-N6-C` đạt `7.36%–10.76%` tương đương `445–651` mẫu bổ trợ nhau, đẩy Oracle Macro-F1 lên `0.8535–0.8697`).
- [x] **P2 — Thử voting/stacking hoặc ghi rõ lý do skip.** (`EXPERIMENTALLY VALIDATED` 2026-11-24 — Chỉ dùng xác suất hợp lệ; từ chối `LinearSVC.decision_function` thô bằng unit test. Trong nội bộ TV4: `ENS-TV4-INTRA-3` đạt `0.7834 ± 0.0058` (`+1.02%`), `ENS-TV4-INTRA-4` đạt `0.7855 ± 0.0053` (`+1.23%`), `ENS-TV4-DIVERSE-6` đạt `0.7859 ± 0.0078` (`+1.26%` ở `@0.50`) và `ENS-TV4-OOF-PLATT` đạt `0.7900 ± 0.0051` (`+1.67%` so với best single TV4 `0.7733`, vượt ngưỡng ADR `>= +1.0%`). Liên thành viên TV2+TV4: `ENS-CROSS-OOF-PLATT` đạt `0.7989 ± 0.0003` (`+1.53%` so với `EXP-N6-C-calibrated @0.50` `0.7836`, `+0.98%` so với `EXP-N6-C` `0.7892`) và `ENS-CROSS-OOF-STACKING-LR` đạt `0.8002 ± 0.0019` (`+1.10%` so với `0.7892`). Ghi rõ quyết định skip Train-OOF Stacking 90 GPU runs trong production, ưu tiên Calibrated Weighted Probability Voting theo `AGENTS.md` §6).
