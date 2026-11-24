# Project TODO — Scratch-first v4.0
**Cập nhật 2026-11-24.** Đây là trạng thái công việc; chỉ đánh dấu hoàn tất khi có artifact/bằng chứng.

**Cập nhật kế hoạch 2026-11-23:** D1–D4 thay contract ba lớp bằng dữ liệu gộp nhị phân. Các mục dưới giữ nguyên trạng thái và lịch sử G0; kết quả cũ không phải kết quả của split mới. TV1 là Tam, TV2 là Vinh, TV3 là Huy, TV4 là TheAnh, theo xác nhận ngày 2026-11-23. Deadline chưa xác định.

## Nhóm 0 — Data contract (TV3 lead; TV1 hỗ trợ) — G0 OPEN
- [x] Kiểm chứng source/revision/schema/nhãn; chấp nhận giới hạn về permission UIT và lịch sử test exposure.
- [x] Kiểm tra exact-text duplicates, conflict groups, split leakage; khóa seed 42, split 70/15/15.
- [x] Tạo manifest ID→split, hashes và contract v1; dataset chính thức: `data/processed/vihsd_g0_v1/`.
- [ ] Cả 4 thành viên xác nhận trực tiếp đã dùng đúng contract và dataset; hiện có evidence các nhánh dùng D1–D4 nhưng chưa đủ xác nhận riêng cho từng người.
- [ ] Giới hạn đã biết: near-duplicate/thread leakage chưa xác minh; test không được xem để phát triển/tuning.

## Nhóm 1 — Chạy mô hình song song trên G0 (mỗi thành viên sở hữu nhánh riêng)
- [ ] **TV1 — EXP-01:** EDA; LR với raw/basic/teencode và TF-IDF word/char; tối thiểu 3 official runs.
- [ ] **TV2 — EXP-02:** LinearSVC + TF-IDF; tuning C/n-grams/feature selection; tối thiểu 3 official runs.
- [x] **TV3 — EXP-03:** scratch LSTM → BiLSTM → BiLSTM+Attention; 3 official runs completed on G0 train/validation with train-only vocabulary. No locked test used.
  - [x] Pipeline: best-checkpoint validation, data hashes, CUDA/CPU auto-selection, complete optimizer/RNG checkpoints and resume support. Artifacts: `models/recurrent_official/EXP-03/`.
- [x] **TV4 — EXP-04:** scratch TextCNN single/multi-kernel; tối thiểu 3 official runs. (`EXPERIMENTALLY VALIDATED` 2026-11-24 — 18 biến thể × 3 seeds = 54 official GPU runs trên `binary_n2_v1` & `binary_n4_training_v1` tại `models/cnn/TV4-run-*/` và MLflow `TV4-CNN-Binary`).
- [x] **Bàn giao chung mỗi mô hình:** config/seed/dataset hash, MLflow run IDs, checkpoint, validation predictions, Macro-F1/per-class F1 và thời gian/size.

## Nhóm 2 — Ablation và phân tích (sau khi có validation results)
- [x] **TV1:** preprocessing ablation và lỗi LR trên validation (N6; xem `docs/TV1_N6_REPORT.md`).
- [x] **TV2:** SVM feature/hyperparameter ablation và training-cost comparison trên validation (N6; xem `docs/todos/TV2.md`).
- [x] **TV3 lead; TV1–TV4 đóng góp — EXP-05/N15 (limited technical acceptance):** Validation-only paired error analysis and AI-assisted review of the fixed 30-case sample completed from saved artifacts; the group lead limits acceptance to this scope. See `docs/experiments/error_analysis.md`, `docs/experiments/manual_review_checklist.md`, and `docs/experiments/reproducibility_report.md`.
- [ ] **EXP-05 deferred follow-up:** Member review/adjudication, privacy/legal clearance and dataset rights, plus comprehensive near-duplicate/thread leakage review remain unfinished. Do not publish the redacted validation worksheet.
- [x] **TV4:** CNN kernel/dropout ablation. (`EXPERIMENTALLY VALIDATED` 2026-11-24 — 6 vòng ablation kernel/filter/dropout/weight_decay/imbalance/augmentation/char-cnn/hybrid tại `docs/TV4_CNN_REPORT.md` & `artifacts/TV4-cnn-analysis/`).
- [x] **TV4, có điều kiện — EXP-06:** chỉ thử voting/stacking khi validation predictions cho thấy error diversity; stacking phải dùng OOF train predictions. Nếu không phù hợp, ghi lý do skip. (`EXPERIMENTALLY VALIDATED` 2026-11-24 — đo Error Diversity `disagreement_rate = 6.13%–9.62%`, `Oracle Macro-F1 = 0.8535–0.8693`; `ENS-TV4-DIVERSE-6` đạt Val Macro-F1 `0.7859` (`+1.26%`), `ENS-TV4-OOF-PLATT` đạt `0.7900` (`+1.67%`), `ENS-CROSS-OOF-PLATT` đạt `0.7989`; ghi rõ quyết định skip 90-run Train-OOF Stacking trong production tại `artifacts/TV4-ensemble-analysis/`).
- [x] **TV3, tùy chọn — EXP-07:** PhoBERT benchmark đã chạy theo N13; artifact và metrics validation tại `docs/todos/TV3.md`.

## Nhóm 3 — Chọn mô hình và đánh giá cuối
- [ ] **TV3 lead:** freeze candidate và hyperparameters bằng validation; ghi danh sách trước khi mở test.
- [x] **TV1 điều phối, cả nhóm tham gia, TV3 duyệt — EXP-08/N10:** `CLOSED — LIMITED HISTORICAL ACCEPTANCE`, based on the user's report that the group confirms N10. Historical/test-exposure and protocol limitations remain in `artifacts/TV4-locked-test/tv4_n10_reconciliation.md`; this is not blind-test `PASS`. Individual member statements were not attached; don't infer named approvals.
- [x] **Cả nhóm:** reproducibility report/notebook validation-only và 4 phần báo cáo cá nhân dạng draft đã tổng hợp. 16/16 V3 retrained checkpoints có artifacts/validation comparisons; inference replay exact-label passed, max probability diff `0.00011591`. Xem `docs/experiments/reproducibility_report.md`, `notebooks/EXP-05-reproducibility-analysis.ipynb`, `docs/experiments/member_report_sections.md`, `docs/experiments/manual_review_checklist.md`.
- [ ] **Cả nhóm:** Minimal FastAPI `/predict`, `/health`, `/model-info`; chỉ phục vụ model đã chọn, hosting sau khi kiểm tra eligibility.

## Nhóm 4 — Phối hợp
- [ ] Mỗi thành viên dùng branch riêng và PR review; không tự sửa split hoặc contract.
- [ ] Nếu dùng chung GPU 8GB, lên lịch DL jobs; tránh chạy hai job lớn đồng thời.
- [ ] Review MLflow runs/artifacts hàng tuần; ghi trạng thái theo bằng chứng.

Nguồn chi tiết: [spec](../docs/spec.md) · [experiment plan](../docs/plans/ml-experiments.md) · [task breakdown](../docs/plans/task-breakdown.md) · [G0 acceptance](../docs/G0_FINAL_REVIEW.md).

## D1–D4 — Cần làm lại / bị thay thế

- [ ] G0 ba lớp và EXP-01–EXP-04 cần chạy lại theo D1/D2 trên split nhị phân mới. Các mục `[x]` phía trên vẫn là lịch sử đã hoàn thành, không chuyển thành chưa làm.
- [x] Kiểm tra lịch sử test exposure trước khi xây dựng split mới; test mới không blind vì test cũ từng lộ.
- [ ] Các công việc API-only, ensemble và TextCNN trước đây không thay thế yêu cầu React + FastAPI và neural mới. PhoBERT pretrained fine-tune đã được chấp nhận theo xác nhận người dùng.

## Checklist D1–D4 theo thành viên

### TV1 — Tam
- [x] N1: Duyệt tài liệu D1–D4 cùng nhóm.
- [x] N2: Kiểm kê nguồn, mapping, dedupe và thống kê split theo nguồn được ghi tại `docs/dataset-requirements.md`.
- [x] N6: TF-IDF + Logistic Regression cùng TV2. Chọn checkpoint theo val Macro-F1, báo F1 TOXIC, xuất baseline calibrated `.joblib` dưới 15 MB. Phụ thuộc N2.
  - [x] Chạy ít nhất ba seed trên split nhị phân; lưu config, hash dữ liệu, predictions, metrics và artifact.
  - [x] Thực hiện preprocessing ablation và phân tích lỗi LR trên validation.
- [ ] N10: Cùng nhóm duyệt danh mục candidate chỉ dựa trên validation và giới hạn exposure/phạm vi lịch sử; không mở test hoặc đánh giá lại. Chỉ xác nhận nghiệm thu có giới hạn qua checklist chung: `artifacts/TV4-locked-test/tv4_n10_reconciliation.md`.

### TV2 — Vinh
- [x] N1: Duyệt tài liệu D1–D4 cùng nhóm.
- [x] N6: TF-IDF + LinearSVC cùng TV1. Hiệu chuẩn margins, cùng protocol và điều kiện nghiệm thu N6.
  - [x] Chạy ít nhất ba seed trên split nhị phân; lưu config, hash dữ liệu, predictions, metrics và artifact.
  - [x] Thực hiện SVM feature/hyperparameter ablation và so sánh chi phí train trên validation.
- [ ] N10: Xác nhận nhóm chấp nhận phạm vi/exposure test lịch sử và giới hạn tái lập đã ghi; không tune bằng test hoặc mở lại test. Ghi xác nhận trong checklist chung.

### TV3 — Huy
- [x] N1: Duyệt tài liệu D1–D4 cùng nhóm.
- [x] N2: Dataset nhị phân `data/processed/binary_n2_v1/` đã tạo; seed, hash và manifest có trong artifacts.
  - Người dùng xác nhận nhóm đã xin phép UIT; chưa có bằng chứng giấy phép trong repo. Khác biệt ngữ nghĩa nhãn và giới hạn near-duplicate được ghi tại `docs/dataset-requirements.md`.
- [x] N7: Huy hoàn tất recurrent (LSTM/BiLSTM/Attention); TV4 hoàn tất Char-CNN. Artifacts và metrics validation được ghi trong checklist TV3/TV4.
- [x] N10: `CLOSED — LIMITED HISTORICAL ACCEPTANCE` theo xác nhận nhóm do người dùng báo lại; chưa có sign-off cá nhân kèm bằng chứng. Không phải blind-test PASS, không mở lại test.
- [x] N13: TV3 hoàn tất benchmark PhoBERT pretrained fine-tune đã được chấp nhận; checkpoint/config/log, ba seed và VRAM được ghi tại `docs/todos/TV3.md`.
- [x] **N14:** Completed PhoBERT N2 validation-only LR comparison (1e-5 and 3e-5, 3 seeds each; only LR changed). Mean validation Macro-F1: `0.793272 ± 0.002131` and `0.800336 ± 0.004193`, respectively; N13 default `2e-5` remains higher at `0.801469 ± 0.003773`. No alternative promoted; evidence: `docs/experiments/phobert_extra_val.csv`. No locked-test access.
- [x] N15 (limited technical scope): Validation-only paired error analysis and AI-assisted review of the fixed 30-case sample completed; no test accessed and no production model selected. Scope decision/evidence: `docs/experiments/error_analysis.md` and `manual_review_checklist.md`.
  - [ ] Deferred beyond limited acceptance: human review/adjudication, privacy/legal/data-rights clearance, and comprehensive near-duplicate/thread leakage review. The worksheet stays internal.
- [x] N3: Kiểm thử leakage split N2 và augmentation giả lập đã chạy. Xem kết quả và giới hạn trong `docs/dataset-requirements.md`.
- [x] N4: Train-only TOXIC augmentation under D3; 5,675 rows yield a 30.0% train TOXIC share. Actual artifact passed N2/N4 leakage checks; see `docs/dataset-requirements.md`.
- [x] N5: FastText train-only trên N2, 3 seeds; validation Macro-F1 mean 0.622218, population std 0.001276. Config, hashes, models, predictions, metrics và MLflow runs ở `models/baselines/N5-run-4cef5024bbdb437296384e56b471e20b/`.
- [x] N8: Calibration và chọn threshold chỉ dùng validation. LinearSVC raw chỉ xuất decision margin, không phải probability; N6 rerun đã bổ sung raw model weights có thể load lại, nhưng khi cần calibrated probability chỉ dùng SVM đã calibration. Bằng chứng rerun: `docs/TV2_N6_REPORT.md` §5.
- [x] N9: Validation-only ablations and PhoBERT comparison complete; 63-row evidence and protocol/results at `artifacts/N9/README.md` and `artifacts/N9/ablation_val.csv`.
- [x] N11: Hoàn tất inference React + FastAPI local với adapter baseline N6-LR-D; giữ contract CLEAN/TOXIC, chưa deploy public/chốt production trước N10. Bằng chứng: `docs/todos/TV3.md`.
- [ ] N12: **Preparation only; public deployment not cleared.** N11 local evidence exists. Before launch: evidence required acknowledgement of N10's limited historical exposure; reverify HF account/Space eligibility; resolve truthful disclosure for uncalibrated N6-LR-D; freeze production candidate/artifact; reconcile conflicting deployment-contract docs; and pass container/security/data-exclusion smoke checks. See `docs/todos/TV3.md`; no deployment authorized.

### TV4 — TheAnh
- [x] N1: Duyệt tài liệu D1–D4 cùng nhóm.
- [x] N7: Phối hợp neural với TV3 theo vai trò recurrent và CNN hiện có. Sau N4, N5, thử BiGRU/BiLSTM + Char-CNN, checkpoint theo val Macro-F1, ghi F1 TOXIC và VRAM mục tiêu dưới 7 GB. (`HYBRID-RNN-01` BiGRU+Char-CNN đạt Val Macro-F1 `0.7733 ± 0.0072`, F1-TOXIC `0.6149 ± 0.0130`, best seed 42 `0.7806`, VRAM `230.8 MB << 7 GB`; `HYBRID-RNN-WD` đạt `0.7731 ± 0.0052`; `HYBRID-BILSTM-01` đạt `0.7665 ± 0.0027`, VRAM `230.6 MB`).
  - [x] Thực hiện CNN ablation trên validation; lưu config, hash dữ liệu, ba seed và metrics. (18 biến thể × 3 seeds = 54 runs tại `models/cnn/TV4-run-*/`, báo cáo và biểu đồ tại `artifacts/TV4-cnn-analysis/`, `notebooks/EXP-04-cnn-analysis.ipynb` và `docs/TV4_CNN_REPORT.md`).
- [x] TV4 — historical N10 batch: frozen before the recorded one-shot test; scope is TV4 singles, one TV2 reference, and TV4/TV2+TV4 ensembles (test `Macro-F1` range `0.7855–0.8128`). This does not establish team-wide N10 completion; see `artifacts/TV4-locked-test/tv4_n10_reconciliation.md`. Do not reopen locked test.

### Cả nhóm
- [x] N1: Cả nhóm duyệt spec, decisions, plan và todo D1–D4.
  Hoàn thành 2026-11-23 — cả nhóm duyệt D1–D4. N2 đã được mở khóa.
- [ ] Deadline: Chưa xác định.

