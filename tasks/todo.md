# Project TODO — Scratch-first v4.0
**Plan only — 2026-11-11.** Trạng thái dưới đây chỉ phản ánh **kế hoạch mới**, không suy ra code/scaffold cũ đã bị xóa hoặc đã chạy thí nghiệm.

**Cập nhật kế hoạch 2026-11-23:** D1–D4 thay contract ba lớp bằng dữ liệu gộp nhị phân. Các mục dưới giữ nguyên trạng thái và lịch sử G0; kết quả cũ không phải kết quả của split mới. TV1 là Tam, TV2 là Vinh, TV3 là Huy, TV4 là TheAnh, theo xác nhận ngày 2026-11-23. Deadline chưa xác định.

## G0 — Data Contract (TV3 nhóm trưởng)
- [ ] Kiểm chứng source/license/schema/label 3 lớp của UIT-ViHSD (TV1 hỗ trợ audit).
- [ ] Check duplicate, nhãn mâu thuẫn và leakage; quyết định official split hoặc stratified 70/15/15.
- [ ] Khóa test; lưu dataset version/hash, seed, ID→split manifest; ban hành input/output contract.
- [ ] Cả 4 thành viên xác nhận dùng chung contract.

## Parallel work — các nhánh độc lập sau G0
- [ ] **TV1 EXP-01:** EDA; raw/basic/teencode + TF-IDF word/char + Logistic Regression; ≥3 official runs.
- [ ] **TV2 EXP-02:** LinearSVC + TF-IDF, tuning C/n-grams, feature selection; ≥3 official runs. Optional RBF SVC.
- [ ] **TV3 EXP-03:** random-init LSTM → BiLSTM → BiLSTM+Attention; ≥3 official runs.
- [ ] **TV4 EXP-04:** random-init TextCNN one/multi-kernel, filters/dropout; ≥3 official runs.
- [ ] Mỗi TV xuất checkpoint/config/hash/MLflow run IDs + validation predictions + Macro-F1/per-class F1.

## Improvement and research
- [ ] TV1: preprocessing ablation và phân tích lỗi Logistic Regression.
- [ ] TV2: SVM feature/hyperparameter ablation và training-cost comparison.
- [ ] TV3 EXP-05: cross-model error analysis ≥30 validation errors, hypotheses for improvement.
- [ ] TV4: CNN kernel/dropout ablation.
- [ ] **Conditional TV4 EXP-06:** error diversity → voting/stacking (OOF nếu dùng); hoặc ghi lý do skipped.
- [ ] **Optional TV3 EXP-07:** pretrained PhoBERT fine-tuning **chỉ sau khi scratch BiLSTM hoàn thành**, hoặc ghi lý do skipped.

## Finalization
- [ ] TV3 freeze candidate models/hyperparameters bằng validation, ghi danh sách cuối.
- [ ] TV1 + cả nhóm EXP-08: chạy locked test final một đợt, không tune theo test.
- [ ] Cả nhóm: reproducibility report/notebooks, 4 phần thí nghiệm cá nhân.
- [ ] Minimal FastAPI demo `/predict`, `/health`, `/model-info`; hosting chỉ sau khi kiểm tra eligibility.

## Coordination
- [ ] Mỗi thành viên branch riêng, PR review; không sửa split độc lập.
- [ ] Nếu chung GPU 8GB: schedule DL jobs, không chạy hai job lớn đồng thời.
- [ ] Mỗi tuần review MLflow runs và artifacts, không chỉ báo cáo miệng.

Source of truth: [spec](../docs/spec.md) · [experiment plan](../docs/plans/ml-experiments.md) · [task breakdown](../docs/plans/task-breakdown.md).
**Chỉ sửa PLAN/SPEC lần này — không triển khai bất cứ TODO nào.**

## D1–D4 — Cần làm lại / bị thay thế
- [ ] G0 ba lớp và EXP-01–EXP-04 cần chạy lại theo D1/D2 trên split nhị phân mới. Các mục `[x]` phía trên vẫn là lịch sử đã hoàn thành, không chuyển thành chưa làm.
- [x] Kiểm tra lịch sử test exposure trước khi xây dựng split mới; test mới không blind vì test cũ từng lộ.
- [ ] Các công việc API-only, ensemble và TextCNN trước đây không thay thế yêu cầu React + FastAPI và neural mới. PhoBERT pretrained fine-tune đã được chấp nhận theo xác nhận người dùng.

## Checklist D1–D4 theo thành viên

### TV1 — Tam
- [x] N1: Duyệt tài liệu D1–D4 cùng nhóm.
- [x] N2: Kiểm kê nguồn, mapping, dedupe và thống kê split theo nguồn được ghi tại `docs/dataset-requirements.md`.
- [ ] N6: TF-IDF + Logistic Regression cùng TV2. Chọn checkpoint theo val Macro-F1, báo F1 TOXIC, xuất baseline calibrated `.joblib` dưới 15 MB. Phụ thuộc N2.
  - [ ] Chạy ít nhất ba seed trên split nhị phân; lưu config, hash dữ liệu, predictions, metrics và artifact.
  - [ ] Thực hiện preprocessing ablation và phân tích lỗi LR trên validation.
- [ ] N10: Điều phối cả nhóm đánh giá locked test một đợt, sau N9 và N14 nếu có. Đóng băng candidate và ba seed trước khi mở, tạo file khóa và báo cáo `locked_test.md`.

### TV2 — Vinh
- [x] N1: Duyệt tài liệu D1–D4 cùng nhóm.
- [ ] N6: TF-IDF + LinearSVC cùng TV1. Hiệu chuẩn margins, cùng protocol và điều kiện nghiệm thu N6.
  - [ ] Chạy ít nhất ba seed trên split nhị phân; lưu config, hash dữ liệu, predictions, metrics và artifact.
  - [ ] Thực hiện SVM feature/hyperparameter ablation và so sánh chi phí train trên validation.
- [ ] N10: Cùng nhóm đánh giá các candidate đã đóng băng trong một đợt test; không dùng kết quả test để tuning.

### TV3 — Huy
- [x] N1: Duyệt tài liệu D1–D4 cùng nhóm.
- [x] N2: Dataset nhị phân `data/processed/binary_n2_v1/` đã tạo; seed, hash và manifest có trong artifacts.
  - Người dùng xác nhận nhóm đã xin phép UIT; chưa có bằng chứng giấy phép trong repo. Khác biệt ngữ nghĩa nhãn và giới hạn near-duplicate được ghi tại `docs/dataset-requirements.md`.
- [ ] N7: Phụ trách LSTM, BiLSTM và BiLSTM + Attention tự huấn luyện; phối hợp TV4 cho nhánh Char-CNN sau N4 và N5. Lưu checkpoint, config, ba seed và metrics validation.
- [ ] N10: Duyệt danh sách candidate, threshold và protocol đã đóng băng trước đợt test.
- [ ] N13: TV3 phụ trách benchmark PhoBERT pretrained fine-tune đã được chấp nhận. Phụ thuộc N2, thêm N4 nếu augment. Lưu checkpoint/config/log, chạy ba seed và ghi VRAM.
- [ ] N14: Nếu có thời gian, thử learning rate, epoch và chiến lược PhoBERT pretrained fine-tune trên val sau N13. Mỗi lần đổi một yếu tố, lưu `docs/experiments/phobert_extra_val.csv`.
- [ ] N15: Lead phân tích lỗi sau N10. Viết `docs/experiments/error_analysis.md`, so sánh hai chiều, có ví dụ và số mẫu theo nguồn, độ dài và teencode.
- [x] N3: Kiểm thử leakage split N2 và augmentation giả lập đã chạy. Xem kết quả và giới hạn trong `docs/dataset-requirements.md`.
- [ ] N4: Đã mở khóa sau N3. Augment train-only TOXIC, thêm `is_augmented`, mục tiêu 30–35% nếu đạt được với giới hạn số bản. Bắt buộc chạy lại leakage tests trên augmentation thật, không dùng test để tuning.
- [ ] N5: Sau N2, train FastText bằng gensim chỉ trên train, lưu embedding và báo kích thước.
- [ ] N8: Sau N6, N7 và N13, calibration và chọn threshold trên val, báo ECE trước/sau.
- [ ] N9: Sau N7, N8 và N13, ablation trên val và so sánh với PhoBERT pretrained fine-tune. Lưu `ablation_val.csv`.
- [ ] N11: Sau N6 và N8, cập nhật inference/API/frontend cho nhị phân; test API và fallback `.joblib`.
- [ ] N12: Sau N10 và N11, Docker + HF Spaces, smoke test.

### TV4 — TheAnh
- [x] N1: Duyệt tài liệu D1–D4 cùng nhóm.
- [ ] N7: Phối hợp neural với TV3 theo vai trò recurrent và CNN hiện có. Sau N4, N5, thử BiGRU/BiLSTM + Char-CNN, checkpoint theo val Macro-F1, ghi F1 TOXIC và VRAM mục tiêu dưới 7 GB.
  - [ ] Thực hiện CNN ablation trên validation; lưu config, hash dữ liệu, ba seed và metrics.
- [ ] N10: Cùng nhóm đánh giá các candidate đã đóng băng trong một đợt test; không dùng kết quả test để tuning.

### Huy
- [ ] N14: Nếu có thời gian và giảng viên cho phép PhoBERT pretrained fine-tune, thử learning rate, epoch và chiến lược fine-tune trên val sau N13. Mỗi lần đổi một yếu tố, lưu `docs/experiments/phobert_extra_val.csv`.
- [ ] N15: Phối hợp TV3 phân tích lỗi sau N10 như trên.

### Cả nhóm hoặc chưa phân công
- [x] N1: Cả nhóm duyệt spec, decisions, plan và todo D1–D4.
  Hoàn thành 2026-11-23 — cả nhóm duyệt D1–D4. N2 đã được mở khóa.
- [ ] Deadline: Chưa xác định.

