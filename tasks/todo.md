# Project TODO — Scratch-first v4.0
**Cập nhật 2026-11-22.** Đây là trạng thái công việc; chỉ đánh dấu hoàn tất khi có artifact/bằng chứng.

**Cập nhật kế hoạch 2026-11-23:** D1–D4 thay contract ba lớp bằng dữ liệu gộp nhị phân. Các mục dưới giữ nguyên trạng thái và lịch sử G0; kết quả cũ không phải kết quả của split mới. TV1 là Tam, TV2 là Vinh, TV3 là Huy, TV4 là TheAnh, theo xác nhận ngày 2026-11-23. Deadline chưa xác định.

## Nhóm 0 — Data contract (TV3 lead; TV1 hỗ trợ) — G0 OPEN
- [x] Kiểm chứng source/revision/schema/nhãn; chấp nhận giới hạn về permission UIT và lịch sử test exposure.
- [x] Kiểm tra exact-text duplicates, conflict groups, split leakage; khóa seed 42, split 70/15/15.
- [x] Tạo manifest ID→split, hashes và contract v1; dataset chính thức: `data/processed/vihsd_g0_v1/`.
- [ ] Cả 4 thành viên xác nhận dùng đúng contract và dataset; không dùng `vihsd.jsonl` hoặc `g0_v1_candidate*`.
- [ ] Giới hạn đã biết: near-duplicate/thread leakage chưa xác minh; test không được xem để phát triển/tuning.

## Nhóm 1 — Chạy mô hình song song trên G0 (mỗi thành viên sở hữu nhánh riêng)
- [ ] **TV1 — EXP-01:** EDA; LR với raw/basic/teencode và TF-IDF word/char; tối thiểu 3 official runs.
- [ ] **TV2 — EXP-02:** LinearSVC + TF-IDF; tuning C/n-grams/feature selection; tối thiểu 3 official runs.
- [x] **TV3 — EXP-03:** scratch LSTM → BiLSTM → BiLSTM+Attention; 3 official runs completed on G0 train/validation with train-only vocabulary. No locked test used.
  - [x] Pipeline: best-checkpoint validation, data hashes, CUDA/CPU auto-selection, complete optimizer/RNG checkpoints and resume support. Artifacts: `models/recurrent_official/EXP-03/`.
- [ ] **TV4 — EXP-04:** scratch TextCNN single/multi-kernel; tối thiểu 3 official runs.
- [ ] **Bàn giao chung mỗi mô hình:** config/seed/dataset hash, MLflow run IDs, checkpoint, validation predictions, Macro-F1/per-class F1 và thời gian/size.

## Nhóm 2 — Ablation và phân tích (sau khi có validation results)
- [ ] **TV1:** preprocessing ablation và lỗi LR.
- [ ] **TV2:** SVM feature/hyperparameter ablation và training-cost comparison.
- [ ] **TV3 lead; TV1–TV4 đóng góp:** **EXP-05**, phân tích ít nhất 30 lỗi validation cross-model, nhóm lỗi và giả thuyết kiểm chứng.
- [ ] **TV4:** CNN kernel/dropout ablation.
- [ ] **TV4, có điều kiện — EXP-06:** chỉ thử voting/stacking khi validation predictions cho thấy error diversity; stacking phải dùng OOF train predictions. Nếu không phù hợp, ghi lý do skip.
- [ ] **TV3, tùy chọn — EXP-07:** PhoBERT chỉ sau nghiệm thu EXP-03; được skip nếu không đủ tài nguyên, không chặn final.

## Nhóm 3 — Chọn mô hình và đánh giá cuối
- [ ] **TV3 lead:** freeze candidate và hyperparameters bằng validation; ghi danh sách trước khi mở test.
- [ ] **TV1 điều phối, cả nhóm tham gia, TV3 duyệt — EXP-08:** đánh giá locked test đúng một đợt; không tuning/đổi candidate theo kết quả test. Báo cáo rõ test từng có exposure trong artifact cũ.
- [ ] **Cả nhóm:** reproducibility report/notebooks và 4 phần thí nghiệm cá nhân.
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
- [x] N4: Train-only TOXIC augmentation under D3; 5,675 rows yield a 30.0% train TOXIC share. Actual artifact passed N2/N4 leakage checks; see `docs/dataset-requirements.md`.
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

### Cả nhóm
- [x] N1: Cả nhóm duyệt spec, decisions, plan và todo D1–D4.
  Hoàn thành 2026-11-23 — cả nhóm duyệt D1–D4. N2 đã được mở khóa.
- [ ] Deadline: Chưa xác định.

