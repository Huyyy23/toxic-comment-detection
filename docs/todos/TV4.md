# TODO — TV4: TheAnh — TextCNN, tuning kiến trúc, Ensemble/Stacking (EXP-04/06)

## Kế hoạch hiện hành D1–D4

Các mục G0 bên dưới là lịch sử ba lớp; giữ trạng thái đã ghi và làm lại phần cần thiết trên split nhị phân mới.

- [ ] **N1:** Duyệt spec, decisions, plan và checklist cùng nhóm.
- [ ] **N7:** Sau N4 và N5, phối hợp Huy thử BiGRU/BiLSTM với Char-CNN. Chạy ba seed, chọn checkpoint theo val Macro-F1, báo F1 TOXIC và VRAM mục tiêu dưới 7 GB.
- [ ] **N7:** Thực hiện ablation Char-CNN trên validation; chỉ thay một yếu tố mỗi lần. Lưu config, hash, checkpoint, MLflow run ID, metrics và predictions theo lớp, nguồn.
- [ ] Bàn giao validation predictions và lỗi CNN cho Huy; không gọi raw scores là xác suất calibrated.
- [ ] **N10:** Cùng nhóm đánh giá danh sách đã freeze trong một đợt locked test; không tuning theo test.

## Lịch sử G0 trước D1–D4


**Snapshot audit:** chưa thấy implementation TextCNN, config, test, run metrics, validation predictions hay checkpoints riêng cho TV4. Vì vậy EXP-04 **chưa bắt đầu/nghiệm thu**; ensemble EXP-06 cũng chưa có căn cứ vì chưa có outputs từ các model. G0 hiện OPEN, có thể bắt đầu pipeline và chạy official experiments song song với các thành viên khác.

**Quy ước chung:** dùng [Data Contract v1](../DATA_CONTRACT_v1.md), `data/processed/vihsd_g0_v1/train_val.jsonl`; 3 lớp chuẩn; embedding random-init; vocabulary fit train-only; chọn theo validation Macro-F1; locked test chỉ EXP-08. Softmax/score của CNN không tự động là probability đã calibration.

## P0 — nghiệm thu EXP-04

- [ ] **P0 — Chốt input/config của TextCNN trên G0.**
  - **Đầu ra:** config versioned cho dataset/split, vocabulary, embedding, kernel/filter/dropout, seed và output path.
  - **Nghiệm thu:** chỉ dùng `train_val.jsonl`; ghi manifest/hash; embedding random-init; không đọc locked test.
  - **Dependencies:** G0 OPEN; không cần đợi EDA TV1.

- [ ] **P0 — Tạo TextCNN single-kernel và train/evaluate entrypoint.**
  - **Đầu ra:** model/runner cho baseline single-kernel, validation metrics và prediction artifact.
  - **Nghiệm thu:** fixture smoke kiểm tra forward/train/eval, 3-class logits, checkpoint; smoke không tính là official run.
  - **Dependencies:** config G0 P0; có thể tham khảo recurrent utilities nhưng model thuộc TV4.

- [ ] **P0 — Chạy EXP-04-A: single-kernel TextCNN baseline.**
  - **Đầu ra:** config, checkpoint, history, metrics, validation predictions, MLflow run ID.
  - **Nghiệm thu:** G0 split/hash; Macro-F1/per-class/confusion, seed/compute/time/size log; artifact load lại được.
  - **Dependencies:** entrypoint P0.

- [ ] **P0 — Thêm multi-kernel TextCNN variant.**
  - **Đầu ra:** config/model cho nhiều kernel và run metrics/predictions.
  - **Nghiệm thu:** thay đổi kiến trúc được ghi rõ; cùng G0 split; không mặc định multi-kernel tốt hơn.
  - **Dependencies:** single-kernel baseline và runner.

- [ ] **P0 — Chạy biến thể thứ ba với một thay đổi có giả thuyết.**
  - **Đầu ra:** run thay đổi filters/dropout/pooling/class weighting so với baseline; config, checkpoint, metrics và predictions.
  - **Nghiệm thu:** ≥3 biến thể có ý nghĩa tổng cộng; chỉ thay đổi khai báo; cùng data hash; Macro-F1/per-class/confusion lưu đủ.
  - **Dependencies:** baseline và multi-kernel runs; validation G0.

## P1 — ablation và bàn giao

- [ ] **P1 — Tổng hợp ablation TextCNN.**
  - **Đầu ra:** bảng so sánh kernel/filter/dropout/pooling, chi phí và kết luận.
  - **Nghiệm thu:** kết luận chỉ từ validation; nêu biến thể không cải thiện/giới hạn; configs và run IDs dẫn nguồn được.
  - **Dependencies:** ba official runs.

- [ ] **P1 — Kiểm tra predictions/artifacts để ghép cross-model.**
  - **Đầu ra:** validation predictions có ID, true/predicted labels, run ID và hash; model/checkpoint có thể load.
  - **Nghiệm thu:** ID khớp G0 validation; không có test rows; score không bị gọi là calibrated probability nếu chưa hiệu chuẩn.
  - **Dependencies:** official runs; TV3 có thể nhận mỗi variant khi sẵn sàng.

- [ ] **P1 — Gửi error examples và kết luận CNN cho TV3 EXP-05.**
  - **Đầu ra:** các lỗi validation đại diện, nhóm lỗi và nhận xét CNN bắt/nhầm lớp nào.
  - **Nghiệm thu:** cùng IDs với các nhánh khác; chỉ validation; không thay thế phân tích cross-model của TV3.
  - **Dependencies:** validation predictions; các model khác có thể bàn giao sau.

## P2 — Ensemble/Stacking có điều kiện

- [ ] **P2 — Đo error diversity trước khi chọn ensemble.**
  - **Đầu ra:** bảng lỗi bổ sung/chồng lặp giữa các model có validation predictions.
  - **Nghiệm thu:** dùng cùng validation IDs; nếu không có diversity/cơ sở cải thiện, ghi lý do skip và giữ best single.
  - **Dependencies:** predictions từ ít nhất hai mô hình.

- [ ] **P2 — Thử voting/stacking hoặc ghi rõ lý do skip.**
  - **Đầu ra:** so sánh ensemble với best single về Macro-F1 và latency/overhead, hoặc quyết định skip.
  - **Nghiệm thu:** voting chỉ dùng xác suất hợp lệ; stacking meta-learner chỉ fit OOF predictions trên train; chỉ đề xuất giữ nếu đạt điều kiện ADR: validation Macro-F1 tăng ≥1% so best single.
  - **Dependencies:** diversity evidence; calibration/OOF predictions hợp lệ; không chặn EXP-04/final evaluation.
