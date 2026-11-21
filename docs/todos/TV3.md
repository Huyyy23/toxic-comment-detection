# TODO — TV3 (nhóm trưởng): Data Contract, recurrent, Error Analysis, tích hợp

**Snapshot audit:** `docs/DATA_CONTRACT_v1.md` được user duyệt; G0 acceptance PASS và Gate OPEN theo `docs/G0_FINAL_REVIEW.md`. Recurrent model/trainer/checkpoint/loader/tracking/report đã có và từng được test trên fixture CPU theo `docs/TV3_SESSION_REPORT.md`; hiện có thêm official driver/config cho G0. Chưa có bằng chứng official EXP-03 training runs/checkpoints. EXP-05 và EXP-07 chưa có kết quả nghiên cứu.

**Quy ước chung:** canonical dataset `data/processed/vihsd_g0_v1/`; development chỉ nhận `train_val.jsonl`; label mapping `0=CLEAN, 1=OFFENSIVE, 2=HATE`; fit vocabulary trên train; chọn theo validation Macro-F1. Không đọc locked test trước EXP-08. Báo cáo test phải khai báo historical exposure; không nói test blind.

## P0 — governance và EXP-03

- [x] **P0 — Gate G0 và Data Contract v1 được duyệt.**
  - **Đầu ra:** `docs/DATA_CONTRACT_v1.md`, `docs/G0_FINAL_REVIEW.md`, candidate `data/processed/vihsd_g0_v1/`.
  - **Nghiệm thu:** acceptance checks PASS; gate OPEN; train/val được dùng cho official training. Lịch sử test exposure, permission và near-duplicate limitations vẫn được công bố.
  - **Dependencies:** hoàn tất; không lặp lại audit/rebuild nếu không có bằng chứng lỗi mới.

- [ ] **P0 — Thông báo canonical path/protocol và thu xác nhận của TV1–TV4.**
  - **Đầu ra:** xác nhận từng nhánh dùng G0 path/hash, label mapping, train-only rules, validation Macro-F1 và prediction fields.
  - **Nghiệm thu:** ghi lại đủ bốn xác nhận; không tự ý đổi contract hoặc dùng legacy candidate.
  - **Dependencies:** G0 đã OPEN; không cần chờ EDA TV1.

- [ ] **P0 — Xác minh recurrent loader/driver đọc đúng contract G0.**
  - **Đầu ra:** bằng chứng train/val IDs/counts và data hash từ `train_val.jsonl` + manifest; kiểm tra config/CLI trước official runs.
  - **Nghiệm thu:** chỉ nạp train/val, không load `test_locked.jsonl`; 23,002 train và 4,929 val; vocabulary fit train-only; fixture smoke được ghi riêng, không tính là nghiên cứu.
  - **Dependencies:** G0; `src/models/train_recurrent_official.py`, `configs/recurrent_official.json`.

- [ ] **P0 — Chạy EXP-03 LSTM baseline (random initialization).**
  - **Đầu ra:** config, checkpoint, history/curves, metrics, validation predictions và MLflow run ID.
  - **Nghiệm thu:** train G0; ghi seed/hash/compute, Macro-F1/per-class/confusion; checkpoint load lại được.
  - **Dependencies:** loader/driver validation; cùng model code có sẵn.

- [ ] **P0 — Chạy EXP-03 BiLSTM.**
  - **Đầu ra:** artifacts tương ứng với baseline LSTM.
  - **Nghiệm thu:** cùng data hash/protocol; biến thể bidirectional được khai báo; kết quả không suy diễn rằng BiLSTM tốt hơn nếu metric không chứng minh.
  - **Dependencies:** driver P0; có thể chạy sau hoặc schedule riêng từ LSTM.

- [ ] **P0 — Chạy EXP-03 BiLSTM + Attention.**
  - **Đầu ra:** artifacts tương ứng với các run recurrent còn lại.
  - **Nghiệm thu:** cùng data hash/protocol; attention variant được khai báo; metrics/predictions và checkpoint đọc lại được.
  - **Dependencies:** driver P0; phối hợp lịch GPU với TV4 nếu dùng chung thiết bị.

## P1 — Error Analysis và tích hợp

- [ ] **P1 — Ghép validation predictions cho EXP-05.**
  - **Đầu ra:** bảng predictions các nhánh đã bàn giao, căn chỉnh theo cùng G0 validation IDs.
  - **Nghiệm thu:** ID/label khớp; loại run không cùng data hash; không đưa locked-test records vào phân tích.
  - **Dependencies:** predictions từ EXP-01…04; có thể tích lũy khi từng TV bàn giao.

- [ ] **P1 — Review tối thiểu 30 lỗi và lập taxonomy/hypotheses.**
  - **Đầu ra:** ít nhất 30 validation errors, nhóm CLEAN/OFFENSIVE/HATE và hiện tượng ngôn ngữ; hypothesis có thể kiểm chứng.
  - **Nghiệm thu:** nêu evidence, không suy rộng từ vài ví dụ; không chia sẻ nội dung nhận diện cá nhân/locked-test examples; phân biệt quan sát và giả thuyết.
  - **Dependencies:** task ghép predictions; đầy đủ cross-model comparison sau khi EXP-01…04 bàn giao.

- [ ] **P1 — Tích hợp artifacts và freeze candidates bằng validation.**
  - **Đầu ra:** inventory owner/run/config/data hash/metrics/artifact path/prediction schema; danh sách model/ensemble candidates đã freeze.
  - **Nghiệm thu:** candidate truy vết/load được; chọn chỉ bằng validation; nêu rõ deliverable thiếu hoặc nhánh chưa nghiệm thu.
  - **Dependencies:** bàn giao TV1–TV4; EXP-05 và EXP-06 nếu ensemble được xem xét.

- [ ] **P1 — Giám sát EXP-08 one-shot locked-test evaluation.**
  - **Đầu ra:** báo cáo final Macro-F1/per-class/confusion/latency/model size và test limitations.
  - **Nghiệm thu:** mở test sau freeze, chỉ chạy danh sách đã định trước; không tuning/chọn threshold/sửa hypothesis sau khi xem; không tuyên bố test blind.
  - **Dependencies:** candidate freeze; TV1 điều phối theo experiment plan, TV3 review.

## P2 — benchmark tùy chọn

- [ ] **P2 — Quyết định khả thi EXP-07 PhoBERT hoặc ghi rõ skip.**
  - **Đầu ra:** đánh giá budget/GPU/model revision; kế hoạch run hoặc lý do không chạy.
  - **Nghiệm thu:** chỉ dùng `vinai/phobert-base-v2` pretrained base, ghi rõ benchmark transfer learning; không trì hoãn scratch models/final evaluation.
  - **Dependencies:** EXP-03 scratch đã nghiệm thu.

- [ ] **P2 — Fine-tune PhoBERT nếu được chọn.**
  - **Đầu ra:** config/revision, run ID, metrics, compute/VRAM, latency/size và validation predictions.
  - **Nghiệm thu:** cùng G0 split và validation protocol; không dùng toxic-finetuned checkpoint; ghi kết quả dù không vượt scratch.
  - **Dependencies:** quyết định khả thi ở task trước; tài nguyên truy cập/máy phù hợp.

## Ghi chú audit

- `docs/G0_TV1_REVIEW_CHECKLIST.md` và các đoạn trạng thái pending trong `docs/tv1_g0_report.md` là tài liệu lịch sử; không ghi đè quyết định mới. Dùng `docs/TEAM_WORK_CONTEXT.md`, `docs/DATA_CONTRACT_v1.md`, `docs/G0_FINAL_REVIEW.md` làm nguồn trạng thái G0.
- UIT permission/original license chưa xác minh; near-duplicate/thread leakage chưa chứng nhận; test từng có historical exposure. Đây là giới hạn phải công bố, không đóng Gate OPEN theo review hiện hành.
