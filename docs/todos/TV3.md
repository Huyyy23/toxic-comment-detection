# TODO — TV3: Huy (nhóm trưởng) — Data Contract, recurrent, Error Analysis, tích hợp

## Kế hoạch hiện hành D1–D4

Các mục G0 bên dưới là lịch sử ba lớp. `tasks/todo.md` ghi ba official EXP-03 runs đã hoàn thành trên G0 train/validation; các run này cần chạy lại cho split nhị phân, không xóa lịch sử.

- [x] **N1:** Duyệt spec, decisions, plan và checklist cùng nhóm.
- [x] **N2:** Cùng Tam kiểm tra nguồn, permission UIT và mapping; gộp ViHSD/ViCTSD, tạo nhãn CLEAN/TOXIC và `source`, dedupe trong và giữa nguồn, split 70/15/15 stratified theo `(source,label)`. Lưu Parquet, manifest, seed, hash và thống kê split × source. Tình trạng xác nhận quyền UIT và giới hạn về nhãn được ghi tại `docs/dataset-requirements.md`.
- [x] **N3:** Đã kiểm tra split N2 và augmentation giả lập. Xem `docs/dataset-requirements.md` về kết quả và giới hạn near-duplicate/thread.
- [x] **N4:** Tạo augmentation TOXIC train-only theo D3, artifact 5.675 dòng đạt 30,0% TOXIC; leakage tests trên dữ liệu thật pass (10 passed). Xem hash, provenance và giới hạn trong `docs/dataset-requirements.md`.
- [x] **N5:** Train FastText chỉ trên N2 train; 3 seeds 42/123/2026. Validation Macro-F1 0.622207/0.623787/0.620661, mean 0.622218, population std 0.001276. Model, embedding, config, hash, predictions, metrics và MLflow runs tại `models/baselines/N5-run-4cef5024bbdb437296384e56b471e20b/`; FastText mỗi seed khoảng 10.85 MB.
- [x] **N8:** Hoàn tất calibration và threshold validation cho 90 runs (LR/SVM, FastText, recurrent, CNN và PhoBERT); ECE trước/sau, per-run thresholds và calibrated probabilities tại `artifacts/N8/`. Các metric là in-sample validation diagnostics.
- [x] **N9:** Hoàn tất validation ablations sau N7, N8 và N13; tổng hợp 63 dòng tại `artifacts/N9/ablation_val.csv`, protocol/kết quả tại `artifacts/N9/README.md`. Không dùng locked test.
  - [x] Controlled BiLSTM+Attention random-init vs N5 FastText-init trên N4, cùng config 100d và ba seed: Macro-F1 mean `0.584855` vs `0.682165`; paired Δ `+0.097309 ± 0.022045`. F1 TOXIC mean `0.438565` vs `0.512659`; paired Δ `+0.074094 ± 0.015478`. Threshold/ECE fit trên cùng validation là in-sample diagnostics.
- [x] **N11:** Local React + FastAPI, adapter N6-LR-D raw; `GET /health`, `GET /model-info`, `POST /predict` thành công. `pytest tests/test_inference_api.py -q`: 1 passed; `npm run build`: pass; Playwright E2E nhập bình luận, nhận kết quả CLEAN (95%/5%), backend sẵn sàng, 0 browser errors. Fallback tái tạo cục bộ từ config N6 chính thức, seed 42, 3.2625 MB; `is_calibrated=false`, threshold 0.5. Chưa dùng locked test, deploy public hay chốt production.
- [ ] **N12 — PREPARATION ONLY; public deploy not cleared.** Local N11 endpoints/API/build/E2E are documented as passed. Before launch: record required acknowledgement of N10's limited historical exposure; recheck HF account/Space eligibility (repo report dated 2026-11-11 says token pending); disclose N6-LR-D probabilities as uncalibrated (`is_calibrated=false`, threshold 0.5); freeze candidate/artifact; reconcile binary React/FastAPI contract across deployment docs; and pass container, security and data-exclusion smoke checks. N15 acceptance does not satisfy these gates. No deploy in this task.
- [x] **N7 overall:** TV3 recurrent branch and TV4 Char-CNN are complete; see `docs/todos/TV4.md` for Char-CNN evidence.
  - [x] **TV3 recurrent:** LSTM, BiLSTM, BiLSTM + Attention with random-init embeddings; N4 binary train/validation; three seeds (42, 123, 2026), max 30 epochs with early stopping (`patience=3`), best checkpoint by validation Macro-F1. No test data used.

    | Model | Macro-F1 seeds (42 / 123 / 2026) | Mean ± population SD | Mean F1 CLEAN | Mean F1 TOXIC | MLflow run IDs |
    |---|---|---:|---:|---:|---|
    | LSTM | 0.655419 / 0.632095 / 0.678985 | 0.655499 ± 0.019143 | 0.818278 | 0.492721 | `189d4f9e`, `5cb3386d`, `84454486` |
    | BiLSTM | 0.645421 / 0.633061 / 0.670050 | 0.649511 ± 0.015375 | 0.811610 | 0.487412 | `522ee1d8`, `c9829129`, `535dae51` |
    | BiLSTM + Attention | 0.625645 / 0.578585 / 0.630971 | 0.611734 ± 0.023540 | 0.767139 | 0.456328 | `1e1a3744`, `27950743`, `319745b3` |

    Artifacts and full MLflow run IDs: `models/recurrent_n7/mlruns/` (experiment `EXP-03-official`). All nine runs used identical N4 train/validation hashes; each has checkpoint, config, metrics, and validation predictions.
- [x] **TV4 Char-CNN:** completed; validation runs and artifacts are documented in `docs/todos/TV4.md`.
- [x] **N13:** PhoBERT pretrained binary benchmark hoàn tất; cùng validation qua hai gói N2/N4 và ba seed 42/123/2026; best checkpoint chọn theo validation Macro-F1; locked test chưa mở.
  - [x] **N2 baseline, không augmentation:** `vinai/phobert-base`, 4 epoch tối đa, patience 2, max length 160, batch 8 × accumulation 2, FP16; PyVi 0.1.1, Transformers 4.57.6, model revision `01daacda68afe13d83023d16ec647239e344a1e6`.

    | Seed | Macro-F1 | F1 CLEAN | F1 TOXIC | Best epoch | Skipped optimizer steps | Peak VRAM (bytes) | MLflow run ID |
    |---:|---:|---:|---:|---:|---:|---:|---|
    | 42 | 0.796222 | 0.938577 | 0.653867 | 4 | 7 | 2,946,551,808 | `462781192f5d4c46bd41c7e035203923` |
    | 123 | 0.804934 | 0.937365 | 0.672503 | 3 | 7 | 2,952,196,096 | `79180ecaf6f94579896419362f0fc8cb` |
    | 2026 | 0.803251 | 0.941314 | 0.665188 | 3 | 5 | 2,953,908,224 | `eeda02eadf084fb59ceabbf3da0924c4` |
    | Mean ± population SD | 0.801469 ± 0.003773 | 0.939085 ± 0.001652 | 0.663853 ± 0.007666 | — | — | — | — |

    N2 train SHA256 `097b32bea731bf1a4771a8485155d339a96027d2819fe8a38aa2c9d2e2f32a73`; validation SHA256 `4c33430a520818078aef6bcc34b3e2573873445dc7040c5729313e11bb60197a`. Artifacts/config/checkpoints/predictions are in local MLflow `models/N13-phobert/mlruns/`; only train and validation were read.
  - [x] **N4 augmentation ablation:** cùng PhoBERT revision/config và validation với baseline N2.

    | Seed | Macro-F1 | F1 CLEAN | F1 TOXIC | Best epoch | Skipped optimizer steps | Peak VRAM (bytes) | MLflow run ID |
    |---:|---:|---:|---:|---:|---:|---:|---|
    | 42 | 0.786239 | 0.933412 | 0.639066 | 4 | 6 | 2,950,361,088 | `fdb1a801f08947fd81635f891713cd65` |
    | 123 | 0.793745 | 0.936603 | 0.650888 | 2 | 6 | 2,944,192,512 | `cf1d45b2d7244957a948647b64a1666f` |
    | 2026 | 0.795027 | 0.934827 | 0.655226 | 4 | 6 | 2,949,136,384 | `87df4ae80b434a37ae266ba5318c045c` |
    | Mean ± population SD | 0.791670 ± 0.003876 | 0.934947 ± 0.001305 | 0.648393 ± 0.006829 | — | — | — | — |

    N4 train SHA256 `f57fbf0a55de6b08c8cba1cd366c3895bfed495f2817069f892ba27b5786542e`; validation hash matches N2 (`4c33430a520818078aef6bcc34b3e2573873445dc7040c5729313e11bb60197a`). Compared with N2, N4 means are lower by 0.009799 Macro-F1, 0.004138 F1 CLEAN and 0.015460 F1 TOXIC on validation; descriptive result only. Initial N4 seed-42 run `8bc3a002797444c491cabf20945f5288` remains tagged `INVALID`; initial N2 runs remain tagged `SUPERSEDED`.
- [x] **N14:** Validation-only PhoBERT learning-rate comparison on N2 completed; 3 seeds per setting, only LR changed. Macro-F1 mean ± population SD: `1e-5` = `0.793272 ± 0.002131`; `3e-5` = `0.800336 ± 0.004193`; N13 baseline `2e-5` = `0.801469 ± 0.003773`. No alternative LR promoted. Six run IDs, hashes, F1 by class, checkpoint/prediction hashes, and the failed timeout/retry are recorded in `docs/experiments/phobert_extra_val.csv`. Validation-only; no locked-test access.
- [x] **N10:** `CLOSED — LIMITED HISTORICAL ACCEPTANCE` theo xác nhận nhóm do người dùng báo lại; reconciliation hậu nghiệm, không blind test và không mở lại test. Chưa có sign-off cá nhân kèm bằng chứng; xem `artifacts/TV4-locked-test/tv4_n10_reconciliation.md`.
- [x] **N15 — limited technical scope accepted by group lead:** Validation-only paired error analysis and AI-assisted review of the fixed 30-case sample are complete from existing artifacts. The 16/16 V3 retrain/replay remains separately documented in `reproducibility_report.md`; it does not establish gold adjudication, member sign-off or privacy/data clearance. Deferred human/data tasks remain open in `tasks/todo.md` and `manual_review_checklist.md`.

## Lịch sử G0 trước D1–D4


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
