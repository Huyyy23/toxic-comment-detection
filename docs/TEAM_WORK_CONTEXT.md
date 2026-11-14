# TEAM WORK CONTEXT — Scratch-first v4

> Tra cứu nhanh cho Agent; không thay thế SPEC/Plan. Cập nhật: **2026-11-14**. Nhóm 1–2 đã kiểm thử fixture; nhóm 3 tích hợp tracking/report, không official training/dataset.
> **Nguồn đã đối chiếu:** GitHub `master` tại `bd9602300616cf0f01d5b4f04a463cb0148181be` (SPEC/Plan v4, 2026-11-11), yêu cầu nhóm trưởng trong phiên này; local HEAD `c2270f4344112fa6f68256b251e96ae6b1a7d853` còn SPEC/Plan v3. Khi hai bản khác nhau, dùng nguồn v4 được ghim ở §12 và yêu cầu mới nhất; không lấy EXP ID v3 để triển khai.
> Báo cáo bàn giao phiên, lệnh kiểm thử tổng hợp và lưu ý commit/môi trường: [`TV3_SESSION_REPORT.md`](TV3_SESSION_REPORT.md). Đọc khi tiếp quản hoặc chuẩn bị tích hợp nhánh; không thay contract trong context này.

## 1. Project Overview

Đồ án cuối kỳ ML: phân loại tiếng Việt **CLEAN=0 / OFFENSIVE=1 / HATE=2**. Bốn nhánh chính tự train: LR, SVM, recurrent DL, TextCNN. Hướng 2 là cách nghiên cứu/trình bày: baseline, vấn đề, giả thuyết, thay đổi có kiểm soát, kết quả, kết luận; bảng đối sánh là bằng chứng bổ trợ. Không mặc định mô hình phức tạp hơn sẽ thắng.

Research là trọng tâm; FastAPI tối giản chỉ phục vụ model được chọn: `POST /predict`, `GET /health`, `GET /model-info`, Swagger `/docs`. Không frontend, auth, DB người dùng hoặc dashboard. Chưa có bằng chứng train/deploy đạt nghiệm thu.

## 2. Team Responsibilities

| Thành viên | Trách nhiệm chính | Phối hợp / quyền duyệt |
|---|---|---|
| TV1 | EDA, preprocessing, word/char TF-IDF, class weighting, Logistic Regression | Chuẩn bị dữ liệu chung theo contract; review audit G0; điều phối final test |
| TV2 | LinearSVC, feature selection, tuning; RBF SVM nếu khả thi | Dùng cùng dữ liệu/features contract với TV1; đóng góp validation errors |
| TV3 — nhóm trưởng | LSTM, BiLSTM, BiLSTM+Attention scratch; chủ trì error analysis; optional PhoBERT | Sở hữu G0/data contract theo v4; duyệt thay đổi contract, tính nhất quán kỹ thuật, dependencies, tiến độ và final evaluation |
| TV4 | TextCNN scratch, single/multi-kernel, pooling/filters/regularization; ensemble có điều kiện | Đóng góp errors; phối hợp lịch GPU; hỗ trợ demo/API theo plan |

**G0 không chuyển EDA chuyên sâu sang TV3:** TV3 chỉ điều phối/xác minh quản trị dữ liệu tối thiểu trước phát hành contract; TV1 vẫn sở hữu EDA và preprocessing. Phiên này không thực hiện audit dữ liệu hay phát hành split.

## 3. Experiment Ownership Matrix

Trạng thái hiện tại xem §9; đây là ownership/acceptance, không phải kết quả đã đạt.

| EXP | Owner | Nội dung / dependencies | Deliverables và điều kiện nghiệm thu chính |
|---|---|---|---|
| EXP-01 | TV1 | LR + preprocessing/TF-IDF; official runs cần G0 | EDA; ≥3 biến thể raw/basic/teencode, word/char/class weight; validation metrics, pipeline, predictions, ablation |
| EXP-02 | TV2 | LinearSVC/feature selection/tuning; G0; phối hợp TV1 | ≥3 biến thể; configs/grid results, SVM artifact, val predictions, compute comparison; RBF tùy khả thi |
| EXP-03 | TV3 | LSTM → BiLSTM → BiLSTM+Attention; G0 trước official training | ≥3 official architecture runs; random embeddings, train-only vocabulary; diagrams, loss curves, metrics, checkpoints/config, val predictions, ablation |
| EXP-04 | TV4 | TextCNN single/multi-kernel; G0 | ≥3 biến thể random-init; kernels/filters/pooling/dropout ablation; curves, val metrics, checkpoints/predictions |
| EXP-05 | TV3 chủ trì; mọi TV góp dữ liệu | Cross-model error analysis; EXP-01…04 validation outputs | ≥30 validation predictions sai được review theo taxonomy; tỷ lệ lỗi theo lớp, hypotheses kiểm chứng được |
| EXP-06 | TV4 chủ trì | Optional ensemble; ≥2 model predictions và bằng chứng error diversity | Soft/weighted voting xác suất hợp lệ hoặc train-fold OOF stacking; so với best single cả chất lượng/overhead; lưu kết quả hoặc lý do bỏ qua |
| EXP-07 | TV3 | Optional pretrained benchmark; **EXP-03 đạt nghiệm thu + G0** | Fine-tune `vinai/phobert-base-v2` base, không dùng toxic-finetuned checkpoint; so với bốn nhánh scratch; Macro-F1/time/VRAM/latency/size, base revision, init head và budget |
| EXP-08 | TV1 điều phối; TV3 duyệt | EXP-01…04, danh sách candidates và mọi lựa chọn đã freeze | Một đợt locked-test cho fixed candidates; per-class/Macro-F1/confusion/latency/size; không tuning ngược |

Ngoài ma trận yêu cầu: v4 còn **EXP-00 = G0** (TV3 lead, TV1 review) và **EXP-09 = báo cáo/tái lập/demo** (cả nhóm). Giữ nguyên ID; không đổi EXP-00 thành EXP-01 hay dùng EXP-009 cũ cho final.

## 4. Technical Decisions & Constraints

- Bốn nhánh chính **fit-from-scratch/random-init**; EXP-03/04 không pretrained embeddings/weights. PhoBERT chỉ EXP-07, không thay EXP-03, không chặn nghiệm thu scratch.
- Tracking: **MLflow local file store**; không cần server. Optuna tùy chọn, không là dependency bắt buộc.
- Mỗi người ≥3 cấu hình có ý nghĩa và ≥1 ablation; lưu cả thất bại. Không tính sửa lỗi chạy thành biến thể nghiên cứu.
- `.py` giữ logic tái sử dụng; notebook import modules, đọc run thật, trình bày curves/metrics. Chưa train thì ghi **CHƯA THỰC HIỆN**, không vẽ kết quả giả.
- Config/hardware values còn provisional; RTX 4060 8GB là mục tiêu, không phải bằng chứng đã profile. Device/batch/sequence length/seed/paths cấu hình được; mixed precision chỉ khi hỗ trợ.
- Nếu chung GPU 8GB: phát triển song song, xếp lịch DL jobs; không chạy hai job lớn đồng thời. Checkpoints nặng ngoài Git, metadata/version truy vết được.
- Không auto-download dữ liệu hoặc train lúc API startup. Docker chỉ khi hosting cần; deployment/hosting chưa được chứng minh ở đây.
- **Phạm vi hiện tại (2026-11-14):** người dùng cho phép nhóm 3: MLflow local, report helpers/notebook và end-to-end tracking fixture. Đã duyệt cài `mlflow>=2,<3`, Matplotlib và `numpy<2` trong `.venv`; không server, official training, download/đổi dataset, thay SPEC/Plan hoặc commit/push.

## 5. Shared Data Contract

**Contract được SPEC v4 yêu cầu; chưa có bằng chứng Data Contract v1 đã phát hành/được bốn TV xác nhận.** Thiết kế interface theo schema này, không coi là đã có official dataset.

| Thành phần | Contract v4 |
|---|---|
| Input tối thiểu | `id,text,label,split`; IDs ổn định, `label ∈ {0,1,2}`, split train/validation/test |
| Nhãn | `0:CLEAN`, `1:OFFENSIVE`, `2:HATE`; không tự đổi taxonomy/gộp ViCTSD |
| Nguồn | Ưu tiên UIT-ViHSD sau xác minh source/revision/license/schema/mapping; ViCTSD chỉ OOD/đối chiếu nếu phù hợp taxonomy |
| Split | Ưu tiên official split nếu phù hợp; nếu không mới stratified 70/15/15, seed và group/thread policy ghi manifest; không chia lại riêng từng EXP |
| Metadata G0 | Dataset version/hash, label mapping, normalization rules bất biến, manifest `id→split`, seed, exact/near-duplicate/conflicting-label/group leakage checks |
| Phát hành | Chỉ Train/Validation cho development; locked test do quy trình final quản lý |
| Fit | Vocabulary/tokenizer, TF-IDF, feature selection, class statistics và mọi transform có học chỉ fit train/train-fold; validation/test chỉ transform |
| Thay đổi | TV3 duyệt, version contract, thông báo nhóm và chạy lại kết quả bị ảnh hưởng trước final; không âm thầm đổi IDs/split |

Loader phải kiểm tra schema, IDs, text hợp lệ và label mapping; không tự tạo split khi thiếu manifest. `cleaned_text`/`label_name` nằm trong draft dataset requirements cũ, **không phải input tối thiểu bắt buộc của v4**. Định dạng file, tên giá trị `val` hay `validation`, kiểu ID và đường dẫn manifest còn cần chốt ở G0. Không hard-code `train.parquet` chỉ vì plan v3 từng ghi như vậy.

### G0 — Checklist bàn giao và phê duyệt (chưa mở gate)

| Owner | Bằng chứng cần bàn giao | Điều kiện đóng |
|---|---|---|
| TV1 cung cấp; TV3 review | Source/revision/license, schema thực và mapping nhãn; raw-data reference/hash | Truy vết được nguồn, taxonomy và quyền dùng; unknowns được xử lý trước official training |
| TV1 cung cấp dữ liệu; TV3 duyệt split | Official split hoặc lý do tự chia; seed, `id→split`, dataset/split hashes; exact/near-duplicate, conflicting-label/group leakage report | Cùng IDs cho bốn nhánh, không known overlap chưa xử lý; test tách khỏi development |
| TV1 đề xuất; TV3 duyệt normalization | Quy tắc bất biến và version preprocessing; train/val artifacts hoặc references; length/class statistics có provenance | Không fit validation/test, không thay split; đủ thông tin quyết định config official |
| TV3 điều phối; TV1–TV4 xác nhận | Data Contract v1, nơi giữ locked test và quyền chạy final; prediction schema; acknowledgment bốn thành viên | Có bằng chứng approval, không chỉ checklist; official training mới được mở |

Adapter kỹ thuật nhóm 1 dùng JSONL UTF-8, manifest truyền tường minh dưới dạng mapping ID→split, split development `train`/`val`, ID string hoặc integer (không bool). Đây **không phải** quyết định định dạng official dataset hay G0 approval. Tokenization fixture dùng `str.split` giữ case/diacritics, chưa có normalization theo EDA. Batch trả Python lists để không cần PyTorch ở lớp dữ liệu; tensorization thuộc pipeline huấn luyện sau.

Luồng gọi nhóm 1 (chỉ dữ liệu development đã có manifest):
```python
from src.data.loader import load_records
from src.data.tokenizer import Vocabulary, collate_records

records = load_records(path, manifest=manifest)
vocab = Vocabulary()
vocab.fit([row for row in records if row["split"] == "train"])
batch = collate_records(records[:2], vocab, max_length=32)
```
`batch` gồm `ids,input_ids,labels,lengths,attention_mask`; PAD=0, UNK=1, right-padding theo chuỗi dài nhất sau truncation. `max_length=32` chỉ minh họa gọi hàm, không phải config nghiên cứu được duyệt. Gọi `vocab.save(path)` / `Vocabulary.load(path)` để giữ mapping khi inference/resume. Development loader từ chối records `test`; kiểm tra này không thay cơ chế giữ kín locked test ở G0.

## 6. Model Evaluation Contract

- Chọn checkpoint/model bằng **validation Macro-F1**. Bổ sung per-class precision/recall/F1, weighted-F1, balanced accuracy, confusion matrix, training time, latency và model size; EXP-07 thêm VRAM. Ghi budget, hardware, batch/precision và cách đo để so công bằng.
- Prediction artifact dùng schema v4: `id,true_label,predicted_label,probabilities_if_available,run_id`. Join bằng ID trên cùng validation split/hash; không join theo thứ tự dòng. Nếu xuất xác suất: thứ tự `[CLEAN, OFFENSIVE, HATE]`, hữu hạn, không âm, tổng xấp xỉ 1; không có xác suất thì giữ rõ là không có.
- LR dùng `predict_proba`; `LinearSVC.decision_function` là margins, không phải probabilities. Calibrator fit train/CV train-only. Softmax DL không mặc nhiên đã calibration; lưu rõ phương pháp/diagnostics nếu dùng confidence hoặc voting. API phải theo `backend/schemas.py` hiện hành khi tích hợp, không tự đổi contract.
- Official MLflow run cần EXP/run IDs, hypothesis/variant, seed, config/architecture, dataset/split hash, environment/code revision, train duration, train/validation metrics, artifacts/checkpoint và prediction IDs.
- Freeze candidates/hyperparameters/thresholds/calibration/ensemble bằng validation trước EXP-08. Final là **một đợt** cho toàn bộ candidates đã khóa, không bắt buộc một model duy nhất; không xem test để tune hoặc viết lại giả thuyết.
- Fixture/synthetic smoke metrics chỉ chứng minh kỹ thuật, gắn nhãn riêng; **không là research results/official runs**, không đếm vào ≥3 biến thể.

## 7. Deliverables per Member

**Chung:** source tái lập, versioned config/data-split hash/seed, ≥3 official runs + MLflow IDs, validated metrics, best artifact/checkpoint, validation predictions có IDs, ablation/error commentary, notebook/report dẫn run thật.

| Thành viên | Bàn giao riêng |
|---|---|
| TV1 | EDA, preprocessing decisions/variants, dữ liệu chung theo G0, TF-IDF+LR pipeline; điều phối final benchmark |
| TV2 | LinearSVC/features/tuning artifacts, so sánh cost và limits; calibration metadata nếu dùng; RBF kết quả hoặc lý do bỏ qua |
| TV3 | G0 manifest/contract được xác nhận; recurrent architecture diagrams/ablation/curves/checkpoints; cross-model error report; freeze/review final; optional PhoBERT benchmark hoặc lý do bỏ qua |
| TV4 | TextCNN kernel/filter/pooling/regularization study; diversity và ensemble assessment nếu đủ điều kiện, so best single và overhead |

## 8. Dependencies & Parallel Work

- **Trước G0:** TV1 EDA/preprocessing; TV2–TV4 interface, pipelines và smoke tests trên fixture. TV3 chốt thiết kế contract/điều phối G0, không làm thay EDA TV1.
- **Sau G0 + dữ liệu/split được phê duyệt:** EXP-01…04 chạy độc lập cùng Train/Validation; không cần đợi EXP-01 train xong. Nếu pipeline EXP-03 cần preprocessing artifact của TV1 chưa bàn giao thì chỉ phần tích hợp đó chờ TV1.
- Predictions từ EXP-01…04 mở EXP-05; ≥2 models với diversity có cơ sở mới mở EXP-06. EXP-07 chờ EXP-03 đạt chuẩn. Optional runs không chặn freeze/final nếu bỏ qua có lý do.
- EXP-08 chờ official scratch deliverables, error analysis/selection và frozen candidate manifest; TV1 tổ chức, TV3 duyệt. Tiến độ GPU phối hợp TV4, không suy diễn có GPU rảnh.

## 9. Current Progress

**Quy ước:** DONE = đủ bằng chứng đạt DoD; IN PROGRESS = có bằng chứng đang làm; BLOCKED = có dependency chặn cụ thể; NOT STARTED = có xác nhận chưa bắt đầu (ghi thời điểm); UNKNOWN = không đủ thông tin hiện tại. Kế hoạch/checkbox/file scaffold không chứng minh hoàn thành.

**Bằng chứng 2026-11-14:** TV1 chưa xong EDA/preprocessing; TV2/TV4 chưa xác nhận đầy đủ. Nhóm 1–3 đã triển khai/kiểm thử kỹ thuật trên fixture, chưa có official runs. `.venv` Python 3.12.5: torch 2.15.0+cpu, pytest 9.2.0, MLflow 2.22.5, Matplotlib 3.11.2, NumPy 1.26.4; package cài theo quyền được duyệt. Có notebook phân tích không lưu outputs. Không thay dataset/split chính thức hoặc truy cập runs máy khác.

| Đối tượng | Trạng thái hiện tại / phạm vi | Bằng chứng hoặc điểm chặn |
|---|---|---|
| G0 / EXP-00 — TV3 lead | UNKNOWN về tiến độ thực hiện; chưa chứng minh gate mở | Chưa có contract/manifest/hash/approval bàn giao trong bằng chứng đã xem |
| TV1 / EXP-01 | EDA/preprocessing chưa hoàn thành; UNKNOWN đang làm tới đâu; official LR BLOCKED theo gate chưa được chứng minh | Xác nhận người dùng; không suy ra IN PROGRESS chỉ từ “chưa xong” |
| TV2 / EXP-02 | UNKNOWN cá nhân; official runs BLOCKED theo G0 chưa được chứng minh | Chưa có run/model/acceptance bàn giao |
| TV3 / EXP-03 | IN PROGRESS; nhóm 1–3 IMPLEMENTED + TESTED fixture CPU; official training BLOCKED theo G0/data approval | Model/trainer/checkpoint, MLflow local thật, report/notebook đã smoke; chưa có official runs/research metrics |
| TV4 / EXP-04 | UNKNOWN cá nhân; official runs BLOCKED theo G0 chưa được chứng minh | Chưa xác nhận tiến độ hay run ngoài repo |
| EXP-05 | BLOCKED phần phân tích thực nghiệm | Thiếu predictions/metrics EXP-01…04 |
| EXP-06 | BLOCKED, conditional | Thiếu ≥2 model predictions và diversity evidence |
| EXP-07 | BLOCKED, optional | Chưa chứng minh EXP-03 đạt nghiệm thu; không triển khai trước |
| EXP-08 | BLOCKED | Chưa có G0, official candidates và freeze được xác nhận |

Không EXP nào được ghi DONE. Trạng thái NOT STARTED ngày 2026-11-11 không tự động xác nhận trạng thái cá nhân ngày 2026-11-14. Bổ sung run IDs/PR/artifact + DoD trước khi đổi trạng thái.

## 10. TV3 Independent Work Plan

Nhóm 1–2 đã kiểm chứng. Nhóm 3 chia subagent tracker và report/notebook sở hữu file riêng; parent tích hợp CLI/logging và tests end-to-end. Mã `N/W/X/O` không thay EXP/task IDs chính thức; chỉ đổi trạng thái theo bằng chứng. GPU profiling vẫn chưa kiểm chứng.

### CAN START NOW

Nhóm 1–3 có data/model/training/checkpoint/CLI/config, tracking tại `src/tracking/`, report helper `src/evaluation/recurrent_report.py`, notebook `notebooks/EXP-03-recurrent-analysis.ipynb` và tests. GPU profiling chưa kiểm chứng. Chỉ mở module liên quan, không quét lại toàn repo.

| Task / mục tiêu | Dependencies | Deliverables / DoD | Trạng thái | File/module dự kiến |
|---|---|---|---|---|
| N1 — Thiết kế input/output adapter | Schema §5–6; chưa cần official data | JSONL records + explicit manifest, validation schema/IDs/labels/text/split; metadata version/hash official chờ G0, không auto-split | DONE adapter fixture; G0 chưa phê duyệt | `src/data/loader.py`; §5 |
| N2 — Dataset Loader | N1 | Đọc fixture train/val giữ ID; kiểm tra manifest, duplicate IDs/exact cross-split overlap; batches padding/lengths/masks, reject locked-test records | DONE kỹ thuật / TESTED | `src/data/loader.py`, `src/data/tokenizer.py`, `tests/test_data_loader.py`, `tests/test_data_pipeline.py` |
| N3 — Tokenizer/vocabulary | N1–N2 | Train-only fit; PAD/UNK, truncation/padding/mask, JSON save/load; validation-only tokens thành UNK, không refit hoặc mutate vocab khi encode | DONE kỹ thuật / TESTED | `src/data/tokenizer.py`, `tests/test_tokenizer.py`, `tests/test_data_pipeline.py` |
| N4 — Random embedding + LSTM baseline | N2–N3 | Random embedding, packed sequences, logits ba lớp; fixture forward/backward/update và padding invariance | DONE kỹ thuật / TESTED CPU | `src/models/recurrent.py`, `tests/test_recurrent.py` |
| N5 — BiLSTM | N4 | Bidirectional final states, cùng interface, lengths/padding đúng | DONE kỹ thuật / TESTED CPU | Cùng recurrent module/tests |
| N6 — BiLSTM+Attention | N5 | Mask padding trước softmax; backward/padding/batch-composition invariance | DONE kỹ thuật / TESTED CPU | Cùng recurrent module/tests |
| N7 — Train/validation loop | N4–N6 | Train gradients/clipping, eval no-grad, sample-weighted loss; Macro-F1 cả ba lớp, per-class/confusion, early stopping và callback | DONE kỹ thuật / TESTED CPU | `src/models/training.py`, `tests/test_training.py` |
| N8 — Checkpoint/resume/seed | N7 | Atomic save; model/Adam/RNG/config/data/vocab metadata; weights-only load; resume epoch/best/patience và CPU deterministic continuation | DONE kỹ thuật / TESTED CPU; CUDA chưa kiểm chứng | `src/models/checkpoint.py`, `tests/test_checkpoint.py` |
| N9 — MLflow integration | N7–N8 | Tracker local thật; epoch metrics/config/seed/hash/environment/time/checkpoints/predictions có run IDs; FAILED/FINISHED và store isolation đã test | DONE kỹ thuật / TESTED fixture; chưa official runs | `src/tracking/mlflow_tracker.py`, `recurrent_tracking.py`, `tests/test_mlflow_tracker.py`, `tests/test_tracking_pipeline.py` |
| N10 — Unit/smoke gate EXP-03 | N2–N9 | Data/model/train/eval/checkpoint/tracking/artifact readback trên fixture; không tính official acceptance | DONE kỹ thuật / TESTED CPU; nghiên cứu vẫn chờ G0 | Tests module và tracking/recurrent pipeline |
| N11 — Notebook phân tích | Output schema N9 | Notebook import helper; mặc định CHƯA THỰC HIỆN; chọn artifact/run ID tường minh, đọc metadata/curves/per-class/confusion/comparison từ artifacts | DONE cấu trúc / TESTED default + synthetic local/MLflow; official results chưa có | `notebooks/EXP-03-recurrent-analysis.ipynb`, `src/evaluation/recurrent_report.py`, `tests/test_recurrent_report.py` |
| N12 — Config/tuning + profile RTX 4060 8GB | N7–N10; GPU thực tế cho profile | JSON config có embedding/hidden/dropout/LR/length/batch/seed/device, ba architecture variants; chỉ smoke CPU, chưa HPO/VRAM/mixed precision | IN PROGRESS: config kỹ thuật xong; GPU profile BLOCKED bởi môi trường CPU | `configs/recurrent_config.json`, `src/models/smoke_recurrent.py` |
| N13 — G0 coordination/design | SPEC v4 T0.1–T0.3; phối hợp TV1 | Checklist provenance/schema/split/manifest/approval và yêu cầu bàn giao tại §5; G0 chỉ mở sau xác minh thực tế và cả bốn TV xác nhận | DONE phần checklist; G0 chưa phê duyệt | §5 context này; nguồn SPEC v4 §3 và task breakdown M0 |

**Thứ tự gọn:** N1–N4 + một smoke loop; N5–N8; N9–N10; N11–N12. N13 điều phối song song. Synthetic/mock data nhỏ chỉ làm fixture kỹ thuật tách biệt, không trộn dữ liệu chính thức. Profiling cần GPU khả dụng và quyền chạy smoke; không tự khởi chạy sweep/official train.

**Kiểm chứng nhóm 1 trước đó (2026-11-14):** `python -m pytest tests/test_data_loader.py tests/test_tokenizer.py tests/test_data_pipeline.py` — **121 passed in 0.52s**. Được chạy lại cùng nhóm 2 dưới `.venv` như dưới. Không có official data/results.

### Nhóm 2 — API và cách chạy smoke (IMPLEMENTED + TESTED CPU)

- `src/models/recurrent.py`: `RecurrentClassifier`, random embedding, packed sequences; kiến trúc `lstm`, `bilstm`, `bilstm_attention`, output logits ba lớp.
- `src/models/training.py`: `train_epoch`, `evaluate`, `fit`; tensor hóa batches nhóm 1, validation Macro-F1/early stopping, gradient clipping; callback cuối epoch phục vụ checkpoint.
- `src/models/checkpoint.py`: seed, save/load model+optimizer+RNG+config/data metadata/vocabulary; last checkpoint để resume, best checkpoint cho model selection.
- `configs/recurrent_config.json`: **provisional synthetic-only**, không phải thông số tối ưu từ EDA. Smoke chạy CPU; chưa profile RTX 4060 hoặc mixed precision.
- CLI chỉ chứa fixture nhỏ gắn nhãn `SYNTHETIC_NOT_RESEARCH`, không nhận official dataset: `.venv/Scripts/python.exe -m src.models.smoke_recurrent --architecture bilstm_attention --output-dir <thu-muc-tam>`; `--resume <last.pt>` tiếp tục tới tổng epochs trong cùng config, không thêm epochs ngầm.
- Artifacts smoke gồm vocabulary JSON, `last.pt`, `best.pt`, JSON history/predictions. Không ghi MLflow official run, không dùng metrics này làm kết quả nghiên cứu. Lưu ngoài Git; không ghi đè artifact có giá trị.

**Kiểm chứng nhóm 1–2:** `.venv/Scripts/python.exe -m pytest tests/test_recurrent.py tests/test_training.py tests/test_checkpoint.py tests/test_recurrent_pipeline.py tests/test_data_loader.py tests/test_tokenizer.py tests/test_data_pipeline.py` — **227 passed, 1 warning in 6.57s**. Warning PyTorch không khởi tạo NumPy vì `.venv` chưa có numpy; không dùng numpy bridge ở pipeline đã kiểm thử, không cài ngoài phạm vi được duyệt.

**CLI thực tế:** chạy `python -m src.models.smoke_recurrent` bằng Python `.venv` cho `bilstm_attention`, sau đó `--resume .../last.pt` sang output dir mới — cả hai exit 0, status `SYNTHETIC_NOT_RESEARCH`, device CPU. Artifacts ở thư mục tạm đã dọn. Tests kiểm chứng continuation model+optimizer+RNG khớp uninterrupted; CLI resume checkpoint đã hoàn tất thì không tự train thêm.

**Giới hạn nhóm 2:** resume ở epoch boundary, thứ tự batches caller quản lý; trainer giữ last state, callback lưu best theo validation, không ngầm restore best. Softmax chưa calibrated. Không có scheduler/scaler; chưa official training, GPU/AMP hoặc VRAM/latency benchmark. MLflow đã kiểm chứng ở nhóm 3 dưới đây.

### Nhóm 3 — Tracking và phân tích (IMPLEMENTED + TESTED fixture)

- CLI thêm `--tracking-uri <local-path-hoac-file-URI>` để log vào experiment **EXP-03-synthetic-smoke**. Tags: `result_type=synthetic`, `research_result=false`, `pretrained=false`; remote/network URI bị từ chối. Không khởi chạy server hoặc official training.
- Log config/model/seed, dataset-split hash, environment/code revision, duration, train/val loss/Macro-F1/per-class từng epoch. Artifacts: `config.json`, `data_metadata.json`, `vocabulary.json`, checkpoints có sẵn, `smoke_results.json`, `validation_predictions.json`; predictions có IDs/run ID, **last checkpoint**, không nhầm với best.
- Tracker giữ wrapper cũ; sửa lifecycle FAILED/FINISHED, local URI Windows, experiment/store isolation và directory artifacts. Fluent active runs chỉ dùng tuần tự, không tuyên bố concurrent/thread-safe.
- Notebook chỉ phân tích, mặc định `artifact_dirs=[]`, `tracking_uri=None`, `run_ids=[]` và in **CHƯA THỰC HIỆN**. Chọn local artifact hoặc explicit MLflow run IDs để download/read. Helper `load_report`, `render_report`, `compare_reports` kiểm tra evidence, xuất loss curves/Macro-F1/confusion và phân biệt synthetic/official, last/best. `OFFICIAL` trong artifact chỉ là khai báo, không tự chứng minh G0 approval.
- Chạy `.venv/Scripts/python.exe -m pytest tests/test_mlflow_tracker.py tests/test_recurrent_report.py tests/test_tracking_pipeline.py tests/test_recurrent_pipeline.py`: **30 passed in 13.45s**. Sau sửa layout tiêu đề figure bị cắt: report suite **15 passed in 0.85s**.
- CLI tracked `bilstm_attention` exit 0; readback MLflow FINISHED, metrics/predictions/artifacts đúng. Smoke run kỹ thuật `cd4ef842204549119e6254b48262336c` ở temporary store (đã dọn), không phải research result. Notebook code cells đã thực thi mặc định, local artifacts và MLflow download bằng Python, chưa kiểm UI Jupyter; outputs notebook commit vẫn rỗng. Đã mở PNG confusion matrix và sửa/xác minh tiêu đề không bị cắt.
- Chưa chạy full suite, official train, GPU, hoặc EXP-05 cross-model errors. NumPy đã cài nên warning thiếu NumPy của phiên nhóm 2 không còn trong checks nhóm 3. Không commit/push.

### WAITING FOR TV1

| Task / mục tiêu | Dependencies | Deliverables / DoD | Trạng thái | File/module dự kiến |
|---|---|---|---|---|
| W1 — Tích hợp dữ liệu/preprocessing bàn giao | TV1 cung cấp dữ liệu/EDA/preprocessing liên quan; TV3 chốt G0 và data/split approval | Loader đọc đúng train/val/manifest/hash, mapping và normalization đã duyệt; kiểm tra không tự split, không fit val/test | BLOCKED | Adapter `src/data/` đề xuất; đường dẫn dữ liệu thực tế chờ G0 |
| W2 — Chốt config official và chạy EXP-03 | W1 + N10; length/class statistics được duyệt hoặc số liệu tối thiểu G0 đủ quyết định | ≥3 official LSTM/BiLSTM/Attention runs; MLflow/checkpoint/predictions/curves/ablation; chọn theo val Macro-F1, không đòi mô hình sau thắng | BLOCKED | Trainer/config/notebook đề xuất ở trên |

**Không chờ toàn bộ EDA chuyên sâu một cách máy móc:** nếu G0 đã cung cấp dữ liệu/split và preprocessing đủ được phê duyệt, W1–W2 được mở dù báo cáo EDA của TV1 còn tiếp tục. Chưa có approval thì chưa official training.

### WAITING FOR OTHER EXPERIMENTS

| Task / mục tiêu | Dependencies | Deliverables / DoD | Trạng thái | File/module dự kiến |
|---|---|---|---|---|
| X1 — EXP-05 thực nghiệm | EXP-01…04 validation predictions đúng §6 | Join cùng IDs/hash, review ≥30 errors, confusion theo lớp, teencode/slang/emoji/phủ định/ngữ cảnh; hypotheses kiểm chứng được | BLOCKED | `src/evaluation/` hiện có; report/notebook vị trí cần chốt |
| X2 — Freeze/review EXP-08 | Scratch runs nghiệm thu; error analysis/ablation; optional hoàn tất hoặc skipped | Frozen candidate/config manifest, TV3 duyệt; TV1 chạy một final batch; không tune theo test | BLOCKED | Final evaluator đề xuất trong `src/evaluation/`; vị trí manifest/report chưa chốt |

### OPTIONAL / LATER

| Task / mục tiêu | Dependencies | Deliverables / DoD | Trạng thái | File/module dự kiến |
|---|---|---|---|---|
| O1 — EXP-07 PhoBERT | EXP-03 đạt toàn bộ DoD, G0, GPU/budget được cho phép | `vinai/phobert-base-v2` pretrained base + head fine-tune; cùng split/metrics, Macro-F1/time/VRAM/latency/size; ghi pretrained vs scratch, không toxic-finetuned checkpoint | BLOCKED, optional | `src/models/`, `configs/` có thể mở rộng sau; không triển khai trước gate |
| O2 — Nhiều seeds / HPO mở rộng | Official baseline EXP-03 + approved compute budget | Độ biến thiên/so sánh có kiểm soát; dùng config N12; Optuna chỉ nếu cần | BLOCKED, optional | Config/trainer đề xuất; không thêm dependency lúc này |
| O3 — Góp predictions/review EXP-06 | TV4 chủ trì, ≥2 model outputs + diversity evidence | Predictions đúng contract, OOF train-only nếu stacking, báo gain/loss vs best single; không nhận ownership TV4 | BLOCKED, conditional | Validation artifacts, `src/evaluation/` nếu cần |

## 11. Blockers & Open Questions

1. **Local v3 khác GitHub v4:** nguồn local còn EXP-001…009 và Transformer là bước chính; không dùng các ID/đường găng đó. Phiên này không sync/checkout/viết lại SPEC; Agent tiếp theo dùng bản v4 ghim bên dưới khi cần chi tiết.
2. **ADR/dataset docs chưa đồng bộ ngay cả trên GitHub:** ADR-02 còn final EXP-009 và split cứng 70/15/15; ADR-09 gọi ensemble EXP-007, ngưỡng ≥1%; dataset draft có extra columns và chốt sau EDA. SPEC/Plan v4 mới hơn quy định final EXP-08, ensemble EXP-06, ưu tiên official split và G0 không chờ EDA sâu. **Ngưỡng ≥1% không được v4 xác nhận lại; cần làm rõ trước nghiệm thu ensemble, không tự coi đã bị bỏ hay vẫn là gate.** Không đổi SPEC trong task này.
3. Chưa xác minh source/license/revision, schema thực, official split, normalization, manifest/data hash, nơi giữ test, G0 approval. Không suy ra dataset hợp lệ chỉ từ link.
4. Chưa có bằng chứng tiến độ/run/artifact bên ngoài repo của TV1–TV4; cần run IDs/commit/PR/artifacts và đối chiếu DoD, không chỉ checkbox.
5. Có tracker file nhưng chưa kiểm tra API, package MLflow cài sẵn hay chưa, resume/reproducibility, GPU/VRAM, notebook hoặc training runtime. Chưa có số đo thực nghiệm.
6. Chưa chốt định dạng file/manifest, `val` naming, chỗ lưu artifacts/chia sẻ runs nhiều máy và lịch GPU. Đây là chi tiết bàn giao cần giải quyết, không đổi ownership hoặc chặn việc fixture.

## 12. Update Rules & Source References

- Đọc file này trước cho ownership/dependencies/progress. Chỉ mở source/module đúng phần cần triển khai hoặc kiểm chứng; không đọc lại toàn repository.
- Đây là cache bối cảnh. Khi quyết định/phân công/contract thay đổi hoặc nghi sai lệch, đối chiếu yêu cầu mới nhất và SPEC/Plan đúng revision; không tự sửa source of truth theo bản tóm tắt.
- Cập nhật đúng hàng/section thay đổi, kèm ngày, nguồn, commit/run/artifact và DoD; giữ khác biệt giữa plan, implemented, tested, experimentally validated và deployed. Không đổi UNKNOWN thành DONE vì có code hay checkbox.
- Khi thay đổi contract: version + approval TV3 + thông báo thành viên, ghi ảnh hưởng/rerun. Task scratch triển khai tiếp theo bắt đầu N1–N4, không nhảy EXP-07 hoặc official training trước gate.

**Nguồn chính thức đã kiểm tra — GitHub revision `bd9602300616cf0f01d5b4f04a463cb0148181be`:**

| Đường dẫn | Mở khi cần |
|---|---|
| [`SPEC.md`](https://github.com/Huyyy23/toxic-comment-detection/blob/bd9602300616cf0f01d5b4f04a463cb0148181be/SPEC.md) | Hướng scratch-first, phân công, giới hạn |
| [`docs/spec.md`](https://github.com/Huyyy23/toxic-comment-detection/blob/bd9602300616cf0f01d5b4f04a463cb0148181be/docs/spec.md) | §3 G0/schema; §5 predictions/logging; §6 scratch/optional; §7 final; §9 acceptance |
| [`docs/plans/ml-experiments.md`](https://github.com/Huyyy23/toxic-comment-detection/blob/bd9602300616cf0f01d5b4f04a463cb0148181be/docs/plans/ml-experiments.md) | EXP-00…09, owners, dependencies, hypotheses/deliverables |
| [`docs/plans/task-breakdown.md`](https://github.com/Huyyy23/toxic-comment-detection/blob/bd9602300616cf0f01d5b4f04a463cb0148181be/docs/plans/task-breakdown.md) | T0.1–T0.3 G0; T1.3 EXP-03; T2.3 errors; T3.2 PhoBERT; M4 final DoD |
| [`docs/plans/implementation-plan.md`](https://github.com/Huyyy23/toxic-comment-detection/blob/bd9602300616cf0f01d5b4f04a463cb0148181be/docs/plans/implementation-plan.md) | G0–G3, branch conventions và lịch GPU |
| [`tasks/plan.md`](https://github.com/Huyyy23/toxic-comment-detection/blob/bd9602300616cf0f01d5b4f04a463cb0148181be/tasks/plan.md), [`tasks/todo.md`](https://github.com/Huyyy23/toxic-comment-detection/blob/bd9602300616cf0f01d5b4f04a463cb0148181be/tasks/todo.md) | Workflow/checklist v4; PLAN ONLY không phải bằng chứng nghiệm thu |

**Nguồn phụ cần cảnh giác lỗi thời:** `docs/decisions.md` ADR-01…09, `docs/dataset-requirements.md` §4, `docs/open-questions.md` DD-01…10 local; xem mâu thuẫn §11 trước khi áp dụng. **Bằng chứng local có giới hạn:** listings `src/`, `configs/`, `data/`, `models/`; graphify đã query nhưng graph vẫn có frontend/ADR cũ, không dùng làm bằng chứng tiến độ. Yêu cầu nhóm trưởng ngày 2026-11-14 xác nhận TV1 chưa hoàn thành EDA/preprocessing và chưa rõ tiến độ TV2/TV4.
