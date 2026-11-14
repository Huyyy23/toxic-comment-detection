# Báo cáo bàn giao TV3 — 2026-11-14

## 1. Đọc gì ở phiên sau

Đọc `docs/TEAM_WORK_CONTEXT.md` trước để biết ownership/contracts/dependencies; dùng báo cáo này để biết thay đổi và bằng chứng của phiên. Không cần quét lại repository. SPEC/Plan chính thức v4 được ghim tại GitHub revision `bd9602300616cf0f01d5b4f04a463cb0148181be` trong context; SPEC/Plan local xuất phát từ `c2270f4344112fa6f68256b251e96ae6b1a7d853` còn v3. Không dùng ID EXP-001…009 cũ thay EXP-01…08 hiện hành.

## 2. Phạm vi và trạng thái

TV3 phụ trách EXP-03 scratch recurrent DL, EXP-05 error analysis, optional EXP-07 PhoBERT; quản lý G0/technical consistency và duyệt final EXP-08 do TV1 điều phối. Phiên này hoàn thành **hạ tầng kỹ thuật nhóm 1–3, IMPLEMENTED + TESTED trên fixture CPU**. **EXP-03 vẫn IN PROGRESS, chưa nghiệm thu nghiên cứu; G0 chưa được chứng minh đã phê duyệt.**

TV1 chưa hoàn thành EDA/preprocessing theo xác nhận người dùng; chưa đủ bằng chứng tiến độ cá nhân TV2/TV4. Không có official training, research metrics, pretrained weights, dataset download/split changes hoặc deployment trong phiên. Con số 30–40% đã trao đổi chỉ là ước lượng khối lượng, không phải metric nghiệm thu.

## 3. Những gì đã làm

| Nhóm | Files | Chức năng |
|---|---|---|
| Context | `docs/TEAM_WORK_CONTEXT.md`, `AGENTS.md` | Ownership EXP-01…08, G0/EXP-00, dependencies, DoD, tiến độ và kế hoạch TV3; Agent đọc context trước, cập nhật theo bằng chứng |
| 1 — Data | `src/data/loader.py`, `src/data/tokenizer.py` | JSONL UTF-8, explicit ID→split manifest; schema/labels/IDs/duplicates/exact cross-split overlap; reject test records; train-only vocabulary, deterministic lexical IDs, PAD/UNK, JSON save/load, batches Python lists |
| 2 — Models | `src/models/recurrent.py` | Random embedding; LSTM/BiLSTM/BiLSTM+Attention; packed sequences, unsorted lengths, masked attention, logits ba lớp |
| 2 — Training | `src/models/training.py`, `src/models/checkpoint.py` | Sample-weighted loss, Macro-F1/per-class/confusion, early stopping, gradient clipping; checkpoint atomic, model/optimizer/RNG/config/data/vocab, safe weights-only load và epoch-boundary resume |
| 2 — Integration | `src/models/smoke_recurrent.py`, `configs/recurrent_config.json` | Synthetic-only CPU CLI, ba architectures; provisional config, last/best checkpoint, history/predictions; output dir mới/rỗng để tránh ghi đè |
| 3 — Tracking | `src/tracking/mlflow_tracker.py`, `src/tracking/recurrent_tracking.py` | Tái dùng tracker local; sửa FAILED/FINISHED, store/experiment isolation, directory artifacts; log epoch metrics/config/seed/hash/environment/time/checkpoints/predictions run IDs |
| 3 — Report | `src/evaluation/recurrent_report.py`, `notebooks/EXP-03-recurrent-analysis.ipynb` | Load/validate artifacts, loss/Macro-F1/confusion PNGs, comparison last/best; notebook chọn artifacts/run IDs tường minh, mặc định CHƯA THỰC HIỆN, outputs rỗng, không train |

Subagents được giao files riêng: data loader và tokenizer; recurrent model, trainer và checkpoint; tracker và report/notebook. Parent giữ integration/tests/context, không chồng chéo sở hữu files.

**Tests:** `tests/test_data_loader.py`, `test_tokenizer.py`, `test_data_pipeline.py`, `test_recurrent.py`, `test_training.py`, `test_checkpoint.py`, `test_recurrent_pipeline.py`, `test_mlflow_tracker.py`, `test_recurrent_report.py`, `test_tracking_pipeline.py`.

## 4. Bằng chứng kiểm chứng

Lệnh kiểm tra tổng hợp cuối trước commit:

```bash
.venv/Scripts/python.exe -m pytest tests/test_data_loader.py tests/test_tokenizer.py tests/test_data_pipeline.py tests/test_recurrent.py tests/test_training.py tests/test_checkpoint.py tests/test_recurrent_pipeline.py tests/test_mlflow_tracker.py tests/test_recurrent_report.py tests/test_tracking_pipeline.py -q
```

**Kết quả: 252 passed in 10.06s.** Đây là mười files tests của phạm vi triển khai, không tuyên bố full repository suite.

Kiểm chứng thực tế khác đã chạy:
- CLI `bilstm_attention` CPU và resume exit 0; tests xác nhận resumed model/optimizer/RNG khớp uninterrupted CPU.
- MLflow file store thật: FINISHED/FAILED, epoch steps, params/hash, artifacts/predictions đọc lại thành công; synthetic experiment riêng.
- Notebook code cells thực thi ba đường: default, local artifacts, MLflow download. Không chạy Jupyter UI/kernel; không cài IPython, notebook vẫn xuất PNG paths.
- Đã mở confusion matrix PNG, sửa tiêu đề provenance bị cắt và xác minh lại. Smoke metrics chỉ fixture; không lưu thành kết quả nghiên cứu.
- Synthetic run/store/downloads/figures tạm đã dọn; notebook không lưu outputs/checkpoint/dataset.

## 5. Môi trường và cách chạy

Người dùng đã duyệt tạo `.venv` Python **3.12.5**, cài torch **2.15.0+cpu**, pytest **9.2.0**, MLflow **2.22.5**, Matplotlib **3.11.2**, NumPy **1.26.4** cùng dependencies. Python 3.15 hệ thống không có torch; dùng đúng executable `.venv/Scripts/python.exe`.

`.venv` không được commit. `requirements.txt` đã có MLflow/numpy và torch bị loại trên Windows theo marker cũ; `requirements-dev.txt` đã có Matplotlib/pytest. Không sửa manifests trong phiên. **Máy Windows mới cần cài torch CPU/CUDA phù hợp một cách tường minh; không coi requirements hiện tại đã tái tạo đủ môi trường này.** Không tự cài CUDA hoặc package hệ thống.

```bash
.venv/Scripts/python.exe -m src.models.smoke_recurrent --architecture bilstm_attention --output-dir <thu-muc-moi> --tracking-uri <thu-muc-MLflow-local>
```

- Chỉ fixture nhúng trong CLI, không nhận official dataset. Tags `result_type=synthetic`, `research_result=false`, `pretrained=false`; experiment `EXP-03-synthetic-smoke`.
- Notebook: đặt `artifact_dirs=[...]` hoặc `tracking_uri` + `run_ids=[...]`, không tự chọn tất cả runs.
- `--resume <last.pt>` tiếp tục đến tổng epochs trong cùng config, không tự tăng epochs. Trainer giữ last weights, callback lưu best. Predictions/report đang là **last validation**, không được trình bày như best-checkpoint predictions.

## 6. Việc còn lại / điểm chặn

1. **G0:** TV1 bàn giao source/revision/license/schema, normalization/data references; TV3 review manifest/hash/split/leakage và cả bốn thành viên xác nhận. Checklist tại context §5. Không tự chia lại split hoặc làm thay EDA TV1.
2. Sau G0: nối official data adapter/config được duyệt, official train ≥3 meaningful LSTM/BiLSTM/Attention variants; tuning/architecture ablation, MLflow artifacts và nhận xét trung thực.
3. Profile RTX 4060 8GB, sequence length/batch/VRAM, AMP nếu thực sự hỗ trợ; CPU smoke không chứng minh GPU throughput hoặc reproducibility CUDA.
4. EXP-05 cần predictions EXP-01…04, ≥30 validation errors có taxonomy/hypotheses; công cụ confusion không thay phân tích lỗi thực nghiệm.
5. EXP-07 optional chỉ sau EXP-03 nghiệm thu; pretrained base `vinai/phobert-base-v2`, không toxic-finetuned checkpoint. Chưa triển khai.
6. Freeze candidates bằng validation và duyệt EXP-08; TV1 điều phối locked test một đợt. Không dùng test tuning.

**Giới hạn kỹ thuật:** whitespace tokenizer/JSONL/train-val naming là adapter convention, chưa là Data Contract v1 được duyệt; chưa near-duplicate/group audit. Batch ordering caller quản lý, resume epoch boundary; chưa scheduler/AMP scaler. Softmax chưa calibrated. MLflow fluent runs tuần tự, không thread-safe. `OFFICIAL` trong report là khai báo, không tự chứng minh G0. Không có CLI official training trong phiên này.

## 7. Git/GitHub bàn giao

Người dùng yêu cầu commit và đưa công việc lên GitHub. **Nhánh giao: `feature/tv3-scratch-pipeline`**, xuất phát local HEAD `c2270f4`; không push thẳng/force-push/merge vào `master`. GitHub `master` tại kiểm tra đầu task là `bd96023`, có SPEC/Plan v4; cần review divergence trước khi tích hợp nhánh này vào master.

Commit bao gồm source/config/tests/context/notebook và báo cáo này. `AGENTS.md` đang bị `.gitignore` loại trừ; file đã sửa được đưa riêng bằng explicit force-add theo yêu cầu lưu công việc, không đổi `.gitignore`. Không đưa `.venv`, caches, graphify outputs, raw data, checkpoints, MLflow stores hoặc secrets vào commit. Commit hash và kết quả push được báo sau khi lệnh thực tế hoàn tất, không đoán hash trong file.
