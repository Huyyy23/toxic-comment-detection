# Task Breakdown & Acceptance Criteria — Scratch-first v4.0

**Ngày:** 2026-11-11 · **Trạng thái:** Chỉ lập kế hoạch, chưa triển khai.
**Quy tắc:** mọi task dưới đây là TODO, không đánh dấu hoàn tất nếu chưa có bằng chứng. Mỗi task có owner, output và acceptance.

## M0 — Governance / G0 (TV3 lead)
- **T0.1** Inspect UIT-ViHSD source, license, schema, class mapping, official splits; phối hợp TV1 EDA sơ bộ. **DoD:** tài liệu xác minh provenance và label taxonomy; unknowns ghi rõ.
- **T0.2** Dedup/check label conflicts, leakage across texts/groups; chốt official split phù hợp hoặc stratified 70/15/15; tạo dataset hash/ID manifest, seed. **DoD:** reproducible `id→split`, no known overlap; test locked.
- **T0.3** Chuẩn hóa input/output contract, val Macro-F1, MLflow run naming; cấp train/val. **DoD:** TV1–TV4 xác nhận dùng chung v1.

## M1 — Parallel pipelines (TV1–TV4)
- **T1.1 TV1:** EDA + preprocessing (raw/basic/teencode variants); LR + word/char TF-IDF. **DoD:** ≥3 official variants, fit train-only, report validation + artifacts.
- **T1.2 TV2:** LinearSVC + word/char TF-IDF, tuning C, optional feature selection and RBF SVC. **DoD:** ≥3 official variants, val comparison, best SVM checkpoint; no invented probabilities.
- **T1.3 TV3:** Random-initialized embeddings, LSTM → BiLSTM → Attention. **DoD:** ≥3 official architecture runs, val Macro-F1, curves, checkpoints, ablation; no pretrained weights.
- **T1.4 TV4:** Random-initialized embeddings, single/multi-kernel TextCNN and regularization variants. **DoD:** ≥3 official variants, val metrics, plots, checkpoints.

## M2 — Error Analysis & Controlled Improvements
- **T2.1 TV1:** quantify LR preprocessing/teencode/class weighting effects; diagnose representative errors. **DoD:** ablation and limitations.
- **T2.2 TV2:** analyze feature/C/class weighting trade-off; compare compute cost. **DoD:** study and limitations.
- **T2.3 TV3:** review cross-model validation error samples, identify error categories and specific next hypotheses. **DoD:** ≥30 reviewed wrong predictions with taxonomy.
- **T2.4 TV4:** TextCNN kernel/dropout ablation, compare errors with other branches. **DoD:** full val prediction artifact.

## M3 — Optional research, never gate 4 scratch models
- **T3.1 TV4:** test error diversity; if justified, calibrated soft/weighted voting; OOF stacking only if train folds valid. **DoD:** report gain/loss vs best single, latency/cost or explicit skipped rationale.
- **T3.2 TV3:** optional PhoBERT pretrained-base fine-tune benchmark **after LSTM scratch runs**. **DoD:** label as pretrained; record base checkpoint/version, training budget, Macro-F1 and compute cost or explicit skipped rationale.

## M4 — Final science & demo
- **T4.1 TV3:** freeze final candidate list, configs and selection based on validation only. **DoD:** signed-off frozen manifest.
- **T4.2 TV1 lead, all contributors:** one-shot locked test evaluation for all fixed candidates. **DoD:** unbiased comparison Macro-F1/per-class, confusion, latency, size; no retuning from test.
- **T4.3 All:** reproducibility notebooks/report + slides, individual experiment sections. **DoD:** results sourced from actual MLflow runs.
- **T4.4 Shared:** minimal FastAPI `POST /predict`, `GET /health`, `GET /model-info`; Swagger demo, hosting only after eligibility checked. **DoD:** endpoint smoke tests and best model loads.

## Shared submission contract
Every TV hands over: experiment source, versioned config and data hash, ≥3 distinct runs, MLflow ID, validated metrics, model checkpoint/artifact, validation predictions with IDs, ablation/error commentary, notebook or report. All four use shared split G0 and fixed Macro-F1 metric; scratch means no pretrained embeddings/weights for four core models. No code changes in this docs-only revision.

## Dependencies at a glance
`M0/G0 → {T1.1 || T1.2 || T1.3 || T1.4} → {M2 parallel} → [optional M3] → M4`.
Pre-G0, pipelines may be authored and smoke-tested independently against fixtures. Optional tasks do **not** block final scratch evaluation.
