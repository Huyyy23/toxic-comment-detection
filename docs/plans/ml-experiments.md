# ML experiments plan

**Status:** Plan only. No new experiments or results are asserted. The N1–N15 task sequence and D1–D4 decisions are in [the team execution plan](../../tasks/plan.md).

## Data and evaluation protocol
Combine ViHSD and ViCTSD as a binary task: `CLEAN=0`, `TOXIC=1`. Map ViHSD labels greater than zero and ViCTSD Toxicity to TOXIC. Record `source` as `vihsd` or `victsd`. The source label meanings differ, and UIT permission remains pending. Deduplicate before stratified 70/15/15 split by `(source,label)` and lock test. Verify data availability, source revisions, schema, ViCTSD encoding, mapping, and G0 artifact compatibility before use.

Use validation Macro-F1 as the primary metric and checkpoint criterion. Report TOXIC F1, PR-AUC, per-class precision/recall/F1, confusion matrix, metrics by source, runtime, model size, and ECE. Use three seeds and retain configs, dataset hashes, run IDs, checkpoints, and validation predictions.

Fit vocabulary, embeddings, and learned preprocessing on training data only. Fit calibration and choose thresholds on validation, alongside model selection, imbalance method, and optional extras. Do not use locked test for tuning, selection, calibration, thresholds, or debugging. Freeze all candidates and analysis settings in a versioned lock file before the sole final test batch. Disclose historical test exposure. Report bootstrap confidence intervals or McNemar comparisons in the final batch.

## Core models
Train LR and SVM, LSTM, BiLSTM with attention, character CNN, and FastText without pretrained embeddings. Do not use LLMs or back-translation in the core comparison.

| Bước | Đối chứng trên validation |
|---|---|
| 1 | TF-IDF + LR với balanced class weights, baseline riêng. |
| 2 | BiLSTM, embedding ngẫu nhiên, cross-entropy. |
| 3 | Bước 2 + FastText tự train trên train. |
| 4 | Bước 3 + Char-CNN; chốt kiến trúc trước bước 5–7. |
| 5 | Bước 4 + một phương án imbalance chọn từ class weights, focal loss hoặc sampler. |
| 6 | Bước 5 + augmentation TOXIC train-only, so với bước 5. |
| 7 | Bước 6 + calibration và threshold trên validation. |

Augmentation mặc định thêm một bản mỗi mẫu TOXIC, tối đa hai bản; bảo vệ token độc hại bằng lexicon train và loại bản trùng validation/test. Mục tiêu 30–35% không được bảo đảm.

Change one factor at a time. Fix the architecture before ablation steps 5–7. Keep the same data split and report three seeds for each configured candidate.

## PhoBERT benchmark
PhoBERT is an accepted pretrained fine-tuning benchmark, not a core scratch model. P1 uses the default `phobert-base` configuration. P2 adds the imbalance treatment selected under D3. P3 optionally changes learning rate, epochs, or fine-tuning strategy one factor at a time on validation.

Use word segmentation and record its tool and version, 128–160 tokens with a hard limit of about 256, learning rate `2e-5`, batch size 16, 10% warmup, weight decay `0.01`, 3–4 epochs, and early stopping by validation Macro-F1. Use fp16 and gradient accumulation when supported.

## Final test and error analysis
Run one test batch after the candidate list, configurations, thresholds, and analysis methods are frozen. Test results and errors must not feed model selection or tuning. Perform error analysis after the test only as reporting; keep any follow-up hypotheses and experiments on validation. Optional extras run on validation and enter the test batch only if frozen beforehand.

The previous CLEAN/OFFENSIVE/HATE three-class protocol is superseded. Preserve it as history, not as the current experiment contract.
