# ML experiments plan

**Status:** N1–N15 remains the plan. N12 calibration diagnostics were fitted on existing validation predictions; results and exposure limits are recorded below. See [the team execution plan](../../tasks/plan.md).

## Data and evaluation protocol
Use the binary dataset at `data/processed/binary_n2_v1/`. Source label meanings differ. The user confirms the team sought UIT permission, but independent license evidence is not in the repo. See `docs/dataset-requirements.md` for the mapping, provenance and test exposure.

Use validation Macro-F1 as the primary metric and checkpoint criterion. Report TOXIC F1, PR-AUC, per-class precision/recall/F1, confusion matrix, metrics by source, runtime, model size, and ECE. Use three seeds and retain configs, dataset hashes, run IDs, checkpoints, and validation predictions.

Fit vocabulary, embeddings, and learned preprocessing on training data only. Fit calibration and choose thresholds on validation, alongside model selection, imbalance method, and optional extras. Do not use locked test for tuning, selection, calibration, thresholds, or debugging. Freeze all candidates and analysis settings in a versioned lock file before the sole final test batch. Disclose historical test exposure. Report bootstrap confidence intervals or McNemar comparisons in the final batch.

## N12 post-hoc calibration diagnostics

The selected N12 checkpoints were calibrated without changing model weights. The fitter used 4,838 rows for fitting and one reserved 1,210-row validation holdout for evaluation. It split the sorted common IDs with seed 42, stratified by source and label. The metrics use 10-bin ECE, binary Brier Score, and Macro-F1 at the unchanged 0.5 threshold.

| Model | Method and fitted parameter | ECE before → after | Brier before → after | Macro-F1 before → after | Applied |
|---|---|---:|---:|---:|---|
| N6-LR-D | Platt, coef `0.805251`, intercept `-0.510152` | 0.029170 → 0.025675 | 0.071368 → 0.069688 | 0.805684 → 0.792381 | No. Macro-F1 decreased. |
| V3-TRI-GATED-FT | Temperature, `T=0.871645` | 0.035690 → 0.020551 | 0.071554 → 0.070881 | 0.811471 → 0.811471 | Yes. |
| V3-PHOBERT-V2 | Temperature, `T=1.103778` | 0.016673 → 0.023613 | 0.063272 → 0.063148 | 0.841012 → 0.841012 | No. ECE increased. |

The activation gate requires ECE and Brier to decrease strictly and Macro-F1 not to decrease. Runtime therefore applies Temperature Scaling only to TRI; N6 and PhoBERT return raw probabilities. The reported “after” metrics for disabled calibrators are diagnostics, not deployed outputs. Calibration did not change candidate selection, which remains based on the original validation Macro-F1.

No independent evaluation set exists for these exact artifacts. The reserved holdout is disjoint from calibration fitting but was part of the full validation set used to select checkpoints. Treat these metrics as exposed validation diagnostics, not independent evaluation or generalization evidence. The fitting tool reads only the three saved N2 validation prediction files; it does not access locked test data. Parameters, source hashes, split hashes, and aggregate metrics are in `deployment/hf-gradio/models/calibration.json`.


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
