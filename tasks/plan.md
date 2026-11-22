# Team Execution Plan — Vietnamese Toxic Comment Detection

**Version 4.0.0 · 2026-11-11 · PLAN ONLY**

## Quyết định hiện hành
- **D1 — dataset and split:** Combine UIT-ViHSD and ViCTSD as a binary task with `CLEAN=0` and `TOXIC=1`. Map ViHSD labels greater than zero to TOXIC and map ViCTSD's Toxicity label to TOXIC. Record `vihsd` or `victsd` as `source`. Dataset meanings differ, and UIT permission remains pending. Deduplicate before a stratified 70/15/15 split by `(source, label)`. Lock the test split. Record source versions, mapping, dataset hashes, IDs, seed, and manifest. Do not treat unavailable data or results as verified.
- **D2 — model scope:** Train LR and SVM from scratch, plus LSTM, BiLSTM, attention, character CNN, and FastText using train-only vocabulary or embeddings. PhoBERT fine-tuning is an accepted pretrained benchmark, separate from the core comparison. The core comparison uses no pretrained embeddings, LLMs, or back-translation.
- **D3 — imbalance and augmentation:** Augment only TOXIC training samples, at most twice per original, targeting 30–35% of training data without promising that the target is reachable. Preserve toxic tokens and exclude generated text that duplicates validation or test text. Select exactly one of class weights, focal loss, or a sampler using validation. Fit calibration and choose any decision threshold on validation; report ECE.
- **D4 — comparison and final test:** Compare models on the same split and binary labels. Select checkpoints by validation Macro-F1 for every model; report TOXIC F1 as a secondary metric. Run three seeds. Freeze all candidates and settings before one final test batch. Report bootstrap confidence intervals or McNemar comparisons. Perform final error analysis after test without tuning; run optional extras on validation only.

## N1–N15 task sequence
| ID | Work | Depends on | Owner |
|---|---|---|---|
| N1 | Publish and obtain team approval of D1–D4 documentation. | — | Cả nhóm |
| N2 | Build the combined, deduplicated binary dataset and locked 70/15/15 manifest. | N1 | TV3 lead; TV1 hỗ trợ |
| N3 | Audit duplicate and split leakage after N2; document remaining limits. | N2 | Huy (TV3) |
| N4 | Create and validate train-only TOXIC augmentation under D3. | N3 | Huy (TV3) |
| N5 | Train and evaluate train-only FastText embeddings. | N2 | Huy (TV3) |
| N6 | Train LR and SVM baselines on the shared split. | N2 | TV1 LR; TV2 SVM |
| N7 | Train scratch LSTM, BiLSTM/attention, and character CNN models. | N4, N5 | TV3 và TV4 |
| N8 | Calibrate and choose thresholds on validation for all eligible models, including PhoBERT. | N6, N7, N13 | Huy (TV3) |
| N9 | Run controlled validation ablations, including the PhoBERT comparison. | N7, N8, N13 | Huy (TV3) |
| N10 | Run the single locked-test comparison after all candidates are frozen. | N9, N14 if applicable | TV1 điều phối; cả nhóm |
| N11 | Integrate the selected model into the API and frontend. | N6, N8 | Huy (TV3) |
| N12 | Deploy after test and integration gates pass. Consider ONNX only if it offers a measured benefit. | N10, N11 | Huy (TV3) |
| N13 | Run the accepted pretrained PhoBERT fine-tuning benchmark. | N2, N4 if augmentation is used | TV3 |
| N14 | Run optional pretrained PhoBERT fine-tuning extras on validation only. | N13, if applicable | Huy |
| N15 | Analyze errors after the final locked-test batch without tuning candidates. | N10 | Huy (TV3) lead |

TV1 là Tam, TV2 là Vinh, TV3 là Huy và TV4 là TheAnh, theo xác nhận của người dùng ngày 2026-11-23.

## Integration order and controlled experiments
The integration order is data → leakage → augmentation → embeddings → baselines → neural models → calibration → ablation → locked test → web/deploy. N5 and N6 can run in parallel after N2; the order above is not a dependency between them. The dependencies in the task table govern execution.

Use validation Macro-F1 to choose checkpoints and candidates. Report binary label counts, per-class precision/recall/F1, Macro-F1, TOXIC F1, confusion matrix, calibration/ECE when applicable, and run-level seed and data hash. Keep train, validation, and locked test roles distinct.

Compare LR with balanced class weights, BiLSTM with random embeddings and cross-entropy, FastText, and character CNN. Evaluate one selected imbalance treatment, augmentation, and calibration/threshold as separate ablations. Change one factor at a time. Fix the architecture before steps 5–7 so later ablations do not also change model structure. Run each configured candidate with three seeds.

PhoBERT is an accepted pretrained fine-tuning benchmark. P1 is `phobert-base` with the default D4 configuration; P2 adds the D3 imbalance treatment; optional P3 varies learning rate, epochs, or fine-tuning strategy one factor at a time on validation. Record the word segmentation tool and version. Use sequence length 128–160, hard limit about 256, learning rate `2e-5`, batch size 16, 10% warmup, weight decay `0.01`, 3–4 epochs, and validation Macro-F1 early stopping. Use fp16 and gradient accumulation where supported.

Before opening the test set, freeze every candidate, configuration, threshold, and analysis method in a versioned file lock. Run one test batch across that complete list. Do not use test results to tune, choose candidates, or revise thresholds. Any later test access is a new, disclosed evaluation, not part of model development.

## Pending verification
Confirm data access and UIT permission, source revisions and schemas, ViCTSD label encoding, the final source-specific mapping, and whether the existing G0 artifacts match D1. Check the split manifest and locked-test access controls before running experiments. No new dataset state or result is asserted here.

**Đợt này chỉ sửa SPEC/PLAN; không code, train, download, sửa config, hay deploy.**
