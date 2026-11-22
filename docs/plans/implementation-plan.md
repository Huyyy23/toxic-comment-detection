# Implementation plan — binary toxic-comment detection

**Status:** Plan only. The agreed D1–D4 decisions and N1–N15 sequence are maintained in [the team execution plan](../../tasks/plan.md). This page records how owners and integration should follow that source of truth.

## Scope and decisions
Combine UIT-ViHSD and ViCTSD as `CLEAN=0`, `TOXIC=1`. Map ViHSD labels greater than zero and ViCTSD Toxicity to TOXIC, retaining `vihsd` or `victsd` as `source`. Their label meanings differ, and UIT permission remains pending. Deduplicate before stratifying 70/15/15 by `(source, label)`; lock the test split. Verify existing G0 artifacts against this contract before reuse.

Train LR, SVM, LSTM, BiLSTM with attention, character CNN, and FastText without pretrained embeddings. PhoBERT is an optional pretrained benchmark and requires instructor approval. Do not use LLMs or back-translation in the core comparison. Augment only training TOXIC samples, at most twice per original, targeting 30–35% without guaranteeing that proportion. Preserve toxic tokens and exclude validation/test duplicates. Select one imbalance method on validation from class weights, focal loss, or a sampler. Select checkpoints by validation Macro-F1; report TOXIC F1 and ECE as additional metrics.

## Ownership and work order
| Task | Owner | Dependencies |
|---|---|---|
| N1 documentation | Cả nhóm | — |
| N2 combined data and locked split | TV3 lead; TV1 hỗ trợ | N1 |
| N3 leakage audit | Chưa phân công | N2 |
| N4 train-only augmentation | Chưa phân công | N3 |
| N5 FastText | Chưa phân công | N2 |
| N6 LR and SVM baselines | TV1 LR; TV2 SVM | N2 |
| N7 scratch neural models | TV3 và TV4 | N4, N5 |
| N8 validation calibration and threshold | Chưa phân công | N6, N7, N13 if PhoBERT is permitted |
| N9 validation ablations | Chưa phân công | N7, N8, N13 if permitted |
| N10 one locked-test batch | TV1 điều phối; cả nhóm | N9, N14 if applicable |
| N11 API and frontend integration | Chưa phân công | N6, N8 |
| N12 deployment | Chưa phân công | N10, N11 |
| N13 optional PhoBERT | TV3 | N2, N4 if augmentation is used, instructor approval |
| N14 optional extras | Huy | N13, if applicable |
| N15 error analysis | TV3 lead; Huy supports | N10 |

The existing `tasks/todo.md` assigns PhoBERT to TV3 and error analysis to TV3 lead, with all members contributing. It does not identify Huy as TV3. Confirm the N14 and N15 supporting assignment with the team; do not infer Huy's member number.

Integrate in this order: data, leakage audit, augmentation, embeddings, baselines, neural models, calibration, ablation, locked test, web and deployment. N5 and N6 may run in parallel after N2. This integration order does not add a dependency between them.

## Reproducibility and final evaluation
Use the same binary split for all models. Log source and dataset hashes, configuration, seed, run ID, checkpoint, and validation predictions. Run three seeds. Report class counts, per-class precision/recall/F1, Macro-F1, TOXIC F1, confusion matrix, calibration/ECE where applicable, runtime, and model size.

Compare LR with balanced weights, BiLSTM with random embeddings and cross-entropy, FastText, and character CNN. Test one imbalance treatment, augmentation, and calibration/threshold in separate ablations; change one factor at a time. Fix the architecture before ablation steps 5–7.

PhoBERT is a pretrained fine-tuning benchmark pending lecturer confirmation. P1 uses the default `phobert-base` configuration, P2 adds the selected imbalance treatment, and optional P3 changes one fine-tuning factor at a time on validation. Record the segmentation tool and version, use 128–160 tokens, learning rate `2e-5`, batch size 16, 10% warmup, weight decay `0.01`, 3–4 epochs, and early stopping on validation Macro-F1. Use fp16 and gradient accumulation when supported.

Before any test access, freeze all candidates, configurations, thresholds, and analysis methods in a versioned lock file. Run one final test batch over that fixed list. Never use test metrics or errors to tune or select models. Report bootstrap confidence intervals or McNemar comparisons in that batch. Any known earlier test exposure must be disclosed. Consider ONNX only if measurements show a benefit.

## Pending checks
UIT permission, source revisions and schemas, ViCTSD label encoding, source-specific mapping, correspondence of existing G0 artifacts with D1, test exposure history, and locked-test access controls remain to verify. This document does not claim that the revised dataset or results exist.
