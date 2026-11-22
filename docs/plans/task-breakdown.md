# Task breakdown and acceptance criteria

**Status:** Plan only. N1–N15 and D1–D4 are defined in [the team execution plan](../../tasks/plan.md). Do not mark a task complete without its artifact or evidence.

## Tasks
| ID | Deliverable and acceptance criteria | Dependencies | Owner |
|---|---|---|---|
| N1 | Document D1–D4 and obtain team approval. | — | Cả nhóm |
| N2 | Create binary combined data, source tags, deduplication, stratified 70/15/15 split, Parquet and source/split statistics. Record unresolved permission and mapping checks. | N1 | TV3 lead; TV1 hỗ trợ |
| N3 | Audit exact duplicates, conflicts, and split leakage; record near-duplicate/thread limitations and evidence. | N2 | Chưa phân công |
| N4 | Generate augmentation from training TOXIC examples only, at most twice per original; preserve toxic tokens and reject validation/test duplicates. Report achieved proportion against the 30–35% target. | N3 | Chưa phân công |
| N5 | Train FastText from training data only and publish three-seed validation predictions, configs, hashes, and metrics. | N2 | Chưa phân công |
| N6 | Train calibrated LR and SVM; select by validation Macro-F1, report TOXIC F1, export baseline `.joblib` under 15 MB. | N2 | TV1 LR; TV2 SVM |
| N7 | Train scratch neural models including BiLSTM and Char-CNN; record VRAM under 7 GB target and select by validation Macro-F1. | N4, N5 | TV3 và TV4 |
| N8 | Fit calibration and choose thresholds on validation only for eligible N6/N7/N13 models. Report ECE and record the selection rule. | N6, N7, N13 if PhoBERT is permitted | Chưa phân công |
| N9 | Run controlled validation ablations. Change one factor at a time and use the specified baseline set. Fix the architecture before ablation steps 5–7. | N7, N8, N13 if permitted | Chưa phân công |
| N10 | Evaluate all frozen candidates and three seeds in one locked-test batch, create lock file and `locked_test.md`, report class metrics and bootstrap CI or McNemar. | N9, N14 if applicable | TV1 điều phối; cả nhóm |
| N11 | Integrate the selected model into API and frontend. Keep model selection and thresholds from validation. | N6, N8 | Chưa phân công |
| N12 | Deploy after N10 and N11 gates. Consider ONNX only if measured benefit warrants it. | N10, N11 | Chưa phân công |
| N13 | Optional PhoBERT benchmark, only after instructor approval. Use `phobert-base`, word segmentation, 128–160 tokens, learning rate `2e-5`, batch size 16, 10% warmup, weight decay `0.01`, 3–4 epochs, validation Macro-F1 early stopping, and fp16 or accumulation when supported. | N2, N4 if augmentation is used, instructor approval | TV3 |
| N14 | Optional extras, using validation only. | N13, if applicable | Huy |
| N15 | Analyze errors after N10 without using test errors to change candidates or settings. | N10 | TV3 lead; Huy supports |

The current TODO assigns PhoBERT to TV3 and error analysis to the TV3 lead with all members contributing. It does not identify Huy as TV3. Confirm the Huy assignments without inferring a member number.

## Shared contract and experiment controls
Use `id,text,label,source,split` as the dataset schema and `CLEAN=0`, `TOXIC=1` as labels. Keep source-specific mappings explicit because dataset meanings differ. UIT permission, source revisions, ViCTSD encoding, data availability, and correspondence of existing G0 outputs remain pending verification.

Use validation Macro-F1 for checkpoint and candidate selection. Report label counts, per-class precision/recall/F1, Macro-F1, TOXIC F1, confusion matrix, ECE when probabilities are available, runtime, and model size. Log three seeds, configs, dataset hashes, run IDs, checkpoints, and validation predictions.

The ablation set includes LR with balanced weights, BiLSTM with random embeddings and cross-entropy, FastText, character CNN, one validation-selected imbalance treatment, augmentation, and calibration/threshold. Change one factor per comparison. Do not use pretrained embeddings, LLMs, or back-translation in the core models.

Deduplicate before splitting. Keep the locked test unavailable for training, selection, calibration, threshold choice, extras, and debugging. Freeze candidates, configs, thresholds, and analysis methods in a versioned lock file before the single test batch. Disclose known historical test exposure; do not describe test analysis as tuning.

## Historical plan
The prior three-class CLEAN/OFFENSIVE/HATE breakdown is superseded. Retain it in repository history, not as current instructions.
