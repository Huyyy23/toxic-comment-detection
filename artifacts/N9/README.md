# N9 — Validation ablations

## Phạm vi và protocol

`ablation_val.csv` hợp nhất 63 dòng: candidate context N5/N6/N7 (18), controlled TV4 CNN ablations (33), PhoBERT N2-vs-N4 (6), và BiLSTM+Attention initialization N9 (6). Mọi candidate đều dùng validation hash `4c33430a520818078aef6bcc34b3e2573873445dc7040c5729313e11bb60197a`; train hashes khớp N2 hoặc N4 theo từng slice.

`val_macro_f1` và `val_f1_toxic` là raw validation metrics tại threshold 0.50. `threshold` và `calibrated_ece` theo N8: Platt scaling và threshold sweep fit/chọn trên cùng validation, nên là số liệu in-sample. Không đọc hoặc dùng locked test trong N9.

## N9 BiLSTM+Attention: random vs N5 FastText initialization

Controlled comparison trên N4, cùng split, kiến trúc, embedding dimension 100, hyperparameters và từng seed; chỉ thay initialization. N5 FastText model khớp seed được dùng cho từng FastText arm.

| Seed | Random Macro-F1 | FastText Macro-F1 | Δ Macro-F1 | Random F1 TOXIC | FastText F1 TOXIC | Δ F1 TOXIC |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.616598 | 0.700357 | +0.083759 | 0.460207 | 0.530222 | +0.070015 |
| 123 | 0.556772 | 0.685172 | +0.128400 | 0.419363 | 0.514120 | +0.094758 |
| 2026 | 0.581196 | 0.660965 | +0.079769 | 0.436124 | 0.493634 | +0.057509 |
| Mean ± population SD | 0.584855 ± 0.024560 | 0.682165 ± 0.016222 | +0.097309 ± 0.022045 | 0.438565 ± 0.016764 | 0.512659 ± 0.014973 | +0.074094 ± 0.015478 |

FastText initialization scored higher on these three paired validation seeds. This is descriptive evidence on the fixed validation split, not a significance or generalization claim.

## Other slices

- TV4 CNN comparisons are in `slices/tv4_ablation/ablation.csv`; mean raw Macro-F1 deltas vs CNN-02: CNN-03a +0.001458, CNN-03b +0.001233, CNN-04a dropout −0.008811, CNN-04b weight decay +0.002343, class weights −0.001452, focal loss −0.003312, weighted sampler −0.007125, N4 augmentation −0.001561.
- PhoBERT N4 augmentation minus N2 baseline paired mean deltas: Macro-F1 −0.009799, F1 TOXIC −0.015460; rows and per-seed evidence are in `slices/phobert_ablation/ablation.csv`.
- Candidate context rows summarize N5/N6/N7 evidence; they are not a same-protocol controlled comparison. See `slices/core_candidates/README.md` for metric/source conventions.

## Evidence

Official N9 recurrent runs and validation predictions are under `models/recurrent_n9_fasttext_init/mlruns/256095358878457704/`. N4 train SHA-256: `f57fbf0a55de6b08c8cba1cd366c3895bfed495f2817069f892ba27b5786542e`. No final test result is reported here; candidate freeze and locked-test evaluation remain separate plan gates.
