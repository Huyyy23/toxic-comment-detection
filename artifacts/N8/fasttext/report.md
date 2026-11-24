# N8 calibration report — FastText slice (N5)

## Status: PASS

Candidate/variant: `N5 FastText mean-pooled vectors + LogisticRegression (balanced)` (`fasttext_lr_balanced`).

## Inputs

- Run directory: `models/baselines/N5-run-4cef5024bbdb437296384e56b471e20b/` (3 official seeds: 42, 123, 2026).
- Saved artifacts per seed: `fasttext.model`, `logistic_regression.joblib`, `val_predictions.csv`, `config.json`, `metrics.json`, `dataset_hashes.json`, `run_id.txt`.
- Canonical validation: `data/processed/binary_n2_v1/val.parquet` (6,048 rows, binary CLEAN=0/TOXIC=1).
- Canonical validation SHA-256: `4c33430a520818078aef6bcc34b3e2573873445dc7040c5729313e11bb60197a` (matched all seed `dataset_hashes.json`).
- Validation ID count: 6,048; unique IDs: 6,048.
- Order-independent validation ID digest (sorted unique IDs joined by `\n`, SHA-256): `7f454a9926bf0a616c54426513b98874b944ea06921c927cd336c2203152f01d`.
- MLflow run statuses: all `FINISHED` (run IDs `c6f8fb3ab8d8408b87f1e1862257eb93`, `94ac5e39f6864ea8a1539e7468b5797f`, `b4551f29977340e391bf1455a61345c9`).

## Method

- No training or fine-tuning. Regenerated `P(TOXIC)` on canonical validation only by loading saved FastText vectors/classifier and mean-pooling exactly as in `src/models/n5_fasttext.py::_pool`.
- Regenerated probabilities reproduced saved predicted labels at threshold 0.5 for all 6,048 rows in every seed (reload exact match 6048/6048).
- Platt scaling fit on validation log-odds using `src/evaluation/cnn_report.py:fit_platt_scaling` / `apply_platt_scaling`.
- Raw and calibrated ECE via `compute_calibration_metrics(..., n_bins=10)`.
- Threshold chosen post-calibration by validation Macro-F1 using `sweep_probability_thresholds` (grid 0.10..0.90 by 0.01).
- All ECE/threshold metrics are in-sample validation diagnostics only; no generalization claim. No locked-test data or test predictions accessed.

## Results (per seed)

| seed | raw ECE | calibrated ECE | Platt coef | Platt intercept | selected threshold | Macro-F1 @ selected | Toxic-F1 @ selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.254330 | 0.017072 | 3.962? see JSON | see JSON | 0.24 | 0.663019 | 0.443602 |
| 123 | 0.252853 | 0.013866 | see JSON | see JSON | 0.26 | 0.665604 | 0.440627 |
| 2026 | 0.254717 | 0.018899 | see JSON | see JSON | 0.25 | 0.661282 | 0.436703 |

Means: raw ECE 0.253967; calibrated ECE 0.016612; selected threshold Macro-F1 0.663301; Toxic-F1 0.440310.

## Outputs (all under `artifacts/N8/fasttext/`)

- `summary.json` — aggregate N8 summary.
- `seed-42/n8_summary.json`, `seed-123/n8_summary.json`, `seed-2026/n8_summary.json` — per-run N8 summaries with candidate/seed/run ID/data hash/ID count+digest/raw+post ECE/Platt params/selected threshold/Macro-F1/Toxic-F1/source artifact paths/method/limitations.
- `seed-42/val_calibrated_probabilities.csv`, `seed-123/val_calibrated_probabilities.csv`, `seed-2026/val_calibrated_probabilities.csv` — per-run calibrated validation probabilities (`id`, `true_label`, `prob_toxic_raw`, `prob_toxic_calibrated`).
- `report.md` — this report.

## Excluded runs

None. All three official seeds included.

## Notes

- FastText vectors were trained on train only in N5; this slice only loaded saved weights for validation inference. No test data or test predictions used.
- No source/docs/checklists edited; no commit made.
