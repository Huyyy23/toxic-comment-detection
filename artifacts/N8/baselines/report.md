# N8 calibration report — Baselines slice (N6 LR + LinearSVC)

## Status: PASS (with documented limitations)

Candidate families:
- `N6 Logistic Regression` variants A, B, C, D (raw) and `EXP-N6-LR-D-calibrated`
- `N6 LinearSVC` variant `N6-C-calibrated` (genuine sigmoid-calibrated probabilities)

## Inputs

- Run directories:
  - `models/baselines/N6-LR-run-0ede1adcfaa84d56803375b97621ac4f/` (EXP-N6-LR-A/B/C/D, 3 seeds each)
  - `models/baselines/N6-LR-run-1533b39d1efa4b70b4179f884d1143f0/` (EXP-N6-LR-D-calibrated, 3 seeds)
  - `models/baselines/N6-run-faffb8025fbc41fb8d9b463468bdd5f3/` (EXP-N6-C-calibrated, 3 seeds)
- Canonical validation: `data/processed/binary_n2_v1/val.parquet` (6,048 rows, binary CLEAN=0/TOXIC=1).
- Canonical validation SHA-256: `4c33430a520818078aef6bcc34b3e2573873445dc7040c5729313e11bb60197a` (matched all run configs).
- Validation ID count: 6,048; unique IDs: 6,048.
- Order-independent validation ID digest (sorted unique IDs joined by `\n`, SHA-256): `7f454a9926bf0a616c54426513b98874b944ea06921c927cd336c2203152f01d`.
- All configs report `n_val=6048` and `val_parquet_sha256` equal to canonical hash.

## Method

- No training or fine-tuning. Used existing validation predictions; regenerated P(TOXIC) from stored probabilities.
- Platt scaling fit on validation log-odds using `src/evaluation/cnn_report.py:fit_platt_scaling` / `apply_platt_scaling`.
- Raw and calibrated ECE via `compute_calibration_metrics(..., n_bins=10)`.
- Threshold chosen post-calibration by validation Macro-F1 using `sweep_probability_thresholds` (grid 0.10..0.90 by 0.01, existing tie behavior).
- All ECE/threshold metrics are in-sample validation diagnostics only; no generalization claim. No locked-test data or test predictions accessed.
- For artifacts already marked calibrated (`input_is_calibrated=true`), N8 Platt is an additional validation-fitted post-calibration; same-validation fit/evaluation optimism disclosed.
- Raw LinearSVC decision scores are not probabilities; genuine probabilities exist only for the calibrated SVM artifact.

## Results (aggregate)

- Included runs: 18 (15 LR + 3 SVM).
- Variants/seeds:
  - LR: EXP-N6-LR-A/B/C/D × seeds 42/43/44 (12 runs, raw probabilities)
  - LR: EXP-N6-LR-D-calibrated × seeds 42/43/44 (3 runs, train-fold calibrated probabilities)
  - SVM: N6-C-calibrated × seeds 42/43/44 (3 runs, genuine sigmoid-calibrated probabilities)
- Per-run summaries: `artifacts/N8/baselines/logistic_regression/*/seed_*/n8_summary.json`, `artifacts/N8/baselines/linear_svc/N6-C/seed_*/n8_summary.json`.
- Aggregate: `artifacts/N8/baselines/summary.json`.

## Excluded / blocked runs

- `models/baselines/N6-run-931168a6f09241c6a8c943fd99f26c7f/` (N6-A/B/C/D raw) and `models/baselines/N6-run-ff683d828c7244ee9124b396bf2988d4/` (N6-D): val predictions have `probabilities: null`; no saved model weights/joblib in those run directories. Raw SVM margins are unavailable, so raw pre-calibration ECE cannot be computed and no inference regeneration is possible. Reported as BLOCKED for raw probability calibration; only the calibrated `N6-C` artifact is included with genuine probabilities.
- Historical G0 three-class runs and N5 runs are out of scope for this slice.

## Outputs (all under `artifacts/N8/baselines/`)

- `summary.json` — aggregate N8 summary.
- `logistic_regression/<variant>/seed_<seed>/n8_summary.json` — per-run summaries with candidate/variant/seed/run ID/data hash/ID count+digest/raw+post ECE/Platt params/selected threshold/Macro-F1/Toxic-F1/source artifact paths/method/limitations.
- `logistic_regression/<variant>/seed_<seed>/val_calibrated_probabilities.csv` — calibrated validation probabilities.
- `linear_svc/N6-C/seed_<seed>/n8_summary.json` and `val_calibrated_probabilities.csv` — same.
- `report.md` — this report.

## Notes

- No model training or fine-tuning performed. No locked-test data or test predictions used. No source/docs/checklists edited; no commit made.
