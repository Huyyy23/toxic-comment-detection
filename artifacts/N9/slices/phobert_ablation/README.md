# N13 PhoBERT train-only augmentation ablation

**Status: PASS.** The CSV contains six per-seed rows: paired N2 baseline and N4 augmentation runs for seeds 42, 123, and 2026. The only contrast is the training dataset/augmentation arm. `val_macro_f1` and `val_f1_toxic` are the raw validation metrics recorded by each N13 run; `threshold`, `calibrated_ece`, and `raw_ece` come from its corresponding N8 calibration entry.

N13 sources: each row's `models/N13-phobert/mlruns/598020008752758771/<run_id>/artifacts/metrics.json`. N8 source: `artifacts/N8/phobert/n8_phobert_summary.json` and per-run `artifacts/N8/phobert/calibrated_validation_probabilities_<run_id>.json` (paths are listed in `source_artifact`). The six official run IDs are 462781192f5d4c46bd41c7e035203923, fdb1a801f08947fd81635f891713cd65, 79180ecaf6f94579896419362f0fc8cb, cf1d45b2d7244957a948647b64a1666f, eeda02eadf084fb59ceabbf3da0924c4, and 87df4ae80b434a37ae266ba5318c045c.

All runs use canonical validation data: file SHA-256 `4c33430a520818078aef6bcc34b3e2573873445dc7040c5729313e11bb60197a`; sorted validation-ID SHA-256 `7f454a9926bf0a616c54426513b98874b944ea06921c927cd336c2203152f01d`. N2 train SHA-256: `097b32bea731bf1a4771a8485155d339a96027d2819fe8a38aa2c9d2e2f32a73`; N4 train SHA-256: `f57fbf0a55de6b08c8cba1cd366c3895bfed495f2817069f892ba27b5786542e`.

N8 fits Platt scaling on validation log-odds and selects a threshold by maximizing validation Macro-F1. Thus threshold and calibrated ECE are optimistic, in-sample validation diagnostics, not generalization estimates. No test data or test artifacts were accessed. All requested metric fields are available; no missing fields.
