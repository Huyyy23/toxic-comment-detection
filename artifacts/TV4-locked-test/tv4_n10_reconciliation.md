# N10 reconciliation — historical TV4 batch

**Status: BLOCKED for team-wide completion.** This record reconciles existing evidence only; it does not reopen or rerun the locked test.

## Existing batch scope

`tv4_n10_freeze_manifest.json` and `tv4_n10_locked_test_report.json` record a one-shot evaluation with `test_blind: false`. The manifest freezes seven TV4 single-model candidates, the TV2 `EXP-N6-C-calibrated` reference (seed 42), three TV4-only ensembles, and two TV2+TV4 cross-member ensembles. It includes TV4 seeds 42/43/44 and the TV2 reference configuration, run ID, validation threshold, and data hashes.

This proves coverage of TV4 and one TV2 reference only. It does not establish that TV1 and TV3 candidates were evaluated, that all four members signed off on the candidate list/protocol, or that the result is the approved team-wide batch. The existing report must remain historical evidence, not a blind test or a newly-run group result.

## N14 and access decision

**N14: COMPLETE — validation-only PhoBERT learning-rate comparison.** N2 results over seeds 42/123/2026: `1e-5` Macro-F1 `0.793272 ± 0.002131`, `3e-5` `0.800336 ± 0.004193`; N13 default `2e-5` baseline `0.801469 ± 0.003773`. Paired mean deltas vs N13 baseline: `-0.008197` (`1e-5`) and `-0.001133` (`3e-5`); neither alternate LR improves the three-seed mean, so no candidate was promoted. All runs used PhoBERT revision `01daacda68afe13d83023d16ec647239e344a1e6`, PyVi `0.1.1`, Transformers `4.57.6`; detailed run lineage and raw validation metrics are in `docs/experiments/phobert_extra_val.csv`. One initial `3e-5`/seed-42 run ended at the 3600s command timeout and was marked `FAILED`; successful retry `5bea48668cfa462e9206a188415e2c96` is used. Validation only; do not append these runs to the historical locked-test batch.

The shared runner hard-codes MLflow experiment/run labels as N13; N14 configs isolate the six completed runs in `models/N14-phobert/mlruns`, and `docs/experiments/phobert_extra_val.csv` maps each N14 setting to its config and run ID. Reported metrics are raw validation metrics at the default 0.50 decision boundary; no extra calibration or threshold tuning was performed.

**No further locked-test access.** No evidence in the current plan/checklists authorizes another access. Any later test access would be a separate, explicitly disclosed evaluation, not a continuation of this N10 batch.

## Remaining gate

Do not mark team-wide N10 complete. Keep all four member confirmations and EXP-08 open until the team lead reconciles the candidate inventory and obtains evidence-based sign-off. Existing validation inventory is documented in `artifacts/N8/n8_calibration_report.md` and `artifacts/N9/README.md`; it is not a freeze manifest. No new team-wide freeze manifest can be truthfully issued from the available records.

The remaining decision is governance, not a missing test run: TV1 lead Tam and TV3 reviewer Huy must explicitly decide whether to report the existing disclosed TV4+TV2 batch as the final historical N10 evidence, or state that team-wide N10 is not reproducible. TV2 Vinh and TV4 TheAnh must confirm the candidate-scope record; capture each response in the corresponding checklist/team context. N1 approval does not imply N10 sign-off. Without these confirmations, keep N10 BLOCKED; do not reopen test to fill the gap.

## Audit trail

- Repository baseline reviewed: `f357a44fb33f419009795e7f7df83a4b242ac521`.
- Initial N10 reconciliation method: read-only inspection of N8/N9 summaries, current task/checklist records, and the existing TV4 N10 manifest/report; no test-data access.
- N14 run evidence: six official N2 train/validation runs, two learning rates (`1e-5`, `3e-5`) × seeds `[42, 123, 2026]`; train SHA `097b32bea731bf1a4771a8485155d339a96027d2819fe8a38aa2c9d2e2f32a73`, validation SHA `4c33430a520818078aef6bcc34b3e2573873445dc7040c5729313e11bb60197a`; no test file was opened.
- Verification: `.venv/Scripts/python.exe -m pytest tests/test_n13_phobert.py -q` → `4 passed`; `docs/experiments/phobert_extra_val.csv` was checked against all six MLflow metrics/data hashes and the interrupted run's FAILED state.
- Additional read-only branch audit: `origin/tam` (`dfb9238`), `origin/feature/svm` (`096b815`), `origin/feature/tv3-scratch-pipeline` (`2f28246`), `origin/TheAnh` (`a33c10a`), and `origin/master` (`d767390`). Member checklists still show N10 pending for TV1/TV2/TV3; TV4 records its completed historical batch. No explicit all-member N10 sign-off was found in those snapshots.
- Owners with explicit N10 sign-off in the inspected records: none evidenced for the full team.
