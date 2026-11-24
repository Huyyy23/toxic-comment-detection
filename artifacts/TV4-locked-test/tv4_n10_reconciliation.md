# N10 reconciliation — historical TV4 batch

**Status: BLOCKED for team-wide completion.** This record reconciles existing evidence only; it does not reopen or rerun the locked test.

## Existing batch scope

`tv4_n10_freeze_manifest.json` and `tv4_n10_locked_test_report.json` record a one-shot evaluation with `test_blind: false`. The manifest freezes seven TV4 single-model candidates, the TV2 `EXP-N6-C-calibrated` reference (seed 42), three TV4-only ensembles, and two TV2+TV4 cross-member ensembles. It includes TV4 seeds 42/43/44 and the TV2 reference configuration, run ID, validation threshold, and data hashes.

This proves coverage of TV4 and one TV2 reference only. It does not establish that TV1 and TV3 candidates were evaluated, that all four members signed off on the candidate list/protocol, or that the result is the approved team-wide batch. The existing report must remain historical evidence, not a blind test or a newly-run group result.

## N14 and access decision

**N14: SKIPPED for this freeze.** N14 is optional; no N14 learning-rate/epoch/strategy extras are evidenced in the current record. N9's PhoBERT N2-vs-N4 augmentation comparison is not N14. Skipping adds no validation-selected candidate to the already-exposed batch.

**No further locked-test access.** No evidence in the current plan/checklists authorizes another access. Any later test access would be a separate, explicitly disclosed evaluation, not a continuation of this N10 batch.

## Remaining gate

Do not mark team-wide N10 complete. Keep TV1/TV2/TV3 sign-off and EXP-08 open until the team lead reconciles the candidate inventory and obtains evidence-based sign-off. Existing validation inventory is documented in `artifacts/N8/n8_calibration_report.md` and `artifacts/N9/README.md`; it is not a freeze manifest. No new team-wide freeze manifest can be truthfully issued from the available records.

## Audit trail

- Repository baseline reviewed: `f357a44fb33f419009795e7f7df83a4b242ac521`.
- Method: read-only inspection of N8/N9 summaries, current task/checklist records, and the existing TV4 N10 manifest/report. No execution, training, or access to test data.
- Owners with explicit N10 sign-off in the inspected records: none evidenced for the full team.
