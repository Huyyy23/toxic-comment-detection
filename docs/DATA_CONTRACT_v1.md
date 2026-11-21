# Data Contract v1.0 — UIT-ViHSD candidate

**Status: APPROVED BY USER; G0 acceptance checks PASS. G0 is OPEN subject to the documented source-permission and historical-test limitations.** The single shared candidate for TV1–TV4 is `data/processed/vihsd_g0_v1/`. User approval and technical acceptance do not claim that the test is blind or that external source permissions were granted. See `docs/G0_FINAL_REVIEW.md` for evidence.

## Record schema

UTF-8 JSONL. Every record contains exactly:

| Field | Type | Contract |
|---|---|---|
| `id` | string | Source-stable, unique identifier. |
| `text` | string | Original source text, nonblank; unchanged. |
| `label` | integer | `0=CLEAN`, `1=OFFENSIVE`, `2=HATE`; no relabeling. |
| `split` | string | `train`, `val`, or `test`. |

No normalization is performed when building the dataset. Experiment-specific preprocessing variants remain separate; any learned vocabulary/vectorizer/statistic is fit on train only and applied to validation/test without fitting.

## Dataset/version provenance

- Dataset: UIT-ViHSD, Hugging Face mirror `ura-hcmut/UIT-ViHSD` ([dataset card](https://huggingface.co/datasets/ura-hcmut/UIT-ViHSD), [API](https://huggingface.co/api/datasets/ura-hcmut/UIT-ViHSD)). The API reports immutable repo SHA `16c4f67cf509d4f9f36ca5b63c5503c7c8830557` (API checked 2026-11-21); generated local raw file hashes are recorded separately.
- The HF card tags `CC-BY-NC-SA-4.0`, but explicitly describes this as a copy of UIT's original and says to obtain permission from [UIT NLP datasets](https://nlp.uit.edu.vn/datasets) before use. **The original UIT permission/license and a project-specific permission grant have not been verified.** Treat the HF tag as a mirror's metadata, not confirmation of UIT terms; this is a limitation requiring documentation/permission review, not a reason to infer rights.
- Citation: Luu, Son T.; Nguyen, Kiet Van; Nguyen, Ngan Luu-Thuy. “A Large-scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts.” *Advances and Trends in Artificial Intelligence* (2021). [arXiv:2103.11528](https://arxiv.org/abs/2103.11528), [Springer DOI:10.1007/978-3-030-79457-6_35](https://link.springer.com/chapter/10.1007/978-3-030-79457-6_35).
- Raw files are immutable inputs. The builder records SHA-256 for each raw file and every generated artifact.
- All text groups whose exact text has conflicting labels are excluded from the modeling candidate; their source records remain in raw. No label is inferred or altered. Excluded source IDs are recorded in `excluded_conflict_ids.json`.
- Exact duplicate records with the same label are retained; exact text is the group key and all members receive one split.

## Split and test handling

- Deterministic Python stdlib seeded greedy stratified group assignment; seed `42`; target train/val/test ratios `70/15/15`.
- Group integrity and class proportions are prioritized; realized counts can differ from exact ratios when groups are indivisible. Realized counts live in `dataset_info.json`.
- `train_val.jsonl` contains only train and val. `test_locked.jsonl` contains only test and must not be supplied to development/training/tuning/model selection. User-approved EXP-08 policy accepts the existing pool with historical exposure disclosed; do not claim a blind test.
- Manifest maps each included candidate ID to its split, including test, for identity/hash consistency. It is metadata, not a mechanism for access control. Test rows remain in a separate file; access should be limited operationally before EXP-08.
- Historical exposure: the previous re-split test was present in shared processed artifacts. The new test is locked for future development use only; it cannot undo possible prior exposure. Report this limitation in final results.
- No thread/group identifier exists in the current raw schema. Exact-text group leakage is prevented; near-duplicate/thread leakage is not certified.

## Rebuild

```bash
python -m src.data.build_g0_dataset --raw-dir data/raw --output-dir data/processed/vihsd_g0_v1
```

The builder refuses to overwrite a nonempty output directory. Use a new versioned destination for each change. It never writes raw files. Generated SHA-256 values, split statistics, conflict counts/ledger hash and historical test warning are in `dataset_info.json`.
