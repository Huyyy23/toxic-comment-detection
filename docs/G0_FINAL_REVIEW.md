# G0 / Data Contract v1 — acceptance

**Gate: OPEN** — user-approved policy; dataset acceptance checks pass. No official training was run. Source permission and historical test exposure are disclosed limitations.

## Official shared dataset

All team experiments use **`data/processed/vihsd_g0_v1/`** only. Contract: [`DATA_CONTRACT_v1.md`](DATA_CONTRACT_v1.md). Rebuild:

```text
python -m src.data.build_g0_dataset --raw-dir data/raw --output-dir data/processed/vihsd_g0_v1
```

Builder refuses to overwrite a non-empty output directory and never writes raw inputs. Do not use the legacy `data/processed/vihsd.jsonl` or earlier `g0_v1_candidate*` outputs for official experiments.

| Split | Records | CLEAN | OFFENSIVE | HATE |
|---|---:|---:|---:|---:|
| Train | 23,002 | 19,168 | 1,476 | 2,358 |
| Validation (`val`) | 4,929 | 4,108 | 316 | 505 |
| Locked test | 4,929 | 4,108 | 316 | 505 |
| **Total** | **32,860** | **27,384** | **2,108** | **3,368** |

Approved policy: seed 42, deterministic stdlib greedy stratification grouped by exact text, target 70/15/15. Exclude all 241 conflicting-text groups (540 source records), without changing labels; retain same-label duplicates, assigned as a group. Exact cross-split overlap is zero. No thread IDs exist in raw; near-duplicate/thread leakage is not certified.

`train_val.jsonl` contains train+val only; `test_locked.jsonl` is separate. A prior re-split test was present in shared artifacts, so this replacement test is **not claimed blind**. It is frozen for one-shot EXP-08; do not use it for tuning, model/preprocessing selection, or development. File separation is not OS-level access control.

## Provenance, license, citation

Verified from primary sources on 2026-11-21:

- HF mirror [`ura-hcmut/UIT-ViHSD`](https://huggingface.co/datasets/ura-hcmut/UIT-ViHSD); pinned repository SHA `16c4f67cf509d4f9f36ca5b63c5503c7c8830557` ([HF API](https://huggingface.co/api/datasets/ura-hcmut/UIT-ViHSD)).
- HF card says `CC-BY-NC-SA-4.0`, but explicitly calls the repo a copy of UIT's original and requests usage permission from [UIT NLP datasets](https://nlp.uit.edu.vn/datasets). The original UIT license and a project-specific permission grant are **unverified**. HF mirror metadata is not treated as proof of original-source permission.
- Citation: Luu, Son T.; Nguyen, Kiet Van; Nguyen, Ngan Luu-Thuy. “A Large-scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts.” *Advances and Trends in Artificial Intelligence* (2021). [arXiv:2103.11528](https://arxiv.org/abs/2103.11528), [DOI 10.1007/978-3-030-79457-6_35](https://doi.org/10.1007/978-3-030-79457-6_35).
- Snapshot download date and UIT permission grant were not independently established. Candidate metadata records revision, source links, paper, mirror-license caveat, raw and generated artifact hashes.

## Acceptance results

| Criterion | Result | Evidence |
|---|---|---|
| Raw data unchanged | **PASS** | Raw SHA-256 before/after identical. |
| Conflicting labels | **PASS** | 241 exact-text conflict groups / 540 records excluded by ID; no relabeling. |
| Split counts/class distribution | **PASS** | 23,002 / 4,929 / 4,929; class proportions aligned across splits. |
| IDs/manifest | **PASS** | 32,860 unique IDs; manifest exactly matches every record split. |
| Test separated from development | **PASS** | 27,931 train+val records in `train_val.jsonl`; test only in `test_locked.jsonl`. Test used only for integrity/hash/group checks, not model or preprocessing selection. |
| Exact-text leakage | **PASS** | Zero exact-text groups across splits; same-label duplicates group-assigned. |
| SHA-256 consistency | **PASS** | All raw and generated hashes match candidate metadata. |
| Reproducibility | **PASS** | Two independent builds from same raw+seed produced byte-identical artifacts. |
| Automated tests | **PASS** | `python -m pytest tests/test_g0_dataset.py -q` — 2 passed. |
| Historical exposure | **ACCEPTED LIMITATION** | Prior test was in shared artifact; disclose in EXP-08 report. |
| UIT original permission | **UNVERIFIED** | HF mirror requests permission; no grant in repository. Do not publicly redistribute without permission review. |
| Near-duplicate/thread leakage | **NOT VERIFIED** | No thread/group IDs; exact-text scope only. |

Raw SHA-256:

| File | SHA-256 |
|---|---|
| `vihsd_train.jsonl` | `42a4fb3a948e5a9018e0854a14653825437d10419e952704080835d75fc068e7` |
| `vihsd_validation.jsonl` | `02155da4eafc7b7731da6ad2409463ce06ceab387302a0f369da101411404363` |
| `vihsd_test.jsonl` | `f3f890659dae2fdace5ab3ae3229eeb642d543c887a0f1db46a15a294aa10f77` |

Candidate SHA-256:

| File | SHA-256 |
|---|---|
| `train_val.jsonl` | `20a00f47d09d3439de83d4b371b253bd5991bd3bf6ff61871176381868d9d778` |
| `test_locked.jsonl` | `b88a13c7718073ac85a777d9458a22c90e55f04be8e062a96ae76e26ba244f87` |
| `manifest.json` | `1a3364ae6dcdc50e4f55e4bda665c5b4065c2835d2d98db808e60f4932c2edb2` |
| `excluded_conflict_ids.json` | `37041f6782581b0febcb5ae2da529ee121ac3f7c8b454d95c53163212bbf4421` |
| `dataset_info.json` | `f8406608dbf4dda0e1bca9b347a7b76b0b438530d9361af0b930f19c9d8ec015` |

## Final status

**G0: OPEN.** User-approved policy and technical acceptance pass. Official training is authorized by this gate, not run in this task. All team members use the single shared directory above, fit learned transforms on train only, and reserve test for one-shot EXP-08; report historical exposure. Original UIT permission remains unverified; public redistribution is not authorized by this report.
