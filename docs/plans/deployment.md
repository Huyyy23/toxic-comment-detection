# N12 Gradio ZeroGPU deployment

## Target

Prepare `deployment/hf-gradio/` as an isolated Hugging Face Gradio Space bundle. Gradio is the demo UI, FastAPI preserves the REST API in the same Uvicorn process. The app listens on port 7860 and does not use Docker.

Public deployment is not cleared. Account eligibility, artifact/data distribution rights, N10 acknowledgement evidence, and ZeroGPU allocation for direct FastAPI routes remain unresolved. No Space was created or uploaded.

## Inference artifacts

| Model | Original artifact | Validation selection | Inference contract |
|---|---|---|---|
| N6-LR-D | `models/baselines/N6-LR-run-ea238c630a2f41b4b5be6b58edf4cdcd/EXP-N6-LR-D/seed_42/fallback_pipeline.joblib` | Macro-F1 `0.790095`; seeds 42/43/44 tied, seed 42 selected as the lowest-seed tie-break. | Teencode normalization, `predict_proba`, threshold 0.5, uncalibrated. |
| V3-TRI-GATED-FT | `models/retrained/v3/retrained_20261124/cnn/V3-TRI-GATED-FT/seed_43/best.pt` | Macro-F1 `0.793046`, F1 TOXIC `0.646727`. | `v3_word_norm`, `v2_char_norm`, frozen vocabulary, raw softmax, uncalibrated. |
| V3-PHOBERT-V2 | `models/retrained/v3/retrained_20261124/transformers/V3-PHOBERT-V2/seed_44/best.pt` | Macro-F1 `0.819487`, F1 TOXIC `0.694340`. | PyVi segmentation, pinned tokenizer revision, raw softmax, uncalibrated. |

The two V3 weights are validation-only replacements because the historical checkpoints were unavailable. No training was run. Validation selection did not use N10, N15, or locked test data.

## Package contents

- `app.py`: Gradio Blocks and FastAPI routes.
- `model_runtime.py`: fixed three-model registry, CPU baseline, on-demand neural inference, shared GPU lock.
- `preprocessing.py`: only inference-time preprocessing variants and vocabulary encoding required by the selected artifacts.
- `models/`: copied N6 joblib, TRI checkpoint/vocabulary/config, PhoBERT checkpoint/config.
- `requirements.txt`, `README.md`, `smoke_test.py`.

The bundle contains no datasets, validation predictions, training scripts, notebooks, experiment logs or unrelated checkpoints. Original files remain in place.

## Local verification

The local smoke test uses synthetic input and passed the REST routes, Swagger HTML, all three model IDs, comparison, binary labels, normalized probabilities, latency and calibration status. The browser check showed the single-model Gradio flow, three-model comparison and Swagger UI. A first REST pass hit a transient local CUDA out-of-memory error; a later complete pass succeeded. Hosted resource behavior is not verified.

## Space configuration

`README.md` contains frontmatter for `sdk: gradio`, Python 3.12, Gradio 5.50.0 and `app_file: app.py`. Uvicorn binds port 7860. `app_port` is deliberately omitted because it is Docker-only. `requirements.txt` uses PyTorch 2.8+ for ZeroGPU while retaining Transformers 4.57.6 and Hub below 1.0.

PhoBERT checkpoint already includes fine-tuned backbone weights. Runtime downloads only `config` and tokenizer at pinned revision. It does not fetch base-model weights again, quantize, or alter model architecture.

## Publication gates

1. Verify account email, age, good standing, ZeroGPU quota, and whether private ZeroGPU Spaces are available.
2. Verify publication rights for selected checkpoints and tokenizer/config; check dataset source permissions before public examples or uploads.
3. Locate existing N10 acknowledgement evidence without reopening N10 or the test set.
4. On an eligible private Space, verify direct REST inference allocation and all endpoints. If ZeroGPU requires a public Space, do not upload until rights gates close.
5. Free ZeroGPU documents a 5 GPU-minute daily quota for Free accounts and supports up to two ZeroGPU Spaces. Do not promise sustained demo availability.

After gates close, create a Gradio Space on ZeroGPU Free and upload only this directory with `hf upload <username>/<space-name> . --repo-type space`. Do not upload the project root, `data/`, training artifacts, notebooks or logs.
