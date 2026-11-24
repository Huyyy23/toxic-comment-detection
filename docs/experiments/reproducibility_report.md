# Báo cáo tái lập — trạng thái hiện có

**Trạng thái: V3 validation-only retrain và inference replay đạt 16/16.** Báo cáo đối chiếu artifacts lịch sử với artifacts retrained; không mở test, không tuyên bố blind test. Notebook validation-only: [`notebooks/EXP-05-reproducibility-analysis.ipynb`](../../notebooks/EXP-05-reproducibility-analysis.ipynb).

## Split và dữ liệu

- Dataset: `data/processed/binary_n2_v1/`; nhãn `CLEAN=0`, `TOXIC=1`.
- Validation: 6,048 dòng, SHA-256 `4c33430a520818078aef6bcc34b3e2573873445dc7040c5729313e11bb60197a`.
- Train N2 SHA-256: `097b32bea731bf1a4771a8485155d339a96027d2819fe8a38aa2c9d2e2f32a73`; train augmentation N4 hash `2f42eaaee33efe96a41816ecbb112df07473da20ef0641d7f3b6df2754bd488d`; validation giữ nguyên.
- Dataset provenance, mapping và giới hạn nhãn/quyền: `docs/dataset-requirements.md`.

## Bằng chứng theo nhóm thí nghiệm

| Nhóm | Bằng chứng hiện có | Giới hạn tái lập |
|---|---|---|
| N6 Logistic Regression | `docs/TV1_N6_REPORT.md`; raw/calibrated run artifacts gồm config, metrics, validation predictions, seeds và run metadata. Các metric replay khớp báo cáo ở độ chính xác đã ghi. | Môi trường của run lịch sử không đủ giải thích chênh lệch kích thước serialization. YAML C/D `C=1.0` khác run config được chọn trên validation `C=5.0`; dùng config theo từng run, không thay bằng YAML. |
| N6 LinearSVC | `docs/TV2_N6_REPORT.md`; run configs/metrics/predictions và seeds được lưu. | Raw margins không phải xác suất. Calibrated threshold `0.34` được chẩn đoán trên cùng validation nên không phải đánh giá độc lập. |
| N7/N8/N9 | `docs/todos/TV3.md`, `artifacts/N8/`, `artifacts/N9/`; seed/run/split/ablation summaries. | Threshold/calibration cùng validation; không diễn giải như kết quả test. |
| N13/N14 PhoBERT | `docs/todos/TV3.md`, `docs/experiments/phobert_extra_val.csv`; model revision, package versions, run IDs, configs/checkpoints/predictions và metric theo seed. | Runner ghi một phần N14 dưới tên N13; mapping được mô tả trong `docs/experiments/phobert_extra_val.csv` và N10 reconciliation. |
| TV4 CNN V1 / V3 | `docs/TV4_CNN_REPORT.md`, `docs/TV4_V3_PROTOCOL_AND_LOG.md`; V3 TRI và PhoBERT có prediction/metrics/config cho seeds 42/43/44. | Báo cáo V3 ghi kỳ vọng `best.pt` ở sáu seed dirs (21.41 MB TRI, 515.06 MB PhoBERT), nhưng file không có local hoặc trên public `master` (`28f14b7cb50cad91ec57ae7eef91a3653fc1c1d3`). GitHub API listing/path-history cho sáu `best.pt` không có checkpoint; hai selected paths trên `TheAnh` cũng không có; query các branch khác cho selected paths trả 404; releases/tags rỗng. Không có checksum/URI/artifact-store record. Không thay bằng weights tương tự. |

## V3 checkpoint retraining and inference replay — 16/16 PASS on validation

- Current artifact inventory found **16 missing checkpoints** for runs with saved validation results: 12 scratch runs (four variants × seeds 42/43/44), PhoBERT seeds 42/43/44, and the completed XLM-R seed 42. XLM-R seeds 43/44 were incomplete/not run and are excluded.
- 12 CNN and three PhoBERT runs completed in the first pass. XLM-R seed 42 first failed three times while creating two CPU copies of its 768 MB embedding tensor (`detach().cpu().clone()`); changing to one independent `to(device="cpu", copy=True)` snapshot let the unchanged config/seed complete. Retry run ID `4c73a3a054c44021bff7a0edf590a01e`, best epoch 2, 678.77 s, peak VRAM 5,497.85 MB.
- XLM-R training log also emitted PyTorch's warning about `lr_scheduler.step()` preceding an optimizer step. The scheduler/training logic was left unchanged to preserve the requested run settings; this remains a limitation for exact reproduction.
- All 16 missing-checkpoint runs now have new artifacts under `models/retrained/v3/retrained_20261124/`; original paths remain untouched. Validation comparisons and checkpoint inference replay completed. Every prediction label matched saved validation predictions; no test file or test prediction/label was opened.
- `sha256_manifest.json` inventories 1,454 files (including run logs, configs, metrics, predictions, checkpoints, and tracking outputs); manifest SHA-256 `d8d0bdbca7db6bddf469f3e3e2497e5847b02cbb4353027dd6b602d9a18963d9`. XLM-R `best.pt` SHA-256 `0bc96250f03983d553c01b7b13c67112b7f771eb0fee96ce9370b76cbbcc6918`.
- Historical V3 did not record immutable PhoBERT/XLM-R revisions. The new runs pin current Hugging Face revisions (`86cd7fd4c148980922ac11a2cf5e257f2ba639e1` and `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`); this does not prove byte-identical reproduction of the original initialization.

Kết quả dưới đây là mean ± population SD qua ba seed, với Δ = retrained − historical; mô tả validation, không phải bằng chứng test hay lý do chọn candidate mới.

| Variant | Macro-F1 historical | Macro-F1 retrained | Δ | F1 TOXIC historical | F1 TOXIC retrained | Δ |
|---|---:|---:|---:|---:|---:|---:|
| V3-BIGRU-GATED-MSD | 0.785325 ± 0.002594 | 0.785483 ± 0.003759 | +0.000157 | 0.633529 ± 0.007007 | 0.634501 ± 0.005153 | +0.000971 |
| V3-BIGRU-GATED-FT | 0.789995 ± 0.002388 | 0.791347 ± 0.000752 | +0.001351 | 0.642047 ± 0.005541 | 0.642040 ± 0.002856 | −0.000007 |
| V3-BIGRU-PYVI-FT | 0.787890 ± 0.001940 | 0.791747 ± 0.001176 | +0.003858 | 0.638811 ± 0.003072 | 0.643394 ± 0.002574 | +0.004583 |
| V3-TRI-GATED-FT | 0.790701 ± 0.003377 | 0.791102 ± 0.002246 | +0.000401 | 0.642125 ± 0.008268 | 0.642689 ± 0.003540 | +0.000564 |
| V3-PHOBERT-V2 | 0.813315 ± 0.002310 | 0.816248 ± 0.002293 | +0.002933 | 0.685050 ± 0.002330 | 0.689356 ± 0.003616 | +0.004305 |
| V3-XLMR-BASE seed 42 | 0.800593 | 0.793331 | −0.007261 | 0.660764 | 0.649675 | −0.011089 |

## Validation pair được dùng trong EXP-05/N15

- Scratch: `V3-TRI-GATED-FT`, seeds 42/43/44, raw Macro-F1 `0.790701 ± 0.003377`.
- PhoBERT: `V3-PHOBERT-V2`, seeds 42/43/44, raw Macro-F1 `0.813315 ± 0.002310`.
- Sample 30 cũ không có membership manifest và không thể xác minh chính xác; không dùng sample-hit counts cũ. `EXP05-REVIEW-V2` là sample mới với quota/tie-break/hash được freeze rõ, fingerprint `33c19e709c013b8dc7be740687635e26e4215ee37727becb621495dffbfde2bb`; 30 interpretations remain AI-provisional. Nhóm trưởng chấp nhận phạm vi kỹ thuật N15 chỉ là phân tích lỗi validation có hỗ trợ AI; member adjudication, privacy/legal/data rights và comprehensive leakage review vẫn mở. Chi tiết/checklist: `error_analysis.md`, `manual_review_checklist.md`.
- Notebook integrity cells verify saved historical validation/prediction hashes, run/config mapping, ID/label alignment, calibration boundary, paired errors, and the V2 sample fingerprint. Its retrained replay cell was executed for all 16 runs: exact label matches, maximum absolute probability difference `0.00011591` (tolerance `1.5e-4`, `1e-5` relative). Notebook outputs contain no text/IDs; the separate N15 checklist now contains redacted validation text and must remain internal.

## Rủi ro và phần chưa chứng minh

- N10 là limited historical acceptance; test split đã có historical exposure được ghi trong `artifacts/TV4-locked-test/tv4_n10_reconciliation.md`. Báo cáo này không dùng test metrics làm evidence mới, không mở test files và không kết luận blind evaluation.
- Original checkpoints remain unavailable and were not restored. New retrained files are not byte-identical recovery of historical weights; inference replay verifies only that each new checkpoint reproduces its own saved validation outputs within the recorded tolerance.
- Quyền phân phối/permission, khác biệt taxonomy nguồn, và near-duplicate/thread leakage chưa được giải quyết đầy đủ (`docs/dataset-requirements.md`).
- Historical training's immutable base-model revisions/runtime versions and TRI FastText initialization hash are not available, so original runs cannot be proven byte-identical. New transformer retrains pin current resolved revisions; that does not establish historical initialization. New checkpoint inference replay passed, but original-checkpoint replay remains impossible.
- N6 LR vẫn thiếu một phần metadata môi trường lịch sử để giải thích serialization-size drift; YAML C/D `C=1.0` không khớp run config validation-selected `C=5.0`, nên giữ config theo run, không thay bằng YAML.
- Chưa có xác nhận cá nhân của từng thành viên trong các phần báo cáo; bản tổng hợp ở `docs/experiments/member_report_sections.md` chỉ là draft evidence-based.

## Kiểm tra local đã chạy

- Notebook code was syntax-checked and all cells executed against validation artifacts. Replay passed for all 16 checkpoints with exact labels and maximum probability drift `0.00011591`; no test artifacts were opened and no notebook outputs containing text/IDs were saved. N15 review text is only in the internal, redacted checklist.
