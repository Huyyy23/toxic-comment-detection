# EXP-05 / N15 — Phân tích validation

**Status: TECHNICAL COMPLETE — LIMITED N15 SCOPE ACCEPTED BY GROUP LEAD; HUMAN/DATA FOLLOW-UP OPEN.** Scope: validation-only paired error analysis and AI-assisted review of the fixed 30-case sample using existing artifacts. No production model selection; that remains gated by N12 and related decisions. No test dataset, prediction, label or evaluation was opened or run. Human adjudication, privacy/legal clearance, dataset rights and comprehensive near-duplicate/thread leakage remain unfinished and are not implied by this acceptance.

## N15 — Chọn cặp ứng viên theo validation

Quy tắc: xếp hạng từng variant single-model bằng trung bình raw validation Macro-F1 của ba seed hiện có; không xếp từng seed như ứng viên độc lập, không dùng threshold/calibration hoặc ensemble để chọn. Phạm vi là official single-model validation inventory đã ghi nhận; đây không phải parse mới toàn bộ artifacts legacy/undocumented.

| Candidate | Seeds | Raw validation Macro-F1 (mean ± population SD) | F1 TOXIC (mean ± population SD) |
|---|---|---:|---:|
| Scratch: `V3-TRI-GATED-FT` | 42/43/44 | 0.790701 ± 0.003377 | 0.642125 ± 0.008268 |
| PhoBERT: `V3-PHOBERT-V2` | 42/43/44 | 0.813315 ± 0.002310 | 0.685050 ± 0.002330 |

`V3-TRI-GATED-FT` là scratch single-model có validation Macro-F1 cao nhất trong inventory được báo cáo; kết quả N6-LR-D là 0.7901, thấp hơn. `V3-PHOBERT-V2` là PhoBERT single-model có validation Macro-F1 cao nhất trong inventory này; N13 PhoBERT N2 baseline là 0.801469 ± 0.003773. PhoBERT là pretrained benchmark riêng, không tham gia chọn scratch. Không chọn mô hình production.

**Artifacts:**
- Scratch predictions/config/metrics: `models/cnn_v3/TV4-V3-scratch/V3-TRI-GATED-FT/seed_{42,43,44}/`
- PhoBERT predictions/config/metrics: `models/transformers_v3/V3-PHOBERT-V2/seed_{42,43,44}/`
- Shared validation: 6,048 rows, SHA-256 `4c33430a520818078aef6bcc34b3e2573873445dc7040c5729313e11bb60197a`.
- Per-seed run IDs and prediction hashes được lưu trong artifacts trên và tổng hợp trong `docs/TV4_V3_PROTOCOL_AND_LOG.md`, `docs/TV4_V3_FINAL_REPORT.md`.

| Seed | TRI errors | PhoBERT errors | Both wrong | TRI-only wrong | PhoBERT-only wrong | Prediction disagreements |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 606 | 584 | 357 | 249 | 227 | 476 |
| 43 | 620 | 625 | 389 | 231 | 236 | 467 |
| 44 | 657 | 579 | 363 | 294 | 216 | 510 |

Không gộp examples giữa seeds. Với seed 42: TRI false negatives 447, false positives 159; PhoBERT false negatives 327, false positives 257. Hai mô hình có trade-off khác nhau.

## EXP-05 — Error analysis validation-only

The previously reported seed-42 sample's selected IDs and complete tie-break implementation were not retained, so exact membership is unverified. Its sample-specific stratum/proxy-hit counts are withdrawn. `EXP05-REVIEW-V2` below is a new, reproducible review sample; it is not claimed to match the earlier 30.

### EXP05-REVIEW-V2 — provisional review

Because the prior selection could not be traced, this is a newly versioned seed-42 sample, not the old 30. It contains 10 each from both-wrong, TRI-only-wrong, and PhoBERT-only-wrong. For each group, allocate 10 proportional largest-remainder quotas by `(lowercase(source), gold label)`; floors first, then descending fractional remainder with ties by source then numeric label. Select within each stratum by ascending SHA-256 of UTF-8 `str(id)`, then order the selected ten by the same hash within each error group. No quota ties occurred.

**Retraining boundary:** `EXP05-REVIEW-V2` membership and seed-42 predictions remain historical validation artifacts; no new sample was selected after retraining. Three independent AI arms have now reviewed all 30 cases; human review and privacy clearance remain pending.

Stratum counts: both wrong `2/5/0/3`, TRI-only `2/5/1/2`, PhoBERT-only `5/3/2/0` (order: ViHSD/CLEAN, ViHSD/TOXIC, ViCTSD/CLEAN, ViCTSD/TOXIC). Ordered sample fingerprint (IDs joined by LF, no trailing LF): `33c19e709c013b8dc7be740687635e26e4215ee37727becb621495dffbfde2bb`. No sample IDs are persisted in the member worksheet; it now contains text with direct names/aliases redacted and must remain internal pending rights/privacy clearance.

The table below is the initial assistant-generated draft and is **superseded by the three-arm review** in [`manual_review_checklist.md`](manual_review_checklist.md). Categories are qualitative prompts, not verified causes, gold-label adjudication, or member approval. The row-level revised review lives in that checklist.

| # | Error group | Source | Gold | Provisional category | Confidence | Ambiguity |
|---:|---|---|---:|---|---|---|
| 1 | Both wrong | ViCTSD | 1 | Indirect/context-dependent meaning | Medium | High |
| 2 | Both wrong | ViCTSD | 1 | Lexical false alarm | Medium | Medium |
| 3 | Both wrong | ViHSD | 1 | Direct abuse | High | Low |
| 4 | Both wrong | ViHSD | 0 | Direct abuse | High | Low |
| 5 | Both wrong | ViHSD | 1 | Lexical false alarm | Medium | Medium |
| 6 | Both wrong | ViHSD | 1 | Indirect/context-dependent meaning | Medium | Medium |
| 7 | Both wrong | ViHSD | 1 | Other / unclear | Low | High |
| 8 | Both wrong | ViHSD | 1 | Indirect/context-dependent meaning | Medium | Medium |
| 9 | Both wrong | ViHSD | 0 | Indirect/context-dependent meaning | Medium | High |
| 10 | Both wrong | ViCTSD | 1 | Indirect/context-dependent meaning | Medium | High |
| 11 | TRI-only wrong | ViHSD | 1 | Indirect/context-dependent meaning | Medium | Medium |
| 12 | TRI-only wrong | ViCTSD | 1 | Indirect/context-dependent meaning | Medium | Medium |
| 13 | TRI-only wrong | ViHSD | 1 | Indirect/context-dependent meaning | Medium | Medium |
| 14 | TRI-only wrong | ViHSD | 1 | Direct abuse | High | Low |
| 15 | TRI-only wrong | ViCTSD | 0 | Lexical false alarm | Medium | Low |
| 16 | TRI-only wrong | ViHSD | 1 | Direct abuse | Medium | Medium |
| 17 | TRI-only wrong | ViCTSD | 1 | Lexical false alarm | High | Low |
| 18 | TRI-only wrong | ViHSD | 1 | Direct abuse | Medium | Medium |
| 19 | TRI-only wrong | ViHSD | 0 | Lexical false alarm | Medium | Low |
| 20 | TRI-only wrong | ViHSD | 0 | Other / unclear | Low | High |
| 21 | PhoBERT-only wrong | ViHSD | 0 | Indirect/context-dependent meaning | Medium | High |
| 22 | PhoBERT-only wrong | ViHSD | 0 | Direct abuse | High | Low |
| 23 | PhoBERT-only wrong | ViCTSD | 0 | Lexical false alarm | Medium | Medium |
| 24 | PhoBERT-only wrong | ViHSD | 1 | Direct abuse | High | Low |
| 25 | PhoBERT-only wrong | ViHSD | 0 | Other / unclear | High | High |
| 26 | PhoBERT-only wrong | ViCTSD | 0 | Lexical false alarm | High | Low |
| 27 | PhoBERT-only wrong | ViHSD | 0 | Lexical false alarm | Medium | Medium |
| 28 | PhoBERT-only wrong | ViHSD | 0 | Lexical false alarm | Medium | Medium |
| 29 | PhoBERT-only wrong | ViHSD | 1 | Direct abuse | Medium | Medium |
| 30 | PhoBERT-only wrong | ViHSD | 1 | Obfuscation / nonstandard form | Medium | Medium |

| Provisional category | Cases |
|---|---:|
| Indirect/context-dependent meaning | 9 |
| Lexical false alarm | 9 |
| Direct abuse | 8 |
| Other / unclear | 3 |
| Obfuscation / nonstandard form | 1 |
| **Total** | **30** |

The category/confidence totals below describe the initial assistant draft only; they are withdrawn as final findings. Use the revised row-by-row table in `manual_review_checklist.md`; all AI interpretations remain provisional until human review. The checklist has blank member decisions; do not publish it or infer approval.

| Slice | n | TRI errors | PhoBERT errors | Both wrong | Union errors | Union error rate |
|---|---:|---:|---:|---:|---:|---:|
| CLEAN | 5,084 | 159 | 257 | 91 | 325 | 6.4% |
| TOXIC | 964 | 447 | 327 | 266 | 508 | 52.7% |
| ViHSD | 4,559 | 449 | 444 | 260 | 633 | 13.9% |
| ViCTSD | 1,489 | 157 | 140 | 97 | 200 | 13.4% |
| Comment dài 0–49 ký tự | 3,476 | 258 | 267 | 160 | 365 | 10.5% |
| 50–99 ký tự | 1,413 | 169 | 168 | 99 | 238 | 16.8% |
| 100–199 ký tự | 790 | 113 | 98 | 60 | 151 | 19.1% |
| 200+ ký tự | 369 | 66 | 51 | 38 | 79 | 21.4% |

### Probes heuristic, không phải nhãn ngôn ngữ đã xác minh

Validation text was read only in memory for the heuristic probes and the provisional `EXP05-REVIEW-V2` review; no text/ID is persisted or published. Regex teencode, emoji Unicode, and punctuation-between-word probes have false positives/negatives and are not human-validated annotations.

| Proxy | Proxy-present rows | Union errors / rate | Proxy-absent rows | Union errors / rate |
|---|---:|---:|---:|---:|
| Teencode keyword regex | 819 | 128 / 15.6% | 5,229 | 705 / 13.5% |
| Emoji Unicode range | 637 | 67 / 10.5% | 5,411 | 766 / 14.2% |
| Punctuation obfuscation regex | 289 | 44 / 15.2% | 5,759 | 789 / 13.7% |

Tỷ lệ lỗi union tăng theo độ dài trong seed 42; đây là mô tả, chưa chứng minh nguyên nhân. Source rates gần nhau; không kết luận có source effect. Giả thuyết về ngữ cảnh, annotation ambiguity, độ dài và teencode cần blinded review/privacy approval để kiểm chứng.

## Giới hạn và bằng chứng còn thiếu

- `EXP05-REVIEW-V2` has an assistant-generated provisional interpretation, not human/member confirmation. The original sample remains untraceable. Category agreement, gold-label adjudication, and any privacy-safe paraphrase require the blank `manual_review_checklist.md`; no example is approved for publication.
- Proxy regex không được human-validated; phân tích theo bins không kiểm soát đồng thời source và class.
- Quyền sử dụng dữ liệu và kiểm tra near-duplicate/thread leakage còn thiếu theo `docs/dataset-requirements.md` §§N2–N3.
- Candidate inventory dùng phạm vi official đã ghi nhận; undocumented/superseded artifacts không được quét lại toàn diện.
- Cùng validation split được dùng qua nhiều lựa chọn trước đây; các kết quả không phải đánh giá độc lập. Không dùng các con số này để tuyên bố blind test, production selection hoặc cải thiện tổng quát hóa.
- Test boundary: chỉ validation dataset/predictions/configs và tài liệu hiện có được dùng; không mở test data/predictions/labels.
