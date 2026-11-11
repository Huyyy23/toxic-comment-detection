# Project TODO — Scratch-first v4.0
**Plan only — 2026-11-11.** Trạng thái dưới đây chỉ phản ánh **kế hoạch mới**, không suy ra code/scaffold cũ đã bị xóa hoặc đã chạy thí nghiệm.

## G0 — Data Contract (TV3 nhóm trưởng)
- [ ] Kiểm chứng source/license/schema/label 3 lớp của UIT-ViHSD (TV1 hỗ trợ audit).
- [ ] Check duplicate, nhãn mâu thuẫn và leakage; quyết định official split hoặc stratified 70/15/15.
- [ ] Khóa test; lưu dataset version/hash, seed, ID→split manifest; ban hành input/output contract.
- [ ] Cả 4 thành viên xác nhận dùng chung contract.

## Parallel work — các nhánh độc lập sau G0
- [ ] **TV1 EXP-01:** EDA; raw/basic/teencode + TF-IDF word/char + Logistic Regression; ≥3 official runs.
- [ ] **TV2 EXP-02:** LinearSVC + TF-IDF, tuning C/n-grams, feature selection; ≥3 official runs. Optional RBF SVC.
- [ ] **TV3 EXP-03:** random-init LSTM → BiLSTM → BiLSTM+Attention; ≥3 official runs.
- [ ] **TV4 EXP-04:** random-init TextCNN one/multi-kernel, filters/dropout; ≥3 official runs.
- [ ] Mỗi TV xuất checkpoint/config/hash/MLflow run IDs + validation predictions + Macro-F1/per-class F1.

## Improvement and research
- [ ] TV1: preprocessing ablation và phân tích lỗi Logistic Regression.
- [ ] TV2: SVM feature/hyperparameter ablation và training-cost comparison.
- [ ] TV3 EXP-05: cross-model error analysis ≥30 validation errors, hypotheses for improvement.
- [ ] TV4: CNN kernel/dropout ablation.
- [ ] **Conditional TV4 EXP-06:** error diversity → voting/stacking (OOF nếu dùng); hoặc ghi lý do skipped.
- [ ] **Optional TV3 EXP-07:** pretrained PhoBERT fine-tuning **chỉ sau khi scratch BiLSTM hoàn thành**, hoặc ghi lý do skipped.

## Finalization
- [ ] TV3 freeze candidate models/hyperparameters bằng validation, ghi danh sách cuối.
- [ ] TV1 + cả nhóm EXP-08: chạy locked test final một đợt, không tune theo test.
- [ ] Cả nhóm: reproducibility report/notebooks, 4 phần thí nghiệm cá nhân.
- [ ] Minimal FastAPI demo `/predict`, `/health`, `/model-info`; hosting chỉ sau khi kiểm tra eligibility.

## Coordination
- [ ] Mỗi thành viên branch riêng, PR review; không sửa split độc lập.
- [ ] Nếu chung GPU 8GB: schedule DL jobs, không chạy hai job lớn đồng thời.
- [ ] Mỗi tuần review MLflow runs và artifacts, không chỉ báo cáo miệng.

Source of truth: [spec](../docs/spec.md) · [experiment plan](../docs/plans/ml-experiments.md) · [task breakdown](../docs/plans/task-breakdown.md).
**Chỉ sửa PLAN/SPEC lần này — không triển khai bất cứ TODO nào.**
