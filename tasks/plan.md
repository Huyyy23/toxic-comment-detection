# Implementation Plan: Vietnamese Toxic Comment Detection System

## Overview
Dự án xây dựng một hệ thống Machine Learning hoàn chỉnh từ khâu nghiên cứu dữ liệu tiếng Việt, đối sánh khoa học khách quan giữa mô hình truyền thống (Traditional ML) và mô hình học sâu Transformer (PhoBERT/ViSoBERT) trên **cùng một tập test 3 lớp duy nhất (CLEAN / OFFENSIVE / HATE) bị khóa hoàn toàn trong quá trình phát triển**, đóng gói suy luận đa backend với hợp đồng xác suất hiệu chuẩn, và triển khai ứng dụng Web hiện đại (**React frontend + FastAPI backend trong Docker container**) trên Hugging Face Spaces.

Tài liệu chi tiết được tổ chức tại thư mục `docs/plans/`:
- Tổng quan kế hoạch & Critical Path: [`docs/plans/implementation-plan.md`](../docs/plans/implementation-plan.md)
- Kế hoạch Thực nghiệm Học máy & Evaluation Protocol: [`docs/plans/ml-experiments.md`](../docs/plans/ml-experiments.md)
- Kế hoạch Triển khai, Docker & Giao nhận Artifact: [`docs/plans/deployment.md`](../docs/plans/deployment.md)
- Chi tiết Phân rã 24 Tasks & Acceptance Criteria: [`docs/plans/task-breakdown.md`](../docs/plans/task-breakdown.md)

---

## Architecture Decisions & Strategy
1. **Giao thức Đánh giá (Evaluation Protocol)**:
   - Tập Train (70%): Huấn luyện weights.
   - Tập Validation (15%): Tuning siêu tham số, early stopping, và lựa chọn mô hình (Model Selection).
   - Tập Test (15% - **LOCKED**): Khóa hoàn toàn, chỉ mở ra đúng 1 lần tại `T6.1` để đối sánh khách quan.
2. **Label Taxonomy**: Phân loại 3 lớp chuẩn: `0: CLEAN`, `1: OFFENSIVE`, `2: HATE`. Không tự chuyển về Binary.
3. **Hợp đồng Xác suất (Probability Contract)**:
   - Logistic Regression dùng `predict_proba`.
   - LinearSVC bắt buộc dùng `CalibratedClassifierCV` (Platt scaling).
   - Tổng xác suất 3 lớp $\approx 1.0$. Không giả mạo confidence.
4. **Siêu tham số Tạm thời (`[PROVISIONAL]`)**:
   - `max_length = 160`, `batch_size = 16`, `learning_rate = 2e-5`, `VRAM budget < 7.0GB`, TF-IDF `max_features = 10000` được đánh dấu tạm thời, chốt sau EDA và hardware profiling.
5. **Tách biệt Kiểm thử Tích hợp**:
   - `T4.2`: Baseline Integration Test (kiểm thử Baseline sớm, unblock Web MVP).
   - `T4.3`: Transformer Integration Test (kiểm thử Transformer sau huấn luyện trên RTX 4060).
   - `T4.4`: Conditional ONNX Benchmark (chỉ chuyển đổi khi có lợi ích thực tế).
6. **Quy trình Artifacts cho Production Docker**:
   - Version hóa qua Hugging Face Model Hub (pinned tag/revision); tự động fallback sang Baseline `.joblib` đóng gói sẵn nếu thiếu weights, ngăn ngừa container crash.
7. **Đánh giá Khách quan (Objective Acceptance)**:
   - Báo cáo trung thực toàn bộ chỉ số bất kể Transformer cao hơn, tương đương hay thấp hơn Baseline.

---

## Critical Path & Execution Order
`M0` (Setup & HF Check) ──→ `M1` (Data & EDA) ──→ `M2` (Baseline on Val) ──→ `T4.1` (Inference Interface) ──→ `T4.2` (Baseline Integration Test) ──→ `M5` (React + FastAPI Web MVP) ──→ `M3` (Transformer on RTX 4060) ──→ `T4.3` (Transformer Integration Test) ──→ `T4.4` (ONNX Benchmark) ──→ `T6.1` (Locked Test Eval) ──→ `T6.2` (Docker) ──→ `T6.3` (Deploy) ──→ `T6.4` (Acceptance)

---

## Tasks Index (24 Tasks)
Theo dõi chi tiết tại: **[`tasks/todo.md`](todo.md)**.
- Milestone 0: Cấu trúc Dự án, Khung Kiểm chuẩn & Xác minh HF (`T0.1` - `T0.3`)
- Milestone 1: Pipeline Dữ liệu, EDA & Phân chia 3 Lớp (`T1.1` - `T1.4`)
- Milestone 2: Mô hình Cơ sở Truyền thống (Validation Tuned) (`T2.1` - `T2.3`)
- Milestone 3: Thực nghiệm Transformer trên GPU RTX 4060 (`T3.1` - `T3.3`)
- Milestone 4: Inference Engine Đa Backend & Kiểm thử Tích hợp Tách biệt (`T4.1` - `T4.4`)
- Milestone 5: Xây dựng Ứng dụng Web React + FastAPI (`T5.1` - `T5.4`)
- Milestone 6: Đánh giá Tập Test Đã Khóa, Docker & Deployment (`T6.1` - `T6.4`)
