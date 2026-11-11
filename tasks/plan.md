# Implementation Plan: Vietnamese Toxic Comment Detection System

## Overview
Dự án xây dựng một hệ thống Machine Learning hoàn chỉnh từ khâu nghiên cứu dữ liệu tiếng Việt, đối sánh khoa học giữa mô hình truyền thống (Traditional ML) và mô hình học sâu Transformer (PhoBERT/ViSoBERT) trên **cùng một tập test 3 lớp duy nhất (CLEAN / OFFENSIVE / HATE)**, đóng gói suy luận đa backend, và triển khai ứng dụng Web hiện đại (**React frontend + FastAPI backend trong Docker container**) trên Hugging Face Spaces.

Tài liệu chi tiết được tổ chức tại thư mục `docs/plans/`:
- Tổng quan kế hoạch & Critical Path: [`docs/plans/implementation-plan.md`](../docs/plans/implementation-plan.md)
- Kế hoạch Thực nghiệm Học máy: [`docs/plans/ml-experiments.md`](../docs/plans/ml-experiments.md)
- Kế hoạch Triển khai & Ràng buộc Hạ tầng: [`docs/plans/deployment.md`](../docs/plans/deployment.md)
- Chi tiết Phân rã Công việc & Acceptance Criteria: [`docs/plans/task-breakdown.md`](../docs/plans/task-breakdown.md)

---

## Architecture Decisions & Strategy
1. **Label Taxonomy**: Phân loại 3 lớp chuẩn: `0: CLEAN`, `1: OFFENSIVE`, `2: HATE`. Không tự chuyển về Binary.
2. **Dataset Strategy**: Tải cả `ViCTSD` và `UIT-ViHSD` cho EDA. Ưu tiên `UIT-ViHSD` làm dataset chính.
3. **Training Hardware**: Local GPU NVIDIA GeForce RTX 4060 8GB (Mixed Precision fp16, VRAM target < 7GB). Google Colab/Kaggle là backup.
4. **Inference Engine**: Interface `BasePredictor` đa backend (`SklearnBackend`, `PyTorchBackend`, `ONNXBackend` có điều kiện theo benchmark).
5. **Web Architecture**: React (Vite) Frontend + FastAPI Backend trong Multi-stage Docker container.
6. **Hosting**: Hugging Face Spaces (Docker SDK, port 7860). Xác minh điều kiện tạo Docker Space ngay tại Milestone 0.

---

## Critical Path & Execution Order
`M0` (Setup & HF Check) ──→ `M1` (Data & EDA) ──→ `M2` (Baseline) ──→ `M4` (Inference Engine) ──→ `M5` (React + FastAPI) ──→ `M3` (Transformer on RTX 4060) ──→ `T4.2` (Integration Test) ──→ `M6` (Docker & HF Deploy)

---

## Tasks Index
Chi tiết từng task, tiêu chí nghiệm thu và quy trình kiểm thử được theo dõi trực tiếp tại: **[`tasks/todo.md`](todo.md)**.
- Milestone 0: Cấu trúc Dự án, Khung Kiểm chuẩn & Xác minh HF (`T0.1` - `T0.3`)
- Milestone 1: Pipeline Dữ liệu & Báo cáo EDA 3 Lớp (`T1.1` - `T1.4`)
- Milestone 2: Mô hình Cơ sở Truyền thống 3 Lớp (`T2.1` - `T2.3`)
- Milestone 3: Thực nghiệm Transformer trên GPU RTX 4060 (`T3.1` - `T3.3`)
- Milestone 4: Inference Engine Đa Backend & Kiểm thử Tích hợp (`T4.1` - `T4.3`)
- Milestone 5: Xây dựng Ứng dụng Web React + FastAPI (`T5.1` - `T5.4`)
- Milestone 6: Dockerization, Hugging Face Deployment & Nghiệm thu (`T6.1` - `T6.3`)
