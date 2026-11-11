# Kế hoạch Thực thi Dự án (Implementation Plan)
## Vietnamese Toxic Comment Detection System (ML-first, API-minimal)

**Mã kế hoạch**: `PLAN-TOXIC-COMMENT-2026-V3.0-ML-FIRST`  
**Phiên bản**: `3.0.0-ML-FIRST`  
**Trạng thái**: `DRAFT — PENDING USER REVIEW`  
**Dựa trên**: `SPEC.md`, `docs/spec.md`, `docs/plans/ml-experiments.md`, `docs/decisions.md`, `docs/dataset-requirements.md`, `docs/deployment-plan.md`, `docs/open-questions.md`  
**Ngày cập nhật**: 2026-11-11  

---

## 1. TÓM TẮT THAY ĐỔI CHÍNH (Phiên này)

Dự án được định hướng lại từ **full-stack (React + FastAPI trong Docker)** sang **ML-first, API-minimal**:

- **Giữ**: FastAPI REST API tối giản (3 endpoint: `POST /predict`, `GET /health`, `GET /model-info`), Swagger UI tại `/docs`, probability contract, locked test set, MLflow local tracking.
- **Loại bỏ khỏi phạm vi**: `frontend/` (React + Vite) — sẽ xóa hoàn toàn ở phiên implementation sau; `Dockerfile`/`docker-compose.yml` chỉ giữ nếu hosting yêu cầu; auth/JWT, user DB, dashboard, CRUD không liên quan ML, CI/CD phức tạp, distributed training.
- **MLflow**: local (file store), không chạy server riêng.
- **Ensemble/Stacking**: không bắt buộc, chỉ nếu có bằng chứng thực nghiệm.
- **Không xóa source code trong phiên này.** Chỉ lập kế hoạch loại bỏ an toàn ở các phiên sau.

---

## 2. PHÂN LOẠI THÀNH PHẦN (Status Ledger)

| Nhóm | Thành phần | Ý nghĩa | Hành động đề xuất |
|---|---|---|---|
| **GIỮ** | `SPEC.md`, `docs/`, `backend/schemas.py`, `configs/*.yaml`, `tests/test_smoke.py`, `src/__init__.py`, `backend/__init__.py` | Hoạt động tốt và cần thiết | Giữ nguyên, cập nhật nếu cần |
| **CẢI TIẾN** | `docs/decisions.md`, `docs/deployment-plan.md`, `docs/open-questions.md`, `docs/hf_eligibility_report.md`, `docs/dataset-requirements.md` | Có thể tái sử dụng nhưng cần nâng cấp cho ML-first | Đã cập nhật trong phiên này |
| **BỔ SUNG** | MLflow local config, experiment docs, ablation docs, error analysis docs, inference.py (production), minimal FastAPI `main.py` | Chức năng còn thiếu | Tạo trong các phiên implementation sau |
| **LOẠI BỎ (kế hoạch, chưa xóa code)** | `frontend/` (React + Vite) | Không còn phục vụ mục tiêu ML-first | Kế hoạch xóa hoàn toàn ở phiên implementation sau; không xóa trong phiên này |
| **TÙY CHỌN** | `Dockerfile`, `docker-compose.yml` | Giữ nếu hosting yêu cầu container | Không bắt buộc trong phiên này |

---

## 3. ĐỒ THỊ PHỤ THUỘC & ĐƯỜNG DẪN GĂNG (Critical Path)

```
[M0: Audit & Architecture Simplification]   (chỉ tài liệu, đã làm trong phiên này)
       │
       ▼
[M1: Data & EDA]                           (EXP-001)
       │
       ▼
[M2: Baseline Models]                      (EXP-002, EXP-003)
       │
       ▼
[M3: Advanced Transformer Training]        (EXP-004, EXP-005)
       │
       ▼
[M4: Ensemble & Ablation]                  (EXP-006, EXP-007 conditional, EXP-008)
       │
       ▼
[M5: Evaluation & Research Documentation]  (EXP-009, Notebook reports)
       │
       ▼
[M6: REST API Deployment]                  (FastAPI tối giản, 3 endpoint, Swagger UI)
```

**Đường dẫn găng (Critical Path):**
`M0` → `M1` → `M2` → `M3` → `M4` → `M5` → `M6`

M3 (Transformer) có thể chạy song song với một phần M4/M5 sau khi có checkpoint, nhưng model selection phải chờ M3 xong.

---

## 4. TỔNG KẾT CÁC MILESTONES (6 Milestones)

| Milestone | Mục tiêu | Experiments liên quan | Đặc trưng task |
|---|---|---|---|
| **M0** | Audit & Architecture Simplification | — | XS/S (tài liệu) |
| **M1** | Data & EDA | EXP-001 | XS/S |
| **M2** | Baseline Models (Traditional ML) | EXP-002, EXP-003 | S/M |
| **M3** | Advanced Transformer Training | EXP-004, EXP-005 | M/L (GPU) |
| **M4** | Ensemble & Ablation Study | EXP-006, EXP-007 (conditional), EXP-008 | S/M |
| **M5** | Evaluation & Research Documentation | EXP-009 | XS/S |
| **M6** | REST API Deployment (API-minimal) | — | S/M |

**Tổng số milestone**: 6 (M0–M6).  
**Tổng số task ước lượng**: ~20–24 atomic tasks (phân rã chi tiết ở `docs/plans/task-breakdown.md`).

---

## 5. MA TRẬN RỦI RO & PHƯƠNG ÁN XỬ LÝ (Risk Register)

| Mã | Rủi ro | Mức độ | Kế hoạch dự phòng / Giảm thiểu |
|---|---|---|---|
| R1 | Hugging Face Space chưa có token / yêu cầu thẻ tín dụng | Cao | Phương án dự phòng: Render Free Docker, Railway, hoặc chạy container Docker local. |
| R2 | Mất cân bằng 3 lớp (HATE ít mẫu nhất) | Cao | Class-weighted loss, Macro-F1, stratified sampling, không dùng Accuracy đơn lẻ. |
| R3 | Transformer OOM trên GPU 8GB | Trung bình | Giảm `max_length`, `batch_size`; mixed precision `fp16`; chuyển Colab/Kaggle T4 làm backup. |
| R4 | Dùng Test set để tuning (data leakage) | Chí mạng | Khóa hoàn toàn `test.parquet`; toàn bộ chọn mô hình trên Validation; chỉ unseal Test một lần tại EXP-009. |
| R5 | Dataset cũ không tổng quát hóa ngôn ngữ mạng xã hội mới | Trung bình | Đánh giá cross-dataset (ViCTSD vs UIT-ViHSD), ghi nhận giới hạn trong báo cáo. |
| R6 | Ensemble/Stacking quá tốn tài nguyên, không mang lại lợi ích | Trung bình | Có phương án thay thế: Soft Voting; nếu không cải thiện ≥ 1% Macro-F1, giữ mô hình đơn tốt nhất. |

---

## 6. NHỮNG VIỆC KHÔNG LÀM TRONG PHIÊN NÀY (Confirmation)

- Không viết code Python.
- Không sửa training pipeline.
- Không refactor source code.
- **Không xóa React/frontend** (chỉ lập kế hoạch xóa ở phiên sau).
- Không cài đặt package.
- Không thay đổi môi trường.
- Không download dataset.
- Không xử lý dữ liệu.
- Không chạy training.
- Không chạy inference.
- Không chạy experiment.
- Không tạo Notebook thực thi.
- Không tạo API implementation.
- Không tạo Dockerfile.
- Không public deployment.
- Không git commit/push nếu chưa được yêu cầu.

Chỉ tạo hoặc chỉnh sửa file tài liệu (Markdown): SPEC.md, docs/spec.md, docs/plans/*.md, docs/decisions.md, docs/deployment-plan.md, docs/open-questions.md, docs/hf_eligibility_report.md, docs/dataset-requirements.md.
