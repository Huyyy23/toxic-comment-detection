# Kế hoạch Thực thi Dự án (Implementation Plan)
## Vietnamese Toxic Comment Detection System (3-Class: React + FastAPI in Docker)

- **Mã kế hoạch**: `PLAN-TOXIC-COMMENT-2026-V2.1`
- **Phiên bản**: `2.1.0-REVISED`
- **Trạng thái**: **REVISED & SYNCHRONIZED**
- **Dựa trên**: `SPEC.md`, `docs/spec.md`, `docs/decisions.md`, `docs/dataset-requirements.md`, `docs/deployment-plan.md`, `docs/open-questions.md`
- **Ngày cập nhật**: 2026-11-11

---

## 1. PHÂN LOẠI THÔNG TIN DỰ ÁN (Status Ledger)

| Trạng thái | Hạng mục |
| :--- | :--- |
| **Confirmed** | - Mục tiêu: Nhận diện bình luận độc hại tiếng Việt, phân loại **3 lớp CLEAN / OFFENSIVE / HATE**.<br>- Dataset: Tải cả `ViCTSD` và `UIT-ViHSD` cho EDA; ưu tiên `UIT-ViHSD` làm tập chính.<br>- Giao thức đánh giá: **Validation set dùng cho tuning và model selection; Test set được KHÓA (LOCKED)** chỉ đánh giá đối kháng cuối cùng.<br>- Hợp đồng xác suất: Calibrated probabilities ($\sum P \approx 1.0$), bọc LinearSVC bằng `CalibratedClassifierCV`.<br>- Tiêu chí so sánh: Đánh giá khách quan, báo cáo trung thực, không đặt điều kiện Transformer bắt buộc vượt Baseline.<br>- Phần cứng train Transformer: **Local GPU NVIDIA GeForce RTX 4060 8GB** (Google Colab/Kaggle là backup).<br>- Kiến trúc Web: **React (Frontend) + FastAPI (Backend)** đóng gói trong **Docker container**.<br>- Hosting mục tiêu: **Hugging Face Spaces** (Docker SDK, port 7860).<br>- Quản lý Artifact: Pinned revision từ HF Model Hub + Fallback baseline joblib tích hợp sẵn. |
| **Provisional** | - Các siêu tham số chưa kiểm chứng: `max_length = 160`, `batch_size = 16`, `learning_rate = 2e-5`, `VRAM budget < 7.0GB`, TF-IDF `max_features = 10000`. Sẽ quyết định sau EDA và hardware profiling. |
| **Blocked / Pending Token** | - Xác minh quyền triển khai Hugging Face Docker Space (`T0.3`): Script đã có, báo cáo ghi nhận `ATTENTION / PENDING TOKEN` do chưa có `HF_TOKEN`. Giữ nguyên blocker và phương án dự phòng. |

---

## 2. ĐỒ THỊ PHỤ THUỘC & ĐƯỜNG DẪN GĂNG (Dependency Graph & Critical Path)

### 2.1. Đồ thị Phụ thuộc Điều chỉnh (Decoupled Baseline & Transformer Integration)

```
[M0: Project Setup & HF Check (Pending Token)]
       │
       ▼
[M1: Data Pipeline & EDA 3 Lớp]
       │
       ▼
[M2: Traditional ML Baseline (Tuned on Val)]
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼ (Song song trên RTX 4060)
[T4.1: Inference Engine Interface]           [M3: Transformer Fine-tuning]
       │                                          │ (Tuned on Val)
       ▼                                          │
[T4.2: Baseline Integration Test]                 │
       │                                          │
       ▼ (Unblock Web MVP sớm)                    │
[M5: React + FastAPI Web App MVP]                 │
       │                                          │
       └──────────────────┬───────────────────────┘
                          ▼
             [T4.3: Transformer Integration Test]
                          │
                          ▼
             [T4.4: Conditional ONNX Benchmark]
                          │
                          ▼
             [T6.1: Locked Test Set Final Evaluation]
                          │
                          ▼
             [T6.2: Docker Multi-stage & Artifacts]
                          │
                          ▼
             [T6.3: HF Spaces Deploy & Live Smoke Test]
                          │
                          ▼
             [T6.4: Final README & Acceptance Report]
```

### 2.2. Đường dẫn Găng (Critical Path)
`M0` (Setup) ──→ `M1` (Data & EDA) ──→ `M2` (Baseline on Val) ──→ `T4.1` (Inference Interface) ──→ `T4.2` (Baseline Integration Test) ──→ `M5` (React + FastAPI) ──→ `M3` (Transformer on RTX 4060) ──→ `T4.3` (Transformer Integration Test) ──→ `T4.4` (ONNX Benchmark) ──→ `T6.1` (Locked Test Eval) ──→ `T6.2` (Docker) ──→ `T6.3` (Deploy) ──→ `T6.4` (Acceptance)

---

## 3. TỔNG KẾT CÁC MILESTONES (7 Milestones - 24 Tasks)

- **Milestone 0: Project Setup, Quality Scaffold & HF Verification** (3 tasks: `T0.1`, `T0.2`, `T0.3`).
- **Milestone 1: Data Pipeline, EDA & 3-Class Preprocessing** (4 tasks: `T1.1`, `T1.2`, `T1.3`, `T1.4`).
- **Milestone 2: Traditional ML Baseline (Validation Tuned)** (3 tasks: `T2.1`, `T2.2`, `T2.3`).
- **Milestone 3: Advanced Transformer Experiments on RTX 4060** (3 tasks: `T3.1`, `T3.2`, `T3.3`).
- **Milestone 4: Multi-Backend Inference Engine & Decoupled Integration Tests** (4 tasks: `T4.1`, `T4.2`, `T4.3`, `T4.4`).
- **Milestone 5: Web Application Development (React + FastAPI)** (4 tasks: `T5.1`, `T5.2`, `T5.3`, `T5.4`).
- **Milestone 6: Final Evaluation, Dockerization & Deployment** (4 tasks: `T6.1`, `T6.2`, `T6.3`, `T6.4`).

---

## 4. MA TRẬN RỦI RO & PHƯƠNG ÁN XỬ LÝ (Risk Register)

| Mã | Rủi ro tiềm ẩn | Mức độ | Kế hoạch dự phòng / Giảm thiểu |
| :--- | :--- | :--- | :--- |
| **R1** | Hugging Face Space chưa có token / yêu cầu thẻ tín dụng | Cao | Ghi nhận blocker tại `T0.3` (`docs/hf_eligibility_report.md`). Chuẩn bị sẵn phương án Render Free Docker hoặc local container; không tự ý đổi nền tảng. |
| **R2** | Mất cân bằng giữa 3 lớp (Lớp HATE chiếm tỉ lệ nhỏ nhất) | Cao | Áp dụng Class-weighted Cross-Entropy Loss; đo đạc Macro-F1 và Per-class F1 trên tập Validation; không dùng Accuracy đơn lẻ. |
| **R3** | Huấn luyện Transformer trên GPU RTX 4060 bị tràn VRAM (OOM) | Trung bình | Đánh dấu `max_length = 160`, `batch_size = 16` là PROVISIONAL; chạy profiling bước đầu; nếu VRAM vượt 7GB thì giảm `max_length = 128` hoặc chuyển sang Colab/Kaggle T4. |
| **R4** | Rò rỉ dữ liệu hoặc lạm dụng tập Test để tuning mô hình | Cao | Khóa hoàn toàn `test.parquet`; toàn bộ việc chọn mô hình diễn ra trên Validation set; chỉ unseal test set một lần duy nhất tại `T6.1`. |
| **R5** | Container Docker thiếu model weights khi khởi chạy | Cao | Tích hợp script `src/tools/download_artifacts.py` với 3 tầng dự phòng: Local -> Tải HF Hub theo pinned revision -> Fallback sang Baseline joblib đóng gói sẵn. |
