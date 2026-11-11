# Kế hoạch Thực thi Dự án (Implementation Plan)
## Vietnamese Toxic Comment Detection System (3-Class: React + FastAPI in Docker)

- **Mã kế hoạch**: `PLAN-TOXIC-COMMENT-2026-V2`
- **Phiên bản**: `2.0.0-APPROVED`
- **Trạng thái**: **APPROVED & READY FOR M0 EXECUTION**
- **Dựa trên**: `SPEC.md`, `docs/spec.md`, `docs/decisions.md`, `docs/dataset-requirements.md`, `docs/deployment-plan.md`, `docs/open-questions.md`
- **Ngày cập nhật**: 2026-11-11

---

## 1. PHÂN LOẠI THÔNG TIN DỰ ÁN (Status Ledger)

| Trạng thái | Hạng mục |
| :--- | :--- |
| **Confirmed** | - Mục tiêu: Nhận diện bình luận độc hại tiếng Việt, phân loại **3 lớp CLEAN / OFFENSIVE / HATE**.<br>- Dataset: Tải cả `ViCTSD` và `UIT-ViHSD` cho EDA; ưu tiên `UIT-ViHSD` làm tập chính.<br>- Mô hình: Đối sánh thực nghiệm khoa học giữa Traditional ML và Transformer trên **cùng một tập test duy nhất**.<br>- Phần cứng train Transformer: **Local GPU NVIDIA GeForce RTX 4060 8GB** (Google Colab/Kaggle là backup).<br>- Kiến trúc Web: **React (Frontend) + FastAPI (Backend)** đóng gói trong **Docker container**.<br>- Hosting mục tiêu: **Hugging Face Spaces** (Docker SDK, port 7860).<br>- Xác minh điều kiện tạo Docker Space trên Hugging Face ngay tại **Milestone 0**.<br>- Inference Engine: Hỗ trợ đa backend (`SklearnBackend`, `PyTorchBackend`, `ONNXBackend` tùy chọn dựa trên benchmark).<br>- Kiểm thử tích hợp cả hai nhóm mô hình trước khi deploy. |
| **Proposed** | - Multi-stage Dockerfile (Node Alpine build React -> Python slim serve FastAPI & static assets).<br>- Quản lý VRAM RTX 4060: Mixed Precision (`fp16`), Gradient Accumulation steps = 2, Batch Size = 16, VRAM budget < 7.0GB.<br>- Pydantic models cho schema API; Axios/Fetch cho React client. |
| **TBD (Chờ EDA & Train)** | - Cấu hình `max_length` cụ thể cho tokenizer sau khi đo phân vị 95% độ dài chuỗi ở M1.<br>- Trọng số lớp `class_weights` tính toán từ tần suất thực tế của 3 lớp trong tập train.<br>- Quyết định có kích hoạt ONNX hay giữ PyTorch dựa trên bài đo benchmark ở M4. |
| **Blocked** | - Training ML bị chặn cho đến khi hoàn thành M0 và M1 (Data Pipeline & EDA). |

---

## 2. TỔNG QUAN CÁC MILESTONE & DEPENDENCY GRAPH

Dự án được phân rã thành **7 Milestones** với **22 Tasks nguyên tử**, đảm bảo lát cắt dọc (Vertical Slicing), cho phép M3 (Transformer) chạy song song với M4/M5 (Web Scaffold với Baseline), sau đó tích hợp cả 2 mô hình qua bài test tích hợp trước khi deploy.

### 2.1. Đồ thị Phụ thuộc Điều chỉnh (Adjusted Dependency Graph)

```
[M0: Project Setup & HF Eligibility Check]
       │
       ▼
[M1: Data Pipeline & EDA (3-Class)]
       │
       ▼
[M2: Traditional ML Baseline (3-Class)] ───────+
       │                                       │
       ├────────────────────────────────┐      │ (Tích hợp MVP Web sớm)
       ▼ (Song song trên RTX 4060)      ▼      ▼
[M3: Transformer Fine-tuning]     [M4: Multi-backend Inference Engine]
       │                                │
       +───────────────┬────────────────┤
                       ▼                │
           [T4.2: Tích hợp & Kiểm thử   │
            cả 2 nhóm mô hình]          ▼
                       │          [M5: React + FastAPI Web App]
                       │                │
                       +────────────────┤
                                        ▼
                           [M6: Docker & Public HF Deployment]
```

### 2.2. Đường dẫn Găng (Critical Path)
`M0` (Setup & HF Check) ──→ `M1` (Data & EDA) ──→ `M2` (Baseline) ──→ `M4` (Inference Engine) ──→ `M5` (React + FastAPI) ──→ `M3` (Transformer on RTX 4060) ──→ `T4.2` (Integration Test) ──→ `M6` (Docker & HF Deploy)

---

## 3. TỔNG KẾT CÁC MILESTONES

- **Milestone 0: Project Setup, Quality Scaffold & HF Verification** (3 tasks: `T0.1`, `T0.2`, `T0.3`).
- **Milestone 1: Data Pipeline, EDA & 3-Class Preprocessing** (4 tasks: `T1.1`, `T1.2`, `T1.3`, `T1.4`).
- **Milestone 2: Traditional ML Baseline & 3-Class Evaluation** (3 tasks: `T2.1`, `T2.2`, `T2.3`).
- **Milestone 3: Advanced Transformer Experiments on RTX 4060** (3 tasks: `T3.1`, `T3.2`, `T3.3`).
- **Milestone 4: Multi-Backend Inference Engine & Integration Testing** (3 tasks: `T4.1`, `T4.2`, `T4.3`).
- **Milestone 5: Web Application Development (React + FastAPI)** (4 tasks: `T5.1`, `T5.2`, `T5.3`, `T5.4`).
- **Milestone 6: Dockerization, Hugging Face Deployment & Acceptance** (3 tasks: `T6.1`, `T6.2`, `T6.3`).

---

## 4. MA TRẬN RỦI RO & PHƯƠNG ÁN XỬ LÝ (Risk Register)

| Mã | Rủi ro tiềm ẩn | Mức độ | Kế hoạch dự phòng / Giảm thiểu |
| :--- | :--- | :--- | :--- |
| **R1** | Hugging Face yêu cầu thẻ tín dụng hoặc chặn Docker Space miễn phí | Cao | **Xác minh ngay tại T0.3**. Nếu bị chặn, ghi nhận blocker và đề xuất phương án Render Free Docker hoặc local container, không tự ý đổi nền tảng. |
| **R2** | Mất cân bằng giữa 3 lớp (Lớp HATE thường chiếm tỉ lệ nhỏ nhất) | Cao | Áp dụng Class-weighted Cross-Entropy Loss; đo đạc Macro-F1 và Per-class F1 cho từng lớp; không dùng Accuracy đơn lẻ. |
| **R3** | Huấn luyện Transformer trên GPU RTX 4060 bị tràn bộ nhớ VRAM (OOM) | Trung bình | Sử dụng Mixed Precision (`fp16`), giới hạn `max_length = 128-160`, batch size 16 kèm gradient accumulation = 2; nếu vẫn OOM thì chuyển sang Colab/Kaggle T4. |
| **R4** | Sai lệch nhãn hoặc rò rỉ dữ liệu giữa Traditional ML và Transformer | Cao | Sử dụng chung 1 tập test chuẩn `data/processed/test.parquet` cho cả hai nhóm mô hình; test case `test_leakage.py` tự động chặn nếu có overlap. |
| **R5** | Mô hình nạp vào container tốn quá nhiều RAM gây chậm cold start | Trung bình | Thiết kế đa backend: có cơ chế fallback sang `SklearnBackend` (<15MB) nếu PyTorch gặp sự cố; chạy benchmark ONNX ở `T4.3` để tối ưu kích thước. |

---

## 5. ĐỊNH NGHĨA HOÀN THÀNH (Definition of Done - DoD)
Dự án được nghiệm thu khi:
1. 100% các bài test tự động (`pytest`) vượt qua, bao gồm unit tests, integration tests đa backend và API tests.
2. Cả hai nhóm mô hình được huấn luyện và đối sánh trên cùng một tập test 3 lớp chuẩn, có bảng so sánh số liệu trung thực và phân tích lỗi.
3. Ứng dụng Web với React frontend và FastAPI backend hoạt động trơn tru trong Docker container.
4. Triển khai công khai thành công trên Hugging Face Spaces (hoặc phương án dự phòng đã được phê duyệt).
