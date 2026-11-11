# ĐẶC TẢ HỆ THỐNG: NHẬN DẠNG BÌNH LUẬN ĐỘC HẠI TIẾNG VIỆT
## (Vietnamese Toxic Comment Detection System - Specification)

- **Mã tài liệu**: `SPEC-TOXIC-COMMENT-2026`
- **Phiên bản**: `2.0.0-APPROVED`
- **Trạng thái**: `APPROVED / IMPLEMENTATION READY`
- **Tác giả**: Senior ML Engineer / NLP Researcher / Software Architect
- **Ngày cập nhật**: 2026-11-11
- **Phân loại**: Đồ án tốt nghiệp / Đồ án cuối kỳ môn Xử lý Ngôn ngữ Tự nhiên & Ứng dụng Web

---

## 1. MỤC TIÊU & PHẠM VI DỰ ÁN (Objective & Scope)

### 1.1. Bối cảnh & Mục tiêu
- **Mục tiêu**: Xây dựng hệ thống Machine Learning hoàn chỉnh từ khâu nghiên cứu dữ liệu tiếng Việt, đối sánh khoa học giữa mô hình truyền thống (Traditional ML) và mô hình học sâu Transformer, đóng gói inference engine đa backend và triển khai ứng dụng Web tương tác hiện đại (React frontend + FastAPI backend trong Docker container) công khai trên Hugging Face Spaces.
- **Đối tượng người dùng**:
  1. *Giảng viên & Hội đồng chấm điểm*: Cần kiểm chứng phương pháp khoa học, số liệu thực nghiệm trung thực trên cùng một tập test chuẩn, độ trễ và khả năng demo trực tiếp.
  2. *Người dùng & Kiểm duyệt viên (Content Moderators)*: Cần giao diện web mượt mà, phản hồi nhanh để kiểm tra câu đơn lẻ hoặc tải file CSV hàng loạt.

### 1.2. Phân loại Trạng thái Thông tin
- **`[Confirmed]`**: Yêu cầu đã được phê duyệt:
  - Dataset: Tải cả UIT-ViCTSD và UIT-ViHSD cho EDA; ưu tiên UIT-ViHSD làm tập chính.
  - Nhãn phân loại: **3 lớp CLEAN (0) / OFFENSIVE (1) / HATE (2)**. Không tự gộp về Binary.
  - Mô hình: Triển khai cả Traditional ML (TF-IDF + Linear Classifier) và Transformer (PhoBERT/ViSoBERT).
  - Web & Deploy: **Docker container với React + FastAPI**, triển khai trên **Hugging Face Spaces**.
  - Phần cứng huấn luyện: **Local GPU NVIDIA RTX 4060 8GB** (Google Colab/Kaggle là backup).
  - Inference: Hỗ trợ đa backend (`SklearnBackend`, `PyTorchBackend`, `ONNXBackend` tùy chọn).
  - Điều kiện tiên quyết: Xác minh tính khả thi của Hugging Face Docker Space ngay tại M0.
- **`[Proposed]`**: Các thiết kế kỹ thuật phục vụ triển khai (Vite cho React, Pydantic cho FastAPI schemas, Multi-stage Dockerfile).
- **`[TBD]`**: Các siêu tham số tối ưu (learning rate cụ thể, epochs, class weights chính xác sau khi chạy EDA).

---

## 2. BẢN ĐỒ KHẢ NĂNG & THỨ TỰ THỰC HIỆN (Capability Map)

| Module ID | Tên Khả năng | Trách nhiệm chính | Phụ thuộc |
| :--- | :--- | :--- | :--- |
| `mod-setup` | **Project Scaffold & HF Verification** | Khởi tạo cấu trúc dự án, dependencies, khung test tự động, kiểm tra điều kiện Hugging Face Docker Space. | — |
| `mod-data` | **Data Pipeline & EDA** | Tải raw data (`ViCTSD`, `UIT-ViHSD`), EDA, tiền xử lý Unicode/teencode, chuẩn hóa 3-class, chia split không rò rỉ. | `mod-setup` |
| `mod-trad-ml` | **Traditional ML Baseline** | Huấn luyện TF-IDF + Logistic Regression / LinearSVC trên 3 lớp, đánh giá trên Test set. | `mod-data` |
| `mod-trans-ml`| **Transformer Experiments** | Fine-tuning PhoBERT/ViSoBERT trên GPU RTX 4060 (mixed precision fp16), đánh giá trên cùng Test set. | `mod-data` |
| `mod-engine` | **Inference Engine & Integration** | Xây dựng `BasePredictor` đa backend, benchmark PyTorch vs ONNX, kiểm thử tích hợp cả 2 nhóm mô hình. | `mod-trad-ml` & `mod-trans-ml` |
| `mod-web` | **Web App (React + FastAPI)** | Phát triển REST API (FastAPI) và giao diện React (Single comment, Batch CSV, Model Card). | `mod-engine` |
| `mod-deploy` | **Docker & HF Deployment** | Multi-stage Docker build, kiểm thử container local, deploy lên Hugging Face Space và smoke test. | `mod-web` |

**Thứ tự thực hiện (Execution Order)**:  
`mod-setup` ──→ `mod-data` ──→ `mod-trad-ml` ──→ `mod-engine` (với Baseline) ──→ `mod-web` ──→ `mod-trans-ml` (song song) ──→ Tích hợp Transformer & Benchmark ──→ `mod-deploy`

---

## 3. THIẾT KẾ USE CASES CHO ỨNG DỤNG WEB

### UC-01: Phân tích Bình luận Đơn lẻ (Single Comment Analysis)
- **Giao diện**: Ô nhập văn bản tiếng Việt (tối đa 500 ký tự), nút "Phân tích".
- **Kết quả trả về**:
  - Nhãn dự đoán: **CLEAN (0)** [Xanh lá], **OFFENSIVE (1)** [Vàng cam], hoặc **HATE (2)** [Đỏ đậm].
  - Bảng/Thanh phân bố xác suất của cả 3 lớp: `P(Clean)`, `P(Offensive)`, `P(Hate)`.
  - Chỉ số thời gian phản hồi (Latency ms) và mô hình đang phục vụ (Model Identifier).

### UC-02: Phân tích Bình luận Hàng loạt (Batch CSV Analysis)
- **Giao diện**: Upload file `.csv` (giới hạn tối đa 500 dòng), chọn cột văn bản, nút "Quét tệp".
- **Kết quả trả về**:
  - Bảng dữ liệu hiển thị kèm các cột `predicted_label`, `confidence`.
  - Biểu đồ phân bố tỉ lệ 3 nhãn trong tệp.
  - Nút bấm tải về file CSV hoàn chỉnh.

### UC-03: Trang Thông tin Mô hình & So sánh Khoa học (Model Card)
- **Giao diện**: Trình bày kiến trúc hệ thống, bảng so sánh đối đầu giữa Traditional ML và Transformer trên cùng tập test chuẩn, Confusion Matrix 3 lớp, và phân tích các trường hợp khó/lỗi.

---

## 4. THIẾT KẾ KIẾN TRÚC ML & INFERENCE ENGINE

### 4.1. Không gian Nhãn Chuẩn (3-Class Taxonomy)
- `0: CLEAN`: Bình luận bình thường, văn minh hoặc tranh luận mang tính xây dựng.
- `1: OFFENSIVE`: Bình luận có từ ngữ thô tục, chửi thề, khiêu khích nhưng không nhắm vào việc thù ghét/hủy diệt một nhóm xã hội.
- `2: HATE`: Bình luận kích động thù hận, phân biệt đối xử (vùng miền, giới tính, tôn giáo, chủng tộc) hoặc xúc phạm nhân phẩm nghiêm trọng.

### 4.2. Thiết kế Đa Backend cho Inference (`BasePredictor`)
Interface trừu tượng cho phép cắm rút bất kỳ backend nào mà tầng Web không cần thay đổi code:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

@dataclass(frozen=True)
class PredictionResult:
    text: str
    cleaned_text: str
    label_id: int          # 0, 1, 2
    label_name: str        # CLEAN, OFFENSIVE, HATE
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    model_backend: str     # sklearn | pytorch | onnx

class BasePredictor(ABC):
    @abstractmethod
    def predict(self, text: str) -> PredictionResult:
        pass

    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[PredictionResult]:
        pass
```

- **`SklearnBackend`**: Nạp model `.joblib`, chạy TF-IDF + Classifier.
- **`PyTorchBackend`**: Nạp model weights PyTorch FP32/FP16, chạy tokenizer và forward pass.
- **`ONNXBackend`** (Tùy chọn): Nạp `.onnx` với ONNX Runtime, chỉ kích hoạt khi thực nghiệm chứng minh cải thiện tốc độ và bộ nhớ trên CPU.

---

## 5. CẤU TRÚC THƯ MỤC DỰ ÁN (Project Structure)

```
Detecting Toxic Comment/
├── .github/
│   └── workflows/              # GitHub Actions (CI lint/test & HF deploy sync)
├── backend/                    # Mã nguồn FastAPI Backend
│   ├── __init__.py
│   ├── main.py                 # FastAPI application, CORS, static mount
│   ├── api/                    # REST API routers
│   │   ├── __init__.py
│   │   ├── routes_predict.py   # Endpoints /api/predict & /api/predict-batch
│   │   └── routes_meta.py      # Endpoints /api/health & /api/model-info
│   └── schemas.py              # Pydantic request/response models
├── frontend/                   # Ứng dụng React Frontend (Vite)
│   ├── package.json
│   ├── vite.config.js
│   ├── public/
│   └── src/
│       ├── App.jsx
│       ├── components/         # Header, SingleComment, BatchUpload, ModelCard
│       └── services/api.js     # Axios/Fetch API client
├── configs/                    # File cấu hình (YAML)
│   ├── baseline_config.yaml    # Config cho TF-IDF + LogisticRegression/SVC
│   └── transformer_config.yaml # Config cho PhoBERT/ViSoBERT (RTX 4060)
├── data/                       # Dữ liệu (ignored by git)
│   ├── raw/                    # Dữ liệu gốc Hugging Face
│   └── processed/              # Dữ liệu sạch (train/val/test parquet)
├── docs/                       # Tài liệu kỹ thuật
│   ├── spec.md                 # Bản đặc tả này
│   ├── decisions.md            # Các ADR đã phê duyệt
│   ├── dataset-requirements.md # Yêu cầu dataset và checklist EDA
│   ├── deployment-plan.md      # Kế hoạch Docker & HF Spaces
│   ├── open-questions.md       # Trạng thái câu hỏi mở
│   └── plans/                  # Implementation plan chi tiết
├── models/                     # Checkpoints & artifacts (ignored by git)
│   ├── baselines/              # Model sklearn .joblib
│   └── transformer_best/       # Checkpoints transformer PyTorch
├── notebooks/                  # Notebooks nghiên cứu & EDA
│   └── 01_dataset_eda.ipynb
├── src/                        # ML Core Package
│   ├── __init__.py
│   ├── data/                   # download, preprocess, split
│   ├── models/                 # baseline train, transformer train, export
│   ├── evaluation/             # metrics 3 lớp, error analysis, reporting
│   └── inference/              # BasePredictor, SklearnBackend, PyTorchBackend
├── tests/                      # Toàn bộ automated tests
│   ├── test_smoke.py           # Scaffold smoke test
│   ├── test_preprocessing.py   # Test làm sạch tiếng Việt
│   ├── test_leakage.py         # Test chống rò rỉ train/test
│   ├── test_inference_backends.py # Test tích hợp đa backend
│   └── test_api.py             # Test FastAPI endpoints với TestClient
├── Dockerfile                  # Multi-stage Dockerfile (React build + FastAPI)
├── docker-compose.yml          # Local container testing
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development, training & testing dependencies
├── README.md                   # Hướng dẫn dự án & nghiệm thu
└── SPEC.md                     # File spec tham chiếu gốc
```

---

## 6. DANH MỤC LỆNH VẬN HÀNH (Commands Specification)

```bash
# 1. Khởi tạo môi trường ảo Python 3.10 / 3.12 (trên Windows)
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt

# 2. Cài đặt môi trường Frontend React
cd frontend && npm install && cd ..

# 3. Kiểm tra chất lượng mã nguồn & Kiểm thử
pytest tests/ -v --cov=src/ --cov=backend/
flake8 src/ backend/ tests/ --max-line-length=100
cd frontend && npm run lint && cd ..

# 4. Pipeline Dữ liệu
python -m src.data.download --dataset all --output data/raw/
python -m src.data.preprocessor --input data/raw/ --output data/processed/

# 5. Huấn luyện Mô hình
# Traditional ML:
python -m src.models.baseline --config configs/baseline_config.yaml
# Transformer trên GPU local RTX 4060:
python -m src.models.transformer --config configs/transformer_config.yaml

# 6. Đánh giá Đối kháng trên Cùng Test Set
python -m src.evaluation.compare_models --test-data data/processed/test.parquet

# 7. Khởi chạy Local Development (2 processes)
# Backend:
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
# Frontend:
cd frontend && npm run dev

# 8. Build & Chạy thử nghiệm Docker Container
docker build -t vietnamese-toxic-detector:latest .
docker run -p 7860:7860 vietnamese-toxic-detector:latest
```

---

## 7. CHIẾN LƯỢC KIỂM THỬ (Testing Strategy)

1. **Unit Testing**:
   - `test_smoke.py`: Kiểm tra môi trường hoạt động.
   - `test_preprocessing.py`: Kiểm tra chuẩn hóa Unicode NFC, xử lý teencode, loại bỏ ký tự lạ.
   - `test_leakage.py`: Khẳng định `intersection(train_texts, test_texts) == 0`.
2. **Integration Testing (`test_inference_backends.py`)**:
   - Khởi tạo và suy luận độc lập qua `SklearnBackend` và `PyTorchBackend`.
   - Đảm bảo cả hai backend đều trả về dataclass `PredictionResult` đúng schema 3 lớp (`CLEAN`, `OFFENSIVE`, `HATE`).
3. **API Testing (`test_api.py`)**:
   - Dùng `fastapi.testclient.TestClient` kiểm tra mã phản hồi HTTP 200, schema JSON đầu ra, xử lý input rỗng và input vượt độ dài.
4. **Production Smoke Testing**:
   - Kiểm tra container Docker chạy trên port 7860 (chuẩn cổng Hugging Face Spaces), phục vụ đúng giao diện React và trả lời API.

---

## 8. RANH GIỚI VẬN HÀNH (Boundaries)

- **Always Do**:
  - Giữ nguyên nhãn 3 lớp `CLEAN` / `OFFENSIVE` / `HATE` cho bài toán chính.
  - Sử dụng chung 1 tập test chuẩn duy nhất cho cả mô hình truyền thống và Transformer.
  - Kiểm tra tính hợp lệ của Hugging Face Docker Space ngay tại M0.
  - Tách biệt hoàn toàn tầng Inference và tầng Web API.
- **Ask First**:
  - Thay đổi kiến trúc container hoặc đổi nền tảng hosting nếu Hugging Face Docker Space phát sinh rào cản.
  - Bổ sung thêm các trọng số mô hình hoặc dataset ngoài phạm vi đã duyệt.
- **Never Do**:
  - Không tự động gộp nhãn 3 lớp thành Binary.
  - Không commit dataset raw hoặc checkpoint dung lượng lớn lên GitHub.
  - Không sử dụng tập test để tuning siêu tham số hoặc early stopping.
  - Không bắt buộc ONNX nếu benchmark chưa chứng minh được hiệu quả.

---

## 9. TIÊU CHÍ NGHIỆM THU DỰ ÁN (Acceptance Criteria)

- [ ] **AC-ML-01**: Cả hai nhóm mô hình (Traditional ML và Transformer tiếng Việt) được huấn luyện thành công và đánh giá trên **cùng một tập test chuẩn 3 lớp**.
- [ ] **AC-ML-02**: Có bảng so sánh Macro-F1, Per-class Precision/Recall (Clean, Offensive, Hate), Confusion Matrix và Latency.
- [ ] **AC-ML-03**: Mô hình Transformer đạt chỉ số Macro-F1 trên tập test vượt trội so với Baseline.
- [ ] **AC-ML-04**: Đảm bảo 100% không rò rỉ dữ liệu giữa Train và Test (được chứng minh bằng test tự động).
- [ ] **AC-WEB-01**: Ứng dụng có backend FastAPI và frontend React hoạt động mượt mà, hỗ trợ phân tích đơn lẻ và tải file CSV hàng loạt (≤ 500 dòng).
- [ ] **AC-WEB-02**: Có trang Model Card hiển thị báo cáo khoa học trực tiếp.
- [ ] **AC-WEB-03**: Ứng dụng được đóng gói thành Docker container và triển khai công khai thành công trên Hugging Face Spaces (hoặc phương án dự phòng nếu ghi nhận blocker).
