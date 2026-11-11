# ĐẶC TẢ HỆ THỐNG: NHẬN DẠNG BÌNH LUẬN ĐỘC HẠI TIẾNG VIỆT
## (Vietnamese Toxic Comment Detection System - Specification)

- **Mã tài liệu**: `SPEC-TOXIC-COMMENT-2026`
- **Phiên bản**: `1.0.0-DRAFT`
- **Trạng thái**: `PROPOSED / PENDING APPROVAL`
- **Tác giả**: Senior ML Engineer / NLP Researcher / Software Architect
- **Ngày lập**: 2026-11-11
- **Phân loại**: Đồ án tốt nghiệp / Đồ án cuối kỳ môn Xử lý Ngôn ngữ Tự nhiên & Ứng dụng Web

---

## 1. MỤC TIÊU & PHẠM VI DỰ ÁN (Objective & Scope)

### 1.1. Bối cảnh & Mục tiêu (What & Why)
- **Bối cảnh**: Trên không gian mạng xã hội tiếng Việt, các bình luận mang tính xúc phạm, thù ghét, tục tĩu (toxic/hate speech) gây ảnh hưởng nghiêm trọng đến trải nghiệm người dùng và an toàn văn hóa số.
- **Mục tiêu**: Xây dựng một giải pháp Machine Learning hoàn chỉnh từ khâu nghiên cứu dữ liệu, huấn luyện mô hình đối sánh khoa học đến đóng gói thành ứng dụng Web tương tác công khai, phục vụ mục đích học thuật và thử nghiệm thực tế.
- **Đối tượng người dùng mục tiêu (Target Users)**:
  1. *Giảng viên & Hội đồng đánh giá*: Cần kiểm tra tính đúng đắn khoa học, số liệu thực nghiệm, độ trễ và khả năng demo trực tiếp.
  2. *Người dùng cuối / Người kiểm duyệt nội dung (Content Moderators)*: Cần giao diện trực quan để kiểm tra nhanh mức độ độc hại của văn bản đơn lẻ hoặc danh sách bình luận.

### 1.2. Phân biệt Trạng thái Thông tin (Information Classification)
Toàn bộ thông tin trong tài liệu này được phân định nghiêm ngặt:
- **`[Verified]`**: Đã kiểm chứng từ mã nguồn, tệp cấu hình hoặc dữ liệu hiện hữu trong repo (ví dụ: liên kết dataset tại `link dataset.txt`).
- **`[Proposed]`**: Đề xuất thiết kế kỹ thuật của Kỹ sư, dựa trên best practices của NLP tiếng Việt và công nghệ web.
- **`[Assumption]`**: Giả định logic cần được xác thực lại khi có dữ liệu.
- **`[TBD]`**: Vấn đề chưa đủ căn cứ quyết định, bắt buộc chờ kết quả thực nghiệm.

### 1.3. Phạm vi MVP (Minimum Viable Product Scope)
- **Trong phạm vi (In-Scope)**:
  - Ingestion và chuẩn hóa 1 hoặc cả 2 bộ dữ liệu tiếng Việt từ Hugging Face (`tarudesu/ViCTSD` hoặc `ura-hcmut/UIT-ViHSD`).
  - Xây dựng Baseline cổ điển: TF-IDF + Logistic Regression / LinearSVC.
  - Xây dựng Mô hình Nâng cao: Fine-tuning Pretrained Transformer tiếng Việt (PhoBERT-base-v2 hoặc ViSoBERT).
  - Báo cáo đối sánh khoa học (Macro-F1, Precision, Recall, Confusion Matrix, Latency, Model Size).
  - Ứng dụng Web demo hoàn chỉnh bằng Streamlit:
    - Chức năng 1: Phân tích 1 bình luận (dự đoán nhãn Toxic/Non-toxic, hiển thị độ tin cậy Confidence Score, highlight từ khóa).
    - Chức năng 2: Phân tích hàng loạt (Upload file CSV/TXT, trả kết quả thống kê và cho phép tải CSV đã gán nhãn).
    - Chức năng 3: Giới thiệu mô hình, kiến trúc pipeline và bảng so sánh kết quả thực nghiệm.
  - Triển khai công khai (Public Deployment) miễn phí trên Hugging Face Spaces.
- **Ngoài phạm vi MVP (Out-of-Scope / Deferred)**:
  - Chưa hỗ trợ đa ngôn ngữ (chỉ tập trung vào tiếng Việt).
  - Chưa hỗ trợ xử lý luồng thời gian thực từ API mạng xã hội (Streaming Real-time scraping).
  - Chưa triển khai hệ thống xác thực người dùng (Auth/Login/RBAC).
  - Chưa xây dựng database lưu trữ lịch sử bình luận lâu dài (tránh vi phạm quyền riêng tư và chi phí hạ tầng).

---

## 2. BẢN ĐỒ KHẢ NĂNG & THỨ TỰ THỰC HIỆN (Capability Map)

Theo nguyên tắc phân rã độc lập và kiểm thử được:

| Module ID | Tên Khả năng (Capability) | Trách nhiệm chính | Phụ thuộc (Depends on) |
| :--- | :--- | :--- | :--- |
| `mod-data` | **Data Pipeline & EDA** | Tải raw data, EDA, chuẩn hóa Unicode/teencode, chia train/val/test, chống data leakage. | — |
| `mod-train` | **Model Training & Evaluation** | Huấn luyện Baseline & Advanced Transformers, log metrics, lưu evaluation report. | `mod-data` |
| `mod-export` | **Model Packaging & Optimization** | Export model sang ONNX/PyTorch tối ưu, kiểm tra latency và kích thước artifact. | `mod-train` |
| `mod-ui` | **Web Application (Streamlit)** | Xây dựng UI cho single input, batch CSV input, trang giới thiệu mô hình. | `mod-export` |
| `mod-deploy` | **Public Cloud Deployment** | Cấu hình Hugging Face Spaces, Dockerfile/requirements, smoke test live. | `mod-ui` |

**Thứ tự triển khai (Build Order)**:  
`mod-data` ──→ `mod-train` ──→ `mod-export` ──→ `mod-ui` ──→ `mod-deploy`

---

## 3. THIẾT KẾ USE CASES CHO ỨNG DỤNG WEB

### UC-01: Phân tích Bình luận Đơn lẻ (Single Comment Analysis)
- **Mục tiêu**: Người dùng nhập trực tiếp 1 đoạn văn bản tiếng Việt để xem kết quả đánh giá.
- **Luồng hoạt động**:
  1. Người dùng nhập văn bản vào Text Area (tối đa 500 ký tự).
  2. Bấm nút "Phân tích nội dung" (hoặc tự động cập nhật khi gõ).
  3. Hệ thống tiền xử lý chuỗi: chuẩn hóa khoảng trắng, Unicode dựng sẵn, xử lý teencode.
  4. Mô hình suy luận xác suất.
  5. Hiển thị kết quả:
     - Badge nhãn: **Độc hại (Toxic)** màu đỏ cảnh báo hoặc **Lành mạnh (Clean)** màu xanh lá.
     - Thanh đo điểm tin cậy (Confidence Gauge/Bar): ví dụ `Toxic: 92.4%`, `Clean: 7.6%`.
     - Giải thích sơ bộ (highlight các cụm từ nghi vấn nếu dùng mô hình diễn giải được).

### UC-02: Phân tích Bình luận Hàng loạt (Batch Comment Analysis)
- **Mục tiêu**: Cho phép tải lên file CSV hoặc TXT chứa danh sách nhiều bình luận.
- **Luồng hoạt động**:
  1. Người dùng kéo-thả file `.csv` có cột chứa văn bản (tối đa 500 dòng cho bản demo miễn phí để tránh timeout).
  2. Chọn tên cột văn bản cần phân tích.
  3. Bấm "Bắt đầu phân tích".
  4. Hệ thống chạy batch inference kèm thanh tiến trình (Progress Bar).
  5. Trả về bảng dữ liệu kèm cột `predicted_label` và `confidence_score`.
  6. Biểu đồ tròn (Pie Chart) thể hiện tỉ lệ Toxic vs Clean trong tệp.
  7. Nút "Tải kết quả (Download CSV)".

### UC-03: Trang Giới thiệu & Báo cáo Thực nghiệm (Model Card & Experiment Reports)
- **Mục tiêu**: Cung cấp tài liệu khoa học trực tiếp ngay trên web cho hội đồng chấm điểm.
- **Nội dung hiển thị**:
  - Tóm tắt đề tài, các nhà nghiên cứu, liên kết mã nguồn GitHub.
  - Bảng so sánh hiệu năng các mô hình (Baseline vs Advanced).
  - Confusion Matrix tương tác và các chỉ số Macro-F1, Precision, Recall.
  - Giới hạn của mô hình (Known limitations) và hướng dẫn sử dụng có trách nhiệm.

---

## 4. THIẾT KẾ KIẾN TRÚC ML PIPELINE

```
+-----------------------------------------------------------------------------------+
|                                ML DATA PIPELINE                                   |
|                                                                                   |
|  [Hugging Face Hub]                                                               |
|  - tarudesu/ViCTSD                                                                |
|  - ura-hcmut/UIT-ViHSD                                                            |
|          |                                                                        |
|          v                                                                        |
|  [1. Ingestion] ──────> Lưu raw vào data/raw/                                     |
|          |                                                                        |
|          v                                                                        |
|  [2. Preprocessing] ──> Unicode Normalization (NFC)                               |
|                         Xử lý teencode & icon (regex + dict lookup)               |
|                         Tách từ tiếng Việt (nếu dùng PhoBERT)                     |
|          |                                                                        |
|          v                                                                        |
|  [3. Validation] ─────> Loại bỏ trùng lặp, giải quyết mâu thuẫn nhãn              |
|                         Kiểm tra rò rỉ dữ liệu giữa các tập                       |
|          |                                                                        |
|          v                                                                        |
|  [4. Splitting] ──────> Stratified Split (Train 70% - Val 15% - Test 15%)         |
|                         hoặc sử dụng Official Benchmark Split                     |
|          |                                                                        |
|          v                                                                        |
|  [Processed Data] ────> Lưu data/processed/ (train.parquet, val, test)            |
+-----------------------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------------------+
|                                TRAINING & EVALUATION                              |
|                                                                                   |
|      +--------------------------------+   +---------------------------------+     |
|      |        BASELINE PIPELINE       |   |       TRANSFORMER PIPELINE      |     |
|      |  - TF-IDF N-grams (1, 3)       |   |  - Pretrained PhoBERT / ViSoBERT|     |
|      |  - Logistic Regression / SVM   |   |  - Fine-tuning with AdamW       |     |
|      |  - Hyperparameter Grid Search  |   |  - Class-weighted Loss          |     |
|      +----------------+---------------+   +----------------+----------------+     |
|                       |                                    |                      |
|                       +------------------+-----------------+                      |
|                                          |                                        |
|                                          v                                        |
|                       [Evaluation Protocol & Benchmarking]                        |
|                       - Macro-F1, Precision, Recall, Accuracy                     |
|                       - Confusion Matrix, Error Analysis                          |
|                       - Latency & Model Size Profiling                            |
|                                          |                                        |
|                                          v                                        |
|                       [Selection of Best Production Model]                        |
+-----------------------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------------------+
|                             PACKAGING & INFERENCE ENGINE                          |
|                                                                                   |
|  - Export PyTorch weights sang ONNX format (src/models/export_onnx.py)             |
|  - Dynamic INT8 Quantization (giảm dung lượng < 150MB, tăng tốc CPU)              |
|  - Đóng gói inference engine độc lập (src/inference/predictor.py)                 |
|  - Tích hợp vào Web UI (Streamlit - app/main.py)                                  |
+-----------------------------------------------------------------------------------+
```

---

## 5. THIẾT KẾ ĐỐI SÁNH MÔ HÌNH (Baseline vs Advanced)

### 5.1. Nhóm Baseline (Chuẩn cơ sở)
- **Mô hình**: `TF-IDF (Word n-gram 1-2 & Char n-gram 2-4) + Logistic Regression / LinearSVC`.
- **Mục đích**:
  - Thiết lập cận dưới hiệu năng (Performance Lower Bound).
  - Đảm bảo có một mô hình siêu nhẹ (< 15MB, latency < 10ms) có thể chạy mọi lúc mọi nơi kể cả khi hạ tầng gặp sự cố.
  - Phân tích feature weights để khám phá các từ ngữ mang tính chỉ thị độc hại mạnh nhất trong dataset.

### 5.2. Nhóm Advanced Models (Học sâu Tiền huấn luyện)
- **Mô hình 1 [Proposed]**: `vinai/phobert-base-v2`
  - Mô hình nền tảng ngôn ngữ tiếng Việt phổ biến nhất, huấn luyện trên 20GB văn bản báo chí và bách khoa.
  - Cần thư viện tách từ (`pyvi` hoặc `rdrsegmenter`).
- **Mô hình 2 [Proposed]**: `uitnlp/visobert`
  - Mô hình tiền huấn luyện chuyên biệt trên dữ liệu mạng xã hội tiếng Việt (Social Media Domain). Rất nhạy với teencode, slang và ngữ cảnh bình luận Facebook/YouTube.
- **Tiêu chuẩn lựa chọn mô hình triển khai**:
  - Không chọn theo cảm tính; chọn mô hình có chỉ số **Macro-F1 trên tập Test** cao nhất, đồng thời thỏa mãn:
    1. Kích thước artifact sau đóng gói ≤ 500MB (hoặc ≤ 150MB nếu convert ONNX).
    2. Độ trễ suy luận trên 1 vCPU ≤ 300ms/câu.

---

## 6. GIAO THỨC ĐÁNH GIÁ (Evaluation Protocol, Imbalance & Leakage)

### 6.1. Xử lý Mất cân bằng Lớp (Class Imbalance Strategy)
- Văn bản bình luận độc hại trong thực tế thường là lớp thiểu số (Minority Class), chiếm tỷ lệ nhỏ (<15-20%).
- **Quy tắc tuyệt đối**: **KHÔNG** dùng Accuracy làm thước đo quyết định vì mô hình đoán toàn bộ là "Clean" vẫn có thể đạt Accuracy 85-90%.
- **Chỉ số Đánh giá Chính (Primary Metric)**: **Macro-averaged F1-Score** (trọng số bình đẳng giữa lớp Toxic và Non-toxic) cùng với **Precision / Recall của riêng lớp Toxic**.
- **Kỹ thuật Huấn luyện hỗ trợ**:
  - Thiết lập trọng số hàm mất mát (Class-weighted Cross-Entropy Loss):
    $$\text{weight}_c = \frac{N_{\text{total}}}{C \times N_c}$$
  - Thử nghiệm Focal Loss nếu tỉ lệ mất cân bằng vượt quá 1:10.

### 6.2. Kiểm soát và Ngăn chặn Rò rỉ Dữ liệu (Data Leakage Prevention)
- **Quy tắc Leakage 1 (Feature Extraction Leakage)**: TF-IDF vectorizer hoặc bất kỳ phương pháp tính thống kê vocab nào chỉ được `fit` trên tập **Train**. Tập Val và Test chỉ được `transform`.
- **Quy tắc Leakage 2 (Text Duplication Leakage)**: Thực hiện kiểm tra trùng lặp văn bản (exact match và fuzzy match) giữa Train và Test trước khi đưa vào pipeline huấn luyện. Nếu phát hiện mẫu trùng lặp, phải loại bỏ khỏi Test hoặc gộp nhóm.
- **Quy tắc Leakage 3 (Preprocessing Leakage)**: Bảng từ điển teencode/slang phải là bộ từ điển cố định độc lập, không được tự động học từ tập Test.

---

## 7. CẤU TRÚC THƯ MỤC DỰ ÁN (Project Structure)

```
Detecting Toxic Comment/
├── .github/
│   └── workflows/              # CI/CD pipelines (test, lint, deploy sync)
├── app/                        # Ứng dụng Web (Streamlit)
│   ├── main.py                 # File khởi chạy giao diện chính
│   ├── pages/                  # Các trang con (nếu dùng Streamlit multipage)
│   │   ├── 1_Single_Comment.py
│   │   ├── 2_Batch_Analysis.py
│   │   └── 3_Model_Card.py
│   └── components/             # Reusable UI widgets (charts, badges)
├── configs/                    # File cấu hình (YAML / JSON)
│   ├── baseline_config.yaml    # Hyperparameters cho TF-IDF + Classifier
│   └── transformer_config.yaml # Hyperparameters cho Transformer fine-tuning
├── data/                       # Quản lý dữ liệu (bị git ignore trừ README)
│   ├── raw/                    # Dữ liệu gốc tải về từ Hugging Face
│   └── processed/              # Dữ liệu sạch sau tiền xử lý và chia split
├── docs/                       # Tài liệu kỹ thuật
│   ├── spec.md                 # Bản đặc tả hệ thống này
│   ├── decisions.md            # Các quyết định thiết kế kiến trúc (ADR)
│   ├── dataset-requirements.md # Yêu cầu dataset và checklist EDA
│   ├── deployment-plan.md      # Kế hoạch hosting và ngân sách tài nguyên
│   └── open-questions.md       # Các vấn đề mở cần xác minh
├── models/                     # Lưu trữ weights và artifacts
│   ├── baselines/              # Model TF-IDF joblib
│   └── exported/               # Model ONNX / PyTorch đã tối ưu sẵn sàng deploy
├── notebooks/                  # Jupyter notebooks cho nghiên cứu & EDA
│   ├── 01_dataset_eda.ipynb
│   └── 02_error_analysis.ipynb
├── src/                        # Mã nguồn cốt lõi (Core Python Package)
│   ├── __init__.py
│   ├── data/                   # Module xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── download.py         # Script tải dữ liệu từ Hugging Face Hub
│   │   ├── preprocessor.py     # Chuẩn hóa Unicode, regex, teencode cleaner
│   │   └── splitter.py         # Chia tập train/val/test an toàn
│   ├── models/                 # Module mô hình hóa
│   │   ├── __init__.py
│   │   ├── baseline.py         # Huấn luyện TF-IDF + LogisticRegression/SVM
│   │   ├── transformer.py      # Fine-tuning PhoBERT/ViSoBERT với PyTorch
│   │   └── export_onnx.py      # Script chuyển đổi PyTorch -> ONNX & INT8
│   ├── evaluation/             # Đánh giá & Báo cáo
│   │   ├── __init__.py
│   │   ├── metrics.py          # Tính Macro-F1, Confusion Matrix, Latency
│   │   └── reporter.py         # Xuất bảng biểu và đồ thị markdown/json
│   └── inference/              # Inference Engine phục vụ Web
│       ├── __init__.py
│       └── predictor.py        # Class Predictor độc lập (Load ONNX/Torch)
├── tests/                      # Kiểm thử tự động (Unit & Integration tests)
│   ├── test_preprocessing.py   # Test làm sạch chuỗi, teencode, unicode
│   ├── test_inference.py       # Test engine dự đoán đúng định dạng output
│   └── test_leakage.py         # Test đảm bảo train/test không overlap
├── link dataset.txt            # [Verified] Chứa link nguồn dataset Hugging Face
├── requirements.txt            # Dependencies cho web & inference (tối giản)
├── requirements-dev.txt        # Dependencies cho training, lint, pytest
├── README.md                   # Hướng dẫn tổng quan dự án
└── SPEC.md                     # File spec tham chiếu gốc
```

---

## 8. DANH MỤC LỆNH VẬN HÀNH (Commands Specification)

Mọi thao tác phát triển, kiểm thử, huấn luyện và chạy app đều được định nghĩa thành các lệnh CLI rõ ràng:

```bash
# 1. Cài đặt môi trường phát triển (Development Setup)
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/macOS:
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt

# 2. Kiểm tra chất lượng mã nguồn (Lint & Formatting)
flake8 src/ app/ tests/ --max-line-length=100
black --check src/ app/ tests/

# 3. Chạy toàn bộ kiểm thử tự động (Automated Testing)
pytest tests/ -v --cov=src/

# 4. Tải và tiền xử lý dữ liệu (Data Pipeline)
python -m src.data.download --dataset uit-vihsd --output data/raw/
python -m src.data.preprocessor --input data/raw/ --output data/processed/

# 5. Huấn luyện mô hình Baseline (TF-IDF)
python -m src.models.baseline --config configs/baseline_config.yaml

# 6. Huấn luyện mô hình Nâng cao (Transformer)
python -m src.models.transformer --config configs/transformer_config.yaml

# 7. Đánh giá & Xuất báo cáo khoa học
python -m src.evaluation.reporter --model-dir models/exported/ --test-data data/processed/test.parquet

# 8. Export mô hình sang ONNX & Quantization
python -m src.models.export_onnx --input models/transformer_best/ --output models/exported/

# 9. Khởi chạy ứng dụng Web cục bộ (Local Web Development)
streamlit run app/main.py --server.port 8501
```

---

## 9. QUY CHUẨN MÃ NGUỒN (Code Style & Engineering Conventions)

- **Ngôn ngữ**: Python 3.10+
- **Chuẩn format**: PEP 8, Black (line length: 100 ký tự), Isort để gom import.
- **Type Annotations**: Bắt buộc gắn kiểu dữ liệu đầy đủ cho các hàm và phương thức.
- **Clean Architecture Pattern**: Tách biệt hoàn toàn tầng Web Presentation và tầng Inference Logic. UI chỉ gọi qua interface `Predictor.predict(text: str) -> PredictionResult`.

### Ví dụ Mã Nguồn Mẫu (Reference Implementation Style)

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass(frozen=True)
class PredictionResult:
    """Immutable data container cho kết quả dự đoán bình luận."""
    text: str
    cleaned_text: str
    is_toxic: bool
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float

class BasePredictor:
    """Abstract interface cho tất cả inference engines (Baseline hoặc ONNX)."""
    
    def predict(self, text: str) -> PredictionResult:
        raise NotImplementedError("Subclasses must implement predict()")

    def predict_batch(self, texts: List[str]) -> List[PredictionResult]:
        return [self.predict(t) for t in texts]
```

---

## 10. CHIẾN LƯỢC KIỂM THỬ (Testing Strategy)

| Tầng kiểm thử | Đối tượng kiểm tra | Vị trí file test | Tiêu chí vượt qua |
| :--- | :--- | :--- | :--- |
| **Unit Test: Preprocessing** | Xử lý Unicode dựng sẵn (NFC), xóa ký tự rác, map teencode cơ bản, chuẩn hóa khoảng trắng. | `tests/test_preprocessing.py` | Chuỗi đầu ra không lỗi encoding, không vỡ từ tố, chuẩn hóa đúng kỳ vọng 100%. |
| **Unit Test: Inference Engine** | Khởi tạo mô hình, suy luận với input bình thường, input chuỗi rỗng `""`, input chuỗi siêu dài (>1000 ký tự). | `tests/test_inference.py` | Không phát sinh ngoại lệ không bắt được (Unhandled Exception); trả về đối tượng `PredictionResult` đúng schema; thời gian xử lý < 500ms. |
| **Integration Test: Data Leakage** | Kiểm tra giao tập giữa tập Train và tập Test. | `tests/test_leakage.py` | Số lượng mẫu trùng lặp giữa Train và Test bằng 0 (`intersection == 0`). |
| **Smoke Test: Web UI** | Khởi động Streamlit process, kiểm tra HTTP status 200 tại port 8501. | CI pipeline step | Ứng dụng load không bị OOM, render trang chủ thành công. |

---

## 11. RANH GIỚI VẬN HÀNH (Boundaries: Always / Ask First / Never)

### Luôn luôn làm (Always Do)
- [x] Luôn kiểm tra phân bố nhãn (Class Distribution) trước khi chọn hàm Loss và Metric.
- [x] Luôn chuẩn hóa Unicode sang dạng NFC (`unicodedata.normalize('NFC', text)`) cho toàn bộ dữ liệu tiếng Việt.
- [x] Luôn đóng gói hàm suy luận độc lập với UI để kiểm thử tự động được bằng `pytest`.
- [x] Luôn cắt ngắn (truncate) văn bản đầu vào ở mức giới hạn an toàn khi inference trên Web.
- [x] Luôn ghi rõ nguồn trích dẫn học thuật của dataset (`ViCTSD`, `UIT-ViHSD`) và các mô hình nền tảng (`PhoBERT`, `ViSoBERT`).

### Hỏi trước khi làm (Ask First)
- [?] Thay đổi bài toán phân loại từ Binary (Toxic vs Clean) sang Multi-class (Clean, Offensive, Hate) hoặc Multi-label.
- [?] Bổ sung thêm thư viện hệ thống ngoài Python (như Java cho RDRsegmenter hoặc các C++ binaries).
- [?] Thêm dependency có dung lượng lớn vào file `requirements.txt` của app deploy.
- [?] Tự ý thay đổi tỉ lệ phân chia train/val/test nếu đã có split chuẩn từ benchmark của tác giả.

### Tuyệt đối không làm (Never Do)
- [!] **TUYỆT ĐỐI KHÔNG** commit các file dataset raw nặng hàng trăm MB hoặc file trọng số mô hình PyTorch `.bin` / `.pth` lớn lên GitHub.
- [!] **TUYỆT ĐỐI KHÔNG** dùng Accuracy làm tiêu chí đánh giá mô hình duy nhất khi dữ liệu mất cân bằng.
- [!] **TUYỆT ĐỐI KHÔNG** đưa logic huấn luyện (training loop) hoặc tải dataset raw vào trong code chạy Web Streamlit (`app/main.py`).
- [!] **TUYỆT ĐỐI KHÔNG** xóa hoặc sửa các test case kiểm tra data leakage để qua mặt CI.
- [!] **TUYỆT ĐỐI KHÔNG** bịa đặt số liệu thống kê hoặc kết quả accuracy/F1 mà không có log thực nghiệm chứng minh.

---

## 12. TIÊU CHÍ NGHIỆM THU DỰ ÁN (Acceptance Criteria)

### 12.1. Tiêu chí cho Mô hình Machine Learning (ML Criteria)
- [ ] **AC-ML-01**: Có ít nhất 1 mô hình Baseline truyền thống (TF-IDF + Classifier) và ít nhất 1 mô hình Deep Learning (Transformer tiếng Việt) được huấn luyện và đối sánh trên cùng một tập Test chuẩn.
- [ ] **AC-ML-02**: Có bảng so sánh chi tiết các chỉ số: Macro-F1, Precision, Recall, Accuracy, Inference Latency (ms), Artifact Size (MB).
- [ ] **AC-ML-03**: Mô hình đạt chỉ số **Macro-F1 trên tập Test ≥ 0.70** (đối với Baseline) và **≥ 0.80** (đối với Transformer).
- [ ] **AC-ML-04**: Có kiểm thử chứng minh không có data leakage giữa tập Train và tập Test.
- [ ] **AC-ML-05**: Có ma trận nhầm lẫn (Confusion Matrix) và phân tích các trường hợp mô hình dự đoán sai (Error Analysis).

### 12.2. Tiêu chí cho Ứng dụng Web & Triển khai (Web & Deployment Criteria)
- [ ] **AC-WEB-01**: Giao diện Web hiển thị trực quan, hỗ trợ tiếng Việt hoàn chỉnh, không có lỗi hiển thị font hoặc vỡ layout.
- [ ] **AC-WEB-02**: Cho phép người dùng nhập 1 câu bình luận bất kỳ và trả về nhãn + điểm tin cậy trong thời gian < 500ms.
- [ ] **AC-WEB-03**: Cho phép upload file CSV (tối đa 500 dòng), xử lý theo batch và xuất kết quả file CSV đã gán nhãn cho người dùng tải về.
- [ ] **AC-WEB-04**: Có trang "Model Card / Báo cáo" trình bày minh bạch kiến trúc, kết quả thực nghiệm và giới hạn của mô hình.
- [ ] **AC-WEB-05**: Ứng dụng được triển khai công khai thành công (Publicly accessible URL trên Hugging Face Spaces hoặc Streamlit Cloud), hoạt động ổn định trên Free-tier không bị sập (crash) hay tràn bộ nhớ (OOM).
- [ ] **AC-WEB-06**: Toàn bộ unit tests trong thư mục `tests/` vượt qua 100% trước khi deploy.

---

## 13. LIÊN KẾT TÀI LIỆU LIÊN QUAN (Associated Documents)
- `docs/decisions.md`: Quyết định kiến trúc chi tiết (Streamlit vs FastAPI, PhoBERT vs ViSoBERT, ONNX vs PyTorch).
- `docs/dataset-requirements.md`: Yêu cầu dữ liệu, phân tích ViCTSD vs UIT-ViHSD, checklist EDA.
- `docs/deployment-plan.md`: Kiến trúc triển khai chi tiết, ngân sách tài nguyên và giải pháp cold-start.
- `docs/open-questions.md`: Danh sách câu hỏi mở và quyết định cần chờ dữ liệu thực tế.
