# Vietnamese Toxic Comment Detection System — Specification (SPEC.md)

Bản đặc tả chi tiết và đầy đủ nhất được lưu trữ tại: **[`docs/spec.md`](docs/spec.md)**  
Các tài liệu thành phần đi kèm:
- Kiến trúc & Quyết định đánh đổi: [`docs/decisions.md`](docs/decisions.md)
- Yêu cầu Dữ liệu & Quy trình Kiểm tra EDA: [`docs/dataset-requirements.md`](docs/dataset-requirements.md)
- Kế hoạch Triển khai & Ràng buộc Tài nguyên: [`docs/deployment-plan.md`](docs/deployment-plan.md)
- Danh sách Câu hỏi Mở & Quyết định Phụ thuộc Dataset: [`docs/open-questions.md`](docs/open-questions.md)

---

## Tóm tắt Nhanh (Executive Summary)

### 1. Bản đồ Khả năng & Thứ tự Xây dựng (Capability Map)
| Module ID | Trách nhiệm | Phụ thuộc |
| :--- | :--- | :--- |
| `mod-data` | Tải dữ liệu từ Hugging Face (`ViCTSD` / `UIT-ViHSD`), EDA, chuẩn hóa Unicode/teencode, chống leakage | — |
| `mod-train` | Huấn luyện Baseline (TF-IDF + LR/SVM) và Transformer (PhoBERT/ViSoBERT), đối sánh khoa học | `mod-data` |
| `mod-export` | Đóng gói PyTorch weights sang ONNX + INT8 Quantization, tối ưu bộ nhớ | `mod-train` |
| `mod-ui` | Xây dựng giao diện Web Streamlit (Single input, Batch CSV, Model Card report) | `mod-export` |
| `mod-deploy` | Triển khai công khai miễn phí trên Hugging Face Spaces (CPU Basic) | `mod-ui` |

**Build Order**: `mod-data` ──→ `mod-train` ──→ `mod-export` ──→ `mod-ui` ──→ `mod-deploy`

### 2. Danh mục Lệnh Chính (Core Commands)
```bash
# Setup
pip install -r requirements-dev.txt

# Test & Quality
pytest tests/ -v --cov=src/
flake8 src/ app/ tests/

# Data & Training
python -m src.data.download --dataset uit-vihsd --output data/raw/
python -m src.data.preprocessor --input data/raw/ --output data/processed/
python -m src.models.baseline --config configs/baseline_config.yaml
python -m src.models.transformer --config configs/transformer_config.yaml
python -m src.models.export_onnx --input models/transformer_best/ --output models/exported/

# Local Demo
streamlit run app/main.py --server.port 8501
```

### 3. Tiêu chí Nghiệm thu Cốt lõi (Key Acceptance Criteria)
- **ML**: Macro-F1 trên tập Test ≥ 0.70 (Baseline) và ≥ 0.80 (Transformer); không có data leakage; có confusion matrix & error analysis.
- **Web**: Phản hồi suy luận đơn lẻ < 500ms; hỗ trợ batch CSV ≤ 500 dòng; có trang giải thích mô hình; deploy live thành công trên Hugging Face Spaces (0 VNĐ chi phí hạ tầng).

---
*Vui lòng xem chi tiết từng mục trong [`docs/spec.md`](docs/spec.md).*
