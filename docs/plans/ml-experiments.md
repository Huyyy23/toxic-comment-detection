# Kế hoạch Thực nghiệm Học máy (ML Experiments Plan)
## Vietnamese Toxic Comment Detection (3-Class Classification)

Trạng thái: **APPROVED & SYNCHRONIZED**  
Ngày cập nhật: 2026-11-11  
Dự án: **Vietnamese Toxic Comment Detection System**

---

## 1. NGUYÊN TẮC THỰC NGHIỆM KHOA HỌC & ĐỐI SÁNH CÔNG BẰNG

1. **Không gian Nhãn 3 Lớp Bắt buộc**:
   - `0: CLEAN`: Lành mạnh / không độc hại.
   - `1: OFFENSIVE`: Xúc phạm, thô tục, chửi thề.
   - `2: HATE`: Thù ghét, kỳ thị, công kích định danh nghiêm trọng.
   - *Tuyệt đối không chuyển đổi về nhãn nhị phân (Binary)* để đảm bảo độ mịn của bài toán.
2. **Nguyên tắc "Cùng Tập Test Chuẩn" (Identical Test Set)**:
   - Mô hình truyền thống (Traditional ML) và mô hình Transformer bắt buộc phải được đánh giá trên **cùng một tập test duy nhất** (`data/processed/test.parquet`) với nhãn tương thích 100%.
   - Tập Test không được dùng cho hyperparameter tuning, early stopping hay threshold tuning.
3. **Chống Rò rỉ Thông tin (Leakage Prevention)**:
   - TF-IDF vectorizer chỉ `fit` trên tập Train.
   - Kiểm tra giao thoa văn bản: `set(train_texts) ∩ set(test_texts) == 0`.

---

## 2. KẾ HOẠCH HUẤN LUYỆN TRÊN GPU LOCAL (NVIDIA RTX 4060 8GB)

### 2.1. Cấu hình Môi trường Phần cứng & Tối ưu VRAM
- **GPU**: NVIDIA GeForce RTX 4060 Laptop/Desktop (8GB GDDR6 VRAM).
- **Hệ thống CUDA & PyTorch**: PyTorch với CUDA 11.8 hoặc 12.x hỗ trợ Tensor Cores.
- **Kỹ thuật tiết kiệm VRAM (VRAM Budget < 7.0 GB)**:
  - **Mixed Precision (`fp16` hoặc `bf16`)**: Giảm 50% bộ nhớ activations và weights.
  - **Batch Size & Gradient Accumulation**:
    - `per_device_train_batch_size = 16`
    - `gradient_accumulation_steps = 2` (tạo effective batch size = 32)
  - **Độ dài Tokenizer giới hạn (`max_length`)**: Đặt ở mức 128 hoặc 160 tokens (dựa trên phân vị 95% của EDA).
  - **DataLoader**: `pin_memory=True`, `num_workers=2`.
- **Dự phòng (Fallback)**: Google Colab / Kaggle T4 GPU (chỉ sử dụng khi local gặp lỗi CUDA hoặc tràn VRAM).

### 2.2. Không gian Siêu Tham số Huấn luyện (Hyperparameter Search Space)
- **Mô hình Ứng viên**:
  1. `uitnlp/visobert` (Ưu tiên số 1: Pretrained trên mạng xã hội tiếng Việt).
  2. `vinai/phobert-base-v2` (Pretrained tổng quát, cần word segmentation).
- **Hàm mất mát**: `CrossEntropyLoss(weight=class_weights)` trong đó `class_weights` được tính toán tự động nghịch đảo tỉ lệ 3 lớp từ tập train.
- **Tối ưu hóa**: `AdamW` với learning rate: $2\times 10^{-5}$ đến $3\times 10^{-5}$, weight decay = 0.01.
- **Scheduler**: Linear Warmup 10% và Cosine Annealing.
- **Epochs & Early Stopping**: Tối đa 5 epochs, early stopping sau 2 epochs nếu Validation Macro-F1 không tăng.

---

## 3. THIẾT KẾ ĐỐI SÁNH KHOA HỌC (Benchmarking Protocol)

### 3.1. Nhóm 1: Mô hình Cơ sở Truyền thống (Traditional ML Baselines)
- **Đặc trưng**: TF-IDF Word N-grams (1, 2) + Char N-grams (2, 4), `max_features = 10000`.
- **Phân loại**:
  - `Baseline 1`: Multinomial Logistic Regression (`multi_class='multinomial'`, `class_weight='balanced'`).
  - `Baseline 2`: LinearSVC (`multi_class='ovr'`, `class_weight='balanced'`).
- **Thời gian chạy**: < 2 phút trên CPU.

### 3.2. Nhóm 2: Mô hình Transformer Nâng cao
- Fine-tuned trên GPU RTX 4060, lưu checkpoint tốt nhất tại `models/transformer_best/`.

### 3.3. Bảng Đối sánh Chuẩn (Comparison Matrix)
*Điền bằng số liệu thực nghiệm thực tế sau khi chạy mã nguồn:*

| Mô hình (Model) | Backend | Macro-F1 | Clean F1 | Offensive F1 | Hate F1 | Latency (CPU ms) | Artifact Size (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + Logistic Reg.** | Sklearn | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **TF-IDF + LinearSVC** | Sklearn | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **PhoBERT-base-v2** | PyTorch | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **ViSoBERT** | PyTorch | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **ViSoBERT (ONNX INT8)** | ONNX | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 4. QUY TRÌNH KIỂM THỬ TÍCH HỢP ĐA BACKEND & BENCHMARK ONNX

1. **Kiểm thử Tích hợp Đa Backend**:
   - Trước khi triển khai lên Web, module `tests/test_inference_backends.py` bắt buộc phải kiểm tra cả `SklearnBackend` và `PyTorchBackend`:
     - Nhận cùng chuỗi text mẫu.
     - Trả về đúng 3 xác suất $P(\text{Clean}) + P(\text{Offensive}) + P(\text{Hate}) \approx 1.0$.
     - Trả về nhãn id (0, 1, 2) và nhãn chữ tương ứng.
2. **Quy tắc Benchmark ONNX (Có điều kiện)**:
   - Chạy script đo lường so sánh giữa `PyTorch (CPU)` và `ONNX (CPU)`:
     - Đo độ trễ trung bình trên 100 câu mẫu.
     - Đo bộ nhớ RAM đỉnh điểm (Peak RAM).
   - **Tiêu chuẩn kích hoạt ONNX**: Chỉ chuyển sang ONNX làm default engine nếu ONNX giúp giảm độ trễ ít nhất 25% hoặc giảm dung lượng RAM/Artifact rõ rệt mà không làm sụt giảm Macro-F1 (> 0.5%). Nếu không, giữ nguyên PyTorch backend.
