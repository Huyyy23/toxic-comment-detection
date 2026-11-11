# Kế hoạch Thực nghiệm Học máy (ML Experiments Plan)
## Vietnamese Toxic Comment Detection (3-Class Classification)

Trạng thái: **APPROVED & SYNCHRONIZED**  
Ngày cập nhật: 2026-11-11  
Dự án: **Vietnamese Toxic Comment Detection System**

---

## 1. NGUYÊN TẮC THỰC NGHIỆM KHOA HỌC & ĐỐI SÁNH CÔNG BẰNG

1. **Phân định Nghiêm ngặt Vai trò các Phân vùng Dữ liệu**:
   - **Tập Train (70%)**: Huấn luyện trọng số mô hình và trích xuất đặc trưng TF-IDF.
   - **Tập Validation (15%)**: **DUY NHẤT** được dùng để điều chỉnh siêu tham số (tuning C, learning rate, batch size), early stopping và lựa chọn mô hình tốt nhất (Model Selection).
   - **Tập Test (15%) - BỊ KHÓA (LOCKED / SEALED)**:
     - Bị khóa hoàn toàn trong toàn bộ quá trình nghiên cứu và tối ưu.
     - Chỉ mở ra đúng một lần ở bước đánh giá cuối cùng để đối sánh khách quan giữa Baseline và Transformer đã chốt từ tập Validation.
     - **Tuyệt đối không dùng kết quả Test set để tối ưu ngược lại mô hình, chọn tham số hay sửa đổi pipeline**.
2. **Không gian Nhãn 3 Lớp Bắt buộc**:
   - `0: CLEAN`: Lành mạnh / không độc hại.
   - `1: OFFENSIVE`: Xúc phạm, thô tục, chửi thề.
   - `2: HATE`: Thù ghét, kỳ thị, công kích định danh nghiêm trọng.
   - *Tuyệt đối không chuyển đổi về nhãn nhị phân (Binary)* để đảm bảo độ mịn của bài toán.
3. **Chống Rò rỉ Thông tin (Leakage Prevention)**:
   - TF-IDF vectorizer chỉ `fit` trên tập Train.
   - Kiểm tra giao thoa văn bản: `set(train_texts) ∩ set(test_texts) == 0`.
4. **Hợp đồng Xác suất Nghiêm ngặt (Probability Contract)**:
   - Logistic Regression sử dụng `.predict_proba()` chính thống.
   - LinearSVC sử dụng `CalibratedClassifierCV(estimator=LinearSVC(...), cv=5, method='sigmoid')` (Platt scaling) để sinh xác suất hiệu chuẩn thực sự.
   - Không được giả mạo confidence bằng hàm softmax tự chế trên uncalibrated decision margins.

---

## 2. KẾ HOẠCH HUẤN LUYỆN TRÊN GPU LOCAL (NVIDIA RTX 4060 8GB)

### 2.1. Cấu hình Môi trường & Siêu Tham số Tạm thời (`[PROVISIONAL]`)
*Mọi thông số dưới đây là **`[PROVISIONAL / TẠM THỜI]`**, sẽ được điều chỉnh chính thức dựa trên kết quả EDA và benchmark VRAM:*
- **Mô hình Ứng viên**: `uitnlp/visobert` (Social Media Domain) và `vinai/phobert-base-v2`.
- **Cấu hình Huấn luyện Tạm thời (Chờ Validation Tuning)**:
  - `per_device_train_batch_size = 16` [PROVISIONAL]
  - `gradient_accumulation_steps = 2` (effective batch size = 32) [PROVISIONAL]
  - `learning_rate = 2e-5` (grid search [1e-5, 2e-5, 3e-5] trên Validation set) [PROVISIONAL]
  - `max_length = 160` (sẽ chốt dựa trên phân vị 95% độ dài chuỗi ở EDA) [PROVISIONAL]
  - `VRAM budget < 7.0GB` (sẽ đo đạc bộ nhớ thực tế qua batch profiling) [PROVISIONAL]
  - Mixed precision `fp16 = True` [PROVISIONAL: phụ thuộc kiểm tra driver CUDA local]
  - Early stopping: Dừng sau 2 epochs nếu Validation Macro-F1 không tăng.

---

## 3. THIẾT KẾ ĐỐI SÁNH KHOA HỌC KHÁCH QUAN (Objective Benchmarking)

### 3.1. Nhóm 1: Mô hình Cơ sở Truyền thống (Traditional ML Baselines)
- **Đặc trưng**: TF-IDF Word N-grams (1, 2) + Char N-grams (2, 4), `max_features = 10000` [PROVISIONAL].
- **Mô hình**:
  1. `Logistic Regression` (`class_weight='balanced'`).
  2. `LinearSVC` bọc trong `CalibratedClassifierCV(cv=5, method='sigmoid')`.
- **Lựa chọn Baseline Tốt nhất**: Chọn mô hình có Macro-F1 cao nhất trên tập **Validation**.

### 3.2. Nhóm 2: Mô hình Transformer Nâng cao
- Fine-tuned trên GPU RTX 4060 (Train set), lưu checkpoint tốt nhất dựa trên **Validation Macro-F1**.

### 3.3. Đánh giá Đối kháng trên Tập Test Đã Khóa (Final Locked Test Evaluation)
- Cả Best Baseline và Best Transformer cùng chạy dự đoán trên tập `test.parquet`.
- **Nguyên tắc Đánh giá Khách quan**: Không bắt buộc Transformer phải vượt Baseline; mục tiêu là ghi nhận trung thực toàn bộ chỉ số:
  - Macro-F1, Per-class Precision/Recall/F1 cho 3 lớp (`CLEAN`, `OFFENSIVE`, `HATE`).
  - Ma trận nhầm lẫn (Confusion Matrix).
  - Độ trễ suy luận trên CPU (ms) và dung lượng mô hình (MB).
  - Phân tích nguyên nhân và trade-offs (ví dụ: Transformer chính xác hơn nhưng tốn tài nguyên hơn, hoặc ngược lại).

| Mô hình (Model) | Backend | Val Macro-F1 | Test Macro-F1 | Test Clean F1 | Test Offensive F1 | Test Hate F1 | Latency (CPU ms) | Size (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Best Baseline (Calibrated)** | Sklearn | *TBD (Val)* | *TBD (Test)* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Best Transformer** | PyTorch | *TBD (Val)* | *TBD (Test)* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Best Transformer (ONNX INT8)** | ONNX | *TBD (Val)* | *TBD (Test)* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 4. QUY TRÌNH KIỂM THỬ TÍCH HỢP TÁCH BIỆT & BENCHMARK ONNX

1. **Baseline Integration Test (`tests/test_baseline_integration.py`)**:
   - Kiểm thử độc lập ngay sau M2: nạp Baseline `.joblib`, kiểm tra xác suất hiệu chuẩn, tích hợp với API schema, unblock phát triển Web M5.
2. **Transformer Integration Test (`tests/test_transformer_integration.py`)**:
   - Kiểm thử độc lập sau M3: nạp checkpoint PyTorch, kiểm tra output xác suất 3 lớp, so sánh latency với Baseline.
3. **Quy tắc Benchmark ONNX (Có điều kiện)**:
   - Đo lường so sánh giữa `PyTorch (CPU)` và `ONNX (CPU)`:
     - Đo độ trễ trung bình trên 100 câu mẫu.
     - Đo bộ nhớ RAM đỉnh điểm (Peak RAM).
   - **Tiêu chuẩn kích hoạt ONNX**: Chỉ chuyển sang ONNX làm default engine nếu ONNX giúp giảm độ trễ $\ge 25\%$ hoặc giảm dung lượng RAM rõ rệt mà không làm sụt giảm Macro-F1 (> 0.5%). Nếu không, giữ nguyên PyTorch backend.
