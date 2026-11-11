# Báo cáo Quyết định Kiến trúc & Phân tích Đánh đổi (Architecture Decisions & Trade-offs)

Trạng thái: **PROPOSED / SPECIFICATION REVIEW**  
Ngày: 2026-11-11  
Dự án: **Vietnamese Toxic Comment Detection System**

---

## ADR-01: Lựa chọn Giao diện & Kiến trúc Web (Streamlit vs React + FastAPI)

### 1. Bối cảnh
Hệ thống cần một giao diện trực quan cho giảng viên và hội đồng nghiệm thu kiểm thử:
- Nhập trực tiếp văn bản bình luận tiếng Việt.
- Xem kết quả phân loại độc hại (Toxicity), điểm tin cậy (Confidence Score/Probabilities), và nhãn giải thích.
- Chế độ nhập hàng loạt (Batch input qua file CSV hoặc text area).
- Xem thông tin mô hình, kiến trúc, và kết quả thực nghiệm khoa học.

### 2. So sánh Toàn diện

| Tiêu chí | Phương án A: Streamlit (All-in-one Python) | Phương án B: React (Frontend) + FastAPI (Backend) |
| :--- | :--- | :--- |
| **Độ phức tạp phát triển** | **Rất thấp**: 1 file / 1 repo Python duy nhất, không cần viết HTML/CSS/JS/API glue code. | **Cao**: Cần duy trì 2 codebases (hoặc monorepo), viết REST API, xử lý CORS, state management frontend. |
| **Thời gian hoàn thiện (Time to Market)** | **1 - 2 ngày** cho UI hoàn chỉnh có biểu đồ, upload file, metric display. | **4 - 7 ngày** (cần thiết kế giao diện, build pipeline, kết nối API). |
| **Tài nguyên Hosting & Triển khai** | **Tối ưu**: Triển khai dễ dàng trên **Hugging Face Spaces** (Streamlit SDK) hoặc Streamlit Community Cloud hoàn toàn miễn phí. | Cần 2 dịch vụ riêng biệt (Vercel/Netlify cho React + Render/Fly.io cho FastAPI) hoặc đóng gói Docker đa tầng. |
| **Khả năng mở rộng (Scalability)** | Kém hơn khi tải cao (stateful execution model của Streamlit tiêu tốn RAM khi có nhiều session đồng thời). | Rất tốt, API stateless, scale độc lập backend và frontend. |
| **Độ linh hoạt tùy biến UI/UX** | Bị giới hạn trong widget layout của Streamlit (tuy nhiên có sẵn components cho charts, cards, tabs, dataframe). | Hoàn toàn tự do tùy biến design, responsive, styling. |
| **Phù hợp với mục tiêu Đồ án** | **9.5/10**: Tập trung tối đa nguồn lực vào chất lượng mô hình ML, pipeline và benchmark khoa học thay vì tốn thời gian cho web boilerplate. | **6.5/10**: Nguy cơ overengineering, tốn thời gian debug CORS, frontend packaging thay vì ML. |

### 3. Quyết định (Decision)
- **Chọn Phương án A (Streamlit)** cho giai đoạn MVP và đồ án kết thúc môn.
- **Dự phòng (Proposed Fallback)**: Cấu trúc mã nguồn theo hướng **Clean Architecture**: Tách biệt hoàn toàn `src/inference/` (Core predictor) độc lập với `app/` (UI). Nếu cần chuyển sang FastAPI về sau, chỉ cần viết một file `app_api.py` bọc quanh `src/inference/engine.py` mà không phải sửa logic ML.

---

## ADR-02: Chiến lược Lựa chọn Mô hình (Baseline vs Advanced Transformer)

### 1. Bối cảnh
Đồ án học thuật bắt buộc phải có tính so sánh khoa học: không thể chỉ chọn một mô hình Transformer hiện đại mà không chứng minh được sự vượt trội so với các mô hình baseline truyền thống, đồng thời phải cân nhắc ràng buộc tài nguyên triển khai (RAM < 2GB trên môi trường free-tier).

### 2. Ma trận So sánh Ứng viên Mô hình

| Nhóm | Mô hình Ứng viên | Ưu điểm | Nhược điểm / Rủi ro | Đánh giá |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline 1** | **TF-IDF (Word + N-gram) + Logistic Regression / LinearSVC** | - Huấn luyện < 1 phút trên CPU.<br>- Kích thước artifact cực nhỏ (< 20MB).<br>- Tốc độ suy luận tính bằng mili-giây.<br>- Khả năng giải thích cao (Feature importances). | - Không nắm bắt được ngữ cảnh phức tạp.<br>- Nhạy cảm với teencode, typo và từ đồng nghĩa. | **Bắt buộc có** làm chuẩn đối sánh (Lower-bound Benchmark). |
| **Baseline 2 (Tùy chọn)** | **Bi-LSTM / TextCNN + FastText Vietnamese Embedding** | - Bắt được thứ tự từ.<br>- Chi phí tính toán vừa phải. | - Cần embedding pretrained tiếng Việt dung lượng lớn (~1-2GB).<br>- Hiệu quả thường thua Transformer. | Không ưu tiên nếu thời gian giới hạn. |
| **Advanced 1** | **PhoBERT-base (`vinai/phobert-base-v2`)** | - Pretrained mạnh nhất trên dữ liệu tiếng Việt.<br>- Độ chính xác ngôn ngữ cao.<br>- Hỗ trợ word-level tokenization (pyvi / RDRsegmenter). | - Kích thước model ~540MB (float32).<br>- Cần bước tách từ tiếng Việt (word segmentation). | **Ứng viên sáng giá nhất cho Advanced**. |
| **Advanced 2** | **ViSoBERT (`uitnlp/visobert`)** | - Pretrained chuyên sâu trên dữ liệu mạng xã hội tiếng Việt (Social Media Domain).<br>- Nhận biết teencode, slang và icon cực tốt. | - Kích thước ~500MB.<br>- Cần xác minh độ tương thích tokenizer. | **Ứng viên ưu tiên số 1 nếu dataset là ViHSD/ViCTSD**. |
| **Advanced 3** | **Multilingual (XLM-RoBERTa-base)** | - Không cần tách từ.<br>- Xử lý đa ngôn ngữ (Việt - Anh xen kẽ). | - Kích thước lớn (~1.1GB), nặng RAM khi inference trên CPU free-tier. | Dự phòng. |

### 3. Quyết định (Decision)
- **Baseline**: Xây dựng TF-IDF (char n-gram + word n-gram) kết hợp Logistic Regression / LinearSVC.
- **Advanced Model**: Thực nghiệm đối sánh giữa **PhoBERT-base-v2** và **ViSoBERT**.
- Chọn mô hình có Macro-F1 cao nhất trên tập Test, đồng thời thỏa mãn giới hạn RAM/Latency khi deploy.

---

## ADR-03: Tối ưu Hóa Kích thước & Tốc độ Suy luận (Inference Optimization)

### 1. Bối cảnh
Môi trường hosting miễn phí (như Hugging Face Spaces free-tier hoặc Render Free) giới hạn nghiêm ngặt:
- CPU: 2 vCPU.
- RAM: 2GB (Render chỉ có 512MB RAM - rất dễ bị OOM; Hugging Face Spaces cấp 16GB RAM cho CPU Basic - an toàn hơn nhiều).
- Không có GPU cho live demo.

### 2. Phương án Đề xuất
1. **PyTorch Native (FP32)**:
   - Baseline triển khai ban đầu: load trực tiếp weights PyTorch qua Hugging Face `transformers`.
   - Dung lượng: ~500MB. Startup time: ~5-10s.
2. **ONNX Runtime + Dynamic INT8 Quantization (Đề xuất tối ưu cho Production)**:
   - Convert PyTorch weights sang `.onnx` và áp dụng dynamic quantization (quantize các lớp Linear sang INT8).
   - Dung lượng giảm từ ~500MB xuống còn ~135MB.
   - Tốc độ suy luận trên CPU tăng gấp 2 - 3 lần.
   - Dung lượng RAM runtime giảm rõ rệt.

### 3. Quyết định
- MVP giai đoạn 1: Load PyTorch FP32 để đảm bảo tính sẵn sàng nhanh.
- Bước tối ưu hóa sau training: Script convert ONNX được định nghĩa sẵn trong pipeline (`src/models/export_onnx.py`). Nếu host gặp giới hạn RAM, chuyển đổi ngay sang ONNX Runtime.

---

## ADR-04: Nền tảng Triển khai Demo Công khai (Public Hosting)

### 1. Phân tích các Nền tảng Miễn phí

| Nền tảng | Cấu hình Miễn phí | Giới hạn / Rủi ro | Đánh giá |
| :--- | :--- | :--- | :--- |
| **Hugging Face Spaces (Streamlit SDK)** | - 2 vCPU, 16GB RAM, 50GB Disk.<br>- Hoàn toàn miễn phí, không sleep nếu có traffic nhẹ.<br>- Tích hợp sẵn với Hugging Face Model Hub. | Cần tài khoản Hugging Face; cộng đồng ML quốc tế dễ truy cập. | **LỰA CHỌN TỐI ƯU NHẤT (Top 1)** |
| **Streamlit Community Cloud** | - Miễn phí kết nối GitHub repo.<br>- RAM ~1GB. | Thỉnh thoảng bị lỗi resource limit khi load mô hình Transformer nặng (>500MB). | Lựa chọn dự phòng cho Baseline. |
| **Render.com (Web Service)** | - 0.1 CPU, 512MB RAM.<br>- Sleep sau 15 phút không dùng. | **Rất dễ bị dính lỗi Out-of-Memory (OOM)** khi import `torch` và load transformer model. | Không khuyến nghị cho deep learning model. |

### 2. Quyết định (Decision)
- **Primary Hosting Platform**: **Hugging Face Spaces** sử dụng **Streamlit SDK**.
  - Đảm bảo đủ RAM (16GB RAM miễn phí) cho mô hình NLP tiếng Việt hoạt động trơn tru, không lo OOM.
  - Có link public trực tiếp dạng `https://huggingface.co/spaces/<username>/vietnamese-toxic-detector` để nộp đồ án.
