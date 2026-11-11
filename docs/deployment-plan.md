# Kế hoạch Triển khai & Ràng buộc Tài nguyên (Deployment Plan)

Trạng thái: **PROPOSED / SPECIFICATION**  
Ngày: 2026-11-11  
Dự án: **Vietnamese Toxic Comment Detection System**

---

## 1. Kiến trúc Hệ thống Triển khai (System Architecture)

Hệ thống được thiết kế theo kiến trúc **Monolith Tinh gọn (Lightweight Modular Monolith)**, đóng gói toàn bộ logic inference và presentation trong một ứng dụng duy nhất để tối đa hóa độ tin cậy và tối thiểu hóa chi phí hạ tầng.

```
+-----------------------------------------------------------------------+
|                            USER BROWSER                               |
|              (Nhập bình luận / Batch CSV / Xem metrics)               |
+-----------------------------------+-----------------------------------+
                                    | HTTPS (Web UI)
                                    v
+-----------------------------------------------------------------------+
|                 HUGGING FACE SPACES (Free CPU Tier)                   |
|                  2 vCPU | 16 GB RAM | 50 GB Storage                   |
|                                                                       |
|   +---------------------------------------------------------------+   |
|   |                 Streamlit Application Layer                   |   |
|   |  - app/main.py (Multi-page or Tabs layout)                    |   |
|   |  - Components: Input form, Batch uploader, Score Gauge        |   |
|   +-------------------------------+-------------------------------+   |
|                                   | In-memory function call           |
|                                   v                                   |
|   +---------------------------------------------------------------+   |
|   |               Core ML Inference Engine (Singleton)            |   |
|   |  - src/inference/predictor.py (Decorated with cache_resource) |   |
|   |  - Preprocessing: regex, emoji map, vietnamese word segment   |   |
|   |  - Inference: ONNX Runtime (hoặc PyTorch FP32)                |   |
|   +-------------------------------+-------------------------------+   |
|                                   | Read-only access                  |
|                                   v                                   |
|   +---------------------------------------------------------------+   |
|   |                    Packaged Model Artifacts                   |   |
|   |  - models/exported/model.onnx (hoặc pytorch_model.bin)        |   |
|   |  - models/exported/tokenizer_config.json, vocab.txt           |   |
|   |  - models/exported/label_encoder.json, metadata.json          |   |
|   +---------------------------------------------------------------+   |
+-----------------------------------------------------------------------+
```

---

## 2. Ngân sách Tài nguyên & Giới hạn Hạ tầng (Resource Budget & Limits)

| Tham số hạ tầng | Ngưỡng cho phép (Budget) | Giải pháp kiểm soát |
| :--- | :--- | :--- |
| **Dung lượng Artifacts (Model Weight)** | **≤ 500 MB** (PyTorch) hoặc **≤ 150 MB** (ONNX INT8) | Loại bỏ optimizer states (`optimizer.pt`), chỉ lưu model weights và tokenizer assets; nén hoặc tải dynamic từ Hugging Face Hub. |
| **Dung lượng RAM tối đa** | **< 2.0 GB** khi tải 10 requests/phút | Dùng PyTorch with `torch.no_grad()`, giải phóng cache; dùng Singleton pattern để chỉ load 1 instance model vào RAM. |
| **CPU Spikes** | Giới hạn 2 vCPU | Sử dụng ONNX Runtime tối ưu phép nhân ma trận trên CPU; batch size khi suy luận batch không quá 32. |
| **Thời gian khởi động (Startup Time)** | **< 30 giây** | Model load dùng lazy loading hoặc pre-load trong `@st.cache_resource` khi app init; không thực hiện train hoặc fetch file dung lượng lớn lúc startup. |
| **Độ trễ suy luận (Inference Latency)** | **Single comment: < 300ms**<br>**Batch 20 comments: < 2.0s** | Cache tokenization, không lặp lại khởi tạo model, chạy inference trên ONNX/CPU. |
| **Chi phí hạ tầng (Cloud Billing)** | **0 VNĐ (Hoàn toàn miễn phí)** | Sử dụng Hugging Face Spaces CPU Basic Tier hoặc Streamlit Community Cloud. |

---

## 3. Quy trình Đóng gói & Quản lý Artifact (Model Packaging)

Để tránh tình trạng repo Git bị phình to (vượt quá 100MB của GitHub):

### Chiến lược Quản lý Trọng số Mô hình (Proposed):
- **Phương án 1 (Khuyến nghị)**: Đẩy model weights đã huấn luyện lên **Hugging Face Model Hub** cá nhân (vd: `username/vietnamese-toxic-phobert-onnx`). Ứng dụng Streamlit trên Spaces sẽ tự động pull model về cache khi khởi chạy lần đầu:
  ```python
  from huggingface_hub import hf_hub_download
  model_path = hf_hub_download(repo_id="your-username/model-name", filename="model.onnx")
  ```
- **Phương án 2 (Dự phòng cho Baseline TF-IDF)**: Model `.joblib` hoặc `.pkl` của Logistic Regression thường < 10MB, có thể lưu trực tiếp trong thư mục `models/` của repo.

---

## 4. Các Bước Triển khai Chi tiết (Deployment Procedure)

1. **Chuẩn bị Repository**:
   - File cấu hình `requirements.txt` tối giản chỉ chứa dependencies chạy web và inference (không kèm thư viện train nặng như `accelerate`, `jupyter`, `wandb`).
   - File `packages.txt` (nếu cần thư viện C++ hệ thống cho `pyvi` hoặc `underthesea`).
   - File `.gitignore` chặn đẩy các file dữ liệu raw và checkpoint huấn luyện dở dang lên Git.
2. **Khởi tạo Hugging Face Space**:
   - Chọn SDK: `Streamlit`.
   - Hardware: `CPU basic - 2 vCPU - 16 GB - Free`.
   - Kết nối với GitHub Repo (Sync) hoặc Push trực tiếp vào Git của Hugging Face Space.
3. **Smoke Test sau Deploy**:
   - Kiểm tra UI hiển thị đúng giao diện.
   - Nhập một bình luận mẫu lành mạnh: "Bài viết này rất bổ ích, cảm ơn bạn." -> Kết quả: `Clean` / `Non-toxic` (Confidence > 0.85).
   - Nhập một bình luận mẫu có từ ngữ xúc phạm -> Kết quả: `Toxic` / `Hate` (Confidence > 0.85).
   - Kiểm tra tab thông tin và báo cáo khoa học.

---

## 5. Giám sát & Quản lý Sự cố (Monitoring & Fallback)

- **Crash do OOM (Out of Memory)**: Streamlit Spaces tự khởi động lại container. Hệ thống ghi log kích thước input vào stdout để debug.
- **Input quá dài (Adversarial Long Input)**: Cắt ngắn văn bản đầu vào (`truncate`) ở mức tối đa 256 tokens hoặc 500 ký tự trước khi đưa vào mô hình, ngăn chặn việc tràn bộ nhớ hoặc treo CPU.
- **Fallback Rule**: Nếu mô hình Transformer gặp lỗi không lường trước lúc load, app có cơ chế fallback chuyển sang dùng Baseline model TF-IDF nhẹ để người dùng không bị gián đoạn trải nghiệm demo.
