# Danh sách Câu hỏi Mở & Trạng thái Xác minh (Open Questions & Verification Ledger)

Trạng thái: **SYNCHRONIZED WITH APPROVED SPEC**  
Ngày cập nhật: 2026-11-11  
Dự án: **Vietnamese Toxic Comment Detection System**

---

## 1. Các Quyết định Đã được Phê duyệt (Confirmed & Closed)

| ID | Vấn đề | Quyết định đã phê duyệt |
| :--- | :--- | :--- |
| **DD-01** | **Chiến lược Dataset** | **Confirmed**: Tải cả `ViCTSD` và `UIT-ViHSD` để chạy EDA so sánh. Ưu tiên `UIT-ViHSD` làm tập chính, chốt chính thức sau EDA. |
| **DD-02** | **Không gian Nhãn (Label Taxonomy)** | **Confirmed**: Sử dụng phân loại **3 lớp CLEAN / OFFENSIVE / HATE**. Không gộp về Binary. |
| **DD-03** | **Kiến trúc Web & Triển khai** | **Confirmed**: Ứng dụng **Docker container** kết hợp **React (Frontend) + FastAPI (Backend)**, deploy lên **Hugging Face Spaces**. |
| **DD-04** | **Phần cứng Huấn luyện Transformer** | **Confirmed**: Ưu tiên huấn luyện trên **Local GPU NVIDIA GeForce RTX 4060 8GB** (Google Colab/Kaggle là backup). |
| **DD-05** | **Kiến trúc Inference Engine** | **Confirmed**: Đa backend (`SklearnBackend`, `PyTorchBackend`, `ONNXBackend` tùy chọn dựa trên benchmark). Không ép buộc ONNX từ đầu. |
| **DD-06** | **Tính công bằng trong Thực nghiệm** | **Confirmed**: Mô hình truyền thống và Transformer bắt buộc dùng chung 1 tập test chuẩn và nhãn 3 lớp đồng nhất. |

---

## 2. Các Hạng mục Xác minh Kỹ thuật Cần Thực hiện (Pending Technical Verifications)

### TV-01: Xác minh Điều kiện Triển khai Hugging Face Docker Space (Thực hiện tại M0)
- **Mục tiêu**: Kiểm tra xem tài khoản Hugging Face của người dùng có đủ điều kiện tạo và chạy Docker Space miễn phí mà không bị chặn xác thực danh tính/thẻ tín dụng hay không.
- **Tiêu chuẩn kiểm tra**:
  - Có Hugging Face User Access Token (với quyền Write).
  - Khởi tạo thử nghiệm hoặc kiểm tra API endpoint của Hugging Face Spaces.
- **Phương án xử lý nếu không đủ điều kiện (Blocker Fallback)**:
  - Ghi nhận blocker vào `docs/plans/deployment.md`.
  - Giữ nguyên kiến trúc Docker và đề xuất phương án hosting thay thế (Render Free Docker Web Service, Fly.io hoặc chạy local container) cho người dùng phê duyệt, không tự ý đổi nền tảng.

### TV-02: Quyết định Phụ thuộc Dataset (Thực hiện tại M1 sau khi tải dữ liệu)
- **Tỉ lệ phân bố 3 lớp**: Đo đếm số lượng và % của `CLEAN`, `OFFENSIVE`, `HATE` trong `UIT-ViHSD`. Tính toán ma trận trọng số `class_weight` cho hàm mất mát.
- **Phân bố độ dài chuỗi**: Đo độ dài từ/ký tự ở phân vị 95% để thiết lập `max_length` cho tokenizer (dự kiến 128 hoặc 192 tokens để tiết kiệm VRAM trên RTX 4060).
- **Đặc trưng ngôn ngữ tiếng Việt**: Thống kê mức độ teencode, từ lóng, icon để xây dựng từ điển chuẩn hóa trong `src/data/preprocessor.py`.
