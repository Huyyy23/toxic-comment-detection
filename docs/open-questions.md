# Danh sách Câu hỏi Mở & Các Quyết định Phụ thuộc Dataset (Open Questions & Dataset-Dependent Decisions)

Trạng thái: **ACTIVE / AWAITING EDA & VERIFICATION**  
Ngày lập: 2026-11-11  
Dự án: **Vietnamese Toxic Comment Detection System**

---

## 1. Các Quyết định Bắt buộc Chờ Dữ liệu Thực tế (Dataset-Dependent Decisions)

Mọi quyết định dưới đây đang ở trạng thái **[TBD]** hoặc **[Assumption]** và **không được phép tự ý chốt cứng** trước khi hoàn thành kịch bản Khám phá Dữ liệu (EDA):

| ID | Vấn đề cần quyết định | Trạng thái hiện tại | Điều kiện để chốt (Verification Trigger) |
| :--- | :--- | :--- | :--- |
| **DD-01** | **Lựa chọn Dataset chính: ViCTSD hay UIT-ViHSD hay cả hai?** | **[Assumption]**: UIT-ViHSD phù hợp trực diện cho Hate/Toxic Speech, còn ViCTSD phù hợp cho Constructive/Toxic Speech. | Cần tải mẫu (sample preview) của cả 2 bộ để xem chất lượng gán nhãn, độ dài trung bình và mức độ phù hợp với mục tiêu đề tài. |
| **DD-02** | **Định nghĩa Không gian Nhãn (Label Taxonomy): Binary vs 3-Class vs Multi-label** | **[TBD]** | Kiểm tra phân bố nhãn trong dataset raw: nếu nhãn thù ghét quá ít (<3%), có thể phải gộp `Offensive` và `Hate` thành `Toxic` (Binary: Toxic vs Clean) để đảm bảo mô hình học ổn định. |
| **DD-03** | **Chiến lược Tokenization tiếng Việt (Word-level vs Subword-level)** | **[Proposed]**: Dùng `pyvi` hoặc `rdrsegmenter` nếu chọn PhoBERT; dùng Subword trực tiếp nếu chọn ViSoBERT/XLM-R. | Xác định sau khi kiểm tra tỉ lệ từ lóng, teencode và ký tự né kiểm duyệt trong dữ liệu. Nếu teencode quá nhiều, công cụ tách từ truyền thống có thể phân rã sai. |
| **DD-04** | **Cấu hình `max_length` của Tokenizer** | **[TBD]** (Dự kiến 128 hoặc 256) | Cần vẽ đồ thị phân bố độ dài ký tự/từ của bình luận. Chọn `max_length` bao phủ ≥ 95% số lượng mẫu để tối ưu tốc độ và bộ nhớ GPU/CPU. |
| **DD-05** | **Xử lý Class Imbalance (Tỉ lệ mất cân bằng)** | **[TBD]** | Thống kê số lượng từng class trong tập train. Nếu tỉ lệ mất cân bằng > 1:5, bắt buộc bật `class_weight` trong hàm Loss và đo đạc Macro-F1. |
| **DD-06** | **Chính sách Tiền xử lý Icon / Emoji / Ký tự đặc biệt** | **[Proposed]**: Giữ lại Emoji biểu thị cảm xúc tiêu cực/tích cực bằng cách map sang text hoặc giữ nguyên cho Pretrained model; chuẩn hóa Unicode sang NFC. | Kiểm tra xem emoji có tương quan cao với nhãn Toxic không qua ma trận tương quan ở bước EDA. |

---

## 2. Các Câu hỏi Cần Thảo luận với Giảng viên / Nhóm Đồ án (Open Questions)

### Q1: Mục tiêu nghiệp vụ của Demo Web là gì?
- **Lựa chọn 1**: Phục vụ người dùng thông thường tự kiểm tra xem nội dung của mình có chứa từ ngữ độc hại trước khi đăng bài (Content Moderation Assistant).
- **Lựa chọn 2**: Phục vụ Quản trị viên diễn đàn (Admin/Moderator) tải lên file CSV hàng loạt để lọc và kiểm duyệt bình luận tự động.
- *Đề xuất*: Hỗ trợ cả 2 chế độ thông qua 2 tab trên giao diện Streamlit.

### Q2: Phạm vi xử lý có bao gồm tiếng lóng (Slang) và từ ngữ teencode nhạy cảm không?
- Mạng xã hội tiếng Việt biến đổi liên tục với các cách viết lách né bộ lọc (vd: "đ-m", "clm", "d!t", "mẹ m").
- *Khuyến nghị*: Đồ án nên có một module từ điển teencode/slang chuẩn hóa nhẹ (`vietnamese_teencode.json`) được thu thập từ các nguồn mã nguồn mở uy tín để tăng độ bao phủ cho mô hình.

### Q3: Ngưỡng chấp nhận (Acceptance Threshold) về độ chính xác tối thiểu là bao nhiêu?
- Với bài toán dữ liệu mất cân bằng (imbalanced data), Accuracy là thước đo không đáng tin cậy.
- *Đề xuất*:
  - **Macro-F1 (Tập Test)**: Đạt tối thiểu **≥ 0.70** (với Baseline) và **≥ 0.80** (với Transformer SOTA).
  - **Latency**: < 500ms / 1 câu bình luận trên CPU.

---

## 3. Quy tắc Xác nhận Trạng thái (Status Transition Rules)

Mỗi mục trong danh sách này sẽ chỉ được chuyển trạng thái:
- Từ **[TBD]** sang **[Proposed]**: Khi đã có cơ sở lý thuyết và phương án thiết kế.
- Từ **[Proposed]** sang **[Verified]**: Khi đã thực hiện code EDA hoặc thực nghiệm kiểm chứng có lưu log kết quả.
- Tuyệt đối không xóa bỏ các mục đã từng [TBD] mà phải lưu vết trong `docs/decisions.md`.
