# Danh sách Câu hỏi Mở & Trạng thái Xác minh (Open Questions & Verification Ledger)

Trạng thái: **DRAFT — PENDING USER REVIEW**  
Ngày cập nhật: 2026-11-11  
Dự án: **Vietnamese Toxic Comment Detection**  
Phương pháp: **ML-first, API-minimal**

---

## 1. Các Quyết định Đã được Phê duyệt (Confirmed & Closed)

| ID | Vấn đề | Quyết định đã phê duyệt |
|---|---|---|
| **DD-01** | **Chiến lược Dataset** | Tải cả `ViCTSD` và `UIT-ViHSD` cho EDA so sánh. Ưu tiên `UIT-ViHSD` làm tập chính, chốt chính thức sau EDA (EXP-001). |
| **DD-02** | **Không gian Nhãn** | Sử dụng phân loại **3 lớp CLEAN / OFFENSIVE / HATE**. Không gộp về Binary. |
| **DD-03** | **Hướng nghiên cứu chính** | **Hướng 2** (baseline → Transformer → Ensemble nếu có bằng chứng). Hướng 1 giữ lại như đối sánh khách quan trên Test đã khóa. |
| **DD-04** | **Phần cứng Huấn luyện Transformer** | Ưu tiên GPU local RTX 4060 8GB; Colab/Kaggle T4 làm backup. |
| **DD-05** | **Kiến trúc Inference** | FastAPI REST API tối giản (3 endpoint). Không bắt buộc frontend. Swagger UI tại `/docs`. |
| **DD-06** | **Tính công bằng Thực nghiệm** | Traditional ML và Transformer dùng chung 1 tập test chuẩn, nhãn 3 lớp đồng nhất. |
| **DD-07** | **Tracking** | MLflow local là công cụ tracking chính. Optuna tùy chọn cho HPO. |
| **DD-08** | **Frontend** | Đã xóa hoàn toàn thư mục `frontend/` (React + Vite) trong phiên này. |
| **DD-09** | **Docker** | Giữ nếu hosting yêu cầu (HF Spaces Docker, Render Docker). Không bắt buộc trong phiên này. |
| **DD-10** | **Ensemble/Stacking** | Không bắt buộc. Chỉ thực hiện nếu cải thiện Macro-F1 ≥ 1% so với mô hình đơn tốt nhất; Soft Voting là phương án thay thế. |

---

## 2. Các Hạng mục Xác minh Kỹ thuật Cần Thực hiện (Pending)

### TV-01: Xác minh Điều kiện Triển khai Hugging Face (M6)
- **Mục tiêu**: Kiểm tra tài khoản HF có đủ điều kiện tạo Space miễn phí không bị chặn xác thực/thẻ tín dụng.
- **Tiêu chuẩn**: Có HF Token (quyền write); kiểm tra API endpoint HF Spaces.
- **Phương án dự phòng**: Render Free Docker, Railway, hoặc container Docker local.

### TV-02: Quyết định Phụ thuộc Dataset (M1, sau EXP-001)
- **Tỉ lệ phân bố 3 lớp**: Đếm số lượng và % `CLEAN` / `OFFENSIVE` / `HATE` trong `UIT-ViHSD`; tính `class_weight`.
- **Phân bố độ dài chuỗi**: Đo độ dài từ/ký tự ở phân vị 95% để chốt `max_length` (dự kiến 128 hoặc 192 để tiết kiệm VRAM).
- **Đặc trưng ngôn ngữ tiếng Việt**: Thống kê teencode, từ lóng, icon để xây dựng từ điển chuẩn hóa.

### TV-03: (ĐÃ QUYẾT ĐỊNH — DD-08) Xóa hoàn toàn frontend
- **Quyết định**: Loại bỏ hoàn toàn `frontend/`. Đã thực hiện xóa thư mục trong phiên này.
- **Hành động**: Không còn cần xóa code; chỉ cần xác nhận `pytest` vẫn pass sau khi xóa.

### TV-04: (ĐÃ QUYẾT ĐỊNH — DD-09) Docker giữ nếu hosting yêu cầu
- **Quyết định**: Không bắt buộc Docker. Giữ `Dockerfile`/`docker-compose.yml` chỉ nếu nền tảng hosting yêu cầu container.

### TV-05: (ĐÃ QUYẾT ĐỊNH — DD-07) MLflow local
- **Quyết định**: Sử dụng MLflow local (file store). Không triển khai MLflow server trong phiên này.

### TV-06: (ĐÃ QUYẾT ĐỊNH — DD-10) Ensemble/Stacking không bắt buộc
- **Quyết định**: Không bắt buộc. Chỉ thực hiện nếu có bằng chứng thực nghiệm. Soft Voting là phương án thay thế; nếu không cải thiện, giữ mô hình đơn tốt nhất.
