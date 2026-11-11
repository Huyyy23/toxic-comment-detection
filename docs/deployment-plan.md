# Kế hoạch Triển khai & Ràng buộc Tài nguyên (Deployment Plan)

Trạng thái: **APPROVED & SYNCHRONIZED**  
Ngày cập nhật: 2026-11-11  
Dự án: **Vietnamese Toxic Comment Detection System**

---

## 1. Kiến trúc Hệ thống Triển khai (System Architecture)

Hệ thống được đóng gói thành một Docker Container độc lập, kết hợp **React (Frontend)** và **FastAPI (Backend)** chạy trên cùng một port `7860` (cổng tiêu chuẩn của Hugging Face Spaces).

- Chi tiết kế hoạch triển khai, Dockerfile đa tầng và kịch bản nạp artifact được quy định tại: **[`docs/plans/deployment.md`](plans/deployment.md)**.

---

## 2. Ngân sách Tài nguyên & Giới hạn Hạ tầng (Dựa trên Đo đạc Thực tế)

| Tham số hạ tầng | Ngưỡng dự kiến (Budget) | Giải pháp kiểm soát |
| :--- | :--- | :--- |
| **Dung lượng Artifacts** | PyTorch (~500MB) hoặc ONNX (~150MB) | Quản lý version hóa qua Hugging Face Model Hub (`v1.0.0`), nạp qua `src/tools/download_artifacts.py`. |
| **Dung lượng RAM** | Quota 16 GB RAM của HF Spaces | Cần đo đạc bộ nhớ thực tế (memory profiling) khi nạp model; fallback sang baseline nếu vượt hạn mức. |
| **Thời gian khởi động** | Cần đo lường thực tế lúc deploy | Khởi tạo model một lần duy nhất qua FastAPI lifespan event; không train hoặc download nặng lúc runtime nếu đã có local cache. |
| **Độ trễ suy luận** | Cần benchmark thực tế trên CPU | Hỗ trợ đa backend; chỉ bật ONNX nếu benchmark chứng minh giảm độ trễ $\ge 25\%$. |
| **Chi phí hạ tầng** | 0 VNĐ | Triển khai trên Hugging Face Spaces (CPU Basic) hoặc Render/Local Docker nếu vướng thẻ tín dụng. |

---

## 3. Xác minh Điều kiện Triển khai Hugging Face (M0 Status)

- **Trạng thái**: Báo cáo tại `docs/hf_eligibility_report.md` hiện ghi nhận **`[ATTENTION / PENDING TOKEN]`**.
- **Quy tắc**: Không tự tiện đánh dấu triển khai thành công khi chưa có `HF_TOKEN` và chưa xác thực quyền tạo Docker Space thực tế.
