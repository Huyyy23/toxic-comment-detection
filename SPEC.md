# Vietnamese Toxic Comment Detection System — Specification (SPEC.md)

Tài liệu đặc tả chi tiết và chuẩn mực: **[`docs/spec.md`](docs/spec.md)**  
Tài liệu quyết định kiến trúc: **[`docs/decisions.md`](docs/decisions.md)**  
Kế hoạch triển khai & phân rã: **[`tasks/plan.md`](tasks/plan.md)** và **[`tasks/todo.md`](tasks/todo.md)**

---

## Tóm tắt Quyết định & Giao thức Khoa học (Approved Principles)

1. **Giao thức Đánh giá (Evaluation Protocol)**:
   - **Validation Set**: Dùng để tuning siêu tham số, early stopping, và lựa chọn mô hình.
   - **Locked Test Set**: Khóa hoàn toàn trong quá trình phát triển; chỉ mở ra đánh giá đối kháng cuối cùng đúng một lần. Tuyệt đối không dùng kết quả test để tối ưu mô hình.
2. **Label Taxonomy**: Phân loại **3 lớp chuẩn**: `0: CLEAN`, `1: OFFENSIVE`, `2: HATE`. Không tự động chuyển về Binary.
3. **Hợp đồng Xác suất (Probability Contract)**:
   - Logistic Regression dùng `predict_proba`.
   - LinearSVC bắt buộc dùng `CalibratedClassifierCV` (Platt scaling) để sinh xác suất hiệu chuẩn thực sự.
   - Tổng xác suất 3 lớp $\approx 1.0$. Không giả mạo confidence.
4. **Siêu tham số Tạm thời (`[PROVISIONAL]`)**:
   - `max_length`, `batch_size`, learning rate, VRAM budget được gắn cờ tạm thời; chỉ chốt chính thức sau EDA và benchmark thực tế.
5. **Inference & Tách biệt Kiểm thử Tích hợp**:
   - `T4.2`: Baseline Integration Test (kiểm thử Baseline sớm, unblock Web MVP).
   - `T4.3`: Transformer Integration Test (kiểm thử Transformer sau huấn luyện trên RTX 4060).
   - `T4.4`: Conditional ONNX Benchmark (chỉ chuyển đổi khi có lợi ích thực tế).
6. **Quy trình Artifacts cho Docker Space**:
   - Version hóa qua Hugging Face Model Hub (pinned tag/revision); tự động fallback sang Baseline `.joblib` đóng gói sẵn nếu thiếu weights, ngăn ngừa container crash.
7. **Đánh giá Khách quan (Objective Acceptance)**:
   - Thay yêu cầu "Transformer bắt buộc vượt Baseline" bằng tiêu chí so sánh đối đầu khách quan, báo cáo trung thực số liệu và phân tích trade-offs.
