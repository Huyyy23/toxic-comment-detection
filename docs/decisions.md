# Báo cáo Quyết định Kiến trúc & Phân tích Đánh đổi (Architecture Decisions & Trade-offs)

Trạng thái: **DRAFT — PENDING USER REVIEW**  
Ngày cập nhật: 2026-11-11  
Dự án: **Vietnamese Toxic Comment Detection**  
Phương pháp: **ML-first, API-minimal, reproducibility-first**

---

## ADR-01: Hướng nghiên cứu chính — Hướng 2 (Model A → B → C)

- **Quyết định**: Chọn **Hướng 2** (baseline A → đánh giá hạn chế → cải thiện thành Transformer B → cải thiện thành Ensemble/Stacking C nếu có bằng chứng) làm phương pháp nghiên cứu chính.
- **Lý do**: Phù hợp mục tiêu chứng minh quá trình cải tiến, khuyến khích error analysis, dễ建构 câu chuyện nghiên cứu xuyên suốt báo cáo, có thể kết hợp traditional ML → Transformer → Ensemble.
- **Hướng 1** (thiết kế N mô hình, so sánh, chọn tốt nhất) được giữ lại như **khía cạnh đối sánh khách quan trên tập test đã khóa**, không làm hai quy trình training độc lập.
- **Ràng buộc**: Không tự khẳng định mô hình sau tốt hơn trước khi có kết quả thực nghiệm. Mỗi vòng có giả thuyết, baseline, metric và tiêu chí quyết định rõ ràng.

---

## ADR-02: Phân loại Nhãn & Giao thức Đánh giá Khoa học

- **Không gian Nhãn**: **3 lớp CLEAN (0) / OFFENSIVE (1) / HATE (2)**. Tuyệt đối không tự động gộp về nhị phân trừ khi có lý do kỹ thuật và bằng chứng rõ ràng, ghi nhận trong SPEC.
- **Nguyên tắc Khóa Tập Test**:
  - **Train (70%)**: Huấn luyện trọng số.
  - **Validation (15%)**: Tuning hyperparameters, early stopping, model selection, calibration.
  - **Test (15%)**: **Bị khóa hoàn toàn**; chỉ mở đúng một lần tại EXP-009 để đánh giá cuối cùng. Tuyệt đối không dùng kết quả Test để tối ưu ngược lại mô hình.
- **Tiêu chí So sánh Khách quan**: Không đặt yêu cầu Transformer bắt buộc vượt Baseline; báo cáo trung tính toàn bộ chỉ số và phân tích trade-offs.

---

## ADR-03: Hợp đồng Xác suất & Hiệu chuẩn Mô hình

- **Vấn đề**: `LogisticRegression` hỗ trợ `.predict_proba()` tự nhiên. `LinearSVC` chỉ sinh `decision_function()` (uncalibrated margins), không phải xác suất.
- **Quyết định**:
  - Không tự viết softmax trên decision scores để giả tạo confidence.
  - Khi dùng LinearSVC, bắt buộc `CalibratedClassifierCV(estimator=LinearSVC(...), cv=5, method='sigmoid')` (Platt scaling).
  - Hợp đồng API: $\sum P(\text{class}) \approx 1.0$, `confidence` là xác suất của lớp chiến thắng.

---

## ADR-04: Siêu tham số `[PROVISIONAL]`

- Mọi giá trị trong `configs/` (`max_length=160`, `batch_size=16`, `learning_rate=2e-5`, `VRAM budget < 7.0GB`, TF-IDF `max_features=10000`) là tạm thời.
- Chốt chính thức sau EDA (M1) và hardware profiling thực tế.

---

## ADR-05: Kiến trúc Inference — FastAPI tối giản (3 endpoint), không frontend

- **Quyết định**: Production inference chỉ cần **FastAPI REST API** với 3 endpoint:
  - `POST /predict` — nhận văn bản, trả về nhãn + xác suất hiệu chuẩn + latency.
  - `GET /health` — kiểm tra model_loaded, backend, artifact status, device.
  - `GET /model-info` — metadata mô hình, label taxonomy, calibration method, artifact version.
- Swagger UI tại `/docs` dùng để kiểm thử và demo.
- **Không bắt buộc frontend riêng.** Không React, không Vite, không CSS/UI framework, không dashboard, không auth/JWT, không user database.

---

## ADR-06: Xóa hoàn toàn frontend (đã thực hiện)

- **Quyết định**: Đã xóa hoàn toàn thư mục `frontend/` (React + Vite) khỏi dự án. Dự án không còn phát triển giao diện web kiểm duyệt bình luận.
- **Lý do**: Dự án được định hướng lại thành đồ án nghiên cứu ML, không phải phát triển full-stack. Frontend không phục vụ mục tiêu ML-first và tạo ra độ phức tạp không cần thiết.
- **Hành động trong phiên này**: Đã xóa thư mục `frontend/` và các file liên quan (`package.json`). Đã cập nhật `setup.cfg`, `tests/test_smoke.py`, `.gitignore` để loại bỏ tham chiếu đến frontend.
- **Rủi ro**: Nếu inference engine hoặc API có phụ thuộc ngầm vào frontend (ví dụ: serve static file), phải tách biệt rõ ràng. Smoke test đã pass sau khi xóa.

---

## ADR-07: Docker — giữ nếu hosting yêu cầu (không bắt buộc)

- **Quyết định**: `Dockerfile` và `docker-compose.yml` **không bắt buộc** trong phiên này. Giữ lại chỉ nếu nền tảng hosting yêu cầu container (ví dụ Hugging Face Spaces Docker SDK, Render Docker).
- **Lý do**: FastAPI có thể chạy trực tiếp local mà không cần Docker. Việc bắt buộc Docker tạo ra độ phức tạp triển khai không cần thiết cho giai đoạn nghiên cứu.
- **Phương án triển khai**: FastAPI chạy local / Uvicorn, hoặc Docker nếu hosting yêu cầu. Kiến trúc API không đổi giữa hai trường hợp.

---

## ADR-08: MLflow local

- **Quyết định**: Sử dụng **MLflow local** (file store) làm công cụ tracking chính.
- **Lý do**: Đơn giản, không cần triển khai và duy trì MLflow server riêng. Có thể nâng lên remote server sau nếu nhóm nhiều máy cần tổng hợp runs.
- **Mỗi run lưu**: Experiment ID, Run ID, dataset version/hash, split config, model architecture, hyperparameters, random seed, training duration, train/val metrics, environment metadata, checkpoint/artifacts, git commit (nếu có).

---

## ADR-09: Ensemble/Stacking không bắt buộc (conditional)

- **Quyết định**: Ensemble / Stacking **không bắt buộc**. Chỉ thực hiện EXP-007 nếu có bằng chứng thực nghiệm cho thấy base models có lỗi bổ sung cho nhau và ensemble cải thiện Macro-F1 ≥ 1% so với mô hình đơn tốt nhất trên Validation.
- **Phương án thay thế**: Soft Voting (nếu Stacking quá tốn tài nguyên). Nếu ensemble không mang lại lợi ích, giữ mô hình đơn tốt nhất.
- **Ràng buộc**: Meta-learner không học từ Test set; out-of-fold predictions hợp lệ; không xem decision score của LinearSVC là xác suất.
