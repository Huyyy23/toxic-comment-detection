# Vietnamese Toxic Comment Detection — Specification (SPEC.md)

Tài liệu đặc tả chi tiết: [`docs/spec.md`](docs/spec.md)  
Kế hoạch thực nghiệm: [`docs/plans/ml-experiments.md`](docs/plans/ml-experiments.md)  
Kế hoạch triển khai: [`docs/plans/implementation-plan.md`](docs/plans/implementation-plan.md)  
Phân rã task chi tiết: [`docs/plans/task-breakdown.md`](docs/plans/task-breakdown.md)  
Quyết định kiến trúc: [`docs/decisions.md`](docs/decisions.md)  
Kế hoạch triển khai API tối giản: [`docs/deployment-plan.md`](docs/deployment-plan.md)  
Yêu cầu dữ liệu: [`docs/dataset-requirements.md`](docs/dataset-requirements.md)

---

## Tóm tắt

Đồ án **Vietnamese Toxic Comment Detection** được định hướng lại thành một **đồ án nghiên cứu Machine Learning có độ khó cao**, theo nguyên tắc **ML-first, API-minimal, reproducibility-first**.

- **Không phải** ứng dụng web kiểm duyệt bình luận full-stack.
- **Là** quá trình xây dựng, huấn luyện, so sánh, tối ưu và cải thiện nhiều mô hình phân loại văn bản tiếng Việt (3 lớp: CLEAN / OFFENSIVE / HATE), có bằng chứng thực nghiệm, ghi log đầy đủ qua **MLflow local**, và triển khai bằng **FastAPI REST API tối giản** (3 endpoint: `POST /predict`, `GET /health`, `GET /model-info`).

**Phương pháp nghiên cứu chính thức:** **Hướng 2** — Xây dựng mô hình A (baseline), đánh giá hạn chế, cải thiện thành mô hình B (Transformer), tiếp tục C (Ensemble/Stacking) **không bắt buộc**, chỉ nếu có bằng chứng thực nghiệm. Không tự khẳng định mô hình sau tốt hơn trước khi có kết quả thực nghiệm.

**4 quyết định đã khóa và đã thực hiện trong phiên này:**
1. **Xóa hoàn toàn `frontend/` (React + Vite)** — đã xóa thư mục, cập nhật `setup.cfg`, `tests/test_smoke.py`, `.gitignore`.
2. **Docker giữ nếu hosting yêu cầu** — không bắt buộc; FastAPI có thể chạy local trực tiếp.
3. **MLflow local** — đã thiết lập `src/tracking/mlflow_tracker.py` (file store), chưa cài package `mlflow` (sẽ cài ở phiên implementation).
4. **Ensemble/Stacking không bắt buộc** — chỉ thực hiện nếu cải thiện Macro-F1 ≥ 1% so với mô hình đơn tốt nhất; Soft Voting là phương án thay thế.

Phiên này đã thực hiện: xóa `frontend/`, thiết lập MLflow local code. Không train, không deploy.
