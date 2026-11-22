# Báo cáo Quyết định Kiến trúc & Phân tích Đánh đổi (Architecture Decisions & Trade-offs)

Trạng thái: **APPROVED — D1–D4 được cả nhóm duyệt, 2026-11-23**  
Ngày cập nhật: 2026-11-23  
Dự án: **Vietnamese Toxic Comment Detection**  
Phương pháp: **ML-first, API-minimal, reproducibility-first**

---

## ADR-01: Hướng nghiên cứu chính — Hướng 2 (Model A → B → C)

**SUPERSEDED by D2, 2026-11-23:** Transformer là đối chứng pretrained có điều kiện, không phải bước cải tiến bắt buộc.

- **Quyết định**: Chọn **Hướng 2** (baseline A → đánh giá hạn chế → cải thiện thành Transformer B → cải thiện thành Ensemble/Stacking C nếu có bằng chứng) làm phương pháp nghiên cứu chính.
- **Lý do**: Phù hợp mục tiêu chứng minh quá trình cải tiến, khuyến khích error analysis, dễ建构 câu chuyện nghiên cứu xuyên suốt báo cáo, có thể kết hợp traditional ML → Transformer → Ensemble.
- **Hướng 1** (thiết kế N mô hình, so sánh, chọn tốt nhất) được giữ lại như **khía cạnh đối sánh khách quan trên tập test đã khóa**, không làm hai quy trình training độc lập.
- **Ràng buộc**: Không tự khẳng định mô hình sau tốt hơn trước khi có kết quả thực nghiệm. Mỗi vòng có giả thuyết, baseline, metric và tiêu chí quyết định rõ ràng.

---

## ADR-02: Phân loại Nhãn & Giao thức Đánh giá Khoa học

**SUPERSEDED by D1, 2026-11-23:** Quyết định ba lớp dưới đây được giữ làm lịch sử; contract mới là nhị phân.

- **Không gian Nhãn**: **3 lớp CLEAN (0) / OFFENSIVE (1) / HATE (2)**. Tuyệt đối không tự động gộp về nhị phân trừ khi có lý do kỹ thuật và bằng chứng rõ ràng, ghi nhận trong SPEC.
- **Nguyên tắc Khóa Tập Test**:
  - **Train (70%)**: Huấn luyện trọng số.
  - **Validation (15%)**: Tuning hyperparameters, early stopping, model selection, calibration.
  - **Test (15%)**: **Bị khóa hoàn toàn**; chỉ mở đúng một lần tại EXP-009 để đánh giá cuối cùng. Tuyệt đối không dùng kết quả Test để tối ưu ngược lại mô hình.
- **Tiêu chí So sánh Khách quan**: Không đặt yêu cầu Transformer bắt buộc vượt Baseline; báo cáo trung tính toàn bộ chỉ số và phân tích trade-offs.

---

## ADR-03: Hợp đồng Xác suất & Hiệu chuẩn Mô hình

**SUPERSEDED một phần by D3, 2026-11-23:** Calibration và threshold được chọn trên validation theo D3; không giả định confidence luôn là xác suất lớn nhất khi dùng threshold riêng.

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

**SUPERSEDED by D2, 2026-11-23:** Kế hoạch mới khôi phục React + FastAPI; không khẳng định frontend đã được triển khai.

- **Quyết định**: Production inference chỉ cần **FastAPI REST API** với 3 endpoint:
  - `POST /predict` — nhận văn bản, trả về nhãn + xác suất hiệu chuẩn + latency.
  - `GET /health` — kiểm tra model_loaded, backend, artifact status, device.
  - `GET /model-info` — metadata mô hình, label taxonomy, calibration method, artifact version.
- Swagger UI tại `/docs` dùng để kiểm thử và demo.
- **Không bắt buộc frontend riêng.** Không React, không Vite, không CSS/UI framework, không dashboard, không auth/JWT, không user database.

---

## ADR-06: Xóa hoàn toàn frontend (đã thực hiện)

**SUPERSEDED by D2, 2026-11-23:** Việc xóa là lịch sử; kế hoạch mới yêu cầu lại frontend.

- **Quyết định**: Đã xóa hoàn toàn thư mục `frontend/` (React + Vite) khỏi dự án. Dự án không còn phát triển giao diện web kiểm duyệt bình luận.
- **Lý do**: Dự án được định hướng lại thành đồ án nghiên cứu ML, không phải phát triển full-stack. Frontend không phục vụ mục tiêu ML-first và tạo ra độ phức tạp không cần thiết.
- **Hành động trong phiên này**: Đã xóa thư mục `frontend/` và các file liên quan (`package.json`). Đã cập nhật `setup.cfg`, `tests/test_smoke.py`, `.gitignore` để loại bỏ tham chiếu đến frontend.
- **Rủi ro**: Nếu inference engine hoặc API có phụ thuộc ngầm vào frontend (ví dụ: serve static file), phải tách biệt rõ ràng. Smoke test đã pass sau khi xóa.

---

## ADR-07: Docker — giữ nếu hosting yêu cầu (không bắt buộc)

**SUPERSEDED by D2, 2026-11-23:** Kế hoạch triển khai mới dùng Docker và HF Spaces.

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

---

## D1: Gộp dữ liệu và phân loại nhị phân

- **Bối cảnh**: Dữ liệu ViHSD G0 mất cân bằng; OFFENSIVE và CLEAN có ranh giới khó phân biệt. ViCTSD dùng nhãn Toxicity nhị phân. G0 ghi nhận 23.002 mẫu train sau xử lý conflict và chia lại, không phải số train chính thức của nguồn; không suy diễn nguyên nhân thành dedupe đơn thuần.
- **Quyết định**: ViHSD `label_id == 0` thành CLEAN (0), `label_id > 0` thành TOXIC (1). ViCTSD dùng `Toxicity`, không đưa `Constructiveness`, `Title`, `Topic` vào huấn luyện. Lưu `source` là `vihsd` hoặc `victsd`; dedupe trong và giữa nguồn trước split 70/15/15 stratify theo `(source, label)`.
- **Lý do**: Giảm độ khó phân biệt ba lớp mất cân bằng và thống nhất nhãn đầu ra của hai nguồn. Hiệu quả phải được kiểm chứng, không mặc định đã cải thiện.
- **Hệ quả**: Cần nhóm xác nhận khác biệt định nghĩa Toxicity với OFFENSIVE/HATE. Văn phong báo và mạng xã hội có thể tạo shortcut theo nguồn; báo cáo metric theo `source`. Quyền dùng UIT chưa xác minh. G0 cũ và lịch sử test exposure phải được giữ và công bố; split mới chưa tồn tại hoặc chưa được kiểm chứng.
- **Ngày**: 2026-11-23.
- **Trạng thái**: Accepted.

## D2: Nhóm tự huấn luyện và đối chứng PhoBERT

- **Bối cảnh**: Giảng viên yêu cầu sinh viên tự huấn luyện. Người dùng xác nhận ngày 2026-11-23 rằng PhoBERT đã được chấp nhận.
- **Quyết định**: Nhóm chính gồm TF-IDF + LR/LinearSVC, LSTM, BiLSTM, BiLSTM + Attention và nhánh Char-CNN; FastText tự train bằng gensim chỉ trên train. Không dùng pretrained weights, pretrained embeddings, back-translation hoặc paraphrase/LLM sinh dữ liệu. PhoBERT fine-tune từ pretrained là benchmark riêng, không thay nhóm chính.
- **Lý do**: Tách kết quả tự huấn luyện khỏi đối chứng khởi tạo pretrained để tránh nhầm khi chấm điểm.
- **Hệ quả**: PhoBERT pretrained fine-tune đã được chấp nhận theo xác nhận người dùng. Báo cáo tách bảng hoặc gắn nhãn rõ. Kế hoạch tích hợp React + FastAPI, Docker và HF Spaces; đây là thay đổi kế hoạch, không phải bằng chứng triển khai. ONNX T4.4 chỉ xem xét nếu có lợi.
- **Ngày**: 2026-11-23.
- **Trạng thái**: Accepted; PhoBERT đã được chấp nhận theo xác nhận người dùng.

## D3: Mất cân bằng và calibration

- **Bối cảnh**: G0 cho thấy phân phối lớp không đều; nguyên nhân cụ thể của lỗi dự đoán và các số liệu từ phân tích trước cần kiểm chứng trên artifacts hợp lệ.
- **Quyết định**: Augment chỉ TOXIC của train, mặc định thêm một bản và tối đa hai bản mỗi mẫu. Mục tiêu 30–35% TOXIC là mục tiêu, không bảo đảm đạt được dưới giới hạn này. Dùng deletion/swap, nhiễu dấu hoặc ký tự và từ điển teencode hai chiều; bảo vệ token độc hại bằng lexicon dựng từ train. Loại bản trùng val/test bằng kiểm tra trùng lặp, không xem nội dung test để tuning. Chọn đúng một trong class weights, focal loss, WeightedRandomSampler theo validation, không chồng các phương án.
- **Lý do**: Xử lý mất cân bằng có thể làm xác suất lệch; temperature scaling hoặc Platt trên val giúp đánh giá và hiệu chuẩn xác suất, không bảo đảm calibration hoàn hảo.
- **Hệ quả**: Chọn checkpoint thống nhất bằng val Macro-F1; báo thêm precision, recall, F1 từng lớp, PR-AUC, F1 TOXIC và metric theo nguồn. Chọn threshold tối ưu F1 TOXIC trên val, báo ECE trước/sau. Lưu threshold và trạng thái calibration; xác suất hai lớp cộng xấp xỉ 1.
- **Ngày**: 2026-11-23.
- **Trạng thái**: Accepted.

## D4: Đối chứng PhoBERT và giao thức so sánh

- **Bối cảnh**: Cần so sánh công bằng nhóm tự huấn luyện với PhoBERT pretrained fine-tune đã được chấp nhận.
- **Quyết định**: Dùng cùng split nhị phân, metric, nguyên tắc xử lý mất cân bằng và một đợt test khóa cho LR, SVM, LSTM, BiLSTM, BiLSTM + Attention và PhoBERT. Chạy ba seed mỗi mô hình, báo trung bình ± độ lệch chuẩn, bootstrap CI hoặc McNemar, số tham số, thời gian train, latency và kích thước.
- **Lý do**: Tách biến nghiên cứu và công bố độ bất định thay vì suy ra cải thiện từ một run.
- **Hệ quả**: PhoBERT pretrained fine-tune đã được chấp nhận; cấu hình phobert-base là dự kiến, tách từ có ghi công cụ/phiên bản, max length 128–160, tối đa khoảng 256, lr 2e-5, batch 16, warmup 10%, weight decay 0,01, 3–4 epoch, early stopping theo val Macro-F1; fp16 và accumulation khi phù hợp thiết bị. Thử bổ sung chỉ trên val và đổi một yếu tố mỗi lần. Sau khi đóng băng mọi candidate và seed, mở test một đợt có file khóa; không chọn lại theo test. Phân tích lỗi đối chiếu hai chiều, theo nguồn, độ dài, teencode và nhóm lỗi, chỉ kết luận về ngữ cảnh khi có ví dụ và đủ mẫu; ghi nhận nhãn mơ hồ.
- **Ngày**: 2026-11-23.
- **Trạng thái**: Accepted; PhoBERT đã được chấp nhận theo xác nhận người dùng.
