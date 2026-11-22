# TODO — TV1: Tam — EDA, preprocessing, TF-IDF + Logistic Regression (EXP-01)

## Kế hoạch hiện hành D1–D4

Các mục G0 bên dưới là lịch sử ba lớp; giữ trạng thái đã ghi và chạy lại các bước cần thiết trên split nhị phân mới. Không dùng kết quả G0 như kết quả của split mới.

- [ ] **N1:** Duyệt spec, decisions, plan và checklist cùng nhóm.
- [ ] **N2:** Hỗ trợ Huy xác minh nguồn, mapping ViHSD/ViCTSD, dedupe và thống kê theo split × source. Phụ thuộc N1; quyền UIT và cách xử lý khác biệt nhãn còn chờ xác nhận.
- [ ] **N6:** Sau N2, chạy TF-IDF + Logistic Regression trên split nhị phân với ba seed. Fit vectorizer trên train; lưu config, hash, MLflow run ID, model, validation predictions và metrics theo lớp và nguồn.
- [ ] **N6:** So sánh preprocessing trên validation, chọn checkpoint theo Macro-F1, báo F1 TOXIC; phối hợp calibration trên validation và bàn giao baseline `.joblib` dưới 15 MB.
- [ ] **N10:** Sau N9 và N14 nếu có, điều phối cả nhóm freeze candidates, seed, threshold và protocol; tạo file khóa và chạy một đợt locked test. Báo cáo `locked_test.md`, công bố lịch sử test exposure và không tuning theo test.
- [ ] Bàn giao validation predictions và lỗi LR cho Huy; lưu lại evidence theo từng run.

## Lịch sử G0 trước D1–D4


**Snapshot audit:** EDA notebook và EXP-01 A/B/C artifacts đã có, nhưng chúng dùng `data/processed/vihsd.jsonl` (run ghi 24,048 train / 2,672 val), không phải candidate G0 hiện hành `data/processed/vihsd_g0_v1/` (23,002 train / 4,929 val). Vì vậy đây là **đã chạy trên dữ liệu cũ; chưa nghiệm thu official EXP-01**. Không dùng kết quả cũ để so sánh chính thức. G0 hiện OPEN; có thể làm song song với TV2–TV4.

**Quy ước chung:** theo [Data Contract v1](../DATA_CONTRACT_v1.md), chỉ dùng `train_val.jsonl`; fit mọi bước có học trên train; không đọc `test_locked.jsonl` trước EXP-08. Ghi `Macro-F1` validation, per-class metrics, confusion matrix, thời gian/size, seed, hash và MLflow run ID; không khẳng định cải thiện nếu chưa có bằng chứng.

## P0 — nghiệm thu EXP-01 trên candidate G0

- [ ] **P0 — Chốt input/run contract của EXP-01.**
  - **Đầu ra:** config EXP-01 trỏ tới `data/processed/vihsd_g0_v1/train_val.jsonl`; ghi manifest/hash và số lượng split đã đọc.
  - **Nghiệm thu:** chỉ nạp train/val, đúng 23,002 train và 4,929 val; không mở `test_locked.jsonl`; mọi run dùng cùng contract/hash.
  - **Dependencies:** G0 OPEN; pipeline `src/models/lr_pipeline.py`. TV2–TV4 không chờ task này.

- [ ] **P0 — Cập nhật EDA nhãn và độ dài cho G0.**
  - **Đầu ra:** notebook EDA và biểu đồ phân phối nhãn, độ dài trên dữ liệu G0 train/val.
  - **Nghiệm thu:** thống kê tái lập được; ghi rõ split/phạm vi dùng; loại bỏ số liệu của `vihsd.jsonl` cũ khỏi kết luận hiện hành.
  - **Dependencies:** chốt input/run contract; không đọc locked test.

- [ ] **P0 — Cập nhật EDA pattern tiếng Việt và nhận xét preprocessing.**
  - **Đầu ra:** thống kê URL/emoji/teencode/ký tự chuẩn hóa và ghi chú ảnh hưởng tới các preprocessing variant.
  - **Nghiệm thu:** nhận xét có số liệu từ G0 train/val; không tự thay đổi text/schema/split của candidate.
  - **Dependencies:** chốt input/run contract; có thể chạy song song với TV2–TV4.

- [ ] **P0 — Chạy EXP-01-A: raw + word TF-IDF + Logistic Regression.**
  - **Đầu ra:** config, model/vectorizer, metrics, validation predictions, MLflow run ID.
  - **Nghiệm thu:** fit vectorizer trên train; log seed/hash, Macro-F1, per-class F1, confusion matrix và thời gian; artifact đọc lại được.
  - **Dependencies:** input/run contract; pipeline LR hiện có.

- [ ] **P0 — Chạy EXP-01-B: basic cleanup + word/char TF-IDF + Logistic Regression.**
  - **Đầu ra:** cùng artifact set như A cho variant B.
  - **Nghiệm thu:** chỉ thay đổi đã khai báo trong config; dùng cùng split/hash với A; metrics và predictions được lưu.
  - **Dependencies:** input/run contract; độc lập với A sau khi pipeline sẵn sàng.

- [ ] **P0 — Chạy EXP-01-C: teencode + word/char TF-IDF + class weighting.**
  - **Đầu ra:** cùng artifact set như A/B cho variant C.
  - **Nghiệm thu:** preprocessing/class weighting được ghi trong config; cùng split/hash; metrics và predictions được lưu.
  - **Dependencies:** input/run contract; độc lập với A/B sau khi pipeline sẵn sàng.

## P1 — phân tích và bàn giao

- [ ] **P1 — Xác minh tracking và tính truy vết của A/B/C.**
  - **Đầu ra:** run IDs, MLflow experiment/store và đường dẫn artifacts có thể mở lại; hoàn thiện metadata dataset/split hash.
  - **Nghiệm thu:** metrics/config/predictions truy về đúng run; nếu không truy cập được các run cũ thì ghi là historical, không hợp thức hóa bằng run ID đơn lẻ.
  - **Dependencies:** ba run G0 P0.

- [ ] **P1 — So sánh/ablation A/B/C và kết luận EXP-01.**
  - **Đầu ra:** bảng Macro-F1/per-class, confusion matrices, chi phí và kết luận có giới hạn.
  - **Nghiệm thu:** mọi con số từ run G0; so sánh đúng thay đổi đã thực hiện; notebook `EXP-01-LR-analysis.ipynb` không còn placeholder hoặc kết quả bản cũ.
  - **Dependencies:** metrics của A/B/C.

- [ ] **P1 — Bàn giao validation predictions/error examples cho TV3.**
  - **Đầu ra:** predictions có validation ID, nhãn thật/dự đoán, run ID/hash; một số lỗi LR đại diện.
  - **Nghiệm thu:** ID khớp validation G0, không có test rows; TV3 xác nhận format ghép được cho EXP-05.
  - **Dependencies:** A/B/C; TV3 có thể bắt đầu EXP-05 khi các nhánh khác lần lượt bàn giao.

## P2 — mở rộng

- [ ] **P2 — Đo độ biến thiên seed cho cấu hình LR được chọn nếu ngân sách cho phép.**
  - **Đầu ra:** run seed bổ sung và độ biến thiên validation metrics.
  - **Nghiệm thu:** cùng config/data hash; ghi seed; không chặn EXP-01 hoặc final evaluation.
  - **Dependencies:** A/B/C hoàn tất.

## Ghi chú audit

- `models/baselines/EXP-01-A/B/C/` có metrics, config, prediction và run ID; summary có số liệu validation. Tuy nhiên config hiện trỏ tới `vihsd.jsonl`, nên các số đó không phải kết quả trên candidate được duyệt.
- `notebooks/EXP-01-EDA.ipynb` ghi rõ EDA và leakage 318 text cross-split trên bản cũ; không tái dùng nhận định đó cho G0 mới. `docs/G0_FINAL_REVIEW.md` là nguồn hiện trạng G0; checklist/báo cáo TV1 cũ còn trạng thái pending là lịch sử.

## EDA N2/N4 (2026-11-23)

- Notebook `notebooks/EXP-EDA-N2-N4.ipynb` có bảng, biểu đồ, kết quả chạy và có thể chạy lại bằng Restart & Run All.
- N2 train có 28.231 mẫu, 15,93% TOXIC. N4 thêm 5.675 mẫu TOXIC vào train, tăng tỷ lệ TOXIC lên 30,00%; số CLEAN giữ nguyên 23.734. Validation và test mỗi split có 6.048 mẫu và khớp đầy đủ với N2.
- Cả 5.675 mẫu augmentation thuộc train và trỏ tới 2.872 parent trong N2 train. Không có overlap exact hoặc theo khóa chuẩn hóa với validation và test.
- Không có giá trị null hoặc chuỗi rỗng trên trường bắt buộc; `parent_id` null ở 28.231 dòng gốc N4 train theo thiết kế. Không có exact duplicate text. Có 364 nhóm khóa chuẩn hóa trùng trong split, không có overlap khóa chuẩn hóa giữa các split.
- Chưa kiểm chứng near-duplicate ngữ nghĩa tổng quát hoặc thread leakage. Test không blind. Không huấn luyện mô hình và không kết luận augmentation cải thiện mô hình.
