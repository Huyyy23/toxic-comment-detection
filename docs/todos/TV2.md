# TODO — TV2: LinearSVC, feature engineering, tuning, ablation (EXP-02)

**Snapshot audit:** chưa thấy pipeline/config/run artifacts riêng cho LinearSVC hoặc kết quả EXP-02. `src/features/tfidf_pipeline.py` là hạ tầng TF-IDF dùng chung, không phải bằng chứng đã huấn luyện SVM. **EXP-02 chưa bắt đầu/nghiệm thu.** G0 hiện OPEN, nên TV2 có thể làm song song ngay với TV1, TV3, TV4.

**Quy ước chung:** dùng [Data Contract v1](../DATA_CONTRACT_v1.md), dataset `data/processed/vihsd_g0_v1/`, chỉ `train_val.jsonl`; Macro-F1 validation là metric chọn mô hình; fit vectorizer/feature selection/calibration chỉ trên train hoặc train-fold; tuyệt đối không biến `LinearSVC.decision_function` thành xác suất.

## P0 — nghiệm thu EXP-02

- [ ] **P0 — Chốt LinearSVC config và input G0.**
  - **Đầu ra:** config versioned cho dataset/split, TF-IDF, C, class weight, feature selection và seed.
  - **Nghiệm thu:** trỏ `data/processed/vihsd_g0_v1/train_val.jsonl`; chỉ đọc train/val; ghi hash; không mở locked test.
  - **Dependencies:** G0 OPEN; EDA chi tiết TV1 không phải dependency.

- [ ] **P0 — Hoàn thiện LinearSVC train/evaluate entrypoint.**
  - **Đầu ra:** pipeline dùng TF-IDF utility, train/evaluate trên contract chung và lưu model/metrics/predictions.
  - **Nghiệm thu:** fixture smoke chạy hết train→validation; vectorizer/feature selector fit train-only; decision scores không được gắn nhãn probability.
  - **Dependencies:** config task trước; `src/features/tfidf_pipeline.py` có sẵn.

- [ ] **P0 — Chạy EXP-02-A: word TF-IDF + LinearSVC baseline.**
  - **Đầu ra:** config, model, metrics, validation predictions, MLflow run ID.
  - **Nghiệm thu:** Macro-F1/per-class metrics/confusion, thời gian và model size được lưu cùng seed/hash; artifact mở lại được.
  - **Dependencies:** entrypoint P0; G0. Độc lập với EXP-01.

- [ ] **P0 — Chạy EXP-02-B: word+char TF-IDF + LinearSVC.**
  - **Đầu ra:** artifact set cho variant B trên cùng split.
  - **Nghiệm thu:** n-grams khai báo trong config; cùng data hash với A; metrics/predictions truy về đúng run.
  - **Dependencies:** entrypoint P0; sau A để so sánh có baseline.

- [ ] **P0 — Chạy EXP-02-C: feature selection/tuning C và class weight.**
  - **Đầu ra:** config và kết quả variant C; lưu lựa chọn tham số, metrics và predictions.
  - **Nghiệm thu:** feature selection/C/class weight chỉ chọn theo validation; mọi bước học fit trên train hoặc train-fold; ghi chi phí và kết quả dù không cải thiện.
  - **Dependencies:** A/B baseline; validation G0.

## P1 — phân tích và bàn giao

- [ ] **P1 — Tổng hợp tuning/ablation LinearSVC.**
  - **Đầu ra:** bảng A/B/C so sánh Macro-F1/per-class, confusion, C/n-grams/feature selection, thời gian và kích thước model.
  - **Nghiệm thu:** kết luận khớp configs/runs, không dùng test để chọn; nêu giới hạn thiết kế.
  - **Dependencies:** A/B/C hoàn tất.

- [ ] **P1 — Kiểm tra run lineage và predictions.**
  - **Đầu ra:** MLflow run IDs/store, data hash, prediction artifact có ID/true/predicted label/run ID.
  - **Nghiệm thu:** load lại được artifacts; IDs trùng validation G0; không có test rows; không gọi raw margins là probability.
  - **Dependencies:** A/B/C.

- [ ] **P1 — Bàn giao lỗi SVM cho EXP-05 và metadata cho EXP-06.**
  - **Đầu ra:** phân nhóm lỗi đại diện và xác định outputs có thể ghép; nếu cần voting xác suất, đề xuất calibrator.
  - **Nghiệm thu:** validation-only; calibration chỉ fit trong train CV; TV3/TV4 xác nhận predictions ghép được.
  - **Dependencies:** prediction artifacts; TV3/TV4 có thể nhận từng nhánh độc lập.

## P2 — tùy chọn

- [ ] **P2 — Chạy RBF SVC hoặc ghi rõ lý do skip.**
  - **Đầu ra:** run riêng kèm compute/memory/time hoặc ghi quyết định skip.
  - **Nghiệm thu:** không trì hoãn LinearSVC; cùng validation G0; calibration đúng nếu xuất probabilities.
  - **Dependencies:** EXP-02 A/B/C; tài nguyên cho phép.
