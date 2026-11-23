**Cập nhật 2026-11-22:** G0 ba lớp đã được duyệt và OPEN theo `docs/G0_FINAL_REVIEW.md`; giữ nguyên dataset legacy/raw và locked test. G0 không phải dataset nhị phân N2.

**Cập nhật 2026-11-23:** Người dùng xác nhận TV1 = Tam, TV2 = Vinh, TV3 = Huy, TV4 = TheAnh. Checklist D1–D4 của từng thành viên nằm trong `docs/todos/`; quyền UIT và mapping nhãn giữa nguồn còn chờ xác nhận.

**Cập nhật 2026-11-23:** Người dùng phân N3, N4, N5, N8, N9, N11 và N12 cho Huy (TV3). Xem `docs/todos/TV3.md` và kế hoạch N1–N15 để biết phụ thuộc.

**Cập nhật 2026-11-23:** Người dùng xác nhận PhoBERT pretrained fine-tune đã được chấp nhận. PhoBERT vẫn là benchmark tách biệt với nhóm tự huấn luyện; chưa có xác nhận rằng thí nghiệm đã chạy.

**Cập nhật 2026-11-23:** N1 hoàn thành theo xác nhận người dùng: Tam, Vinh, Huy và TheAnh đã duyệt D1–D4. N2 đã được mở khóa; quyền UIT và mapping nhãn vẫn chờ xác minh.

**Cập nhật 2026-11-23:** N2 đã tạo và kiểm tra cơ bản tại `data/processed/binary_n2_v1/`; N3 tiếp tục audit leakage. Người dùng xác nhận nhóm đã xin phép UIT cho đồ án, nhưng repo chưa có bằng chứng để kiểm chứng giấy phép. Xem `docs/dataset-requirements.md` về mapping và giới hạn test.

**Cập nhật 2026-11-23:** N3 hoàn thành; N4 tạo augmentation TOXIC train-only và chạy leakage tests trên artifact thật (10 passed). Artifact, hash và giới hạn tại `docs/dataset-requirements.md`; N5 tiếp tục theo plan.

**Cập nhật 2026-11-23:** N5 hoàn thành trên N2 train-only với seeds 42, 123, 2026. Validation Macro-F1 mean 0.622218, population std 0.001276. Artifacts, hashes và MLflow run IDs tại `models/baselines/N5-run-4cef5024bbdb437296384e56b471e20b/`; không dùng N4 hoặc mở test set. Nguồn: `summary.json` và per-seed artifacts.

**Cập nhật 2026-11-23:** N5 tuning hoàn tất với seeds 42, 123, 2026 trên N2 train/validation. Logistic Regression không trọng số với ngưỡng TOXIC chọn trên validation đạt F1 TOXIC mean 0.454134, population std 0.001391, tăng 0.018243 so baseline 0.435891 nhưng chưa đạt mục tiêu 0.55. Macro-F1 mean 0.658345; F1 CLEAN mean 0.862556. Candidate có Macro-F1 cao nhất là Logistic Regression balanced với ngưỡng validation, mean 0.661476; F1 TOXIC mean 0.451987. Artifacts và MLflow summary run `86516c6788154b5f9a8628332b43e4c8` tại `models/baselines/N5-toxic-tuning-run-d15f90a55f944793b61a49dfcdb9360b/`. Threshold được chọn và đánh giá trên cùng validation nên điểm tuned có bias lạc quan. Hash N2 và artifacts baseline không đổi; không dùng N4 hoặc test. Nguồn: `summary.json`, per-seed artifacts và MLflow.
