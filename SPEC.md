# Vietnamese Toxic Comment Detection — Project Specification

**Version:** 4.0.0 — SCRATCH-FIRST | **Date:** 2026-11-11 | **Status:** APPROVED DIRECTION / IMPLEMENTATION NOT STARTED

## Mục tiêu
Đồ án cuối kỳ Machine Learning: phân loại bình luận tiếng Việt thành **CLEAN (0), OFFENSIVE (1), HATE (2)**. Trọng tâm là **tự huấn luyện, so sánh, cải tiến và lý giải kết quả**; không lấy mô hình phân loại đã huấn luyện sẵn làm sản phẩm chính.

## Bốn hướng nghiên cứu chính (training from scratch)
| Thành viên | Mô hình chính | Thí nghiệm |
|---|---|---|
| TV1 | TF-IDF + Logistic Regression | EDA, preprocessing, word/char n-grams, class weighting |
| TV2 | TF-IDF + LinearSVC | C, n-grams, feature selection; RBF SVC chỉ khi khả thi |
| TV3 — nhóm trưởng | Random-initialized LSTM → BiLSTM → BiLSTM + Attention | Kiến trúc, hidden size, dropout, ablation; sở hữu data contract |
| TV4 | Random-initialized TextCNN → multi-kernel TextCNN | Kernel/filter/dropout, ablation; ensemble khi có cơ sở |

**Pretrained benchmark (riêng biệt):** TV3 fine-tune `vinai/phobert-base-v2` chỉ để **đối chứng Transfer Learning**, sau khi mô hình tự train đã có kết quả. Không coi PhoBERT là mô hình scratch. TV4 có thể hỗ trợ vận hành. Phần mở rộng, không là điều kiện chặn nghiệm thu bốn mô hình chính.

## Giao thức khoa học
- Dataset ưu tiên: UIT-ViHSD, phải xác minh schema, nguồn, giấy phép và split chính thức trước khi khóa. ViCTSD chỉ làm cross-dataset/OOD nếu taxonomy tương thích; không tự ghép nhãn.
- Data Contract do TV3 ban hành ở **Gate G0** sau kiểm tra tối thiểu (nhãn, trùng lặp, leakage, split). TV1 tiếp tục EDA chi tiết song song với TV2–TV4 viết code.
- Nếu official split phù hợp, ưu tiên giữ nguyên; nếu không, stratified 70/15/15, seed được ghi trong manifest. Chỉ fit tokenizer/vocabulary/TF-IDF và mọi tham số preprocessing có học trên train.
- **Macro-F1** là metric chọn mô hình. Báo cáo per-class precision/recall/F1, confusion matrix, training time, inference latency, model size. Validation dùng tuning; **locked test chỉ dùng ở giai đoạn cuối**, không tối ưu theo test.
- Tối thiểu **3 biến thể/cấu hình có ý nghĩa mỗi người**, ghi MLflow run/config/seed/dataset hash, lưu artifact và phân tích nguyên nhân kết quả; chạy nhiều seed cho các cấu hình quan trọng nếu tài nguyên cho phép.
- Ensemble chỉ thử khi dự đoán validation từ các mô hình chứng tỏ lỗi bổ sung; stacking phải dùng out-of-fold đúng quy tắc.
- Không bịa kết quả, không khẳng định mô hình mới tốt hơn trước khi đo. Mọi kết quả train/deploy hiện là **chưa thực hiện/chưa xác minh**.

## Phạm vi ứng dụng
Giữ **FastAPI tối giản** làm demo suy luận (`POST /predict`, `GET /health`, `GET /model-info`), Swagger `/docs`; không phát triển React/frontend, auth, database hay dashboard. Hosting/Hugging Face Spaces chỉ quyết định sau khi kiểm tra điều kiện thật.

## Tài liệu nguồn chuẩn
- [Đặc tả chi tiết](docs/spec.md)
- [Kế hoạch thí nghiệm](docs/plans/ml-experiments.md)
- [Kế hoạch giai đoạn và phối hợp](docs/plans/implementation-plan.md)
- [Task và nghiệm thu](docs/plans/task-breakdown.md)
- [Kế hoạch nhóm tóm tắt](tasks/plan.md)
- [Checklist](tasks/todo.md)

**Chỉ cập nhật SPEC/PLAN trong thay đổi này; tuyệt đối chưa sửa code, dataset, config, checkpoint hoặc triển khai.**
