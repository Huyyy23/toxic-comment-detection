# ĐẶC TẢ DỰ ÁN — VIETNAMESE TOXIC COMMENT DETECTION

**Version:** 4.0.0-SCRATCH-FIRST · **Ngày:** 2026-11-11 · **Trạng thái:** Đã chốt hướng nghiên cứu, chưa triển khai.

## 1. Bài toán và mục tiêu
Phân loại bình luận tiếng Việt thành **0 CLEAN**, **1 OFFENSIVE**, **2 HATE**. Đồ án tập trung vào học máy thực nghiệm: thiết kế pipeline, **train từ đầu bốn họ mô hình**, tuning, phân tích sai, ablation, đối sánh công bằng. Không phải dự án full-stack.

### Phân biệt tự train và pretrained
- **Scratch models (bắt buộc):** Logistic Regression, LinearSVC, LSTM/BiLSTM/Attention, TextCNN; không khởi tạo bằng pretrained weights hoặc pretrained embeddings. Được sử dụng scikit-learn/PyTorch để cài đặt và huấn luyện.
- **Transfer-learning control (tùy chọn, không thay thế):** PhoBERT pretrained base được **fine-tune trên train set chung**, rồi so sánh chi phí/lợi ích. Không nhận PhoBERT là scratch và không tải model đã fine-tune sẵn cho toxic classification.
- Chỉ thử RBF SVC/ensemble nếu có căn cứ và đủ tài nguyên.

## 2. Định nghĩa nhãn và đánh giá
| ID | Nhãn | Ý nghĩa |
|---|---|---|
| 0 | CLEAN | Bình luận không độc hại |
| 1 | OFFENSIVE | Xúc phạm, chửi bới, thô tục |
| 2 | HATE | Ngôn từ thù ghét/kỳ thị theo taxonomy của dataset |

Không tự chuyển bài toán sang nhị phân hoặc đổi nghĩa nhãn. **Metric chính:** validation Macro-F1. **Phụ:** per-class Precision, Recall, F1; weighted F1, balanced accuracy, confusion matrix; thời gian train, latency, kích thước. Với xác suất API: LR có `predict_proba`; LinearSVC cần calibrator được fit bằng train/cross-validation; softmax Transformer/CNN/LSTM **không mặc nhiên là xác suất đã calibration**.

## 3. Dataset và Data Contract (Gate G0)
**Owner: TV3/nhóm trưởng**, không cần đợi TV1 kết thúc EDA chuyên sâu.
1. Kiểm tra UIT-ViHSD: khả dụng, schema thực, mapping nhãn, giấy phép, nguồn; ghi nhận điều chưa xác minh. ViCTSD chỉ là đối chiếu/OOD nếu định nghĩa nhãn phù hợp.
2. Phân tích tối thiểu: rỗng, nhãn mâu thuẫn, exact/near duplicates, phân bố lớp, rò rỉ do cùng văn bản/thread; xử lý trùng lặp trước khi split nếu tự chia.
3. **Ưu tiên split chính thức** nếu phù hợp benchmark; nếu không, stratified train/val/test = 70/15/15 với seed lưu manifest; tránh đưa cùng group/thread qua split khi có metadata.
4. Lưu Data Contract v1 (schema `id,text,label,split`, label mapping, normalization rules bất biến, dataset source/revision/hash, manifest ID từng split, seed, leakage checks); giữ test khóa.
5. Chỉ phát hành Train/Validation cho development. Mọi tokenizer, TF-IDF, vocabulary, embedding từ đầu phải train/fit **chỉ trên Train**. Preprocessing có nhiều biến thể được fit riêng trong pipeline train/val của mỗi experiment, không làm thay đổi split.

**TV1** có thể tiếp tục EDA chuyên sâu và đề xuất chỉnh Data Contract nếu phát hiện lỗi nghiêm trọng. Khi đó nhóm trưởng tạo version mới, thông báo cả nhóm, và chạy lại kết quả bị ảnh hưởng **trước final evaluation**; không âm thầm thay split.

## 4. Phân công nghiên cứu độc lập
| Chủ sở hữu | Nhánh | Điều phải chứng minh | Deliverables |
|---|---|---|---|
| TV1 | Logistic Regression | Hiệu quả raw/basic/teencode preprocessing, word/char TF-IDF, n-grams, class weighting | EDA, LR scripts, model, MLflow, báo cáo ablation |
| TV2 | LinearSVC | C, n-grams, feature selection/regularization; RBF SVC conditional | SVM scripts, val comparison, model, MLflow |
| TV3 (lead) | LSTM → BiLSTM → BiLSTM + Attention, thêm PhoBERT benchmark | Bidirectionality và attention có giúp thực nghiệm không; trade-off với pretrained | Scratch DL runs, error analysis, optional PhoBERT run |
| TV4 | TextCNN → Multi-kernel TextCNN, conditional ensemble | Kernel/filter/dropout; diversity và hiệu quả ensemble | CNN runs, ablation, ensemble study nếu đủ điều kiện |

Tất cả làm việc **song song sau Gate G0**. Trước G0, TV1 chạy EDA; TV2–TV4 lập code/training skeleton và smoke tests bằng dữ liệu mẫu không dùng làm final result.

## 5. Thực nghiệm và tái lập
- Ít nhất **3 cấu hình/biến thể có ý nghĩa cho từng thành viên**, không tính sửa lỗi chạy.
- Mỗi thí nghiệm: hypothesis → baseline → thay đổi có kiểm soát → train → val metrics → error analysis → kết luận → quyết định tiếp.
- Mỗi official run dùng MLflow local: experiment ID, run ID, dataset/split hash, seed, hyperparameters, model architecture, train time, validation metrics, artifacts, git revision (nếu có).
- Lưu predictions Validation theo schema `id,true_label,predicted_label,probabilities_if_available,run_id`; không biến raw LinearSVC margins thành probabilities. Notebook trình bày kết quả thực, import code từ `.py`, không chứa logic sản xuất duy nhất.
- Ít nhất một ablation mỗi nhánh, thêm đánh giá nhiều seeds/độ biến thiên cho mô hình mạnh nếu chi phí cho phép. Thí nghiệm thất bại vẫn phải lưu và lý giải.

## 6. Chi tiết kiến trúc/đường cải tiến
### TV1: Logistic Regression
Word TF-IDF (1–2 gram) vs char TF-IDF, kết hợp word+char, raw/basic/teencode cleaning; thay đổi `C`, class_weight. Fit vectorizer trên train, select theo validation Macro-F1. Báo cáo lỗi theo nhóm từ viết tắt/emoji/xúc phạm.

### TV2: SVM
`LinearSVC` là SVM tuyến tính chính (phù hợp sparse TF-IDF); tune `C`, feature count, word/char features, selection, class_weight. `SVC(kernel="rbf")` chỉ thử khi kích thước/cost cho phép; không coi nó là bắt buộc và không đánh đồng với LinearSVC. Nếu cần API probabilities, calibrate đúng trên train.

### TV3: LSTM/BiLSTM
Tự xây tokenizer/vocabulary từ train; embedding **random initialization**, LSTM baseline → BiLSTM → BiLSTM + Attention; kiểm tra hidden size, dropout, sequence length, class weights. Chọn checkpoint theo validation Macro-F1, early stopping, gradient clipping và seed. Error analysis để tạo giả thuyết cải tiến, không mặc định Attention thắng.

### TV4: TextCNN
Embedding khởi tạo ngẫu nhiên; TextCNN single-kernel → multi-kernel; tune window sizes, filters, dropout, pooling và class weighting. Ensemble (soft/weighted voting) chỉ nếu models có xác suất phù hợp và lỗi bổ sung; stacking chỉ với out-of-fold hợp lệ, so với **best single**.

### Optional: PhoBERT benchmark — TV3
Fine-tune `vinai/phobert-base-v2` trên cùng train/val split, cùng metric; ghi rõ pretrained weights và budget, báo cáo macro-F1/time/latency/size. Chỉ làm sau khi TV3 đã có kết quả scratch hợp lệ; TV4 hỗ trợ GPU nếu cần. Không dùng PhoBERT làm nền tảng bắt buộc hoặc lý do hoãn 4 nhánh chính.

## 7. Công bằng, leakage và final test
Đóng băng test cho đến khi lựa chọn mô hình/hyperparameters/calibration/ensemble strategy hoàn tất bằng validation; test cuối chỉ chạy một đợt đánh giá chốt cho tất cả ứng viên đã xác định. Không học meta-learner trên test, không chỉnh mô hình theo test; báo cáo trung tính dù mô hình mới yếu hơn. So sánh cần ghi compute/training budget.

## 8. Sản phẩm nghiên cứu và API
Bắt buộc: EDA, 4 nhánh code/weights, logs và bảng so sánh, error analysis, ablation, notebooks giải thích phương pháp, final locked test report. Demo **FastAPI API-minimal**: POST `/predict`, GET `/health`, GET `/model-info`, Swagger `/docs`. Không xây frontend React/Vite, login, dashboard hay DB. Hosting nếu có chỉ thực hiện sau khi xác minh thực tế. Kế hoạch ≠ đã hoàn thành.

## 9. Điều kiện nghiệm thu và giới hạn thay đổi
Mỗi người có ít nhất 1 thí nghiệm tự huấn luyện, **≥3 biến thể**, metrics validation, MLflow ID, artifact chạy lại, bảng so sánh, nhận xét thành công/thất bại, Notebook. TV3 quản lý G0 và review integration; các thành viên tự chịu trách nhiệm thí nghiệm của mình. Không sửa code/config/dataset hay chạy training trong lần cập nhật tài liệu này.

Tài liệu liên quan: [Experiments](plans/ml-experiments.md) · [Execution plan](plans/implementation-plan.md) · [Tasks](plans/task-breakdown.md).
