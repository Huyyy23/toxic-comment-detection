# Đặc tả dự án: Vietnamese Toxic Comment Detection

**Version:** 5.0.0 · **Ngày:** 2026-11-23 · **Trạng thái:** KẾ HOẠCH cho D1–D4; công việc cũ có trạng thái riêng.

## 1. Bài toán và trạng thái quyết định

Phân loại bình luận tiếng Việt thành **CLEAN=0** và **TOXIC=1**. Dataset nhị phân ViHSD và ViCTSD đã được tạo tại `data/processed/binary_n2_v1/`; xem `dataset-requirements.md` về nguồn và giới hạn.

| Nguồn | Giá trị nguồn | Quy tắc nhãn dự kiến |
|---|---|---|
| ViHSD | `vihsd` | `label > 0` thành TOXIC=1; `label = 0` thành CLEAN=0 |
| ViCTSD | `victsd` | `Toxicity` thành TOXIC=1 hoặc CLEAN=0 |

Taxonomy gốc khác nhau và không được coi là tương đương hoàn toàn. Người dùng xác nhận nhóm đã xin phép UIT cho đồ án; repo chưa có bằng chứng giấy phép để kiểm chứng độc lập. Yêu cầu bài tập và deadline vẫn pending. Lịch sử ba lớp CLEAN/OFFENSIVE/HATE ở v4 được giữ làm lịch sử superseded.

### 1.2 Information Ledger

- [Confirmed 2026-11-23] Người dùng xác nhận PhoBERT pretrained fine-tune đã được chấp nhận làm benchmark riêng.
- [Confirmed 2026-11-23] Người dùng xác nhận nhóm đã xin phép sử dụng dữ liệu UIT cho đồ án; chưa có bằng chứng giấy phép trong repo.
- [Pending] Deadline, yêu cầu bài tập và hosting. Báo cáo theo `source` vì văn phong nguồn có thể tạo shortcut.

## 2. Dữ liệu, split và nguồn gốc
Schema chuẩn gồm `id`, `text`, `source`, `label`, `split`. `label` chỉ nhận 0 hoặc 1; `source` nhận `vihsd` hoặc `victsd`; `split` nhận `train`, `val` hoặc `test`. Giữ source để báo cáo metric theo nguồn và giữ liên kết ID với bản ghi gốc.

Dataset đã deduplicate trong và giữa nguồn, loại 205 nhóm mâu thuẫn và chia gần 70/15/15 theo `(source,label)`, giữ nhóm text chuẩn hóa cùng split. Manifest và hashes nằm trong `data/processed/binary_n2_v1/`; test mới không blind.
Validation Macro-F1 là tiêu chí thống nhất để chọn checkpoint và so sánh mọi mô hình; F1 TOXIC là metric bổ sung, không phải tiêu chí chọn riêng. Mọi biến đổi học được chỉ fit trên train. FastText phải được huấn luyện trên train, không nạp pretrained embeddings. Không dùng LLM hoặc backtranslation cho nhóm chính.

Giữ riêng file test và giới hạn truy cập. Không dùng test để chọn mô hình, calibration hay threshold. Mọi bước bên dưới là dự kiến. Không khẳng định kết quả chưa đo.

### Lịch sử split và trạng thái

Data Contract v1 và `data/processed/vihsd_g0_v1/` mô tả split ba lớp cũ, không phải split nhị phân của đặc tả này. G0 review ghi nhận 23,002 train, 4,929 validation và 4,929 test sau khi loại 540 bản ghi thuộc 241 nhóm xung đột nhãn. Same-label exact duplicates được giữ và gán cùng split. Test lịch sử từng hiện diện trong artifact chia sẻ, nên không được gọi là blind. Không kế thừa số liệu hay trạng thái chấp thuận đó cho split mới.

## 3. Huấn luyện và ablation

Nhóm mô hình chính gồm Logistic Regression, SVM, LSTM, BiLSTM, BiLSTM + Attention và nhánh Char-CNN. Embedding ngẫu nhiên hoặc FastText tự train trên train. PhoBERT là benchmark pretrained fine-tune riêng, không phải mô hình tự train.

| Bước | Ablation trên validation |
|---|---|
| 1 | TF-IDF + LR balanced, baseline riêng. |
| 2 | BiLSTM embedding ngẫu nhiên, cross-entropy. |
| 3 | Bước 2 + FastText train-only. |
| 4 | Bước 3 + Char-CNN. Chốt kiến trúc trước bước 5–7. |
| 5 | Bước 4 + chọn đúng một class weights, focal loss hoặc sampler. |
| 6 | Bước 5 + augmentation, so với bước 5 không augment. |
| 7 | Bước 6 + calibration và threshold. |

Augmentation chỉ áp dụng TOXIC train-only, mặc định thêm một bản và tối đa hai bản mỗi mẫu. Mục tiêu 30–35% không bảo đảm. Dùng deletion/swap, nhiễu dấu hoặc ký tự và teencode hai chiều; bảo vệ token độc hại bằng lexicon train và lọc bản trùng val/test. Mỗi bước đổi một yếu tố, cố định seed, không tuyên bố cải thiện nếu chưa đo.

PhoBERT là benchmark pretrained fine-tune đã được chấp nhận theo xác nhận người dùng. Dùng `vinai/phobert-base`, tách từ và ghi công cụ/phiên bản, 128–160 tokens, giới hạn khoảng 256, learning rate `2e-5`, batch 16, warmup 10%, weight decay `.01`, 3–4 epochs, early stopping theo validation Macro-F1. Dùng fp16 và gradient accumulation nếu hỗ trợ. P1 là cấu hình mặc định; P2 thêm phương án imbalance D3; P3 tùy chọn thay một yếu tố trên val.

## 4. Metrics và đánh giá cuối

Validation Macro-F1 là metric chọn checkpoint cho mọi mô hình. Báo F1 TOXIC bổ sung, precision/recall/F1 theo lớp, PR-AUC, metrics theo `source`, confusion matrix, ECE trước/sau calibration, thời gian train, số tham số, latency và kích thước. Chạy ba seed, báo trung bình ± độ lệch chuẩn và lưu metrics từng seed, config, data hash, validation predictions, checkpoint và log.

Chỉ sau khi freeze candidates, threshold và protocol trên validation mới chạy một batch test cuối cho mọi ứng viên. Báo cáo bootstrap confidence intervals hoặc McNemar, theo một phương pháp thống nhất được chọn trước khi mở test. Không chỉnh mô hình theo kết quả test. Ghi rõ nếu test không blind hoặc có lịch sử tiếp xúc.

Calibration và threshold tuning trên validation có thể làm metric validation lạc quan nếu dùng cùng tập đó để so sánh candidates. Chọn kiến trúc/checkpoint bằng raw predictions và validation Macro-F1 trước. Sau khi chọn candidate, fit calibrator và threshold trên validation chỉ cho artifact inference. Ghi calibration diagnostics validation là in-sample, không dùng để xếp hạng model hoặc tuyên bố so sánh công bằng. Freeze artifact trước test và không fit lại trên test.

Các endpoint local đã triển khai là `POST /predict`, `GET /health`, `GET /model-info`. `/predict` nhận `{text}` và trả `label_id` (0 CLEAN, 1 TOXIC), `label_name`, `confidence`, `probabilities` (`CLEAN`, `TOXIC`), `is_calibrated`, `threshold`, `latency_ms`, `model_backend`. `/model-info` công bố backend, label mapping, calibration status, threshold và artifact version. FastAPI dùng adapter inference thay thế được; frontend React gọi contract này. Hiện chỉ bật baseline N6-LR-D raw trên local; xác suất chưa hiệu chuẩn (`is_calibrated=false`), threshold mặc định 0.5. Chưa deploy public hoặc chốt production model trước N10.


## 5. Capability Map

| Module | Trách nhiệm và phụ thuộc |
|---|---|
| `mod-data-merge` | Map nhị phân, `source`, dedupe trước split và manifest. |
| `mod-augment` | Augment TOXIC train-only sau kiểm tra leakage. |
| `mod-embedding` | FastText tự train trên train. |
| `mod-neural` | LSTM, BiLSTM, Attention và Char-CNN tự huấn luyện. |
| `mod-trans` | PhoBERT pretrained fine-tune làm benchmark đã được chấp nhận. |
| `mod-phobert-exp` | Thí nghiệm PhoBERT pretrained fine-tune bổ sung trên val. |
| `mod-calibration` | Calibration, threshold và ECE trên val cho các mô hình đủ điều kiện. |

## 6. Artifact và ứng dụng

Inference artifacts remain in their original project paths. N12 copies only the selected N6 joblib, TRI checkpoint/vocabulary/config and PhoBERT checkpoint/config into `deployment/hf-gradio/models/`; it does not move or overwrite any source artifact.

N11 local React/FastAPI remains unchanged. N12 uses Gradio as the primary Vietnamese interface and FastAPI REST in the same process on port 7860, with ZeroGPU Free as the target host and no Docker. Three validation-selected models are exposed individually and in a side-by-side comparison. All probabilities are explicitly uncalibrated. Local UI/REST/Swagger checks pass; public eligibility, rights, N10 evidence, and hosted FastAPI GPU allocation remain unresolved.

## 7. Điều kiện nghiệm thu D1–D4

| Giai đoạn | Điều kiện dự kiến |
|---|---|
| **D1 Data** | Dataset nhị phân đã tạo và kiểm tra cơ bản; N3 tiếp tục audit leakage. |
| **D2 Models** | LR balanced, SVM, BiLSTM random-init/cross-entropy, FastText train-only và Char-CNN; checkpoint theo validation Macro-F1; 3 seeds; không pretrained embeddings, LLM hay backtranslation trong nhóm chính; PhoBERT pretrained fine-tune là benchmark riêng đã được chấp nhận. |
| **D3 Improvements** | TOXIC augmentation train-only theo giới hạn; chọn đúng một imbalance method; calibration, threshold và ECE; không dùng test. |
| **D4 Evaluation** | Cùng split, binary metrics, 3 seeds; freeze candidates trước một batch test; bootstrap CI hoặc McNemar; error analysis; artifacts và local Gradio/FastAPI evidence. Public HF deployment chỉ sau khi xác minh account eligibility, artifact/data rights và ZeroGPU REST allocation. |

Test chỉ mở một đợt sau khi đóng băng mọi candidate, seed, threshold và protocol; file khóa ghi danh sách trước đợt đánh giá và ngăn chạy lại. Báo riêng nhóm tự huấn luyện và PhoBERT pretrained fine-tune. Phân tích lỗi sau test chỉ để báo cáo, không tuning. Ghi ví dụ, số mẫu, lỗi tiếng lóng, viết tắt bị che, mỉa mai, câu dài, emoji và nhãn nhiễu; cắt lát theo nguồn, độ dài và teencode.

Giấy phép UIT chưa được kiểm chứng độc lập trong repo; yêu cầu bài tập và deadline chưa được xác minh. Các điều kiện còn lại là tiêu chí dự kiến, không phải kết quả nghiệm thu.

## 8. Tài liệu liên quan

[Yêu cầu dữ liệu](dataset-requirements.md) · [Quyết định kiến trúc](decisions.md) · [Câu hỏi mở](open-questions.md) · [Kế hoạch thí nghiệm](plans/ml-experiments.md) · [Kế hoạch triển khai](deployment-plan.md).
