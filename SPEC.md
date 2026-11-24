# Vietnamese Toxic Comment Detection — Project Specification

**Version:** 5.0.0 · **Date:** 2026-11-23 · **Status:** PLANNED cho D1–D4; các công việc cũ có trạng thái riêng.

## Mục tiêu và trạng thái

Đồ án phân loại bình luận tiếng Việt thành **CLEAN=0** hoặc **TOXIC=1**. Người dùng xác nhận nhóm đã xin phép UIT cho đồ án; repo chưa có bằng chứng giấy phép để kiểm chứng độc lập. Yêu cầu bài tập và deadline còn pending.

ViHSD dùng `label > 0` làm TOXIC; ViCTSD dùng `Toxicity` làm nhãn nhị phân. Hai taxonomy gốc khác nhau và không được coi là tương đương hoàn toàn. CLEAN luôn là 0, TOXIC là 1. Không dùng pretrained embeddings, LLM hay backtranslation cho nhóm mô hình chính.

**Lịch sử:** đặc tả ba lớp CLEAN/OFFENSIVE/HATE ở v4 được giữ làm lịch sử đã superseded, không còn là taxonomy hiện hành.

## Dữ liệu và giao thức

- Dataset nhị phân ViHSD và ViCTSD tại `data/processed/binary_n2_v1/` đã deduplicate và chia gần 70/15/15 theo `(source,label)` với seed 42. Xem `docs/dataset-requirements.md` và manifest để biết provenance và giới hạn.
- Validation Macro-F1 là tiêu chí thống nhất để chọn checkpoint và so sánh mọi mô hình; F1 TOXIC là metric bổ sung.
- Fit mọi biến đổi học được chỉ trên train. Không dùng pretrained embeddings, LLM hoặc backtranslation cho nhóm chính.
- Dùng 3 seed và báo metrics từng seed. Test khóa, chỉ một batch cuối cho candidates đã freeze. Dùng bootstrap CI hoặc McNemar theo protocol thống nhất trước khi mở test.
- Không khẳng định kết quả chưa đo. Mọi bước bên dưới là dự kiến.

### Lịch sử split và trạng thái

G0 review ghi nhận split ba lớp cũ có 23,002 train, 4,929 validation và 4,929 test sau khi loại 540 bản ghi thuộc 241 nhóm xung đột nhãn. Same-label exact duplicates được giữ cùng split. Test từng hiện diện trong artifacts chia sẻ, nên không được gọi là blind. Split cũ không phải split binary mới và không chuyển trạng thái chấp thuận sang split mới.


## Mô hình và ablation

Các nhóm chính tự huấn luyện Logistic Regression (LR), SVM, BiLSTM random-init với cross-entropy, FastText, và Char-CNN. Ablation gồm LR balanced; BiLSTM random CE; FastText train-only; Char-CNN; chọn đúng một phương án imbalance theo validation; augmentation train-only; calibration cùng threshold. Mỗi ablation đổi một yếu tố so với baseline, không giả định cải thiện.

Augmentation chỉ áp dụng cho TOXIC trên train, tối đa +1 mẫu sinh mỗi mẫu gốc và không quá 2x. Mục tiêu TOXIC chiếm 30–35% không bảo đảm. Bảo vệ từ độc hại và lọc bản trùng với validation/test. Chọn duy nhất một trong class weights, focal loss hoặc sampler bằng validation.

PhoBERT là benchmark pretrained fine-tune riêng, đã được chấp nhận theo xác nhận người dùng, không phải mô hình scratch. Dùng `vinai/phobert-base`, word segmentation có ghi công cụ và phiên bản, 128–160 tokens, giới hạn khoảng 256, learning rate `2e-5`, batch 16, warmup 10%, weight decay `.01`, 3–4 epochs, early stopping theo validation Macro-F1. Dùng fp16 và gradient accumulation nếu hỗ trợ. P1 là cấu hình mặc định; P2 thêm phương án imbalance D3; P3 tùy chọn thử một yếu tố trên val.
Calibration và threshold validation có thể tạo optimistic bias nếu dùng để xếp hạng model. So sánh raw predictions và chọn candidate bằng validation Macro-F1 trước. Sau đó fit calibration và threshold trên validation chỉ cho artifact inference; ghi diagnostics là in-sample, không dùng làm bằng chứng so sánh công bằng. Freeze artifact trước test, không fit lại trên test.


## API và artifacts inference local

Prediction binary gồm `label_id`, `label_name`, `confidence`, `probabilities` (`CLEAN`, `TOXIC`), `is_calibrated`, `threshold`, `latency_ms`, `model_backend`. Probabilities có tổng xấp xỉ 1; `confidence` là xác suất của nhãn trả về. `is_calibrated` chỉ true khi có calibration hợp lệ; raw `predict_proba` và softmax không tự động là calibrated probabilities.

N11 local React/FastAPI hiện có tiếp tục giữ nguyên. N12 dùng bundle riêng `deployment/hf-gradio/`, với Gradio làm UI, FastAPI REST cùng process và port 7860; không dùng Docker.

N12 inference candidates được chọn trên validation cùng split N2:

| Model | Artifact dùng trong N12 | Val Macro-F1 | F1 TOXIC | Calibration |
|---|---|---:|---:|---|
| N6-LR-D | Official C=5, seed 42; seed 42/43/44 tie, chọn seed thấp nhất | 0.7901 | 0.6419 | Raw `predict_proba`, chưa calibration |
| V3-TRI-GATED-FT | Validation-only replacement, seed 43 | 0.7930 | 0.6467 | Raw softmax, chưa calibration |
| V3-PHOBERT-V2 | Validation-only replacement, seed 44 | 0.8195 | 0.6943 | Raw softmax, chưa calibration |

N11 local fallback không phải artifact N12. Hai V3 historical weights không có; N12 bundle copy các checkpoint replacement đã tồn tại, không train lại. Chi tiết hash, provenance và inference preprocessing ở [deployment README](deployment/hf-gradio/README.md). Không có dataset, logs, notebooks hoặc validation predictions trong bundle.

Local Gradio, REST, Swagger và cả ba model đã smoke test bằng synthetic input. Public ZeroGPU Space chưa triển khai. Account eligibility, quyền phân phối artifacts/dữ liệu, sign-off N10 hiện có và GPU allocation cho direct FastAPI routes chưa được xác minh. Không truy cập test set hoặc mở lại N10/N15.

- **D1 Data:** Dataset nhị phân đã được tạo và kiểm tra cơ bản; test không blind. Kiểm tra leakage sâu hơn thuộc N3.
- **D2 Models:** LR balanced, SVM, BiLSTM random-init/cross-entropy, FastText train-only và Char-CNN; checkpoint theo validation Macro-F1; 3 seeds; không pretrained embeddings, LLM hay backtranslation trong nhóm chính. PhoBERT pretrained fine-tune là benchmark riêng.
- **D3 Improvements:** augmentation TOXIC chỉ trên train theo giới hạn; chọn đúng một imbalance method; calibration, threshold và ECE; không dùng test.
- **D4 Evaluation:** cùng split, binary metrics, 3 seeds; freeze candidates trước một batch test; bootstrap CI hoặc McNemar; error analysis; N12 có bundle và local evidence, còn public deployment chỉ sau khi đóng các gate nêu trên.

Giấy phép UIT và quyền phân phối artifacts chưa được kiểm chứng độc lập trong repo; yêu cầu bài tập và deadline còn pending.

Tài liệu liên quan: [Đặc tả chi tiết](docs/spec.md), [kế hoạch thí nghiệm](docs/plans/ml-experiments.md), [yêu cầu dữ liệu](docs/dataset-requirements.md), [deployment](docs/deployment-plan.md).
