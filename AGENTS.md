# AGENTS.md — Vietnamese Toxic Comment Detection

> Quy tắc làm việc lâu dài cho AI coding agents trong repository này. Áp dụng cho mọi phiên audit, lập kế hoạch, triển khai, huấn luyện, đánh giá và triển khai REST API.

## 1. Sứ mệnh và thứ tự ưu tiên

Đây là **đồ án cuối kỳ Machine Learning** về phân loại bình luận tiếng Việt **nhị phân** (`CLEAN=0`, `TOXIC=1`), với đánh giá có thể tái lập và ứng dụng theo SPEC hiện hành.

Ưu tiên: (1) tính đúng đắn khoa học, không leakage; (2) evidence có thể truy vết; (3) đánh giá công bằng trên cùng split; (4) tái lập và phối hợp nhóm; (5) code đơn giản, ít phụ thuộc, tiết kiệm tài nguyên.

Không coi checklist là bằng chứng kết quả; không khẳng định cải thiện nếu metrics thực tế không chứng minh.

## 2. Nguồn sự thật và quy trình bắt đầu mỗi phiên

**Phân công, ownership, dependency, trạng thái:** dùng `docs/TEAM_WORK_CONTEXT.md`, `tasks/plan.md`, `tasks/todo.md` và checklist thành viên liên quan trong `docs/todos/`. Plan quy định scope/dependencies; TODO/context ghi tiến độ và bằng chứng. Nếu khác nhau hoặc còn mục legacy, không tự chọn trạng thái: ưu tiên thông tin mới hơn có nguồn/evidence và nêu mâu thuẫn khi ảnh hưởng quyết định.

Trước khi sửa, chỉ đọc các tài liệu liên quan đang tồn tại: `SPEC.md` là contract hiện hành; `tasks/plan.md` và `tasks/todo.md` là plan/checklist; `docs/plans/ml-experiments.md` là protocol thí nghiệm; các tài liệu dataset/deployment/ADR khi task cần. Tài liệu lịch sử không thay contract hiện hành.

Trước mỗi thay đổi: kiểm tra `git status`, tìm đúng vị trí cần sửa, xem dependency và tác động. Không ghi đè thay đổi chưa commit hoặc làm mất dữ liệu/nghiên cứu của nhóm.

## 3. Phạm vi kiến trúc đã chốt

- **Research:** dataset nhị phân ViHSD/ViCTSD, EDA, preprocessing, baseline, mô hình tự huấn luyện, ablation, đánh giá và error analysis theo plan.
- **Ứng dụng:** React + FastAPI, Docker và Hugging Face Spaces theo SPEC; trạng thái triển khai/eligibility phải có bằng chứng, không mặc định đã hoàn tất.
- Không tự ý mở rộng sang dashboard, tài khoản, JWT, database người dùng, proxy hay service nền.

Chỉ phục vụ model được chọn cùng preprocessing/tokenizer, nhãn và artifact inference cần thiết. Không train hoặc khởi chạy MLflow trong ứng dụng public.

## 4. Protocol nghiên cứu

- So sánh trên cùng binary split; chọn checkpoint/candidate bằng validation Macro-F1, báo F1 `TOXIC` và metrics theo lớp.
- Core comparison dùng mô hình scratch, không dùng pretrained embeddings, LLM hoặc back-translation. PhoBERT fine-tuning được chấp nhận riêng làm pretrained benchmark.
- Ablation chỉ đổi một yếu tố mỗi lần; ghi đối chứng, protocol, seed, kết quả và giới hạn. Kết quả không tốt cũng là kết quả hợp lệ.
- Mọi bước chọn mô hình, imbalance method, calibration/threshold và optional extras dùng validation. Trước test phải freeze candidate/config/threshold/phương pháp phân tích; chỉ một final test batch. Không dùng test để tune, chọn hoặc sửa giả thuyết; công khai historical exposure.

## 5. Dataset và chống rò rỉ dữ liệu

- Taxonomy hiện hành: `CLEAN=0`, `TOXIC=1`; không dùng schema ba lớp CLEAN/OFFENSIVE/HATE cũ ngoài phần lịch sử.
- Dùng binary dataset `data/processed/binary_n2_v1/`; mapping, provenance, giới hạn nhãn/quyền và split tại `docs/dataset-requirements.md`.
- Nguồn nhãn khác nghĩa; không khẳng định tương đương hoàn toàn hoặc quyền phân phối chưa được kiểm chứng.
- Chỉ fit vocabulary, embeddings, preprocessing và augmentation trên train; giữ validation/test tách biệt. Không dùng test khóa để tune hoặc debug.

## 6. Mô hình và thiết kế thí nghiệm

Các ứng viên và ablations theo plan hiện hành:
1. Baselines: TF-IDF + Logistic Regression và Linear SVM.
2. Core scratch models: LSTM, BiLSTM/Attention, FastText train-only và Char-CNN theo task/dependency.
3. PhoBERT fine-tuning là benchmark pretrained riêng, không phải scratch model.
4. Chỉ chạy optional extras/ensemble khi plan cho phép và có căn cứ validation.

- Metric chọn checkpoint/candidate: validation Macro-F1; bổ sung F1 TOXIC, per-class precision/recall/F1, confusion matrix, PR-AUC/ECE khi phù hợp, runtime, model size và seed.
- Dùng protocol/split chung, ba seeds theo plan; lưu config, data hashes, run IDs, checkpoint và validation predictions.
- LinearSVC margins không phải xác suất; calibration/threshold chỉ fit/chọn trên validation và nêu rõ bias nếu đánh giá trên cùng validation.
- Stacking chỉ dùng out-of-fold predictions từ train để fit meta-learner. Không ép ensemble nếu thiếu diversity hoặc kém hơn model đơn.

## 7. Python `.py`, Notebook và MLflow: ranh giới trách nhiệm

- **`.py` là nguồn sự thật cho logic tái sử dụng:** preprocessing, feature pipeline, training, evaluation, inference; dùng config/CLI để chạy.
- **`.ipynb` chỉ để EDA, phân tích và lưu kết quả thí nghiệm:** import module `.py`, đọc dữ liệu/MLflow/artifacts, vẽ biểu đồ, diễn giải. Không copy logic train vào notebook; không phụ thuộc thứ tự cell tùy tiện; không nhúng checkpoint, dataset lớn hoặc bình luận nhận diện cá nhân.
- **MLflow là sổ ghi thí nghiệm:** mọi run train chính thức phải lưu run ID, model/config, seed, dataset version/hash, split, metrics, thời gian, phiên bản code/môi trường, checkpoint hoặc tham chiếu artifacts. Optuna dùng nếu thực sự cần tuning.
- Dữ liệu và kết quả Notebook phải dựa trên run thực tế. Nếu chưa có run, ghi trạng thái `CHƯA THỰC HIỆN`, không vẽ biểu đồ kết quả giả.
- Tách experiments và runtime: không cần MLflow, Optuna, notebook hay training datasets để chạy API production.

## 8. Làm việc nhóm trên nhiều máy

- Không hard-code GPU/CPU, RAM, OS, đường dẫn cá nhân, token hoặc vị trí dataset.
- Cấu hình thiết bị qua `auto/cpu/cuda` hoặc cơ chế phù hợp thư viện; chỉ bật mixed precision khi được hỗ trợ.
- Batch size, workers, sequence length, precision, seed, đường dẫn artifacts và ngân sách tuning phải cấu hình được.
- Thiết kế run IDs, config snapshots, dataset/split hashes và phiên bản model giúp **gộp kết quả độc lập từ nhiều máy**; không bắt buộc dựng MLflow remote server nếu chia sẻ artifact là đủ.
- Hỗ trợ checkpoint/resume khi có ích. Không giả định thời gian train cố định cho mọi máy.
- Không đưa secret, access token, thông tin đăng nhập vào code, notebook, log hoặc Git.

## 9. Quy tắc triển khai FastAPI

- API dùng contract nhị phân hiện hành của `SPEC.md`/`backend/schemas.py`, gồm `CLEAN`/`TOXIC`; giữ thứ tự xác suất và ý nghĩa calibration nhất quán.
- Chỉ trả confidence/probabilities khi cách tính hợp lệ; raw margins/softmax không tự động là calibrated probabilities.
- Không train hoặc tự tải dữ liệu huấn luyện lúc API startup; kiểm tra input/readiness và tính nhất quán offline/API.
- React, FastAPI, Docker và HF Spaces là hướng ứng dụng trong SPEC; xác minh eligibility, smoke test và URL trước khi tuyên bố deployment.

## 10. Kỷ luật thay đổi code và kiểm thử

- Mỗi phiên chỉ thực hiện **task được giao** và dependency trực tiếp; không tự ý chuyển milestone hoặc thêm tính năng ngoài phạm vi.
- Ưu tiên thay đổi nhỏ, rõ trách nhiệm, dễ review; tái sử dụng code có sẵn; tránh tạo module rỗng, duplicate implementation và abstraction không có nhu cầu.
- Trước khi sửa: kiểm tra trạng thái file và ảnh hưởng; sau khi sửa: chạy test/lint/smoke **phù hợp phạm vi**, không tự khởi chạy full train.
- Thay đổi preprocessing/labels/splits/API contract phải cập nhật test và tài liệu tương ứng.
- Không xóa artifacts, notebooks, dataset, checkpoint hoặc lịch sử thí nghiệm nếu chưa được giao và chưa bảo đảm an toàn.
- Không tự ý commit, push, tạo PR, triển khai public, cài package hệ thống, khởi chạy service nền hoặc chi tiền.
- Các lệnh có thể mất nhiều tài nguyên (fine-tuning, tải dataset lớn, sweep, OOF Transformer, benchmark toàn bộ) chỉ chạy khi người dùng **yêu cầu rõ**. Có thể chạy unit tests nhỏ và kiểm tra tĩnh trong phạm vi task.
- Không lén chuyển từ phương án được duyệt sang phương án khác khi gặp lỗi; nêu vấn đề, chọn fallback an toàn nếu có trong SPEC hoặc yêu cầu quyết định khi thay đổi lớn.

## 11. Quy ước tài liệu và báo cáo tiến độ

- Ưu tiên **tiếng Việt rõ ràng, súc tích**, giữ nguyên tên thuật ngữ/biến/method tiếng Anh khi cần.
- Không tạo tài liệu trùng lặp. Cập nhật SPEC, ADR, Experiment Plan, Task Breakdown đã có; thêm file mới chỉ khi có mục đích riêng.
- Duy trì trạng thái tách biệt: `PLANNED`, `IMPLEMENTED`, `TESTED`, `EXPERIMENTALLY VALIDATED`, `DEPLOYED` — chỉ dùng trạng thái có bằng chứng.
- Với ablation, ghi rõ đối chứng và thay đổi từng yếu tố một; không tuyên bố cải thiện trước khi có metrics.
- Khi xong task, báo cáo ngắn: đã làm gì; file thay đổi; lệnh test và kết quả; điều chưa xác minh; rủi ro; task tiếp theo theo plan.
- Không ngụy tạo lệnh test đã chạy, metric, đường dẫn artifact hay trạng thái triển khai.

## 12. Tiêu chí hoàn thành toàn dự án

Hoàn thiện theo các gate trong `tasks/plan.md`/`tasks/todo.md`: dataset và leakage limits có provenance; baseline và scratch/pretrained benchmark theo protocol chung; artifacts/results có thể truy vết; ablation/error analysis dựa trên validation; candidate được freeze trước final locked-test batch và historical exposure được công bố; React/FastAPI API và deployment chỉ được đánh dấu hoàn thành khi có kiểm thử thực tế.

## 13. Quy tắc truy vấn graphify (bắt buộc)

Trước khi đọc hoặc grep bất kỳ file nguồn nào trong repository, agent **phải truy vấn graphify knowledge graph** trước:

`graphify query "<câu hỏi về cấu trúc, phụ thuộc, luồng dữ liệu>"`

Chỉ truy vấn trực tiếp file khi graph không trả lời được câu hỏi, hoặc khi cần xem nội dung chi tiết của file cụ thể đã được graph chỉ ra. Quy tắc này giúp hiểu kiến trúc tổng thể trước khi chi tiết, tránh bỏ lỡ liên kết giữa tài liệu và code.

Ưu tiên câu hỏi dạng `graphify path "X" "Y"`, `graphify explain "<node>"`, hoặc `graphify query "<câu hỏi chung>"`.
