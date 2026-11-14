# AGENTS.md — Vietnamese Toxic Comment Detection

> Quy tắc làm việc lâu dài cho AI coding agents trong repository này. Áp dụng cho mọi phiên audit, lập kế hoạch, triển khai, huấn luyện, đánh giá và triển khai REST API.

## 1. Sứ mệnh và thứ tự ưu tiên

Đây là **đồ án cuối kỳ Machine Learning**, không phải dự án full-stack. Mục tiêu là xây dựng hệ thống phân loại bình luận độc hại tiếng Việt **end-to-end, có thể tái lập, có bằng chứng thực nghiệm**, và phục vụ mô hình đã chọn qua **FastAPI REST API**.

Ưu tiên theo thứ tự:
1. Tính đúng đắn khoa học: không data leakage, đánh giá công bằng, kết quả có thể truy vết.
2. Chất lượng nghiên cứu ML: baseline → giả thuyết → thí nghiệm → phân tích → cải tiến.
3. Khả năng tái lập, kiểm thử và làm việc nhóm trên nhiều máy.
4. Code đơn giản, dễ bảo trì, ít phụ thuộc và tiết kiệm tài nguyên.
5. REST API công khai tối giản; không đầu tư vào UI/UX.

**Không tối ưu vì số mô hình nhiều, GPU chạy lâu, code phức tạp hoặc benchmark đẹp.** Chỉ kết luận từ bằng chứng thực tế.

## 2. Nguồn sự thật và quy trình bắt đầu mỗi phiên

**Phân công, EXP ownership, dependencies hoặc tiến độ nhóm:** đọc `docs/TEAM_WORK_CONTEXT.md` trước. Khi context đủ thông tin, chỉ đọc file/section chi tiết liên quan đến task, không tự động đọc lại toàn repository. Đối chiếu SPEC/Plan đúng phiên bản khi có thay đổi, nghi sai lệch hoặc cần xác minh kỹ thuật; context không thay nguồn sự thật. Cập nhật đúng phần trong context khi phân công, dependencies hoặc trạng thái thay đổi có bằng chứng, kèm ngày và nguồn.

Trước khi sửa, đọc những tài liệu **đang tồn tại** và liên quan trong repo (không mặc định file nào cũng có):
- `SPEC.md`, `docs/spec.md`: phạm vi, contract, acceptance criteria.
- `docs/decisions.md`: các quyết định đã chốt (ADR).
- `docs/plans/implementation-plan.md`, `docs/plans/task-breakdown.md`: milestone, dependency, task.
- `docs/plans/ml-experiments.md`: giả thuyết và thiết kế thực nghiệm.
- `docs/dataset-requirements.md`, `docs/deployment-plan.md`, `docs/open-questions.md` nếu liên quan.
- `README.md`, cấu hình, kiểm thử và code thực tế liên quan.

Quy tắc xử lý mâu thuẫn:
- Yêu cầu mới nhất của người dùng và yêu cầu giảng viên là ưu tiên cao nhất.
- `AGENTS.md` quy định cách làm; SPEC/ADR quy định sản phẩm; Plan quy định trình tự công việc.
- Nếu tài liệu mâu thuẫn hoặc lỗi thời: nêu rõ và đề xuất cập nhật; **không âm thầm chọn một giả định quan trọng**.
- **Code được kiểm tra thực tế** quyết định cái gì đã triển khai; tài liệu không phải bằng chứng code đã chạy.

Trước mỗi thay đổi: kiểm tra `git status`, tìm đúng vị trí cần sửa, xem dependency và tác động. Không ghi đè thay đổi chưa commit hoặc làm mất dữ liệu/nghiên cứu của nhóm.

## 3. Phạm vi kiến trúc đã chốt

- **Research:** dữ liệu, EDA, preprocessing, features, training, tuning, tracking, ensemble, evaluation, ablation, error analysis.
- **Production:** Python inference + FastAPI REST API; Swagger `/docs` dùng để demo.
- Các endpoint mục tiêu tối thiểu: `POST /predict`, `GET /health`, `GET /model-info` (theo contract hiện hành trong SPEC).
- **Không xây dựng frontend** React/Vite/Streamlit/HTML riêng, dashboard, tài khoản, JWT hoặc database người dùng nếu không có yêu cầu mới rõ ràng.
- Loại bỏ frontend cũ theo task được giao; **không tự ý xóa** file/code khi đang làm task khác.
- Giữ Dockerfile nếu cách triển khai đã chọn cần Docker; không tự mặc định Docker hoặc Docker Compose là thừa.
- Không thêm proxy, service nền thường trực, orchestrator, MLOps stack hoặc CI/CD phức tạp nếu nhu cầu thực tế chưa chứng minh.

Chỉ phục vụ **model hoặc ensemble cuối cùng** cùng tokenizer, label mapping và preprocessing artifacts cần thiết. Không khởi chạy train/MLflow server trong ứng dụng public.

## 4. Phương pháp nghiên cứu của giảng viên

Giảng viên cho phép **một trong hai hướng**, không phải hai pipeline bắt buộc:

- **Hướng 1 — So sánh N mô hình:** huấn luyện các kiến trúc khác nhau trên cùng protocol, so sánh công bằng và chọn mô hình phù hợp nhất dựa trên validation; chỉ kiểm thử cuối cùng trên locked test.
- **Hướng 2 — Cải tiến có căn cứ:** A → đánh giá/nhận diện điểm yếu → đặt giả thuyết → thiết kế B → đánh giá đối chứng → chấp nhận/bác bỏ giả thuyết → lặp lại nếu cần.

**Hướng 2 là hướng trình bày và nghiên cứu chính** theo quyết định hiện tại. Dùng bảng so sánh kiểu Hướng 1 như bằng chứng bổ trợ; không lập hai luồng training trùng lặp. Mọi vòng cải tiến phải ghi: vấn đề, bằng chứng, giả thuyết, thay đổi, protocol, kết quả và kết luận. Khi chưa train, dùng từ **dự kiến/giả thuyết**, tuyệt đối không ghi “đã cải thiện”. Thử nghiệm thất bại vẫn là kết quả hợp lệ.

## 5. Dataset và chống rò rỉ dữ liệu

- Bài toán mục tiêu: 3 lớp `CLEAN`, `OFFENSIVE`, `HATE`; dùng **label mapping và schema chuẩn** được định nghĩa trong repo.
- Nguồn đang xem xét: ViHSD, ViCTSD và các nguồn có tài liệu hợp lệ. **Không gộp nhãn/tập dữ liệu khác taxonomy** nếu chưa có ánh xạ được biện minh và kiểm chứng.
- Ghi rõ nguồn, phiên bản, giấy phép, ngày/phạm vi dữ liệu, hash, phương pháp gán nhãn và hạn chế. Không bịa thông tin chưa kiểm chứng.
- Kiểm tra missing, duplicate, near-duplicate, phân phối nhãn và trùng lặp giữa train/validation/test; bảo vệ riêng tư khi lưu các bình luận mẫu.
- Chỉ fit vectorizer, tokenizer đặc thù được học từ dữ liệu, resampling, thống kê, normalization và các bước có học tham số **trên train/fold train**. Transform validation/test bằng artifact đã fit.
- **Locked test** chỉ dành cho đánh giá cuối cùng sau khi chọn mô hình và đóng băng hyperparameters/ensemble. Không xem test để tuning, chọn mô hình, chọn ngưỡng, chọn ensemble hay sửa giả thuyết. Nếu đã có vi phạm, báo cáo rõ và thiết kế lại quy trình đánh giá.
- External test (nếu có): độc lập, thu thập hợp pháp, khử thông tin định danh, chuẩn gán nhãn rõ ràng; không train/tune trên đó nếu dùng làm đánh giá ngoài phân phối.

## 6. Mô hình và thiết kế thí nghiệm

Các nhóm ứng viên để **đưa vào kế hoạch**, không có nghĩa tất cả đều phải cài ngay:
1. Baseline bắt buộc: TF-IDF + Logistic Regression; TF-IDF + Linear SVM.
2. Deep learning ưu tiên: PhoBERT và ViSoBERT (fine-tuning có kiểm soát).
3. Mở rộng có điều kiện: XGBoost với đặc trưng phù hợp, BiLSTM hoặc TextCNN.
4. Ensemble: Soft Voting, Weighted Voting; Stacking nếu hợp lý về chi phí và tính đúng đắn.
5. Ablation: feature/preprocessing, class weights, tuning, kiến trúc và thành phần ensemble.

- **Metric chính:** Macro-F1, trừ khi SPEC ghi quyết định khác. Bổ sung per-class precision/recall/F1, confusion matrix, weighted-F1, training time, model size, inference latency; calibration nếu sử dụng xác suất.
- Chia dữ liệu và chọn metric/protocol **trước** khi xem kết quả. Cố định seed, lưu split IDs/hash; so sánh trên cùng split và điều kiện phù hợp.
- `LinearSVC.decision_function` **không phải** xác suất. Nếu cần Voting xác suất, dùng phương pháp hiệu chuẩn đúng protocol, chỉ dựa trên dữ liệu train/validation hợp lệ.
- **Stacking:** meta-learner chỉ học từ out-of-fold predictions tạo trên train; không dùng prediction in-sample hoặc nhãn validation/test để fit. Cân nhắc chi phí OOF của Transformer; nếu không khả thi, ưu tiên Voting hợp lệ.
- Không mặc định mô hình lớn hơn tốt hơn; không ép chọn ensemble nếu nó kém hơn mô hình đơn.
- Khi phân tích lỗi, kiểm tra nhầm lẫn `OFFENSIVE`/`HATE`, false positives/negatives, teencode, tiếng lóng, emoji, sai chính tả và phân phối mới.

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

- API chỉ load mô hình đã được chọn và preprocessing/tokenizer/label mapping tương ứng.
- Đầu ra tuân thủ **probability contract hiện hữu** trong `backend/schemas.py` hoặc contract chuẩn đã được SPEC phê duyệt. Không tự ý đổi schema/label mà chưa cập nhật tests và tài liệu.
- Trả `confidence`/`probabilities` chỉ khi cách tính hợp lệ, mô tả rõ ý nghĩa. Tổng xác suất cần hợp lệ trong sai số số học; thứ tự lớp nhất quán.
- Không train hoặc tự tải dữ liệu huấn luyện lúc API startup. Tránh khởi động dịch vụ không cần thiết.
- Phải có kiểm tra input sai, rỗng, quá dài; kiểm tra model readiness và tính nhất quán offline vs API.
- Kiểm chứng điều kiện hosting trước khi quyết định HF Spaces/Render/Railway; không tuyên bố deploy thành công khi chưa kiểm thử URL public.

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
- Đối với Hướng 2, log quyết định theo cấu trúc: **Baseline → Vấn đề → Giả thuyết → Thay đổi → Kết quả → Kết luận**.
- Khi xong task, báo cáo ngắn: đã làm gì; file thay đổi; lệnh test và kết quả; điều chưa xác minh; rủi ro; task tiếp theo theo plan.
- Không ngụy tạo lệnh test đã chạy, metric, đường dẫn artifact hay trạng thái triển khai.

## 12. Tiêu chí hoàn thành toàn dự án

Một đồ án hoàn chỉnh phải có: dữ liệu có nguồn gốc và split an toàn; pipeline `.py` có thể tái lập; baseline + mô hình nâng cao được so sánh công bằng; các vòng Hướng 2 được chứng minh bằng log; đánh giá ensemble/ablation/error analysis; notebook báo cáo kết quả thực; MLflow tracking; mô hình được chọn theo validation và đánh giá cuối trên locked test; FastAPI REST API public hoạt động; tài liệu hướng dẫn tái lập và các giới hạn của kết quả.

## 13. Quy tắc truy vấn graphify (bắt buộc)

Trước khi đọc hoặc grep bất kỳ file nguồn nào trong repository, agent **phải truy vấn graphify knowledge graph** trước:

`graphify query "<câu hỏi về cấu trúc, phụ thuộc, luồng dữ liệu>"`

Chỉ truy vấn trực tiếp file khi graph không trả lời được câu hỏi, hoặc khi cần xem nội dung chi tiết của file cụ thể đã được graph chỉ ra. Quy tắc này giúp hiểu kiến trúc tổng thể trước khi chi tiết, tránh bỏ lỡ liên kết giữa tài liệu và code.

Ưu tiên câu hỏi dạng `graphify path "X" "Y"`, `graphify explain "<node>"`, hoặc `graphify query "<câu hỏi chung>"`.
