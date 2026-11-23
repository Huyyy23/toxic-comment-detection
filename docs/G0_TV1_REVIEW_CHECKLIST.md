# G0 — Checklist hoàn tất bàn giao dữ liệu của TV1

**Mục đích:** checklist hành động để TV1 hoàn tất hồ sơ G0 và gửi TV3 review. Đây là đề xuất; chưa phê duyệt G0 và chưa cho phép official training.

## Quyết định đã thống nhất

- Dataset: UIT-ViHSD từ `ura-hcmut/UIT-ViHSD`; người dùng xác nhận chấp nhận nguồn là đúng và đáng tin cậy.
- Split: ngẫu nhiên 70/15/15, seed 42.
- Nhãn: `0=CLEAN`, `1=OFFENSIVE`, `2=HATE`.
- Không dùng test để tuning, chọn mô hình/ngưỡng hoặc sửa preprocessing; test chỉ được mở theo protocol EXP-08.

## Checklist TV1

### A. Chốt một phiên bản dữ liệu

- [ ] Xác nhận processed dataset đang xin duyệt được tạo từ UIT-ViHSD và nêu rõ file raw đầu vào; giải thích chênh lệch giữa raw files hiện có và processed records.
- [ ] Ghi revision/phiên bản nguồn và ngày tải nếu biết. Ghi license kèm link/bằng chứng nguồn; không suy đoán license.
- [ ] Ghi schema thực, cột text/label, mapping nhãn và số mẫu trước/sau từng bước xử lý.
- [ ] Ghi lệnh/script/config tái tạo processed dataset, split và manifest.
- [ ] Tính SHA-256 cho từng raw input, processed dataset và manifest; ghi hash, số records và ngày tạo vào `data/processed/dataset_info.json`.

### B. Khóa split 70/15/15, seed 42

- [ ] Xác nhận thuật toán chia ngẫu nhiên/phân tầng thực tế; nếu đã dùng stratified split, ghi rõ. Không gọi là stratified nếu không có bằng chứng.
- [ ] Xác minh tổng số và tỷ lệ từng split, phân phối từng lớp, và ID duy nhất.
- [ ] Xác minh manifest khớp chính xác với `split` trong từng record; mọi model dùng cùng hash manifest.
- [ ] Ghi rõ test hiện tại là split mới tạo từ dữ liệu nguồn, không phải official test gốc nếu đúng như metadata hiện tại.
- [ ] Lưu test riêng có kiểm soát truy cập; TV1–TV4 không dùng test để phát triển/tuning. Ghi người giữ test và điều kiện mở theo EXP-08. Nếu test đã được xem để phát triển, khai báo rõ với TV3 trước khi duyệt.

### C. Audit duplicate, overlap và nhãn mâu thuẫn

Bằng chứng kiểm đếm trước đó trên `data/processed/vihsd.jsonl`: **1.253 text xuất hiện ở nhiều split** và **241 text có nhiều label**. TV1 cần tái tạo kiểm đếm trên đúng phiên bản split sẽ đóng băng.

- [ ] Xuất báo cáo exact duplicate theo nhóm text, kèm ID, split và label; phân biệt trùng trong cùng split với trùng qua split.
- [ ] Xuất riêng các nhóm cross-split và conflicting-label; kiểm tra label gắn cùng text trước khi chọn cách xử lý.
- [ ] Đề xuất quy tắc xử lý có thể tái lập, không quyết định nhãn theo suy đoán. Xin TV3 duyệt trước khi thay đổi dữ liệu.
- [ ] Sau khi áp dụng quyết định được duyệt, tái tạo processed dataset/manifest/hash và chạy lại audit; ghi số mẫu/split/class trước-sau.
- [ ] Nếu còn overlap có chủ đích, ghi lý do và đánh giá nguy cơ; không để known leakage chưa được giải thích.
- [ ] Kiểm tra near-duplicate hoặc thread/group leakage nếu dữ liệu có metadata phù hợp; nếu không có, ghi rõ giới hạn và không tuyên bố đã kiểm tra được.

### D. Chốt preprocessing và data contract

- [ ] Ghi preprocessing/normalization bất biến, phiên bản và trường văn bản nào được dùng (giữ `text` gốc để truy vết).
- [ ] Xác nhận tokenizer/vocabulary, vectorizer, feature selection và mọi bước có học tham số chỉ fit trên train/train-fold.
- [ ] Chốt schema dùng chung: `id,text,label,split`; kiểu ID, tên split, đường dẫn data/manifest và quy ước prediction IDs.
- [ ] Ghi class/length statistics để các nhóm chọn config; các thay đổi sau này phải tăng version contract và hash.

### E. Gửi duyệt

- [ ] Cập nhật `docs/tv1_g0_report.md` bằng bằng chứng/đường dẫn cho từng mục; không đánh dấu hoàn tất chỉ dựa vào kế hoạch.
- [ ] Đính kèm provenance, split summary, hash, manifest, duplicate/conflicting-label report, EDA/preprocessing note và quyết định test.
- [ ] Gửi TV3 review để duyệt G0/Data Contract v1; sau khi TV3 duyệt, thông báo TV1–TV4 cùng version/hash.

## Điều kiện mở G0

TV3 chỉ xem xét mở G0 khi provenance của processed data truy vết được; split seed 42 và manifest tái lập/khớp; test được khóa; overlap và nhãn mâu thuẫn có kết quả audit cùng phương án được duyệt; preprocessing/contract rõ; artifacts và hash đồng bộ. Checklist này không thay thế phê duyệt của TV3 hoặc xác nhận Data Contract v1 của nhóm.

## Trạng thái hiện tại

- Tỷ lệ hiện tại được ghi nhận: train 23.380 (70%), val 5.010 (15%), test 5.010 (15%), tổng 33.400; manifest có 33.400 IDs.
- Các số liệu duplicate/overlap cần TV1 tái tạo trên artifact cuối cùng trước khi kết luận hoặc xử lý.
- **G0: CHƯA DUYỆT. Official training: CHƯA ĐƯỢC MỞ.** TV2–TV4 có thể tiếp tục code và smoke test trên fixture.
