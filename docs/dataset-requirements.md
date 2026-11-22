# Yêu cầu Dữ liệu & Quy trình Xác minh Dataset (Dataset Requirements)

Trạng thái tài liệu: **DRAFT / SPECIFICATION**  
Ngày cập nhật: 2026-11-11  
Dự án: **Hệ thống Nhận dạng Bình luận Độc hại (Vietnamese Toxic Comment Detection)**  
Phương pháp: **ML-first, reproducibility-first**

## Kiểm kê N2 ngày 2026-11-23

**N2 hoàn thành về tạo dữ liệu và kiểm tra cơ bản.** Người dùng xác nhận nhóm đã xin phép UIT cho đồ án; repo chưa có bằng chứng cấp phép để kiểm chứng độc lập. Các mục ba lớp bên dưới là lịch sử.

Sáu file trong `data/raw/` đọc được bằng UTF-8 và mỗi dòng là một JSON object. ViHSD có `id` (str), `text` (str), `label` (int), `split` (str). ViCTSD có `Comment`, `Title`, `Topic`, `_split` (str), `Toxicity`, `Constructiveness`, `Unnamed: 0`, `_idx` (int).

| File | Số dòng | Phân bố nhãn nguồn | SHA-256 |
|---|---:|---|---|
| `vihsd_train.jsonl` | 24048 | 0: 19886; 1: 1606; 2: 2556 | `42a4fb3a948e5a9018e0854a14653825437d10419e952704080835d75fc068e7` |
| `vihsd_validation.jsonl` | 2672 | 0: 2190; 1: 212; 2: 270 | `02155da4eafc7b7731da6ad2409463ce06ceab387302a0f369da101411404363` |
| `vihsd_test.jsonl` | 6680 | 0: 5548; 1: 444; 2: 688 | `f3f890659dae2fdace5ab3ae3229eeb642d543c887a0f1db46a15a294aa10f77` |
| `victsd_train.jsonl` | 7000 | 0: 6241; 1: 759 | `0638ff0a55b9cd3e786936b27cb56c36c719d98b453eea3aa04e5d5d31dbc935` |
| `victsd_validation.jsonl` | 2000 | 0: 1768; 1: 232 | `da4b0ef6a4087e3348e89dabc17645dd5349eed1f19287557b70fd3ba6676dc4` |
| `victsd_test.jsonl` | 1000 | 0: 890; 1: 110 | `522e9983d920c572a6001e84994dfa6b5cd56ee4c4de2e59a26ece895d358b32` |

ViHSD: metadata G0 ghi mirror `ura-hcmut/UIT-ViHSD` revision `16c4f67cf509d4f9f36ca5b63c5503c7c8830557`; ba hash raw khớp metadata G0. Chưa chứng minh thời điểm tải tương ứng revision này. [Mirror](https://huggingface.co/datasets/ura-hcmut/UIT-ViHSD) yêu cầu xin phép UIT trước khi dùng; tag CC-BY-NC-SA-4.0 của mirror không chứng minh quyền từ UIT. Trang `https://nlp.uit.edu.vn/datasets` chuyển tới trang chủ UIT và không cung cấp bằng chứng cấp phép cho dự án.

ViCTSD: [dataset card chính thức](https://huggingface.co/datasets/tarudesu/ViCTSD) ghi chỉ dùng cho nghiên cứu; không thấy giấy phép cụ thể. [API](https://huggingface.co/api/datasets/tarudesu/ViCTSD) trả revision `65a073f2c48401410b264213229a6c52417f367a` khi kiểm tra; chưa chứng minh raw local được tải từ revision đó. [Kho tác giả](https://github.com/tarudesu/ViCTSD) nhận đây là dataset chính thức. Quyền phân phối lại và sử dụng ngoài nghiên cứu chưa được xác minh.

Mapping áp dụng theo yêu cầu người dùng: ViHSD `0→0`, `1/2→1`; ViCTSD `Toxicity 0→0`, `1→1`. [Bài báo ViHSD](https://arxiv.org/html/2103.11528) định nghĩa OFFENSIVE theo harassment không nhắm đối tượng và HATE theo nội dung lạm dụng nhắm đối tượng. [Bài báo ViCTSD](https://arxiv.org/html/2103.10069) gộp cả mức quite toxic, gồm nội dung có thể gây hại trong ngữ cảnh, vào toxic. Hai định nghĩa có giao nhau nhưng không chứng minh tương đương; đây là giới hạn của nhãn gộp.

Kiểm tra exact text trên sáu file: 2698 nhóm trùng, gồm 2659 nhóm chỉ ViHSD, 33 nhóm chỉ ViCTSD và 6 nhóm giữa hai nguồn. Đã loại 205 nhóm, 466 bản ghi mang nhãn nhị phân mâu thuẫn: 203 nhóm ViHSD, 1 nhóm ViCTSD và 1 nhóm giữa hai nguồn. Không sửa raw. Các nhóm còn lại giữ một bản, manifest giữ liên kết mọi ID nguồn; source của bản giữ là nguồn của ID đầu tiên theo thứ tự từ điển.

[G0 review](G0_FINAL_REVIEW.md) ghi test của lần chia lại trước đã có trong artifacts chia sẻ. Chưa chứng minh split tương lai độc lập với dữ liệu đã lộ; không gọi split tương lai là blind. G0 là dữ liệu ba lớp, không phải dataset nhị phân mới.

Dataset nhị phân nằm tại `data/processed/binary_n2_v1/`: train 28231, val 6048, test 6048. Seed 42; 20 fold stratified theo `(source,label)`, gộp 14/3/3 fold và giữ các text trùng sau chuẩn hóa cùng split. `manifest.json` ánh xạ ID nguồn sang ID nội dung và split, hoặc đánh dấu conflict; `dataset_info.json` ghi hash và thống kê. Tái tạo vào thư mục chưa tồn tại bằng `.venv/Scripts/python.exe -m src.data.build_binary_dataset --output-dir <thư_mục_mới>`.

Kiểm tra sau tạo cho thấy không trùng exact text giữa split và không trùng sau NFKC, casefold và bỏ ký tự không thuộc `\w`. Phương pháp này không phát hiện mọi near-duplicate hoặc thread leakage. Test mới không blind do test cũ từng lộ.

---

## 1. Nguồn Dữ liệu Được Cung cấp (Input Datasets)

Theo file `link dataset.txt`, hệ thống có hai nguồn tham chiếu trên Hugging Face:

1. `https://huggingface.co/datasets/tarudesu/ViCTSD` (Vietnamese Constructive and Toxic Speech Detection)
2. `https://huggingface.co/datasets/ura-hcmut/UIT-ViHSD` (UIT Vietnamese Hate Speech Detection)

### Trạng thái dữ liệu hiện tại
- **[Verified]**: Link nguồn đã được cung cấp.
- **[Assumption]**: Dữ liệu ở định dạng văn bản tiếng Việt (comments/posts) thu thập từ mạng xã hội (Facebook/YouTube/diễn đàn).
- **[TBD]**: Chưa tải local vào repo, chưa inspect raw schema, chưa có thống kê số lượng mẫu và tỉ lệ phân bố nhãn.

---

## 2. Tiêu chí Đánh giá & So sánh 2 Dataset (Đề xuất)

| Tiêu chí | ViCTSD (`tarudesu/ViCTSD`) | UIT-ViHSD (`ura-hcmut/UIT-ViHSD`) | Ghi chú & Đánh giá |
|---|---|---|---|
| **Mục tiêu chính** | Nhận diện tính xây dựng (Constructiveness) và tính độc hại (Toxicity) | Phân loại mức độ thù ghét/xúc phạm (Clean, Offensive, Hate) | Cần xác nhận bài toán Binary hay Multi-class |
| **Nhãn dự kiến** | Toxicity (0/1) & Constructiveness (0/1) [TBD verify] | 3 nhãn: 0 (Clean), 1 (Offensive), 2 (Hate) [TBD verify] | UIT-ViHSD khớp taxonomy 3 lớp của đồ án |
| **Định dạng dữ liệu** | Hugging Face Dataset (Parquet / JSON / CSV) | Hugging Face Dataset (CSV / TSV / Parquet) | Cần script chuẩn hóa về schema chung |
| **Mức độ phổ biến trong NCKH** | Benchmark chuẩn VLSP 2021 | Benchmark nghiên cứu UIT-HCMUT | Cả 2 đều có tính học thuật cao, trích dẫn rõ ràng |

**Khuyến nghị chọn lựa [Proposed]**:
- **Phương án A (Ưu tiên)**: Dùng **UIT-ViHSD** làm tập chính (3-class Clean/Offensive/Hate khớp taxonomy).
- **Phương án B (Nghiên cứu sâu)**: Dùng UIT-ViHSD làm tập huấn luyện chính, ViCTSD làm cross-dataset evaluation (OOD generalization).
- **Quyết định chính thức**: Chốt sau EDA (EXP-001, M1).

---

## 3. Tiêu chí Chọn Dữ liệu (bắt buộc)

Nguồn dữ liệu phải đáp ứng:
- Có xuất xứ rõ ràng, đơn vị phát hành hoặc công trình nghiên cứu xác thực.
- Có thông tin giấy phép sử dụng.
- Có phương pháp gán nhãn đáng tin cậy.
- Có tính phù hợp với tiếng Việt mạng xã hội hiện tại.
- Nếu dùng dataset cũ: phải giải thích phạm vi sử dụng và đánh giá khả năng tổng quát hóa với ngôn ngữ mới.

**Không bịa nguồn dữ liệu hoặc giấy phép chưa được xác minh.**  
**Không thu thập dữ liệu trong phiên này.**

---

## 4. Schema Chuẩn hóa Kỳ vọng (Normalized Schema Standard)

Sau bước Data Ingestion & Preprocessing, dữ liệu phải chuyển về schema đồng nhất:

```json
{
  "id": "string | int",
  "text": "string (văn bản bình luận nguyên bản)",
  "cleaned_text": "string (văn bản sau khi chuẩn hóa ký tự, teencode, v.v.)",
  "label": "int (0, 1, 2)",
  "label_name": "string (CLEAN, OFFENSIVE, HATE)",
  "split": "string ('train' | 'val' | 'test')"
}
```

### Ràng buộc toàn vẹn dữ liệu
- `text` không được null, rỗng (`""`), hoặc chỉ chứa khoảng trắng / ký tự điều khiển vô nghĩa.
- `label` phải nằm trong `{0, 1, 2}`.
- Không được có duplicate text có nhãn mâu thuẫn mà không qua xử lý conflict resolution.

---

## 5. Checklist Kiểm tra Dữ liệu Sau khi Tải (EDA & Verification Checklist)

- [ ] **1. Tải và Lưu trữ Nguyên bản (Raw Ingestion)**
  - Tải dữ liệu từ Hugging Face Hub lưu vào `data/raw/<dataset_name>/`.
  - Không sửa đổi trực tiếp dữ liệu trong `data/raw/`.
- [ ] **2. Thống kê Cơ bản (Descriptive Statistics)**
  - Tổng số mẫu; phân bố theo train/val/test.
  - Phân bố nhãn (% CLEAN / OFFENSIVE / HATE) để xác định mức độ mất cân bằng.
  - Độ dài chuỗi (token/character): Min, Max, Median, p95 (dùng để cấu hình `max_length`).
- [ ] **3. Phân tích Dị biệt Ngôn ngữ Tiếng Việt**
  - Tỉ lệ teencode, viết tắt (vd: "đcm", "vl", "ko", "dc", "j").
  - Ký tự né filter / adversarial spelling (vd: "đ.m", "đ*m", "clm").
  - Icon / Emoji và vai trò ngữ nghĩa.
  - Chuẩn mã hóa Unicode (NFD vs NFC).
- [ ] **4. Kiểm tra Data Leakage & Overlap**
  - Kiểm tra trùng lặp văn bản giữa Train và Test/Val.
  - Kiểm tra trùng lặp ngữ cảnh / cùng thread thảo luận.
- [ ] **5. Ghi nhận Báo cáo EDA**
  - Xuất `docs/eda_report.md` kèm biểu đồ phân bố nhãn và độ dài.

---

## 6. Rủi ro & Chiến lược Xử lý

| Rủi ro | Mức độ | Giải pháp đề xuất |
|---|---|---|
| Class Imbalance nặng (HATE ít mẫu) | Cao | Class-weighted loss, Macro-F1, stratified sampling, không dùng Accuracy đơn lẻ. |
| Nhiễu gán nhãn (Label Noise) | Trung bình | EDA mẫu bất đồng thuận; đánh giá bằng threshold tuning. |
| Domain Mismatch khi demo API | Cao | Preprocessing linh hoạt; cảnh báo giới hạn mô hình (out-of-domain warning). |
| Data Leakage giữa Train/Val/Test | Chí mạng | Deduplicate trước khi split; kiểm tra overlap văn bản nếu dataset đã chia sẵn. |
| Dataset cũ không tổng quát ngôn ngữ mới | Trung bình | Đánh giá cross-dataset, ghi nhận giới hạn trong báo cáo. |
