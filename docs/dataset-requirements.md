# Yêu cầu Dữ liệu & Quy trình Xác minh Dataset (Dataset Requirements)

Trạng thái tài liệu: **DRAFT / SPECIFICATION**  
Ngày lập: 2026-11-11  
Dự án: **Hệ thống Nhận dạng Bình luận Độc hại (Vietnamese Toxic Comment Detection)**

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

## 2. Tiêu chí Đánh giá & So sánh Sơ bộ 2 Dataset (Đề xuất)

| Tiêu chí | ViCTSD (`tarudesu/ViCTSD`) | UIT-ViHSD (`ura-hcmut/UIT-ViHSD`) | Ghi chú & Đánh giá |
| :--- | :--- | :--- | :--- |
| **Mục tiêu chính** | Nhận diện tính xây dựng (Constructiveness) và tính độc hại (Toxicity) | Phân loại mức độ thù ghét/xúc phạm (Clean, Offensive, Hate) | Cần quyết định bài toán là Binary hay Multi-class/Multi-label |
| **Nhãn dự kiến** | Thường là nhãn Toxicity (0/1) & Constructiveness (0/1) [TBD verify] | Thường là 3 nhãn: 0 (Clean), 1 (Offensive), 2 (Hate) [TBD verify] | Tùy mục tiêu bài toán thực tế của đồ án |
| **Định dạng dữ liệu** | Hugging Face Dataset (Parquet / JSON / CSV) | Hugging Face Dataset (CSV / TSV / Parquet) | Cần script chuẩn hóa về schema chung |
| **Mức độ phổ biến trong NCKH** | Benchmark chuẩn tại VLSP 2021 | Benchmark chuẩn từ nhóm nghiên cứu UIT-HCMUT | Cả 2 đều có tính học thuật cao, trích dẫn rõ ràng |

**Khuyến nghị chọn lựa [Proposed]**:
- **Phương án A (Ưu tiên MVP đơn giản, tập trung vào Toxic Detection)**: Dùng **UIT-ViHSD** với bài toán 3-class (Clean vs Offensive vs Hate) hoặc map về Binary (Clean vs Toxic/Hate) HOẶC dùng **ViCTSD** tập trung vào cột nhãn `Toxicity`.
- **Phương án B (Nghiên cứu sâu)**: Sử dụng một dataset làm tập huấn luyện chính, dataset còn lại làm Cross-dataset evaluation (kiểm tra Out-of-Distribution generalization).
- **Quyết định chính thức**: Sẽ chốt sau giai đoạn EDA (Bước 1 của Implementation Plan).

---

## 3. Schema Chuẩn hóa Kỳ vọng (Normalized Schema Standard)

Dù dataset raw có cấu trúc cột như thế nào, ML Pipeline sau bước Data Ingestion & Preprocessing **phải** chuyển đổi về một schema đồng nhất:

```json
{
  "id": "string | int",
  "text": "string (văn bản bình luận nguyên bản)",
  "cleaned_text": "string (văn bản sau khi chuẩn hóa ký tự, teencode, v.v.)",
  "label": "int (0, 1, ...) hoặc list[int] nếu multi-label",
  "label_name": "string (Clean, Toxic, Hate, ...)",
  "split": "string ('train' | 'val' | 'test')"
}
```

### Các ràng buộc toàn vẹn dữ liệu (Data Integrity Constraints)
- `text` không được null, rỗng (`""`), hoặc chỉ chứa khoảng trắng / ký tự điều khiển vô nghĩa.
- `label` phải nằm trong tập nhãn hợp lệ được định nghĩa trước.
- Không được có duplicate text có nhãn mâu thuẫn (cùng text nhưng 1 record gán 0, 1 record gán 1) mà không qua xử lý conflict resolution.

---

## 4. Checklist Kiểm tra Dữ liệu Sau khi Tải (EDA & Verification Checklist)

Agent hoặc Kỹ sư thực hiện tải dataset **bắt buộc** phải hoàn thành checklist sau trước khi bắt đầu code pipeline huấn luyện:

- [ ] **1. Tải và Lưu trữ Nguyên bản (Raw Ingestion)**
  - Tải dữ liệu từ Hugging Face Hub lưu vào thư mục `data/raw/<dataset_name>/`.
  - Không sửa đổi trực tiếp dữ liệu trong `data/raw/`.
- [ ] **2. Thống kê Cơ bản (Descriptive Statistics)**
  - Tổng số lượng mẫu (total instances).
  - Phân bổ mẫu theo tập train/val/test có sẵn (nếu dataset đã chia sẵn) hoặc kiểm tra split.
  - Phân bố nhãn (Class distribution): Tỉ lệ % của từng nhãn để xác định mức độ mất cân bằng (Class Imbalance Ratio).
  - Độ dài chuỗi (Token/Character length distribution): Min, Max, Median, 95th-percentile (để cấu hình `max_length` cho tokenizer).
- [ ] **3. Phân tích Dị biệt Ngôn ngữ Tiếng Việt (Vietnamese NLP Anomalies)**
  - Tỉ lệ Teencode, viết tắt (vd: "đcm", "vl", "vcl", "ko", "dc", "j", "hok").
  - Ký tự cố tình né filter (Obfuscation / Adversarial spelling, vd: "đ.m", "đ*m", "clm", "d!t").
  - Icon / Emoji cảm xúc và vai trò ngữ nghĩa (toxic hay non-toxic).
  - Chuẩn mã hóa Unicode: Kiểm tra xung đột giữa Unicode Tổ hợp (NFD) và Unicode Dựng sẵn (NFC).
- [ ] **4. Kiểm tra Data Leakage & Overlap**
  - Kiểm tra trùng lặp văn bản giữa tập Train và tập Test/Val.
  - Kiểm tra trùng lặp ngữ cảnh hoặc cùng một thread thảo luận.
- [ ] **5. Ghi nhận Tài liệu EDA**
  - Xuất báo cáo tóm tắt ra `docs/eda_report.md` kèm biểu đồ phân bố nhãn và phân bố độ dài.

---

## 5. Các Rủi ro & Chiến lược Xử lý Liên quan đến Dữ liệu

| Rủi ro | Mức độ ảnh hưởng | Giải pháp đề xuất |
| :--- | :--- | :--- |
| **Class Imbalance nặng** (vd: Toxic chỉ chiếm < 10%) | Cao (mô hình dự đoán lệch về đa số) | Áp dụng Class Weight trong Loss function (Focal Loss hoặc Weighted Cross-Entropy); stratified sampling; theo dõi Macro-F1 / PR-AUC thay vì Accuracy. |
| **Nhiễu gán nhãn (Label Noise / Subjectivity)** | Trung bình (độc hại mang tính chủ quan của annotator) | EDA mẫu bất đồng thuận; loại bỏ hoặc giữ nguyên có ghi nhận; đánh giá mô hình bằng threshold tuning. |
| **Domain Mismatch khi Demo Web** | Cao (người dùng web nhập văn bản phong cách khác dữ liệu train) | Preprocessing module linh hoạt; cảnh báo giới hạn mô hình trên UI (Out-of-domain warning). |
| **Data Leakage giữa Train/Val/Test** | Chí mạng (kết quả đánh giá ảo) | Luôn deduplicate trước khi split; nếu dùng split có sẵn của tác giả, kiểm tra overlap văn bản giữa các split. |
