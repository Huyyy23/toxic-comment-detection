# Yêu cầu Dữ liệu & Quy trình Xác minh Dataset (Dataset Requirements)

Trạng thái tài liệu: **DRAFT / SPECIFICATION**  
Ngày cập nhật: 2026-11-11  
Dự án: **Hệ thống Nhận dạng Bình luận Độc hại (Vietnamese Toxic Comment Detection)**  
Phương pháp: **ML-first, reproducibility-first**

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
