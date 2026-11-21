# TV1 — Báo cáo bàn giao G0 (EXP-00)

> **TV1 cung cấp; TV3 review và phê duyệt**  
> Ngày: 2026-11-21 | Trạng thái: **READY FOR TV3 REVIEW**

## 1. Thông tin dataset

| Mục | Giá trị |
|---|---|
| Tên dataset | UIT-ViHSD (Vietnamese Hate Speech Detection) |
| Nguồn | HuggingFace: `ura-hcmut/UIT-ViHSD` |
| URL | https://huggingface.co/datasets/ura-hcmut/UIT-ViHSD |
| Tổ chức | University of Information Technology, VNU-HCM |
| Giấy phép | CC BY-NC 4.0 (cho mục đích nghiên cứu học thuật) |
| Phương pháp gán nhãn | Crowdsourcing với multi-annotator agreement |

## 2. Schema nhãn

| Nhãn số | Tên | Ý nghĩa |
|---|---|---|
| 0 | CLEAN | Bình luận bình thường, không có nội dung độc hại |
| 1 | OFFENSIVE | Nội dung xúc phạm nhưng không nhắm vào nhóm cụ thể |
| 2 | HATE | Ngôn ngữ thù ghét nhắm vào cá nhân/nhóm |

**Mapping tương thích contract v4:** ViHSD nhãn gốc đã là 0/1/2 (`label_id`) $\rightarrow$ khớp 100% contract v4, không cần biến đổi.

## 3. Thống kê phân phối Split và Hash

| Split | Tổng số mẫu | CLEAN | OFFENSIVE | HATE | SHA-256 Hash |
|---|---|---|---|---|---|
| **Train** | 24,048 | 19,886 (82.7%) | 1,606 (6.7%) | 2,556 (10.6%) | `42a4fb3a948e5a9018e0854a14653825437d10419c952704080835d75fc068e7` |
| **Val** (dev) | 2,672 | 2,190 (82.0%) | 212 (7.9%) | 270 (10.1%) | `02155da4eafc7b7731da6ad2409463ce06ceab387302a0f369da101411404363` |
| **Test** (locked) | 6,680 | 5,548 (83.1%) | 444 (6.6%) | 688 (10.3%) | `f3f890659dae2fdace5ab3ae3229eeb642d543c887a0f1db46a15a294aa10f77` |
| **Tổng cộng** | 33,400 | 27,624 (82.7%) | 2,262 (6.8%) | 3,514 (10.5%) | — |

- File JSONL phát hành cho Development (Train + Val): `data/processed/vihsd.jsonl` (26,720 dòng)
- SHA-256 Hash của `vihsd.jsonl`: `fb6f075da1eb0f39990f1bd78eefd015c296e580f7b99920b8ee7905155aedf3`
- *Ghi chú bảo mật:* Tập Test hoàn toàn bị khóa và lưu tách biệt tại `data/raw/vihsd_test.jsonl`, không đưa vào development set.

## 4. Báo cáo Data Leakage & Trùng lặp
1. **Exact duplicate toàn bộ tập dữ liệu:** 3,586 mẫu có chuỗi văn bản trùng lặp (hiện tượng spam bình luận phổ biến trên mạng xã hội).
2. **Cross-split overlap (Train $\cap$ Val):** Phát hiện **318 mẫu text trùng lặp** giữa tập Train và Val của official split (ví dụ: các câu khẩu ngữ, trend mạng như *"Hóng B52 debut cùng J97"*, *"Thầy hãy acp fb em :(( "*, *"Fu thê"*).
   - *Quyết định:* Giữ nguyên official split để đảm bảo tính đối sánh công bằng tuyệt đối giữa 4 thành viên (LR, SVM, BiLSTM, TextCNN).

## 5. Checklist bàn giao

- [x] Source/revision/license đã xác minh (UIT-ViHSD, CC BY-NC 4.0)
- [x] Schema thực và mapping nhãn đã kiểm tra sample (khớp contract v4)
- [x] Raw data hash đã tính và lưu vết
- [x] Official split đã load và kiểm tra
- [x] Exact duplicate check hoàn thành (3,586 duplicates)
- [x] Cross-split text overlap check hoàn thành (318 mẫu overlap)
- [x] Class distribution có bằng chứng (82.7% CLEAN, 6.7% OFFENSIVE, 10.6% HATE)
- [x] Processed JSONL theo contract v4 đã xuất (`data/processed/vihsd.jsonl`)
- [x] Manifest id→split đã tạo (`data/processed/manifest.json`)
- [x] dataset_info.json với provenance đầy đủ đã lưu

## 6. Files bàn giao

```
data/raw/vihsd_train.jsonl        # Dữ liệu gốc train (24,048 samples)
data/raw/vihsd_validation.jsonl   # Dữ liệu gốc validation (2,672 samples)
data/raw/vihsd_test.jsonl         # Dữ liệu gốc test (6,680 samples - locked)
data/processed/vihsd.jsonl        # Contract v4 train+val (26,720 samples)
data/processed/manifest.json      # Mapping id -> split
data/processed/dataset_info.json   # Provenance & hashes
```

## 7. Đề xuất cho TV3 (Trưởng nhóm)
- Gate G0 đủ điều kiện phê duyệt để mở khóa huấn luyện chính thức cho cả 4 nhánh: TV1 (LR), TV2 (SVM), TV3 (BiLSTM), TV4 (TextCNN).
