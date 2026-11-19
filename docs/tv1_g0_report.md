# TV1 — Báo cáo bàn giao G0 (EXP-00)

> **TV1 cung cấp; TV3 review và phê duyệt**  
> Ngày: 2026-11-20 | Trạng thái: IN PROGRESS

## 1. Thông tin dataset

| Mục | Giá trị |
|---|---|
| Tên dataset | UIT-ViHSD (Vietnamese Hate Speech Detection) |
| Nguồn | HuggingFace: `ura-hcmut/UIT-ViHSD` |
| URL | https://huggingface.co/datasets/ura-hcmut/UIT-ViHSD |
| Tổ chức | University of Information Technology, VNU-HCM |
| Giấy phép | Cần xác minh tại trang HuggingFace (thường CC BY-NC 4.0 cho nghiên cứu) |
| Phương pháp gán nhãn | Crowdsourcing với multi-annotator agreement |

## 2. Schema nhãn

| Nhãn số | Tên | Ý nghĩa |
|---|---|---|
| 0 | CLEAN | Bình luận bình thường, không có nội dung độc hại |
| 1 | OFFENSIVE | Nội dung xúc phạm nhưng không nhắm vào nhóm cụ thể |
| 2 | HATE | Ngôn ngữ thù ghét nhắm vào cá nhân/nhóm |

**Mapping tương thích contract v4:** ViHSD nhãn gốc đã là 0/1/2 → không cần chuyển đổi.

## 3. Split

- Dùng **official split** từ HuggingFace (train/dev/test)
- `dev` → map thành `val` trong contract v4
- Locked test không phát hành cho development

## 4. Checklist bàn giao

- [ ] Source/revision/license đã xác minh
- [ ] Schema thực và mapping nhãn đã kiểm tra sample
- [ ] Raw data hash đã tính
- [ ] Official split đã load và kiểm tra
- [ ] Exact duplicate check hoàn thành
- [ ] Cross-split text overlap check hoàn thành  
- [ ] Class distribution có bằng chứng (thống kê)
- [ ] Processed JSONL theo contract v4 đã xuất
- [ ] Manifest id→split đã tạo
- [ ] dataset_info.json với provenance đã lưu

## 5. Files bàn giao

```
data/raw/vihsd_train.jsonl      # Dữ liệu gốc train
data/raw/vihsd_dev.jsonl        # Dữ liệu gốc dev  
data/processed/vihsd.jsonl      # Schema contract v4 (train+val)
data/processed/manifest.json    # id→split mapping
data/processed/dataset_info.json # Provenance đầy đủ
```

## 6. Trạng thái

> **PENDING** — Chờ chạy `python -m src.data.download_dataset` với mạng  
> Sau khi chạy: điền hash, thống kê, và gửi TV3 review để đóng G0.

## 7. ViCTSD (phụ)

ViCTSD (`tarudesu/ViCTSD`) có taxonomy khác (Constructive/Toxic/Severe Toxic).  
**Chưa gộp vào training** — chỉ dùng OOD/đối chiếu nếu cần, sau khi xác minh ánh xạ nhãn.
