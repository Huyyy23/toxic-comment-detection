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

## 8. Review G0 — 2026-11-21

**Quyết định: CHƯA DUYỆT G0; không mở official training.** Tỷ lệ 70/15/15 đã có trong artifact hiện tại, nhưng chỉ xác nhận số lượng theo split, không chứng minh split an toàn hay hợp lệ.

Đối chiếu `data/processed/vihsd.jsonl` với `manifest.json` và `dataset_info.json`:

- 33,400 records: train 23,380 (70%), val 5,010 (15%), test 5,010 (15%); manifest có 33,400 IDs.
- `dataset_info.json` ghi split được chia lại theo stratified 70/15/15, seed=42; split test hiện tại không còn là locked test gốc.
- Kiểm đếm exact text: 2,661 text xuất hiện nhiều lần; 1,253 text có mặt ở nhiều split; 241 text gắn nhiều hơn một label. Các overlap/conflicting labels chưa có báo cáo phân loại hoặc quyết định xử lý. Đây là blocker; không deduplicate hoặc đổi split âm thầm.
- Report chưa xác minh license/revision, và chưa ghi nhận bằng chứng kiểm tra group/thread leakage hay quyết định normalization được TV3 duyệt.
- Các raw JSONL hiện có không khớp số lượng/split của processed data; cần lập provenance thống nhất và chứng minh nguồn/hash cho đúng artifact đang xin duyệt.

**Điều kiện để gửi duyệt lại:** chốt và ghi lý do split lại (bao gồm xử lý test gốc/locked test theo protocol); báo cáo các overlap train/val/test và nhãn mâu thuẫn cùng phương án xử lý được TV3 duyệt; xác minh revision/license/schema/mapping; đối chiếu raw → processed bằng hash; hoàn thành contract và normalization approval theo G0 checklist. Không dùng test mới cho tuning; nếu test gốc đã bị lộ vào development, phải ghi nhận và xác lập test lock mới độc lập trước EXP-08.
