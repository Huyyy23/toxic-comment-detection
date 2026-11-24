# EXP05-REVIEW-V2 — checklist xác nhận thủ công

**Checklist status: HUMAN REVIEW PENDING. N15 scope decision (group lead): LIMITED TECHNICAL ACCEPTANCE** — Scope is validation error analysis supported by AI using existing validation artifacts. This acceptance does not mean member case sign-off, gold-label adjudication, privacy/legal clearance, dataset-rights approval or comprehensive leakage certification; these remain open.

**Group-lead report:** The user reports that TV1–TV4 completed their assigned work. This is recorded as group-level progress only; it does not identify who reviewed these 30 cases, record case-level decisions, or establish category agreement/privacy approval. Keep individual fields below blank until corresponding evidence is supplied.

## Phạm vi và cách xem mẫu

- Sample mới `EXP05-REVIEW-V2`, chỉ lấy từ validation seed 42: 10 mẫu cả hai sai, 10 TRI-only sai, 10 PhoBERT-only sai. Đây **không** phải sample 30 cũ; membership của sample cũ không thể khôi phục vì không có manifest/ID selection.
- Bộ phân tầng: mỗi error group lấy đúng 10, quota proportional largest remainder theo `(lowercase(source), gold label)`; floors trước, slot còn lại theo remainder giảm dần, ties theo source rồi label; chọn ID theo SHA-256 của UTF-8 `str(id)`, sau đó đánh số theo error group và hash. Fingerprint có thứ tự: `33c19e709c013b8dc7be740687635e26e4215ee37727becb621495dffbfde2bb`.
- Sample selection uses historical saved seed-42 predictions. Retraining/replay does not change its membership or assistant-provisional categories; no human review or sample from retrained predictions is implied here.
- Validation-only source: `data/processed/binary_n2_v1/val.parquet` và prediction seed 42 của `V3-TRI-GATED-FT` / `V3-PHOBERT-V2`. Không mở test data/predictions/labels; không dùng lỗi test để điều chỉnh model.
- Bảng bên dưới chứa nguyên văn validation đã **che tên/biệt danh/định danh trực tiếp** để TV1–TV4 soát thuận tiện; không lưu sample ID. Đây là bản nội bộ N15, không phải phê duyệt quyền phân phối hay privacy clearance. Không đăng công khai, không commit/push lên remote public trước khi chủ sở hữu xác nhận quyền dùng/phân phối. Nếu cần đối chiếu các từ đã che hoặc ngữ cảnh gốc, mở `review_sample` trong notebook trên máy được phép; không lưu cell output, export text hoặc ID.
- Cột “category/confidence” bên dưới là chỗ reviewer tự ghi sau khi đọc; không dùng category của assistant làm nhãn chuẩn. Báo cáo assistant provisional: [`error_analysis.md`](error_analysis.md#exp05-review-v2--provisional-review).

## Mẫu review theo từng member

Trong ô reviewer ghi `agree/disagree/abstain`, category tự đánh giá và confidence (high/medium/low). Không ghi ID hoặc thêm nguyên văn/chi tiết nhận dạng vào phần xác nhận.
**Quy trình review rút gọn — đề xuất, chưa được nhóm chấp thuận:** mỗi mẫu có 2 đánh giá độc lập. Pair theo `(n−1) mod 6`: `0→TV1+TV2`, `1→TV1+TV3`, `2→TV1+TV4`, `3→TV2+TV3`, `4→TV2+TV4`, `5→TV3+TV4`. Lặp 5 chu kỳ, nên mỗi thành viên review 15/30 mẫu. Chỉ hai cột reviewer được pair chọn điền; các cột khác để trống. Chỉ disagreement/abstain hoặc gold/rubric conflict chuyển sang adjudication; không tự đổi gold. Chỉ áp dụng sau khi nhóm chấp thuận; đây không phải phân công hay sign-off.

| # | Error group | Source / gold | TV1 — Tâm | TV2 — Vinh | TV3 — Huy | TV4 — Thế Anh |
|---:|---|---|---|---|---|---|
| 1 | Both wrong | ViCTSD / 1 | ____ | ____ | ____ | ____ |
| 2 | Both wrong | ViCTSD / 1 | ____ | ____ | ____ | ____ |
| 3 | Both wrong | ViHSD / 1 | ____ | ____ | ____ | ____ |
| 4 | Both wrong | ViHSD / 0 | ____ | ____ | ____ | ____ |
| 5 | Both wrong | ViHSD / 1 | ____ | ____ | ____ | ____ |
| 6 | Both wrong | ViHSD / 1 | ____ | ____ | ____ | ____ |
| 7 | Both wrong | ViHSD / 1 | ____ | ____ | ____ | ____ |
| 8 | Both wrong | ViHSD / 1 | ____ | ____ | ____ | ____ |
| 9 | Both wrong | ViHSD / 0 | ____ | ____ | ____ | ____ |
| 10 | Both wrong | ViCTSD / 1 | ____ | ____ | ____ | ____ |
| 11 | TRI-only wrong | ViHSD / 1 | ____ | ____ | ____ | ____ |
| 12 | TRI-only wrong | ViCTSD / 1 | ____ | ____ | ____ | ____ |
| 13 | TRI-only wrong | ViHSD / 1 | ____ | ____ | ____ | ____ |
| 14 | TRI-only wrong | ViHSD / 1 | ____ | ____ | ____ | ____ |
| 15 | TRI-only wrong | ViCTSD / 0 | ____ | ____ | ____ | ____ |
| 16 | TRI-only wrong | ViHSD / 1 | ____ | ____ | ____ | ____ |
| 17 | TRI-only wrong | ViCTSD / 1 | ____ | ____ | ____ | ____ |
| 18 | TRI-only wrong | ViHSD / 1 | ____ | ____ | ____ | ____ |
| 19 | TRI-only wrong | ViHSD / 0 | ____ | ____ | ____ | ____ |
| 20 | TRI-only wrong | ViHSD / 0 | ____ | ____ | ____ | ____ |
| 21 | PhoBERT-only wrong | ViHSD / 0 | ____ | ____ | ____ | ____ |
| 22 | PhoBERT-only wrong | ViHSD / 0 | ____ | ____ | ____ | ____ |
| 23 | PhoBERT-only wrong | ViCTSD / 0 | ____ | ____ | ____ | ____ |
| 24 | PhoBERT-only wrong | ViHSD / 1 | ____ | ____ | ____ | ____ |
| 25 | PhoBERT-only wrong | ViHSD / 0 | ____ | ____ | ____ | ____ |
| 26 | PhoBERT-only wrong | ViCTSD / 0 | ____ | ____ | ____ | ____ |
| 27 | PhoBERT-only wrong | ViHSD / 0 | ____ | ____ | ____ | ____ |
| 28 | PhoBERT-only wrong | ViHSD / 0 | ____ | ____ | ____ | ____ |
| 29 | PhoBERT-only wrong | ViHSD / 1 | ____ | ____ | ____ | ____ |
| 30 | PhoBERT-only wrong | ViHSD / 1 | ____ | ____ | ____ | ____ |

## Checklist mỗi reviewer

- [ ] Tôi đã xác nhận đây là sample `EXP05-REVIEW-V2`, không phải sample cũ.
- [ ] Tôi đã review độc lập các dòng được phân công nếu nhóm chấp thuận rotation đề xuất; không mở test.
- [ ] Với các dòng được giao, tôi đã tự kiểm tra gold, hai prediction, category và confidence; không coi model disagreement là bằng chứng gold sai.
- [ ] Tôi đã ghi bất đồng/không chắc chắn, không suy luận intent hoặc causal mechanism từ text thiếu context.
- [ ] Tôi không sao chép **nguyên văn chưa ẩn danh**, record ID, URL, handle, screenshot hay dữ liệu nhận dạng vào phần xác nhận; bảng nội bộ chỉ dùng nội dung đã rà soát/che tên.
- [ ] Tôi xác nhận bảng nội dung đã được screening kỹ thuật sơ bộ; đây không phải privacy/legal clearance, và quyền sử dụng/phân phối dữ liệu vẫn cần chủ sở hữu xác nhận.
- [ ] Tôi xác nhận các giới hạn về taxonomy khác nhau giữa nguồn, permission, duplicate/thread leakage, validation reuse, và việc replay checkpoint retrained không khôi phục original weights.

| Member | Reviewed / decision | Date | Category agreement / caveat | Privacy review reference (no text/ID) |
|---|---|---|---|---|
| TV1 — Tâm | ____ | ____ | ____ | ____ |
| TV2 — Vinh | ____ | ____ | ____ | ____ |
| TV3 — Huy | ____ | ____ | ____ | ____ |
| TV4 — Thế Anh | ____ | ____ | ____ | ____ |

## Tổng hợp nhóm — để trống đến khi thành viên phản hồi

- Categories được thống nhất / sửa / để mở: ____________________
- Bất đồng về gold label hoặc rubric và cách xử lý: ____________________
- Privacy reviewer và paraphrase an toàn (nếu có, không ghi text/ID): ____________________
- Tuyên bố tái lập được chấp nhận/sửa: ____________________
- Dissent, limitation còn mở: ____________________
- N15 decision và ngày xác nhận: ____________________

## Evidence cần xác nhận — chưa có approval

| Nội dung | Trạng thái | Bằng chứng hiện có |
|---|---|---|
| V3 candidates / validation counts | Evidence-backed | `error_analysis.md`, saved validation predictions/configs/metrics. |
| Limited N15 scope | Accepted by group lead | Technical scope only: validation error analysis with AI support; no member sign-off or gold adjudication inferred. |
| Case categories and causes | Provisional AI interpretation; 3-arm review complete | Updated row-level synthesis below; member adjudication still pending. |
| Technical source/gold/model fields, 30 rows | Evidence-backed | Deterministic validation-only reconstruction and sample fingerprint check match all 30 source/gold/prediction/P(TOXIC) entries; no IDs are persisted. |
| Independent AI review | Completed; not human approval | Three full-coverage review arms, 30 rows each, ascending order; rank-all synthesis below. Interpretations remain provisional until member review. |
| Exact V3 weight recovery | Historical originals unavailable; retrain separate | 16 checkpoints mới được lưu riêng; không khôi phục hoặc thay thế weights gốc. Xem `reproducibility_report.md`. |
| Inference replay | Evidence-backed, limited | 16 checkpoint retrained replay exact labels trên validation; max probability diff `0.00011591`. Không chứng minh replay original checkpoints. |
| Dataset rights / label equivalence / leakage | Partially checked; owner evidence pending | Exact train–val text overlap: 0; IDs giao nhau: 0. One empty normalized key bucket (160 train/28 val) is not a raw-text match; see builder content-ID fallback note in `dataset-requirements.md`. 30 mẫu không có exact/non-empty normalized match với train; char n-gram screening không có cặp ≥0.85 (max 0.737). Near-duplicate tổng quát/thread leakage và quyền dataset chưa được xác minh. |
| Test/blindness | Historical exposure; no new test access | `../../artifacts/TV4-locked-test/tv4_n10_reconciliation.md`; do not claim blind test. |

## Tổng hợp swarm — independent AI review

- **Shape:** 3 identical full-coverage review arms; **rank all**. Mỗi arm review 30 hàng theo thứ tự 1–30, base SHA `f676e3fd86f3f8f1c1ab61a989ec0a034abfdced`, trên live working tree (checklist/docs có thay đổi chưa commit). Cả 3 trả `ISSUES` về diễn giải, không phải lỗi kỹ thuật; không dropout. Không tuyên bố các arm dùng model khác nhau.
- **Kỹ thuật:** arm A và kiểm tra deterministic của parent xác nhận cả 30 source/gold/nhãn dự đoán/P(TOXIC)/thứ tự khớp artifacts validation seed 42. Arm B/C không nối được hàng 7 và 25 chỉ từ text đã che; đây không phải bằng chứng sai số. Không có ID được ghi vào checklist.
- **Điều chỉnh hội tụ:** bỏ “lexical false alarm” ở các false-negative/gold-context cases 2, 5, 17 và 23; thay nhận định nguyên nhân chưa có bằng chứng ở 15, 26, 27, 28; sửa category/uncertainty ở 6, 9, 12, 14, 16, 19, 22, 25, 29, 30. Rows 1/14 dùng mức thận trọng low/high; bảng draft cũ trong `error_analysis.md` được đánh dấu superseded. Gold chưa được adjudicate.
- **Ưu tiên người review:** (1) gold/rubric conflict 4, 9, 22; (2) direct/group-targeted interpretation 3, 12, 18, 24; (3) ngữ cảnh/giải mã 1, 6–10, 14, 16, 18–21, 23, 25, 27, 30; (4) taxonomy/source mapping hoặc target scope 2, 5, 11–13, 15, 17, 24, 26, 28, 29. Đây là ưu tiên, không miễn review các dòng được pair giao.
- **Giới hạn:** nhận định AI không xác lập nguyên nhân lỗi hoặc gold label đúng/sai. Privacy/legal clearance và quyền dataset còn pending; không suy diễn member approval.

**The worksheet remains open** for member review, case adjudication, privacy/data-rights clearance and unresolved data limitations. These do not extend the accepted N15 technical scope or imply member approval.

## Bảng 30 mẫu — nội dung đã che định danh, dự đoán và hướng review

**Nguồn:** validation N2, cặp prediction seed 42 lịch sử `V3-TRI-GATED-FT` và `V3-PHOBERT-V2`; đúng fingerprint ở trên, 10 mẫu mỗi nhóm lỗi. Nhãn thực tế là `gold`, không phải nhãn đã adjudicate lại. `0=CLEAN`, `1=TOXIC`. Xác suất là `P(TOXIC)` từ artifact; `P(CLEAN)=1−P(TOXIC)`. Hai model outputs có `is_calibrated=false`; không diễn giải xác suất như confidence đã calibration hoặc so sánh trực tiếp như cùng thang đo.

**Privacy screen:** không phát hiện URL/email/handle dạng `@`/phone/chuỗi số dài trong 30 mẫu; tên/biệt danh khả dĩ ở các dòng 3, 6, 7, 18, 19, 22, 24, 25 đã được che. Screening thủ công vẫn có thể bỏ sót danh tính gián tiếp. Đây không phải privacy/legal clearance; quyền sử dụng/phân phối dataset chưa được xác minh. Giữ nội bộ nhóm, không xuất bản.

| # | Nội dung validation (nguyên văn trừ tên/định danh đã che) | Source / gold | TRI pred / P(TOXIC) | PhoBERT pred / P(TOXIC) | Nhận định sơ bộ / đề xuất đánh giá |
|---:|---|---:|---:|---:|---|
| 1 | Xong rồi sao | ViCTSD / 1 | 0 / 0.004636 | 0 / 0.033558 | Indirect/context-dependent; thấp / cao. Câu quá ngắn, xác suất cả hai rất thấp. Cần ngữ cảnh trước/sau; không kết luận gold sai chỉ từ câu đứng riêng. |
| 2 | nghe câu Có Cơ Hội Đòi Lại Món Nợ mà có vẻ Hư Cấu Quá , Cả Liverpool và Bayern hiện tại đều cửa trên so với Barcelona , họ ko những mạnh mà còn Bản Lĩnh | ViCTSD / 1 | 0 / 0.063653 | 0 / 0.138316 | Possible source-label/context mismatch; medium/high. Excerpt reads as sports commentary without obvious abuse, but both models miss gold TOXIC. Check authorized context and ViCTSD label scope; do not call this a model false alarm or relabel gold. |
| 3 | [Biệt danh đã ẩn] là chị chó phèn ak haha | ViHSD / 1 | 0 / 0.064341 | 0 / 0.254642 | Direct abuse; cao / thấp. Có từ miệt thị hướng tới người được nhắc. Reviewer xác định mục tiêu và mức công kích theo rubric; không suy luận danh tính. |
| 4 | Con quỷ này nó dở quá a c e  đừng nói tới nói mở miệng ra là cái đó không | ViHSD / 0 | 1 / 0.798638 | 1 / 0.945349 | Direct abuse apparent; high/medium. The CLEAN gold conflicts with visible insult. Check quotation/target/context and source rubric; do not change gold automatically. |
| 5 | Hoặc do xe vinfast dễ gây tai nạn cho các xe khác khi cùng lưu thông; Hoặc chính quyền dùng quyền lực xoá lợi ích người dân để phục vụ cho lợi ích nhóm? | ViHSD / 1 | 0 / 0.007694 | 0 / 0.399929 | Possible source-label/context mismatch; medium/high. Visible policy/business criticism, while both models miss gold TOXIC. Review source criteria and context; no false-alarm explanation or gold correction without adjudication. |
| 6 | [Biệt danh/handle đã ẩn] đi chơi gái bị tống à :)) | ViHSD / 1 | 0 / 0.006757 | 0 / 0.059129 | Sexualized insinuation; low/high. Target and context are obscured; assess whether it is degrading/targeted without treating the insinuation as fact. |
| 7 | [Tên đã ẩn] bồ của [tên đã ẩn] corona | ViHSD / 1 | 0 / 0.008134 | 0 / 0.275446 | Other/unclear; thấp / cao. Câu rời, có tên riêng đã che và tham chiếu COVID mơ hồ. Đề nghị abstain nếu không có ngữ cảnh; kiểm tra rủi ro nhắm cá nhân trước khi ghi category. |
| 8 | Lỡ rồi múc luôn đi. Mấy ảnh về lấy tiền của dân mua lại =]] | ViHSD / 1 | 0 / 0.014578 | 0 / 0.092955 | Context-dependent idiom/target; medium/high. Determine “múc” and “mấy ảnh” from permitted context; do not infer a threat from the excerpt alone. |
| 9 | Chưởi bk chó là nó sắp tới số nữa con này khg bc loại nào ln | ViHSD / 0 | 1 / 0.942668 | 1 / 0.856002 | Abusive wording / possible threat; medium/high. Abbreviations and target are unclear, but the CLEAN gold conflicts with the apparent insult. Check decoding, quotation/context and source rubric; preserve gold pending adjudication. |
| 10 | Mọi thứ đều đẹp trừ 2 quả thận | ViCTSD / 1 | 0 / 0.009864 | 0 / 0.313646 | Indirect/context-dependent; vừa / cao. Có thể là đùa đen/ám chỉ nguy cơ lấy nội tạng. Cần context để phân biệt ẩn dụ với nội dung đe dọa hoặc gây hại. |
| 11 | Chúng nó đang bận vơ vét thanh lý đất nước này | ViHSD / 1 | 0 / 0.091869 | 1 / 0.942348 | Indirect/context-dependent; vừa / vừa. TRI bỏ sót lời cáo buộc gay gắt, PhoBERT nhận TOXIC. Xác định “chúng nó” chỉ ai và áp dụng nhất quán giữa phê bình chính trị với công kích cá nhân/nhóm. |
| 12 | Điều này chỉ chứng tỏ thiệt thòi người tiêu dùng hàn, khi không được sở hữu xe nhật đẳng cấp thế giới, mà phải sử dụng xe hàn kém mọi mặt, từ thương hiệu đến đẳng cấp, hạ tầm đẳng cấp con người. | ViCTSD / 1 | 0 / 0.077383 | 1 / 0.701764 | Group-targeted derogation; medium/medium. Review nationality-group target and source binary mapping; model agreement does not establish gold correctness. |
| 13 | Cướp đêm là giặc, cướp ngày là quan. 24/7 lúc nào cũng phải nơm nớp. Than ôi cái đất nước hạnh phúc | ViHSD / 1 | 0 / 0.049088 | 1 / 0.677154 | Indirect/context-dependent; vừa / vừa. Lời châm biếm chính quyền/xã hội, không rõ có nhắm cá nhân hay không. Đánh giá theo phạm vi nhãn nguồn và ghi nhận khác biệt taxonomy. |
| 14 | Sat mui vo lon no co hieu no vua sau vua nao yeu nua | ViHSD / 1 | 0 / 0.449839 | 1 / 0.788013 | Other/unclear; possible insult only if decoding is correct; low/high. Nonstandard unaccented wording needs fluent-speaker/context review before category agreement. |
| 15 | Đau lòng cho những người làm cha, làm mẹ. Giờ đây một bên thì lo đám tang, một bên thì lo đi hầu tòa và bồi thường. | ViCTSD / 0 | 1 / 0.623441 | 0 / 0.184859 | Possible model false positive; medium/low. Excerpt appears sympathetic; check source context and TRI’s prediction, without asserting a lexical cause or changing gold. |
| 16 | Kéo đàn violin thì làm đếch gì được. | ViHSD / 1 | 0 / 0.279810 | 1 / 0.830216 | Profanity/dismissive language; target unclear; medium/high. Apply the ViHSD rubric to untargeted profanity; do not assume direct person-directed abuse. |
| 17 | Bộ giáo dục thì bảo cấm không cho trẻ lớp lá học chữ trước nhưng thử hỏi không học trước mà vô lớp 1 học theo kịp các bạn tui khen thiên tài. Nghĩ sao mới học tuần 4 thôi mà đã phải đọc 1 đoạn văn về bài "Su su" 22 chữ rồi?. Thật là thiếu tính logic và thực tế. | ViCTSD / 1 | 0 / 0.322743 | 1 / 0.853577 | Policy criticism / possible source-label-context mismatch; medium/high. TRI predicts CLEAN and PhoBERT TOXIC; check omitted context and source label scope, not a lexical false alarm. |
| 18 | Con dĩ nào bên có gì [biệt danh đã ẩn] có đó :)) đẳng ngang với di kkk | ViHSD / 1 | 0 / 0.008114 | 1 / 0.835752 | Direct abuse; vừa / vừa. Có từ miệt thị giới tính và tham chiếu khó hiểu. Reviewer xác định mục tiêu, ngữ cảnh và mức chắc chắn; tránh diễn giải phần viết tắt quá mức. |
| 19 | [Biệt danh đã ẩn] .bản lĩnh con cháu hai bà trưng bà triệu👍🙏❤🌍 | ViHSD / 0 | 1 / 0.809748 | 0 / 0.078997 | Possible model false positive; low/high. Visible text appears positive, but the redacted opening could change tone. Review permitted context; do not infer sarcasm or identity. |
| 20 | Răng kêu ngột ngạt đồ | ViHSD / 0 | 1 / 0.531843 | 0 / 0.113973 | Other/unclear; thấp / cao. Cụm phương ngữ/thiếu ngữ cảnh, TRI gần ngưỡng phân loại. Reviewer nên abstain nếu không chắc cách hiểu, không tự gán nghĩa. |
| 21 | Sao a ý không dùng lựu đạn nhỉ, tiếc quá, hảo hán chưa thỏa căm hờn | ViHSD / 0 | 0 / 0.094361 | 1 / 0.735535 | Indirect/context-dependent; vừa / cao. Có ngôn ngữ tán thành bạo lực nhưng đối tượng/ngữ cảnh không rõ. Xác định đích và liệu đây là đe dọa, cổ vũ hay trích dẫn. |
| 22 | Tát dẹo sống mũi [đã ẩn] lun đi chị. Cái mũi nó giả tạo quá, chị tạo phước cho nó đi gắn lại cái ống thở mới nó sẽ biết ơn chị lắm đó. | ViHSD / 0 | 0 / 0.343810 | 1 / 0.715561 | Direct abuse/advocacy of violence appears; high/medium. The gold conflict needs context/quotation and rubric adjudication; do not change it from this review alone. |
| 23 | Sống dai quá | ViCTSD / 0 | 0 / 0.122939 | 1 / 0.731502 | Other/unclear, target-dependent phrase; low/high. A short excerpt cannot establish hostile meaning; request context rather than calling it a lexical false alarm. |
| 24 | [Tên đã ẩn] nói thật bây gio tui ghét bọn hàn như ghét cức | ViHSD / 1 | 1 / 0.844479 | 0 / 0.474022 | Direct abuse (assistant); cao / thấp. Có khái quát miệt thị theo quốc tịch. Reviewer xác định tiêu chí group-targeted toxicity và ghi sự khác nhau giữa hai model. |
| 25 | [Chuỗi có thể là tên riêng — đã ẩn] | ViHSD / 0 | 0 / 0.041490 | 1 / 0.512815 | Other/unclear; insufficient content. High confidence applies only to the insufficiency, not a toxicity judgment; abstain/request context and do not speculate about identity. |
| 26 | Thật sự phẫn nộ những quái xế như thế này. Ám ảnh kinh hoàng phiên bản xa lộ. | ViCTSD / 0 | 0 / 0.291304 | 1 / 0.782932 | Behavior-directed condemnation, possibly derogatory; medium/medium. Review whether the target is conduct or people under ViCTSD criteria; do not assume a lexical cause. |
| 27 | Trà ơ t vẫn chúc sinh nhật m thế đéo nào nhỉ | ViHSD / 0 | 0 / 0.199042 | 1 / 0.610047 | Profanity, target/relationship-dependent; medium/high. Review the conversational context and source rule for profanity; do not conclude toxicity from the token alone. |
| 28 | Cắt đất tsn mở sân golf giờ quá tải mở sân bay long thành(hốt 2 đầu) lãnh đạo quá sáng suốt. | ViHSD / 0 | 0 / 0.229447 | 1 / 0.628920 | Political sarcasm/criticism; no clear person-directed abuse; medium/medium. Review target and source rubric; do not attribute the PhoBERT false positive to a specific lexical cue. |
| 29 | Bài hát phải nói là như con cac. | ViHSD / 1 | 1 / 0.760035 | 0 / 0.458106 | Profane negative evaluation of a work, not clearly person-directed; medium/high. Ask whether ViHSD’s mapped binary label includes this target scope; no automatic gold correction. |
| 30 | Duma tức ák | ViHSD / 1 | 1 / 0.646027 | 0 / 0.053949 | Nonstandard profanity/exclamation, no clear target; low/high. Human reviewer should decode the spelling and apply the source rule for untargeted profanity. |

**Hướng dẫn người review:** mọi “nhận định sơ bộ” và gợi ý trên là prompt để kiểm tra, không phải gold label/cause đã xác minh. Ghi `agree / disagree / abstain`, category và confidence ở bảng member phía trên. Nếu bất đồng, ghi lý do ngắn không kèm quote/ID; gold label chỉ được đề xuất sửa qua quy trình adjudication của nhóm.
