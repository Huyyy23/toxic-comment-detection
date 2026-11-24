# Báo cáo xác minh Hugging Face Gradio ZeroGPU

Trạng thái tài khoản: **CHƯA XÁC MINH**. Chưa có quyền public deployment được xác nhận.

## Kết quả kiểm tra tài khoản

| Điều kiện | Bằng chứng hiện có |
|---|---|
| `HF_TOKEN` hoặc token đăng nhập | Không tìm thấy |
| Tên tài khoản | Chưa xác định |
| Email đã xác minh | Chưa xác định |
| Tài khoản đủ tuổi và good standing | Chưa xác định |
| Quyền tạo Space và quota ZeroGPU | Chưa xác định |

Đã thử kiểm tra token/API và mở phiên HF đã đăng nhập. Môi trường không có `HF_TOKEN`; browser relay không khả dụng. Vì vậy không thể đọc trạng thái tài khoản hay tạo Space. Không có credential nào được yêu cầu hoặc ghi vào repo.

## Điều kiện nền tảng đã đối chiếu

- Free ZeroGPU dùng Gradio SDK, yêu cầu email xác minh, tài khoản trên 30 ngày và good standing; tài khoản Free có quota tối đa 5 GPU-phút mỗi ngày và tối đa 2 ZeroGPU Spaces theo tài liệu HF hiện hành.
- ZeroGPU yêu cầu PyTorch 2.8 trở lên. README deployment pin Gradio 5.50.0 để tương thích với Transformers 4.57.6 và Hub 0.x trong pipeline hiện có.
- `app_port` trong metadata chỉ áp dụng Docker SDK. Gradio SDK dùng port mặc định 7860.
- HF mô tả ZeroGPU cho Gradio. FastAPI custom routes có thể cùng process local, nhưng chưa có bằng chứng remote rằng route REST trực tiếp nhận GPU allocation.
- CPU Basic không được chọn. Không tạo Docker Space, không chọn hardware trả phí.

Nguồn: [Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview), [ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu), [Spaces configuration](https://huggingface.co/docs/hub/en/spaces-config-reference), [Gradio FastAPI mount](https://www.gradio.app/docs/gradio/mount_gradio_app).

## Quyết định và blocker

- Kế hoạch N12 chuyển sang một Space Gradio + ZeroGPU Free, port 7860, FastAPI REST custom routes được giữ cùng process.
- `deployment/hf-gradio/` chỉ chứa inference code và ba checkpoint đã chọn. Không có dataset, notebook, validation predictions, training logs hoặc experiment artifacts.
- Chỉ upload private để smoke test nếu tài khoản cho phép. Nếu ZeroGPU yêu cầu Space public, không upload trước khi có bằng chứng quyền phân phối model/tokenizer và dữ liệu nguồn.
- Chưa xác minh HF account eligibility, quyền upload model artifacts, N10 acknowledgement hiện có, hoặc ZeroGPU allocation cho direct FastAPI REST. Chưa deploy.
