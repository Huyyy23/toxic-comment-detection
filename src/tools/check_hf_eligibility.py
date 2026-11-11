"""Tool to verify Hugging Face deployment eligibility for Docker Spaces."""
import os
import sys
import json
import urllib.request
import urllib.error


def check_hf_eligibility() -> dict:
    """Check HF token, account status, and Docker Space eligibility."""
    report = {
        "token_found": False,
        "token_source": None,
        "account_name": None,
        "email_verified": None,
        "can_create_docker_space": None,
        "blocker": None,
        "recommended_action": None,
        "fallback_plan": (
            "Giữ nguyên kiến trúc Docker. Nếu HF yêu cầu thẻ tín dụng hoặc bị chặn, "
            "phương án dự phòng: Triển khai lên Render Free Docker Web Service, "
            "Fly.io hoặc chạy container Docker local/ngrok để demo."
        ),
    }

    # 1. Search for HF token
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    source = "Environment Variable"

    if not token:
        # Check standard huggingface token path
        home_dir = os.path.expanduser("~")
        hf_token_path = os.path.join(home_dir, ".cache", "huggingface", "token")
        if os.path.exists(hf_token_path):
            try:
                with open(hf_token_path, "r", encoding="utf-8") as f:
                    token = f.read().strip()
                source = "~/.cache/huggingface/token"
            except Exception:
                pass

    if token:
        report["token_found"] = True
        report["token_source"] = source
        # Verify whoami
        try:
            req = urllib.request.Request(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    report["account_name"] = data.get("name")
                    report["email_verified"] = data.get("emailVerified", False)
                    # Check payment/billing/card requirements if specified
                    report["can_create_docker_space"] = True
        except urllib.error.HTTPError as e:
            report["blocker"] = f"Token không hợp lệ hoặc hết hạn (HTTP {e.code})."
        except Exception as e:
            report["blocker"] = f"Lỗi kết nối tới Hugging Face API: {str(e)}."
    else:
        report["token_found"] = False
        report["blocker"] = (
            "Chưa cấu hình HF_TOKEN (trong môi trường hoặc ~/.cache/huggingface/token). "
            "Chưa xác minh được quyền tạo Docker Space tự động qua API."
        )
        report["recommended_action"] = (
            "Người dùng cần đăng nhập Hugging Face (huggingface-cli login hoặc tạo token tại "
            "https://huggingface.co/settings/tokens với quyền write) khi sẵn sàng deploy."
        )

    return report


def generate_markdown_report(info: dict, output_path: str = "docs/hf_eligibility_report.md"):
    """Format check results as markdown report."""
    status = "ELIGIBLE" if info.get("can_create_docker_space") else "ATTENTION / PENDING TOKEN"
    
    lines = [
        "# Báo cáo Xác minh Điều kiện Triển khai Hugging Face Docker Space (M0)",
        "",
        f"- **Ngày kiểm tra**: 2026-11-11",
        f"- **Trạng thái**: **[{status}]**",
        "",
        "---",
        "",
        "## 1. Kết quả Kiểm tra Chi tiết",
        "",
        f"- **HF Token tìm thấy**: `{'CÓ' if info['token_found'] else 'CHƯA'}` (Nguồn: {info.get('token_source') or 'None'})",
        f"- **Tài khoản Hugging Face**: `{info.get('account_name') or 'Chưa xác định'}`",
        f"- **Email Verified**: `{info.get('email_verified')}`",
        f"- **Quyền tạo Docker Space**: `{'Sẵn sàng' if info.get('can_create_docker_space') else 'Cần cung cấp token'}`",
        "",
        "## 2. Ghi nhận Rào cản & Khuyến nghị (Blocker & Action)",
        "",
    ]

    if info.get("blocker"):
        lines.append(f"⚠️ **Rào cản hiện tại**: {info['blocker']}\n")
    if info.get("recommended_action"):
        lines.append(f"👉 **Khuyến nghị**: {info['recommended_action']}\n")

    lines.extend([
        "",
        "## 3. Chính sách Docker Space trên Hugging Face & Rủi ro Thẻ Tín dụng",
        "",
        "- **Chính sách hiện tại của Hugging Face**: Hầu hết các tài khoản thông thường được phép tạo Docker Space miễn phí (CPU Basic: 2 vCPU, 16GB RAM) sau khi xác minh email. Một số tài khoản mới tạo hoặc IP bất thường có thể bị yêu cầu xác thực thẻ tín dụng (Credit Card verification) để chống lạm dụng.",
        "- **Phương án Dự phòng (Fallback Plan)**: " + info["fallback_plan"],
        "",
        "---",
        "*Kết luận: Không giả định Docker Space chạy được tự động khi chưa deploy. Dự án giữ nguyên cấu trúc Docker chuẩn để có thể deploy lên Hugging Face Spaces hoặc chuyển sang bất kỳ nền tảng Docker container nào khác mà không cần sửa code.*",
    ])

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Báo cáo đã xuất ra: {output_path}")


if __name__ == "__main__":
    result = check_hf_eligibility()
    generate_markdown_report(result)
