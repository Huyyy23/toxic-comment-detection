"""
Preprocessing pipeline cho TV1 — EXP-01.

Ba variant:
  - raw:      Không xử lý gì, giữ nguyên text gốc
  - basic:    Lowercase + strip URL/HTML + chuẩn hóa khoảng trắng
  - teencode:  basic + từ điển teencode + chuẩn hóa dấu tiếng Việt + xử lý emoji

Quy tắc:
  - Không fit bất kỳ tham số nào từ validation/test
  - Từ điển teencode là tĩnh (không học từ dữ liệu), nên áp dụng được cho mọi split
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

# ---------------------------------------------------------------------------
# Từ điển teencode tiếng Việt phổ biến
# (tĩnh, không học từ dữ liệu — an toàn áp dụng mọi split)
# ---------------------------------------------------------------------------
TEENCODE_DICT: dict[str, str] = {
    # Phủ định / khẳng định
    "ko": "không",
    "k": "không",
    "khong": "không",
    "kh": "không",
    "hem": "không",
    "hok": "không",
    "hong": "không",
    "dc": "được",
    "đc": "được",
    "dk": "được",
    "bk": "biết",
    "bt": "bình thường",
    "mk": "mình",
    "mik": "mình",
    "mn": "mọi người",
    "mng": "mọi người",
    "ng": "người",
    "nge": "nghe",
    "ngta": "người ta",
    "nt": "nhắn tin",
    "ns": "nói",
    "nv": "nhân vật",
    "cx": "cũng",
    "cg": "cũng",
    "cũg": "cũng",
    "vs": "với",
    "voi": "với",
    "vz": "vậy",
    "v": "vậy",
    "vay": "vậy",
    "z": "vậy",
    "na": "này",
    "nay": "này",
    "ny": "này",
    "nè": "này",
    "ne": "này",
    "ni": "này",
    "nc": "nước",
    "oke": "ok",
    "oce": "ok",
    "okela": "ok",
    "ừa": "ừ",
    "ừm": "ừ",
    "uhm": "ừ",
    "bjo": "bây giờ",
    "h": "giờ",
    "gio": "giờ",
    "trc": "trước",
    "tg": "thời gian",
    "tgian": "thời gian",
    "nhieu": "nhiều",
    "nhiu": "nhiều",
    "nhé": "nhé",
    "nhe": "nhé",
    "nha": "nhé",
    "hn": "hà nội",
    "hcm": "hồ chí minh",
    "bth": "bình thường",
    "bình thg": "bình thường",
    "ntn": "như thế nào",
    "sao": "sao",
    "s": "sao",
    "sp": "sản phẩm",
    "xài": "dùng",
    "dung": "dùng",
    "làm j": "làm gì",
    "lam j": "làm gì",
    "j": "gì",
    "gi": "gì",
    "nào": "nào",
    "nao": "nào",
    "luon": "luôn",
    "lun": "luôn",
    "cuoi": "cuối",
    "dau": "đầu",
    "diem": "điểm",
    "nhin": "nhìn",
    "tinh": "tình",
    "khoe": "khỏe",
    "chet": "chết",
    "that": "thật",
    "chan": "chán",
    "buon": "buồn",
    "vui": "vui",
    "xau": "xấu",
    "dep": "đẹp",
    "tot": "tốt",
    "gioi": "giỏi",
    "ngu": "ngu",
    "xin loi": "xin lỗi",
    "xl": "xin lỗi",
    "cm": "comment",
    "cmt": "comment",
    "fb": "facebook",
    "ytb": "youtube",
    "gg": "google",
    "đm": "đm",
    "dm": "đm",
    "vcl": "vcl",
    "vl": "vl",
    "clgt": "clgt",
    "cc": "cc",
    "đkm": "đkm",
    "dkm": "đkm",
    "bitch": "bitch",
    "wtf": "wtf",
    "omg": "omg",
    "lmao": "lmao",
    "haha": "haha",
    "hihi": "haha",
    "hehe": "haha",
    "huhu": "buồn",
    "tks": "cảm ơn",
    "thks": "cảm ơn",
    "ty": "cảm ơn",
    "tq": "cảm ơn",
    "ib": "inbox",
    "pm": "nhắn tin",
    "rep": "trả lời",
    "like": "thích",
    "lk": "thích",
    "share": "chia sẻ",
    "tag": "gắn thẻ",
    "follow": "theo dõi",
}

# Pattern cho URL
_URL_RE = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)

# Pattern HTML tags
_HTML_RE = re.compile(r"<[^>]+>")

# Pattern khoảng trắng
_WS_RE = re.compile(r"\s+")

# Pattern emoji (cơ bản — ký tự Unicode ngoài BMP)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

# Pattern lặp ký tự (vd: "đẹppppp" → "đẹp")
_REPEAT_RE = re.compile(r"(.)\1{2,}")


def _strip_url(text: str) -> str:
    return _URL_RE.sub(" ", text)


def _strip_html(text: str) -> str:
    return _HTML_RE.sub(" ", text)


def _normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _normalize_unicode(text: str) -> str:
    """NFC normalize — quan trọng cho tiếng Việt có nhiều cách encode dấu."""
    return unicodedata.normalize("NFC", text)


def _reduce_repeated_chars(text: str) -> str:
    """'đẹppppp' → 'đẹpp' (giữ 2 để còn nhận ra từ)."""
    return _REPEAT_RE.sub(r"\1\1", text)


def _replace_emoji(text: str, placeholder: str = " <emoji> ") -> str:
    return _EMOJI_RE.sub(placeholder, text)


def _apply_teencode(text: str) -> str:
    """Áp dụng từ điển teencode theo word boundary."""
    tokens = text.split()
    replaced = []
    for tok in tokens:
        lower = tok.lower()
        if lower in TEENCODE_DICT:
            replaced.append(TEENCODE_DICT[lower])
        else:
            replaced.append(tok)
    return " ".join(replaced)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

PreprocessVariant = Literal["raw", "basic", "teencode"]


def preprocess(text: str, variant: PreprocessVariant = "raw") -> str:
    """Áp dụng preprocessing theo variant.

    Args:
        text: Văn bản đầu vào.
        variant: "raw" | "basic" | "teencode"

    Returns:
        Văn bản đã xử lý.
    """
    if not isinstance(text, str):
        text = str(text)

    if variant == "raw":
        return _normalize_unicode(text)
    elif variant in ("basic", "teencode"):
        text = _normalize_unicode(text)
        text = _strip_url(text)
        text = _strip_html(text)
        text = text.lower()
        text = _reduce_repeated_chars(text)
        text = _normalize_whitespace(text)

        if variant == "basic":
            return text

        # --- Teencode: basic + emoji + teencode dict ---
        text = _replace_emoji(text, placeholder=" <emoji> ")
        text = _apply_teencode(text)
        text = _normalize_whitespace(text)
        return text
    else:
        raise ValueError(f"Unknown variant: {variant!r}. Must be 'raw', 'basic', or 'teencode'.")


def preprocess_series(
    texts: list[str], variant: PreprocessVariant = "raw"
) -> list[str]:
    """Batch preprocessing."""
    return [preprocess(t, variant) for t in texts]
