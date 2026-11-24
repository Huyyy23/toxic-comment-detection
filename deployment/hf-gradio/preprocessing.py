from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Literal

TEENCODE_DICT: dict[str, str] = {
    "ko": "không", "k": "không", "khong": "không", "kh": "không",
    "hem": "không", "hok": "không", "hong": "không", "dc": "được",
    "đc": "được", "dk": "được", "bk": "biết", "bt": "bình thường",
    "mk": "mình", "mik": "mình", "mn": "mọi người", "mng": "mọi người",
    "ng": "người", "nge": "nghe", "ngta": "người ta", "nt": "nhắn tin",
    "ns": "nói", "nv": "nhân vật", "cx": "cũng", "cg": "cũng",
    "cũg": "cũng", "vs": "với", "voi": "với", "vz": "vậy", "v": "vậy",
    "vay": "vậy", "z": "vậy", "na": "này", "nay": "này", "ny": "này",
    "nè": "này", "ne": "này", "ni": "này", "nc": "nước", "oke": "ok",
    "oce": "ok", "okela": "ok", "ừa": "ừ", "ừm": "ừ", "uhm": "ừ",
    "bjo": "bây giờ", "h": "giờ", "gio": "giờ", "trc": "trước",
    "tg": "thời gian", "tgian": "thời gian", "nhieu": "nhiều", "nhiu": "nhiều",
    "nhé": "nhé", "nhe": "nhé", "nha": "nhé", "hn": "hà nội", "hcm": "hồ chí minh",
    "bth": "bình thường", "bình thg": "bình thường", "ntn": "như thế nào",
    "sao": "sao", "s": "sao", "sp": "sản phẩm", "xài": "dùng", "dung": "dùng",
    "làm j": "làm gì", "lam j": "làm gì", "j": "gì", "gi": "gì", "nào": "nào",
    "nao": "nào", "luon": "luôn", "lun": "luôn", "cuoi": "cuối", "dau": "đầu",
    "diem": "điểm", "nhin": "nhìn", "tinh": "tình", "khoe": "khỏe", "chet": "chết",
    "that": "thật", "chan": "chán", "buon": "buồn", "vui": "vui", "xau": "xấu",
    "dep": "đẹp", "tot": "tốt", "gioi": "giỏi", "ngu": "ngu", "xin loi": "xin lỗi",
    "xl": "xin lỗi", "cm": "comment", "cmt": "comment", "fb": "facebook",
    "ytb": "youtube", "gg": "google", "đm": "đm", "dm": "đm", "vcl": "vcl",
    "vl": "vl", "clgt": "clgt", "cc": "cc", "đkm": "đkm", "dkm": "đkm",
    "bitch": "bitch", "wtf": "wtf", "omg": "omg", "lmao": "lmao", "haha": "haha",
    "hihi": "haha", "hehe": "haha", "huhu": "buồn", "tks": "cảm ơn", "thks": "cảm ơn",
    "ty": "cảm ơn", "tq": "cảm ơn", "ib": "inbox", "pm": "nhắn tin", "rep": "trả lời",
    "like": "thích", "lk": "thích", "share": "chia sẻ", "tag": "gắn thẻ",
    "follow": "theo dõi",
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
_REPEAT_RE = re.compile(r"(.)\1{2,}")


def preprocess(text: str, variant: Literal["raw", "basic", "teencode"] = "raw") -> str:
    text = unicodedata.normalize("NFC", str(text))
    if variant == "raw":
        return text
    if variant not in ("basic", "teencode"):
        raise ValueError(f"Unknown preprocessing variant: {variant!r}")
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text).lower()
    text = _REPEAT_RE.sub(r"\1\1", text)
    text = _WS_RE.sub(" ", text).strip()
    if variant == "basic":
        return text
    text = _EMOJI_RE.sub(" <emoji> ", text)
    text = " ".join(TEENCODE_DICT.get(token.lower(), token) for token in text.split())
    return _WS_RE.sub(" ", text).strip()


TEENCODE_V2_DICT = {
    **TEENCODE_DICT,
    "vlxd": "vật liệu xây dựng", "cmnr": "rồi", "xamlol": "xàm lồn",
    "xàmlol": "xàm lồn", "vkl": "vcl", "đhs": "đéo hiểu sao", "dhs": "đéo hiểu sao",
    "ak": "à", "uk": "ừ", "thik": "thích", "iu": "yêu", "wa": "quá", "qa": "quá",
    "éo": "đéo", "méo": "đéo", "chả": "chẳng", "đéo": "đéo",
}
_V2_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_V2_HTML_RE = re.compile(r"<[^>]+>")
_V2_ZW_RE = re.compile(r"[\u200b-\u200f\ufeff]")
_V2_LETTER_REP_RE = re.compile(r"([a-zà-ỹ])\1{2,}", flags=re.IGNORECASE)
_V2_CHAR_REP3_RE = re.compile(r"(.)\1{3,}")
_V2_PUNCT_SEP_RE = re.compile(r"([^\w\s<>])", flags=re.UNICODE)
_V2_WS_RE = re.compile(r"\s+")
_V3_VI_TONE_FIXES = (
    ("òa", "oà"), ("óa", "oá"), ("ỏa", "oả"), ("õa", "oã"), ("ọa", "oạ"),
    ("òe", "oè"), ("óe", "oé"), ("ỏe", "oẻ"), ("õe", "oẽ"), ("ọe", "oẹ"),
    ("ùy", "uỳ"), ("úy", "uý"), ("ủy", "uỷ"), ("ũy", "uỹ"), ("ụy", "uỵ"),
)
_V3_OBFUSC_RE = (
    (re.compile(r"\bđ[\.\*\-_]+m\b", re.IGNORECASE), "đm"),
    (re.compile(r"\bd[\.\*\-_]+m\b", re.IGNORECASE), "đm"),
    (re.compile(r"\bv[\.\*\-_]+l\b", re.IGNORECASE), "vl"),
    (re.compile(r"\bv[\.\*\-_]+c[\.\*\-_]+l\b", re.IGNORECASE), "vcl"),
    (re.compile(r"\bc[\.\*\-_]+c\b", re.IGNORECASE), "cc"),
    (re.compile(r"\bl[\.\*\-_]+n\b", re.IGNORECASE), "lồn"),
)


def _normalize_v3_word_text(raw: str) -> str:
    text = unicodedata.normalize("NFC", str(raw))
    text = _V2_ZW_RE.sub("", text)
    text = _V2_URL_RE.sub(" <url> ", text)
    text = _V2_HTML_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" <emoji> ", text).lower()
    for old, new in _V3_VI_TONE_FIXES:
        text = text.replace(old, new)
    for pattern, replacement in _V3_OBFUSC_RE:
        text = pattern.sub(f" {replacement} ", text)
    text = _V2_LETTER_REP_RE.sub(r"\1", text)
    text = _V2_CHAR_REP3_RE.sub(r"\1\1", text)
    text = _V2_PUNCT_SEP_RE.sub(r" \1 ", text)
    tokens = [TEENCODE_V2_DICT.get(token, token) for token in text.split()]
    return _V2_WS_RE.sub(" ", " ".join(tokens)).strip() or "<unk>"


def preprocess_v2(texts: list[str], variant: str) -> list[str]:
    if variant in ("raw", "basic", "teencode"):
        return [preprocess(text, variant) for text in texts]
    if variant == "v3_word_norm":
        return [_normalize_v3_word_text(text) for text in texts]
    if variant == "v2_char_norm":
        out = []
        for raw in texts:
            text = unicodedata.normalize("NFC", str(raw))
            text = _V2_ZW_RE.sub("", text)
            text = _V2_URL_RE.sub(" ", text)
            text = _V2_HTML_RE.sub(" ", text)
            text = _V2_CHAR_REP3_RE.sub(r"\1\1\1", text)
            out.append(_V2_WS_RE.sub(" ", text).strip() or " ")
        return out
    raise ValueError(f"Unsupported preprocessing variant: {variant!r}")


class SequenceVocab:
    pad_id = 0
    unk_id = 1

    def __init__(self, payload: dict[str, Any]) -> None:
        self.mode = payload["mode"]
        if self.mode not in ("word", "char"):
            raise ValueError(f"Unsupported vocabulary mode: {self.mode!r}")
        self.tokens = {token: index for index, token in enumerate(payload["tokens"], start=2)}
        self.size = len(self.tokens) + 2

    def encode(self, text: str, max_length: int) -> list[int]:
        tokens = text.split() if self.mode == "word" else list(text)
        ids = [self.tokens.get(token, self.unk_id) for token in tokens[:max_length]]
        return ids or [self.unk_id]
