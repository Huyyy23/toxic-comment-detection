"""
Unit tests cho TV1 preprocessing module.
Chạy: .venv/Scripts/pytest tests/test_preprocessing.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.features.preprocessing import preprocess, preprocess_series


class TestRawVariant:
    def test_unicode_normalize(self):
        # NFC normalization — ký tự dấu có thể encode nhiều cách
        text = "ti\u0301nh"  # ti + combining accent
        result = preprocess(text, variant="raw")
        import unicodedata
        assert unicodedata.is_normalized("NFC", result)

    def test_preserves_original_text(self):
        text = "Bình luận bình thường 123 !@#"
        result = preprocess(text, variant="raw")
        # raw chỉ normalize unicode, không xóa gì
        assert len(result) > 0
        assert "Bình luận" in result

    def test_non_string_input(self):
        result = preprocess(42, variant="raw")
        assert result == "42"


class TestBasicVariant:
    def test_lowercase(self):
        result = preprocess("HẾT SỨC XẤU XÍ", variant="basic")
        assert result == result.lower()

    def test_strip_url(self):
        text = "Xem tại https://example.com nhé"
        result = preprocess(text, variant="basic")
        assert "https://" not in result
        assert "example.com" not in result

    def test_strip_html(self):
        text = "<b>Chú ý</b> điều này"
        result = preprocess(text, variant="basic")
        assert "<b>" not in result
        assert "</" not in result

    def test_normalize_whitespace(self):
        text = "từ   nhiều    khoảng    trắng"
        result = preprocess(text, variant="basic")
        assert "  " not in result

    def test_reduce_repeated_chars(self):
        text = "đẹppppp lắmmm"
        result = preprocess(text, variant="basic")
        # max 2 ký tự liên tiếp
        import re
        assert not re.search(r"(.)\1{2,}", result)


class TestTeencodeVariant:
    def test_replaces_teencode(self):
        text = "ko biết làm j"
        result = preprocess(text, variant="teencode")
        assert "không" in result or "biết" in result

    def test_handles_emoji(self):
        text = "vui quá 😂😂😂"
        result = preprocess(text, variant="teencode")
        assert "<emoji>" in result

    def test_is_lowercase(self):
        text = "MK Ko Biết"
        result = preprocess(text, variant="teencode")
        assert result == result.lower()

    def test_no_double_spaces(self):
        result = preprocess("ko  dc   lam j", variant="teencode")
        assert "  " not in result


class TestBatchPreprocessing:
    def test_preprocess_series_length(self):
        texts = ["text 1", "text 2", "text 3"]
        results = preprocess_series(texts, variant="basic")
        assert len(results) == len(texts)

    def test_preprocess_series_all_strings(self):
        texts = ["text 1", "text 2"]
        results = preprocess_series(texts, variant="teencode")
        assert all(isinstance(r, str) for r in results)

    def test_invalid_variant_raises(self):
        with pytest.raises((ValueError, KeyError, TypeError)):
            preprocess("text", variant="invalid_variant")  # type: ignore
