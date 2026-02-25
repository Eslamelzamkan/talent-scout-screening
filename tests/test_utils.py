"""
Tests for core/utils.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

from core.utils import (  # pyre-ignore[21]
    normalize_whitespace,
    normalize_for_matching,
    chunk_text,
    extract_emails,
    extract_phones,
    extract_urls,
    TextChunk,
)

import pytest  # pyre-ignore[21]


# ---- Whitespace normalization ----

class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        assert normalize_whitespace("hello   world") == "hello world"

    def test_collapses_tabs(self):
        assert normalize_whitespace("hello\t\tworld") == "hello world"

    def test_preserves_single_newline(self):
        result = normalize_whitespace("hello\nworld")
        assert "\n" in result

    def test_collapses_many_newlines(self):
        result = normalize_whitespace("hello\n\n\n\n\nworld")
        assert result == "hello\n\nworld"

    def test_strips_control_chars(self):
        result = normalize_whitespace("hello\x00\x01world")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_empty_string(self):
        assert normalize_whitespace("") == ""

    def test_strips_result(self):
        assert normalize_whitespace("  hello  ") == "hello"


# ---- Normalize for matching ----

class TestNormalizeForMatching:
    def test_lowercases(self):
        assert normalize_for_matching("HELLO World") == "hello world"

    def test_removes_control_chars(self):
        result = normalize_for_matching("Hello\x00World")
        assert "\x00" not in result

    def test_empty_string(self):
        assert normalize_for_matching("") == ""


# ---- Text chunking ----

class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "This is a short sentence."
        chunks = chunk_text(text, max_chars=2000)
        assert len(chunks) == 1
        assert chunks[0].text.strip() == text.strip()

    def test_long_text_multiple_chunks(self):
        text = "Hello world. " * 200  # ~2600 chars
        chunks = chunk_text(text, max_chars=500, overlap=50)
        assert len(chunks) > 1

    def test_chunk_overlap(self):
        text = "A" * 1000
        chunks = chunk_text(text, max_chars=500, overlap=100, respect_sentence_boundaries=False)
        if len(chunks) >= 2:
            # Second chunk should start before end of first chunk
            assert chunks[1].start < chunks[0].end

    def test_empty_text(self):
        chunks = chunk_text("")
        assert chunks == []

    def test_chunks_are_text_chunk_type(self):
        chunks = chunk_text("Hello world. This is a test.")
        for c in chunks:
            assert isinstance(c, TextChunk)
            assert isinstance(c.text, str)  # pyre-ignore[16]
            assert isinstance(c.start, int)  # pyre-ignore[16]
            assert isinstance(c.end, int)  # pyre-ignore[16]


# ---- Email extraction ----

class TestExtractEmails:
    def test_finds_email(self):
        emails = extract_emails("Contact me at john@example.com for details")
        assert "john@example.com" in emails

    def test_multiple_emails(self):
        text = "Send to alice@foo.org or bob@bar.co.uk"
        emails = extract_emails(text)
        assert len(emails) == 2

    def test_no_emails(self):
        emails = extract_emails("No email here, just text")
        assert emails == []

    def test_empty_string(self):
        assert extract_emails("") == []


# ---- Phone extraction ----

class TestExtractPhones:
    def test_international_format(self):
        phones = extract_phones("Call +20 10 12345678 for info")
        assert len(phones) >= 1

    def test_no_phone(self):
        phones = extract_phones("No phone number here")
        assert phones == []

    def test_short_numbers_filtered(self):
        phones = extract_phones("Room 123 on floor 4")
        assert phones == []


# ---- URL extraction ----

class TestExtractUrls:
    def test_http_url(self):
        urls = extract_urls("Visit http://example.com for more")
        assert len(urls) == 1

    def test_https_url(self):
        urls = extract_urls("Check https://github.com/user/repo")
        assert len(urls) == 1

    def test_no_urls(self):
        urls = extract_urls("No links here")
        assert urls == []
