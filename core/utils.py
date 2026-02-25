"""
utils.py

Utility helpers for the Talent Scout project.

Design goals
------------
1) Robust text normalization & chunking (for embedding + LLM prompting)
2) Safe, dependency-light document reading (PDF optional)
3) Simple contact extraction (email / phone / links)
4) Consistent logging

This module intentionally keeps hard dependencies minimal.

Optional extras
--------------
- pdfplumber (better PDF text extraction)

pip install pdfplumber
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

# -----------------------------
# Logging
# -----------------------------


def get_logger(name: str = "talent_scout", level: int = logging.INFO) -> logging.Logger:
    """
    Create/get a logger with a consistent format. Idempotent.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = get_logger()


# -----------------------------
# Text normalization
# -----------------------------

_WS_RE = re.compile(r"[ \t]+")
_LINES_RE = re.compile(r"\n{3,}")
_NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving newlines.
    - Collapses repeated spaces/tabs
    - Collapses 3+ blank lines to 2
    - Removes non-printable control chars
    """
    if not text:
        return ""
    text = _NON_PRINTABLE_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _LINES_RE.sub("\n\n", text)
    return text.strip()


def normalize_for_matching(text: str) -> str:
    """
    Heavier normalization for deterministic matching:
    - lowercases
    - removes non-printable
    - standardizes whitespace
    """
    return normalize_whitespace(text).lower()


# -----------------------------
# Chunking helpers (for embeddings / prompting)
# -----------------------------


@dataclass(frozen=True)
class TextChunk:
    text: str
    start: int
    end: int


def chunk_text(
    text: str,
    max_chars: int = 2000,
    overlap: int = 200,
    respect_sentence_boundaries: bool = True,
) -> List[TextChunk]:
    """
    Chunk text into overlapping character windows.

    Why char-based?
    - Works without tokenizer deps.
    - Stable across environments (Windows / Colab).

    Parameters
    ----------
    max_chars : max length per chunk
    overlap : overlap between consecutive chunks (chars)
    respect_sentence_boundaries : attempt to end chunks near sentence boundary.

    Returns
    -------
    List[TextChunk] with original offsets.
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    max_chars = int(max(200, max_chars))
    overlap = int(max(0, min(overlap, max_chars - 50)))

    chunks: List[TextChunk] = []
    i = 0
    n = len(text)

    while i < n:
        j = min(n, i + max_chars)
        if respect_sentence_boundaries and j < n:
            # try to end near last sentence boundary
            window = text[i:j]
            boundary = max(window.rfind(". "), window.rfind("! "), window.rfind("? "), window.rfind("\n"))
            if boundary > max(100, int(0.6 * max_chars)):
                j = i + boundary + 1

        chunk = text[i:j].strip()
        if chunk:
            chunks.append(TextChunk(text=chunk, start=i, end=j))

        if j == n:
            break
        i = max(0, j - overlap)

    return chunks


# -----------------------------
# File reading
# -----------------------------


def read_text_file(path: str | Path, encoding: str = "utf-8") -> str:
    p = Path(path)
    return p.read_text(encoding=encoding, errors="ignore")


def read_pdf(path: str | Path, max_pages: Optional[int] = None) -> str:
    """
    Read a PDF into text. Uses pdfplumber if available; otherwise returns
    a helpful error message to install it.

    On Windows: pdf extraction can be flaky without pdfplumber.
    """
    p = Path(path)
    try:
        import pdfplumber  # type: ignore
    except Exception:
        raise ImportError(
            "PDF reading requires 'pdfplumber'. Install with: pip install pdfplumber"
        )

    texts: List[str] = []
    with pdfplumber.open(str(p)) as pdf:
        pages = pdf.pages[: max_pages or len(pdf.pages)]
        for pg in pages:
            texts.append(pg.extract_text() or "")
    return normalize_whitespace("\n\n".join(texts))


def read_document(path: str | Path, max_pdf_pages: Optional[int] = 5) -> str:
    """
    Read common document types.
    - .txt / .md / .json / .yaml / .yml
    - .pdf (optional pdfplumber)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    suffix = p.suffix.lower()
    if suffix in {".txt", ".md", ".log"}:
        return normalize_whitespace(read_text_file(p))
    if suffix in {".json"}:
        return json.dumps(json.loads(read_text_file(p)), ensure_ascii=False, indent=2)
    if suffix in {".yml", ".yaml"}:
        # avoid hard dependency on PyYAML in utils
        try:
            import yaml  # type: ignore
        except Exception:
            raise ImportError("YAML reading requires PyYAML. Install with: pip install pyyaml")
        data = yaml.safe_load(read_text_file(p))
        return normalize_whitespace(yaml.dump(data, allow_unicode=True, sort_keys=False))
    if suffix == ".pdf":
        return read_pdf(p, max_pages=max_pdf_pages)

    # fallback: try reading as text
    return normalize_whitespace(read_text_file(p))


# -----------------------------
# Contact extraction
# -----------------------------

_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b")
# permissive: +20 10 1234 5678, (010)12345678, 010-1234-5678 etc.
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}(?!\w)"
)
_URL_RE = re.compile(r"\bhttps?://[^\s)>\]]+\b", re.IGNORECASE)


def extract_emails(text: str) -> List[str]:
    text = text or ""
    emails = sorted(set(_EMAIL_RE.findall(text)))
    return emails


def extract_urls(text: str) -> List[str]:
    text = text or ""
    urls = sorted(set(_URL_RE.findall(text)))
    return urls


def extract_phones(text: str) -> List[str]:
    text = text or ""
    phones = set()
    for m in _PHONE_RE.finditer(text):
        s = re.sub(r"[^\d+]", "", m.group(0))
        # heuristic: ignore very short numbers
        digits = re.sub(r"\D", "", s)
        if len(digits) >= 8:
            phones.add(s)
    return sorted(phones)


# -----------------------------
# JSON helpers
# -----------------------------


def safe_json_dump(obj, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_json_load(path: str | Path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def timestamp_id(fmt: str = "%Y%m%d-%H%M%S") -> str:
    return datetime.now().strftime(fmt)
