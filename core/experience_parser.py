"""
experience_parser.py

Extracts (approx.) total professional experience from free-form resume text.

Why this exists
--------------
Recruiting resumes are messy: mixed formats, partial dates, overlapping roles,
and sometimes explicit statements like "5+ years of experience".

This parser uses TWO signals and then reconciles them:
1) Explicit experience mentions (e.g., "3+ years", "at least 5 years")
2) Date ranges (e.g., "Jan 2020 - Mar 2022", "2021 - Present")

It returns:
- years (float)
- months (int)
- method ("explicit", "date_ranges", "combined")
- confidence score (0..1)
- evidence (snippets)

No heavy deps (no dateutil). Pure regex + month mapping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple

from core.utils import normalize_whitespace  # pyre-ignore[21]


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

PRESENT_TOKENS = {"present", "current", "now", "to date", "till date"}


@dataclass(frozen=True)
class ExperienceResult:
    years: float
    months: int
    method: str
    confidence: float
    evidence: Dict[str, List[str]]


# -----------------------------
# Regex patterns
# -----------------------------

# "5 years", "5+ yrs", "3.5 years", "at least 5 years", "minimum 3 years"
EXPLICIT_YEARS_RE = re.compile(
    r"(?P<prefix>\b(?:at\s+least|minimum|min\.?|over|more\s+than)\b\s*)?"
    r"(?P<y1>\d+(?:\.\d+)?)\s*"
    r"(?P<plus>\+)?\s*"
    r"(?P<unit>years?|yrs?)\b",
    flags=re.IGNORECASE,
)

# "3-5 years", "3 to 5 years", "3–5 yrs"
YEARS_RANGE_RE = re.compile(
    r"(?P<y1>\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(?P<y2>\d+(?:\.\d+)?)\s*(years?|yrs?)\b",
    flags=re.IGNORECASE,
)

# Date ranges like:
# "Jan 2020 - Mar 2022", "2020 - 2022", "2021 - Present"
DATE_RANGE_RE = re.compile(
    r"(?P<m1>\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b)?"
    r"\s*"
    r"(?P<y1>\b(?:19|20)\d{2}\b)"
    r"\s*(?:-|–|to)\s*"
    r"(?P<m2>\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b|present|current|now)?"
    r"\s*"
    r"(?P<y2>\b(?:19|20)\d{2}\b)?",
    flags=re.IGNORECASE,
)


def _month_to_int(m: Optional[str]) -> Optional[int]:
    if not m:
        return None
    m = m.strip().lower()
    if m in PRESENT_TOKENS:
        return None
    return MONTHS.get(m)


def _to_month_index(y: int, m: int) -> int:
    # e.g., 2020-01 => 2020*12 + 0
    return y * 12 + (m - 1)


def _clamp_month(y: int, m: Optional[int]) -> int:
    return 1 if m is None else max(1, min(12, int(m)))


def _parse_year(s: str) -> int:
    return int(s)


def _today_year_month() -> Tuple[int, int]:
    t = date.today()
    return t.year, t.month


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge [start, end] month-index intervals (inclusive end).
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = intervals[0]  # pyre-ignore
    for s, e in intervals[1:]:  # pyre-ignore
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))  # pyre-ignore
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))  # pyre-ignore
    return merged


class ExperienceParser:
    """
    Extracts experience from resume text.

    Heuristics:
    - explicit years often refer to requirements (JD) but also appear in resumes ("5 years exp").
      If you feed resume text only, it's usually fine.
    - date ranges are better evidence for resumes.
    """

    def __init__(self, max_years_cap: int = 40):
        self.max_years_cap = int(max_years_cap)

    def extract_explicit_years(self, text: str) -> Tuple[Optional[float], List[str]]:
        text_n = normalize_whitespace(text)
        evidence: List[str] = []

        # ranges like 3-5 years (use upper bound as "claimed capability")
        vals: List[float] = []
        for m in YEARS_RANGE_RE.finditer(text_n):
            y2 = float(m.group("y2"))
            vals.append(y2)
            evidence.append(m.group(0).strip())

        # singles like 5+ years
        for m in EXPLICIT_YEARS_RE.finditer(text_n):
            y1 = float(m.group("y1"))
            # If "+", interpret as at least y1; we'll use y1 as conservative.
            vals.append(y1)
            evidence.append(m.group(0).strip())

        if not vals:
            return None, []

        # pick the max mention, cap
        v = min(max(vals), float(self.max_years_cap))
        return float(round(v, 2)), evidence[:8]  # pyre-ignore

    def extract_date_ranges(self, text: str) -> Tuple[Optional[float], List[str]]:
        """
        Parse date ranges and estimate total experience (years).
        """
        text_n = normalize_whitespace(text)
        intervals: List[Tuple[int, int]] = []
        evidence: List[str] = []

        ty, tm = _today_year_month()
        now_idx = _to_month_index(ty, tm)

        for m in DATE_RANGE_RE.finditer(text_n):
            y1 = _parse_year(m.group("y1"))
            m1 = _month_to_int(m.group("m1"))
            m1i = _clamp_month(y1, m1 if m1 is not None else 1)

            y2_raw = m.group("y2")
            m2_raw = m.group("m2")

            if m2_raw and m2_raw.strip().lower() in PRESENT_TOKENS:
                end_idx = now_idx
            elif y2_raw:
                y2 = _parse_year(y2_raw)
                m2 = _month_to_int(m2_raw)
                m2i = _clamp_month(y2, m2 if m2 is not None else 12)
                end_idx = _to_month_index(y2, m2i)
            else:
                # "2020 - " (dangling). ignore as too ambiguous
                continue

            start_idx = _to_month_index(y1, m1i)

            # sanity checks
            if end_idx < start_idx:
                continue
            # reject ranges that are wildly long (e.g., 1900)
            if (end_idx - start_idx) > self.max_years_cap * 12:
                continue

            intervals.append((start_idx, end_idx))
            evidence.append(m.group(0).strip())

        if not intervals:
            return None, []

        merged = _merge_intervals(intervals)
        total_months = 0
        for s, e in merged:
            total_months += (e - s + 1)

        years = total_months / 12.0
        years = min(years, float(self.max_years_cap))

        return float(round(years, 2)), evidence[:10]  # pyre-ignore

    def parse(self, resume_text: str) -> ExperienceResult:
        """
        Main entry point.
        """
        explicit_years, explicit_ev = self.extract_explicit_years(resume_text)
        ranged_years, ranged_ev = self.extract_date_ranges(resume_text)

        evidence = {"explicit": explicit_ev, "date_ranges": ranged_ev}

        # Choose method:
        # - If we have date ranges, trust them more for resumes.
        # - If only explicit exists, use that.
        # - If both exist, take max but confidence is higher if they agree-ish.
        if ranged_years is not None and explicit_years is not None:
            years = max(ranged_years, explicit_years)
            method = "combined"

            # confidence: high if close, lower if far apart
            gap = abs(ranged_years - explicit_years)
            confidence = 0.85 if gap <= 1.0 else 0.65 if gap <= 3.0 else 0.5
        elif ranged_years is not None:
            years = ranged_years
            method = "date_ranges"
            confidence = 0.8
        elif explicit_years is not None:
            years = explicit_years
            method = "explicit"
            confidence = 0.6
        else:
            years = 0.0
            method = "none"
            confidence = 0.1

        months = int(round(years * 12))
        return ExperienceResult(
            years=float(round(years, 2)),  # pyre-ignore
            months=months,
            method=method,
            confidence=float(round(confidence, 2)),  # pyre-ignore
            evidence=evidence,
        )


# Backward-compatible function name (if your code calls parse_experience)
def parse_experience(resume_text: str) -> dict:
    """
    Legacy wrapper: returns dict structure similar to older implementations.
    """
    parser = ExperienceParser()
    res = parser.parse(resume_text)
    return {
        "years": res.years,
        "months": res.months,
        "method": res.method,
        "confidence": res.confidence,
        "evidence": res.evidence,
    }
