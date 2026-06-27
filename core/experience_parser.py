"""
experience_parser.py — Extracts total professional experience from resume text.

Ported from talent-scout-screening/core/experience_parser.py.
Import path changed: core.utils → core.utils
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

from core.utils import normalize_whitespace


MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
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

EXPLICIT_YEARS_RE = re.compile(
    r"(?P<prefix>\b(?:at\s+least|minimum|min\.?|over|more\s+than)\b\s*)?"
    r"(?P<y1>\d+(?:\.\d+)?)\s*"
    r"(?P<plus>\+)?\s*"
    r"(?P<unit>years?|yrs?)\b",
    flags=re.IGNORECASE,
)

YEARS_RANGE_RE = re.compile(
    r"(?P<y1>\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(?P<y2>\d+(?:\.\d+)?)\s*(years?|yrs?)\b",
    flags=re.IGNORECASE,
)

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

EDUCATION_CONTEXT_RE = re.compile(
    r"\b("
    r"education|academic|academics|university|college|school|faculty|"
    r"bachelor(?:'s)?|master(?:'s)?|ph\.?d|doctorate|diploma|degree|"
    r"gpa|cgpa|coursework|thesis|graduat(?:e|ion|ed)|"
    r"undergraduate|postgraduate|major|minor"
    r")\b",
    flags=re.IGNORECASE,
)

SCHOOL_CONTEXT_RE = re.compile(
    r"\b(university|college|school|faculty|institute|polytechnic|academy)\b",
    flags=re.IGNORECASE,
)

WORK_CONTEXT_RE = re.compile(
    r"\b("
    r"experience|employment|work(?:ed|ing)?|professional|"
    r"intern(?:ship)?|engineer|developer|manager|analyst|scientist|consultant|"
    r"specialist|lead|director|officer|coordinator|assistant|associate|"
    r"freelance|contract"
    r")\b",
    flags=re.IGNORECASE,
)

INTERNSHIP_CONTEXT_RE = re.compile(
    r"\b(intern(?:ship)?|trainee|apprentice)\b",
    flags=re.IGNORECASE,
)

NON_PROFESSIONAL_EXPLICIT_CONTEXT_RE = re.compile(
    r"\b(projects?|academic|coursework|capstone|hackathon|training)\b",
    flags=re.IGNORECASE,
)

EDUCATION_HEADING_RE = re.compile(
    r"(?m)^\s*(education|academics?|academic background|qualifications?)\s*$",
    flags=re.IGNORECASE,
)

WORK_HEADING_RE = re.compile(
    r"(?m)^\s*(?:[a-z/&\-\s]+?\s+)?(work experience|professional experience|employment(?: history)?|career summary|projects|experience)\s*$",
    flags=re.IGNORECASE,
)


def _line_bounds(text: str, idx: int) -> Tuple[int, int]:
    if 0 <= idx < len(text) and text[idx] == "\n" and (idx + 1) < len(text):
        idx += 1
    start = text.rfind("\n", 0, idx) + 1
    end = text.find("\n", idx)
    if end == -1:
        end = len(text)
    return start, end


def _in_education_section(text: str, idx: int, lookback_chars: int = 1400) -> bool:
    segment_start = max(0, idx - lookback_chars)
    segment = text[segment_start:idx]

    edu_matches = list(EDUCATION_HEADING_RE.finditer(segment))
    if not edu_matches:
        return False

    last_edu = edu_matches[-1].start()
    work_matches = list(WORK_HEADING_RE.finditer(segment))
    last_work = work_matches[-1].start() if work_matches else -1
    return last_work < last_edu


def _is_education_context(text: str, start: int, end: int) -> bool:
    line_start, line_end = _line_bounds(text, start)
    context_start = max(0, line_start - 140)
    context_end = min(len(text), line_end + 50)
    snippet = text[context_start:context_end]
    tight_snippet = text[max(0, start - 45):min(len(text), end + 45)]
    nearby_school_snippet = text[max(0, start - 130):min(len(text), end + 35)]
    line_snippet = text[line_start:line_end]

    has_edu_signal = bool(EDUCATION_CONTEXT_RE.search(snippet))
    has_work_signal = bool(WORK_CONTEXT_RE.search(tight_snippet))
    has_work_signal_on_line = bool(WORK_CONTEXT_RE.search(line_snippet))
    has_school_signal = bool(SCHOOL_CONTEXT_RE.search(nearby_school_snippet))
    in_edu_section = _in_education_section(text, start)

    if has_school_signal and not has_work_signal and not has_work_signal_on_line:
        return True
    if in_edu_section and has_edu_signal and not has_work_signal:
        return True
    if has_edu_signal and not has_work_signal and not has_work_signal_on_line:
        return True
    return False


def _is_internship_context(text: str, start: int, end: int) -> bool:
    line_start, line_end = _line_bounds(text, start)
    context_start = max(0, line_start - 220)
    context_end = min(len(text), line_end + 80)
    snippet = text[context_start:context_end]
    return bool(INTERNSHIP_CONTEXT_RE.search(snippet))


def _is_non_professional_explicit_context(text: str, start: int, end: int) -> bool:
    context_start = max(0, start - 120)
    context_end = min(len(text), end + 120)
    snippet = text[context_start:context_end]
    return bool(NON_PROFESSIONAL_EXPLICIT_CONTEXT_RE.search(snippet))


def _month_to_int(m: Optional[str]) -> Optional[int]:
    if not m:
        return None
    m = m.strip().lower()
    if m in PRESENT_TOKENS:
        return None
    return MONTHS.get(m)


def _to_month_index(y: int, m: int) -> int:
    return y * 12 + (m - 1)


def _clamp_month(y: int, m: Optional[int]) -> int:
    return 1 if m is None else max(1, min(12, int(m)))


def _parse_year(s: str) -> int:
    return int(s)


def _today_year_month() -> Tuple[int, int]:
    t = date.today()
    return t.year, t.month


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


class ExperienceParser:
    def __init__(self, max_years_cap: int = 40):
        self.max_years_cap = int(max_years_cap)

    def extract_explicit_years(self, text: str) -> Tuple[Optional[float], List[str]]:
        text_n = normalize_whitespace(text)
        evidence: List[str] = []
        vals: List[float] = []

        for m in YEARS_RANGE_RE.finditer(text_n):
            if _is_education_context(text_n, m.start(), m.end()):
                continue
            if _is_non_professional_explicit_context(text_n, m.start(), m.end()):
                continue
            y2 = float(m.group("y2"))
            vals.append(y2)
            evidence.append(m.group(0).strip())

        for m in EXPLICIT_YEARS_RE.finditer(text_n):
            if _is_education_context(text_n, m.start(), m.end()):
                continue
            if _is_non_professional_explicit_context(text_n, m.start(), m.end()):
                continue
            y1 = float(m.group("y1"))
            vals.append(y1)
            evidence.append(m.group(0).strip())

        if not vals:
            return None, []

        v = min(max(vals), float(self.max_years_cap))
        return float(round(v, 2)), evidence[:8]

    def extract_date_ranges(self, text: str) -> Tuple[Optional[float], List[str]]:
        text_n = normalize_whitespace(text)
        all_intervals: List[Tuple[int, int]] = []
        all_evidence: List[str] = []
        professional_intervals: List[Tuple[int, int]] = []
        professional_evidence: List[str] = []

        ty, tm = _today_year_month()
        now_idx = _to_month_index(ty, tm)

        for m in DATE_RANGE_RE.finditer(text_n):
            if _is_education_context(text_n, m.start(), m.end()):
                continue

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
                continue

            start_idx = _to_month_index(y1, m1i)

            if end_idx < start_idx:
                continue
            if (end_idx - start_idx) > self.max_years_cap * 12:
                continue

            matched_range = m.group(0).strip()
            interval = (start_idx, end_idx)
            all_intervals.append(interval)
            all_evidence.append(matched_range)

            if not _is_internship_context(text_n, m.start(), m.end()):
                professional_intervals.append(interval)
                professional_evidence.append(matched_range)

        # If there is at least one non-intern role, internships should not inflate
        # professional years. If everything is internship-only, use all intervals.
        intervals = professional_intervals if professional_intervals else all_intervals
        evidence = professional_evidence if professional_intervals else all_evidence

        if not intervals:
            return None, []

        merged = _merge_intervals(intervals)
        total_months = 0
        for s, e in merged:
            total_months += (e - s + 1)

        years = total_months / 12.0
        years = min(years, float(self.max_years_cap))

        return float(round(years, 2)), evidence[:10]

    def parse(self, resume_text: str) -> ExperienceResult:
        explicit_years, explicit_ev = self.extract_explicit_years(resume_text)
        ranged_years, ranged_ev = self.extract_date_ranges(resume_text)

        evidence = {"explicit": explicit_ev, "date_ranges": ranged_ev}

        if ranged_years is not None and explicit_years is not None:
            years = max(ranged_years, explicit_years)
            method = "combined"
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
            years=float(round(years, 2)),
            months=months,
            method=method,
            confidence=float(round(confidence, 2)),
            evidence=evidence,
        )


def parse_experience(resume_text: str) -> dict:
    """Legacy wrapper: returns dict."""
    parser = ExperienceParser()
    res = parser.parse(resume_text)
    return {
        "years": res.years,
        "months": res.months,
        "method": res.method,
        "confidence": res.confidence,
        "evidence": res.evidence,
    }
