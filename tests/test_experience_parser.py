"""
Tests for core/experience_parser.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

from core.experience_parser import (  # pyre-ignore[21]
    ExperienceParser,
    ExperienceResult,
    parse_experience,
    _merge_intervals,
    _month_to_int,
)

import pytest  # pyre-ignore[21]


# ---- Helpers ----

class TestHelpers:
    def test_month_to_int_valid(self):
        assert _month_to_int("jan") == 1
        assert _month_to_int("december") == 12
        assert _month_to_int("Sept") == 9

    def test_month_to_int_none(self):
        assert _month_to_int(None) is None
        assert _month_to_int("") is None

    def test_month_to_int_present_token(self):
        assert _month_to_int("present") is None
        assert _month_to_int("current") is None

    def test_merge_non_overlapping(self):
        intervals = [(0, 10), (20, 30)]
        merged = _merge_intervals(intervals)
        assert len(merged) == 2

    def test_merge_overlapping(self):
        intervals = [(0, 15), (10, 30)]
        merged = _merge_intervals(intervals)
        assert len(merged) == 1
        assert merged[0] == (0, 30)

    def test_merge_adjacent(self):
        intervals = [(0, 10), (11, 20)]
        merged = _merge_intervals(intervals)
        assert len(merged) == 1
        assert merged[0] == (0, 20)

    def test_merge_empty(self):
        assert _merge_intervals([]) == []


# ---- Explicit years ----

class TestExplicitYears:
    parser: ExperienceParser  # pyre-ignore[13]

    def setup_method(self):
        self.parser = ExperienceParser()

    def test_simple_years(self):
        years, evidence = self.parser.extract_explicit_years("I have 5 years of experience")
        assert years == 5.0
        assert len(evidence) > 0

    def test_plus_years(self):
        years, _ = self.parser.extract_explicit_years("3+ years in software development")
        assert years == 3.0

    def test_at_least_years(self):
        years, _ = self.parser.extract_explicit_years("at least 7 years of experience")
        assert years == 7.0

    def test_yrs_abbreviation(self):
        years, _ = self.parser.extract_explicit_years("10 yrs working in IT")
        assert years == 10.0

    def test_range_takes_upper(self):
        years, _ = self.parser.extract_explicit_years("3-5 years experience")
        assert years == 5.0

    def test_no_match(self):
        years, evidence = self.parser.extract_explicit_years("I worked at Google and Facebook")
        assert years is None
        assert evidence == []

    def test_decimal_years(self):
        years, _ = self.parser.extract_explicit_years("3.5 years of experience")
        assert years == 3.5


# ---- Date ranges ----

class TestDateRanges:
    parser: ExperienceParser  # pyre-ignore[13]

    def setup_method(self):
        self.parser = ExperienceParser()

    def test_full_date_range(self):
        text = "Software Engineer, Jan 2020 - Dec 2022"
        years, evidence = self.parser.extract_date_ranges(text)
        assert years is not None
        assert years >= 2.9  # ~3 years
        assert len(evidence) > 0

    def test_year_only_range(self):
        text = "Data Analyst, 2018 - 2020"
        years, _ = self.parser.extract_date_ranges(text)
        assert years is not None
        assert years >= 1.5

    def test_no_date_ranges(self):
        text = "I love programming and machine learning"
        years, evidence = self.parser.extract_date_ranges(text)
        assert years is None
        assert evidence == []

    def test_multiple_ranges_no_overlap(self):
        text = """
        Software Engineer, Jan 2018 - Dec 2019
        Senior Engineer, Jan 2021 - Dec 2022
        """
        years, _ = self.parser.extract_date_ranges(text)
        assert years is not None
        assert years >= 3.5  # ~4 years total (no overlap)

    def test_overlapping_ranges_merged(self):
        text = """
        Developer, Jan 2020 - Dec 2021
        Lead Developer, Jun 2021 - Dec 2022
        """
        years, _ = self.parser.extract_date_ranges(text)
        assert years is not None
        # Merged: Jan 2020 - Dec 2022 = 3 years, NOT 2 + 1.5 = 3.5
        assert years <= 3.5


# ---- Full parse ----

class TestParse:
    parser: ExperienceParser  # pyre-ignore[13]

    def setup_method(self):
        self.parser = ExperienceParser()

    def test_combined_method(self):
        text = """
        5 years of experience in Python.
        Software Engineer, Jan 2019 - Dec 2023
        """
        result = self.parser.parse(text)
        assert isinstance(result, ExperienceResult)
        assert result.method == "combined"
        assert result.years >= 4.0
        assert result.confidence > 0.0

    def test_explicit_only(self):
        text = "Over 8 years of industry experience."
        result = self.parser.parse(text)
        assert result.method == "explicit"
        assert result.years == 8.0

    def test_date_ranges_only(self):
        text = "Software Engineer, Jan 2020 - Dec 2022"
        result = self.parser.parse(text)
        assert result.method == "date_ranges"
        assert result.years >= 2.5

    def test_no_experience(self):
        text = "Fresh graduate looking for opportunities."
        result = self.parser.parse(text)
        assert result.method == "none"
        assert result.years == 0.0
        assert result.confidence == 0.1

    def test_months_field_consistent(self):
        text = "3 years of experience"
        result = self.parser.parse(text)
        assert result.months == int(round(result.years * 12))


# ---- Legacy wrapper ----

class TestParseExperience:
    def test_returns_dict(self):
        result = parse_experience("5 years experience in ML")
        assert isinstance(result, dict)
        assert "years" in result
        assert "months" in result
        assert "method" in result
        assert "confidence" in result
        assert "evidence" in result
