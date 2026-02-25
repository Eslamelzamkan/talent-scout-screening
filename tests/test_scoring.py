"""
Tests for core/scoring.py
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.scoring import (  # pyre-ignore[21]
    _normalize_weights,
    apply_role_profile,
    list_role_profiles,
    _get_skills_match_rate,
    _get_years_experience,
    _experience_score_cap,
    _experience_score_fit,
    compute_final_score,
    should_shortlist,
    ROLE_PROFILES,
)

import pytest  # pyre-ignore[21]


# ---- Weight normalization ----

class TestNormalizeWeights:
    def test_weights_sum_to_one(self):
        cfg = {"semantic_weight": 0.6, "skills_weight": 0.25, "experience_weight": 0.15}
        result = _normalize_weights(cfg)
        total = result["semantic_weight"] + result["skills_weight"] + result["experience_weight"]
        assert abs(total - 1.0) < 1e-9

    def test_unbalanced_weights_normalized(self):
        cfg = {"semantic_weight": 2.0, "skills_weight": 1.0, "experience_weight": 1.0}
        result = _normalize_weights(cfg)
        assert abs(result["semantic_weight"] - 0.5) < 1e-9
        assert abs(result["skills_weight"] - 0.25) < 1e-9

    def test_zero_weights_fallback(self):
        cfg = {"semantic_weight": 0, "skills_weight": 0, "experience_weight": 0}
        result = _normalize_weights(cfg)
        assert result["semantic_weight"] == 0.6
        assert result["skills_weight"] == 0.25
        assert result["experience_weight"] == 0.15

    def test_does_not_mutate_original(self):
        cfg = {"semantic_weight": 0.6, "skills_weight": 0.25, "experience_weight": 0.15}
        original = dict(cfg)
        _normalize_weights(cfg)
        assert cfg == original


# ---- Role profiles ----

class TestRoleProfiles:
    def test_list_profiles_returns_all(self):
        profiles = list_role_profiles()
        assert "fresh_grad" in profiles
        assert "senior" in profiles
        assert "manager" in profiles

    def test_apply_known_role(self):
        cfg, profile = apply_role_profile({}, "senior")
        assert profile is not None
        assert cfg["role_profile"] == "senior"
        assert "semantic_weight" in cfg

    def test_apply_custom_role(self):
        cfg, profile = apply_role_profile({"semantic_weight": 0.5, "skills_weight": 0.3, "experience_weight": 0.2}, "custom")
        assert profile is None
        assert cfg["role_profile"] == "custom"

    def test_apply_unknown_role_falls_back_to_custom(self):
        cfg, profile = apply_role_profile({}, "astronaut")
        assert profile is None
        assert cfg["role_profile"] == "custom"

    def test_apply_none_role(self):
        cfg, profile = apply_role_profile({}, None)
        assert profile is None
        assert cfg["role_profile"] == "custom"


# ---- Skills match rate extraction ----

class TestGetSkillsMatchRate:
    def test_dict_with_match_rate(self):
        c = {"skills_match": {"match_rate": 75.0}}
        assert _get_skills_match_rate(c) == 75.0

    def test_numeric_skills_match(self):
        c = {"skills_match": 80}
        assert _get_skills_match_rate(c) == 80.0

    def test_skills_match_rate_fallback(self):
        c = {"skills_match_rate": 65.0}
        assert _get_skills_match_rate(c) == 65.0

    def test_missing_returns_zero(self):
        assert _get_skills_match_rate({}) == 0.0

    def test_clamped_to_100(self):
        c = {"skills_match": 150}
        assert _get_skills_match_rate(c) == 100.0


# ---- Years experience extraction ----

class TestGetYearsExperience:
    def test_dict_with_years(self):
        c = {"experience": {"years": 5.0}}
        assert _get_years_experience(c) == 5.0

    def test_numeric_experience(self):
        c = {"experience": 3}
        assert _get_years_experience(c) == 3.0

    def test_years_experience_fallback(self):
        # years_experience is only used when experience is a non-numeric type (e.g. string)
        c = {"experience": "unknown", "years_experience": 7.5}
        assert _get_years_experience(c) == 7.5

    def test_missing_returns_zero(self):
        assert _get_years_experience({}) == 0.0

    def test_negative_clamped_to_zero(self):
        c = {"experience": -2}
        assert _get_years_experience(c) == 0.0


# ---- Experience score (cap-based) ----

class TestExperienceScoreCap:
    def test_at_cap_returns_100(self):
        assert _experience_score_cap(10, 10) == 100.0

    def test_half_cap(self):
        assert _experience_score_cap(5, 10) == 50.0

    def test_over_cap_clamped(self):
        assert _experience_score_cap(15, 10) == 100.0

    def test_zero_years(self):
        assert _experience_score_cap(0, 10) == 0.0

    def test_zero_cap_returns_zero(self):
        assert _experience_score_cap(5, 0) == 0.0


# ---- Experience score (role-fit) ----

class TestExperienceScoreFit:
    def test_inside_range_returns_100(self):
        assert _experience_score_fit(7, (5.0, 10.0)) == 100.0

    def test_below_range_ramps(self):
        score = _experience_score_fit(2.5, (5.0, 10.0))
        assert abs(score - 50.0) < 1e-9

    def test_above_range_penalty(self):
        score = _experience_score_fit(20, (5.0, 10.0))
        assert abs(score - 50.0) < 1e-9  # (10/20) * 100 = 50

    def test_at_min_boundary(self):
        assert _experience_score_fit(5.0, (5.0, 10.0)) == 100.0

    def test_at_max_boundary(self):
        assert _experience_score_fit(10.0, (5.0, 10.0)) == 100.0

    def test_zero_years_zero_min(self):
        assert _experience_score_fit(0, (0.0, 1.0)) == 100.0


# ---- Final score computation ----

class TestComputeFinalScore:
    def test_basic_computation(self):
        candidate = {"score": 0.8, "skills_match": 60, "experience": 5}
        config = {
            "semantic_weight": 0.5,
            "skills_weight": 0.3,
            "experience_weight": 0.2,
            "experience_cap": 10,
            "use_role_experience_fit": False,
        }
        result = compute_final_score(candidate, config)
        assert "final_score" in result
        assert "breakdown" in result
        assert result["final_score"] > 0

    def test_breakdown_has_all_keys(self):
        candidate = {"score": 0.5, "skills_match": 50, "experience": 3}
        config = {"experience_cap": 10, "use_role_experience_fit": False}
        result = compute_final_score(candidate, config)
        bd = result["breakdown"]
        assert "semantic" in bd
        assert "skills" in bd
        assert "experience" in bd
        assert "weights" in bd

    def test_perfect_candidate(self):
        candidate = {"score": 1.0, "skills_match": 100, "experience": 10}
        config = {
            "semantic_weight": 0.33,
            "skills_weight": 0.33,
            "experience_weight": 0.34,
            "experience_cap": 10,
            "use_role_experience_fit": False,
        }
        result = compute_final_score(candidate, config)
        assert result["final_score"] >= 99.0

    def test_zero_candidate(self):
        candidate = {"score": 0, "skills_match": 0, "experience": 0}
        config = {"experience_cap": 10, "use_role_experience_fit": False}
        result = compute_final_score(candidate, config)
        assert result["final_score"] == 0.0


# ---- Shortlisting ----

class TestShouldShortlist:
    def test_above_threshold(self):
        assert should_shortlist({"final_score": 80}, {"shortlist_threshold": 75}) is True

    def test_below_threshold(self):
        assert should_shortlist({"final_score": 60}, {"shortlist_threshold": 75}) is False

    def test_at_threshold(self):
        assert should_shortlist({"final_score": 75}, {"shortlist_threshold": 75}) is True

    def test_missing_score_defaults_to_zero(self):
        assert should_shortlist({}, {"shortlist_threshold": 75}) is False
