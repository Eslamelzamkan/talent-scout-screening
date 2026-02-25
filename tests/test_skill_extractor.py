"""
Tests for core/skill_extractor.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

from core.skill_extractor import SkillExtractor, SkillHit, SkillMatchResult  # pyre-ignore[21]

import pytest  # pyre-ignore[21]


@pytest.fixture
def extractor():
    return SkillExtractor()


# ---- Skill extraction ----

class TestExtract:
    def test_finds_python(self, extractor):
        hits = extractor.extract("Experienced in Python and data analysis")
        assert "python" in hits

    def test_finds_multiple_skills(self, extractor):
        text = "Built ML pipelines using Python, PyTorch, and Docker on AWS"
        hits = extractor.extract(text)
        assert "python" in hits
        assert "pytorch" in hits
        assert "docker" in hits
        assert "aws" in hits

    def test_alias_sklearn(self, extractor):
        hits = extractor.extract("Used sklearn for classification tasks")
        assert "scikit-learn" in hits

    def test_alias_k8s(self, extractor):
        hits = extractor.extract("Deployed services on k8s clusters")
        assert "kubernetes" in hits

    def test_alias_cpp(self, extractor):
        hits = extractor.extract("Proficient in C++ and Python")
        assert "c++" in hits

    def test_go_not_matched_plain(self, extractor):
        """'go' alone should NOT match to avoid false positives; 'golang' should."""
        hits = extractor.extract("I want to go to the store")
        assert "go" not in hits

    def test_golang_matched(self, extractor):
        hits = extractor.extract("Experience with Golang microservices")
        assert "go" in hits

    def test_empty_text(self, extractor):
        hits = extractor.extract("")
        assert hits == {}

    def test_none_text(self, extractor):
        hits = extractor.extract(None)
        assert hits == {}

    def test_count_multiple_mentions(self, extractor):
        text = "Python developer. Used Python for web apps. Python is great."
        hits = extractor.extract(text)
        assert hits["python"].count >= 3

    def test_surface_forms_recorded(self, extractor):
        hits = extractor.extract("Experience with PyTorch and pytorch")
        assert "pytorch" in hits
        assert len(hits["pytorch"].surface_forms) >= 1

    def test_sentence_transformers(self, extractor):
        hits = extractor.extract("Fine-tuned sentence-transformers model for embeddings")
        assert "sentence-transformers" in hits


# ---- Skill normalization ----

class TestNormalizeSkill:
    def test_canonical_name(self, extractor):
        assert extractor.normalize_skill("python") == "python"

    def test_case_insensitive(self, extractor):
        assert extractor.normalize_skill("PyTorch") == "pytorch"

    def test_alias_resolves(self, extractor):
        assert extractor.normalize_skill("sklearn") == "scikit-learn"

    def test_unknown_returns_none(self, extractor):
        assert extractor.normalize_skill("fortran") is None

    def test_empty_returns_none(self, extractor):
        assert extractor.normalize_skill("") is None

    def test_blacklisted_returns_none(self, extractor):
        assert extractor.normalize_skill("and") is None


# ---- Skill matching ----

class TestMatchSkills:
    def test_full_match(self, extractor):
        candidate = extractor.extract("Expert in Python and PyTorch")
        result = extractor.match_skills(["python", "pytorch"], candidate)
        assert result.match_rate == 100.0
        assert len(result.missing) == 0

    def test_partial_match(self, extractor):
        candidate = extractor.extract("Expert in Python")
        result = extractor.match_skills(["python", "pytorch", "docker"], candidate)
        assert 30 <= result.match_rate <= 40  # 1/3 ≈ 33.33%
        assert "python" in result.present
        assert "pytorch" in result.missing
        assert "docker" in result.missing

    def test_no_match(self, extractor):
        candidate = extractor.extract("I like cooking and gardening")
        result = extractor.match_skills(["python", "pytorch"], candidate)
        assert result.match_rate == 0.0
        assert len(result.present) == 0

    def test_empty_requirements(self, extractor):
        candidate = extractor.extract("Expert in Python")
        result = extractor.match_skills([], candidate)
        assert result.match_rate == 100.0

    def test_result_types(self, extractor):
        candidate = extractor.extract("Python and Docker")
        result = extractor.match_skills(["python"], candidate)
        assert isinstance(result, SkillMatchResult)
        assert isinstance(result.required, tuple)
        assert isinstance(result.present, tuple)
        assert isinstance(result.missing, tuple)
