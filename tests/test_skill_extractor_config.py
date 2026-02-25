"""
Tests for external skill alias configuration in core/skill_extractor.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.skill_extractor import SkillExtractor  # pyre-ignore[21]


def test_custom_skill_aliases_loaded_from_file(tmp_path):
    config_file = tmp_path / "skills_aliases.yml"
    config_file.write_text(
        "skill_aliases:\n"
        "  rust:\n"
        "    - rust\n"
        "    - rustlang\n",
        encoding="utf-8",
    )

    extractor = SkillExtractor(skill_aliases_path=str(config_file))
    hits = extractor.extract("Built low-latency systems with Rustlang")

    assert "rust" in hits
    # Built-in defaults should remain available after merge.
    assert extractor.normalize_skill("python") == "python"
