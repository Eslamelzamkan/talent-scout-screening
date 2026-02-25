"""
skills_extractor.py

Grad-level skill extraction that is:
- deterministic (no LLM required)
- fast (single pass regex matching)
- auditable (keeps evidence: surface forms + counts)

This extractor is intentionally conservative to avoid false positives
(e.g., matching "go" in normal text, or "R" in a sentence).

Usage
-----
from skills_extractor import SkillExtractor

extractor = SkillExtractor()
resume_skills = extractor.extract(resume_text)
jd_skills = extractor.extract(jd_text)

match = extractor.match_skills(jd_required=["python","pytorch"], candidate=resume_skills)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from core.utils import normalize_for_matching


# -----------------------------
# Defaults
# -----------------------------

logger = logging.getLogger(__name__)

# Allows users to define extra aliases from disk.
SKILL_ALIASES_ENV_VAR = "SKILL_ALIASES_PATH"
DEFAULT_SKILL_ALIASES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "skills_aliases.yml",
)

# Canonical -> aliases (lowercase expected)
DEFAULT_SKILL_ALIASES: Dict[str, List[str]] = {
    # Languages
    "python": ["python"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "c sharp"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "go": ["golang", "go language"],  # avoid plain "go" (too many false positives)
    "sql": ["sql", "postgresql", "mysql", "sqlite"],
    # ML / DS
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "tf"],  # "tf" is noisy; handled by boundary rules
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "numpy": ["numpy"],
    "pandas": ["pandas"],
    "nlp": ["nlp", "natural language processing"],
    "transformers": ["transformers", "huggingface transformers", "hf transformers"],
    "sentence-transformers": ["sentence-transformers", "sentence transformers", "sbert"],
    # Dev / Infra
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "linux": ["linux"],
    "git": ["git", "github", "gitlab"],
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud"],
    "azure": ["azure"],
    # Data / pipelines
    "etl": ["etl", "data pipeline", "data pipelines"],
    "airflow": ["airflow", "apache airflow"],
    "spark": ["spark", "apache spark"],
    "pyspark": ["pyspark"],
    "kafka": ["kafka", "apache kafka"],
    "databricks": ["databricks"],
    "snowflake": ["snowflake"],
    # Backend / frameworks
    "fastapi": ["fastapi"],
    "flask": ["flask"],
    "django": ["django"],
    "sqlalchemy": ["sqlalchemy"],
    # Ops / tooling
    "terraform": ["terraform"],
    "ansible": ["ansible"],
    "jenkins": ["jenkins"],
    "github-actions": ["github actions", "github workflows"],
    "mlflow": ["mlflow"],
    # BI / analytics
    "tableau": ["tableau"],
    "power-bi": ["power bi", "powerbi"],
    # Frontend (common in full-stack resumes)
    "react": ["react", "reactjs"],
    "node.js": ["node.js", "nodejs"],
}

# Terms that should never be treated as standalone skills (noise)
DEFAULT_BLACKLIST: Set[str] = {
    "in",
    "at",
    "on",
    "and",
    "or",
    "to",
    "for",
    "with",
    "the",
    "a",
    "an",
    "as",
    "of",
    "is",
    "are",
    "was",
    "were",
    "be",
}


def _normalize_alias_map(skill_aliases: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for canon, aliases in skill_aliases.items():
        if not isinstance(canon, str):
            continue
        canon_n = canon.strip().lower()
        if not canon_n:
            continue
        clean_aliases = []
        for alias in aliases or []:
            if not isinstance(alias, str):
                continue
            alias_n = alias.strip().lower()
            if alias_n:
                clean_aliases.append(alias_n)
        if clean_aliases:
            out[canon_n] = clean_aliases
    return out


def _merge_skill_aliases(
    base: Dict[str, List[str]], overrides: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    merged = {
        canon: list(aliases)
        for canon, aliases in _normalize_alias_map(base).items()
    }
    for canon, aliases in _normalize_alias_map(overrides).items():
        current = set(merged.get(canon, []))
        current.update(aliases)
        merged[canon] = sorted(current)
    return merged


def _load_skill_aliases_from_file(path: str) -> Dict[str, List[str]]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            if path.lower().endswith(".json"):
                raw = json.load(f)
            else:
                import yaml  # pyre-ignore[21]

                raw = yaml.safe_load(f)
    except Exception as exc:
        logger.warning("Could not load skill aliases from %s: %s", path, exc)
        return {}

    if not isinstance(raw, dict):
        logger.warning("Skill aliases file must contain an object/map: %s", path)
        return {}

    payload = raw.get("skill_aliases", raw)
    if not isinstance(payload, dict):
        logger.warning("`skill_aliases` must be a mapping in %s", path)
        return {}

    normalized: Dict[str, List[str]] = {}
    for canon, aliases in payload.items():
        if isinstance(aliases, str):
            alias_list: List[str] = [aliases]
        elif isinstance(aliases, (list, tuple, set)):
            alias_list = [a for a in aliases if isinstance(a, str)]
        else:
            continue
        if alias_list:
            normalized[str(canon)] = alias_list

    loaded = _normalize_alias_map(normalized)
    if loaded:
        logger.info("Loaded %d custom skills from %s", len(loaded), path)
    return loaded


# -----------------------------
# Data classes
# -----------------------------


@dataclass(frozen=True)
class SkillHit:
    canonical: str
    count: int
    surface_forms: Tuple[str, ...]


@dataclass(frozen=True)
class SkillMatchResult:
    required: Tuple[str, ...]
    present: Tuple[str, ...]
    missing: Tuple[str, ...]
    match_rate: float  # 0..100
    evidence: Dict[str, SkillHit]


# -----------------------------
# Implementation
# -----------------------------


def _escape_alias(alias: str) -> str:
    """
    Escape alias for regex and normalize common punctuation.
    """
    return re.escape(alias.strip().lower())


def _is_single_token(alias: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9\+#\.\-]+", alias.strip().lower()))


class SkillExtractor:
    """
    Deterministic skill extractor with alias-based matching.

    Notes on precision:
    - Single-token aliases are matched with word boundaries where possible.
    - Very short aliases (<=2) are treated carefully to reduce false positives.
    """

    def __init__(
        self,
        skill_aliases: Optional[Dict[str, List[str]]] = None,
        skill_aliases_path: Optional[str] = None,
        blacklist: Optional[Set[str]] = None,
        min_alias_len: int = 3,
    ):
        if skill_aliases is not None:
            resolved_aliases = _normalize_alias_map(skill_aliases)
        else:
            resolved_path = skill_aliases_path or os.getenv(
                SKILL_ALIASES_ENV_VAR,
                DEFAULT_SKILL_ALIASES_PATH,
            )
            file_aliases = _load_skill_aliases_from_file(resolved_path)
            resolved_aliases = _merge_skill_aliases(DEFAULT_SKILL_ALIASES, file_aliases)

        self.skill_aliases = resolved_aliases
        self.blacklist = blacklist or DEFAULT_BLACKLIST
        self.min_alias_len = min_alias_len

        self._alias_to_canonical: Dict[str, str] = {}
        for canon, aliases in self.skill_aliases.items():
            canon_n = canon.strip().lower()
            for a in aliases:
                a_n = a.strip().lower()
                if not a_n or a_n in self.blacklist:
                    continue
                self._alias_to_canonical[a_n] = canon_n

        self._pattern = self._compile_pattern(self._alias_to_canonical.keys())

    def _compile_pattern(self, aliases: Iterable[str]) -> re.Pattern:
        """
        Compile one big regex to find all aliases in one pass.
        Uses careful boundary handling for short/ambiguous aliases.
        """
        parts = []
        for alias in sorted(set(aliases), key=len, reverse=True):
            if alias in self.blacklist:
                continue

            a = alias.strip().lower()
            if len(a) < self.min_alias_len and a not in {"c#", "c++", "tf", "js", "ts"}:
                continue

            esc = _escape_alias(a)

            if _is_single_token(a):
                # word boundary: for tokens with +/# we can't rely on \b
                if any(ch in a for ch in ["+", "#"]):
                    # require non-word char around it
                    parts.append(rf"(?<!\w){esc}(?!\w)")
                else:
                    parts.append(rf"\b{esc}\b")
            else:
                # phrase: allow flexible whitespace and punctuation between words
                # replace escaped spaces with \s+
                esc = esc.replace(r"\ ", r"\s+")
                parts.append(rf"(?<!\w){esc}(?!\w)")

        if not parts:
            # never happens with defaults, but be safe
            parts = [r"$^"]

        big = "|".join(parts)
        return re.compile(big, flags=re.IGNORECASE)

    def extract(self, text: str) -> Dict[str, SkillHit]:
        """
        Extract skills from text, returning canonical -> SkillHit.
        """
        norm = normalize_for_matching(text or "")
        if not norm:
            return {}

        counts: Dict[str, int] = {}
        surfaces: Dict[str, Set[str]] = {}

        for m in self._pattern.finditer(norm):
            surface = m.group(0).strip().lower()
            canon = self._alias_to_canonical.get(surface)

            # If surface isn't directly in alias map (because of flexible whitespace),
            # try to map by collapsing whitespace.
            if canon is None:
                surface2 = re.sub(r"\s+", " ", surface)
                canon = self._alias_to_canonical.get(surface2)

            if canon is None:
                # Should be rare; ignore unknown match
                continue

            counts[canon] = counts.get(canon, 0) + 1
            surfaces.setdefault(canon, set()).add(surface)

        out: Dict[str, SkillHit] = {
            canon: SkillHit(
                canonical=canon,
                count=c,
                surface_forms=tuple(sorted(surfaces.get(canon, set()))),
            )
            for canon, c in counts.items()
        }
        return out

    def normalize_skill(self, skill: str) -> Optional[str]:
        """
        Normalize a user-entered skill (e.g., from UI) to canonical.
        Returns None if unknown/blacklisted.
        """
        s = (skill or "").strip().lower()
        if not s or s in self.blacklist:
            return None

        if s in self.skill_aliases:
            return s

        if s in self._alias_to_canonical:
            return self._alias_to_canonical[s]

        # try small normalizations
        s2 = re.sub(r"\s+", " ", s)
        return self._alias_to_canonical.get(s2)

    def match_skills(
        self,
        jd_required: Sequence[str],
        candidate: Dict[str, SkillHit],
    ) -> SkillMatchResult:
        """
        Compute match against a list of required skills.
        Returns a result with present/missing and match_rate 0..100.
        """
        req_norm: List[str] = []
        for s in jd_required:
            if (canon := self.normalize_skill(s)):
                req_norm.append(canon)

        req_set = tuple(sorted(set(req_norm)))
        if not req_set:
            # nothing required => full match
            present = tuple(sorted(candidate.keys()))
            return SkillMatchResult(
                required=(),
                present=present,
                missing=(),
                match_rate=100.0,
                evidence=candidate,
            )

        cand_set = set(candidate.keys())
        present = tuple(sorted([s for s in req_set if s in cand_set]))
        missing = tuple(sorted([s for s in req_set if s not in cand_set]))

        match_rate = 100.0 * (len(present) / max(1, len(req_set)))

        return SkillMatchResult(
            required=req_set,
            present=present,
            missing=missing,
            match_rate=float(round(match_rate, 2)),
            evidence=candidate,
        )
