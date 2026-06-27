"""
skill_extractor.py — Deterministic regex-based skill extraction.

Ported from talent-scout-screening/core/skill_extractor.py.
Import path changed: core.utils → core.utils
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

SKILL_ALIASES_ENV_VAR = "SKILL_ALIASES_PATH"
DEFAULT_SKILL_ALIASES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "skills_aliases.yml",
)

DEFAULT_SKILL_ALIASES: Dict[str, List[str]] = {
    # Languages
    "python": ["python"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "c sharp"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "go": ["golang", "go language"],
    "sql": ["sql", "postgresql", "mysql", "sqlite"],
    "r": ["r language", "r programming"],
    "scala": ["scala"],
    # ML / DS
    "machine-learning": ["machine learning", "applied machine learning", "ml models"],
    "deep-learning": ["deep learning", "neural networks"],
    "data-analysis": ["data analysis", "analytical modeling", "data analytics"],
    "statistics": ["statistics", "statistical analysis", "statistical modeling"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "tf"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "xgboost": ["xgboost"],
    "lightgbm": ["lightgbm"],
    "catboost": ["catboost"],
    "numpy": ["numpy"],
    "pandas": ["pandas"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "plotly": ["plotly"],
    "nlp": ["nlp", "natural language processing"],
    "computer-vision": ["computer vision"],
    "transformers": ["transformers", "huggingface transformers", "hf transformers"],
    "sentence-transformers": ["sentence-transformers", "sentence transformers", "sbert"],
    "llm": ["llm", "large language models", "large language model"],
    "langchain": ["langchain"],
    "openai-api": ["openai api", "openai"],
    # Dev / Infra
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "linux": ["linux"],
    "git": ["git", "github", "gitlab"],
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud"],
    "azure": ["azure"],
    "rest-api": ["rest api", "restful api", "rest services"],
    "microservices": ["microservices", "microservice architecture"],
    # Data / pipelines
    "etl": ["etl", "data pipeline", "data pipelines"],
    "airflow": ["airflow", "apache airflow"],
    "spark": ["spark", "apache spark"],
    "pyspark": ["pyspark"],
    "hadoop": ["hadoop", "apache hadoop"],
    "kafka": ["kafka", "apache kafka"],
    "databricks": ["databricks"],
    "snowflake": ["snowflake"],
    "bigquery": ["bigquery", "google bigquery"],
    "redshift": ["redshift", "amazon redshift"],
    # Backend / frameworks
    "fastapi": ["fastapi"],
    "flask": ["flask"],
    "django": ["django"],
    "express": ["express", "express.js"],
    "nestjs": ["nestjs", "nest.js"],
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
    "looker": ["looker", "looker studio"],
    # Frontend
    "react": ["react", "reactjs"],
    "next.js": ["next.js", "nextjs"],
    "vue": ["vue", "vue.js"],
    "angular": ["angular"],
    "tailwind": ["tailwind", "tailwind css"],
    "node.js": ["node.js", "nodejs"],
}

DEFAULT_BLACKLIST: Set[str] = {
    "in", "at", "on", "and", "or", "to", "for", "with", "the",
    "a", "an", "as", "of", "is", "are", "was", "were", "be",
}

JD_REQUIRED_HEADING_RE = re.compile(
    r"^\s*(requirements?|required qualifications?|minimum qualifications?|must[-\s]*have|what you'll need)\s*:?\s*$",
    flags=re.IGNORECASE,
)

JD_PREFERRED_HEADING_RE = re.compile(
    r"^\s*(preferred qualifications?|nice to have|good to have|bonus skills?|plus skills?)\s*:?\s*$",
    flags=re.IGNORECASE,
)

JD_REQUIRED_LINE_RE = re.compile(
    r"\b(required?|must(?:\s+have)?|minimum|need(?:ed)?\s+to\s+have|strong\s+experience|hands[-\s]*on\s+experience)\b",
    flags=re.IGNORECASE,
)

JD_PREFERRED_LINE_RE = re.compile(
    r"\b(preferred|nice\s+to\s+have|good\s+to\s+have|a\s+plus|bonus|optional)\b",
    flags=re.IGNORECASE,
)

SKILL_INFERENCE_MAP: Dict[str, Set[str]] = {
    "machine-learning": {
        "scikit-learn", "xgboost", "lightgbm", "catboost", "tensorflow",
        "pytorch", "deep-learning", "nlp", "computer-vision",
    },
    "deep-learning": {"tensorflow", "pytorch", "transformers", "computer-vision"},
    "etl": {"airflow", "spark", "pyspark", "kafka", "databricks", "snowflake"},
    "data-analysis": {"statistics", "pandas", "sql", "tableau", "power-bi", "looker"},
    "statistics": {"data-analysis", "machine-learning", "scikit-learn"},
    "rest-api": {"fastapi", "flask", "django", "express", "nestjs"},
    "microservices": {"docker", "kubernetes", "rest-api"},
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
                import yaml  # type: ignore
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
    preferred: Tuple[str, ...]
    present: Tuple[str, ...]
    missing: Tuple[str, ...]
    preferred_present: Tuple[str, ...]
    preferred_missing: Tuple[str, ...]
    match_rate: float  # 0..100
    evidence: Dict[str, SkillHit]


@dataclass(frozen=True)
class JobSkillProfile:
    required: Tuple[str, ...]
    preferred: Tuple[str, ...]
    detected: Tuple[str, ...]


# -----------------------------
# Implementation
# -----------------------------

def _escape_alias(alias: str) -> str:
    return re.escape(alias.strip().lower())


def _is_single_token(alias: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9\+#\.\-]+", alias.strip().lower()))


def _dedupe_keep_order(values: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    seen: Set[str] = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return tuple(out)


class SkillExtractor:
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
        parts = []
        for alias in sorted(set(aliases), key=len, reverse=True):
            if alias in self.blacklist:
                continue

            a = alias.strip().lower()
            if len(a) < self.min_alias_len and a not in {"c#", "c++", "tf", "js", "ts"}:
                continue

            esc = _escape_alias(a)

            if _is_single_token(a):
                if any(ch in a for ch in ["+", "#"]):
                    parts.append(rf"(?<!\w){esc}(?!\w)")
                else:
                    parts.append(rf"\b{esc}\b")
            else:
                esc = esc.replace(r"\ ", r"\s+")
                parts.append(rf"(?<!\w){esc}(?!\w)")

        if not parts:
            parts = [r"$^"]

        big = "|".join(parts)
        return re.compile(big, flags=re.IGNORECASE)

    def extract(self, text: str) -> Dict[str, SkillHit]:
        norm = normalize_for_matching(text or "")
        if not norm:
            return {}

        counts: Dict[str, int] = {}
        surfaces: Dict[str, Set[str]] = {}

        for m in self._pattern.finditer(norm):
            surface = m.group(0).strip().lower()
            canon = self._alias_to_canonical.get(surface)

            if canon is None:
                surface2 = re.sub(r"\s+", " ", surface)
                canon = self._alias_to_canonical.get(surface2)

            if canon is None:
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
        s = (skill or "").strip().lower()
        if not s or s in self.blacklist:
            return None
        if s in self.skill_aliases:
            return s
        if s in self._alias_to_canonical:
            return self._alias_to_canonical[s]
        s2 = re.sub(r"\s+", " ", s)
        return self._alias_to_canonical.get(s2)

    def extract_job_profile(self, jd_text: str) -> JobSkillProfile:
        jd_norm = normalize_for_matching(jd_text or "")
        if not jd_norm:
            return JobSkillProfile(required=(), preferred=(), detected=())

        detected_all = tuple(sorted(self.extract(jd_norm).keys()))
        if not detected_all:
            return JobSkillProfile(required=(), preferred=(), detected=())

        required_hits: List[str] = []
        preferred_hits: List[str] = []
        section_mode: Optional[str] = None

        for raw_line in jd_norm.split("\n"):
            line = (raw_line or "").strip(" -\t")
            if not line:
                continue

            if JD_REQUIRED_HEADING_RE.search(line):
                section_mode = "required"
                continue
            if JD_PREFERRED_HEADING_RE.search(line):
                section_mode = "preferred"
                continue

            line_skills = tuple(sorted(self.extract(line).keys()))
            if not line_skills:
                continue

            if section_mode == "preferred" or JD_PREFERRED_LINE_RE.search(line):
                preferred_hits.extend(line_skills)
                continue

            if section_mode == "required" or JD_REQUIRED_LINE_RE.search(line):
                required_hits.extend(line_skills)
                continue

        required = list(_dedupe_keep_order(required_hits))
        preferred = [s for s in _dedupe_keep_order(preferred_hits) if s not in set(required)]

        if not required:
            required = list(detected_all)
            preferred = []

        return JobSkillProfile(
            required=tuple(required),
            preferred=tuple(preferred),
            detected=detected_all,
        )

    @staticmethod
    def _presence_strength(hit: SkillHit) -> float:
        count = max(0, int(hit.count))
        if count >= 3:
            return 1.0
        if count == 2:
            return 0.95
        if count == 1:
            return 0.88
        return 0.0

    def _skill_match_strength(self, skill: str, candidate: Dict[str, SkillHit]) -> float:
        direct_hit = candidate.get(skill)
        if direct_hit is not None:
            return self._presence_strength(direct_hit)

        related = SKILL_INFERENCE_MAP.get(skill, set())
        if not related:
            return 0.0

        related_scores = [self._presence_strength(candidate[r]) for r in related if r in candidate]
        if not related_scores:
            return 0.0

        best = max(related_scores)
        breadth_bonus = 0.03 * min(3, len(related_scores))
        return min(0.9, 0.55 + (0.30 * best) + breadth_bonus)

    def match_skills(
        self,
        jd_required: Sequence[str],
        candidate: Dict[str, SkillHit],
        jd_preferred: Optional[Sequence[str]] = None,
        required_weight: float = 0.82,
        preferred_weight: float = 0.18,
    ) -> SkillMatchResult:
        req_norm: List[str] = []
        for s in jd_required:
            if (canon := self.normalize_skill(s)):
                req_norm.append(canon)

        pref_norm: List[str] = []
        for s in (jd_preferred or []):
            if (canon := self.normalize_skill(s)):
                pref_norm.append(canon)

        req_set = tuple(sorted(set(req_norm)))
        pref_set = tuple(sorted([s for s in set(pref_norm) if s not in set(req_set)]))

        if not req_set and not pref_set:
            # No JD skills defined — return 0.0 so the skills component does not
            # artificially inflate every candidate's score for vague job descriptions.
            present = tuple(sorted(candidate.keys()))
            return SkillMatchResult(
                required=(),
                preferred=(),
                present=present,
                missing=(),
                preferred_present=(),
                preferred_missing=(),
                match_rate=0.0,
                evidence=candidate,
            )

        req_strengths: Dict[str, float] = {s: self._skill_match_strength(s, candidate) for s in req_set}
        pref_strengths: Dict[str, float] = {s: self._skill_match_strength(s, candidate) for s in pref_set}

        present = tuple(sorted([s for s, score in req_strengths.items() if score > 0]))
        missing = tuple(sorted([s for s, score in req_strengths.items() if score <= 0]))
        preferred_present = tuple(sorted([s for s, score in pref_strengths.items() if score > 0]))
        preferred_missing = tuple(sorted([s for s, score in pref_strengths.items() if score <= 0]))

        req_score = 0.0
        if req_set:
            req_score = sum(req_strengths.values()) / len(req_set)

        pref_score = 0.0
        if pref_set:
            pref_score = sum(pref_strengths.values()) / len(pref_set)

        if req_set and pref_set:
            r_w = max(0.0, float(required_weight))
            p_w = max(0.0, float(preferred_weight))
            total_w = max(1e-6, r_w + p_w)
            pref_bonus_scale = p_w / total_w
            req_component = req_score
            # Preferred skills should improve the score, not punish missing optional skills.
            match_rate_01 = req_component + ((1.0 - req_component) * pref_bonus_scale * pref_score)
        elif req_set:
            match_rate_01 = req_score
        else:
            match_rate_01 = pref_score

        return SkillMatchResult(
            required=req_set,
            preferred=pref_set,
            present=present,
            missing=missing,
            preferred_present=preferred_present,
            preferred_missing=preferred_missing,
            match_rate=float(round(100.0 * max(0.0, min(match_rate_01, 1.0)), 2)),
            evidence=candidate,
        )
