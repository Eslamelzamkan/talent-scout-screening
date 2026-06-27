"""
scoring.py — Weighted final score computation + role profiles.

Ported from talent-scout-screening/core/scoring.py.
"""

import logging
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

# Resolve config path relative to the screening agent root
_AGENT_ROOT = Path(__file__).resolve().parent.parent  # agents/screening/
logger = logging.getLogger(__name__)


# -----------------------------
# Role/Seniority presets
# -----------------------------
# NOTE: shortlist_threshold values are authored on the RAW weighted-average
# scale. They are mapped through _calibrate_final_score() before being compared
# against a candidate's calibrated final_score (see should_shortlist).
ROLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "fresh_grad": {
        "semantic_weight": 0.55,
        "skills_weight": 0.35,
        "experience_weight": 0.10,
        "experience_cap": 2,
        "years_range": (0.0, 1.0),
        "shortlist_threshold": 70,
    },
    "junior": {
        "semantic_weight": 0.50,
        "skills_weight": 0.35,
        "experience_weight": 0.15,
        "experience_cap": 5,
        "years_range": (1.0, 3.0),
        "shortlist_threshold": 72,
    },
    "mid": {
        "semantic_weight": 0.50,
        "skills_weight": 0.30,
        "experience_weight": 0.20,
        "experience_cap": 8,
        "years_range": (3.0, 6.0),
        "shortlist_threshold": 75,
    },
    "senior": {
        "semantic_weight": 0.45,
        "skills_weight": 0.30,
        "experience_weight": 0.25,
        "experience_cap": 12,
        "years_range": (6.0, 10.0),
        "shortlist_threshold": 78,
    },
    "lead": {
        "semantic_weight": 0.40,
        "skills_weight": 0.30,
        "experience_weight": 0.30,
        "experience_cap": 15,
        "years_range": (8.0, 15.0),
        "shortlist_threshold": 80,
    },
    "manager": {
        "semantic_weight": 0.45,
        "skills_weight": 0.20,
        "experience_weight": 0.35,
        "experience_cap": 18,
        "years_range": (8.0, 20.0),
        "shortlist_threshold": 82,
    },
}


def _normalize_weights(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure weights sum to 1.0."""
    cfg = deepcopy(cfg)
    w_sem = float(cfg.get("semantic_weight", 0.6))
    w_ski = float(cfg.get("skills_weight", 0.25))
    w_exp = float(cfg.get("experience_weight", 0.15))
    s = w_sem + w_ski + w_exp
    if s <= 0:
        cfg["semantic_weight"], cfg["skills_weight"], cfg["experience_weight"] = 0.6, 0.25, 0.15
        return cfg

    cfg["semantic_weight"] = w_sem / s
    cfg["skills_weight"] = w_ski / s
    cfg["experience_weight"] = w_exp / s
    return cfg


def list_role_profiles() -> Dict[str, Dict[str, Any]]:
    return deepcopy(ROLE_PROFILES)


def _fallback_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _normalize_weights(config)
    cfg["role_profile"] = "custom"
    return cfg


def apply_role_profile(config: Dict[str, Any], role: Optional[str]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    if not role or role == "custom":
        return _fallback_config(config), None

    role_key = str(role).strip().lower()
    profile = ROLE_PROFILES.get(role_key)
    if not profile:
        return _fallback_config(config), None

    new_cfg = deepcopy(config)
    new_cfg.update(profile)
    new_cfg = _normalize_weights(new_cfg)
    new_cfg["role_profile"] = role_key
    return new_cfg, deepcopy(profile)


# -----------------------------
# Config loader
# -----------------------------
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    default_config: Dict[str, Any] = {
        "semantic_weight": 0.6,
        "skills_weight": 0.25,
        "experience_weight": 0.15,
        "experience_cap": 20,
        "minimum_score": 0,
        "shortlist_threshold": 75,
        "role_profile": "custom",
        "use_role_experience_fit": True,
    }

    resolved_path = Path(path) if path else (_AGENT_ROOT / "config" / "scoring.yaml")

    try:
        if resolved_path.exists():
            with open(resolved_path, "r") as f:
                config = yaml.safe_load(f) or {}
                merged = {**default_config, **config}
                return _normalize_weights(merged)
        else:
            # Don't auto-create config files; just use defaults
            return _normalize_weights(default_config)
    except Exception as e:
        logger.warning("Config load failed, using defaults: %s", e)
        return _normalize_weights(default_config)


# -----------------------------
# Robust extractors
# -----------------------------
def _get_skills_match_rate(candidate: Dict[str, Any]) -> float:
    sm = candidate.get("skills_match")
    if isinstance(sm, dict):
        val = sm.get("match_rate", 0.0)
    elif isinstance(sm, (int, float)):
        val = sm
    else:
        val = candidate.get("skills_match_rate", 0.0)

    try:
        val = float(val)
    except Exception:
        val = 0.0

    return max(0.0, min(val, 100.0))


def _get_years_experience(candidate: Dict[str, Any]) -> float:
    exp = candidate.get("experience", 0.0)
    if isinstance(exp, dict):
        val = exp.get("years", 0.0)
    elif isinstance(exp, (int, float)):
        val = exp
    else:
        val = candidate.get("years_experience", 0.0)

    try:
        val = float(val)
    except Exception:
        val = 0.0

    return max(0.0, val)


def _experience_score_cap(years: float, cap: float) -> float:
    cap = float(cap or 0)
    return 0.0 if cap <= 0 else max(0.0, min((years / cap) * 100.0, 100.0))


def _experience_score_fit(years: float, years_range: Tuple[float, float]) -> float:
    mn, mx = float(years_range[0]), float(years_range[1])

    if years <= 0 and mn <= 0:
        return 100.0
    if years < mn:
        return 100.0 if mn <= 0 else max(0.0, min((years / mn) * 100.0, 100.0))
    if years > mx:
        return 0.0 if years <= 0 else max(0.0, min((mx / years) * 100.0, 100.0))
    return 100.0


def _calibrate_final_score(raw_score: float) -> float:
    """
    Map raw weighted-average score to a human-interpretable scale.

    Raw scores empirically cluster in the 30-70 range because:
    - Cosine similarity between different document types rarely exceeds 0.65
    - Skill match rates are depressed by the required/preferred fallback
    - Experience scores are linear against high caps

    This piecewise-linear calibration stretches the distribution so that:
    - A strong candidate scores 75-90% (not 40-55%)
    - A weak candidate still scores below 40%
    - Perfect scores remain near 100%
    """
    breakpoints = [
        (0.0, 0.0),
        (25.0, 15.0),
        (35.0, 35.0),
        (45.0, 55.0),
        (55.0, 70.0),
        (65.0, 82.0),
        (75.0, 90.0),
        (85.0, 95.0),
        (100.0, 100.0),
    ]
    for i in range(len(breakpoints) - 1):
        x0, y0 = breakpoints[i]
        x1, y1 = breakpoints[i + 1]
        if raw_score <= x1:
            t = (raw_score - x0) / (x1 - x0) if x1 > x0 else 0.0
            return max(0.0, min(100.0, y0 + t * (y1 - y0)))
    return 100.0


def compute_final_score(candidate: dict, config: dict) -> dict:
    """
    Calculate weighted final score from semantic, skills, and experience.
    Assumes semantic score in candidate["score"] is 0..1 (SentenceTransformer cosine).
    """
    config = _normalize_weights(config)

    semantic_raw = candidate.get("score", 0.0)
    try:
        semantic_raw = float(semantic_raw)
    except Exception:
        semantic_raw = 0.0
    semantic = max(0.0, min(semantic_raw, 1.0)) * 100.0

    skills = _get_skills_match_rate(candidate)
    years = _get_years_experience(candidate)

    exp_cap = float(config.get("experience_cap", 20))
    exp_score = _experience_score_cap(years, exp_cap)

    if config.get("use_role_experience_fit", True):
        role = config.get("role_profile")
        if role and role != "custom":
            prof = ROLE_PROFILES.get(str(role).lower())
            if prof and "years_range" in prof:
                exp_score = _experience_score_fit(years, prof["years_range"])

    w_semantic = float(config.get("semantic_weight", 0.6))
    w_skills = float(config.get("skills_weight", 0.25))
    w_experience = float(config.get("experience_weight", 0.15))

    raw_score = (w_semantic * semantic) + (w_skills * skills) + (w_experience * exp_score)
    final_score = _calibrate_final_score(raw_score)

    return {
        "final_score": round(final_score, 2),
        "raw_score": round(raw_score, 2),
        "breakdown": {
            "semantic": round(semantic, 2),
            "skills": round(skills, 2),
            "experience": round(exp_score, 2),
            "weights": {
                "semantic": round(w_semantic, 3),
                "skills": round(w_skills, 3),
                "experience": round(w_experience, 3),
            },
            "role_profile": config.get("role_profile", "custom"),
            "years_experience": round(float(years), 2),
        },
    }


def should_shortlist(candidate: dict, config: dict) -> bool:
    final_score = candidate.get("final_score", 0)
    # shortlist_threshold is authored on the raw weighted-average scale; map it
    # through the same calibration curve as final_score before comparing so the
    # two operands share a scale.
    threshold = _calibrate_final_score(float(config.get("shortlist_threshold", 75)))
    try:
        final_score = float(final_score)
    except Exception:
        final_score = 0.0
    return final_score >= threshold
