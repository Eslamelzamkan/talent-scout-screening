import os
import logging
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

# Resolve config path relative to the Screening/ project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


# -----------------------------
# Role/Seniority presets
# -----------------------------
ROLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "fresh_grad": {
        "semantic_weight": 0.55,
        "skills_weight": 0.35,
        "experience_weight": 0.10,
        "experience_cap": 2,
        "years_range": (0.0, 1.0),          # ✅ NEW: expected years for this role
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
    """Ensure weights sum to 1.0 (prevents weird scoring when UI sliders drift)."""
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
    """Expose profiles to UI."""
    return deepcopy(ROLE_PROFILES)


def _fallback_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _normalize_weights(config)
    cfg["role_profile"] = "custom"
    return cfg


def apply_role_profile(config: Dict[str, Any], role: Optional[str]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Apply a seniority preset on top of config.
    Returns (new_config, profile_used).
    """
    if not role or role == "custom":
        return _fallback_config(config), None

    role_key = str(role).strip().lower()
    profile = ROLE_PROFILES.get(role_key)
    if not profile:
        return _fallback_config(config), None

    new_cfg = deepcopy(config)
    new_cfg.update(profile)  # sets weights + caps + thresholds + years_range
    new_cfg = _normalize_weights(new_cfg)
    new_cfg["role_profile"] = role_key
    return new_cfg, deepcopy(profile)


# -----------------------------
# Existing config loader
# -----------------------------
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load scoring configuration from YAML file.
    Defaults to <project_root>/config/scoring.yaml.
    """
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

    # Resolve path: use provided path or default to project root config/
    resolved_path = Path(path) if path else (_PROJECT_ROOT / "config" / "scoring.yaml")

    try:
        if resolved_path.exists():
            with open(resolved_path, "r") as f:
                config = yaml.safe_load(f) or {}
                merged = {**default_config, **config}
                return _normalize_weights(merged)
        else:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            with open(resolved_path, "w") as f:
                yaml.dump(default_config, f, default_flow_style=False)
            return _normalize_weights(default_config)
    except Exception as e:
        logger.warning("Config load failed, using defaults: %s", e)
        return _normalize_weights(default_config)


# -----------------------------
# Robust extractors (✅ NEW)
# -----------------------------
def _get_skills_match_rate(candidate: Dict[str, Any]) -> float:
    """
    Returns 0..100.
    Supports:
    - candidate["skills_match"]["match_rate"]
    - candidate["skills_match"] as numeric
    - candidate["skills_match_rate"] as numeric
    """
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
    """
    Supports:
    - candidate["experience"]["years"]
    - candidate["experience"] numeric
    - candidate["years_experience"]
    """
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
    """0..100 based on cap."""
    cap = float(cap or 0)
    return 0.0 if cap <= 0 else max(0.0, min((years / cap) * 100.0, 100.0))


def _experience_score_fit(years: float, years_range: Tuple[float, float]) -> float:
    """
    0..100 role-fit score using expected [min,max].
    - inside range => 100
    - below min    => linear ramp
    - above max    => gentle penalty (max/years)
    """
    mn, mx = float(years_range[0]), float(years_range[1])

    if years <= 0 and mn <= 0:
        return 100.0

    if years < mn:
        return 100.0 if mn <= 0 else max(0.0, min((years / mn) * 100.0, 100.0))

    if years > mx:
        return 0.0 if years <= 0 else max(0.0, min((mx / years) * 100.0, 100.0))

    return 100.0


def compute_final_score(candidate: dict, config: dict) -> dict:
    """
    Calculate weighted final score from semantic, skills, and experience.
    Assumes semantic score in candidate["score"] is 0..1 (SentenceTransformer cosine).
    """
    config = _normalize_weights(config)

    semantic_raw = candidate.get("score", 0.0)  # 0..1
    try:
        semantic_raw = float(semantic_raw)
    except Exception:
        semantic_raw = 0.0
    semantic = max(0.0, min(semantic_raw, 1.0)) * 100.0

    skills = _get_skills_match_rate(candidate)  # 0..100
    years = _get_years_experience(candidate)

    # Experience: either cap-based or role-fit-based
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

    final_score = (w_semantic * semantic) + (w_skills * skills) + (w_experience * exp_score)

    return {
        "final_score": round(final_score, 2),
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
    threshold = float(config.get("shortlist_threshold", 75))
    try:
        final_score = float(final_score)
    except Exception:
        final_score = 0.0
    return final_score >= threshold
