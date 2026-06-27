# core/pipeline.py
#
# Batch screening orchestrator: rank many resumes against one job description.
#
# Flow (per the project's deployed design):
#   1. Parse the JD once with an LLM -> required/preferred skills, seniority, and a
#      hypothetical "ideal resume" summary (HyRe / HyDE-style query expansion).
#   2. Stage 1 semantic ranking: dense bi-encoder similarity against the ideal-resume
#      query, blended with a lexical keyword-overlap signal.
#   3. Skill match (required + preferred), experience parsing, entity extraction.
#   4. Final weighted, calibrated score (semantic / skills / experience).
#   5. Stage 2 deterministic explanation (+ optional LLM polish).
#
# Every external dependency degrades gracefully: no LLM -> regex JD parsing; no
# embedding model -> lexical-only semantic score; no database -> no persistence.

from __future__ import annotations

import concurrent.futures
import logging
import os
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from db.db_postgres import TalentScoutRepo  # pyre-ignore[21]
from core.jd_parser import parse_jd_with_llm  # pyre-ignore[21]
from core.model import lexical_alignment_score, rank_resumes_stage1  # pyre-ignore[21]
from core.agent import run_stage2  # pyre-ignore[21]
from core.scoring import apply_role_profile, compute_final_score, load_config  # pyre-ignore[21]
from core.entity_extractor import extract_entities  # pyre-ignore[21]
from core.skill_extractor import SkillExtractor  # pyre-ignore[21]
from core.experience_parser import parse_experience  # pyre-ignore[21]
from core.utils import extract_emails, extract_phones, extract_urls  # pyre-ignore[21]

logger = logging.getLogger(__name__)

# Module-level singleton extractor (constructed once, reused across calls)
_skill_extractor = SkillExtractor()

MAX_JOB_DESCRIPTION_CHARS = int(os.getenv("MAX_JOB_DESCRIPTION_CHARS", "20000"))
MAX_RESUME_TEXT_CHARS = int(os.getenv("MAX_RESUME_TEXT_CHARS", "120000"))
MAX_RESUMES_PER_REQUEST = int(os.getenv("MAX_RESUMES_PER_REQUEST", "200"))

# Blend of dense embedding similarity vs lexical keyword overlap (matches deployment).
_EMBED_WEIGHT = float(os.getenv("SEMANTIC_EMBED_WEIGHT", "0.82"))
_LEXICAL_WEIGHT = float(os.getenv("SEMANTIC_LEXICAL_WEIGHT", "0.18"))


def _to_plain_dict(obj: Any) -> Dict[str, Any]:
    """Serialize stage2 output (pydantic / dataclass / dict) to a plain dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict() if hasattr(obj, "dict") else {"value": obj}


def _normalize_resumes(resumes: List[Union[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Accept List[str] or List[{"id","resume_text"}] -> [{"id","resume_text"}]."""
    normalized: List[Dict[str, Any]] = []
    for i, r in enumerate(resumes):
        text = ""
        rid = str(i + 1)
        if isinstance(r, str):
            text = r.strip()
        elif isinstance(r, dict):
            text = str(r.get("resume_text", "")).strip()
            rid = str(r.get("id") or (i + 1))
        elif hasattr(r, "model_dump"):
            data = r.model_dump()
            text = str(data.get("resume_text", "")).strip()
            rid = str(data.get("id") or (i + 1))
        if not text:
            continue
        if len(text) > MAX_RESUME_TEXT_CHARS:
            raise ValueError(f"Resume {i + 1} exceeds max length ({MAX_RESUME_TEXT_CHARS} chars)")
        normalized.append({"id": rid, "resume_text": text})
    return normalized


def _blend_semantic(embedding_01: float, dense_ok: bool, job_description: str, resume_text: str) -> Dict[str, float]:
    """Blend a precomputed dense similarity with a per-resume lexical overlap signal."""
    try:
        lexical_01 = lexical_alignment_score(job_description, resume_text)
    except Exception:
        lexical_01 = 0.0
    if dense_ok:
        hybrid = max(0.0, min(_EMBED_WEIGHT * embedding_01 + _LEXICAL_WEIGHT * lexical_01, 1.0))
    else:
        hybrid = lexical_01
    return {
        "embedding_score_01": round(embedding_01, 4),
        "lexical_score_01": round(lexical_01, 4),
        "hybrid_score_01": round(hybrid, 4),
    }


def _resolve_jd_skills(job_description: str, parsed_jd: Dict[str, Any]) -> Dict[str, List[str]]:
    """Required/preferred skill lists from the LLM (canonicalized) or regex fallback."""
    llm_ok = parsed_jd.get("source") == "groq_llm"
    if llm_ok and (parsed_jd.get("required_skills") or parsed_jd.get("preferred_skills")):
        required: List[str] = []
        for s in parsed_jd.get("required_skills", []):
            canon = _skill_extractor.normalize_skill(s)
            if canon and canon not in required:
                required.append(canon)
        preferred: List[str] = []
        for s in parsed_jd.get("preferred_skills", []):
            canon = _skill_extractor.normalize_skill(s)
            if canon and canon not in preferred and canon not in required:
                preferred.append(canon)
        # Catch skills the LLM missed via regex over the JD.
        profile = _skill_extractor.extract_job_profile(job_description)
        for s in profile.detected:
            if s not in required and s not in preferred:
                preferred.append(s)
        return {"required": required, "preferred": preferred}

    profile = _skill_extractor.extract_job_profile(job_description)
    return {"required": list(profile.required), "preferred": list(profile.preferred)}


def run_pipeline(
    job_title: str,
    job_description: str,
    resumes: List[Union[str, Dict[str, Any]]],
    role_profile: str = "custom",
    model_version_id: Optional[str] = None,
    repo: Optional[TalentScoutRepo] = None,
) -> Dict[str, Any]:
    """
    End-to-end batch screening. Returns:
        {"session_id": <str|None>, "results": [candidate_dicts sorted by final_score]}
    """
    job_title = (job_title or "").strip()
    job_description = (job_description or "").strip()
    if not job_description:
        raise ValueError("job_description is required")
    if len(job_description) > MAX_JOB_DESCRIPTION_CHARS:
        raise ValueError(f"job_description exceeds max length ({MAX_JOB_DESCRIPTION_CHARS} chars)")
    if len(resumes) > MAX_RESUMES_PER_REQUEST:
        raise ValueError(f"Too many resumes ({len(resumes)}). Max {MAX_RESUMES_PER_REQUEST}")

    normalized_resumes = _normalize_resumes(resumes)
    if not normalized_resumes:
        raise ValueError("No valid resumes provided")

    # ---- DB (optional) ----
    if repo is None:
        try:
            repo = TalentScoutRepo()
        except Exception:
            repo = None

    # ---- scoring config + role profile ----
    cfg = load_config()
    cfg, profile_used = apply_role_profile(cfg, role_profile)

    # ---- parse the JD once: HyRe ideal-resume + skill lists + seniority ----
    parsed_jd = parse_jd_with_llm(job_title, job_description)
    llm_ok = parsed_jd.get("source") == "groq_llm"

    # Auto-pick a role profile from detected seniority when the caller left it "custom".
    if role_profile == "custom" and llm_ok and parsed_jd.get("seniority"):
        cfg, profile_used = apply_role_profile(load_config(), parsed_jd["seniority"])

    semantic_query = job_description
    if llm_ok and parsed_jd.get("ideal_resume_summary"):
        semantic_query = parsed_jd["ideal_resume_summary"]

    jd_skills = _resolve_jd_skills(job_description, parsed_jd)

    # ---- stage 1 dense ranking, computed once for the whole batch ----
    # Encode the HyRe query + all resumes in a single call, then map scores back by text.
    resume_texts = [r["resume_text"] for r in normalized_resumes]
    dense_by_text: Dict[str, float] = {}
    dense_ok = True
    try:
        for it in rank_resumes_stage1(semantic_query, resume_texts):
            dense_by_text[str(it.get("resume_text", ""))] = float(it.get("score", 0.0))
    except Exception as exc:  # embedding model unavailable -> lexical-only fallback
        logger.warning("Dense ranking unavailable, falling back to lexical: %s", exc)
        dense_ok = False

    # ---- create session (optional persistence) ----
    session_id: Optional[UUID] = None
    if repo is not None:
        try:
            session_id = repo.create_session(
                job_title=job_title or "(untitled)",
                job_description=job_description,
                role_profile=role_profile,
                scoring_config=cfg,
                model_version_id=None,
            )
        except Exception:
            session_id = None

    def _process_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
        resume_text = str(item.get("resume_text", "")).strip()
        try:
            candidate: Dict[str, Any] = {"id": item.get("id"), "resume_text": resume_text}

            # contacts
            emails, phones, urls = extract_emails(resume_text), extract_phones(resume_text), extract_urls(resume_text)
            candidate["contacts"] = {
                "email": emails[0] if emails else None,
                "phone": phones[0] if phones else None,
                "emails": emails, "phones": phones, "urls": urls,
            }

            # stage 1: semantic (dense + lexical) using the HyRe query
            signals = _blend_semantic(dense_by_text.get(resume_text, 0.0), dense_ok, job_description, resume_text)
            candidate["score"] = signals["hybrid_score_01"]
            candidate["semantic_signals"] = signals

            # skills (required + preferred)
            resume_skills = _skill_extractor.extract(resume_text)
            match = _skill_extractor.match_skills(
                jd_required=jd_skills["required"],
                jd_preferred=jd_skills["preferred"],
                candidate=resume_skills,
            )
            candidate["skills_match"] = {
                "found": list(dict.fromkeys([*match.present, *match.preferred_present])),
                "missing": list(match.missing),
                "match_rate": match.match_rate,
                "required": list(match.required),
                "preferred": list(match.preferred),
                "preferredFound": list(match.preferred_present),
                "preferredMissing": list(match.preferred_missing),
            }

            # experience
            exp = parse_experience(resume_text)
            candidate["experience"] = {
                "years": exp["years"], "months": exp["months"],
                "method": exp["method"], "confidence": exp["confidence"],
            }

            # final weighted + calibrated score
            candidate.update(compute_final_score(candidate, cfg))

            # stage 2 explanation (+ optional Groq polish)
            stage2 = _to_plain_dict(run_stage2(job_description, resume_text, candidate))
            candidate["aiSummary"] = stage2

            # entities (optional)
            with suppress(Exception):
                ents = extract_entities(resume_text)
                candidate["candidate_name"] = ents.get("candidate_name")
                candidate["meta"] = {
                    "recent_companies": ents.get("recent_companies", []),
                    "role_profile": role_profile,
                    "profile_used": profile_used,
                    "model_version_id": model_version_id,
                    "jd_source": parsed_jd.get("source"),
                }

            # persist (optional)
            if repo is not None and session_id is not None:
                with suppress(Exception):
                    repo.save_candidate(session_id=session_id, candidate_data=candidate)

            return candidate
        except Exception as e:  # one bad resume never aborts the run
            logger.warning("Candidate %s failed: %s", item.get("id"), e)
            return {
                "id": item.get("id"), "resume_text": resume_text,
                "score": 0.0, "final_score": 0.0,
                "breakdown": {"semantic": 0.0, "skills": 0.0, "experience": 0.0},
                "aiSummary": {"error": str(e)},
            }

    outputs: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.getenv("PIPELINE_WORKERS", "8"))) as ex:
        for fut in concurrent.futures.as_completed([ex.submit(_process_candidate, it) for it in normalized_resumes]):
            outputs.append(fut.result())

    outputs.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
    return {"session_id": str(session_id) if session_id is not None else None, "results": outputs}
