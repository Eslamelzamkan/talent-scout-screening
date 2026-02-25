# Screening/core/pipeline.py

from __future__ import annotations

import concurrent.futures
import os
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from db.db_postgres import TalentScoutRepo  # pyre-ignore[21]
from core.model import rank_resumes_stage1  # pyre-ignore[21]
from core.agent import run_stage2  # pyre-ignore[21]
from core.scoring import apply_role_profile, compute_final_score, load_config  # pyre-ignore[21]
from core.entity_extractor import extract_entities  # pyre-ignore[21]
from core.skill_extractor import SkillExtractor  # pyre-ignore[21]
from core.experience_parser import parse_experience  # pyre-ignore[21]
from core.utils import extract_emails, extract_phones, extract_urls  # pyre-ignore[21]

# Module-level singleton extractors (constructed once, reused across calls)
_skill_extractor = SkillExtractor()
MAX_JOB_DESCRIPTION_CHARS = int(os.getenv("MAX_JOB_DESCRIPTION_CHARS", "20000"))
MAX_RESUME_TEXT_CHARS = int(os.getenv("MAX_RESUME_TEXT_CHARS", "120000"))
MAX_RESUMES_PER_REQUEST = int(os.getenv("MAX_RESUMES_PER_REQUEST", "200"))


def _to_plain_dict(obj: Any) -> Dict[str, Any]:
    """Make sure we can serialize objects returned by stage2 (pydantic/dataclass/etc)."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    # Pydantic v1/v2 compatibility
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict() if hasattr(obj, "dict") else {"value": obj}


def _normalize_resumes(resumes: List[Union[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Accept resumes as:
      - List[str]
      - List[{"id": "...", "resume_text": "..."}]
    Returns normalized list: [{"id": "...", "resume_text": "..."}]
    """
    normalized: List[Dict[str, Any]] = []
    for i, r in enumerate(resumes):
        if isinstance(r, str):
            if (text := r.strip()):
                if len(text) > MAX_RESUME_TEXT_CHARS:
                    raise ValueError(
                        f"Resume {i + 1} exceeds max length ({MAX_RESUME_TEXT_CHARS} chars)"
                    )
                normalized.append({"id": str(i + 1), "resume_text": text})
        elif isinstance(r, dict):
            if (text := str(r.get("resume_text", "")).strip()):
                if len(text) > MAX_RESUME_TEXT_CHARS:
                    raise ValueError(
                        f"Resume {i + 1} exceeds max length ({MAX_RESUME_TEXT_CHARS} chars)"
                    )
                rid = str(r.get("id") or (i + 1))
                normalized.append({"id": rid, "resume_text": text})
        elif hasattr(r, "model_dump"):
            data = r.model_dump()
            if (text := str(data.get("resume_text", "")).strip()):
                if len(text) > MAX_RESUME_TEXT_CHARS:
                    raise ValueError(
                        f"Resume {i + 1} exceeds max length ({MAX_RESUME_TEXT_CHARS} chars)"
                    )
                rid = str(data.get("id") or (i + 1))
                normalized.append({"id": rid, "resume_text": text})
        else:
            # ignore unknown types
            continue
    return normalized


def run_pipeline(
    job_title: str,
    job_description: str,
    resumes: List[Union[str, Dict[str, Any]]],
    role_profile: str = "custom",
    model_version_id: Optional[str] = None,
    repo: Optional[TalentScoutRepo] = None,
) -> Dict[str, Any]:
    """
    End-to-end pipeline:
      - create session (optional DB)
      - stage1: rank resumes vs JD
      - skill extraction + experience parsing
      - stage2: structured explanation
      - scoring: apply weights + compute final score
      - persist candidates (optional DB)

    Returns:
      {"session_id": <str|None>, "results": [candidate_dicts...]}
    """

    # ---- basic validation ----
    job_title = (job_title or "").strip()
    job_description = (job_description or "").strip()
    if not job_description:
        raise ValueError("job_description is required")
    if len(job_description) > MAX_JOB_DESCRIPTION_CHARS:
        raise ValueError(
            f"job_description exceeds max length ({MAX_JOB_DESCRIPTION_CHARS} chars)"
        )
    if len(resumes) > MAX_RESUMES_PER_REQUEST:
        raise ValueError(
            f"Too many resumes ({len(resumes)}). Max {MAX_RESUMES_PER_REQUEST}"
        )

    normalized_resumes = _normalize_resumes(resumes)
    if not normalized_resumes:
        raise ValueError("No valid resumes provided")

    # ---- repo / persistence ----
    # If no repo is provided and DB is unavailable, run without persistence.
    if repo is None:
        try:
            repo = TalentScoutRepo()
        except Exception:
            repo = None  # Pipeline continues; session_id will stay None

    # ---- scoring config ----
    cfg = load_config()
    cfg, profile_used = apply_role_profile(cfg, role_profile)

    # ---- pre-extract JD skills once (reused for every candidate) ----
    jd_skills = _skill_extractor.extract(job_description)

    # ---- create session ----
    session_id: Optional[UUID] = None
    if repo is not None:
        try:
            session_id = repo.create_session(
                job_title=job_title or "(untitled)",
                job_description=job_description,
                role_profile=role_profile,
                scoring_config=cfg,
                model_version_id=None,  # UUID lookup not yet implemented; kept as None
            )
        except Exception:
            # If DB is down, we still want the pipeline to work.
            session_id = None

    # ---- stage1 ----
    resume_texts = [r["resume_text"] for r in normalized_resumes]
    stage1_results = rank_resumes_stage1(job_description, resume_texts)

    # stage1_results expected: list of dicts (must contain resume_text and score)
    # We'll map back IDs by matching text (safe for your UI use-case).
    text_to_id = {r["resume_text"]: r["id"] for r in normalized_resumes}

    def _process_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
        try:
            resume_text = str(item.get("resume_text", "")).strip()
            candidate_id = item.get("id") or text_to_id.get(resume_text)

            base = dict(item)
            if candidate_id is not None:
                base["id"] = str(candidate_id)

            # ---- contact extraction ----
            emails = extract_emails(resume_text)
            phones = extract_phones(resume_text)
            urls = extract_urls(resume_text)
            base["contacts"] = {
                "email": emails[0] if emails else None,
                "phone": phones[0] if phones else None,
                "emails": emails,
                "phones": phones,
                "urls": urls,
            }

            # ---- skill extraction ----
            resume_skills = _skill_extractor.extract(resume_text)
            skill_match = _skill_extractor.match_skills(
                jd_required=list(jd_skills.keys()),
                candidate=resume_skills,
            )
            base["skills_match"] = {
                "found": list(skill_match.present),
                "missing": list(skill_match.missing),
                "match_rate": skill_match.match_rate,
            }

            # ---- experience parsing ----
            exp_result = parse_experience(resume_text)
            base["experience"] = {
                "years": exp_result["years"],
                "months": exp_result["months"],
                "method": exp_result["method"],
                "confidence": exp_result["confidence"],
            }

            # ---- stage2 ----
            stage2_out = run_stage2(job_description, resume_text, base)
            stage2_dict = _to_plain_dict(stage2_out)

            # Ensure stage1 fields present in final
            stage2_dict.setdefault("resume_text", resume_text)
            stage2_dict.setdefault("id", base.get("id"))
            stage2_dict.setdefault("score", base.get("score", 0.0))  # stage1 semantic score 0..1

            # Carry forward extracted skills/experience so scoring can use them
            stage2_dict.setdefault("skills_match", base["skills_match"])
            stage2_dict.setdefault("experience", base["experience"])
            stage2_dict.setdefault("contacts", base["contacts"])

            # ---- NER Extractor ----
            entities = extract_entities(resume_text)
            stage2_dict["candidate_name"] = entities.get("candidate_name")

            # ---- final scoring ----
            scored = compute_final_score(stage2_dict, cfg)
            stage2_dict.update(scored)

            # Consolidate all meta in a single update (prevents accidental overwrite)
            meta = stage2_dict.get("meta") or {}
            meta.update(
                {
                    "recent_companies": entities.get("recent_companies", []),
                    "role_profile": role_profile,
                    "profile_used": profile_used,
                    "model_version_id": model_version_id,
                }
            )
            stage2_dict["meta"] = meta

            # ---- persist candidate ----
            if repo is not None and session_id is not None:
                with suppress(Exception):
                    repo.save_candidate(session_id=session_id, candidate_data=stage2_dict)

            return stage2_dict

        except Exception as e:
            # Robust failure isolation: one bad candidate doesn't kill the whole run
            return {
                "id": item.get("id"),
                "resume_text": item.get("resume_text", ""),
                "score": item.get("score", 0.0),
                "final_score": 0.0,
                "breakdown": {"semantic": 0.0, "skills": 0.0, "experience": 0.0},
                "stage2": {"error": str(e)},
            }

    # Execute Stage 2 and persistence in parallel
    outputs: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_process_candidate, item) for item in stage1_results]
        for future in concurrent.futures.as_completed(futures):
            outputs.append(future.result())

    # sort results by final_score desc (deterministic output)
    outputs.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)

    return {"session_id": str(session_id) if session_id is not None else None, "results": outputs}
