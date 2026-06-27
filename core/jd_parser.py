"""
jd_parser.py — LLM-powered job description parser with in-memory caching.

Strategy (from the "one call per screening" plan):
  JD arrives → 1 Groq call → structured JSON → cached
  Resume 1..N → deterministic scoring against cached JSON → scores

Returns:
  required_skills, preferred_skills   — proper classification (Fix 1)
  seniority, years_min, years_max     — auto-detected role profile (Fix 3)
  ideal_resume_summary                — HyRe comparison target (Fix 2)

Uses llama-3.1-8b-instant (higher TPM limits, sufficient for JD parsing).
Falls back to empty result if Groq is unavailable — the caller should
then use the existing regex-based extraction as a fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from groq import Groq  # type: ignore
except ImportError:
    Groq = None  # type: ignore

# ---------------------------------------------------------------------------
# In-memory cache: hash(title+jd) → parsed result
# Survives for the lifetime of the worker process. One Groq call per unique
# JD, zero calls for subsequent resumes screened against the same JD.
# ---------------------------------------------------------------------------
_jd_cache: Dict[str, Dict[str, Any]] = {}

_groq_client: Any = None
_groq_initialized: bool = False
_groq_lock = threading.Lock()


def _get_groq_client() -> Any:
    """Lazy-init the Groq client (singleton)."""
    global _groq_client, _groq_initialized
    if _groq_initialized:
        return _groq_client
    with _groq_lock:
        if _groq_initialized:
            return _groq_client
        _groq_initialized = True
        api_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if api_key and Groq is not None:
            try:
                _groq_client = Groq(api_key=api_key)
            except Exception as exc:
                logger.warning("Groq client init failed: %s", str(exc)[:120])
    return _groq_client


def _cache_key(job_title: str, job_description: str) -> str:
    """Deterministic hash so the same JD always maps to the same cache slot."""
    raw = f"{job_title.strip().lower()}|||{job_description.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _empty_result(source: str = "none") -> Dict[str, Any]:
    """Fallback structure when LLM is unavailable."""
    return {
        "required_skills": [],
        "preferred_skills": [],
        "seniority": None,
        "years_min": None,
        "years_max": None,
        "ideal_resume_summary": None,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Prompt — single call extracts everything we need
# ---------------------------------------------------------------------------
_PARSE_PROMPT_TEMPLATE = """\
Parse this job description. Return ONLY valid JSON, no markdown, no explanation.

Schema:
{{
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill3", "skill4"],
  "seniority": "fresh_grad | junior | mid | senior | lead | manager",
  "years_min": <int or null>,
  "years_max": <int or null>,
  "ideal_resume_summary": "<2-3 sentence summary written AS IF it were the opening paragraph of the ideal candidate's resume — use first-person narrative style, mention key technologies and years of experience>"
}}

Rules:
- required_skills: technologies, tools, or domain knowledge that are explicitly required, strongly emphasized, or listed under requirements / must-have sections.
- preferred_skills: technologies or tools that are nice-to-have, bonus, or mentioned casually. If a skill appears alongside words like "preferred", "plus", "bonus", "ideally", or "nice to have", put it here.
- When in doubt, lean toward required.
- seniority: infer from the job title, required years of experience, and scope of responsibilities. Use exactly one of: fresh_grad, junior, mid, senior, lead, manager.
- ideal_resume_summary: write as a real resume opening paragraph, NOT as a job description. Use first person. Example: "Backend engineer with 4+ years of experience building scalable REST APIs in Python and Go. Strong background in PostgreSQL, Redis, and Docker-based microservice deployments."
- Keep skill names lowercase and concise (e.g., "python", "react", "aws", "docker").

Job Title: {job_title}
Job Description:
{job_description}
"""


def parse_jd_with_llm(
    job_title: str,
    job_description: str,
    *,
    max_jd_chars: int = 2000,
) -> Dict[str, Any]:
    """
    Parse a job description using Groq LLM.

    • One call per unique (title, description) pair — result is cached.
    • Uses llama-3.1-8b-instant for higher TPM limits on the free tier.
    • Returns structured dict; caller should check ``result["source"]``
      to know whether LLM parsing succeeded or fell back.
    """
    key = _cache_key(job_title, job_description)

    # Cache hit → zero API calls
    if key in _jd_cache:
        logger.debug("JD parser cache hit (key=%s…)", key[:8])
        return _jd_cache[key]

    client = _get_groq_client()
    if client is None:
        result = _empty_result("fallback_no_client")
        _jd_cache[key] = result
        return result

    prompt = _PARSE_PROMPT_TEMPLATE.format(
        job_title=job_title.strip(),
        job_description=job_description.strip()[:max_jd_chars],
    )

    try:
        model = os.getenv("GROQ_JD_PARSER_MODEL", "llama-3.1-8b-instant")

        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 500,
        }
        # response_format may not be supported on all models/SDK versions
        try:
            request_kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**request_kwargs)
        except TypeError:
            request_kwargs.pop("response_format", None)
            resp = client.chat.completions.create(**request_kwargs)

        content = ((resp.choices[0].message.content if resp and resp.choices else "") or "").strip()
        parsed = _extract_json(content)

        result = {
            "required_skills": _to_str_list(parsed.get("required_skills")),
            "preferred_skills": _to_str_list(parsed.get("preferred_skills")),
            "seniority": _clean_seniority(parsed.get("seniority")),
            "years_min": _safe_int(parsed.get("years_min")),
            "years_max": _safe_int(parsed.get("years_max")),
            "ideal_resume_summary": (parsed.get("ideal_resume_summary") or "").strip() or None,
            "source": "groq_llm",
        }

        _jd_cache[key] = result
        logger.info(
            "Parsed JD with LLM (%s): %d required, %d preferred skills, seniority=%s",
            model,
            len(result["required_skills"]),
            len(result["preferred_skills"]),
            result["seniority"],
        )
        return result

    except Exception as exc:
        logger.warning("Groq JD parsing failed, caller should use regex fallback: %s", str(exc)[:200])
        result = _empty_result(f"fallback_error:{str(exc)[:80]}")
        # Do NOT cache transient failures — the next call should retry the LLM
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_VALID_SENIORITY = {"fresh_grad", "junior", "mid", "senior", "lead", "manager"}


def _clean_seniority(raw: Any) -> Optional[str]:
    if not raw:
        return None
    s = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    return s if s in _VALID_SENIORITY else None


def _to_str_list(val: Any) -> list:
    if not val:
        return []
    if isinstance(val, (list, tuple)):
        return [str(v).strip().lower() for v in val if v and str(v).strip()]
    return []


def _safe_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from LLM output."""
    text = (text or "").strip()
    if not text:
        return {}

    # Try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown fences
    import re
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        try:
            obj = json.loads(fenced.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Try finding first { ... last }
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        try:
            obj = json.loads(text[first : last + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    logger.warning("Could not extract JSON from LLM response (len=%d)", len(text))
    return {}
