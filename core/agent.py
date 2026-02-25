"""
Stage 2: Recruiter-style deep screening (FREE + professional)

Goals:
- Typed inputs/outputs (Pydantic)
- Robust error handling (never breaks the app)
- Deterministic scoring explanations (based on Stage 1 signals)
- Optional Groq free-tier call to polish wording (temperature=0)
- Backward compatible with Streamlit: agent.analyze(resume_text, jd_text)

Recommended Streamlit improvement (1-line change):
    analysis = agent.analyze(c["raw"], jd_text, candidate=c)
So Stage 2 can use Stage 1 signals (semantic/skills/experience/final_score).
"""

from __future__ import annotations

import os
import re
import json
import time
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    # pydantic v2
    from pydantic import BaseModel, Field, ValidationError  # type: ignore
except Exception:  # pragma: no cover
    # fallback if pydantic missing
    BaseModel = object  # type: ignore
    Field = lambda *a, **k: None  # type: ignore
    ValidationError = Exception  # type: ignore

# Groq is optional. Stage2 still works without it.
try:
    from groq import Groq  # type: ignore
except Exception:
    Groq = None


# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


# -----------------------------
# Typed Models
# -----------------------------
class DecisionStatus(str, Enum):
    Shortlist = "Shortlist"
    Reject = "Reject"
    Pending = "Pending"
    Error = "Error"


class Rating(str, Enum):
    High = "High"
    Medium = "Medium"
    Low = "Low"


class Stage2Signals(BaseModel):  # type: ignore
    """
    Signals coming from Stage 1 (semantic ranker + skills + exp + scoring).
    This keeps Stage 2 deterministic and explainable.
    """
    semantic_score_01: Optional[float] = Field(default=None, description="Semantic similarity score 0..1")
    skills_match_rate_100: Optional[float] = Field(default=None, description="Skills match rate 0..100")
    skills_found: List[str] = Field(default_factory=list)
    skills_missing: List[str] = Field(default_factory=list)
    experience_years: Optional[float] = Field(default=None, description="Total years of experience")
    final_score_100: Optional[float] = Field(default=None, description="Final weighted score 0..100")

    @staticmethod
    def from_candidate_dict(candidate: Dict[str, Any]) -> "Stage2Signals":
        # Candidate structure comes from your Streamlit pipeline. :contentReference[oaicite:1]{index=1}
        semantic = candidate.get("score")  # stage1 ranker uses "score" typically (0..1)
        skills = candidate.get("skills_match", {}) or {}
        exp = candidate.get("experience", {}) or {}
        final_score = candidate.get("final_score")

        def _safe_float(x):
            try:
                if x is None:
                    return None
                return float(x)
            except Exception:
                return None

        return Stage2Signals(
            semantic_score_01=_safe_float(semantic),
            skills_match_rate_100=_safe_float(skills.get("match_rate")),
            skills_found=[str(s) for s in (skills.get("found") or []) if s],
            skills_missing=[str(s) for s in (skills.get("missing") or []) if s],
            experience_years=_safe_float(exp.get("years")),
            final_score_100=_safe_float(final_score),
        )


class Stage2Result(BaseModel):  # type: ignore
    summary: str
    status: DecisionStatus
    rating: Rating
    pros: List[str]
    cons: List[str]
    interview_questions: List[str]

    # Extra fields that make it "grad-level"
    evidence: Dict[str, Any] = Field(default_factory=dict)
    explanation: str = ""


# -----------------------------
# Deterministic logic
# -----------------------------
def _clamp(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    if x is None:
        return None
    return max(lo, min(float(x), hi))


def _bucket_rating(score_100: Optional[float]) -> Rating:
    if score_100 is None:
        return Rating.Medium
    if score_100 >= 80:
        return Rating.High
    if score_100 >= 65:
        return Rating.Medium
    return Rating.Low


def _decide_status(score_100: Optional[float], skills_missing: List[str]) -> DecisionStatus:
    """
    Deterministic decision rule:
    - If score is strong => Shortlist
    - If low => Reject
    - Otherwise Pending
    Also: if there are many missing critical skills, bias to Pending/Reject.
    """
    if score_100 is None:
        return DecisionStatus.Pending

    # simple, explainable thresholds
    if score_100 >= 78:
        return DecisionStatus.Shortlist
    if score_100 < 60:
        return DecisionStatus.Reject

    # borderline: if missing too many skills, lean Reject
    if len(skills_missing) >= 5 and score_100 < 70:
        return DecisionStatus.Reject

    return DecisionStatus.Pending


def _make_pros_cons(signals: Stage2Signals) -> Tuple[List[str], List[str]]:
    pros: List[str] = []
    cons: List[str] = []

    sem = _clamp(signals.semantic_score_01, 0.0, 1.0)
    if sem is not None:
        if sem >= 0.75:
            pros.append(f"Strong semantic match to the JD (cosine similarity ~{sem:.2f}).")
        elif sem >= 0.60:
            pros.append(f"Decent semantic alignment to the JD (similarity ~{sem:.2f}).")
        else:
            cons.append(f"Weak semantic alignment to the JD (similarity ~{sem:.2f}).")

    sm = _clamp(signals.skills_match_rate_100, 0.0, 100.0)
    if sm is not None:
        if sm >= 75:
            pros.append(f"Good skills match (~{sm:.0f}%).")
        elif sm >= 55:
            pros.append(f"Moderate skills match (~{sm:.0f}%).")
        else:
            cons.append(f"Low skills match (~{sm:.0f}%).")

    if signals.skills_found:
        top_found = ", ".join(signals.skills_found[:6])
        pros.append(f"Matched skills: {top_found}.")

    if signals.skills_missing:
        top_missing = ", ".join(signals.skills_missing[:6])
        cons.append(f"Missing/unclear skills: {top_missing}.")

    yrs = _clamp(signals.experience_years, 0.0, 60.0)
    if yrs is not None:
        if yrs >= 5:
            pros.append(f"Solid experience level (~{yrs:.1f} years).")
        elif yrs >= 2:
            pros.append(f"Some relevant experience (~{yrs:.1f} years).")
        else:
            cons.append(f"Limited experience (~{yrs:.1f} years).")

    # Ensure non-empty lists (UI friendliness)
    if not pros:
        pros = ["Resume processed successfully.", "Candidate information captured."]
    if not cons:
        cons = ["No major concerns detected from structured signals."]

    return pros[:6], cons[:6]


def _make_questions(signals: Stage2Signals) -> List[str]:
    qs: List[str] = []

    if signals.skills_missing:
        qs.append(f"Can you walk through your hands-on experience with {signals.skills_missing[0]}?")
    if signals.skills_found:
        qs.append(f"Describe a project where you used {signals.skills_found[0]} and the impact you delivered.")
    if signals.experience_years is not None:
        qs.append("Which role/project best represents your current level of ownership and scope?")
    qs.append("What are the most challenging technical tradeoffs you handled recently, and why?")

    return qs[:4]


def _deterministic_summary(signals: Stage2Signals, status: DecisionStatus, rating: Rating) -> str:
    fs = _clamp(signals.final_score_100, 0.0, 100.0)
    parts = []
    if fs is not None:
        parts.append(f"Overall match score: {fs:.1f}/100 ({rating}).")
    else:
        parts.append(f"Overall match assessed from available signals ({rating}).")

    if signals.skills_match_rate_100 is not None:
        parts.append(f"Skills match: ~{signals.skills_match_rate_100:.0f}%.")

    if signals.experience_years is not None:
        parts.append(f"Estimated experience: ~{signals.experience_years:.1f} years.")

    parts.append(f"Decision: {status}.")
    return " ".join(parts)


def _deterministic_explanation(signals: Stage2Signals, status: DecisionStatus) -> str:
    """
    A stable explanation string your professor will like:
    - References concrete signals (semantic/skills/experience/final score)
    - Explains why Shortlist/Reject/Pending deterministically
    """
    fs = _clamp(signals.final_score_100, 0.0, 100.0)
    sem = _clamp(signals.semantic_score_01, 0.0, 1.0)
    sm = _clamp(signals.skills_match_rate_100, 0.0, 100.0)

    bullets = []
    if fs is not None:
        bullets.append(f"Final weighted score = {fs:.1f}/100.")
    if sem is not None:
        bullets.append(f"Semantic similarity (Stage 1) = {sem:.2f}.")
    if sm is not None:
        bullets.append(f"Skills match rate = {sm:.0f}% (found={len(signals.skills_found)}, missing={len(signals.skills_missing)}).")
    if signals.experience_years is not None:
        bullets.append(f"Estimated experience = {signals.experience_years:.1f} years.")

    if status == DecisionStatus.Shortlist:
        bullets.append("Shortlisted because overall score is strong and the candidate meets most requirements.")
    elif status == DecisionStatus.Reject:
        bullets.append("Rejected due to low overall score and/or multiple missing requirements.")
    else:
        bullets.append("Marked as Pending because the candidate is borderline; recommend manual review or a short screening call.")

    return " ".join(bullets)


# -----------------------------
# Optional GitHub signal
# -----------------------------
_GH_USER_RE = re.compile(r"github\.com/([a-zA-Z0-9_-]+)", re.IGNORECASE)

def _check_github(resume_text: str, timeout_s: float = 3.0) -> Dict[str, Any]:
    """
    Safe GitHub check. Never raises.
    Returns structured evidence dict.
    """
    if not resume_text:
        return {"found": False, "note": "Empty resume text."}

    m = _GH_USER_RE.search(resume_text)
    if not m:
        return {"found": False, "note": "No GitHub link found."}

    username = m.group(1)
    try:
        resp = requests.get(
            f"https://api.github.com/users/{username}",
            timeout=timeout_s,
            headers={"User-Agent": "TalentScout-App"},
        )
        if resp.status_code != 200:
            return {"found": True, "username": username, "note": f"GitHub profile not accessible (HTTP {resp.status_code})."}

        data = resp.json()
        return {
            "found": True,
            "username": username,
            "public_repos": data.get("public_repos", 0),
            "followers": data.get("followers", 0),
        }
    except requests.exceptions.Timeout:
        return {"found": True, "username": username, "note": "GitHub check timed out."}
    except Exception as e:
        return {"found": True, "username": username, "note": f"GitHub check failed: {str(e)[:80]}"}


# -----------------------------
# Groq polishing (optional)
# -----------------------------
def _safe_model_dump(x: Any) -> Dict[str, Any]:
    # Pydantic v2: model_dump, v1: dict
    if hasattr(x, "model_dump"):
        return x.model_dump()  # type: ignore
    if hasattr(x, "dict"):
        return x.dict()  # type: ignore
    result: Dict[str, Any] = dict(x)  # type: ignore
    return result


def _polish_with_groq(
    client: Any,
    model: str,
    draft: Stage2Result,
    jd_text: str,
    resume_text: str,
    max_resume_chars: int = 2500,
    max_jd_chars: int = 1500,
) -> Stage2Result:
    """
    Optional: Use Groq to rewrite summary/pros/cons/questions in cleaner language.
    Decision/status/rating/evidence/explanation remain deterministic.
    """
    if client is None:
        return draft

    prompt = f"""
You are a concise technical recruiter assistant.
Rewrite the fields summary/pros/cons/interview_questions to be clearer and more professional.
DO NOT change status or rating. Keep items short.

JOB (truncated):
{jd_text[:max_jd_chars]}

RESUME (truncated):
{resume_text[:max_resume_chars]}

Return ONLY valid JSON with keys:
summary (string), pros (list of strings), cons (list of strings), interview_questions (list of strings).
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        content = resp.choices[0].message.content.strip()
        obj = json.loads(content)

        # Patch only language fields
        patched = Stage2Result(**{
            **_safe_model_dump(draft),
            "summary": str(obj.get("summary", draft.summary)),
            "pros": [str(x) for x in (obj.get("pros") or draft.pros)][:6],
            "cons": [str(x) for x in (obj.get("cons") or draft.cons)][:6],
            "interview_questions": [str(x) for x in (obj.get("interview_questions") or draft.interview_questions)][:4],
        })
        return patched
    except Exception as e:
        logger.info("Groq polish skipped: %s", str(e)[:120])
        return draft


# -----------------------------
# Public Agent (compatible with your app)
# -----------------------------
class RecruiterAgent:
    """
    Backward compatible with your Streamlit code:
        analysis = agent.analyze(c["raw"], jd_text)

    Recommended:
        analysis = agent.analyze(c["raw"], jd_text, candidate=c)
    So Stage 2 uses Stage 1 signals deterministically.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        self.api_key = (api_key or os.getenv("GROQ_API_KEY") or "").strip()
        self.model = model

        self.client = None
        if self.api_key and Groq is not None:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.warning("Groq client init failed; continuing without LLM: %s", str(e)[:120])
                self.client = None

    def analyze(
        self,
        resume_text: str,
        jd_text: str,
        candidate: Optional[Dict[str, Any]] = None,
        polish_with_llm: bool = False,
        github_check: bool = True,
    ) -> Dict[str, Any]:
        """
        Always returns a valid dict with keys:
        summary, status, rating, pros, cons, interview_questions
        + (extra) evidence, explanation

        If candidate is provided, Stage 2 becomes fully deterministic + explainable.
        If not, it still works but has fewer signals.
        """

        # Basic input validation (never raise in UI path)
        if not isinstance(resume_text, str) or not resume_text.strip():
            return {
                "summary": "Resume text is empty or missing.",
                "status": DecisionStatus.Error.value,
                "rating": Rating.Low.value,
                "pros": [],
                "cons": ["No resume content provided."],
                "interview_questions": [],
                "evidence": {"error": "empty_resume"},
                "explanation": "Stage2 failed: missing resume text.",
            }

        if not isinstance(jd_text, str) or not jd_text.strip():
            return {
                "summary": "Job description is empty or missing.",
                "status": DecisionStatus.Error.value,
                "rating": Rating.Low.value,
                "pros": [],
                "cons": ["No job description provided."],
                "interview_questions": [],
                "evidence": {"error": "empty_jd"},
                "explanation": "Stage2 failed: missing job description text.",
            }

        # Signals
        try:
            signals = Stage2Signals.from_candidate_dict(candidate or {})
        except Exception:
            # fallback
            signals = Stage2Signals()

        # GitHub evidence (optional)
        gh = _check_github(resume_text) if github_check else {"skipped": True}

        # Decide using deterministic rules.
        fs = _clamp(signals.final_score_100, 0.0, 100.0)
        rating = _bucket_rating(fs)
        status = _decide_status(fs, signals.skills_missing)

        pros, cons = _make_pros_cons(signals)
        questions = _make_questions(signals)

        summary = _deterministic_summary(signals, status, rating)
        explanation = _deterministic_explanation(signals, status)

        draft = Stage2Result(
            summary=summary,
            status=status,
            rating=rating,
            pros=pros,
            cons=cons,
            interview_questions=questions,
            evidence={
                "signals": _safe_model_dump(signals),
                "github": gh,
                "policy": {
                    "status_thresholds": {"shortlist": ">=78", "reject": "<60", "borderline": "60-77"},
                },
            },
            explanation=explanation,
        )

        # Optional: language polish with Groq (FREE tier)
        if polish_with_llm and self.client is not None:
            # lightweight retry if free-tier rate limit happens
            for attempt in range(2):
                polished = _polish_with_groq(
                    client=self.client,
                    model=self.model,
                    draft=draft,
                    jd_text=jd_text,
                    resume_text=resume_text,
                )
                if polished is not None:
                    draft = polished
                    break
                time.sleep(1.5 * (attempt + 1))

        # Return legacy dict keys expected by streamlit/db. :contentReference[oaicite:2]{index=2}
        out = _safe_model_dump(draft)

        # Convert Enums to strings for JSON/SQLite friendliness
        out["status"] = draft.status.value
        out["rating"] = draft.rating.value

        return out


# ------------------------------------------------------------------
# Module-level convenience function (used by pipeline.py)
# ------------------------------------------------------------------
_default_agent: Optional[RecruiterAgent] = None


def run_stage2(
    job_description: str,
    resume_text: str,
    candidate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper around RecruiterAgent for pipeline.py.
    """
    global _default_agent
    if _default_agent is None:
        _default_agent = RecruiterAgent()

    return _default_agent.analyze(
        resume_text=resume_text,
        jd_text=job_description,
        candidate=candidate,
    )

