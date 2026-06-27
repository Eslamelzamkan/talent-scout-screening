# pyre-ignore-all-errors
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from dotenv import load_dotenv  # type: ignore
from sqlalchemy import (  # type: ignore
    JSON, Boolean, CheckConstraint, Float, ForeignKey, Index, Integer, String, Text,
    TIMESTAMP, create_engine, func, select
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID  # type: ignore
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session  # type: ignore

load_dotenv()

# Optional: the engine/pipeline runs fine without a database (persistence is skipped).
# We only require DATABASE_URL when a DB connection is actually requested (get_engine()).
DATABASE_URL = os.getenv("DATABASE_URL")

# -----------------------------
# Base
# -----------------------------
class Base(DeclarativeBase):
    pass

# -----------------------------
# Models
# -----------------------------
class User(Base):
    __tablename__ = "users"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    model_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    training_config: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        CheckConstraint("kind IN ('baseline','finetuned')", name="chk_model_kind"),
    )

class ScreeningSession(Base):
    __tablename__ = "screening_sessions"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)

    created_by: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    job_title: Mapped[str] = mapped_column(Text, nullable=False)
    job_description: Mapped[str] = mapped_column(Text, nullable=False)
    role_profile: Mapped[str] = mapped_column(String, default="custom", nullable=False)
    scoring_config: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    model_version_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("model_versions.id", ondelete="SET NULL"))
    total_candidates: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    candidates: Mapped[List["Candidate"]] = relationship(back_populates="session", cascade="all, delete-orphan")

class Candidate(Base):
    __tablename__ = "candidates"
    __table_args__ = (
        Index("idx_candidates_session_id", "session_id"),
        Index("idx_candidates_final_score", "final_score"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("screening_sessions.id", ondelete="CASCADE"), nullable=False)

    filename: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    candidate_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    resume_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    email: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    semantic_score: Mapped[float] = mapped_column(Float, default=0, nullable=False)
    skills_match_rate: Mapped[float] = mapped_column(Float, default=0, nullable=False)
    experience_years: Mapped[float] = mapped_column(Float, default=0, nullable=False)
    final_score: Mapped[float] = mapped_column(Float, default=0, nullable=False)

    status: Mapped[str] = mapped_column(String, default="unknown", nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    pros: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    cons: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    interview_questions: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)

    skills_found: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    skills_missing: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    breakdown: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    explanation: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    meta: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    session: Mapped["ScreeningSession"] = relationship(back_populates="candidates")


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    created_by: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))

    dataset_name: Mapped[str] = mapped_column(Text, nullable=False)
    baseline_model_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("model_versions.id", ondelete="SET NULL"))
    finetuned_model_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("model_versions.id", ondelete="SET NULL"))

    summary_metrics: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    perjd_metrics: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    actor_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(String, nullable=False)
    entity_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    entity_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    details: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


# -----------------------------
# Engine + init
# -----------------------------
_engine = None


def get_engine():
    """Lazy engine factory — only connects when first called."""
    global _engine
    if _engine is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL environment variable is required for database features")
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


# Keep a module-level alias so existing imports of `engine` still work.
# This is a property-like accessor; callers that do `from db.db_postgres import engine`
# will get the lazy instance on first access via TalentScoutRepo or init_db().
engine = None  # will be set on first get_engine() call


def init_db() -> None:
    """Create tables if you aren't using Alembic yet."""
    Base.metadata.create_all(get_engine())

# -----------------------------
# Repository (similar to your ResumeDatabase)
# -----------------------------
class TalentScoutRepo:
    def __init__(self):
        init_db()

    def create_session(
        self,
        job_title: str,
        job_description: str,
        created_by: Optional[UUID] = None,
        role_profile: str = "custom",
        scoring_config: Optional[Dict[str, Any]] = None,
        model_version_id: Optional[UUID] = None,
    ) -> UUID:
        with Session(get_engine()) as db:
            sess = ScreeningSession(  # type: ignore
                job_title=job_title,
                job_description=job_description,
                created_by=created_by,
                role_profile=role_profile,
                scoring_config=scoring_config or {},
                model_version_id=model_version_id,
            )
            db.add(sess)
            db.commit()
            db.refresh(sess)

            db.add(AuditEvent(  # pyre-ignore
                actor_id=created_by,
                action="CREATE_SESSION",
                entity_type="session",
                entity_id=sess.id,
                details={"job_title": job_title, "role_profile": role_profile},
            ))
            db.commit()
            return sess.id

    def save_candidate(
        self,
        session_id: UUID,
        candidate_data: Dict[str, Any],
        actor_id: Optional[UUID] = None,
    ) -> UUID:
        """candidate_data can be your stage1/stage2 merged output."""
        def _get(d, path, default=None):
            cur = d
            for k in path:
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur

        # explanation may be a str (from agent.py) or dict; normalize to dict
        raw_explanation = candidate_data.get("explanation", {})
        if isinstance(raw_explanation, str):
            raw_explanation = {"text": raw_explanation}
        elif not isinstance(raw_explanation, dict):
            raw_explanation = {}

        cand = Candidate(  # type: ignore
            session_id=session_id,
            filename=candidate_data.get("filename"),
            candidate_name=candidate_data.get("candidate_name"),
            resume_text=candidate_data.get("resume_text"),
            email=_get(candidate_data, ["contacts", "email"]),
            phone=_get(candidate_data, ["contacts", "phone"]),

            semantic_score=float(candidate_data.get("score", 0.0)),
            skills_match_rate=float(_get(candidate_data, ["skills_match", "match_rate"], 0.0)),
            experience_years=float(_get(candidate_data, ["experience", "years"], 0.0)),
            final_score=float(candidate_data.get("final_score", 0.0)),

            status=str(candidate_data.get("status", "unknown")),
            summary=candidate_data.get("summary"),

            pros=candidate_data.get("pros", []) or [],
            cons=candidate_data.get("cons", []) or [],
            interview_questions=candidate_data.get("interview_questions", []) or [],

            skills_found=_get(candidate_data, ["skills_match", "found"], []) or [],
            skills_missing=_get(candidate_data, ["skills_match", "missing"], []) or [],
            breakdown=candidate_data.get("breakdown", {}) or {},
            explanation=raw_explanation,
            meta=candidate_data.get("meta", {}) or {},
        )

        with Session(get_engine()) as db:
            db.add(cand)

            # keep session counter updated
            db.execute(
                select(ScreeningSession).where(ScreeningSession.id == session_id).with_for_update()
            )
            sess = db.get(ScreeningSession, session_id)
            if sess:
                sess.total_candidates += 1

            db.commit()
            db.refresh(cand)

            db.add(AuditEvent(  # pyre-ignore
                actor_id=actor_id,
                action="SAVE_CANDIDATE",
                entity_type="candidate",
                entity_id=cand.id,
                details={"session_id": str(session_id), "final_score": cand.final_score},
            ))
            db.commit()
            return cand.id

    def get_session_candidates(self, session_id: UUID) -> List[Dict[str, Any]]:
        with Session(get_engine()) as db:
            stmt = select(Candidate).where(Candidate.session_id == session_id).order_by(Candidate.final_score.desc())
            rows = db.scalars(stmt).all()
            return [self._candidate_to_dict(r) for r in rows]

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        with Session(get_engine()) as db:
            stmt = select(ScreeningSession).order_by(ScreeningSession.created_at.desc())
            rows = db.scalars(stmt).all()
            return [{
                "id": str(r.id),
                "job_title": r.job_title,
                "role_profile": r.role_profile,
                "total_candidates": r.total_candidates,
                "created_at": r.created_at.isoformat(),
                "model_version_id": str(r.model_version_id) if r.model_version_id else None,
            } for r in rows]

    def save_benchmark_run(
        self,
        dataset_name: str,
        baseline_model_id: Optional[UUID],
        finetuned_model_id: Optional[UUID],
        summary_metrics: Dict[str, Any],
        perjd_metrics: List[Dict[str, Any]],
        created_by: Optional[UUID] = None,
        notes: Optional[str] = None,
    ) -> UUID:
        with Session(get_engine()) as db:
            run = BenchmarkRun(  # type: ignore
                created_by=created_by,
                dataset_name=dataset_name,
                baseline_model_id=baseline_model_id,
                finetuned_model_id=finetuned_model_id,
                summary_metrics=summary_metrics,
                perjd_metrics=perjd_metrics,
                notes=notes,
            )
            db.add(run)
            db.commit()
            db.refresh(run)

            db.add(AuditEvent(
                actor_id=created_by,
                action="SAVE_BENCHMARK",
                entity_type="benchmark",
                entity_id=run.id,
                details={"dataset": dataset_name},
            ))
            db.commit()
            return run.id

    @staticmethod
    def _candidate_to_dict(c: Candidate) -> Dict[str, Any]:
        return {
            "id": str(c.id),
            "session_id": str(c.session_id),
            "filename": c.filename,
            "candidate_name": c.candidate_name,
            "email": c.email,
            "phone": c.phone,
            "semantic_score": c.semantic_score,
            "skills_match_rate": c.skills_match_rate,
            "experience_years": c.experience_years,
            "final_score": c.final_score,
            "status": c.status,
            "summary": c.summary,
            "pros": c.pros,
            "cons": c.cons,
            "interview_questions": c.interview_questions,
            "skills_found": c.skills_found,
            "skills_missing": c.skills_missing,
            "breakdown": c.breakdown,
            "explanation": c.explanation,
            "meta": c.meta,
            "created_at": c.created_at.isoformat(),
        }
