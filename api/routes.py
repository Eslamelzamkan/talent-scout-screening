import os

from fastapi import APIRouter, Depends, HTTPException, Request, status  # pyre-ignore[21]
from pydantic import BaseModel  # pyre-ignore[21]
from sqlalchemy import select  # pyre-ignore[21]
from sqlalchemy.orm import Session as DBSession  # pyre-ignore[21]
from uuid import UUID

from api.schemas import RunRequest, RunResponse  # pyre-ignore[21]
from api.deps import get_repo  # pyre-ignore[21]
from api.auth import (  # pyre-ignore[21]
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)
from core.pipeline import run_pipeline  # pyre-ignore[21]
from db.db_postgres import User, get_engine  # pyre-ignore[21]
from api.rate_limit import limiter  # pyre-ignore[21]

router = APIRouter(prefix="/v1", tags=["talent-scout"])
RUN_RATE_LIMIT = os.getenv("RUN_RATE_LIMIT", "10/minute")


# ---------- Auth schemas ----------

class RegisterRequest(BaseModel):
    email: str
    full_name: str
    password: str
    role: str = "recruiter"


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---------- Auth endpoints ----------

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest):
    with DBSession(get_engine()) as db:
        if (existing := db.execute(select(User).where(User.email == req.email)).scalar_one_or_none()):
            raise HTTPException(status_code=400, detail="Email already registered")

        user = User(
            email=req.email,
            full_name=req.full_name,
            role=req.role,
            password_hash=hash_password(req.password),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    return TokenResponse(access_token=create_access_token(user.id))  # pyre-ignore[28]


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    with DBSession(get_engine()) as db:
        user = db.execute(select(User).where(User.email == req.email)).scalar_one_or_none()

    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    token = create_access_token(user.id)
    return TokenResponse(access_token=token)  # pyre-ignore[28]


@router.post("/run", response_model=RunResponse)
@limiter.limit(RUN_RATE_LIMIT)
def run(
    request: Request,  # noqa: ARG001
    req: RunRequest,
    user_id: str = Depends(get_current_user),  # noqa: ARG001
    repo=Depends(get_repo),
):
    return run_pipeline(
        job_title=req.job_title,
        job_description=req.job_description,
        resumes=req.resumes,
        role_profile=req.role_profile,
        model_version_id=req.model_version_id,
        repo=repo,
    )


@router.get("/sessions")
def list_sessions(
    limit: int = 50,
    user_id: str = Depends(get_current_user),  # noqa: ARG001
    repo=Depends(get_repo),
):
    """
    Return previous screening sessions for history views in the UI.
    """
    if repo is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    lim = max(1, min(int(limit), 500))
    sessions = repo.get_all_sessions()
    return {"sessions": sessions[:lim]}


@router.get("/sessions/{session_id}")
def session_candidates(
    session_id: UUID,
    user_id: str = Depends(get_current_user),  # noqa: ARG001
    repo=Depends(get_repo),
):
    """
    Return candidates for a previous screening session.
    """
    if repo is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    candidates = repo.get_session_candidates(session_id)
    return {"session_id": str(session_id), "candidates": candidates}
