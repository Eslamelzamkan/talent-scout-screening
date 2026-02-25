import os
from typing import Annotated, Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, StringConstraints, field_validator  # pyre-ignore[21]

MAX_JOB_DESCRIPTION_CHARS = int(os.getenv("MAX_JOB_DESCRIPTION_CHARS", "20000"))
MAX_RESUME_TEXT_CHARS = int(os.getenv("MAX_RESUME_TEXT_CHARS", "120000"))
MAX_RESUMES_PER_REQUEST = int(os.getenv("MAX_RESUMES_PER_REQUEST", "200"))

JobTitle = Annotated[str, StringConstraints(max_length=200)]
JobDescription = Annotated[
    str, StringConstraints(min_length=1, max_length=MAX_JOB_DESCRIPTION_CHARS)
]
ResumeText = Annotated[
    str, StringConstraints(min_length=1, max_length=MAX_RESUME_TEXT_CHARS)
]


class ResumeItem(BaseModel):
    id: Optional[str] = Field(default=None, max_length=200)
    resume_text: ResumeText


# ---------- Request ----------

class RunRequest(BaseModel):
    """Matches what the Streamlit UI sends and what pipeline.run_pipeline expects."""

    job_title: JobTitle = ""
    job_description: JobDescription
    resumes: List[Union[ResumeText, ResumeItem]] = Field(
        min_length=1, max_length=MAX_RESUMES_PER_REQUEST
    )
    role_profile: str = Field(default="custom", max_length=64)
    model_version_id: Optional[str] = None  # UUID string; reserved for future model version tracking
    scoring_config: Optional[Dict[str, Any]] = None  # reserved for future use

    @field_validator("resumes")
    @classmethod
    def validate_non_empty_resumes(cls, value):
        cleaned = 0
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    cleaned += 1
            elif isinstance(item, ResumeItem):
                if item.resume_text.strip():
                    cleaned += 1
            elif isinstance(item, dict):
                if str(item.get("resume_text", "")).strip():
                    cleaned += 1

        if cleaned == 0:
            raise ValueError("At least one non-empty resume is required")
        return value


# ---------- Response ----------

class RunResponse(BaseModel):
    """Matches the dict returned by pipeline.run_pipeline."""

    session_id: Optional[str] = None
    results: List[Dict[str, Any]]


# ---------- Health ----------

class HealthResponse(BaseModel):
    status: str = "ok"
