"""
Smoke tests for core/pipeline.py
"""
import os
import sys
from uuid import UUID

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import pipeline  # pyre-ignore[21]


class FakeRepo:
    def __init__(self):
        self.created_sessions = []
        self.saved_candidates = []

    def create_session(self, **kwargs):
        sid = UUID("11111111-1111-1111-1111-111111111111")
        self.created_sessions.append(kwargs)
        return sid

    def save_candidate(self, session_id, candidate_data):
        self.saved_candidates.append((session_id, candidate_data))
        return UUID("22222222-2222-2222-2222-222222222222")


def test_run_pipeline_smoke_with_repo(monkeypatch):
    def fake_stage1(job_description, resume_texts):
        # Intentionally unsorted to verify final_score sorting.
        return [
            {"resume_text": resume_texts[0], "score": 0.2},
            {"resume_text": resume_texts[1], "score": 0.8},
        ]

    def fake_stage2(job_description, resume_text, candidate):
        return {
            "summary": "stage2 ok",
            "status": "Pending",
            "pros": [],
            "cons": [],
            "interview_questions": [],
            "evidence": {},
            "explanation": "deterministic",
        }

    def fake_compute(candidate, cfg):
        score = float(candidate.get("score", 0.0)) * 100.0
        return {
            "final_score": round(score, 2),
            "breakdown": {"semantic": round(score, 2), "skills": 0.0, "experience": 0.0},
        }

    monkeypatch.setattr(pipeline, "rank_resumes_stage1", fake_stage1)
    monkeypatch.setattr(pipeline, "run_stage2", fake_stage2)
    monkeypatch.setattr(pipeline, "compute_final_score", fake_compute)
    monkeypatch.setattr(pipeline, "load_config", lambda: {"semantic_weight": 1.0})
    monkeypatch.setattr(pipeline, "apply_role_profile", lambda cfg, role: (cfg, None))
    monkeypatch.setattr(
        pipeline,
        "parse_experience",
        lambda text: {"years": 2.0, "months": 24, "method": "stub", "confidence": 1.0},
    )
    monkeypatch.setattr(
        pipeline,
        "extract_entities",
        lambda text: {"candidate_name": "Candidate", "recent_companies": ["Acme"]},
    )

    repo = FakeRepo()
    result = pipeline.run_pipeline(
        job_title="ML Engineer",
        job_description="Need Python and SQL",
        resumes=[
            {"id": "r1", "resume_text": "resume one john@example.com"},
            {"id": "r2", "resume_text": "resume two jane@example.com"},
        ],
        repo=repo,
    )

    assert result["session_id"] == "11111111-1111-1111-1111-111111111111"
    assert len(result["results"]) == 2
    assert result["results"][0]["id"] == "r2"  # higher stage1 score -> higher final_score
    assert result["results"][0]["final_score"] == 80.0
    assert isinstance(result["results"][0].get("contacts"), dict)
    assert result["results"][0]["contacts"].get("email") is not None
    assert len(repo.saved_candidates) == 2

