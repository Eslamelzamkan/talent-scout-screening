import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://talent_admin:talent_password@127.0.0.1:5433/talent_scout",
)
os.environ.setdefault("JWT_SECRET_KEY", "test-only-secret")

from api.auth import get_current_user  # pyre-ignore[21]  # noqa: E402
from api.deps import get_repo  # pyre-ignore[21]  # noqa: E402
from api.main import app  # pyre-ignore[21]  # noqa: E402
from api import routes as routes_module  # pyre-ignore[21]  # noqa: E402


class FakeRepo:
    def get_all_sessions(self):
        return [
            {
                "id": "session-1",
                "job_title": "ML Engineer",
                "role_profile": "junior",
                "total_candidates": 2,
                "created_at": "2026-02-25T00:00:00+00:00",
                "model_version_id": None,
            }
        ]

    def get_session_candidates(self, session_id):
        return [{"id": "candidate-1", "session_id": str(session_id), "final_score": 90.0}]


@pytest.fixture
def client():
    app.dependency_overrides.clear()
    old_max_bytes = app.state.max_request_body_bytes
    with TestClient(app) as test_client:
        yield test_client
    app.state.max_request_body_bytes = old_max_bytes
    app.dependency_overrides.clear()


def test_sessions_requires_auth(client):
    resp = client.get("/api/v1/sessions?limit=10")
    assert resp.status_code == 401


def test_sessions_with_auth_returns_rows(client):
    app.dependency_overrides[get_current_user] = lambda: "user-1"
    app.dependency_overrides[get_repo] = lambda: FakeRepo()

    resp = client.get("/api/v1/sessions?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["job_title"] == "ML Engineer"


def test_run_with_auth_success(client, monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: "user-1"
    app.dependency_overrides[get_repo] = lambda: FakeRepo()

    def fake_run_pipeline(**kwargs):
        return {"session_id": "session-1", "results": [{"id": "r1", "final_score": 88.0}]}

    monkeypatch.setattr(routes_module, "run_pipeline", fake_run_pipeline)

    payload = {
        "job_title": "ML Engineer",
        "job_description": "Need Python and SQL",
        "role_profile": "junior",
        "resumes": [{"id": "r1", "resume_text": "Python engineer with SQL."}],
    }
    resp = client.post("/api/v1/run", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "session-1"
    assert body["results"][0]["id"] == "r1"


def test_run_rejects_oversized_body(client):
    app.dependency_overrides[get_current_user] = lambda: "user-1"
    app.dependency_overrides[get_repo] = lambda: FakeRepo()
    app.state.max_request_body_bytes = 300

    payload = {
        "job_title": "ML Engineer",
        "job_description": "Need Python",
        "resumes": [{"id": "r1", "resume_text": "A" * 500}],
    }

    resp = client.post("/api/v1/run", json=payload)
    assert resp.status_code == 413
