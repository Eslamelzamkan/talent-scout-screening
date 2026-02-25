-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =========================
-- USERS (admin/recruiter)
-- =========================
CREATE TABLE IF NOT EXISTS users (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email           TEXT UNIQUE NOT NULL,
  full_name       TEXT NOT NULL,
  role            TEXT NOT NULL CHECK (role IN ('admin','recruiter','viewer')),
  password_hash   TEXT NOT NULL,                -- store hashed password (bcrypt/argon2)
  is_active       BOOLEAN NOT NULL DEFAULT TRUE,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =========================
-- MODEL VERSIONS (baseline/finetuned)
-- =========================
CREATE TABLE IF NOT EXISTS model_versions (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name            TEXT NOT NULL,                -- e.g., "baseline-allminilm", "finetuned-v1"
  kind            TEXT NOT NULL CHECK (kind IN ('baseline','finetuned')),
  model_path      TEXT NOT NULL,                -- local path or HF repo
  sha256          TEXT,                         -- optional integrity
  training_config JSONB,                        -- store your experiment_config.json here
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_versions_kind ON model_versions(kind);

-- =========================
-- SCREENING SESSIONS
-- =========================
CREATE TABLE IF NOT EXISTS screening_sessions (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_by      UUID REFERENCES users(id) ON DELETE SET NULL,
  job_title       TEXT NOT NULL,
  job_description TEXT NOT NULL,
  role_profile    TEXT NOT NULL DEFAULT 'custom',  -- fresh_grad/junior/senior...
  scoring_config  JSONB NOT NULL DEFAULT '{}'::jsonb,
  model_version_id UUID REFERENCES model_versions(id) ON DELETE SET NULL,
  total_candidates INTEGER NOT NULL DEFAULT 0,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON screening_sessions(created_at DESC);

-- =========================
-- CANDIDATES (one resume scored inside a session)
-- =========================
CREATE TABLE IF NOT EXISTS candidates (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id      UUID NOT NULL REFERENCES screening_sessions(id) ON DELETE CASCADE,

  filename        TEXT,
  candidate_name  TEXT,                         -- extracted by NER (entity_extractor.py)
  resume_text     TEXT,                         -- store extracted text (optional but very useful)
  email           TEXT,
  phone           TEXT,

  -- scores
  semantic_score  REAL NOT NULL DEFAULT 0,       -- 0..1 or 0..100; pick one and be consistent
  skills_match_rate REAL NOT NULL DEFAULT 0,     -- 0..100
  experience_years REAL NOT NULL DEFAULT 0,      -- parsed years
  final_score     REAL NOT NULL DEFAULT 0,       -- 0..100

  status          TEXT NOT NULL DEFAULT 'unknown',  -- shortlist/reject/...
  summary         TEXT,

  -- richer fields (perfect for thesis + UI)
  pros            JSONB NOT NULL DEFAULT '[]'::jsonb,
  cons            JSONB NOT NULL DEFAULT '[]'::jsonb,
  interview_questions JSONB NOT NULL DEFAULT '[]'::jsonb,
  skills_found    JSONB NOT NULL DEFAULT '[]'::jsonb,
  skills_missing  JSONB NOT NULL DEFAULT '[]'::jsonb,
  breakdown       JSONB NOT NULL DEFAULT '{}'::jsonb, -- semantic/skills/experience breakdown
  explanation     JSONB NOT NULL DEFAULT '{}'::jsonb, -- deterministic explanation object
  meta            JSONB NOT NULL DEFAULT '{}'::jsonb, -- anything extra

  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_candidates_session_score ON candidates(session_id, final_score DESC);
CREATE INDEX IF NOT EXISTS idx_candidates_session_id ON candidates(session_id);
CREATE INDEX IF NOT EXISTS idx_candidates_final_score ON candidates(final_score DESC);

-- Keep session candidate count consistent (optional but nice)
CREATE OR REPLACE FUNCTION inc_session_candidate_count() RETURNS TRIGGER AS $$
BEGIN
  UPDATE screening_sessions
  SET total_candidates = total_candidates + 1
  WHERE id = NEW.session_id;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_inc_session_candidate_count ON candidates;
CREATE TRIGGER trg_inc_session_candidate_count
AFTER INSERT ON candidates
FOR EACH ROW EXECUTE FUNCTION inc_session_candidate_count();

-- =========================
-- BENCHMARK RUNS (baseline vs finetuned)
-- =========================
CREATE TABLE IF NOT EXISTS benchmark_runs (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_by       UUID REFERENCES users(id) ON DELETE SET NULL,
  dataset_name     TEXT NOT NULL,               -- e.g. netsol/resume-score-details
  baseline_model_id UUID REFERENCES model_versions(id) ON DELETE SET NULL,
  finetuned_model_id UUID REFERENCES model_versions(id) ON DELETE SET NULL,

  -- store your final df + perjd tables here
  summary_metrics  JSONB NOT NULL DEFAULT '{}'::jsonb,
  perjd_metrics    JSONB NOT NULL DEFAULT '[]'::jsonb,
  notes            TEXT,

  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_benchmark_created_at ON benchmark_runs(created_at DESC);

-- =========================
-- AUDIT EVENTS (professional touch)
-- =========================
CREATE TABLE IF NOT EXISTS audit_events (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  actor_id        UUID REFERENCES users(id) ON DELETE SET NULL,
  action          TEXT NOT NULL,                 -- e.g. "CREATE_SESSION", "SCORE_CANDIDATE"
  entity_type     TEXT,                          -- "session", "candidate", "benchmark"
  entity_id       UUID,
  details         JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_events(created_at DESC);
