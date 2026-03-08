-- ============================================================
-- Project DNA: Database Schema v1.0
-- PostgreSQL 15 — Core tables for RAG pipeline
-- Created: 2026-03-08
--
-- Apply: docker exec -i ai-postgres psql -U igorvl -d project_dna -f /tmp/schema.sql
-- Or:    docker exec -i ai-postgres psql -U igorvl -d project_dna < db/migrations/001_initial_schema.sql
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS projects (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255) NOT NULL,
    slug            VARCHAR(100) NOT NULL UNIQUE,
    status          VARCHAR(20) NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'archived', 'paused')),
    dna_document    TEXT NOT NULL DEFAULT '',
    style_matrix    JSONB NOT NULL DEFAULT '{}',
    tags            TEXT[] DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);
CREATE INDEX IF NOT EXISTS idx_projects_slug ON projects(slug);

CREATE TABLE IF NOT EXISTS accounts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255) NOT NULL,
    provider        VARCHAR(50) NOT NULL DEFAULT 'gemini',
    account_email   VARCHAR(255),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS project_accounts (
    project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    account_id      UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    PRIMARY KEY (project_id, account_id)
);

CREATE TABLE IF NOT EXISTS generations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    account_id      UUID REFERENCES accounts(id) ON DELETE SET NULL,
    seq_num         INTEGER NOT NULL DEFAULT 0,
    prompt          TEXT NOT NULL,
    negative_prompt TEXT DEFAULT '',
    response_text   TEXT DEFAULT '',
    seed            BIGINT,
    model_params    JSONB NOT NULL DEFAULT '{}',
    typography      JSONB DEFAULT '{}',
    mask_source_url VARCHAR(500),
    result_urls     TEXT[] DEFAULT '{}',
    reference_urls  TEXT[] DEFAULT '{}',
    status          VARCHAR(20) NOT NULL DEFAULT 'generated'
                    CHECK (status IN ('generated', 'approved', 'rejected', 'reworking')),
    feedback_note   TEXT DEFAULT '',
    qdrant_point_id UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_generations_project ON generations(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_generations_account ON generations(account_id);
CREATE INDEX IF NOT EXISTS idx_generations_status ON generations(status);
CREATE INDEX IF NOT EXISTS idx_generations_seed ON generations(seed);
CREATE INDEX IF NOT EXISTS idx_generations_seq ON generations(project_id, seq_num);

CREATE OR REPLACE FUNCTION set_generation_seq_num()
RETURNS TRIGGER AS $$
BEGIN
    SELECT COALESCE(MAX(seq_num), 0) + 1 INTO NEW.seq_num
    FROM generations WHERE project_id = NEW.project_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_generation_seq_num ON generations;
CREATE TRIGGER trg_generation_seq_num
    BEFORE INSERT ON generations FOR EACH ROW
    EXECUTE FUNCTION set_generation_seq_num();

CREATE TABLE IF NOT EXISTS context_summaries (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    context_type    VARCHAR(20) NOT NULL
                    CHECK (context_type IN ('strategic', 'tactical')),
    summary_text    TEXT NOT NULL,
    gen_from_seq    INTEGER NOT NULL,
    gen_to_seq      INTEGER NOT NULL,
    gen_count       INTEGER NOT NULL,
    model_used      VARCHAR(100) DEFAULT 'gemini-flash',
    tokens_input    INTEGER DEFAULT 0,
    tokens_output   INTEGER DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_context_project_type ON context_summaries(project_id, context_type, created_at DESC);

CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    account_id      UUID REFERENCES accounts(id) ON DELETE SET NULL,
    title           VARCHAR(255) DEFAULT '',
    notes           TEXT DEFAULT '',
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id, started_at DESC);

CREATE OR REPLACE VIEW v_latest_contexts AS
SELECT DISTINCT ON (project_id, context_type)
    id, project_id, context_type, summary_text,
    gen_from_seq, gen_to_seq, gen_count, created_at
FROM context_summaries
ORDER BY project_id, context_type, created_at DESC;

CREATE OR REPLACE VIEW v_project_stats AS
SELECT p.id, p.name, p.status,
    COUNT(g.id) AS total_generations,
    COUNT(g.id) FILTER (WHERE g.status = 'approved') AS approved_count,
    COUNT(g.id) FILTER (WHERE g.status = 'rejected') AS rejected_count,
    MAX(g.created_at) AS last_generation_at,
    COUNT(DISTINCT g.account_id) AS accounts_used
FROM projects p LEFT JOIN generations g ON g.project_id = p.id
GROUP BY p.id, p.name, p.status;

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$ BEGIN NEW.updated_at = NOW(); RETURN NEW; END; $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_projects_updated_at ON projects;
CREATE TRIGGER trg_projects_updated_at
    BEFORE UPDATE ON projects FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
