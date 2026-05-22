-- =============================================================================
-- Nexus PostgreSQL Schema
-- Run automatically on first docker compose up
-- =============================================================================

-- MLflow database
CREATE DATABASE mlflow;

-- Feature registry metadata
CREATE TABLE IF NOT EXISTS feature_views (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(255) UNIQUE NOT NULL,
    schema_json     JSONB NOT NULL,
    fingerprint     VARCHAR(32) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Feature materialisation history
CREATE TABLE IF NOT EXISTS materialisation_runs (
    id              SERIAL PRIMARY KEY,
    feature_view    VARCHAR(255) NOT NULL,
    status          VARCHAR(32) NOT NULL,  -- ok | error
    rows_written    INTEGER,
    as_of           TIMESTAMPTZ NOT NULL,
    duration_ms     INTEGER,
    error_msg       TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Experiment definitions
CREATE TABLE IF NOT EXISTS experiments (
    id              VARCHAR(64) PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    status          VARCHAR(32) NOT NULL DEFAULT 'draft',
    config_json     JSONB NOT NULL,
    started_at      TIMESTAMPTZ,
    stopped_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Experiment assignments (user → variant mapping log)
CREATE TABLE IF NOT EXISTS experiment_assignments (
    id              BIGSERIAL PRIMARY KEY,
    experiment_id   VARCHAR(64) NOT NULL,
    user_id         BIGINT NOT NULL,
    variant_name    VARCHAR(128) NOT NULL,
    assigned_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(experiment_id, user_id)
);

-- Experiment events (for metric computation)
CREATE TABLE IF NOT EXISTS experiment_events (
    id              BIGSERIAL PRIMARY KEY,
    experiment_id   VARCHAR(64),
    user_id         BIGINT NOT NULL,
    variant_name    VARCHAR(128),
    metric_name     VARCHAR(255) NOT NULL,
    metric_value    DOUBLE PRECISION NOT NULL,
    event_timestamp TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_exp_events_exp_id  ON experiment_events(experiment_id);
CREATE INDEX idx_exp_events_user    ON experiment_events(user_id);
CREATE INDEX idx_exp_events_ts      ON experiment_events(event_timestamp);

-- Model serving records
CREATE TABLE IF NOT EXISTS model_serving_log (
    id              BIGSERIAL PRIMARY KEY,
    request_id      VARCHAR(64) NOT NULL,
    user_id         BIGINT,
    model_name      VARCHAR(128) NOT NULL,
    model_version   VARCHAR(32),
    latency_ms      INTEGER,
    served_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Fraud review queue (Human-in-the-loop)
CREATE TABLE IF NOT EXISTS fraud_review_queue (
    id              BIGSERIAL PRIMARY KEY,
    entity_type     VARCHAR(32) NOT NULL,  -- user | item | transaction
    entity_id       BIGINT NOT NULL,
    fraud_score     DOUBLE PRECISION NOT NULL,
    model_version   VARCHAR(32),
    status          VARCHAR(32) DEFAULT 'pending',  -- pending | reviewed | escalated
    reviewer_id     INTEGER,
    reviewer_notes  TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    reviewed_at     TIMESTAMPTZ
);
CREATE INDEX idx_fraud_queue_status ON fraud_review_queue(status);
CREATE INDEX idx_fraud_queue_score  ON fraud_review_queue(fraud_score DESC);

GRANT ALL PRIVILEGES ON ALL TABLES    IN SCHEMA public TO nexus;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO nexus;
