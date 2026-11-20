CREATE TABLE IF NOT EXISTS auth_rate_limits (
    user_id TEXT PRIMARY KEY,
    window_start TIMESTAMPTZ NOT NULL,
    used INTEGER NOT NULL CHECK (used > 0)
);
CREATE INDEX IF NOT EXISTS idx_auth_rate_limits_window ON auth_rate_limits(window_start);
