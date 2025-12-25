-- ============================================================================
-- POSTGRESQL SCHEMA: Service Credit Union Call Center Platform
-- Version: 1.0.0
-- Description: Complete schema for enterprise call intelligence with 
--              identity verification, real-time transcription, and analytics
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text search

-- ============================================================================
-- CALLS - Master call record
-- ============================================================================
CREATE TABLE IF NOT EXISTS calls (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_sid        VARCHAR(64) UNIQUE NOT NULL,  -- External call ID from PBX
    
    -- Call Identification
    ani             VARCHAR(20),          -- Caller's phone number (ANI)
    dnis            VARCHAR(20),          -- Dialed number (DNIS)
    direction       VARCHAR(10) NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    channel         VARCHAR(20) NOT NULL DEFAULT 'phone',
    
    -- Member Link (NULL until verified)
    member_id       VARCHAR(20),          -- Fiserv member ID
    member_verified BOOLEAN DEFAULT FALSE,
    verification_method VARCHAR(50),      -- 'ani_match', 'kba', 'mfa', 'manual'
    verification_attempts INTEGER DEFAULT 0,
    verification_passed_at TIMESTAMPTZ,
    
    -- Agent Info
    agent_id        VARCHAR(50),
    agent_extension VARCHAR(20),
    queue_id        VARCHAR(50),
    
    -- Call Timing
    call_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    call_answered_at TIMESTAMPTZ,
    call_ended_at   TIMESTAMPTZ,
    hold_duration_sec INTEGER DEFAULT 0,
    wrap_up_duration_sec INTEGER DEFAULT 0,
    
    -- Call Disposition
    status          VARCHAR(20) DEFAULT 'ringing' CHECK (status IN ('ringing', 'in_progress', 'on_hold', 'completed', 'abandoned', 'transferred')),
    disposition     VARCHAR(50),          -- resolved, follow_up_needed, escalated, etc.
    transfer_to     VARCHAR(50),
    
    -- Transcription & Analysis
    transcript_status VARCHAR(20) DEFAULT 'pending' CHECK (transcript_status IN ('pending', 'processing', 'completed', 'failed')),
    transcript_text TEXT,
    transcript_redacted TEXT,
    summary         TEXT,
    summary_type    VARCHAR(20),
    
    -- Sentiment & Intent
    sentiment_score DECIMAL(4,3) CHECK (sentiment_score >= -1 AND sentiment_score <= 1),
    emotion_primary VARCHAR(20),
    intent_detected VARCHAR(100),
    
    -- Problem Detection
    problem_categories JSONB DEFAULT '[]',
    risk_level      VARCHAR(10) CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    
    -- Fiserv Context (cached at call start)
    fiserv_member_data JSONB,
    fiserv_accounts JSONB,
    
    -- Metadata
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for call lookup
CREATE INDEX IF NOT EXISTS idx_calls_ani ON calls(ani);
CREATE INDEX IF NOT EXISTS idx_calls_member ON calls(member_id);
CREATE INDEX IF NOT EXISTS idx_calls_agent ON calls(agent_id);
CREATE INDEX IF NOT EXISTS idx_calls_status ON calls(status);
CREATE INDEX IF NOT EXISTS idx_calls_started ON calls(call_started_at DESC);
CREATE INDEX IF NOT EXISTS idx_calls_verification ON calls(member_verified, verification_method);
CREATE INDEX IF NOT EXISTS idx_calls_problems ON calls USING GIN(problem_categories);
CREATE INDEX IF NOT EXISTS idx_calls_transcript ON calls USING GIN(to_tsvector('english', transcript_text));

-- ============================================================================
-- CALL_EVENTS - Lifecycle events (start, answer, hold, transfer, end)
-- ============================================================================
CREATE TABLE IF NOT EXISTS call_events (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id         UUID NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    
    event_type      VARCHAR(50) NOT NULL,  -- started, ringing, answered, hold_started, hold_ended, transfer, ended
    event_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Event-specific data
    previous_state  VARCHAR(50),
    new_state       VARCHAR(50),
    agent_id        VARCHAR(50),
    queue_id        VARCHAR(50),
    transfer_target VARCHAR(100),
    
    -- CTI metadata
    source_system   VARCHAR(50),          -- asterisk, genesys, etc.
    raw_event       JSONB,
    
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_call_events_call ON call_events(call_id);
CREATE INDEX IF NOT EXISTS idx_call_events_type ON call_events(event_type);
CREATE INDEX IF NOT EXISTS idx_call_events_time ON call_events(event_timestamp DESC);

-- ============================================================================
-- VERIFICATION_ATTEMPTS - Member identity verification log
-- ============================================================================
CREATE TABLE IF NOT EXISTS verification_attempts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id         UUID NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    
    -- What was verified
    method          VARCHAR(50) NOT NULL,  -- ani_lookup, kba_dob, kba_ssn4, pin_entry, mfa_sms, mfa_email, voice_biometric
    challenge_type  VARCHAR(50),           -- dob, ssn_last4, account_number, otp, security_question
    challenge_data  JSONB,                 -- Non-sensitive challenge metadata
    
    -- Result
    success         BOOLEAN NOT NULL,
    failure_reason  VARCHAR(200),
    confidence_score DECIMAL(4,3),         -- For biometrics (0.000 - 1.000)
    
    -- Audit
    attempted_at    TIMESTAMPTZ DEFAULT NOW(),
    response_time_ms INTEGER,
    agent_id        VARCHAR(50),
    
    -- Security
    ip_address      INET,
    user_agent      TEXT
);

CREATE INDEX IF NOT EXISTS idx_verification_call ON verification_attempts(call_id);
CREATE INDEX IF NOT EXISTS idx_verification_method ON verification_attempts(method, success);
CREATE INDEX IF NOT EXISTS idx_verification_time ON verification_attempts(attempted_at DESC);

-- ============================================================================
-- CALL_SEGMENTS - Speaker turns with timestamps
-- ============================================================================
CREATE TABLE IF NOT EXISTS call_segments (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id         UUID NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    
    segment_index   INTEGER NOT NULL,      -- Order within call
    speaker         VARCHAR(20) NOT NULL,  -- agent, member, ivr, hold_music
    speaker_id      VARCHAR(50),           -- agent_id or member_id
    
    start_time_sec  DECIMAL(10,3) NOT NULL,
    end_time_sec    DECIMAL(10,3) NOT NULL,
    duration_sec    DECIMAL(10,3) GENERATED ALWAYS AS (end_time_sec - start_time_sec) STORED,
    
    text            TEXT NOT NULL,
    text_redacted   TEXT,
    
    -- Analysis
    sentiment_score DECIMAL(4,3),
    emotion         VARCHAR(20),
    keywords        TEXT[],
    
    -- Audio reference
    audio_offset_ms BIGINT,
    audio_duration_ms INTEGER,
    
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_segments_call ON call_segments(call_id);
CREATE INDEX IF NOT EXISTS idx_segments_speaker ON call_segments(call_id, speaker);
CREATE INDEX IF NOT EXISTS idx_segments_time ON call_segments(call_id, start_time_sec);
CREATE INDEX IF NOT EXISTS idx_segments_text ON call_segments USING GIN(to_tsvector('english', text));

-- ============================================================================
-- PHONE_NUMBER_REGISTRY - Known member phone numbers for ANI matching
-- ============================================================================
CREATE TABLE IF NOT EXISTS phone_number_registry (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phone_number    VARCHAR(20) NOT NULL,  -- Normalized E.164 format (+16035551234)
    member_id       VARCHAR(20) NOT NULL,
    
    number_type     VARCHAR(20) NOT NULL CHECK (number_type IN ('mobile', 'home', 'work', 'other')),
    is_primary      BOOLEAN DEFAULT FALSE,
    verified        BOOLEAN DEFAULT FALSE,
    verified_at     TIMESTAMPTZ,
    verification_method VARCHAR(50),       -- sms_otp, call_back, agent_manual
    
    -- Risk scoring
    fraud_flags     INTEGER DEFAULT 0,
    last_successful_auth TIMESTAMPTZ,
    failed_auth_count INTEGER DEFAULT 0,
    
    -- Source tracking
    source          VARCHAR(50) DEFAULT 'fiserv_sync',  -- fiserv_sync, member_update, manual_entry
    
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(phone_number, member_id)
);

CREATE INDEX IF NOT EXISTS idx_phone_registry_number ON phone_number_registry(phone_number);
CREATE INDEX IF NOT EXISTS idx_phone_registry_member ON phone_number_registry(member_id);
CREATE INDEX IF NOT EXISTS idx_phone_registry_verified ON phone_number_registry(verified, is_primary);

-- ============================================================================
-- ACTION_ITEMS - Follow-ups from calls
-- ============================================================================
CREATE TABLE IF NOT EXISTS action_items (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id         UUID NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    
    description     TEXT NOT NULL,
    assignee_type   VARCHAR(20) CHECK (assignee_type IN ('agent', 'supervisor', 'member', 'system')),
    assignee_id     VARCHAR(50),
    
    priority        VARCHAR(10) DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    due_date        DATE,
    
    status          VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'in_progress', 'completed', 'cancelled')),
    completed_at    TIMESTAMPTZ,
    completed_by    VARCHAR(50),
    resolution_notes TEXT,
    
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_action_items_call ON action_items(call_id);
CREATE INDEX IF NOT EXISTS idx_action_items_assignee ON action_items(assignee_id, status);
CREATE INDEX IF NOT EXISTS idx_action_items_due ON action_items(due_date, status);
CREATE INDEX IF NOT EXISTS idx_action_items_priority ON action_items(priority, status);

-- ============================================================================
-- PROBLEM_CATEGORIES - Taxonomy and detection patterns
-- ============================================================================
CREATE TABLE IF NOT EXISTS problem_categories (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    category_code   VARCHAR(50) UNIQUE NOT NULL,
    
    display_name    VARCHAR(100) NOT NULL,
    parent_code     VARCHAR(50) REFERENCES problem_categories(category_code),
    description     TEXT,
    
    severity        VARCHAR(10) DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    requires_callback BOOLEAN DEFAULT FALSE,
    escalation_queue VARCHAR(50),
    sla_minutes     INTEGER,
    
    -- Detection
    keywords        TEXT[],
    regex_patterns  TEXT[],
    ml_model_id     VARCHAR(100),
    
    active          BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_problem_categories_parent ON problem_categories(parent_code);
CREATE INDEX IF NOT EXISTS idx_problem_categories_severity ON problem_categories(severity, active);

-- ============================================================================
-- CALL_PROBLEMS - Many-to-many relationship for detected problems
-- ============================================================================
CREATE TABLE IF NOT EXISTS call_problems (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id         UUID NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    category_code   VARCHAR(50) NOT NULL REFERENCES problem_categories(category_code),
    
    confidence      DECIMAL(4,3) DEFAULT 0.8,
    detection_method VARCHAR(50),          -- keyword, ml, agent_tagged
    
    resolved        BOOLEAN DEFAULT FALSE,
    resolved_at     TIMESTAMPTZ,
    resolution_notes TEXT,
    
    detected_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_call_problems_call ON call_problems(call_id);
CREATE INDEX IF NOT EXISTS idx_call_problems_category ON call_problems(category_code);
CREATE INDEX IF NOT EXISTS idx_call_problems_resolved ON call_problems(resolved, detected_at);

-- ============================================================================
-- AGENT_SESSIONS - Track agent availability and stats
-- ============================================================================
CREATE TABLE IF NOT EXISTS agent_sessions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id        VARCHAR(50) NOT NULL,
    
    session_started TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_ended   TIMESTAMPTZ,
    
    status          VARCHAR(20) DEFAULT 'available' CHECK (status IN ('available', 'on_call', 'wrap_up', 'break', 'training', 'offline')),
    current_call_id UUID REFERENCES calls(id),
    
    calls_handled   INTEGER DEFAULT 0,
    avg_handle_time_sec INTEGER,
    avg_wrap_time_sec INTEGER,
    
    queue_ids       TEXT[],
    skills          TEXT[],
    
    metadata        JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_agent_sessions_agent ON agent_sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_status ON agent_sessions(status);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_active ON agent_sessions(session_ended) WHERE session_ended IS NULL;

-- ============================================================================
-- VERIFICATION_CHALLENGES - Define available verification methods
-- ============================================================================
CREATE TABLE IF NOT EXISTS verification_challenges (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    challenge_code  VARCHAR(50) UNIQUE NOT NULL,
    
    display_name    VARCHAR(100) NOT NULL,
    description     TEXT,
    
    challenge_type  VARCHAR(20) NOT NULL CHECK (challenge_type IN ('knowledge', 'possession', 'inherence')),
    security_level  INTEGER DEFAULT 2 CHECK (security_level BETWEEN 1 AND 5),
    
    -- Configuration
    prompt_template TEXT,                  -- "Please provide your date of birth"
    answer_format   VARCHAR(50),           -- mm/dd/yyyy, 4digits, otp6
    max_attempts    INTEGER DEFAULT 3,
    timeout_seconds INTEGER DEFAULT 300,
    
    -- Availability
    requires_otp_provider BOOLEAN DEFAULT FALSE,
    requires_biometric BOOLEAN DEFAULT FALSE,
    
    active          BOOLEAN DEFAULT TRUE,
    sort_order      INTEGER DEFAULT 100,
    
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default verification challenges
INSERT INTO verification_challenges (challenge_code, display_name, challenge_type, security_level, prompt_template, answer_format, sort_order)
VALUES 
    ('kba_dob', 'Date of Birth', 'knowledge', 2, 'Please provide your date of birth.', 'mm/dd/yyyy', 10),
    ('kba_ssn4', 'SSN Last 4 Digits', 'knowledge', 3, 'Please provide the last 4 digits of your Social Security Number.', '4digits', 20),
    ('kba_account', 'Account Number', 'knowledge', 2, 'Please provide your account number.', 'account_number', 30),
    ('kba_recent_tx', 'Recent Transaction', 'knowledge', 4, 'Please describe your most recent transaction amount.', 'amount', 40),
    ('mfa_sms', 'SMS One-Time Password', 'possession', 4, 'We will send a 6-digit code to your phone. Please provide the code.', 'otp6', 50),
    ('mfa_email', 'Email One-Time Password', 'possession', 3, 'We will send a 6-digit code to your email. Please provide the code.', 'otp6', 60),
    ('voice_bio', 'Voice Biometrics', 'inherence', 5, 'Please repeat the following phrase for voice verification.', 'voice', 70)
ON CONFLICT (challenge_code) DO NOTHING;

-- ============================================================================
-- AUDIT_LOG - Compliance and security tracking
-- ============================================================================
CREATE TABLE IF NOT EXISTS audit_log (
    id              BIGSERIAL PRIMARY KEY,
    log_timestamp   TIMESTAMPTZ DEFAULT NOW(),
    
    event_type      VARCHAR(50) NOT NULL,
    resource_type   VARCHAR(50) NOT NULL,  -- call, member, verification, agent
    resource_id     VARCHAR(100),
    
    actor_type      VARCHAR(20) CHECK (actor_type IN ('agent', 'system', 'member', 'admin')),
    actor_id        VARCHAR(50),
    
    action          VARCHAR(50) NOT NULL,
    description     TEXT,
    old_value       JSONB,
    new_value       JSONB,
    
    ip_address      INET,
    user_agent      TEXT,
    session_id      VARCHAR(100),
    
    -- Compliance flags
    pii_accessed    BOOLEAN DEFAULT FALSE,
    financial_data  BOOLEAN DEFAULT FALSE,
    verification_event BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(log_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_log(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_pii ON audit_log(pii_accessed) WHERE pii_accessed = TRUE;

-- ============================================================================
-- VIEWS - Convenience queries
-- ============================================================================

-- Current active calls
CREATE OR REPLACE VIEW active_calls AS
SELECT 
    c.*,
    a.agent_id as session_agent,
    a.status as agent_status
FROM calls c
LEFT JOIN agent_sessions a ON a.current_call_id = c.id AND a.session_ended IS NULL
WHERE c.status IN ('ringing', 'in_progress', 'on_hold');

-- Unverified calls needing attention
CREATE OR REPLACE VIEW unverified_active_calls AS
SELECT 
    c.*,
    COALESCE(c.verification_attempts, 0) as total_attempts
FROM calls c
WHERE c.status IN ('ringing', 'in_progress', 'on_hold')
  AND c.member_verified = FALSE;

-- Daily call metrics
CREATE OR REPLACE VIEW daily_call_metrics AS
SELECT 
    DATE(call_started_at) as call_date,
    COUNT(*) as total_calls,
    COUNT(*) FILTER (WHERE status = 'completed') as completed_calls,
    COUNT(*) FILTER (WHERE status = 'abandoned') as abandoned_calls,
    COUNT(*) FILTER (WHERE member_verified) as verified_calls,
    ROUND(AVG(EXTRACT(EPOCH FROM (call_ended_at - call_started_at)))::numeric, 2) as avg_duration_sec,
    ROUND(AVG(sentiment_score)::numeric, 3) as avg_sentiment,
    COUNT(*) FILTER (WHERE risk_level = 'critical') as critical_issues,
    COUNT(*) FILTER (WHERE risk_level = 'high') as high_issues
FROM calls
WHERE call_started_at > NOW() - INTERVAL '90 days'
GROUP BY DATE(call_started_at)
ORDER BY call_date DESC;

-- Member call history summary
CREATE OR REPLACE VIEW member_call_summary AS
SELECT 
    member_id,
    COUNT(*) as total_calls,
    MIN(call_started_at) as first_call,
    MAX(call_started_at) as last_call,
    ROUND(AVG(EXTRACT(EPOCH FROM (call_ended_at - call_started_at)))::numeric, 2) as avg_duration_sec,
    ROUND(AVG(sentiment_score)::numeric, 3) as avg_sentiment,
    COUNT(*) FILTER (WHERE risk_level IN ('high', 'critical')) as escalated_calls
FROM calls
WHERE member_id IS NOT NULL
GROUP BY member_id;

-- ============================================================================
-- FUNCTIONS - Utility functions
-- ============================================================================

-- Function to normalize phone numbers to E.164
CREATE OR REPLACE FUNCTION normalize_phone(phone TEXT)
RETURNS TEXT AS $$
DECLARE
    cleaned TEXT;
BEGIN
    -- Remove all non-digits
    cleaned := regexp_replace(phone, '[^0-9]', '', 'g');
    
    -- Handle US numbers
    IF length(cleaned) = 10 THEN
        RETURN '+1' || cleaned;
    ELSIF length(cleaned) = 11 AND LEFT(cleaned, 1) = '1' THEN
        RETURN '+' || cleaned;
    ELSE
        RETURN '+' || cleaned;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update trigger to relevant tables
CREATE TRIGGER trigger_calls_updated_at
    BEFORE UPDATE ON calls
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trigger_action_items_updated_at
    BEFORE UPDATE ON action_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trigger_phone_registry_updated_at
    BEFORE UPDATE ON phone_number_registry
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- SEED DATA - Problem categories
-- ============================================================================
INSERT INTO problem_categories (category_code, display_name, severity, keywords) VALUES
    ('account_access', 'Account Access', 'medium', ARRAY['login', 'password', 'locked', 'access']),
    ('account_access.login', 'Login Issues', 'medium', ARRAY['cannot log in', 'login failed', 'password wrong']),
    ('account_access.locked', 'Locked Account', 'high', ARRAY['locked out', 'account frozen', 'suspended']),
    ('account_access.mfa', 'MFA Problems', 'medium', ARRAY['mfa', 'two factor', 'code not working', 'authenticator']),
    
    ('transactions', 'Transactions', 'medium', ARRAY['transfer', 'payment', 'deposit']),
    ('transactions.failed', 'Failed Transfer', 'high', ARRAY['transfer failed', 'didnt go through', 'declined']),
    ('transactions.missing', 'Missing Deposit', 'high', ARRAY['missing deposit', 'not showing', 'where is my']),
    ('transactions.dispute', 'Transaction Dispute', 'medium', ARRAY['dispute', 'unauthorized', 'fraud', 'wrong charge']),
    
    ('cards', 'Cards', 'medium', ARRAY['card', 'debit', 'credit']),
    ('cards.lost_stolen', 'Lost/Stolen Card', 'high', ARRAY['lost card', 'stolen', 'cant find']),
    ('cards.fraud', 'Card Fraud', 'critical', ARRAY['fraud', 'unauthorized purchase', 'didnt make']),
    
    ('loans', 'Loans', 'medium', ARRAY['loan', 'payment', 'interest']),
    ('loans.payment', 'Loan Payment', 'low', ARRAY['payment due', 'how much', 'balance']),
    
    ('fraud', 'Fraud', 'critical', ARRAY['fraud', 'scam', 'unauthorized', 'identity theft']),
    ('fraud.unauthorized', 'Unauthorized Activity', 'critical', ARRAY['didnt authorize', 'wasnt me', 'stolen']),
    ('fraud.identity', 'Identity Theft', 'critical', ARRAY['identity theft', 'someone opened', 'not my account'])
ON CONFLICT (category_code) DO NOTHING;

-- ============================================================================
-- GRANTS - Access control (adjust roles as needed)
-- ============================================================================
-- Example: GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO call_center_app;
-- Example: GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO call_center_app;

COMMENT ON DATABASE postgres IS 'Service Credit Union Call Center Platform - v1.0.0';
