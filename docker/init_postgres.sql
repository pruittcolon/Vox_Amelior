-- PostgreSQL Schema for Nemo Server
-- Matches existing SQLite schemas from rag.db and users.db
-- Run with: docker exec -i refactored_postgres psql -U $POSTGRES_USER -d nemo_queue < init_postgres.sql

-- =============================================================================
-- Users Database (from users.db)
-- =============================================================================

CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(64) PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(50) NOT NULL,
    speaker_id VARCHAR(64),
    email VARCHAR(255),
    created_at DOUBLE PRECISION NOT NULL,
    modified_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_speaker_id ON users(speaker_id);

-- =============================================================================
-- Sessions Table (for distributed session storage)
-- =============================================================================

CREATE TABLE IF NOT EXISTS sessions (
    session_token TEXT PRIMARY KEY,
    user_id VARCHAR(64) REFERENCES users(user_id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    speaker_id VARCHAR(64),
    created_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL,
    ip_address VARCHAR(45),
    last_refresh DOUBLE PRECISION,
    csrf_token TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);

-- =============================================================================
-- RAG Memories (from rag.db)
-- =============================================================================

CREATE TABLE IF NOT EXISTS memories (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding BYTEA,
    metadata JSONB DEFAULT '{}',
    user_id VARCHAR(64),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);

-- =============================================================================
-- Transcript Records (from rag.db)
-- =============================================================================

CREATE TABLE IF NOT EXISTS transcript_records (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    filename VARCHAR(512),
    duration_seconds DOUBLE PRECISION,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_transcript_records_session_id ON transcript_records(session_id);
CREATE INDEX IF NOT EXISTS idx_transcript_records_status ON transcript_records(status);

-- =============================================================================
-- Transcript Segments (from rag.db)
-- =============================================================================

CREATE TABLE IF NOT EXISTS transcript_segments (
    id SERIAL PRIMARY KEY,
    transcript_id INTEGER REFERENCES transcript_records(id) ON DELETE CASCADE,
    speaker VARCHAR(255),
    text TEXT NOT NULL,
    start_time DOUBLE PRECISION,
    end_time DOUBLE PRECISION,
    emotion VARCHAR(50),
    confidence DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transcript_segments_transcript_id ON transcript_segments(transcript_id);
CREATE INDEX IF NOT EXISTS idx_transcript_segments_speaker ON transcript_segments(speaker);

-- =============================================================================
-- QA Items (from gateway)
-- =============================================================================

CREATE TABLE IF NOT EXISTS qa_items (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT,
    category VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending',
    user_id VARCHAR(64),
    created_at TIMESTAMP DEFAULT NOW(),
    answered_at TIMESTAMP
);

-- =============================================================================
-- Meetings (from gateway)
-- =============================================================================

CREATE TABLE IF NOT EXISTS meetings (
    id SERIAL PRIMARY KEY,
    title VARCHAR(512) NOT NULL,
    description TEXT,
    scheduled_at TIMESTAMP,
    duration_minutes INTEGER,
    attendees JSONB DEFAULT '[]',
    status VARCHAR(50) DEFAULT 'scheduled',
    user_id VARCHAR(64),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_meetings_user_id ON meetings(user_id);
CREATE INDEX IF NOT EXISTS idx_meetings_scheduled_at ON meetings(scheduled_at);

-- =============================================================================
-- Automation Rules (from gateway)
-- =============================================================================

CREATE TABLE IF NOT EXISTS automation_rules (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    trigger_type VARCHAR(100) NOT NULL,
    trigger_config JSONB DEFAULT '{}',
    action_type VARCHAR(100) NOT NULL,
    action_config JSONB DEFAULT '{}',
    enabled BOOLEAN DEFAULT true,
    user_id VARCHAR(64),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_automation_rules_user_id ON automation_rules(user_id);
CREATE INDEX IF NOT EXISTS idx_automation_rules_enabled ON automation_rules(enabled);

-- =============================================================================
-- Email Storage (from rag email module)
-- =============================================================================

CREATE TABLE IF NOT EXISTS emails (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(512) UNIQUE,
    thread_id VARCHAR(512),
    subject TEXT,
    sender VARCHAR(512),
    recipients JSONB DEFAULT '[]',
    body TEXT,
    html_body TEXT,
    attachments JSONB DEFAULT '[]',
    labels JSONB DEFAULT '[]',
    received_at TIMESTAMP,
    processed_at TIMESTAMP DEFAULT NOW(),
    embedding BYTEA,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_emails_thread_id ON emails(thread_id);
CREATE INDEX IF NOT EXISTS idx_emails_sender ON emails(sender);
CREATE INDEX IF NOT EXISTS idx_emails_received_at ON emails(received_at);

-- =============================================================================
-- Call Intelligence (from gateway)
-- =============================================================================

CREATE TABLE IF NOT EXISTS call_records (
    id SERIAL PRIMARY KEY,
    call_id VARCHAR(64) UNIQUE NOT NULL,
    caller_id VARCHAR(255),
    callee_id VARCHAR(255),
    duration_seconds DOUBLE PRECISION,
    status VARCHAR(50),
    transcript_id INTEGER REFERENCES transcript_records(id),
    sentiment_score DOUBLE PRECISION,
    topics JSONB DEFAULT '[]',
    action_items JSONB DEFAULT '[]',
    user_id VARCHAR(64),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_call_records_user_id ON call_records(user_id);
CREATE INDEX IF NOT EXISTS idx_call_records_created_at ON call_records(created_at);

-- =============================================================================
-- Loan Applications (Fiserv Banking)
-- =============================================================================

CREATE TABLE IF NOT EXISTS loan_applications (
    id VARCHAR(64) PRIMARY KEY,
    applicant_name VARCHAR(255) NOT NULL,
    amount DECIMAL(12, 2) NOT NULL,
    loan_type VARCHAR(50) NOT NULL,
    credit_score INTEGER NOT NULL,
    monthly_income DECIMAL(12, 2) NOT NULL,
    existing_debt DECIMAL(12, 2) DEFAULT 0,
    member_id VARCHAR(64),
    status VARCHAR(50) DEFAULT 'pending',
    assigned_officer VARCHAR(255),
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_loan_applications_status ON loan_applications(status);
CREATE INDEX IF NOT EXISTS idx_loan_applications_loan_type ON loan_applications(loan_type);
CREATE INDEX IF NOT EXISTS idx_loan_applications_created_at ON loan_applications(created_at);

-- =============================================================================
-- Cases (Fiserv Case Management)
-- =============================================================================

CREATE TABLE IF NOT EXISTS cases (
    id VARCHAR(64) PRIMARY KEY,
    case_type VARCHAR(100) NOT NULL,
    subject VARCHAR(512) NOT NULL,
    description TEXT,
    member_id VARCHAR(64),
    account_id VARCHAR(64),
    priority VARCHAR(50) DEFAULT 'medium',
    status VARCHAR(50) DEFAULT 'open',
    assignee_id VARCHAR(64),
    assignee_name VARCHAR(255),
    resolution_summary TEXT,
    due_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    resolved_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
CREATE INDEX IF NOT EXISTS idx_cases_priority ON cases(priority);
CREATE INDEX IF NOT EXISTS idx_cases_assignee_id ON cases(assignee_id);
CREATE INDEX IF NOT EXISTS idx_cases_created_at ON cases(created_at);

-- Case Notes (for timeline)
CREATE TABLE IF NOT EXISTS case_notes (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(64) REFERENCES cases(id) ON DELETE CASCADE,
    note TEXT NOT NULL,
    user_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_case_notes_case_id ON case_notes(case_id);

-- Grant permissions (if running as different user)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO nemo_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO nemo_user;

