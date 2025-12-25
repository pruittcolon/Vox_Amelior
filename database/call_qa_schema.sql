-- ============================================================================
-- CALL QA SCHEMA: Quality Assurance Analysis Tables
-- Part of Service Credit Union Call Center Platform
-- Version: 1.0.0
-- ============================================================================

-- ============================================================================
-- CALL_QA_CHUNKS - Chunked transcript analysis with Gemma scores
-- ============================================================================
CREATE TABLE IF NOT EXISTS call_qa_chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id         UUID NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    
    -- Chunk identification
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    chunk_text_redacted TEXT,
    token_count     INTEGER,
    
    -- Time reference (if available from segments)
    start_time_sec  DECIMAL(10,3),
    end_time_sec    DECIMAL(10,3),
    
    -- Speaker in chunk (if determinable)
    primary_speaker VARCHAR(20) CHECK (primary_speaker IN ('agent', 'member', 'mixed', 'unknown')),
    
    -- Gemma QA Scores (1-10 scale)
    score_professionalism INTEGER CHECK (score_professionalism BETWEEN 1 AND 10),
    score_compliance INTEGER CHECK (score_compliance BETWEEN 1 AND 10),
    score_customer_service INTEGER CHECK (score_customer_service BETWEEN 1 AND 10),
    score_protocol INTEGER CHECK (score_protocol BETWEEN 1 AND 10),
    
    -- Weighted overall score (computed)
    -- 30% compliance, 25% professionalism, 25% customer_service, 20% protocol
    score_overall DECIMAL(3,1) GENERATED ALWAYS AS (
        ROUND((
            COALESCE(score_professionalism, 5) * 0.25 +
            COALESCE(score_compliance, 5) * 0.30 +
            COALESCE(score_customer_service, 5) * 0.25 +
            COALESCE(score_protocol, 5) * 0.20
        )::numeric, 1)
    ) STORED,
    
    -- Gemma rationales (for human-in-the-loop review)
    rationale_professionalism TEXT,
    rationale_compliance TEXT,
    rationale_customer_service TEXT,
    rationale_protocol TEXT,
    
    -- Flagged issues for compliance review
    compliance_flags JSONB DEFAULT '[]',
    requires_review BOOLEAN DEFAULT FALSE,
    review_reason TEXT,
    
    -- Review tracking
    reviewed_by VARCHAR(50),
    reviewed_at TIMESTAMPTZ,
    review_notes TEXT,
    
    -- Vectorization reference (RAG/ChromaDB)
    vector_id       VARCHAR(100),
    embedding_model VARCHAR(50) DEFAULT 'all-MiniLM-L6-v2',
    
    -- Gemma processing metadata
    gemma_model     VARCHAR(50) DEFAULT 'gemma-3-4b-it',
    gemma_task_id   VARCHAR(100),
    gemma_raw_response JSONB,
    processing_time_ms INTEGER,
    
    analyzed_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_qa_chunks_call ON call_qa_chunks(call_id);
CREATE INDEX IF NOT EXISTS idx_qa_chunks_review ON call_qa_chunks(requires_review) WHERE requires_review = TRUE;
CREATE INDEX IF NOT EXISTS idx_qa_chunks_scores ON call_qa_chunks(score_overall DESC);
CREATE INDEX IF NOT EXISTS idx_qa_chunks_analyzed ON call_qa_chunks(analyzed_at DESC);
CREATE INDEX IF NOT EXISTS idx_qa_chunks_compliance ON call_qa_chunks USING GIN(compliance_flags);

-- ============================================================================
-- AGENT_QA_METRICS - Aggregated agent performance (daily/weekly/monthly)
-- ============================================================================
CREATE TABLE IF NOT EXISTS agent_qa_metrics (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id        VARCHAR(50) NOT NULL,
    
    -- Period this metric covers
    period_type     VARCHAR(10) NOT NULL CHECK (period_type IN ('daily', 'weekly', 'monthly')),
    period_start    DATE NOT NULL,
    period_end      DATE NOT NULL,
    
    -- Call and chunk counts
    calls_analyzed  INTEGER DEFAULT 0,
    chunks_analyzed INTEGER DEFAULT 0,
    
    -- Average scores (1-10 scale)
    avg_professionalism DECIMAL(3,1),
    avg_compliance DECIMAL(3,1),
    avg_customer_service DECIMAL(3,1),
    avg_protocol DECIMAL(3,1),
    avg_overall DECIMAL(3,1),
    
    -- Score distributions (for histograms)
    -- Format: {"1": 0, "2": 1, "3": 5, ..., "10": 12}
    score_distribution JSONB,
    
    -- Issues
    compliance_issues_count INTEGER DEFAULT 0,
    review_required_count INTEGER DEFAULT 0,
    
    -- Trend vs previous period
    trend_vs_previous VARCHAR(10) CHECK (trend_vs_previous IN ('up', 'down', 'stable', 'new')),
    trend_percentage DECIMAL(5,2),  -- e.g., +5.2% or -3.1%
    
    calculated_at   TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(agent_id, period_type, period_start, period_end)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_agent_qa_agent ON agent_qa_metrics(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_qa_period ON agent_qa_metrics(period_type, period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_agent_qa_overall ON agent_qa_metrics(avg_overall DESC);

-- ============================================================================
-- VIEWS - Convenience queries for QA analysis
-- ============================================================================

-- Chunks requiring human review
CREATE OR REPLACE VIEW qa_chunks_pending_review AS
SELECT 
    qc.*,
    c.agent_id,
    c.member_id,
    c.call_started_at
FROM call_qa_chunks qc
JOIN calls c ON c.id = qc.call_id
WHERE qc.requires_review = TRUE
  AND qc.reviewed_at IS NULL
ORDER BY qc.analyzed_at DESC;

-- Agent leaderboard (last 7 days)
CREATE OR REPLACE VIEW agent_qa_leaderboard AS
SELECT 
    c.agent_id,
    COUNT(DISTINCT c.id) as calls_analyzed,
    COUNT(qc.id) as chunks_analyzed,
    ROUND(AVG(qc.score_overall)::numeric, 1) as avg_overall,
    ROUND(AVG(qc.score_professionalism)::numeric, 1) as avg_professionalism,
    ROUND(AVG(qc.score_compliance)::numeric, 1) as avg_compliance,
    ROUND(AVG(qc.score_customer_service)::numeric, 1) as avg_customer_service,
    ROUND(AVG(qc.score_protocol)::numeric, 1) as avg_protocol,
    COUNT(*) FILTER (WHERE qc.requires_review) as review_count
FROM calls c
JOIN call_qa_chunks qc ON qc.call_id = c.id
WHERE c.call_started_at > NOW() - INTERVAL '7 days'
  AND c.agent_id IS NOT NULL
GROUP BY c.agent_id
ORDER BY avg_overall DESC;

-- Daily QA summary
CREATE OR REPLACE VIEW daily_qa_summary AS
SELECT 
    DATE(qc.analyzed_at) as analysis_date,
    COUNT(DISTINCT qc.call_id) as calls_processed,
    COUNT(qc.id) as chunks_processed,
    ROUND(AVG(qc.score_overall)::numeric, 1) as avg_score,
    COUNT(*) FILTER (WHERE qc.requires_review) as flagged_count,
    ROUND(AVG(qc.processing_time_ms)::numeric, 0) as avg_processing_ms
FROM call_qa_chunks qc
WHERE qc.analyzed_at > NOW() - INTERVAL '30 days'
GROUP BY DATE(qc.analyzed_at)
ORDER BY analysis_date DESC;

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE call_qa_chunks IS 'Stores chunked transcript QA analysis from Gemma AI';
COMMENT ON TABLE agent_qa_metrics IS 'Aggregated QA metrics per agent over time periods';
COMMENT ON VIEW qa_chunks_pending_review IS 'Chunks flagged for human review that have not been reviewed';
COMMENT ON VIEW agent_qa_leaderboard IS 'Agent ranking by QA scores over last 7 days';
COMMENT ON VIEW daily_qa_summary IS 'Daily aggregation of QA processing metrics';
