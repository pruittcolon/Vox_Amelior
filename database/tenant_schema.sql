-- Tenant Schema for Multi-Tenant SaaS Architecture
-- Following 2024 best practices: shared database with tenant_id + RLS
-- 
-- Usage: Run this migration to add multi-tenancy support
-- psql -U postgres -d nemo_db -f database/tenant_schema.sql

BEGIN;

-- =============================================================================
-- TENANT TABLES
-- =============================================================================

-- Core tenants table
CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'pending', 'deleted')),
    tier VARCHAR(20) DEFAULT 'free' CHECK (tier IN ('free', 'starter', 'professional', 'enterprise')),
    settings JSONB DEFAULT '{}',
    owner_id UUID,
    billing_email VARCHAR(255),
    stripe_customer_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

-- Index for slug lookups (used in tenant context extraction)
CREATE INDEX IF NOT EXISTS idx_tenants_slug ON tenants(slug);
CREATE INDEX IF NOT EXISTS idx_tenants_status ON tenants(status) WHERE status = 'active';

-- =============================================================================
-- USER-TENANT RELATIONSHIP
-- =============================================================================

-- Add tenant_id to users table (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'tenant_id'
    ) THEN
        ALTER TABLE users ADD COLUMN tenant_id UUID REFERENCES tenants(id);
        CREATE INDEX idx_users_tenant ON users(tenant_id);
    END IF;
END $$;

-- Tenant memberships for users who belong to multiple tenants
CREATE TABLE IF NOT EXISTS tenant_memberships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    role VARCHAR(50) DEFAULT 'user' CHECK (role IN ('admin', 'manager', 'user', 'viewer')),
    is_primary BOOLEAN DEFAULT FALSE,
    permissions JSONB DEFAULT '[]',
    invited_by UUID,
    accepted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, tenant_id)
);

CREATE INDEX IF NOT EXISTS idx_tenant_memberships_user ON tenant_memberships(user_id);
CREATE INDEX IF NOT EXISTS idx_tenant_memberships_tenant ON tenant_memberships(tenant_id);

-- =============================================================================
-- SSO/IDENTITY PROVIDER CONFIG
-- =============================================================================

CREATE TABLE IF NOT EXISTS identity_providers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    provider_type VARCHAR(50) NOT NULL CHECK (provider_type IN ('oidc', 'saml', 'google', 'azure_ad', 'okta')),
    client_id VARCHAR(255),
    client_secret_encrypted TEXT,
    metadata_url TEXT,
    issuer VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ,
    UNIQUE(tenant_id, name)
);

CREATE INDEX IF NOT EXISTS idx_identity_providers_tenant ON identity_providers(tenant_id);

-- =============================================================================
-- SCIM PROVISIONING
-- =============================================================================

CREATE TABLE IF NOT EXISTS scim_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    description VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scim_tokens_tenant ON scim_tokens(tenant_id);
CREATE INDEX IF NOT EXISTS idx_scim_tokens_hash ON scim_tokens(token_hash);

-- SCIM sync log for audit
CREATE TABLE IF NOT EXISTS scim_sync_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    operation VARCHAR(20) NOT NULL CHECK (operation IN ('create', 'update', 'delete', 'activate', 'deactivate')),
    resource_type VARCHAR(20) NOT NULL CHECK (resource_type IN ('User', 'Group')),
    resource_id VARCHAR(255) NOT NULL,
    external_id VARCHAR(255),
    request_body JSONB,
    response_code INTEGER,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scim_sync_log_tenant ON scim_sync_log(tenant_id);
CREATE INDEX IF NOT EXISTS idx_scim_sync_log_time ON scim_sync_log(created_at DESC);

-- =============================================================================
-- ROW LEVEL SECURITY (RLS)
-- =============================================================================

-- Enable RLS on tenant-scoped tables
ALTER TABLE tenant_memberships ENABLE ROW LEVEL SECURITY;
ALTER TABLE identity_providers ENABLE ROW LEVEL SECURITY;
ALTER TABLE scim_tokens ENABLE ROW LEVEL SECURITY;

-- RLS Policies: Restrict access based on current_setting('app.current_tenant')
-- These are enforced when using set_config('app.current_tenant', tenant_id, true)

CREATE POLICY tenant_memberships_isolation ON tenant_memberships
    USING (tenant_id::TEXT = current_setting('app.current_tenant', true));

CREATE POLICY identity_providers_isolation ON identity_providers
    USING (tenant_id::TEXT = current_setting('app.current_tenant', true));

CREATE POLICY scim_tokens_isolation ON scim_tokens
    USING (tenant_id::TEXT = current_setting('app.current_tenant', true));

-- =============================================================================
-- AUDIT TRIGGER
-- =============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_tenants_updated_at
    BEFORE UPDATE ON tenants
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trigger_identity_providers_updated_at
    BEFORE UPDATE ON identity_providers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- DEFAULT DATA
-- =============================================================================

-- Insert default tenant for development
INSERT INTO tenants (id, name, slug, tier, settings)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'Default Tenant',
    'default',
    'enterprise',
    '{"max_users": 100, "enable_sso": true, "enable_scim": true}'
)
ON CONFLICT (slug) DO NOTHING;

COMMIT;
