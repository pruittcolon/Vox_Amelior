#!/bin/bash
# Nemo Server - Secrets Setup Script
#
# This script generates and manages secrets for the Nemo Server.
# It follows the principle of least surprise:
# - Existing secrets are NEVER overwritten (non-destructive)
# - All secrets are archived before any changes
# - File permissions are strictly enforced (600)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
SECRETS_DIR="$PROJECT_ROOT/docker/secrets"
ARCHIVE_DIR="$PROJECT_ROOT/archive/secrets_backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🔐 Nemo Server - Secrets Setup"
echo "================================"
echo ""

# Create directories if needed
mkdir -p "$SECRETS_DIR"
mkdir -p "$ARCHIVE_DIR"

# Function to generate a secret
generate_secret() {
    local name=$1
    local length=${2:-32}
    local file="$SECRETS_DIR/$name"
    
    if [ -f "$file" ] && [ -s "$file" ]; then
        echo -e "${YELLOW}⏭️  Skipping $name (already exists)${NC}"
        return 0
    fi
    
    echo -e "${GREEN}🔑 Generating $name...${NC}"
    openssl rand -base64 "$length" | tr -d '\n' > "$file"
    chmod 600 "$file"
}

# Function to generate a password (alphanumeric for wider compatibility)
generate_password() {
    local name=$1
    local length=${2:-24}
    local file="$SECRETS_DIR/$name"
    
    if [ -f "$file" ] && [ -s "$file" ]; then
        echo -e "${YELLOW}⏭️  Skipping $name (already exists)${NC}"
        return 0
    fi
    
    echo -e "${GREEN}🔑 Generating $name...${NC}"
    # Generate alphanumeric password (safer for database connections)
    openssl rand -base64 "$length" | tr -dc 'a-zA-Z0-9' | head -c "$length" > "$file"
    chmod 600 "$file"
}

# Function to set a static value
set_static() {
    local name=$1
    local value=$2
    local file="$SECRETS_DIR/$name"
    
    if [ -f "$file" ] && [ -s "$file" ]; then
        echo -e "${YELLOW}⏭️  Skipping $name (already exists)${NC}"
        return 0
    fi
    
    echo -e "${GREEN}📝 Setting $name...${NC}"
    echo -n "$value" > "$file"
    chmod 600 "$file"
}

# Archive existing secrets before proceeding
archive_existing() {
    if ls "$SECRETS_DIR"/* 1> /dev/null 2>&1; then
        echo "📦 Archiving existing secrets to $ARCHIVE_DIR/$TIMESTAMP/"
        mkdir -p "$ARCHIVE_DIR/$TIMESTAMP"
        for f in "$SECRETS_DIR"/*; do
            if [ -f "$f" ] && [ "$(basename "$f")" != "README.md" ] && [ "$(basename "$f")" != ".gitkeep" ]; then
                cp "$f" "$ARCHIVE_DIR/$TIMESTAMP/"
            fi
        done
        echo ""
    fi
}

# Main execution
echo "📁 Secrets directory: $SECRETS_DIR"
echo ""

# Archive first (non-destructive)
archive_existing

# Generate JWT secrets (32 bytes minimum for HMAC-SHA256)
generate_secret "jwt_secret_primary" 48
generate_secret "jwt_secret_previous" 48
generate_secret "jwt_secret" 48

# Generate service authentication key
generate_secret "service_api_key" 32

# Generate session encryption key (32 bytes for AES-256)
generate_secret "session_key" 32

# Generate database encryption keys
generate_secret "users_db_key" 32
generate_secret "rag_db_key" 32  
generate_secret "email_db_key" 32

# Generate database credentials
set_static "postgres_user" "nemo_admin"
generate_password "postgres_password" 24

# Generate Redis password
generate_password "redis_password" 24

# HuggingFace token (placeholder - user should set their own)
if [ ! -f "$SECRETS_DIR/huggingface_token" ] || [ ! -s "$SECRETS_DIR/huggingface_token" ]; then
    echo -e "${YELLOW}💡 huggingface_token not set - create manually if needed${NC}"
    touch "$SECRETS_DIR/huggingface_token"
    chmod 600 "$SECRETS_DIR/huggingface_token"
fi

# Set permissions on all secrets
echo ""
echo "🔒 Setting file permissions (600)..."
find "$SECRETS_DIR" -type f -not -name "README.md" -not -name ".gitkeep" -exec chmod 600 {} \;

# Verify
echo ""
echo "✅ Secret files created:"
echo "------------------------"
for f in "$SECRETS_DIR"/*; do
    if [ -f "$f" ] && [ "$(basename "$f")" != "README.md" ] && [ "$(basename "$f")" != ".gitkeep" ]; then
        name=$(basename "$f")
        size=$(wc -c < "$f")
        perms=$(stat -c "%a" "$f")
        if [ "$perms" = "600" ]; then
            status="${GREEN}OK${NC}"
        else
            status="${RED}WARN (perms: $perms)${NC}"
        fi
        printf "  %-25s %3d bytes  [%b]\n" "$name" "$size" "$status"
    fi
done

echo ""
echo "📋 Next steps:"
echo "  1. Set your HuggingFace token: echo 'hf_...' > docker/secrets/huggingface_token"
echo "  2. Run: python3 scripts/verify_security.py"
echo "  3. Start services: docker compose up -d"
echo ""
echo "⚠️  Remember: Never commit these files to Git!"
