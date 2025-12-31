#!/bin/bash
# ============================================================================
# Push Secrets to Remote Windows Machine via Tailscale
# ============================================================================
# Run this from your Linux machine AFTER your brother's Windows PC is on Tailscale
#
# Usage: ./push_secrets_to_windows.sh <TAILSCALE_IP>
# Example: ./push_secrets_to_windows.sh 100.x.x.x
# ============================================================================

set -e

REMOTE_IP="${1:-}"
SECRETS_DIR="$(dirname "$0")/../docker/secrets"
REMOTE_USER="${2:-$USER}"  # Default to current username

if [ -z "$REMOTE_IP" ]; then
    echo "============================================================================"
    echo "PUSH SECRETS TO REMOTE WINDOWS MACHINE"
    echo "============================================================================"
    echo ""
    echo "Usage: $0 <TAILSCALE_IP> [WINDOWS_USERNAME]"
    echo ""
    echo "Examples:"
    echo "  $0 100.123.45.67"
    echo "  $0 100.123.45.67 BrotherName"
    echo ""
    echo "Get the Tailscale IP from your brother's Windows machine:"
    echo "  In PowerShell: tailscale ip -4"
    echo ""
    exit 1
fi

echo "============================================================================"
echo "PUSHING SECRETS TO REMOTE WINDOWS MACHINE"
echo "============================================================================"
echo ""
echo "Remote IP: $REMOTE_IP"
echo "Username:  $REMOTE_USER"
echo "Secrets:   $SECRETS_DIR"
echo ""

# Verify secrets directory exists
if [ ! -d "$SECRETS_DIR" ]; then
    echo "ERROR: Secrets directory not found: $SECRETS_DIR"
    exit 1
fi

# Count secrets
SECRET_COUNT=$(ls -1 "$SECRETS_DIR" 2>/dev/null | wc -l)
echo "Found $SECRET_COUNT secret files to copy."
echo ""

# Destination path on Windows (PowerShell-compatible)
WINDOWS_DEST="C:/Users/$REMOTE_USER/Desktop/Nemo_Server/docker/secrets/"

echo "Copying secrets to Windows machine..."
echo "Destination: $WINDOWS_DEST"
echo ""

# Use SCP to copy secrets
# Note: Windows OpenSSH uses forward slashes or escaped backslashes
scp -r "$SECRETS_DIR"/* "${REMOTE_USER}@${REMOTE_IP}:${WINDOWS_DEST}"

if [ $? -eq 0 ]; then
    echo ""
    echo "SUCCESS: Secrets copied to Windows machine!"
    echo ""
    echo "Next steps for your brother:"
    echo "  1. Open PowerShell as Administrator"
    echo "  2. Navigate to: cd Desktop\\Nemo_Server\\docker"
    echo "  3. Run: docker compose -f docker-compose.remote.yml up -d"
    echo ""
    echo "Then access Nemo Server at: http://$REMOTE_IP:8000"
else
    echo ""
    echo "ERROR: Failed to copy secrets."
    echo ""
    echo "Troubleshooting:"
    echo "  1. Verify Windows OpenSSH is enabled:"
    echo "     Settings > Apps > Optional Features > OpenSSH Server"
    echo "  2. Verify Tailscale is connected on both machines"
    echo "  3. Try: ssh $REMOTE_USER@$REMOTE_IP"
fi
