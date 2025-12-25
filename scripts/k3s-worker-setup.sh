#!/bin/bash
# =============================================================================
# K3s Multi-Node Cluster Setup - Worker Node (Laptop)
# =============================================================================
# This script installs K3s agent on your laptop to join the cluster
# Run this AFTER the master setup script has completed on your main PC
#
# REQUIRED: Set these variables before running:
#   K3S_MASTER_IP   - Tailscale IP of your main PC (e.g., 100.68.213.84)
#   K3S_TOKEN       - Node token from master setup output
# =============================================================================

set -e

echo "=========================================="
echo " K3s Worker Node Setup"
echo "=========================================="

# Check if variables are set
if [ -z "$K3S_MASTER_IP" ]; then
    echo "❌ K3S_MASTER_IP not set."
    echo "   Export it first: export K3S_MASTER_IP=100.x.x.x"
    echo "   (Get this from 'tailscale ip -4' on your main PC)"
    exit 1
fi

if [ -z "$K3S_TOKEN" ]; then
    echo "❌ K3S_TOKEN not set."
    echo "   Export it first: export K3S_TOKEN=<token>"
    echo "   (Get this from 'sudo cat /var/lib/rancher/k3s/server/node-token' on main PC)"
    exit 1
fi

echo "Master IP: $K3S_MASTER_IP"
echo "Token: ${K3S_TOKEN:0:20}..."

# Check for Tailscale
if ! command -v tailscale &> /dev/null; then
    echo ""
    echo "📦 Installing Tailscale..."
    curl -fsSL https://tailscale.com/install.sh | sh
    echo ""
    echo "⚠️  Run 'sudo tailscale up' to connect, then re-run this script."
    exit 1
fi

# Get Tailscale IP
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null)
if [ -z "$TAILSCALE_IP" ]; then
    echo "❌ Could not get Tailscale IP. Connect first:"
    echo "   sudo tailscale up"
    exit 1
fi
echo "✅ Tailscale IP: $TAILSCALE_IP"

# Set hostname if still default
CURRENT_HOSTNAME=$(hostname)
if [ "$CURRENT_HOSTNAME" == "localhost" ] || [ "$CURRENT_HOSTNAME" == "ubuntu" ]; then
    echo ""
    echo "📝 Setting hostname to 'nemo-worker'..."
    sudo hostnamectl set-hostname nemo-worker
    echo "✅ Hostname set to: nemo-worker"
fi

# Disable swap
SWAP_ON=$(swapon --show 2>/dev/null | wc -l)
if [ "$SWAP_ON" -gt 0 ]; then
    echo ""
    echo "⚠️  Disabling swap for K3s..."
    sudo swapoff -a
    sudo sed -i '/ swap / s/^/#/' /etc/fstab
    echo "✅ Swap disabled"
else
    echo "✅ Swap already disabled"
fi

# Configure firewall if ufw is active
if command -v ufw &> /dev/null && sudo ufw status | grep -q "active"; then
    echo ""
    echo "🔥 Configuring firewall..."
    sudo ufw allow 6443/tcp    # K3s API
    sudo ufw allow 8472/udp    # Flannel VXLAN
    sudo ufw allow 10250/tcp   # Kubelet
    sudo ufw allow 51820/udp   # WireGuard
    sudo ufw reload
    echo "✅ Firewall configured"
fi

# Test connectivity to master
echo ""
echo "🔗 Testing connection to master..."
if curl -sk --connect-timeout 5 "https://$K3S_MASTER_IP:6443" &>/dev/null; then
    echo "✅ Master is reachable"
else
    echo "⚠️  Cannot reach master at $K3S_MASTER_IP:6443"
    echo "   Make sure K3s is running on the master and both devices are on Tailscale"
fi

# Install K3s agent
echo ""
echo "📦 Installing K3s agent..."
curl -sfL https://get.k3s.io | K3S_URL="https://$K3S_MASTER_IP:6443" \
    K3S_TOKEN="$K3S_TOKEN" \
    INSTALL_K3S_EXEC="--node-ip=$TAILSCALE_IP --flannel-iface=tailscale0" \
    sh -

# Verify
echo ""
echo "⏳ Waiting for agent to connect..."
sleep 5

if systemctl is-active --quiet k3s-agent; then
    echo "✅ K3s agent is running!"
else
    echo "⚠️  K3s agent may not have started. Check: sudo systemctl status k3s-agent"
fi

echo ""
echo "=========================================="
echo " Worker Node Setup Complete!"
echo "=========================================="
echo ""
echo "Verify on your main PC with:"
echo "  sudo k3s kubectl get nodes"
echo ""
echo "You should see this node listed as 'nemo-worker'"
