#!/bin/bash
# =============================================================================
# K3s Multi-Node Cluster Setup - Master Node (Main PC)
# =============================================================================
# This script installs K3s server on your main PC with Tailscale networking
# Run this FIRST before running the worker setup on your laptop
# =============================================================================

set -e

echo "=========================================="
echo " K3s Master Node Setup"
echo "=========================================="

# Check for Tailscale
if ! command -v tailscale &> /dev/null; then
    echo "❌ Tailscale is not installed. Please install it first:"
    echo "   curl -fsSL https://tailscale.com/install.sh | sh"
    exit 1
fi

# Get Tailscale IP
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null)
if [ -z "$TAILSCALE_IP" ]; then
    echo "❌ Could not get Tailscale IP. Make sure Tailscale is connected:"
    echo "   sudo tailscale up"
    exit 1
fi
echo "✅ Tailscale IP: $TAILSCALE_IP"

# Check for swap
SWAP_ON=$(swapon --show 2>/dev/null | wc -l)
if [ "$SWAP_ON" -gt 0 ]; then
    echo "⚠️  Swap is enabled. Disabling for K3s..."
    sudo swapoff -a
    sudo sed -i '/ swap / s/^/#/' /etc/fstab
    echo "✅ Swap disabled"
else
    echo "✅ Swap already disabled"
fi

# Install K3s server with Tailscale networking
echo ""
echo "📦 Installing K3s server..."
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="\
    --node-ip=$TAILSCALE_IP \
    --flannel-iface=tailscale0 \
    --tls-san=$TAILSCALE_IP \
    --write-kubeconfig-mode=644 \
    --disable=traefik \
    " sh -

# Wait for K3s to be ready
echo ""
echo "⏳ Waiting for K3s to be ready..."
sleep 10

# Verify installation
if sudo k3s kubectl get nodes &>/dev/null; then
    echo "✅ K3s server installed successfully!"
else
    echo "❌ K3s installation may have failed. Check: sudo systemctl status k3s"
    exit 1
fi

# Get node token for worker nodes
echo ""
echo "=========================================="
echo " WORKER NODE SETUP INFO"
echo "=========================================="
NODE_TOKEN=$(sudo cat /var/lib/rancher/k3s/server/node-token)
echo ""
echo "Run this on your laptop (worker node):"
echo ""
echo "curl -sfL https://get.k3s.io | K3S_URL=https://$TAILSCALE_IP:6443 \\"
echo "    K3S_TOKEN=$NODE_TOKEN \\"
echo "    INSTALL_K3S_EXEC=\"--node-ip=\$(tailscale ip -4) --flannel-iface=tailscale0\" \\"
echo "    sh -"
echo ""
echo "=========================================="

# Set up kubectl alias
if ! grep -q "export KUBECONFIG" ~/.bashrc; then
    echo 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml' >> ~/.bashrc
    echo "✅ Added KUBECONFIG to ~/.bashrc"
fi

# Create symlink for nemo-server if it doesn't exist
if [ ! -L "/opt/nemo-server" ]; then
    NEMO_PATH="$HOME/Desktop/Nemo_Server"
    if [ -d "$NEMO_PATH" ]; then
        echo ""
        echo "📁 Creating /opt/nemo-server symlink..."
        sudo ln -sf "$NEMO_PATH" /opt/nemo-server
        echo "✅ Symlink created: /opt/nemo-server -> $NEMO_PATH"
    fi
fi

echo ""
echo "🎉 Master node setup complete!"
echo ""
echo "Next steps:"
echo "1. Run the worker setup command above on your laptop"
echo "2. Verify with: sudo k3s kubectl get nodes"
echo "3. Deploy Nemo: sudo k3s kubectl apply -f /opt/nemo-server/k8s/base/"
