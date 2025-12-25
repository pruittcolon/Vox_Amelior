#!/bin/bash

# Configuration
SERVER_IP="100.68.213.84" # Nemo Server Tailscale IP (Local Machine)
SHARE_PATH="/home/pruittcolon/Desktop/Nemo_Server"
MOUNT_POINT="$HOME/Desktop/Nemo_Share"

# Ensure nfs-common is installed
if ! command -v mount.nfs &> /dev/null; then
    echo "Installing nfs-common (password may be required)..."
    if command -v apt-get &> /dev/null; then
        echo "installing nfs-common"
        # sudo apt-get update && sudo apt-get install -y nfs-common
        # Commenting out auto-install as it might hang on password. 
        # Better to warn user.
        echo "⚠️  nfs-common not found. Please run 'sudo apt install nfs-common' manually if mount fails."
    fi
fi

# Create mount point
mkdir -p "$MOUNT_POINT"

# Check if already mounted
if mount | grep -q "$MOUNT_POINT"; then
    echo "✅ Already mounted."
else
    echo "Mounting $SERVER_IP:$SHARE_PATH to $MOUNT_POINT..."
    sudo mount -t nfs "$SERVER_IP:$SHARE_PATH" "$MOUNT_POINT"
fi

if mount | grep -q "$MOUNT_POINT"; then
    echo "✅ Connection verified!"
    echo "Files available at: $MOUNT_POINT"
else
    echo "❌ Mount failed."
    exit 1
fi
