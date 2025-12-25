#!/bin/bash

# NFS Setup Script for Nemo Server
# Run this script with sudo AFTER installing nfs-kernel-server

SHARED_DIR="/home/pruitt-colon/Desktop/Nemo_Server/shared"
EXPORT_ENTRY="$SHARED_DIR *(rw,sync,no_subtree_check)"

echo "Configuring NFS Share for $SHARED_DIR..."

# 1. Ensure the directory exists and has correct permissions
if [ ! -d "$SHARED_DIR" ]; then
    echo "Creating shared directory..."
    mkdir -p "$SHARED_DIR"
fi

# Set permissions so anyone can read/write (since it's a shared folder)
# You might want to restrict this in a real production env, but for home share 777 is common.
chmod 777 "$SHARED_DIR"

# 2. Add to /etc/exports if not already there
if grep -q "$SHARED_DIR" /etc/exports; then
    echo "Entry already exists in /etc/exports"
else
    echo "Adding entry to /etc/exports..."
    echo "$EXPORT_ENTRY" | tee -a /etc/exports
fi

# 3. Restart NFS Service
echo "Restarting NFS Kernel Server..."
systemctl restart nfs-kernel-server

if [ $? -eq 0 ]; then
    echo "NFS Server restarted successfully."
    echo "Current Exports:"
    exportfs -v
else
    echo "Failed to restart NFS server. Did you install it first? (sudo apt install nfs-kernel-server)"
    exit 1
fi
