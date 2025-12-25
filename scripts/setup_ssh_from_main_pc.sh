#!/bin/bash
# Run this on the remote laptop to authorize SSH access from main PC
# Created by Antigravity - 2025-12-20

echo "Adding SSH key to authorized_keys..."

mkdir -p ~/.ssh
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIE2Tw+NxkVCs66d+YTN8y3fQ0HLL9c6PGSl6jOjey7L1 pruittcolon@github" > ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys

echo ""
echo "✅ SSH key authorized!"
echo ""
echo "Verifying..."
cat ~/.ssh/authorized_keys
ls -la ~/.ssh/

echo ""
echo "Now try SSH from the main PC!"
