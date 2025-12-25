#!/bin/bash
# Run this script ON THE NEMO SERVER to enable SSH access from the laptop
# Execute: bash ~/Desktop/Nemo_Server/setup_laptop_ssh.sh

PUBLIC_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKUpFufZC/YJGXDULVEc5BlhsuGH1mdvkmKHVvfV4lSV laptop-to-nemo-server"

echo "Adding laptop's SSH key to authorized_keys..."

# Create .ssh dir if needed
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Check if key already exists
if grep -q "laptop-to-nemo-server" ~/.ssh/authorized_keys 2>/dev/null; then
    echo "✅ Key already installed!"
else
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    echo "✅ SSH key added successfully!"
fi

echo ""
echo "Testing SSH service..."
systemctl status ssh | head -5 || service ssh status | head -5

echo ""
echo "The laptop should now be able to SSH to this server!"
echo "Test from laptop: ssh $(whoami)@$(hostname -I | awk '{print $1}')"
