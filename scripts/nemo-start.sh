#!/bin/bash
# Nemo Services Start Script
# Starts all Nemo Docker services

cd /home/pruittcolon/Desktop/Nemo_Server/docker

# Start all docker compose services
docker compose up -d 2>/dev/null

# Send desktop notification
notify-send -i emblem-ok-symbolic "Nemo Services" "All services starting..." 2>/dev/null || true

echo "All Nemo services started."
