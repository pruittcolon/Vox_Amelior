#!/bin/bash
# Nemo Services Stop Script
# Stops all Nemo Docker services

cd /home/pruittcolon/Desktop/Nemo_Server/docker

# Stop all docker compose services
docker compose down 2>/dev/null

# Send desktop notification
notify-send -i process-stop "Nemo Services" "All services stopped" 2>/dev/null || true

echo "All Nemo services stopped."
