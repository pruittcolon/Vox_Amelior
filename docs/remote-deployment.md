# Nemo Server - Remote Deployment Guide

Deploy Nemo Server on a secondary laptop (no NVIDIA GPU required) and access it from your main PC over WiFi.

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| Storage | 10 GB | 20 GB |
| CPU | Dual-core | Quad-core |
| GPU | None required | Intel/AMD integrated OK |
| Network | WiFi | Same network as main PC |

## Quick Start

### On the Remote Laptop (Server)

1. **Clone or copy the repository:**
   ```bash
   git clone <your-repo-url> ~/Nemo_Server
   cd ~/Nemo_Server
   ```

2. **Install Docker:**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install docker.io docker-compose python3
   sudo usermod -aG docker $USER
   # Log out and back in for group changes
   ```

3. **Start the server:**
   ```bash
   ./nemo --remote     # or: ./scripts/start-remote.sh
   ```

4. **Note the displayed IP address** (e.g., `http://192.168.1.50:8000`)

### On Your Main PC (Client)

1. **Open your browser and navigate to:**
   ```
   http://<remote-laptop-ip>:8000/ui/login.html
   ```

2. **Login with demo credentials:**
   - Username: `admin`
   - Password: `admin123`

## What's Running

### Services Running (CPU-only)

| Service | Port | Description |
|---------|------|-------------|
| API Gateway | 8000 | Main entry point, serves UI |
| RAG Service | 8004 | Vector search & memory |
| Emotion Service | 8005 | Sentiment analysis |
| Insights Service | 8010 | Analytics dashboard |
| ML Service | 8006 | Machine learning engines |
| Fiserv Service | 8015 | Banking integrations |
| N8N Service | 8011 | Automation webhooks |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Cache & queues |

### Services NOT Running (Require NVIDIA GPU)

| Service | Reason |
|---------|--------|
| Gemma Service | Requires CUDA GPU for LLM inference |
| Transcription Service | Requires CUDA for speech-to-text |

## Commands

```bash
# Start services
./nemo --remote          # or: ./scripts/start-remote.sh

# Start with image rebuild
./nemo --remote --build  # or: ./scripts/start-remote.sh --build

# Stop all services
./nemo --stop            # or: ./scripts/start-stop.sh

# View logs
cd docker && docker compose -f docker-compose.remote.yml logs -f

# Check service status
cd docker && docker compose -f docker-compose.remote.yml ps
```

## Troubleshooting

### Can't connect from main PC

1. Check the remote laptop's IP:
   ```bash
   ip addr show | grep "inet "
   ```

2. Ensure firewall allows port 8000:
   ```bash
   sudo ufw allow 8000/tcp
   ```

3. Verify services are running:
   ```bash
   curl http://localhost:8000/health
   ```

### Services won't start

1. Check Docker is running:
   ```bash
   sudo systemctl start docker
   ```

2. Check logs:
   ```bash
   cd docker && docker compose -f docker-compose.remote.yml logs
   ```

### Low on storage

Prune unused Docker data:
```bash
docker system prune -a
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR MAIN PC (Client)                        │
│                                                                 │
│   Browser → http://192.168.x.x:8000/ui/index.html              │
│                        ↓ WiFi                                   │
└────────────────────────│────────────────────────────────────────┘
                         │
                    [WiFi Network]
                         │
┌────────────────────────│────────────────────────────────────────┐
│               REMOTE LAPTOP (Server)                            │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ docker-compose.remote.yml                               │  │
│   │   • API Gateway (0.0.0.0:8000) ← Network accessible     │  │
│   │   • RAG Service (internal)                              │  │
│   │   • Emotion Service (internal)                          │  │
│   │   • ML Service (internal)                               │  │
│   │   • Fiserv Service (internal)                           │  │
│   │   • PostgreSQL, Redis                                   │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Gemma and Transcription NOT available                     │
└─────────────────────────────────────────────────────────────────┘
```

## Security Notes

- The gateway binds to `0.0.0.0:8000` making it accessible on your local network
- Redis and PostgreSQL remain bound to `127.0.0.1` (local-only)
- Demo users are enabled by default - **change passwords for production**
- Consider using a VPN (WireGuard) for remote access outside your home network
