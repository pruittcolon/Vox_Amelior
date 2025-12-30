# Multi-Computer Development Architecture

# Multi-Computer Development Architecture

**Use Case: Active Development Workflow** - Use this guide if you are developing on a powerful main PC but running the backend/docker containers on a secondary laptop (e.g. to offload compute or manage resources).

This document explains how Nemo Server operates across two computers with WireGuard VPN and VNC for a seamless development experience.

## Overview

**Connection Stack:**
- **WireGuard VPN** - Secure tunnel between Main PC and Remote Laptop
- **VNC Server** - Visual remote access to laptop for Docker management
- **Shared Folder** - Code synced between machines

**Network:**
- Main PC: `<YOUR_LAN_IP>` (LAN), `10.100.0.1` (WireGuard)
- Remote Laptop: WireGuard peer, accessible via VNC

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MAIN PC (Development)                               │
│                                                                             │
│  ┌──────────────────┐    ┌──────────────────────────────────────────────┐  │
│  │   VS Code +      │    │        Shared Folder                         │  │
│  │   Antigravity    │◄──►│  ~/Desktop/Nemo_Server                       │  │
│  │   (This Agent)   │    │                    ↕                         │  │
│  └──────────────────┘    │        (Synced via Samba/NFS/Syncthing)      │  │
│                          └──────────────────────────────────────────────┘  │
│                                          │                                  │
│  Browser → http://localhost:8000         │  (WiFi Network)                  │
│                                          ↓                                  │
└────────────────────────────────────────────────────────────────────────────┘
                                           │
                                    [WiFi/Tailscale]
                                           │
┌────────────────────────────────────────────────────────────────────────────┐
│                      REMOTE LAPTOP (Docker Runtime)                        │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │        Shared Folder (Same as Main PC)                               │  │
│  │  ~/Nemo_Server or /mnt/shared/Nemo_Server                            │  │
│  │                    ↓                                                 │  │
│  │  docker-compose.remote.yml                                           │  │
│  │    • API Gateway (0.0.0.0:8000) ← Network accessible                 │  │
│  │    • RAG, ML, Emotion, Fiserv, Insights                              │  │
│  │    • PostgreSQL, Redis                                               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  NO GPU Services Available (Gemma, Transcription)                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## How Code Changes Propagate

1. **Agent edits code** on Main PC → File saved to shared folder  
2. **Shared folder syncs** to Remote Laptop (instant for NFS/Samba, seconds for Syncthing)  
3. **Docker must be restarted** for Python changes to take effect:
   ```bash
   # On remote laptop
   docker-compose -f docker/docker-compose.remote.yml restart api-gateway
   ```

## Common Issues & Solutions

### Issue: Changes not taking effect

**Cause**: Docker caches Python bytecode. Simply syncing files doesn't restart services.

**Solution**: Restart the affected service on the remote laptop:
```bash
ssh user@remote-laptop "cd ~/Nemo_Server && docker-compose -f docker/docker-compose.remote.yml restart api-gateway"
```

### Issue: Import errors (404s, router not loading)

**Cause**: Bad Python imports often show as "router not available" in logs.

**Solution**: 
1. Check API Gateway logs: `docker logs nemo_api_gateway --tail 100`
2. Look for `ImportError` or `Router not available`
3. Fix the import, sync, restart container

### Issue: WebSocket connection refused

**Cause**: WebSocket endpoints may not be registered if their parent router fails to load.

**Solution**: Check if the parent router loaded. Example: if `/api/v1/salesforce/stream` fails but `/api/v1/salesforce/accounts` returns 404, the entire salesforce router failed to import.

## Network Configuration

### Option A: Direct WiFi Connection
- Both computers on same WiFi network
- Access via IP: `http://192.168.x.x:8000`

### Option B: Tailscale (Recommended)
- Works across different networks
- Access via Tailscale IP: `http://100.x.x.x:8000`
- Setup: Install Tailscale on both machines, `tailscale up`

## Developer Workflow

```
1. Edit code (VS Code/Antigravity on Main PC)
       ↓
2. Save file (auto-syncs to remote)
       ↓
3. Restart container if needed (Python changes)
       ↓
4. Refresh browser to test
```

## Service Restart Commands

SSH to remote laptop, then:

```bash
# Restart API Gateway (most common)
docker-compose -f docker/docker-compose.remote.yml restart api-gateway

# Restart all services
docker-compose -f docker/docker-compose.remote.yml restart

# Rebuild and restart (for Dockerfile changes)
docker-compose -f docker/docker-compose.remote.yml up -d --build api-gateway

# View logs
docker-compose -f docker/docker-compose.remote.yml logs -f api-gateway
```

## Remote Laptop Details

| Component | Details |
|-----------|---------|
| IP Address | Update with your remote laptop IP |
| SSH Command | `ssh <user>@<remote-laptop-ip>` |
| Shared Folder Path | `~/Nemo_Server` or update with your path |
| Docker Compose File | `docker/docker-compose.remote.yml` |

## Changelog

- **2025-12-20**: Created initial documentation after fixing Salesforce router import issue
