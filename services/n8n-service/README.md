# n8n Integration Service

Microservice for integrating n8n automation workflows with the Nemo transcription pipeline.

## Features

- **Voice Command Detection**: Pattern-based matching for triggering smart home commands
- **Emotion Alert Tracking**: Rolling window tracking for consecutive emotion patterns
- **Extensible Registry**: Easy to add new command patterns and triggers
- **Webhook-based**: Communicates with n8n via HTTP webhooks

## Architecture

This service receives enriched transcript segments from the Transcription Service and:
1. Checks text against registered voice command patterns
2. Tracks emotions per speaker for alert thresholds
3. Fires webhooks to n8n when triggers are met

## Authentication

Requires Service-to-Service (S2S) JWT authentication via the `X-Service-Token` header when accessed through the API Gateway.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/process` | POST | Process transcript segments for triggers |
| `/commands` | GET | List registered voice commands |
| `/commands` | POST | Register a new voice command |
| `/alerts/status` | GET | Get current emotion tracking status |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `N8N_WEBHOOK_BASE_URL` | `http://n8n:5678/webhook` | n8n webhook base URL |
| `EMOTION_ALERT_THRESHOLD` | `20` | Consecutive emotions before alert |
| `EMOTION_ALERT_SPEAKERS` | `pruitt,ericah` | Speakers to track |

## Running

```bash
cd services/n8n-service
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 8011
```

---

## Adding Custom Voice Commands

### Via API
```bash
curl -X POST http://localhost:8011/commands \
  -H "Content-Type: application/json" \
  -d '{
    "pattern": "turn on the lights",
    "webhook_path": "/smart-home/lights-on",
    "description": "Activates living room lights"
  }'
```

### Via Code
Edit `src/commands/registry.py`:
```python
VOICE_COMMANDS = {
    "turn on the lights": {
        "webhook": "/smart-home/lights-on",
        "description": "Activates living room lights"
    },
    # Add your command here
    "activate security": {
        "webhook": "/security/arm",
        "description": "Arms the security system"
    }
}
```

---

## Setting Up n8n Webhooks

1. **Create workflow in n8n** with a Webhook trigger node
2. **Configure webhook path** (e.g., `/smart-home/lights-on`)
3. **Register command** in this service pointing to that path
4. **Test** by speaking the trigger phrase during transcription

