# n8n Workflow Integration Guide

This guide explains how to set up n8n workflows to receive events from the Nemo transcription pipeline.

## Overview

The n8n Integration Service sends webhooks to n8n when:
1. **Voice Commands** are detected (e.g., "Honey, can you turn off the lights")
2. **Emotion Alerts** fire (20 consecutive angry emotions from Pruitt/Ericah)

## Setup

### 1. Run n8n

Add n8n to your docker-compose or run standalone:

```bash
docker run -d --name n8n -p 5678:5678 \
  -v n8n_data:/home/node/.n8n \
  --network nemo_network \
  n8nio/n8n
```

### 2. Configure Webhook URL

Set the n8n webhook URL in your environment:

```bash
export N8N_WEBHOOK_BASE_URL=http://n8n:5678/webhook
```

---

## Voice Command Workflow

### Webhook Payload

When a voice command is detected, n8n receives:

```json
{
  "event_type": "voice_command",
  "command_id": "lights_off",
  "n8n_action": "alexa_voice_monkey_lights_off",
  "speaker": "pruitt",
  "original_text": "Honey, can you turn off the lights please",
  "timestamp": "2025-12-08T17:52:44.000Z",
  "metadata": {
    "job_id": "abc-123",
    "session_id": "default",
    "start_time": 0.0,
    "end_time": 2.5
  }
}
```

### n8n Workflow: Alexa Voice Monkey

1. Create a new workflow in n8n
2. Add a **Webhook** trigger node:
   - HTTP Method: POST
   - Path: `/voice-command`
3. Add an **HTTP Request** node:
   - Method: POST
   - URL: `https://api.voicemonkey.io/trigger`
   - Body (JSON):

```json
{
  "token": "YOUR_VOICE_MONKEY_TOKEN",
  "device": "YOUR_DEVICE_ID",
  "preset": "{{ $json.n8n_action }}"
}
```

### Alexa Voice Monkey Curl Example

Direct curl command (for reference):

```bash
curl -X POST "https://api.voicemonkey.io/trigger" \
  -H "Content-Type: application/json" \
  -d '{
    "token": "YOUR_VOICE_MONKEY_TOKEN",
    "device": "YOUR_DEVICE_ID",
    "preset": "lights_off"
  }'
```

---

## Emotion Alert Workflow

### Webhook Payload

When 20 consecutive angry emotions are detected:

```json
{
  "event_type": "emotion_alert",
  "speaker": "pruitt",
  "emotion": "anger",
  "consecutive_count": 20,
  "timestamp": "2025-12-08T17:52:44.000Z",
  "metadata": {
    "job_id": "abc-123",
    "session_id": "default"
  }
}
```

### n8n Workflow: Database Alert

1. Create a new workflow in n8n
2. Add a **Webhook** trigger node:
   - HTTP Method: POST
   - Path: `/emotion-alert`
3. Add a **Postgres** or **MySQL** node to insert the alert:

```sql
INSERT INTO emotion_alerts (speaker, emotion, consecutive_count, timestamp)
VALUES ('{{ $json.speaker }}', '{{ $json.emotion }}', {{ $json.consecutive_count }}, '{{ $json.timestamp }}')
```

---

## Adding New Voice Commands

### Via API

Register new commands at runtime:

```bash
curl -X POST "http://localhost:8011/commands" \
  -H "Content-Type: application/json" \
  -d '{
    "command_id": "lights_on",
    "pattern": "honey.*turn.*on.*light",
    "description": "Turn on lights via Alexa",
    "n8n_action": "alexa_voice_monkey_lights_on"
  }'
```

### In Code

Edit `services/n8n-service/src/command_registry.py`:

```python
def _load_default_commands(self):
    self.register(
        command_id="lights_off",
        pattern=r"honey.*(?:can|could|would).*turn.*off.*(?:the\s+)?light",
        description="Turn off lights via Alexa Voice Monkey",
        n8n_action="alexa_voice_monkey_lights_off"
    )
    # Add your new command here
    self.register(
        command_id="your_command",
        pattern=r"your.*regex.*pattern",
        description="Your command description",
        n8n_action="your_n8n_action"
    )
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /health` | GET | Health check |
| `POST /process` | POST | Process transcript segments |
| `GET /commands` | GET | List all voice commands |
| `POST /commands` | POST | Register new command |
| `DELETE /commands/{id}` | DELETE | Delete a command |
| `GET /alerts/status` | GET | Emotion tracking status |
| `GET /alerts/history` | GET | Alert history |
