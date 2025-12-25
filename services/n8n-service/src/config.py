"""
Configuration for n8n Integration Service
"""

import os

# n8n Webhook Configuration
N8N_WEBHOOK_BASE_URL = os.getenv("N8N_WEBHOOK_BASE_URL", "http://n8n:5678/webhook")
N8N_VOICE_COMMAND_PATH = os.getenv("N8N_VOICE_COMMAND_PATH", "/voice-command")
N8N_EMOTION_ALERT_PATH = os.getenv("N8N_EMOTION_ALERT_PATH", "/emotion-alert")

# Emotion Alert Configuration
EMOTION_ALERT_THRESHOLD = int(os.getenv("EMOTION_ALERT_THRESHOLD", "20"))
EMOTION_ALERT_SPEAKERS_RAW = os.getenv("EMOTION_ALERT_SPEAKERS", "pruitt,ericah")
EMOTION_ALERT_SPEAKERS: list[str] = [s.strip().lower() for s in EMOTION_ALERT_SPEAKERS_RAW.split(",") if s.strip()]
EMOTION_ALERT_TYPE = os.getenv("EMOTION_ALERT_TYPE", "anger")

# Service Configuration
SERVICE_PORT = int(os.getenv("N8N_SERVICE_PORT", "8011"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Redis for state persistence (optional)
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
ENABLE_REDIS_STATE = os.getenv("ENABLE_REDIS_STATE", "false").lower() == "true"

# Voice Monkey Direct Integration
VOICE_MONKEY_TOKEN = os.getenv("VOICE_MONKEY_TOKEN", "")
VOICE_MONKEY_DEVICE_ID = os.getenv("VOICE_MONKEY_DEVICE_ID", "")
VOICE_MONKEY_ENABLED = os.getenv("VOICE_MONKEY_ENABLED", "true").lower() == "true"
