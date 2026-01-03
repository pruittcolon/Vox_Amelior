# Gmail Automation Service

Gmail API integration service for the Vox Amelior platform. Provides OAuth-based authentication to user Gmail accounts and Gemma-powered email analysis capabilities.

## Features

- **OAuth 2.0 Authentication**: Secure Google account connection with token encryption
- **Email Fetching**: Retrieve emails with configurable timeframe and label filters
- **Gemma Analysis**: Batch process emails through the local Gemma LLM with custom prompts
- **Streaming Results**: SSE-based real-time analysis progress feedback

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `GMAIL_CLIENT_ID` | Required | Google OAuth Client ID |
| `GMAIL_CLIENT_SECRET` | Required | Google OAuth Client Secret |
| `GMAIL_REDIRECT_URI` | `http://localhost:8000/api/gmail/oauth/callback` | OAuth redirect URI |
| `GEMMA_SERVICE_URL` | `http://gemma-service:8001` | Gemma service endpoint |
| `RAG_SERVICE_URL` | `http://rag-service:8004` | RAG service for storage |
| `TOKEN_ENCRYPTION_KEY` | From secrets | AES-256 key for token encryption |

## API Endpoints

### OAuth Flow
- `GET /oauth/url` - Generate Google OAuth authorization URL
- `POST /oauth/callback` - Handle OAuth callback and store tokens
- `GET /oauth/status` - Check current OAuth connection status
- `POST /oauth/disconnect` - Revoke tokens and disconnect account

### Email Operations
- `POST /emails/fetch` - Fetch emails for a given timeframe
- `POST /emails/analyze` - Run Gemma analysis on emails
- `GET /emails/analyze/stream` - Stream analysis progress via SSE

## Security

- OAuth tokens are encrypted at rest using AES-256-GCM
- Tokens are stored per-user and never shared
- All endpoints require JWT authentication via the API Gateway
- Refresh tokens enable long-term access without re-authentication

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires Google Cloud credentials)
uvicorn src.main:app --host 0.0.0.0 --port 8016 --reload
```

## Docker

```bash
# Build
docker build -t gmail-service -f docker/gmail_instance/Dockerfile .

# Run
docker run -p 8016:8016 -e GMAIL_CLIENT_ID=... gmail-service
```
