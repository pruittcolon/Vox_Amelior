# Fiserv Automation Service

Automation layer for Fiserv Banking Hub DNA platform with AI-assisted insights.

## Quick Start

### Development Mode (auto-reload, edit code without rebuild)
```bash
cd services/fiserv-service
docker-compose --profile dev up --build
```
**Code changes in `src/` apply immediately!**

### Production Mode (code baked in)
```bash
docker-compose --profile prod up --build
```

## API Usage Tracking

**SANDBOX LIMIT: 1000 API CALLS TOTAL**

Check usage:
```bash
curl http://localhost:8015/api/v1/usage
```

## Authentication

Requires Service-to-Service (S2S) JWT authentication via the `X-Service-Token` header. All endpoints are protected when accessed through the API Gateway.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/usage` | GET | **Check API call count (1000 limit)** |
| `/api/v1/status` | GET | Service status + usage stats |
| `/api/v1/token/refresh` | POST | Refresh OAuth token |
| `/api/v1/party/search` | POST | Search customers |
| `/api/v1/account/lookup` | POST | Get account details |
| `/api/v1/transactions/list` | POST | List transactions + anomaly detection |
| `/api/v1/alerts` | GET | Get flagged items for review |

## Providers

9 providers available: `DNABanking`, `Premier`, `Signature`, `Finxact`, `Precision`, `Cleartouch`, `Portico`, `Identity`, `DNACU`

Use `?provider=Premier` query param to switch.

## Configuration

Set in `docker-compose.yml` or environment:
- `FISERV_API_KEY` - Your API key
- `FISERV_API_SECRET` - Your API secret  
- `FISERV_ORG_ID` - Organization ID (default: DNABanking)

---

## Adding a New Provider Integration

1. **Create provider module** in `src/providers/`:
```python
# src/providers/my_provider.py
class MyProvider:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
    
    async def search_party(self, query: dict) -> dict:
        # Implement provider-specific API call
        pass
```

2. **Register in provider factory** (`src/providers/__init__.py`):
```python
PROVIDERS = {
    "DNABanking": DNABankingProvider,
    "MyProvider": MyProvider,  # Add here
}
```

3. **Restart service**:
```bash
docker restart refactored_fiserv_service
```

---

## Error Handling

| Code | Description |
|------|-------------|
| 401 | Missing or invalid service token |
| 429 | API rate limit exceeded (1000 calls) |
| 503 | Fiserv API unavailable |

