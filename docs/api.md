# NeMo Server API Documentation

> Version: 2.0.0  
> Generated: 2024-12-23  
> **Total Endpoints: 245**

> [!IMPORTANT]
> **This document provides a high-level overview.**
> For the most up-to-date and interactive API documentation, please refer to the [OpenAPI Specification](api/openapi.yaml) or run the server and visit `http://localhost:8000/docs`.

Privacy-first cognitive AI platform providing real-time transcription, LLM chat, semantic search, and ML predictions.

---

## Table of Contents

- [Authentication](#authentication)
- [Common Headers](#common-headers)
- [Error Responses](#error-responses)
- [Endpoint Categories](#endpoint-categories)

---

## Authentication

All protected endpoints require one of:
- **Bearer Token**: `Authorization: Bearer <jwt_token>`
- **Session Cookie**: `ws_session` cookie from login

### Obtaining a Token

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

---

## Common Headers

| Header | Description | Required |
|--------|-------------|----------|
| `Authorization` | Bearer token: `Bearer <token>` | Protected endpoints |
| `X-Request-ID` | Unique request identifier for tracing | Optional |
| `X-CSRF-Token` | CSRF token for mutating requests | POST/PUT/DELETE |
| `Content-Type` | `application/json` | Request body |

---

## Error Responses

All errors follow RFC 7807 format:

```json
{
  "error_code": "ERROR_CODE",
  "message": "Human readable message",
  "details": {},
  "request_id": "req_abc123"
}
```

---

## Endpoint Categories

### Health (4 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |
| GET | `/api/health` | API health check |
| GET | `/ready` | Readiness probe |
| GET | `/api/audit/status` | Audit system status |

---

### Authentication (7 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/login` | Authenticate user |
| POST | `/api/auth/logout` | End session |
| POST | `/api/auth/register` | Register new user |
| GET | `/api/auth/session` | Get current session |
| GET | `/api/auth/check` | Check authentication status |
| GET | `/api/auth/users` | List users (admin) |
| POST | `/api/auth/users/create` | Create user (admin) |

---

### Gemma - LLM (17 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/gemma/generate` | Generate text completion |
| POST | `/api/gemma/chat` | Multi-turn chat conversation |
| POST | `/api/gemma/chat-rag` | RAG-enhanced chat |
| POST | `/api/gemma/analyze` | Document analysis |
| POST | `/api/gemma/warmup` | Pre-load model |
| POST | `/api/gemma/release-session` | Release GPU resources |
| GET | `/api/gemma/stats` | Model statistics |
| POST | `/api/public/chat` | Public chat endpoint |

---

### Call Intelligence (26 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/calls/ingest` | Ingest call recording |
| GET | `/api/v1/calls/{call_id}` | Get call details |
| GET | `/api/v1/calls` | List calls |
| POST | `/api/v1/calls/search` | Search calls |
| POST | `/api/v1/calls/{call_id}/summarize` | Summarize call |
| POST | `/api/v1/calls/{call_id}/analyze` | Analyze call |

---

### Salesforce Integration (56 endpoints)

#### Accounts (5 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/salesforce/accounts` | List accounts |
| POST | `/api/v1/salesforce/accounts` | Create account |
| GET | `/api/v1/salesforce/accounts/{id}` | Get account |
| PATCH | `/api/v1/salesforce/accounts/{id}` | Update account |
| DELETE | `/api/v1/salesforce/accounts/{id}` | Delete account |

#### Leads (5 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/salesforce/leads` | List leads |
| POST | `/api/v1/salesforce/leads` | Create lead |
| GET | `/api/v1/salesforce/leads/{id}` | Get lead |
| PATCH | `/api/v1/salesforce/leads/{id}` | Update lead |
| DELETE | `/api/v1/salesforce/leads/{id}` | Delete lead |

#### Opportunities (7 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/salesforce/opportunities` | List opportunities |
| POST | `/api/v1/salesforce/opportunities` | Create opportunity |
| GET | `/api/v1/salesforce/opportunities/{id}` | Get opportunity |
| PATCH | `/api/v1/salesforce/opportunities/{id}` | Update opportunity |
| DELETE | `/api/v1/salesforce/opportunities/{id}` | Delete opportunity |

#### AI Analytics (21 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/salesforce/analytics/lead-score` | Score lead |
| POST | `/api/v1/salesforce/analytics/opportunity-score` | Score opportunity |
| POST | `/api/v1/salesforce/analytics/gemma-insights` | AI insights |

---

### Enterprise Features (49 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/enterprise/qa/feedback` | Submit QA feedback |
| GET | `/api/enterprise/qa/review` | Review queue |
| POST | `/api/enterprise/qa/approve` | Approve item |
| POST | `/api/enterprise/qa/reject/{id}` | Reject item |
| GET | `/api/enterprise/qa/golden` | Golden dataset |

---

### Fiserv Banking Hub (26 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/fiserv/health` | Fiserv health check |
| GET | `/fiserv/stats` | Service statistics |
| GET | `/fiserv/transactions/weekly` | Weekly transactions |
| GET | `/fiserv/api/v1/token` | Get API token |
| GET | `/fiserv/api/v1/usage` | API usage stats |

---

### Banking (8 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/banking/analyze/{member_id}` | Analyze member |
| GET | `/api/v1/banking/executive/dashboard` | Executive dashboard |
| GET | `/api/v1/banking/executive/trends` | Trend analysis |
| GET | `/api/v1/banking/fraud/portfolio-risk` | Portfolio risk |

---

### ML Analytics (25 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/ml/ingest` | Ingest data |
| POST | `/api/ml/propose-goals` | Propose ML goals |
| POST | `/api/ml/execute-analysis` | Execute analysis |
| POST | `/api/ml/explain-finding` | Explain finding |
| POST | `/upload` | File upload |

---

### RAG & Memory (11 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/rag/query` | RAG query |
| POST | `/api/search/semantic` | Semantic search |
| POST | `/api/rag/memory/search` | Memory search |
| GET | `/api/memory/list` | List memories |
| POST | `/api/memory/search` | Search memories |

---

### Transcription (5 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/transcription/transcribe` | Transcribe audio |
| POST | `/api/transcription/stream` | Stream transcription |
| POST | `/transcribe` | Quick transcribe |
| POST | `/api/transcribe` | API transcribe |
| POST | `/api/emotion/analyze` | Emotion analysis |

---

### Transcripts & Analytics (9 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/analytics/signals` | Emotion signals |
| GET | `/api/analytics/segments` | Transcript segments |
| GET | `/api/transcripts/speakers` | Speaker list |
| POST | `/api/transcripts/query` | Query transcripts |

---

### Emotions (3 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/emotions/stats` | Emotion statistics |
| GET | `/emotions/analytics` | Emotion analytics |
| GET | `/emotions/moments` | Emotion moments |

---

### Email (9 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/email/users` | Email users |
| GET | `/api/email/labels` | Email labels |
| GET | `/api/email/stats` | Email stats |
| POST | `/api/email/query` | Query emails |
| POST | `/api/email/analyze/quick` | Quick analysis |

---

### CTI Integration (10 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/cti/webhook/call_started` | Call started webhook |
| POST | `/api/v1/cti/webhook/call_answered` | Call answered webhook |
| POST | `/api/v1/cti/webhook/call_ended` | Call ended webhook |
| POST | `/api/v1/cti/verify` | Verify CTI |
| GET | `/api/v1/cti/verification/options` | Verification options |

---

### Enrollment (4 endpoints)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/enroll/upload` | Upload enrollment audio |
| GET | `/enroll/speakers` | List enrolled speakers |
| POST | `/api/enroll/upload` | API enrollment upload |
| GET | `/api/enroll/speakers` | API speaker list |

---

## Rate Limiting

| Endpoint Category | Limit | Window |
|-------------------|-------|--------|
| Authentication | 10 | 1 minute |
| General API | 100 | 1 minute |
| GPU Services | 10 | 1 minute |

When rate limited, responses include headers:
- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Reset timestamp

---

## Servers

| Environment | URL |
|-------------|-----|
| Local Development | `http://localhost:8000` |
| Swagger UI | `http://localhost:8000/docs` |

---

*Last updated: 2024-12-23 | Auto-generated from OpenAPI schema*
