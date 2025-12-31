# Vox Amelior

A **Local-First, Privacy-Focused AI Platform** demonstrating enterprise-grade architecture on consumer hardware. This project implements a complete microservices ecosystem with real-time speech recognition, large language model inference, and 22 specialized machine learning engines - all running offline without cloud dependencies.

---

## Technical Overview

| Metric | Value |
|--------|-------|
| **Microservices** | 13 containerized services |
| **ML Engines** | 22 specialized prediction models |
| **Frontend** | ESM-based SPA with real-time visualizations |
| **Mobile** | Flutter companion app with voice control |
| **Infrastructure** | Docker Compose + Kubernetes manifests |

---

## Architecture

The system implements a Hub-and-Spoke pattern with custom GPU orchestration for running multiple AI workloads on a single consumer GPU (6GB VRAM).

```
                    +------------------+
                    |   Nginx (TLS)    |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   API Gateway    |
                    |  (Auth, Routing) |
                    +--------+---------+
                             |
        +--------------------+--------------------+
        |                    |                    |
+-------v-------+    +-------v-------+    +-------v-------+
|  Gemma LLM    |    | Transcription |    |  ML Service   |
|  (4-bit Q)    |    |   (ASR+Diar)  |    | (22 Engines)  |
+---------------+    +---------------+    +---------------+
        |                    |                    |
        +--------------------+--------------------+
                             |
                    +--------v---------+
                    | GPU Coordinator  |
                    | (Redis Semaphore)|
                    +------------------+
```

### Core Services

| Service | Technology | Purpose |
|---------|------------|---------|
| **API Gateway** | FastAPI | JWT auth, rate limiting, request routing |
| **Gemma Service** | llama.cpp | Quantized LLM inference (Gemma 3-4B) |
| **Transcription** | NeMo Parakeet | Real-time ASR with speaker diarization |
| **ML Service** | Scikit-Learn, XGBoost | 22 predictive engines (see below) |
| **RAG Service** | FAISS | Vector search and semantic memory |
| **Emotion Service** | DistilRoBERTa | Sentiment and emotion classification |
| **Insights Service** | Pandas | Business analytics aggregation |
| **GPU Coordinator** | Redis | VRAM allocation via semaphore protocol |

---

## ML Engine Suite

The platform includes 22 specialized engines grouped by domain:

**Financial Analysis**
- Pricing Optimization, Cash Flow Forecasting, Inventory Management
- Resource Utilization, Market Basket Analysis, Cost Analysis

**Predictive Intelligence**
- Titan (ensemble), Mirror (comparative), Chronos (time-series)
- Causality, Correlation, Clustering

**Advanced Analytics**
- Anomaly Detection, Statistical Regression, Survival Analysis
- Sentiment, Risk, Monte Carlo Simulation

**System 2 Verification**
- Cross-validation layer that verifies LLM outputs against deterministic ML predictions

---

## Security Implementation

| Layer | Implementation |
|-------|----------------|
| **Transport** | TLS 1.2+ with HSTS, modern cipher suites |
| **Authentication** | JWT with key rotation, Redis-backed replay protection |
| **Service Mesh** | Mutual JWT for service-to-service communication |
| **Secrets** | Docker secrets mounted at `/run/secrets/` (600 permissions) |
| **Containers** | Read-only root filesystem, dropped capabilities, non-root user |

See [SECURITY.md](SECURITY.md) for the complete security policy.

---

## Deployment

### Prerequisites
- Linux (Ubuntu 22.04+) or WSL2
- NVIDIA GPU with 6GB+ VRAM
- Docker 24.0+ with NVIDIA Container Toolkit
- CUDA 12.6+

### Quick Start

```bash
# Clone repository
git clone https://github.com/pruittcolon/Vox_Amelior.git
cd Vox_Amelior

# Initialize secrets and certificates
./scripts/setup_secrets.sh
./scripts/generate_certs.sh

# Launch all services
./scripts/start.sh

# Access: https://localhost (accept self-signed certificate)
```

---

## Project Structure

```
Vox_Amelior/
├── services/           # 13 Python/FastAPI microservices
├── frontend/           # No-build ESM frontend
├── mobile-app/         # Flutter companion app
├── shared/             # Common libraries (auth, security, logging)
├── docker/             # Docker Compose configuration
├── k8s/                # Kubernetes manifests
├── scripts/            # Automation and deployment scripts
├── tests/              # Unit, integration, and E2E tests
└── docs/               # Architecture and API documentation
```

---

## Documentation

- [Architecture Design](docs/ARCHITECTURE.md) - System diagrams and port mappings
- [API Specification](docs/api/openapi.yaml) - OpenAPI 3.0 spec
- [Service Level Objectives](docs/SLO.md) - Availability and latency targets

---

## Technology Stack

**Backend**: Python 3.12, FastAPI, Redis, PostgreSQL

**ML/AI**: PyTorch, Transformers, Scikit-Learn, XGBoost, FAISS, llama.cpp

**Infrastructure**: Docker, Kubernetes, Nginx, Prometheus

**Frontend**: Vanilla JavaScript (ESM), Chart.js, D3.js

**Mobile**: Flutter/Dart

---

## License

MIT License - See [LICENSE](LICENSE) for details.