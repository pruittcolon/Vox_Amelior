# Developer Guide

Welcome to the NeMo Server repository! This guide will get you up and running in under 1 hour.

## Repository Structure

```
NeMo_Server/
├── services/              # Microservices
│   ├── api-gateway/       # Main API entry point (Port 8000)
│   ├── gemma-service/     # LLM service (Port 8001)
│   ├── transcription-service/  # Audio transcription (Port 8003)
│   ├── ml-service/        # ML analytics (Port 8006)
│   └── fiserv-service/    # Banking integration (Port 8015)
├── shared/                # Shared libraries
│   ├── security/          # Auth, validation, audit
│   ├── config.py          # Centralized configuration
│   ├── constants.py       # Magic number definitions
│   └── errors.py          # Structured error responses
├── frontend/              # HTML/CSS/JS frontend
├── docker/                # Docker Compose files
├── scripts/               # Automation scripts
│   └── browser_tests/     # Playwright E2E tests
└── tests/                 # Unit and integration tests
```

## Quick Start

### Prerequisites

- Docker 24+ with Compose v2
- Python 3.11+
- Node.js 18+ (for Playwright)
- 16GB RAM recommended

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Nemo_Server.git
cd Nemo_Server

# 2. Set up secrets (one-time)
./scripts/setup_secrets.sh

# 3. Start all services
./start.sh

# 4. Access the application
open http://localhost:8000/ui/index.html
```

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Required
GEMMA_URL=http://gemma-service:8001
ML_SERVICE_URL=http://ml-service:8006

# Optional
DEBUG=false
LOG_LEVEL=INFO
```

## Testing Strategy

### Unit Tests

Fast, isolated tests for individual functions:

```bash
cd tests/unit
pytest -v --cov=services
```

### Integration Tests

Test service interactions (requires Docker):

```bash
docker compose up -d
pytest tests/integration/ -v
```

### E2E Browser Tests (Playwright)

Full user flow testing:

```bash
cd scripts/browser_tests
python3 runner.py --all
```

## Code Standards

### Python Style

- **Formatter:** Ruff (runs via pre-commit)
- **Type hints:** Required for all public functions
- **Docstrings:** Google-style for public APIs

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Commit Messages

Use conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation
- `refactor:` Code restructuring

## CI/CD Pipeline

### Pull Request Checks

1. **Ruff lint/format** - Code quality
2. **MyPy** - Type checking
3. **Bandit** - Security scanning
4. **pytest** - Unit tests (80% coverage required)

### Deployment

Merges to `main` trigger:
1. Docker image build
2. Container security scan (Trivy)
3. Deployment to staging

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs api-gateway

# Common fix: secrets not mounted
./scripts/setup_secrets.sh
```

### GPU Not Available

```bash
# Verify CUDA
nvidia-smi

# Check GPU coordinator
docker compose logs gpu-coordinator
```

### Test Failures

```bash
# Run with verbose output
pytest -v --tb=long

# Check RAM usage
free -h
```

## Getting Help

- Check existing issues on GitHub
- Review the `/docs` folder
- Ask in #dev-support channel
