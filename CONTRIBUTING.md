# Contributing to Nemo Server

Thank you for your interest in contributing. We value your time and expertise.

## How to Contribute

### Reporting Bugs

Please use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:
- Steps to reproduce
- Observed vs expected behavior
- Environment details (OS, Python version, Docker version, GPU model)
- Logs and error messages

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Please provide:
- A detailed description of the proposed functionality
- Rationale for the enhancement
- Alternative solutions considered

### Pull Requests

1. **Fork** the repository.
2. **Branch** from `main`.
3. **Implement** changes following code standards.
4. **Test** your changes.
5. **Submit** a pull request.

## Development Setup

### Prerequisites
- Docker 24.0+
- Docker Compose
- NVIDIA GPU with CUDA 12.6+ (recommended)

### Getting Started

```bash
# 1. Clone fork
git clone https://github.com/YOUR_USERNAME/NeMo_Server.git
cd NeMo_Server

# 2. Setup environment
./scripts/setup_secrets.sh
./nemo

# 3. Verify
./scripts/run_tests.sh
```

## Code Standards

### Python
- Follow **PEP 8**.
- Use strict type hints.
- Add Google-style docstrings.
- Max line length: 120 characters.

### Docker
- Minimize image layers.
- Do not include secrets in images.
- Use specific version tags.

### Documentation Standards (For AI Agents)
- **Structure**: Use clear headers and standardized directory structures.
- **Accuracy**: Keep `README.md` and `docs/` synchronized with code changes (ports, env vars).
- **Context**: Explicitly mention file paths and service dependencies to help AI agents parse the repo context.

## Repository Structure

```
NeMo_Server/
├── services/              # Microservices
│   ├── api-gateway/       # Main API entry point (Port 8000)
│   ├── gemma-service/     # LLM service (Port 8001)
│   ├── transcription-service/  # Audio transcription (Port 8003)
│   ├── ml-service/        # ML analytics (Port 8006)
│   ├── fiserv-service/    # Banking integration (Port 8015)
│   ├── rag-service/       # RAG memory service (Port 8004)
│   ├── emotion-service/   # Sentiment analysis (Port 8005)
│   ├── insights-service/  # Business analytics (Port 8010)
│   ├── n8n-service/       # Automation workflows (Port 8011)
│   └── queue-service/     # GPU Coordinator
├── shared/                # Shared libraries
├── frontend/              # HTML/CSS/JS frontend
├── docker/                # Docker Compose files
├── scripts/               # Automation scripts
└── tests/                 # Unit and integration tests
```

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): subject`.

## Testing Strategy

### Unit Tests
Fast, isolated tests for individual functions.
```bash
cd tests/unit
pytest -v --cov=services
```

### Integration Tests
Test service interactions (requires Docker).
```bash
docker compose up -d
pytest tests/integration/ -v
```

### E2E Browser Tests (Playwright)
Full user flow testing.
```bash
cd scripts/browser_tests
python3 runner.py --all
```

Run security verification before submitting:
```bash
./scripts/verify_security.py
```

## Code Review Process

1. Automated checks pass (CI/CD, Security, Tests).
2. Maintainer review (Design, Performance, Security).
3. Address feedback.
4. Squash and merge.

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
nvidia-smi
docker compose logs gpu-coordinator
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
