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
./start.sh

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

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): subject`.

## Testing

- New features must have tests.
- Maintain >80% code coverage.
- Run tests before submitting:
    ```bash
    pytest
    ./scripts/verify_security.py
    ```

## Code Review Process

1. Automated checks pass (CI/CD, Security, Tests).
2. Maintainer review (Design, Performance, Security).
3. Address feedback.
4. Squash and merge.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
