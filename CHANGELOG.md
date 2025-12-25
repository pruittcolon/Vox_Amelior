# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Replay Protection**: Redis-backed JTI tracking prevents S2S token reuse attacks
- **Fail-Closed Startup**: `shared/security/startup_checks.py` with `assert_strong_secret()` and `assert_secure_mode()`
- **Prometheus Metrics**: `/metrics` endpoint via `shared/telemetry/metrics.py`
- **Circuit Breaker**: Inter-service HTTP client with retry/backoff (`shared/clients/base.py`)
- **Router Architecture**: Modular routers in `services/api-gateway/src/routers/`
- **Security Tests**: 34 unit tests covering replay protection, CSRF, rate limiting, security headers
- Enterprise implementation plan with 8 phases
- SECURITY.md with GitHub Private Vulnerability Reporting
- tests/requirements.txt with pytest 9.0

### Changed
- `SECURE_MODE=true` now enforced in docker-compose.yml
- Moved WireGuard configuration to docker/secrets/
- Updated documentation to reflect 36 ML engines (was 27)
- Added Fiserv and n8n services to Architecture documentation
- Updated `CONTRIBUTING.md` setup instructions to use `scripts/setup_secrets.sh`
- Regenerated `REPO_MAP.md` with latest file structure (excluding .venv)

### Bug Fixes
- CODE_OF_CONDUCT.md placeholder email replaced with security@nemoserver.dev

### Security
- **Removed DummyServiceAuth**: Gateway now fails startup without valid auth
- **Removed DummySession**: No fallback unauthenticated sessions
- Removed PII documents from version control (Resume.txt, Job_Description.txt, Application.txt)
- Added .gitignore entries for sensitive files

## [1.0.0] - 2025-12-11

### Added
- **Architecture**: 10 microservices with Docker Compose orchestration
- **AI/ML**: 27 ML prediction engines (Titan, Mirror, Chronos, etc.)
- **LLM**: Gemma 2B integration with GPU coordination
- **Transcription**: Real-time ASR with Parakeet 1.1B + Pyannote diarization
- **RAG**: FAISS-based semantic search with email ingestion
- **Emotion**: DistilRoBERTa emotion classification
- **Mobile**: Flutter app with real-time transcription
- **Auth**: JWT authentication with session management
- **GPU**: Custom semaphore-based GPU coordination
- **Kubernetes**: Full K8s manifests with Kustomize

### Security
- JWT key rotation support
- SQLCipher database encryption
- Docker secrets management
- Network isolation via bridge networks
- CSRF protection with per-session tokens

---

[Unreleased]: https://github.com/pruittcolon/Nemo_Server/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/pruittcolon/Nemo_Server/releases/tag/v1.0.0
