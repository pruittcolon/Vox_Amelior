# Nemo Server

A **Local-First, Offline-Capable** AI orchestration platform designed to manage high-load cognitive workloads on consumer hardware. Nemo Server solves resource contention between Large Language Models (LLMs) and real-time services like Automatic Speech Recognition (ASR) through a custom preemptive resource scheduler.

> **Mission:** "People First AI." Empowering individuals and organizations with enterprise-grade intelligence that runs 100% offline—no data center required.

## Core Architecture

The system operates on a Hub-and-Spoke microservices architecture, implementing a secondary cognitive loop ("System 2") for output verification.

- **Resource Orchestration**: Custom GPU coordinator using Redis-based semaphores to manage VRAM allocation between Gemma (LLM) and Parakeet (ASR).
- **Service Mesh**: Nginx reverse proxy with TLS termination and strict mTLS for inter-service communication.
- **Microservices**: 13 containerized services including:
    - **Transcription**: Real-time ASR with speaker diarization.
    - **Inference**: Quantized LLM execution (Gemma 3B/4B).
    - **Memory**: Vector-based RAG service using FAISS.
    - **Validation**: Scikit-Learn/XGBoost engines for symbolic regression and output verification.
    - **Automation**: n8n workflow automation for voice commands.
    - **Banking**: Fiserv integration for financial operations.

## Agentic Context

**For AI Agents:** This repository is a complex microservices architecture.
- **Entry Point**: `./nemo` (Bash script wrapper around `scripts/start.sh`).
- **Configuration**: Defined in `docker/docker-compose.yml`.
- **Source Code**:
  - `services/`: Backend microservices (Python/FastAPI).
  - `frontend/`: No-build ESM-based frontend.
  - `mobile-app/`: Flutter mobile client.
  - `shared/`: Shared Python libraries for auth, security, and logging.
- **Documentation**: `docs/ARCHITECTURE.md` contains the system diagram and port mappings.

## Deployment

### Prerequisites
- **OS**: Linux (Ubuntu 22.04+) or WSL2
- **Hardware**: NVIDIA GPU (6GB+ VRAM recommended)
- **Drivers**: CUDA 12.6+ with NVIDIA Container Toolkit

### Docker Compose
```bash
# 1. Clone repository
git clone https://github.com/pruittcolon/Nemo_Server.git
cd Nemo_Server

# 2. Setup security
./scripts/setup_secrets.sh
./scripts/generate_certs.sh

# 3. Launch services
./nemo                 # or: ./scripts/start.sh
```

Services will be available at `https://localhost` (accept self-signed certificate).

## Security

Refer to [SECURITY.md](SECURITY.md) for the complete security policy.

- **Transport**: TLS 1.2+ enforced; mTLS for internal traffic.
- **Authentication**: JWT with key rotation; Redis-backed replay protection.
- **Secrets**: Docker secrets management; no hardcoded credentials.

## Documentation

- [Architecture Design](docs/ARCHITECTURE.md)
- [API Specification](docs/api/openapi.yaml)
- [Service Level Objectives](docs/SLO.md)

## License

MIT License