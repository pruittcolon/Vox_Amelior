# Nemo Server

A local-first AI orchestration platform designed to manage high-load cognitive workloads on consumer hardware. Nemo Server solves resource contention between Large Language Models (LLMs) and real-time services like Automatic Speech Recognition (ASR) through a custom preemptive resource scheduler.

## Core Architecture

The system operates on a Hub-and-Spoke microservices architecture, implementing a secondary cognitive loop ("System 2") for output verification.

- **Resource Orchestration**: Custom GPU coordinator using Redis-based semaphores to manage VRAM allocation between Gemma (LLM) and Parakeet (ASR).
- **Service Mesh**: Nginx reverse proxy with TLS termination and strict mTLS for inter-service communication.
- **Microservices**: 11 containerized services including:
    - **Transcription**: Real-time ASR with speaker diarization.
    - **Inference**: Quantized LLM execution (Gemma 3B/4B).
    - **Memory**: Vector-based RAG service using FAISS.
    - **Validation**: Scikit-Learn/XGBoost engines for symbolic regression and output verification.

## Deployment

### Prerequisites
- **OS**: Linux (Ubuntu 22.04+) or WSL2
- **Hardware**: NVIDIA GPU (6GB+ VRAM recommended)
- **Drivers**: CUDA 12.0+ with NVIDIA Container Toolkit

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