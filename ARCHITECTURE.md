# Nemo Server Architecture

## 1. System Overview

Nemo Server is a local-first, privacy-focused cognitive AI platform designed to run enterprise-grade AI workloads on consumer hardware (specifically single-GPU setups). It employs a "System 2" architecture, distinguishing between fast, intuitive responses (LLM) and slow, deliberate reasoning (AutoML/Symbolic Regression).

### Key Design Principles
- **Zero Data Exfiltration**: All processing happens locally.
- **Single GPU Orchestration**: A custom semaphore system prevents VRAM contention between heavy services (LLM, ASR).
- **System 2 Verification**: LLM outputs are cross-verified by deterministic ML engines.

## 2. Microservices Architecture

The system follows a Hub-and-Spoke pattern, with the **API Gateway** acting as the central entry point and the **GPU Coordinator** managing hardware resources.

```mermaid
graph TD
    Client[Frontend / Mobile] -->|HTTPS/WSS| Gateway[API Gateway]
    
    subgraph "Orchestration"
        Gateway -->|AuthZ| Queue[GPU Coordinator]
    end
    
    subgraph "Cognitive Services"
        Gateway -->|Route| Gemma[Gemma Service (LLM)]
        Gateway -->|Route| Transcribe[Transcription Service]
        Gateway -->|Route| Emotion[Emotion Service]
    end
    
    subgraph "Memory & Analysis"
        Gateway -->|Route| RAG[RAG Service]
        Gateway -->|Route| Insights[Insights Service]
        Gateway -->|Route| ML[ML Service]
    end
    
    subgraph "Infrastructure"
        Queue -->|Lock| Redis
        RAG -->|Store| Postgres
    end
```

### Service Descriptions

| Service | Port | Description | Tech Stack |
| :--- | :--- | :--- | :--- |
| **API Gateway** | 8000 | Central entry point, handles Authentication (JWT), Rate Limiting, and Routing. | Python, FastAPI |
| **GPU Coordinator** | 8002 | Manages the GPU semaphore. Pauses background tasks (ASR) when foreground tasks (Chat) need VRAM. | Python, Redis |
| **Gemma Service** | 8001 | Runs the Gemma 3-4B LLM. Optimized for 4-bit quantization. | Llama.cpp, Python |
| **Transcription Service** | 8003 | Real-time ASR and Speaker Diarization using NVIDIA Parakeet and Pyannote. | PyTorch, Nemo |
| **RAG Service** | 8004 | Vector database and semantic search engine. Handles long-term memory. | FAISS, Sentence-Transformers |
| **Emotion Service** | 8005 | Analyzes text/audio for sentiment and emotional tone. | DistilRoBERTa |
| **ML Service** | 8006 | "System 2" engine. Runs AutoML, Symbolic Regression, and Causal Inference. | Scikit-Learn, Genetic Programming |
| **Insights Service** | 8010 | Provides high-level analytics and business insights derived from RAG data. | Python |

## 3. Infrastructure

- **Redis**: Used for the GPU semaphore lock, task queues, and caching.
- **PostgreSQL**: Primary relational database for structured data (users, chat history).
- **Docker Networks**: Services communicate over an internal bridge network (`nemo_network`).

## 4. Data Flow & Security

### GPU Semaphore Protocol
1. **Request**: A service (e.g., Gemma) requests a GPU lock from the Coordinator.
2. **Preemption**: If the GPU is busy with a lower-priority task (e.g., background transcription), the Coordinator signals it to pause.
3. **Grant**: Once the GPU is free, the lock is granted.
4. **Release**: After inference, the service releases the lock.

### Security
- **Secrets Management**: Docker Secrets are used for all sensitive keys (JWT, DB passwords).
- **Network Isolation**: Only the API Gateway is exposed to the host (in production). All other services are internal.
- **Encryption**: Databases use SQLCipher for encryption at rest.

## 5. Deployment

### Docker Compose
The entire stack is defined in `docker/docker-compose.yml`.
- **Development**: Maps ports to localhost for debugging.
- **Production**: Should restrict port mapping and use a reverse proxy (Nginx/Traefik) in front of the API Gateway.

### Kubernetes
Full K8s manifests are provided in `k8s/base/`:
- **GPU Support**: Includes NVIDIA device plugin with time-slicing for shared GPU access.
- **Kind Cluster**: `scripts/kind-gpu.yaml` provides a GPU-enabled local cluster configuration.
- **10 Services**: All microservices are containerized with proper resource limits and health checks.
