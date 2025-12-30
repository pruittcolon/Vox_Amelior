# Kubernetes Deployment

> [!WARNING]
> **Deprecated Deployment Method**
> The Kubernetes manifests in this directory are not actively maintained and may reference outdated image tags.
> The primary supported deployment method is **Docker Compose** (via `./nemo` or `docker/docker-compose.yml`).

This directory contains Kubernetes manifests for deploying the Nemo platform on a local or cloud K8s cluster.

## 📁 Directory Structure

```
k8s/
├── base/                    # Base manifests (kustomize)
│   ├── deployments.yaml     # All 11 service deployments
│   ├── services.yaml        # ClusterIP service definitions
│   ├── secrets.yaml         # Secret references (DO NOT commit real secrets)
│   ├── namespace.yaml       # nemo namespace
│   ├── nvidia-device-plugin.yaml  # GPU support with time-slicing
│   ├── network-policies.yaml      # Default-deny + service allow rules
│   ├── nginx-ingress.yaml   # Ingress controller
│   ├── ingress.yaml         # Routing rules
│   └── kustomization.yaml   # Kustomize config
└── overlays/                # Environment-specific overrides
    ├── dev/
    ├── staging/
    └── production/
```

## 🚀 Quick Start (Kind Cluster with GPU)

### Prerequisites
- Docker with NVIDIA Container Toolkit
- `kind` binary: `go install sigs.k8s.io/kind@latest`
- `kubectl` binary: Install via package manager

### Environment Setup

The K8s manifests expect the project to be accessible at `/opt/nemo-server`. Create a symlink:

```bash
# Create symlink to your cloned repo
sudo ln -s /path/to/your/Nemo_Server /opt/nemo-server

# Example:
sudo ln -s $(pwd) /opt/nemo-server
```

> **Note:** This symlink is required for hostPath volume mounts in the Kind cluster.

### 1. Create GPU-Enabled Cluster

```bash
./bin/kind create cluster --name nemo --config scripts/kind-gpu.yaml
```

### 2. Install NVIDIA Device Plugin

```bash
# Install nvidia-container-toolkit inside Kind node
docker exec nemo-control-plane bash -c \
  "apt-get update && apt-get install -y nvidia-container-toolkit && \
   nvidia-ctk runtime configure --runtime=containerd && \
   systemctl restart containerd"

# Apply device plugin
kubectl apply -f k8s/base/nvidia-device-plugin.yaml
```

### 3. Load Docker Images

```bash
for img in refactored-gateway:local refactored-gemma:local refactored-transcription:local \
           refactored-rag-service:local refactored-emotion-service:local \
           refactored-insights:local refactored-gpu-coordinator:local docker-ml-service:latest; do
  ./bin/kind load docker-image --name nemo "$img"
done
```

### 4. Deploy All Services

```bash
kubectl apply -f k8s/base/
```

### 5. Access the Platform

```bash
kubectl port-forward svc/api-gateway 8000:8000 -n nemo
# Open http://localhost:8000
```

---

## Services Overview

| Service | Port | Description | GPU |
|---------|------|-------------|-----|
| **api-gateway** | 8000 | Central entry point, auth, routing | No |
| **gemma-service** | 8001 | Gemma 3-4B LLM inference | Yes |
| **gpu-coordinator** | 8002 | GPU semaphore & task scheduling | No |
| **transcription-service** | 8003 | ASR + Speaker Diarization (Parakeet) | Yes* |
| **rag-service** | 8004 | Vector DB & semantic search | No |
| **emotion-service** | 8005 | Sentiment analysis | No |
| **ml-service** | 8006 | AutoML & System 2 validation | No |
| **insights-service** | 8010 | Business analytics | No |
| **redis** | 6379 | Caching & semaphore locks | No |
| **postgres** | 5432 | Persistent storage | No |

*Transcription uses GPU via coordinator handoff (starts on CPU).

---

## GPU Sharing Architecture

Since consumer GPUs (6-8GB VRAM) can't run multiple AI models simultaneously, we use:

### 1. NVIDIA Device Plugin with Time-Slicing
```yaml
# nvidia-device-plugin.yaml configures 2 virtual GPUs from 1 physical
sharing:
  timeSlicing:
    resources:
      - name: nvidia.com/gpu
        replicas: 2
```

### 2. Application-Level Coordination
- **gemma-service** requests `nvidia.com/gpu: 1`
- **transcription-service** uses `privileged: true` + hostPath mounts
- **gpu-coordinator** manages handoff: gemma unloads → transcription loads → gemma reloads

---

## 🔐 Secrets Management

**Never commit real secrets!** Use the template:

```bash
cp k8s/base/secrets.yaml.template k8s/base/secrets.yaml
# Edit secrets.yaml with base64-encoded values
```

Required secrets:
- `postgres_user` / `postgres_password`
- `session_key` (32-byte base64)
- `jwt_secret_primary` / `jwt_secret_previous` / `jwt_secret`
- `users_db_key` (encryption key)
- `redis_password` (Redis authentication)

---

## 🔐 Network Policies

The cluster implements **default-deny** policies with explicit service-to-service allow rules:

### Default Deny
```yaml
# All ingress to nemo namespace is denied by default
# Only explicitly allowed traffic passes
```

### Service Allow Rules
| From | To | Port | Purpose |
|------|-----|------|---------|
| nginx | api-gateway | 8000 | Reverse proxy |
| api-gateway | gemma-service | 8001 | LLM inference |
| api-gateway | transcription | 8003 | ASR processing |
| api-gateway | rag-service | 8004 | Semantic search |
| api-gateway | emotion-service | 8005 | Sentiment |
| api-gateway | insights-service | 8010 | Analytics |
| gpu-coordinator | redis | 6379 | Lock management |
| transcription | redis | 6379 | Pause/resume |
| rag-service | postgres | 5432 | Data storage |

---

## 📊 Health Checks

All services expose `/health` endpoints:

```bash
# Check all services
for svc in api-gateway gemma-service transcription-service rag-service; do
  kubectl exec -n nemo deployment/api-gateway -- curl -s http://$svc:*/health
done
```

---

## 🛠️ Troubleshooting

### GPU Not Detected
```bash
# Verify GPU is allocatable
kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}'
# Should return "1" or "2" (with time-slicing)
```

### Pod Stuck in Pending
```bash
kubectl describe pod <pod-name> -n nemo
# Check Events for resource issues
```

### Image Pull Errors
```bash
# Ensure images are loaded into Kind
./bin/kind load docker-image --name nemo <image>:tag
```

---

## 📚 Related Documentation

- [ARCHITECTURE.md](../ARCHITECTURE.md) - System design overview
- [docker/docker-compose.yml](../docker/docker-compose.yml) - Docker deployment
- [scripts/kind-gpu.yaml](../scripts/kind-gpu.yaml) - Kind cluster config
