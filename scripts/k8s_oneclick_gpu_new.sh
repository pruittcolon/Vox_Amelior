#!/usr/bin/env bash
set -euo pipefail

# Smart launcher for GPU-enabled kind cluster.
# - Creates cluster if missing, reuses if present.
# - Checks if images are already loaded in the cluster to avoid rebuilds.
# - Requires models present:
#     models/gemma-3-4b-it-UD-Q4_K_XL.gguf
#     models/diar_sortformer_4spk-v1.nemo
#
# Usage: CLUSTER_NAME=nemo-cluster-new ./scripts/k8s_oneclick_gpu_new.sh

RED() { printf '\033[0;31m%s\033[0m\n' "$*"; }
YEL() { printf '\033[1;33m%s\033[0m\n' "$*"; }
GRN() { printf '\033[0;32m%s\033[0m\n' "$*"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/bin:$PATH"

CLUSTER_NAME="${CLUSTER_NAME:-nemo-cluster-new}"
KCFG="/tmp/${CLUSTER_NAME}.conf"
KIND_CONFIG="${KIND_CONFIG:-$ROOT/scripts/kind-gpu.yaml}"

req() { command -v "$1" >/dev/null 2>&1 || { RED "Missing prerequisite: $1"; exit 1; }; }

YEL "[1/6] Checking prerequisites..."
req docker; req kind; req kubectl
if ! nvidia-smi >/dev/null 2>&1; then
  RED "nvidia-smi not working on host. Install NVIDIA drivers/toolkit first."
  exit 1
fi

YEL "[2/6] Checking required model files..."
for f in "$ROOT/models/gemma-3-4b-it-UD-Q4_K_XL.gguf" "$ROOT/models/diar_sortformer_4spk-v1.nemo"; do
  [[ -f "$f" ]] || { RED "Missing model file: $f"; exit 1; }
done

# CHECK CLUSTER EXISTENCE
if kind get clusters | grep -qx "$CLUSTER_NAME"; then
  GRN "Cluster '$CLUSTER_NAME' already exists. Reusing it."
else
  YEL "[3/6] Creating kind cluster '$CLUSTER_NAME' with GPU support..."
  kind create cluster --name "$CLUSTER_NAME" --config "$KIND_CONFIG"
fi

# Always update kubeconfig
kind get kubeconfig --name "$CLUSTER_NAME" > "$KCFG"
export KUBECONFIG="$KCFG"

YEL "[4/6] Checking images in cluster (skips rebuild if present inside kind)..."

# We'll use crictl inside the node to check for loaded images
NODE_NAME="${CLUSTER_NAME}-control-plane"
EXISTING_IMAGES=$(docker exec "$NODE_NAME" crictl images -o json | grep -o '"repoTags": \[[^]]*' | cut -d'"' -f4)

IMAGES=(
  "refactored-gateway:local|docker/Dockerfile.gateway"
  "refactored-rag-service:local|docker/Dockerfile.rag"
  "refactored-emotion-service:local|docker/Dockerfile.emotion"
  "refactored-transcription:local|docker/Dockerfile.transcription"
  "refactored-gemma:local|docker/Dockerfile.gemma"
  "refactored-gpu-coordinator:local|docker/Dockerfile.queue"
  "refactored-insights:local|docker/Dockerfile.insights"
)

for entry in "${IMAGES[@]}"; do
  IFS="|" read -r TAG DF <<<"$entry"
  
  # Check if tag exists in the cluster output
  if echo "$EXISTING_IMAGES" | grep -q "^docker.io/library/$TAG$"; then
    GRN "Image $TAG is already loaded in cluster. Skipping."
    continue
  fi

  # Check if local image exists
  if [[ -z "$(docker images -q "$TAG")" ]]; then
    YEL "Building $TAG ..."
    docker build -t "$TAG" -f "$ROOT/$DF" "$ROOT"
  else
    YEL "Image $TAG exists locally (host), skipping build."
  fi
  
  YEL "Loading $TAG into kind..."
  kind load docker-image --name "$CLUSTER_NAME" "$TAG"
done

YEL "[5/6] Apply manifests..."
# NVIDIA device plugin for GPU scheduling
kubectl apply -f "$ROOT/k8s/base/nvidia-device-plugin.yaml"
kubectl kustomize "$ROOT/k8s/overlays/install_tmp" | kubectl apply -f -
kubectl apply -f "$ROOT/k8s/base/nginx-ingress.yaml"

# Verify GPU allocatable after device plugin
GPU_ALLOC=$(kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}:{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' || true)
if echo "$GPU_ALLOC" | grep -q ":[1-9]"; then
  GRN "GPU allocatable detected: $GPU_ALLOC"
else
  YEL "Warning: no nvidia.com/gpu allocatable reported; ensure NVIDIA drivers/toolkit and device plugin are working."
fi

YEL "[6/6] Waiting for core services..."
# Use || true to avoid crash if wait fails, allows manual check
kubectl wait -n nemo --for=condition=available deploy/api-gateway --timeout=180s || YEL "Warning: Gateway wait timed out"
kubectl wait -n nemo --for=condition=available deploy/gemma-service --timeout=180s || YEL "Warning: Gemma wait timed out"
kubectl wait -n nemo --for=condition=available deploy/transcription-service --timeout=180s || YEL "Warning: Transcription wait timed out"

GRN "Cluster ready: $CLUSTER_NAME (KUBECONFIG=$KCFG)"
YEL "Models used from: $ROOT/models"
GRN "Check pods: kubectl get pods -A"
GRN "Health: curl -s http://localhost/health"
