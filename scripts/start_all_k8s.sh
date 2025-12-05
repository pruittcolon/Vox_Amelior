#!/usr/bin/env bash
#
# Start/restore the Nemo stack on kind using cached images.
# - Creates the cluster if missing (scripts/kind-gpu.yaml)
# - Applies NVIDIA device plugin
# - Loads local images (no rebuild)
# - Applies kustomize manifests and ingress
#

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/bin:$PATH"

CLUSTER_NAME="${CLUSTER_NAME:-nemo-cluster-new}"
KIND_CONFIG="${KIND_CONFIG:-$ROOT/scripts/kind-gpu.yaml}"
K8S_OVERLAY="${K8S_OVERLAY:-$ROOT/k8s/overlays/install_tmp}"
IMAGES=(
  "refactored-gateway:local"
  "refactored-rag-service:local"
  "refactored-emotion-service:local"
  "refactored-transcription:local"
  "refactored-gemma:local"
  "refactored-gpu-coordinator:local"
  "refactored-insights:local"
)

err() { printf '\033[0;31m%s\033[0m\n' "$*" >&2; }
info() { printf '\033[0;34m%s\033[0m\n' "$*"; }
ok() { printf '\033[0;32m%s\033[0m\n' "$*"; }

require() {
  command -v "$1" >/dev/null 2>&1 || { err "Missing prerequisite: $1"; exit 1; }
}

require kind
require kubectl

info "[1/5] Ensuring kind cluster ${CLUSTER_NAME}..."
if ! kind get clusters | grep -qx "$CLUSTER_NAME"; then
  kind create cluster --name "$CLUSTER_NAME" --config "$KIND_CONFIG"
else
  ok "Cluster exists, reusing."
fi

info "[2/5] Applying NVIDIA device plugin..."
kubectl apply -f "$ROOT/k8s/base/nvidia-device-plugin.yaml" >/dev/null

info "[3/5] Loading cached images into kind..."
for img in "${IMAGES[@]}"; do
  if docker images -q "$img" >/dev/null 2>&1 && [[ -n "$(docker images -q "$img")" ]]; then
    kind load docker-image --name "$CLUSTER_NAME" "$img" >/dev/null || err "Load failed for $img"
  else
    err "Image not found locally: $img (build it first)"
  fi
done

info "[4/5] Applying manifests..."
kubectl kustomize "$K8S_OVERLAY" | kubectl apply -f - >/dev/null
kubectl apply -f "$ROOT/k8s/base/nginx-ingress.yaml" >/dev/null

info "[5/5] Checking core pods..."
kubectl get pods -n nemo
info "GPU allocatable:"
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}:{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'

ok "Done. Use 'kubectl -n nemo get pods' to watch readiness."
