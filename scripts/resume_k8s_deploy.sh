#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd)"
export PATH="$ROOT/bin:$PATH"
CLUSTER_NAME="nemo-cluster-new"
KCFG="/tmp/${CLUSTER_NAME}.conf"

# Ensure kubeconfig exists
echo "Retrieving kubeconfig for $CLUSTER_NAME..."
kind get kubeconfig --name "$CLUSTER_NAME" > "$KCFG"
export KUBECONFIG="$KCFG"

echo "Resuming deployment to $CLUSTER_NAME..."

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
  # Check if image exists locally
  if [[ -z "$(docker images -q "$TAG")" ]]; then
    echo "Building $TAG ..."
    docker build -t "$TAG" -f "$DF" .
  else
    echo "Image $TAG exists locally."
  fi
  echo "Loading $TAG into kind..."
  kind load docker-image --name "$CLUSTER_NAME" "$TAG"
done

echo "Applying manifests..."
# Use the existing install_tmp which we verified has correct paths
kubectl kustomize "k8s/overlays/install_tmp" | kubectl apply -f -
kubectl apply -f "k8s/base/nginx-ingress.yaml"

echo "Waiting for deployments..."
# We use '|| true' to not crash the script if they aren't ready immediately, 
# allowing the user to check status manually.
kubectl wait -n nemo --for=condition=available deploy/api-gateway --timeout=180s || echo "Warning: Gateway wait timed out"
kubectl wait -n nemo --for=condition=available deploy/gemma-service --timeout=180s || echo "Warning: Gemma wait timed out"

echo "Done! Check status with: bin/kubectl --context kind-$CLUSTER_NAME -n nemo get pods"
