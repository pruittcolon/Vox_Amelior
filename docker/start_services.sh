#!/bin/bash
set -e

cd /teamspace/studios/this_studio/Vox_Amelior/docker

echo "1. Creating network..."
docker network create docker_nemo_network || true

echo "2. Starting Infrastructure & CPU Services..."
# Now that depends_on is removed, we can start the gateway easily
docker compose up -d redis postgres api-gateway rag-service emotion-service insights-service n8n-service gpu-coordinator fiserv-service ml-service nginx

echo "3. Starting Gemma Service (GPU)..."
# Using --network-alias to make it discoverable as 'gemma-service'
docker stop refactored_gemma || true
docker rm refactored_gemma || true
docker run -d --name refactored_gemma \
  --restart unless-stopped \
  --gpus all \
  --network docker_nemo_network \
  --network-alias gemma-service \
  -p 8001:8001 \
  -v /teamspace/studios/this_studio/Vox_Amelior/models:/app/models:ro \
  -e GEMMA_MODEL_PATH=/app/models/gemma-3-4b-it-UD-Q4_K_XL.gguf \
  -e GEMMA_GPU_LAYERS=-1 \
  -e GEMMA_CONTEXT_SIZE=2048 \
  -e GPU_COORDINATOR_URL=http://gpu-coordinator:8002 \
  -e JWT_ONLY=true \
  docker-gemma-service

echo "4. Starting Transcription Service (GPU)..."
# Using --network-alias to make it discoverable as 'transcription-service'
docker stop refactored_transcription || true
docker rm refactored_transcription || true
docker run -d --name refactored_transcription \
  --restart unless-stopped \
  --gpus all \
  --network docker_nemo_network \
  --network-alias transcription-service \
  -p 8003:8003 \
  -v /teamspace/studios/this_studio/Vox_Amelior/services/transcription-service/src:/app/src \
  -v /teamspace/studios/this_studio/Vox_Amelior/shared:/app/shared \
  -v /teamspace/studios/this_studio/Vox_Amelior/models/diar_sortformer_4spk-v1.nemo:/app/models/diar_sortformer_4spk-v1.nemo:ro \
  -v /teamspace/studios/this_studio/Vox_Amelior/docker/gateway_instance/uploads:/gateway_instance/uploads:ro \
  -e REDIS_URL=redis://redis:6379 \
  -e GPU_COORDINATOR_URL=http://gpu-coordinator:8002 \
  -e START_ON_CPU=true \
  -e TRANSCRIBE_STRATEGY=parakeet \
  docker-transcription-service

echo "5. Verifying..."
docker ps
