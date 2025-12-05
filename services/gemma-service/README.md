# Gemma AI Service

Large Language Model (LLM) inference service using Google's Gemma 3 model via llama.cpp with GPU acceleration.

## Overview

The Gemma Service provides conversational AI capabilities with:

- **Long Context**: 64K token context window
- **GPU Acceleration**: CUDA-enabled inference via llama.cpp
- **RAG Integration**: Automatic retrieval of relevant memories
- **GPU Coordination**: Dynamic GPU sharing with transcription service
- **Streaming Support**: Token-by-token streaming responses
- **Chat History**: Maintains conversation context

## Architecture

### GPU Coordination Model

The service uses a **request-based GPU access** model:

1. **Idle State**: Model loaded in CPU/RAM (n_gpu_layers=0)
2. **On Request**:
   - Request GPU lock from coordinator
   - Wait for transcription service to pause (~1-2s)
   - Reload model to GPU (n_gpu_layers=25)
   - Process inference
   - Offload model back to CPU
   - Release GPU lock
3. **Transcription Resumes**: Normal operation restored

This allows both services to coexist on a single GPU system.

### Request Flow

```
Client Request
    ↓
Check RAG Context (semantic search)
    ↓
Build Prompt with Context
    ↓
Request GPU from Coordinator
    ↓
Wait for GPU Lock
    ↓
Reload Model to GPU
    ↓
LLM Inference (streaming)
    ↓
Return Response
    ↓
Offload Model to CPU
    ↓
Release GPU Lock
```

## Key Features

### 1. RAG-Enhanced Responses
- Automatically searches memories for relevant context
- Includes transcript segments and notes in prompt
- Configurable retrieval depth (top_k)
- Bias towards recent conversations

### 2. Long Context Support
- 64K token context window (Gemma 3 capability)
- Maintains long conversation histories
- Summarization strategies for ultra-long contexts

### 3. GPU Optimization
- On-demand GPU loading
- Fast CPU ↔ GPU model transfer
- Minimal idle GPU usage
- Efficient batch processing

### 4. Streaming Responses
- Token-by-token streaming via Server-Sent Events (SSE)
- Low latency first token
- Interruptible generation

## API Endpoints

### Chat Completion

```bash
POST /chat
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "What did John say about the project?"}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "context": [],  // Optional: additional context documents
  "use_rag": true,  // Enable RAG retrieval
  "rag_top_k": 5  // Number of relevant memories to retrieve
}
```

Response:
```json
{
  "response": "Based on the transcript from yesterday, John mentioned...",
  "model": "gemma-3-4b-it",
  "usage": {
    "prompt_tokens": 1024,
    "completion_tokens": 156,
    "total_tokens": 1180
  },
  "rag_context_used": 5,
  "gpu_wait_time": 1.2,
  "inference_time": 3.4
}
```

### Streaming Chat

```bash
POST /chat/stream
Content-Type: application/json

(Same body as /chat)
```

Response: Server-Sent Events stream
```
data: {"token": "Based", "done": false}
data: {"token": " on", "done": false}
...
data: {"token": ".", "done": true, "usage": {...}}
```

### Health Check

```bash
GET /health
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMMA_MODEL_PATH` | `/app/models/gemma-3-4b-it-UD-Q4_K_XL.gguf` | Path to GGUF model file |
| `GEMMA_GPU_LAYERS` | `25` | Number of layers to offload to GPU |
| `GEMMA_CONTEXT_SIZE` | `65536` | Maximum context window (64K) |
| `GEMMA_BATCH_SIZE` | `512` | Batch size for token processing |
| `GPU_COORDINATOR_URL` | `http://gpu-coordinator:8002` | Coordinator service URL |
| `RAG_SERVICE_URL` | `http://rag-service:8004` | RAG service for memory retrieval |
| `JWT_ONLY` | `false` | Enforce JWT-only authentication |

## Model Information

### Gemma 3 4B Instruct
- **Architecture**: Decoder-only transformer
- **Parameters**: 4 billion
- **Quantization**: Q4_K_XL (4-bit quantization)
- **Context**: 64K tokens
- **Size on Disk**: ~2.5GB (GGUF format)

### Performance Characteristics
- **VRAM Usage**: ~4-5GB (with 25 GPU layers)
- **CPU RAM Usage**: ~6GB (when offloaded)
- **Inference Speed**: ~20-30 tokens/second (on GPU)
- **First Token Latency**: ~200-500ms
- **GPU Load Time**: ~500-800ms

## Prompt Engineering

### System Prompt
The service uses a carefully tuned system prompt that:
- Establishes assistant persona
- Provides RAG context integration instructions
- Sets response style and format guidelines
- Handles multi-turn conversation context

### RAG Context Injection
When RAG is enabled:
1. Query is embedded and searched against memory database
2. Top K relevant segments retrieved
3. Context formatted with metadata (speaker, time, emotion)
4. Injected into prompt before user message
5. Model instructed to cite sources when using context

## Dependencies

See `requirements.txt`. Key dependencies:
- **fastapi** - Web framework
- **llama-cpp-python** - Installed via pre-built wheel (CUDA support)
- **httpx** - Service communication
- **pydantic** - Data validation

**Note**: llama-cpp-python is installed from a pre-built wheel with CUDA support (see `docker/wheels/`) to avoid Docker build issues with libcuda.so.1 linking.

## Development

### Running Locally
```bash
cd services/gemma-service
pip install -r requirements.txt
# Install llama-cpp-python wheel separately
pip install ../../docker/wheels/llama_cpp_python-*.whl

uvicorn src.main:app --reload --host 0.0.0.0 --port 8001
```

### Running in Docker
```bash
docker compose up gemma-service
```

## GPU Memory Management

### Memory States
1. **Idle (CPU)**: ~6GB RAM, 0GB VRAM
2. **Loading**: ~8GB RAM, 2GB VRAM (transition)
3. **Active (GPU)**: ~2GB RAM, 5GB VRAM
4. **Inference**: ~2GB RAM, 5-6GB VRAM (peak)

### Optimization Tips
- Reduce `GEMMA_GPU_LAYERS` for lower VRAM usage
- Reduce `GEMMA_CONTEXT_SIZE` if hitting memory limits
- Use smaller quantization (Q3, Q2) for even lower memory
- Consider model sharding for multi-GPU setups

## Troubleshooting

### Model Not Loading
- Check model file exists at `GEMMA_MODEL_PATH`
- Verify CUDA drivers installed
- Check VRAM availability with `nvidia-smi`

### Slow Response Times
- Check GPU coordinator logs for contention
- Verify GPU is actually being used (nvidia-smi during inference)
- Consider increasing `GEMMA_GPU_LAYERS`

### Out of Memory
- Reduce `GEMMA_CONTEXT_SIZE`
- Reduce `GEMMA_GPU_LAYERS`
- Check for memory leaks in long-running processes
- Restart service to clear GPU memory

## Examples

### Simple Chat
```python
import httpx

response = httpx.post("http://localhost:8001/chat", json={
    "messages": [
        {"role": "user", "content": "Explain quantum computing"}
    ],
    "max_tokens": 200
})
print(response.json()["response"])
```

### RAG-Enhanced Query
```python
response = httpx.post("http://localhost:8001/chat", json={
    "messages": [
        {"role": "user", "content": "What did Sarah say about the deadline?"}
    ],
    "use_rag": True,
    "rag_top_k": 5,
    "max_tokens": 300
})
print(response.json()["response"])
# Will include context from transcripts mentioning Sarah and deadlines
```

### Streaming Response
```python
import httpx
import json

with httpx.stream("POST", "http://localhost:8001/chat/stream", json={
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "max_tokens": 500
}) as stream:
    for line in stream.iter_lines():
        if line.startswith("data: "):
            token_data = json.loads(line[6:])
            print(token_data["token"], end="", flush=True)
```