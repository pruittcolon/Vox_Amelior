# GPU Acceleration Implementation for ML Engines

## Overview

This document describes the GPU acceleration implementation for the ML Service engines,
integrating with the existing GPU Coordinator service for resource management.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ML Service                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Mirror Engine│  │Galileo Engine│  │ Titan Engine │          │
│  │   (CTGAN)    │  │   (GCN/PyG)  │  │  (XGBoost)   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                  │
│         └────────────────┼──────────────────┘                  │
│                          │                                      │
│                  ┌───────▼───────┐                              │
│                  │  GPU Client   │                              │
│                  │ (gpu_client.py)│                              │
│                  └───────┬───────┘                              │
└─────────────────────────┼───────────────────────────────────────┘
                          │ HTTP (async)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GPU Coordinator                                │
│                   (queue-service:8002)                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              GPULockManager                              │   │
│  │  - Manages GPU ownership between services                │   │
│  │  - Transcription Service = GPU Owner (priority)          │   │
│  │  - Gemma/ML Services = GPU Requesters                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                    Redis Pub/Sub                                │
│   Channels: transcription_pause, transcription_paused,         │
│             transcription_resume, gpu_owner                     │
└─────────────────────────────────────────────────────────────────┘
```

## Files Modified/Created

### New Files

1. **`services/ml-service/src/core/gpu_client.py`** (~400 lines)
   - `GPUClient` class for async GPU coordination
   - `GPURequestPriority` enum (IMMEDIATE, BACKGROUND)
   - `GPUAcquisitionResult` dataclass
   - Context manager for automatic acquire/release
   - Retry logic and CPU fallback
   - Singleton pattern with `get_gpu_client()`

2. **`services/ml-service/tests/test_gpu_client.py`** (~340 lines)
   - Comprehensive unit tests for GPUClient
   - Tests for initialization, acquisition, release, context manager
   - Tests for health check and singleton pattern

3. **`services/ml-service/tests/test_engine_gpu_integration.py`** (~280 lines)
   - Integration tests for engine GPU support
   - Tests for fallback behavior
   - Tests for proper cleanup

### Modified Files

4. **`docker/docker-compose.yml`**
   - Added GPU reservation to ml-service container
   - Added `GPU_COORDINATOR_URL` environment variable
   - Added `GPU_ENABLED`, `GPU_REQUEST_TIMEOUT`, `GPU_FALLBACK_TO_CPU`
   - Added dependency on gpu-coordinator service

5. **`services/ml-service/src/engines/mirror_engine.py`**
   - Added async GPU client integration
   - Modified `CTGANSynthesizer` to use dynamic `cuda=` parameter
   - Added `analyze_async()` method with GPU coordination
   - Updated insights to show GPU acceleration indicator

6. **`services/ml-service/src/engines/galileo_engine.py`**
   - Added async GPU client integration
   - Modified GCN model to use `torch.device` selection
   - Data moved to GPU when acquired
   - Added `analyze_async()` method with GPU coordination
   - Updated insights to show GPU acceleration indicator

7. **`services/ml-service/src/engines/titan_engine.py`**
   - Added optional XGBoost with GPU histogram method
   - Created `_create_model_configs()` for dynamic model selection
   - Added `analyze_async()` method with GPU coordination
   - Added `use_gpu` config option
   - Updated insights and PremiumResult to show GPU usage

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_COORDINATOR_URL` | `http://gpu-coordinator:8002` | URL of GPU Coordinator |
| `GPU_ENABLED` | `true` | Enable GPU features |
| `GPU_REQUEST_TIMEOUT` | `15` | Timeout for GPU acquisition (seconds) |
| `GPU_FALLBACK_TO_CPU` | `true` | Fall back to CPU if GPU unavailable |

### Engine Config Options

Each engine now accepts a `use_gpu` parameter in its config:

```python
config = {
    'use_gpu': True,  # Attempt GPU acceleration
    # ... other engine-specific options
}
result = engine.analyze(df, config)
```

## GPU Support by Engine

| Engine | GPU Library | Method |
|--------|-------------|--------|
| Mirror (CTGAN) | PyTorch | `cuda=True` in CTGANSynthesizer |
| Galileo (GCN) | PyTorch Geometric | `.to(device)` for model and data |
| Titan (XGBoost) | XGBoost | `tree_method='gpu_hist'` |

## Usage Example

```python
from engines.mirror_engine import MirrorEngine
from core.gpu_client import init_gpu_client

# Initialize GPU client at startup
init_gpu_client(coordinator_url="http://gpu-coordinator:8002")

# Create engine (will use singleton GPU client)
engine = MirrorEngine()

# Run analysis - GPU will be automatically acquired/released
config = {
    'epochs': 100,
    'num_rows': 1000,
    'use_gpu': True  # Request GPU acceleration
}

result = engine.analyze(df, config)

# Check if GPU was used
print(f"Used GPU: {result['used_gpu']}")  # True or False
```

## Backward Compatibility

- All engines maintain synchronous `analyze()` method signature
- Engines work without GPU client (fall back to CPU)
- Existing code that doesn't use `use_gpu` config will default to GPU if available
- Engines gracefully handle GPU acquisition failures

## Error Handling

1. **GPU Unavailable**: Falls back to CPU, logs warning
2. **Coordinator Unreachable**: Falls back to CPU after retries
3. **Acquisition Timeout**: Falls back to CPU
4. **Release Failure**: State cleaned up regardless, error logged

## Testing

Run the GPU-related tests:

```bash
cd services/ml-service
pytest tests/test_gpu_client.py -v
pytest tests/test_engine_gpu_integration.py -v
```

## Performance Expectations

Based on your GTX 1660 Ti (6GB VRAM):

| Engine | Expected Speedup | Notes |
|--------|-----------------|-------|
| Mirror (CTGAN) | 2-3x | GAN training parallelizes well |
| Galileo (GCN) | 2-4x | Small graphs benefit less than large |
| Titan (XGBoost) | 3-10x | GPU histogram very efficient |

**Note**: Actual speedup depends on dataset size and model complexity.

## Future Improvements

1. **GPU Memory Management**: Track VRAM usage to prevent OOM
2. **Multi-GPU Support**: Enable spreading across multiple GPUs
3. **Priority Levels**: Different priorities for batch vs real-time
4. **GPU Metrics**: Collect and expose GPU utilization metrics
