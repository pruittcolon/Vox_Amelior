# Nemo Unified CLI

Comprehensive command-line interface for all Nemo Server operations.

## Quick Start

```bash
# From scripts/ directory
python3 -m nemo <command> [options]

# OR use wrapper script
./nemo.sh <command> [options]

# OR create symlink for global access
ln -s /path/to/Nemo_Server/scripts/nemo.sh /usr/local/bin/nemo
nemo <command> [options]
```

## Commands

### Documentation Validation
```bash
# Validate ARCHITECTURE.md against codebase
nemo verify

# JSON output for CI/CD
nemo verify --json

# Alias
nemo docs validate
```

### Service Management
```bash
# Check service health
nemo service gateway health
nemo service all health

# View service logs
nemo service gemma logs
nemo service transcription logs --follow

# Start/stop/restart services
nemo service all start      # Start full stack (uses start.sh)
nemo service gemma restart
nemo service rag stop
```

### Testing
```bash
# Run full test suite
nemo test all

# Test specific service
nemo test transcription
nemo test gemma --verbose

# JSON output
nemo test rag --json

# Feature-specific (future)
nemo test transcription --feature diarization
```

### API Interactions
```bash
# All existing nemo_cli.py commands available via:
nemo api <command> [options]

# Examples:
nemo api auth-info
nemo api transcribe --file audio.wav
nemo api chat --message "Summarize recent conversations"
nemo api semantic --query "security concerns"
nemo api gemma-analyze --statements "This is great!" "Maybe not..."
```

## Available Services

| Service | Name | Port | Health Check |
|---------|------|------|--------------|
| API Gateway | `gateway` | 8000 | `nemo service gateway health` |
| Gemma LLM | `gemma` | 8001 | `nemo service gemma health` |
| GPU Coordinator | `gpu-coordinator` | 8002 | `nemo service gpu-coordinator health` |
| Transcription | `transcription` | 8003 | `nemo service transcription health` |
| RAG | `rag` | 8004 | `nemo service rag health` |
| Emotion | `emotion` | 8005 | `nemo service emotion health` |
| ML Service | `ml-service` | 8006 | `nemo service ml-service health` |
| Insights | `insights` | 8010 | `nemo service insights health` |

## Architecture

The unified CLI is composed of 6 modules:

```
scripts/nemo/
├── __init__.py          # Package metadata
├── __main__.py          # Entry point and command routing
├── api_client.py        # Wrapper for nemo_cli.py (824 lines)
├── service_manager.py   # Docker service control
├── test_runner.py       # Test orchestration
└── validators.py        # Architecture validation
```

### Module Responsibilities

**`__main__.py`** - Command Router
- Parses CLI arguments
- Routes to appropriate handler module
- Provides unified help system

**`service_manager.py`** - Service Control
- Start/stop/restart Docker services
- Health check HTTP endpoints
- Stream service logs
- Maps CLI names to Docker service names

**`test_runner.py`** - Test Orchestration
- Runs `full_system_test.sh` for full suite
- Maps service names to test flags
- Supports JSON output for CI/CD
- Future: Service-specific test endpoints

**`validators.py`** - Documentation Validation
- Wraps `validate_architecture.py`
- Enforces code-docs synchronization
- Mandatory at end of every phase

**`api_client.py`** - API Interactions (Future Refactor)
- Currently forwards to nemo_cli.py
- Future: Refactor into proper module structure
- Maintains backward compatibility

## Examples

### Development Workflow
```bash
# 1. Make code changes
vim services/rag-service/src/main.py

# 2. Restart service
nemo service rag restart

# 3. Check health
nemo service rag health

# 4. Run tests
nemo test rag

# 5. Validate documentation
nemo verify

# 6. If validation fails, update docs
vim ARCHITECTURE.md
nemo verify
```

### Testing Before Commit
```bash
# Run full test suite
nemo test all

# Validate architecture
nemo verify

# Both must pass before committing
```

### Service Health Dashboard
```bash
# Quick health check of all services
nemo service all health

# Output:
# [PASS] gateway: healthy
# [PASS] gemma: healthy
# [PASS] gpu-coordinator: healthy
# [PASS] transcription: healthy
# [PASS] rag: healthy
# [PASS] emotion: healthy
# [FAIL] ml-service: unhealthy or unreachable
# [PASS] insights: healthy
```

### CI/CD Integration
```bash
# In GitHub Actions / GitLab CI
- name: Validate Architecture
  run: |
    cd scripts
    python3 -m nemo verify --json > validation.json
    
- name: Run Tests
  run: |
    cd scripts
    python3 -m nemo test all --json > test_results.json
```

## Help System

```bash
# Top-level help
nemo --help

# Command-specific help
nemo service --help
nemo test --help
nemo verify --help
nemo api --help
```

## Future Enhancements (Phase 2+)

1. **Service-Specific Test Endpoints**
   - Each service exposes `/cli/test` HTTP endpoint
   - `nemo test <service>` calls endpoint directly
   - Feature-specific tests: `nemo test rag --feature embeddings`

2. **Auto-Fix Mode**
   - `nemo docs validate --fix`
   - Suggests ARCHITECTURE.md updates based on code

3. **Diagnostic Mode**
   - `nemo diagnose <test-name>`
   - Analyzes failures and suggests fixes

4. **Interactive Mode**
   - `nemo interactive`
   - REPL for exploring services

5. **Configuration Management**
   - `nemo config list`
   - `nemo config set <key> <value>`

## Backward Compatibility

The original `nemo_cli.py` remains functional:
```bash
# Old way (still works)
python3 scripts/nemo_cli.py transcribe --file audio.wav

# New way (recommended)
python3 -m nemo api transcribe --file audio.wav
```

## Installation (Optional)

For global access, create a symlink:
```bash
sudo ln -s /path/to/Nemo_Server/scripts/nemo.sh /usr/local/bin/nemo
nemo verify  # Works from anywhere!
```

## Exit Codes

All commands follow Unix conventions:
- `0`: Success
- `1`: Failure (tests failed, validation errors, service unhealthy)
- `2`: Critical error (missing files, parse errors)

## Mandatory Validation Policy

**CRITICAL REQUIREMENT:** `nemo verify` MUST pass (exit code 0) at the end of every development phase.

```bash
# End of phase checklist:
nemo test all          # All tests pass
nemo verify            # Documentation synchronized
git commit -m "..."    # Only commit if both pass
```

This ensures `ARCHITECTURE.md` never contradicts the codebase.
