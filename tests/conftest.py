"""
Nemo Server Test Fixtures
Shared fixtures for all test modules.
pytest 9.0 compatible.
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest
from typing import AsyncGenerator, Dict

# Fix import paths for hyphenated service directories
# This allows importing from services/api-gateway/src as 'src.*' and 'core.*'
_project_root = Path(__file__).parent.parent
_api_gateway_src = _project_root / "services" / "api-gateway" / "src"
_api_gateway_core = _project_root / "services" / "api-gateway" / "src" / "core"

# Add project root for 'shared.*' imports
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# Add api-gateway/src for 'src.*' imports
if str(_api_gateway_src) not in sys.path:
    sys.path.insert(0, str(_api_gateway_src))
# Add api-gateway/src for 'core.*' imports (some tests import from core.dependencies)
if str(_api_gateway_src) not in sys.path:
    sys.path.insert(0, str(_api_gateway_src))

# Optional imports with graceful fallback
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# Configuration from environment
BASE_URL = os.environ.get("NEMO_URL", "http://localhost:8000")
TEST_USERNAME = os.environ.get("NEMO_TEST_USER", "admin")
TEST_PASSWORD = os.environ.get("NEMO_TEST_PASS", "admin123")


# ============================================================
# Session-scoped fixtures
# ============================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests (pytest 9.0 style)."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def http_client() -> AsyncGenerator:
    """Shared HTTP client for all tests."""
    if not HAS_HTTPX:
        pytest.skip("httpx not installed")
    
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=30.0,
        follow_redirects=True
    ) as client:
        yield client


@pytest.fixture(scope="session")
async def auth_headers(http_client) -> Dict[str, str]:
    """Authenticate and return headers with session token."""
    response = await http_client.post("/api/auth/login", json={
        "username": TEST_USERNAME,
        "password": TEST_PASSWORD
    })
    
    if response.status_code != 200:
        pytest.skip(f"Authentication failed: {response.status_code}")
    
    data = response.json()
    headers = {}
    
    if data.get("session_token"):
        headers["Authorization"] = f"Bearer {data['session_token']}"
    if data.get("csrf_token"):
        headers["X-CSRF-Token"] = data["csrf_token"]
    
    return headers


# ============================================================
# Function-scoped fixtures
# ============================================================

@pytest.fixture
def sample_csv_content() -> bytes:
    """Sample CSV data for ML tests."""
    return b"""id,value,category,target
1,10.5,A,1
2,20.3,B,0
3,15.7,A,1
4,25.1,C,0
5,12.9,B,1
6,18.2,A,0
7,22.4,C,1
8,14.1,B,0
9,19.8,A,1
10,16.5,C,0
"""


@pytest.fixture
def sample_json_data() -> dict:
    """Sample JSON data for API tests."""
    return {
        "prompt": "Hello, how are you?",
        "max_tokens": 50,
        "temperature": 0.7
    }


# ============================================================
# Markers for test categorization
# ============================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external deps)")
    config.addinivalue_line("markers", "integration: Integration tests (requires Docker)")
    config.addinivalue_line("markers", "e2e: End-to-end tests (full stack)")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "slow: Tests taking >5 seconds")
    config.addinivalue_line("markers", "ml: ML engine tests")
