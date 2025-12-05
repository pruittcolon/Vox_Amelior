"""
Test Runner - Orchestrates test execution

Provides CLI commands to run tests:
- Full test suite (full_system_test.sh)
- Service-specific tests
- Feature-specific tests
- JSON output for CI/CD
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_SYSTEM_TEST = REPO_ROOT / "scripts" / "full_system_test.sh"


def handle_test_command(
    service: str,
    feature: Optional[str] = None,
    json_output: bool = False,
    verbose: bool = False
) -> int:
    """Handle test execution commands"""
    
    if service == "all":
        return run_full_system_test(json_output=json_output, verbose=verbose)
    else:
        return run_service_test(service, feature=feature, json_output=json_output, verbose=verbose)


def run_full_system_test(json_output: bool = False, verbose: bool = False) -> int:
    """Run full system test suite using full_system_test.sh"""
    
    if not FULL_SYSTEM_TEST.exists():
        print(f"Error: full_system_test.sh not found at {FULL_SYSTEM_TEST}", file=sys.stderr)
        return 2
    
    print("Running full system test suite...")
    print(f"Script: {FULL_SYSTEM_TEST.relative_to(REPO_ROOT)}")
    print()
    
    cmd = [
        "bash",
        str(FULL_SYSTEM_TEST),
        "--all"
    ]
    
    if json_output:
        cmd.append("--json")
    
    if verbose:
        cmd.append("--verbose")
    
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def run_service_test(
    service: str,
    feature: Optional[str] = None,
    json_output: bool = False,
    verbose: bool = False
) -> int:
    """Run service-specific tests"""
    
    # Map service names to test flags for full_system_test.sh
    service_flags = {
        "gateway": "--health",  # Gateway is tested via health checks
        "gemma": "--gemma",
        "gpu-coordinator": "--health",  # GPU coordinator tested indirectly
        "transcription": "--transcription",
        "rag": "--rag",
        "emotion": "--emotions",
        "ml-service": "--health",  # ML service has separate test suite
        "insights": "--health",
    }
    
    flag = service_flags.get(service)
    if not flag:
        print(f"Error: Unknown service '{service}'", file=sys.stderr)
        return 1
    
    print(f"Running tests for {service}...")
    
    cmd = [
        "bash",
        str(FULL_SYSTEM_TEST),
        flag
    ]
    
    if json_output:
        cmd.append("--json")
    
    if verbose:
        cmd.append("--verbose")
    
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def run_feature_test(service: str, feature: str) -> int:
    """Run feature-specific test (future implementation)"""
    
    # For now, call service /cli/test endpoint if available
    service_ports = {
        "gateway": 8000,
        "gemma": 8001,
        "gpu-coordinator": 8002,
        "transcription": 8003,
        "rag": 8004,
        "emotion": 8005,
        "ml-service": 8006,
        "insights": 8010,
    }
    
    port = service_ports.get(service)
    if not port:
        print(f"Error: Unknown service '{service}'", file=sys.stderr)
        return 1
    
    url = f"http://localhost:{port}/cli/test/{feature}"
    
    result = subprocess.run(
        ["curl", "-sf", url],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            print(json.dumps(data, indent=2))
            return 0 if data.get("passed") else 1
        except Exception:
            print(result.stdout)
            return 0
    else:
        print(f"Error: Feature test endpoint not available", file=sys.stderr)
        print(f"  Service: {service}", file=sys.stderr)
        print(f"  Feature: {feature}", file=sys.stderr)
        print(f"  URL: {url}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Allow standalone usage
    if len(sys.argv) < 2:
        print("Usage: python -m nemo.test_runner <service> [--feature <name>] [--json] [--verbose]", file=sys.stderr)
        sys.exit(1)
    
    service = sys.argv[1]
    feature = None
    json_output = "--json" in sys.argv
    verbose = "--verbose" in sys.argv
    
    if "--feature" in sys.argv:
        idx = sys.argv.index("--feature")
        if idx + 1 < len(sys.argv):
            feature = sys.argv[idx + 1]
    
    sys.exit(handle_test_command(service, feature=feature, json_output=json_output, verbose=verbose))
