"""
API Client - Wrapper for existing nemo_cli.py functionality

This module provides a compatibility layer to integrate the existing
nemo_cli.py (824 lines) into the unified CLI structure.

For now, it forwards to the original script. In future, we'll refactor
the original into proper modules.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NEMO_CLI_SCRIPT = REPO_ROOT / "scripts" / "nemo_cli.py"


def handle_api_command(api_command: str, api_args: list) -> int:
    """Handle API commands by forwarding to nemo_cli.py"""
    
    if not NEMO_CLI_SCRIPT.exists():
        print(f"Error: nemo_cli.py not found at {NEMO_CLI_SCRIPT}", file=sys.stderr)
        return 2
    
    # Forward to existing nemo_cli.py
    cmd = ["python3", str(NEMO_CLI_SCRIPT), api_command] + api_args
    
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


if __name__ == "__main__":
    # Allow standalone usage
    if len(sys.argv) < 2:
        print("Usage: python -m nemo.api_client <command> [args...]", file=sys.stderr)
        sys.exit(1)
    
    sys.exit(handle_api_command(sys.argv[1], sys.argv[2:]))
