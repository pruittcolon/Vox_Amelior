"""
Validators - Documentation and architecture validation

Provides CLI commands for validation:
- Architecture documentation (ARCHITECTURE.md)
- Configuration consistency
- API coverage
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATE_ARCH_SCRIPT = REPO_ROOT / "scripts" / "validate_architecture.py"


def handle_verify_command(json_output: bool = False, check: str = None) -> int:
    """Handle documentation validation commands"""
    
    if not VALIDATE_ARCH_SCRIPT.exists():
        print(f"Error: validate_architecture.py not found", file=sys.stderr)
        return 2
    
    print("Validating ARCHITECTURE.md against codebase...")
    print()
    
    cmd = ["python3", str(VALIDATE_ARCH_SCRIPT)]
    
    if json_output:
        cmd.append("--json")
    
    if check:
        # Future: support --check flag to run specific validation
        print(f"Note: Specific check filtering ({check}) not yet implemented", file=sys.stderr)
        print("Running all checks...", file=sys.stderr)
    
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


if __name__ == "__main__":
    # Allow standalone usage
    json_output = "--json" in sys.argv
    sys.exit(handle_verify_command(json_output=json_output))
