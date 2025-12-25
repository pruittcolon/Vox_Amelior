#!/usr/bin/env python3
"""
Vulnerability Scanning Script for Nemo Server
ISO 27002 Supply Chain Security

Scans all service dependencies for known vulnerabilities.
Use in CI/CD to enforce security thresholds.
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


SERVICES_DIR = Path(__file__).parent.parent / "services"


def install_safety():
    """Ensure safety is installed."""
    try:
        subprocess.run([sys.executable, "-m", "safety", "--version"], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing safety...")
        subprocess.run([sys.executable, "-m", "pip", "install", "safety", "-q"])
        return True


def scan_service(service_path: Path) -> Dict[str, Any]:
    """Scan a single service for vulnerabilities."""
    service_name = service_path.name
    requirements_file = service_path / "requirements.txt"
    
    if not requirements_file.exists():
        return {"service": service_name, "status": "skipped", "reason": "no requirements.txt"}
    
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "safety", "check",
                "-r", str(requirements_file),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        vulnerabilities = []
        if result.stdout:
            try:
                data = json.loads(result.stdout)
                if isinstance(data, list):
                    vulnerabilities = data
            except json.JSONDecodeError:
                pass
        
        # Count by severity
        high_critical = len([v for v in vulnerabilities if v.get("severity", "").upper() in ["HIGH", "CRITICAL"]])
        
        return {
            "service": service_name,
            "status": "scanned",
            "total_vulnerabilities": len(vulnerabilities),
            "high_critical": high_critical,
            "vulnerabilities": vulnerabilities[:5],  # Limit output
        }
    except subprocess.TimeoutExpired:
        return {"service": service_name, "status": "timeout"}
    except Exception as e:
        return {"service": service_name, "status": "error", "error": str(e)}


def scan_all_services(fail_on_high: bool = True) -> Dict[str, Any]:
    """Scan all services for vulnerabilities."""
    print("=" * 60)
    print("Vulnerability Scan for Nemo Server")
    print(f"Scanned at: {datetime.now().isoformat()}")
    print("=" * 60)
    
    install_safety()
    
    results = []
    total_high_critical = 0
    
    if SERVICES_DIR.exists():
        for service_dir in SERVICES_DIR.iterdir():
            if service_dir.is_dir() and not service_dir.name.startswith("."):
                print(f"\n🔍 Scanning {service_dir.name}...")
                result = scan_service(service_dir)
                results.append(result)
                
                if result["status"] == "scanned":
                    vulns = result["total_vulnerabilities"]
                    high = result["high_critical"]
                    total_high_critical += high
                    
                    if high > 0:
                        print(f"   ⚠️ {vulns} vulnerabilities ({high} HIGH/CRITICAL)")
                    elif vulns > 0:
                        print(f"   ⚠️ {vulns} vulnerabilities (no HIGH/CRITICAL)")
                    else:
                        print(f"   ✅ No known vulnerabilities")
                elif result["status"] == "skipped":
                    print(f"   ⏭️ Skipped: {result['reason']}")
                else:
                    print(f"   ❌ Error: {result.get('error', 'unknown')}")
    
    summary = {
        "scanned_at": datetime.now().isoformat(),
        "services": results,
        "total_services": len(results),
        "total_high_critical": total_high_critical,
        "passed": total_high_critical == 0,
    }
    
    print("\n" + "=" * 60)
    if total_high_critical > 0:
        print(f"❌ FAILED: {total_high_critical} HIGH/CRITICAL vulnerabilities found")
        if fail_on_high:
            print("   Set --no-fail to continue despite vulnerabilities")
    else:
        print("✅ PASSED: No HIGH/CRITICAL vulnerabilities")
    print("=" * 60)
    
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scan for vulnerabilities")
    parser.add_argument("--no-fail", action="store_true", help="Don't fail on HIGH/CRITICAL")
    args = parser.parse_args()
    
    result = scan_all_services(fail_on_high=not args.no_fail)
    
    if not result["passed"] and not args.no_fail:
        sys.exit(1)
