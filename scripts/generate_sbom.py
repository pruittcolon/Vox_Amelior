#!/usr/bin/env python3
"""
SBOM Generation Script for Nemo Server
ISO 27002 Supply Chain Security

Generates CycloneDX Software Bill of Materials (SBOM) for all services.
Use this during CI/CD to track dependencies and vulnerabilities.
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


SERVICES_DIR = Path(__file__).parent.parent / "services"
SBOM_OUTPUT_DIR = Path(__file__).parent.parent / "sbom"


def install_cyclonedx():
    """Ensure cyclonedx-py is installed."""
    try:
        import cyclonedx
        return True
    except ImportError:
        print("Installing cyclonedx-py...")
        subprocess.run([sys.executable, "-m", "pip", "install", "cyclonedx-py", "-q"])
        return True


def generate_sbom_for_service(service_path: Path, output_dir: Path) -> dict:
    """Generate SBOM for a single service."""
    service_name = service_path.name
    requirements_file = service_path / "requirements.txt"
    
    if not requirements_file.exists():
        return {"service": service_name, "status": "skipped", "reason": "no requirements.txt"}
    
    output_file = output_dir / f"{service_name}-sbom.json"
    
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "cyclonedx_py",
                "requirements",
                str(requirements_file),
                "--format", "json",
                "--output-file", str(output_file),
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            return {
                "service": service_name,
                "status": "success",
                "sbom_file": str(output_file)
            }
        else:
            return {
                "service": service_name,
                "status": "error",
                "error": result.stderr[:200]
            }
    except Exception as e:
        return {
            "service": service_name,
            "status": "error",
            "error": str(e)
        }


def generate_all_sboms():
    """Generate SBOMs for all services."""
    print("=" * 60)
    print("SBOM Generation for Nemo Server")
    print(f"Generated at: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Create output directory
    SBOM_OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Install cyclonedx if needed
    install_cyclonedx()
    
    results = []
    
    # Find all services
    if SERVICES_DIR.exists():
        for service_dir in SERVICES_DIR.iterdir():
            if service_dir.is_dir() and not service_dir.name.startswith("."):
                print(f"\n📦 Processing {service_dir.name}...")
                result = generate_sbom_for_service(service_dir, SBOM_OUTPUT_DIR)
                results.append(result)
                
                if result["status"] == "success":
                    print(f"   ✅ SBOM generated: {result['sbom_file']}")
                elif result["status"] == "skipped":
                    print(f"   ⚠️ Skipped: {result['reason']}")
                else:
                    print(f"   ❌ Error: {result.get('error', 'unknown')}")
    
    # Generate summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "services": results,
        "total": len(results),
        "success": len([r for r in results if r["status"] == "success"]),
        "skipped": len([r for r in results if r["status"] == "skipped"]),
        "errors": len([r for r in results if r["status"] == "error"]),
    }
    
    summary_file = SBOM_OUTPUT_DIR / "sbom-summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Summary: {summary['success']} success, {summary['skipped']} skipped, {summary['errors']} errors")
    print(f"Summary file: {summary_file}")
    print("=" * 60)
    
    return summary


if __name__ == "__main__":
    generate_all_sboms()
