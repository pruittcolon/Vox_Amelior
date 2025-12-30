"""
Service Manager - Docker service control and health checks

Provides CLI commands to manage Nemo Server services:
- Start/stop/restart services
- Check health status
- View logs
- Run service-specific tests
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "docker"
START_SCRIPT = REPO_ROOT / "scripts" / "start.sh"

# Service name mapping (CLI name -> Docker service name)
SERVICE_MAP = {
    "gateway": "refactored_gateway",
    "gemma": "refactored_gemma-service",
    "gpu-coordinator": "refactored_gpu-coordinator",
    "transcription": "refactored_transcription-service",
    "rag": "refactored_rag-service",
    "emotion": "refactored_emotion-service",
    "ml-service": "refactored_ml-service",
    "insights": "refactored_insights-service",
}


def run_docker_compose(*args, check=False):
    """Run docker-compose command"""
    cmd = ["docker-compose"] + list(args)
    result = subprocess.run(cmd, cwd=DOCKER_DIR, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
    
    return result


def get_service_names(service_name: str):
    """Get Docker service name(s) for CLI service name"""
    if service_name == "all":
        return list(SERVICE_MAP.values())
    return [SERVICE_MAP.get(service_name, service_name)]


def handle_service_command(service_name: str, action: str, follow: bool = False) -> int:
    """Handle service management commands"""
    
    docker_services = get_service_names(service_name)
    
    if action == "start":
        print(f"Starting {service_name}...")
        if service_name == "all":
            # Use start.sh for full stack startup
            if not START_SCRIPT.exists():
                print(f"Error: start.sh not found at {START_SCRIPT}", file=sys.stderr)
                return 1
            result = subprocess.run(
                ["bash", str(START_SCRIPT), "--no-browser", "--no-logs"],
                cwd=REPO_ROOT
            )
            return result.returncode
        else:
            # Start specific service
            for svc in docker_services:
                run_docker_compose("up", "-d", svc, check=True)
            print(f"✓ {service_name} started")
            return 0
    
    elif action == "stop":
        print(f"Stopping {service_name}...")
        for svc in docker_services:
            run_docker_compose("stop", svc, check=True)
        print(f"✓ {service_name} stopped")
        return 0
    
    elif action == "restart":
        print(f"Restarting {service_name}...")
        for svc in docker_services:
            run_docker_compose("restart", svc, check=True)
        print(f"✓ {service_name} restarted")
        return 0
    
    elif action == "logs":
        if follow:
            # Follow logs (blocks until Ctrl+C)
            for svc in docker_services:
                subprocess.run(["docker-compose", "logs", "-f", svc], cwd=DOCKER_DIR)
        else:
            # Show last logs
            for svc in docker_services:
                result = run_docker_compose("logs", "--tail=50", svc)
                print(result.stdout)
        return 0
    
    elif action == "health":
        print(f"Checking health of {service_name}...")
        
        # Map services to their health check URLs
        health_urls = {
            "gateway": "http://localhost:8000/health",
            "gemma": "http://localhost:8001/health",
            "gpu-coordinator": "http://localhost:8002/health",
            "transcription": "http://localhost:8003/health",
            "rag": "http://localhost:8004/health",
            "emotion": "http://localhost:8005/health",
            "ml-service": "http://localhost:8006/health",
            "insights": "http://localhost:8010/health",
        }
        
        if service_name == "all":
            services_to_check = list(health_urls.keys())
        else:
            services_to_check = [service_name]
        
        all_healthy = True
        for svc in services_to_check:
            url = health_urls.get(svc)
            if not url:
                continue
            
            result = subprocess.run(
                ["curl", "-sf", url],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✓ {svc}: healthy")
            else:
                print(f"✗ {svc}: unhealthy or unreachable")
                all_healthy = False
        
        return 0 if all_healthy else 1
    
    elif action == "test":
        # Delegate to test runner
        from nemo import test_runner
        return test_runner.handle_test_command(service_name)
    
    else:
        print(f"Unknown action: {action}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Allow standalone usage
    if len(sys.argv) < 3:
        print("Usage: python -m nemo.service_manager <service> <action>", file=sys.stderr)
        sys.exit(1)
    
    sys.exit(handle_service_command(sys.argv[1], sys.argv[2]))
