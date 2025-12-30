
import asyncio
import httpx
import json
import logging
import subprocess
import time
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GPU_TEST")

GATEWAY_URL = "http://localhost:8000"
GEMMA_URL = "http://localhost:8001"
COORDINATOR_URL = "http://localhost:8002"
TRANSCRIPTION_URL = "http://localhost:8003"

def get_vram_usage():
    """Get VRAM usage per process via nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return {}
        
        usage = {}
        for line in result.stdout.strip().split("\n"):
            if not line: continue
            pid, mem = line.split(",")
            usage[int(pid)] = int(mem)
        return usage
    except Exception:
        return {}

def get_container_pids():
    """Map container names to PIDs"""
    pids = {}
    for name in ["refactored_gemma", "refactored_transcription"]:
        try:
            res = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Pid}}", name],
                capture_output=True, text=True
            )
            if res.returncode == 0:
                pids[name] = int(res.stdout.strip())
        except Exception:
            pass
    return pids

def check_vram_distribution(stage_name):
    """Check and log VRAM status"""
    logger.info(f"--- VRAM CHECK: {stage_name} ---")
    container_pids = get_container_pids()
    vram_map = get_vram_usage()
    
    # We might need to map container PID to actual process PID if they differ (often do in Docker)
    # But for a rough check, we can look at the top consumers
    
    total_gemma = 0
    total_transcription = 0
    
    # Heuristic: Process names via nvidia-smi
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        print(res.stdout)
        
        # This is hard to attribute exactly without more complex logic, 
        # but we can rely on total utilization and logs.
    except:
        pass

async def run_test():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Startup State
        logger.info("Step 1: Verifying Startup State")
        check_vram_distribution("Startup")
        
        # Check Coordinator Status
        resp = await client.get(f"{COORDINATOR_URL}/status")
        status = resp.json()
        logger.info(f"Coordinator Status: {json.dumps(status, indent=2)}")
        
        if status['lock_status']['state'] != 'transcription':
             logger.warning("⚠️ Expected transcription to own GPU lock at startup (logical state)")

        # 2. Trigger Handoff (Simulate Chat)
        logger.info("Step 2: Triggering Handoff (Chat Request)")
        # We can call warmup or chat. Warmup moves to GPU.
        start_t = time.time()
        try:
            resp = await client.post(f"{GATEWAY_URL}/api/gemma/warmup") # Uses warmup endpoint
            logger.info(f"Warmup response: {resp.status_code}")
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            
        await asyncio.sleep(5) # Wait for handoff
        check_vram_distribution("During Gemma Warmup")

        # 3. Release
        logger.info("Step 3: Triggering Release")
        try:
            resp = await client.post(f"{GEMMA_URL}/move-to-cpu")
            logger.info(f"Release response: {resp.status_code}")
        except Exception as e:
            logger.error(f"Release failed: {e}")

        await asyncio.sleep(5)
        check_vram_distribution("After Release")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        # Continuous monitor mode
        while True:
            subprocess.run(["nvidia-smi"], check=False)
            time.sleep(1)
    else:
        asyncio.run(run_test())
