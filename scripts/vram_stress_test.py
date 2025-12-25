#!/usr/bin/env python3
"""
VRAM Stress Test for Gemma Service
Tests progressively larger context sizes to find VRAM limits.

Usage: python3 scripts/vram_stress_test.py
"""

import subprocess
import json
import time
import sys

def run_cmd(cmd: str) -> str:
    """Run a shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def get_vram_usage() -> tuple[int, int]:
    """Get current VRAM usage (used, free) in MB."""
    output = run_cmd("nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits")
    parts = output.split(",")
    if len(parts) >= 2:
        return int(parts[0].strip()), int(parts[1].strip())
    return 0, 0

def get_gemma_health() -> dict:
    """Get Gemma service health status."""
    output = run_cmd("docker exec refactored_gemma curl -s http://localhost:8001/health")
    try:
        return json.loads(output)
    except:
        return {}

def test_generation(prompt: str, max_tokens: int = 50) -> tuple[bool, float, str]:
    """Test a generation request. Returns (success, time_seconds, response)."""
    cmd = f'''docker exec refactored_gemma curl -s -X POST http://localhost:8001/generate \\
        -H "Content-Type: application/json" \\
        -d '{{"prompt": "{prompt}", "max_tokens": {max_tokens}, "temperature": 0.7}}'
    '''
    start = time.time()
    output = run_cmd(cmd)
    elapsed = time.time() - start
    
    try:
        result = json.loads(output)
        text = result.get("text", result.get("response", ""))
        return True, elapsed, text[:100]
    except:
        return False, elapsed, output[:100]

def restart_gemma_with_context(context_size: int, cache_type: str = "q8_0") -> bool:
    """Restart Gemma with new context size."""
    print(f"\n🔄 Restarting Gemma with context={context_size}, cache={cache_type}...")
    
    # Stop current container
    run_cmd("docker stop refactored_gemma 2>/dev/null")
    time.sleep(2)
    
    # Update environment and restart
    env_update = f"GEMMA_CONTEXT_SIZE={context_size} GEMMA_CACHE_TYPE={cache_type}"
    cmd = f"cd /home/pruittcolon/Desktop/Nemo_Server/docker && {env_update} docker compose up -d gemma-service"
    run_cmd(cmd)
    
    # Wait for healthy
    for i in range(60):
        time.sleep(2)
        health = get_gemma_health()
        if health.get("model_loaded"):
            print(f"   ✅ Gemma healthy after {(i+1)*2}s")
            return True
    
    print("   ❌ Gemma failed to start")
    return False

def run_stress_test():
    """Run progressive VRAM stress test."""
    print("=" * 60)
    print("🧪 VRAM STRESS TEST - Gemma 3 4B")
    print("=" * 60)
    
    # Test configurations: (context_size, cache_type)
    configs = [
        (2048, "q8_0"),   # Baseline
        (4096, "q8_0"),   # 2x context
        (8192, "q8_0"),   # 4x context
        (4096, "fp16"),   # Compare with FP16
        (8192, "fp16"),   # FP16 at 8K
        (16384, "q8_0"),  # Push Q8 limits
    ]
    
    results = []
    
    for ctx, cache in configs:
        print(f"\n{'='*60}")
        print(f"📊 Testing: Context={ctx}, Cache={cache}")
        print("="*60)
        
        # Get baseline VRAM
        vram_before = get_vram_usage()
        print(f"   VRAM before restart: {vram_before[0]} MB used, {vram_before[1]} MB free")
        
        # Restart with new config
        if not restart_gemma_with_context(ctx, cache):
            results.append({
                "context": ctx,
                "cache": cache,
                "success": False,
                "error": "Failed to start"
            })
            continue
        
        # Check VRAM after load
        time.sleep(5)
        vram_after = get_vram_usage()
        health = get_gemma_health()
        
        print(f"   VRAM after load: {vram_after[0]} MB used, {vram_after[1]} MB free")
        print(f"   Model VRAM (self-reported): {health.get('vram_used_mb', '?')} MB")
        print(f"   Cache type: {health.get('cache_type', '?')}")
        
        # Test generation
        print("   Testing generation...")
        success, gen_time, response = test_generation("List 3 random facts about space:")
        
        if success:
            print(f"   ✅ Generation succeeded in {gen_time:.2f}s")
            print(f"   Response: {response}...")
        else:
            print(f"   ❌ Generation failed: {response}")
        
        results.append({
            "context": ctx,
            "cache": cache,
            "success": success,
            "vram_used": vram_after[0],
            "vram_free": vram_after[1],
            "model_vram": health.get("vram_used_mb"),
            "gen_time": gen_time if success else None
        })
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 STRESS TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Context':>8} | {'Cache':>6} | {'VRAM':>8} | {'Free':>8} | {'Status':>10} | {'Gen Time':>10}")
    print("-" * 60)
    
    for r in results:
        status = "✅ PASS" if r.get("success") else "❌ FAIL"
        vram = f"{r.get('vram_used', '?')} MB" if r.get('vram_used') else "N/A"
        free = f"{r.get('vram_free', '?')} MB" if r.get('vram_free') else "N/A"
        gen = f"{r.get('gen_time', 0):.2f}s" if r.get("gen_time") else "N/A"
        print(f"{r['context']:>8} | {r['cache']:>6} | {vram:>8} | {free:>8} | {status:>10} | {gen:>10}")
    
    print("\n🏁 Stress test complete!")
    return results

if __name__ == "__main__":
    run_stress_test()
