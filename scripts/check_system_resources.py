#!/usr/bin/env python3
"""
Check System Resources - RAM/VRAM Verification.

Verifies sufficient resources before starting services.
Required per user rules to prevent OOM conditions.

Usage:
    python scripts/check_system_resources.py
    python scripts/check_system_resources.py --require-gpu
    python scripts/check_system_resources.py --min-ram 8

Exit codes:
    0: Resources sufficient
    1: Insufficient RAM
    2: Insufficient VRAM (when --require-gpu)
"""

import argparse
import subprocess
import sys
from typing import Tuple, Optional


# Minimum requirements (configurable via CLI)
DEFAULT_MIN_RAM_GB = 4.0
DEFAULT_MIN_VRAM_GB = 6.0  # For Gemma/transcription services


def check_ram() -> Tuple[float, float]:
    """
    Check available RAM from /proc/meminfo.

    Returns:
        Tuple of (total_gb, available_gb).
    """
    try:
        with open('/proc/meminfo') as f:
            lines = f.read().strip().split('\n')
        
        mem_info = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                mem_info[key.strip()] = value.strip()
        
        total_kb = int(mem_info.get('MemTotal', '0').split()[0])
        available_kb = int(mem_info.get('MemAvailable', '0').split()[0])
        
        return total_kb / 1024 / 1024, available_kb / 1024 / 1024
    except Exception as e:
        print(f"Error reading /proc/meminfo: {e}")
        return 0.0, 0.0


def check_vram() -> Tuple[float, float]:
    """
    Check available VRAM via nvidia-smi.

    Returns:
        Tuple of (total_gb, free_gb). Returns (0, 0) if no GPU available.
    """
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=memory.total,memory.free',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            # Parse first GPU (primary)
            line = result.stdout.strip().split('\n')[0]
            total_mb, free_mb = map(float, line.split(', '))
            return total_mb / 1024, free_mb / 1024
    except FileNotFoundError:
        pass  # nvidia-smi not found
    except subprocess.TimeoutExpired:
        print("Warning: nvidia-smi timed out")
    except Exception as e:
        print(f"Warning: Error checking VRAM: {e}")
    
    return 0.0, 0.0


def check_swap() -> Tuple[float, float]:
    """
    Check available swap from /proc/meminfo.

    Returns:
        Tuple of (total_gb, free_gb).
    """
    try:
        with open('/proc/meminfo') as f:
            lines = f.read().strip().split('\n')
        
        mem_info = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                mem_info[key.strip()] = value.strip()
        
        total_kb = int(mem_info.get('SwapTotal', '0').split()[0])
        free_kb = int(mem_info.get('SwapFree', '0').split()[0])
        
        return total_kb / 1024 / 1024, free_kb / 1024 / 1024
    except Exception:
        return 0.0, 0.0


def format_size(gb: float) -> str:
    """Format size in GB with appropriate precision."""
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    return f"{gb * 1024:.0f} MB"


def main() -> int:
    """
    Main entry point for resource check.

    Returns:
        Exit code (0=success, 1=insufficient RAM, 2=insufficient VRAM).
    """
    parser = argparse.ArgumentParser(
        description="Check system resources before starting services"
    )
    parser.add_argument(
        '--min-ram',
        type=float,
        default=DEFAULT_MIN_RAM_GB,
        help=f"Minimum required RAM in GB (default: {DEFAULT_MIN_RAM_GB})"
    )
    parser.add_argument(
        '--min-vram',
        type=float,
        default=DEFAULT_MIN_VRAM_GB,
        help=f"Minimum required VRAM in GB (default: {DEFAULT_MIN_VRAM_GB})"
    )
    parser.add_argument(
        '--require-gpu',
        action='store_true',
        help="Fail if GPU/VRAM requirements not met"
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help="Output results as JSON"
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress output (exit code only)"
    )
    
    args = parser.parse_args()
    
    # Collect resource info
    ram_total, ram_available = check_ram()
    vram_total, vram_available = check_vram()
    swap_total, swap_free = check_swap()
    
    ram_ok = ram_available >= args.min_ram
    vram_ok = vram_available >= args.min_vram
    gpu_available = vram_total > 0
    
    # JSON output mode
    if args.json:
        import json
        result = {
            "ram": {
                "total_gb": round(ram_total, 2),
                "available_gb": round(ram_available, 2),
                "required_gb": args.min_ram,
                "sufficient": ram_ok
            },
            "vram": {
                "total_gb": round(vram_total, 2),
                "available_gb": round(vram_available, 2),
                "required_gb": args.min_vram,
                "sufficient": vram_ok,
                "gpu_detected": gpu_available
            },
            "swap": {
                "total_gb": round(swap_total, 2),
                "free_gb": round(swap_free, 2)
            },
            "overall_ok": ram_ok and (not args.require_gpu or vram_ok)
        }
        print(json.dumps(result, indent=2))
        return 0 if result["overall_ok"] else 1
    
    # Human-readable output
    if not args.quiet:
        print("╔══════════════════════════════════════════════╗")
        print("║         System Resource Check                ║")
        print("╠══════════════════════════════════════════════╣")
        
        # RAM
        ram_status = "✅" if ram_ok else "❌"
        print(f"║  RAM:  {format_size(ram_available):>8} / {format_size(ram_total):<8} {ram_status}      ║")
        
        # VRAM
        if gpu_available:
            vram_status = "✅" if vram_ok else "⚠️ "
            print(f"║  VRAM: {format_size(vram_available):>8} / {format_size(vram_total):<8} {vram_status}      ║")
        else:
            print("║  VRAM: No GPU detected                       ║")
        
        # Swap
        print(f"║  Swap: {format_size(swap_free):>8} / {format_size(swap_total):<8}           ║")
        
        print("╠══════════════════════════════════════════════╣")
        
        # Overall status
        if not ram_ok:
            print(f"║  ❌ Insufficient RAM!                        ║")
            print(f"║     Need: {format_size(args.min_ram)}, Have: {format_size(ram_available):<16} ║")
        elif args.require_gpu and not vram_ok:
            print(f"║  ❌ Insufficient VRAM!                       ║")
            print(f"║     Need: {format_size(args.min_vram)}, Have: {format_size(vram_available):<16} ║")
        else:
            print("║  ✅ Resources sufficient for operation       ║")
        
        print("╚══════════════════════════════════════════════╝")
    
    # Determine exit code
    if not ram_ok:
        return 1
    if args.require_gpu and not vram_ok:
        return 2
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
