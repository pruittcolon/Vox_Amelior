#!/usr/bin/env python3
"""
Nemo Browser Test Runner
========================
Entry point for running the modular Playwright browser test suite.

Usage:
    source scripts/.venv/bin/activate
    python scripts/run_browser_tests.py

Options:
    --headless    Run in headless mode (default: True)
    --visible     Run with visible browser
    --base-url    Override base URL (default: http://localhost:8000)
"""

import asyncio
import sys
import os

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from browser_tests.runner import TestRunner
from browser_tests.config import Config


async def main():
    """Main entry point"""
    # Parse simple command line args
    headless = "--visible" not in sys.argv
    base_url = "http://localhost:8000"
    
    for arg in sys.argv:
        if arg.startswith("--base-url="):
            base_url = arg.split("=")[1]

    # Create config
    config = Config(
        base_url=base_url,
        headless=headless,
    )

    # Run tests
    runner = TestRunner(config)
    await runner.run_all()
    
    # Print summary
    success = runner.print_summary()
    
    # Generate report
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "browser_test_report.json"
    )
    runner.generate_report(report_path)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
