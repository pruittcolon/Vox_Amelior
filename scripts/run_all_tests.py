#!/usr/bin/env python3
"""
Comprehensive Test Runner for Enterprise Verification.

Runs all test suites, generates reports, and validates enterprise readiness.

Usage:
    python scripts/run_all_tests.py              # Run all tests
    python scripts/run_all_tests.py --category security
    python scripts/run_all_tests.py --generate-report
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TestResult:
    """Result of a test suite execution."""
    category: str
    name: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration: float = 0.0
    output: str = ""
    
    @property
    def total(self) -> int:
        return self.passed + self.failed + self.skipped + self.errors
    
    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100
    
    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "name": self.name,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "total": self.total,
            "success_rate": round(self.success_rate, 2),
            "duration": round(self.duration, 2),
        }


@dataclass
class TestReport:
    """Comprehensive test report."""
    timestamp: datetime = field(default_factory=datetime.now)
    results: list[TestResult] = field(default_factory=list)
    total_duration: float = 0.0
    
    @property
    def total_passed(self) -> int:
        return sum(r.passed for r in self.results)
    
    @property
    def total_failed(self) -> int:
        return sum(r.failed for r in self.results)
    
    @property
    def total_tests(self) -> int:
        return sum(r.total for r in self.results)
    
    @property
    def overall_success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.total_passed / self.total_tests) * 100
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_tests": self.total_tests,
                "passed": self.total_passed,
                "failed": self.total_failed,
                "success_rate": round(self.overall_success_rate, 2),
                "duration_seconds": round(self.total_duration, 2),
            },
            "results": [r.to_dict() for r in self.results],
        }


# Test suite definitions
TEST_SUITES = {
    "unit": {
        "path": "tests/unit",
        "description": "Unit tests",
    },
    "integration": {
        "path": "tests/integration",
        "description": "Integration tests",
    },
    "security": {
        "path": "tests/integration/test_week*security*.py tests/integration/test_audit*.py tests/integration/test_pii*.py",
        "description": "Security-focused tests",
        "pattern": True,
    },
    "week5": {
        "path": "tests/integration/test_week5_sso.py",
        "description": "Week 5: SSO",
    },
    "week6": {
        "path": "tests/integration/test_week6_ingestion.py",
        "description": "Week 6: Ingestion",
    },
    "week7": {
        "path": "tests/integration/test_week7_connectors.py",
        "description": "Week 7: Connectors",
    },
    "week8": {
        "path": "tests/integration/test_week8_observability.py",
        "description": "Week 8: Observability",
    },
    "week9": {
        "path": "tests/integration/test_week9_ai_governance.py",
        "description": "Week 9: AI Governance",
    },
    "week10": {
        "path": "tests/integration/test_week10_reliability.py",
        "description": "Week 10: Reliability",
    },
    "week11": {
        "path": "tests/integration/test_week11_security.py",
        "description": "Week 11: Frontend Security",
    },
}


def run_pytest(test_path: str, extra_args: list[str] | None = None) -> tuple[int, int, int, int, float, str]:
    """
    Run pytest and parse results.
    
    Returns: (passed, failed, skipped, errors, duration, output)
    """
    cmd = [
        sys.executable, "-m", "pytest",
        test_path,
        "-v",
        "--tb=short",
        "-q",
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    start = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=300,  # 5 minute timeout
        )
        output = result.stdout + result.stderr
        duration = time.time() - start
        
        # Parse pytest output
        passed, failed, skipped, errors = parse_pytest_output(output)
        
        return passed, failed, skipped, errors, duration, output
        
    except subprocess.TimeoutExpired:
        return 0, 0, 0, 1, time.time() - start, "Test timed out"
    except Exception as e:
        return 0, 0, 0, 1, time.time() - start, str(e)


def parse_pytest_output(output: str) -> tuple[int, int, int, int]:
    """Parse pytest summary line for counts."""
    import re
    
    # Look for summary line like "5 passed, 2 failed, 1 skipped"
    summary_pattern = r'(\d+)\s+passed'
    failed_pattern = r'(\d+)\s+failed'
    skipped_pattern = r'(\d+)\s+skipped'
    error_pattern = r'(\d+)\s+error'
    
    passed = 0
    failed = 0
    skipped = 0
    errors = 0
    
    match = re.search(summary_pattern, output)
    if match:
        passed = int(match.group(1))
    
    match = re.search(failed_pattern, output)
    if match:
        failed = int(match.group(1))
    
    match = re.search(skipped_pattern, output)
    if match:
        skipped = int(match.group(1))
    
    match = re.search(error_pattern, output)
    if match:
        errors = int(match.group(1))
    
    return passed, failed, skipped, errors


def run_test_suite(name: str, config: dict) -> TestResult:
    """Run a test suite and return result."""
    print(f"\n{'='*60}")
    print(f"Running: {config['description']} ({name})")
    print(f"{'='*60}")
    
    test_path = config["path"]
    
    passed, failed, skipped, errors, duration, output = run_pytest(test_path)
    
    result = TestResult(
        category=name,
        name=config["description"],
        passed=passed,
        failed=failed,
        skipped=skipped,
        errors=errors,
        duration=duration,
        output=output,
    )
    
    # Print summary
    status = "✅ PASS" if failed == 0 and errors == 0 else "❌ FAIL"
    print(f"\n{status}: {passed} passed, {failed} failed, {skipped} skipped in {duration:.2f}s")
    
    return result


def print_report(report: TestReport) -> None:
    """Print formatted report."""
    print("\n" + "="*80)
    print("TEST REPORT SUMMARY")
    print("="*80)
    print(f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Duration: {report.total_duration:.2f}s")
    print()
    
    print(f"{'Category':<20} {'Passed':>8} {'Failed':>8} {'Skipped':>8} {'Rate':>8}")
    print("-"*60)
    
    for result in report.results:
        rate = f"{result.success_rate:.1f}%"
        print(f"{result.category:<20} {result.passed:>8} {result.failed:>8} {result.skipped:>8} {rate:>8}")
    
    print("-"*60)
    print(f"{'TOTAL':<20} {report.total_passed:>8} {report.total_failed:>8} {'':>8} {report.overall_success_rate:.1f}%")
    print()
    
    if report.total_failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {report.total_failed} TESTS FAILED")


def save_report(report: TestReport, output_path: Path) -> None:
    """Save report to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run enterprise test suites")
    parser.add_argument(
        "--category",
        choices=list(TEST_SUITES.keys()) + ["all", "enterprise"],
        default="enterprise",
        help="Test category to run",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate JSON report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_reports/latest.json"),
        help="Report output path",
    )
    
    args = parser.parse_args()
    
    report = TestReport()
    start_time = time.time()
    
    # Determine which suites to run
    if args.category == "all":
        suites = TEST_SUITES
    elif args.category == "enterprise":
        # Run all week tests (5-11)
        suites = {k: v for k, v in TEST_SUITES.items() if k.startswith("week")}
    else:
        suites = {args.category: TEST_SUITES[args.category]}
    
    # Run test suites
    for name, config in suites.items():
        result = run_test_suite(name, config)
        report.results.append(result)
    
    report.total_duration = time.time() - start_time
    
    # Print summary
    print_report(report)
    
    # Save report if requested
    if args.generate_report:
        save_report(report, args.output)
    
    # Exit with error if any failures
    if report.total_failed > 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
