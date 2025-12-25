#!/usr/bin/env python3
"""
n8n Integration Service - CLI Test Suite

Comprehensive testing for voice commands and emotion alerts using the actual service.

Usage:
    # Run all tests
    python test_n8n_cli.py

    # Run specific test
    python test_n8n_cli.py --test voice_command
    python test_n8n_cli.py --test emotion_alert
    python test_n8n_cli.py --test command_api

    # Test against remote service
    python test_n8n_cli.py --url http://n8n-service:8011

    # Verbose output
    python test_n8n_cli.py -v
"""

import argparse
import sys
import time
from dataclasses import dataclass
from enum import Enum

# Add project root to path for local imports
sys.path.insert(0, "/home/pruittcolon/Desktop/Nemo_Server/services/n8n-service")

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# For local testing without HTTP
try:
    from src.command_registry import CommandRegistry, get_command_registry
    from src.config import EMOTION_ALERT_SPEAKERS, EMOTION_ALERT_THRESHOLD
    from src.emotion_tracker import EmotionAlertTracker, get_emotion_tracker

    HAS_LOCAL = True
except ImportError:
    HAS_LOCAL = False


class TestResult(Enum):
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    SKIP = "⏭️ SKIP"
    ERROR = "💥 ERROR"


@dataclass
class TestCase:
    name: str
    description: str
    result: TestResult = TestResult.SKIP
    message: str = ""
    duration_ms: float = 0.0


class TestRunner:
    """CLI Test Runner for n8n Integration Service"""

    def __init__(self, service_url: str = "http://localhost:8011", verbose: bool = False):
        self.service_url = service_url.rstrip("/")
        self.verbose = verbose
        self.tests: list[TestCase] = []
        self.use_http = service_url.startswith("http")

    def log(self, msg: str, force: bool = False):
        if self.verbose or force:
            print(msg)

    def add_result(self, test: TestCase):
        self.tests.append(test)
        status = test.result.value
        print(f"{status} {test.name} ({test.duration_ms:.1f}ms)")
        if test.message and (self.verbose or test.result == TestResult.FAIL):
            print(f"   └─ {test.message}")

    # =========================================================================
    # Voice Command Tests
    # =========================================================================

    def test_voice_command_match_positive(self) -> TestCase:
        """Test that valid voice commands are detected"""
        test = TestCase(
            name="Voice Command - Positive Match", description="Commands matching the pattern should trigger"
        )
        start = time.time()

        test_phrases = [
            ("Honey, can you turn off the lights", "lights_off"),
            ("Honey, could you turn off the light please", "lights_off"),
            ("Honey, would you turn off the lights for me", "lights_off"),
        ]

        try:
            if self.use_http and HAS_HTTPX:
                # Use HTTP API
                for phrase, expected_cmd in test_phrases:
                    response = httpx.post(
                        f"{self.service_url}/process",
                        json={"segments": [{"text": phrase, "speaker": "pruitt"}], "job_id": "test-voice-positive"},
                        timeout=10.0,
                    )
                    result = response.json()
                    if result.get("voice_commands_triggered", 0) == 0:
                        test.result = TestResult.FAIL
                        test.message = f"No command triggered for: '{phrase}'"
                        break
                else:
                    test.result = TestResult.PASS
                    test.message = f"All {len(test_phrases)} phrases triggered correctly"

            elif HAS_LOCAL:
                # Use local registry
                registry = get_command_registry()
                passed = 0
                for phrase, expected_cmd in test_phrases:
                    match = registry.match(phrase)
                    if match and match.command_id == expected_cmd:
                        passed += 1
                    else:
                        test.result = TestResult.FAIL
                        test.message = f"Phrase '{phrase}' did not match '{expected_cmd}'"
                        break
                else:
                    test.result = TestResult.PASS
                    test.message = f"All {passed} phrases matched correctly"
            else:
                test.result = TestResult.SKIP
                test.message = "Neither HTTP nor local modules available"

        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)

        test.duration_ms = (time.time() - start) * 1000
        return test

    def test_voice_command_match_negative(self) -> TestCase:
        """Test that non-matching phrases don't trigger commands"""
        test = TestCase(name="Voice Command - Negative Match", description="Non-matching phrases should not trigger")
        start = time.time()

        non_matching_phrases = [
            "Hey turn off the lights",  # No "honey"
            "Turn off the lights please",  # No "honey"
            "Alexa, turn off the lights",  # No "honey"
            "Honey, I'm going to the store",  # No "turn off lights"
            "Just a regular conversation",
            "The lights are too bright",
        ]

        try:
            if self.use_http and HAS_HTTPX:
                response = httpx.post(
                    f"{self.service_url}/process",
                    json={
                        "segments": [{"text": p, "speaker": "pruitt"} for p in non_matching_phrases],
                        "job_id": "test-voice-negative",
                    },
                    timeout=10.0,
                )
                result = response.json()
                triggered = result.get("voice_commands_triggered", 0)
                if triggered == 0:
                    test.result = TestResult.PASS
                    test.message = f"Correctly rejected {len(non_matching_phrases)} non-matching phrases"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Incorrectly triggered {triggered} command(s)"

            elif HAS_LOCAL:
                registry = get_command_registry()
                false_positives = []
                for phrase in non_matching_phrases:
                    match = registry.match(phrase)
                    if match:
                        false_positives.append(phrase)

                if not false_positives:
                    test.result = TestResult.PASS
                    test.message = f"Correctly rejected {len(non_matching_phrases)} phrases"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"False positives: {false_positives}"
            else:
                test.result = TestResult.SKIP
                test.message = "Neither HTTP nor local modules available"

        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)

        test.duration_ms = (time.time() - start) * 1000
        return test

    # =========================================================================
    # Emotion Alert Tests
    # =========================================================================

    def test_emotion_alert_threshold(self) -> TestCase:
        """Test that alerts fire at exactly 20 consecutive anger emotions"""
        test = TestCase(
            name="Emotion Alert - 20 Consecutive Threshold",
            description="Alert should fire at exactly 20 consecutive angry emotions",
        )
        start = time.time()

        try:
            if self.use_http and HAS_HTTPX:
                # Reset tracking first
                httpx.post(f"{self.service_url}/alerts/reset", timeout=5.0)

                # Send 19 anger segments - should NOT trigger
                segments_19 = [
                    {"text": f"I am so angry {i}", "speaker": "pruitt", "emotion": "anger"} for i in range(19)
                ]
                response = httpx.post(
                    f"{self.service_url}/process",
                    json={"segments": segments_19, "job_id": "test-emotion-19"},
                    timeout=10.0,
                )
                result_19 = response.json()

                # Send 1 more - should trigger
                response = httpx.post(
                    f"{self.service_url}/process",
                    json={
                        "segments": [{"text": "Still angry", "speaker": "pruitt", "emotion": "anger"}],
                        "job_id": "test-emotion-20",
                    },
                    timeout=10.0,
                )
                result_20 = response.json()

                if (
                    result_19.get("emotion_alerts_triggered", 0) == 0
                    and result_20.get("emotion_alerts_triggered", 0) == 1
                ):
                    test.result = TestResult.PASS
                    test.message = "Alert correctly fired at exactly 20 consecutive"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"19 segments: {result_19.get('emotion_alerts_triggered')}, 20th: {result_20.get('emotion_alerts_triggered')}"

            elif HAS_LOCAL:
                tracker = EmotionAlertTracker(threshold=20, target_emotion="anger", tracked_speakers=["pruitt"])

                # Send 19 - no alert
                for i in range(19):
                    alert = tracker.add_emotion("pruitt", "anger")
                    if alert:
                        test.result = TestResult.FAIL
                        test.message = f"Alert fired too early at emotion #{i + 1}"
                        break
                else:
                    # 20th should trigger
                    alert = tracker.add_emotion("pruitt", "anger")
                    if alert and alert.consecutive_count == 20:
                        test.result = TestResult.PASS
                        test.message = "Alert correctly fired at exactly 20 consecutive"
                    else:
                        test.result = TestResult.FAIL
                        test.message = "Alert did not fire at emotion #20"
            else:
                test.result = TestResult.SKIP
                test.message = "Neither HTTP nor local modules available"

        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)

        test.duration_ms = (time.time() - start) * 1000
        return test

    def test_emotion_alert_reset(self) -> TestCase:
        """Test that non-anger emotion resets consecutive count"""
        test = TestCase(
            name="Emotion Alert - Counter Reset", description="Non-anger emotion should reset consecutive count"
        )
        start = time.time()

        try:
            if self.use_http and HAS_HTTPX:
                # Reset first
                httpx.post(f"{self.service_url}/alerts/reset", timeout=5.0)

                # Send 15 anger segments
                segments = [{"text": f"Angry {i}", "speaker": "ericah", "emotion": "anger"} for i in range(15)]
                httpx.post(
                    f"{self.service_url}/process", json={"segments": segments, "job_id": "test-reset-15"}, timeout=10.0
                )

                # Check status
                status_before = httpx.get(f"{self.service_url}/alerts/status", timeout=5.0).json()
                count_before = (
                    status_before.get("speaker_states", {}).get("ericah", {}).get("current_consecutive_count", 0)
                )

                # Send 1 joy - should reset
                httpx.post(
                    f"{self.service_url}/process",
                    json={
                        "segments": [{"text": "Happy now", "speaker": "ericah", "emotion": "joy"}],
                        "job_id": "test-reset-joy",
                    },
                    timeout=10.0,
                )

                # Check status after
                status_after = httpx.get(f"{self.service_url}/alerts/status", timeout=5.0).json()
                count_after = (
                    status_after.get("speaker_states", {}).get("ericah", {}).get("current_consecutive_count", 0)
                )

                if count_before == 15 and count_after == 0:
                    test.result = TestResult.PASS
                    test.message = f"Counter correctly reset from {count_before} to {count_after}"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Expected 15→0, got {count_before}→{count_after}"

            elif HAS_LOCAL:
                tracker = EmotionAlertTracker(threshold=20, target_emotion="anger", tracked_speakers=["ericah"])

                # Add 15 anger
                for i in range(15):
                    tracker.add_emotion("ericah", "anger")

                status = tracker.get_speaker_status("ericah")
                count_before = status["current_consecutive_count"]

                # Add joy - should reset
                tracker.add_emotion("ericah", "joy")

                status = tracker.get_speaker_status("ericah")
                count_after = status["current_consecutive_count"]

                if count_before == 15 and count_after == 0:
                    test.result = TestResult.PASS
                    test.message = f"Counter correctly reset from {count_before} to {count_after}"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Expected 15→0, got {count_before}→{count_after}"
            else:
                test.result = TestResult.SKIP
                test.message = "Neither HTTP nor local modules available"

        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)

        test.duration_ms = (time.time() - start) * 1000
        return test

    def test_emotion_untracked_speaker(self) -> TestCase:
        """Test that untracked speakers don't trigger alerts"""
        test = TestCase(
            name="Emotion Alert - Untracked Speaker", description="Only pruitt and ericah should be tracked"
        )
        start = time.time()

        try:
            if self.use_http and HAS_HTTPX:
                httpx.post(f"{self.service_url}/alerts/reset", timeout=5.0)

                # Send 25 anger segments from an untracked speaker
                segments = [{"text": f"Angry {i}", "speaker": "john_doe", "emotion": "anger"} for i in range(25)]
                response = httpx.post(
                    f"{self.service_url}/process", json={"segments": segments, "job_id": "test-untracked"}, timeout=10.0
                )
                result = response.json()

                if result.get("emotion_alerts_triggered", 0) == 0:
                    test.result = TestResult.PASS
                    test.message = "Correctly ignored untracked speaker 'john_doe'"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Incorrectly triggered alert for untracked speaker"

            elif HAS_LOCAL:
                tracker = EmotionAlertTracker(
                    threshold=20, target_emotion="anger", tracked_speakers=["pruitt", "ericah"]
                )

                alerts = []
                for i in range(25):
                    alert = tracker.add_emotion("john_doe", "anger")
                    if alert:
                        alerts.append(alert)

                if not alerts:
                    test.result = TestResult.PASS
                    test.message = "Correctly ignored untracked speaker"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Incorrectly fired {len(alerts)} alerts for untracked speaker"
            else:
                test.result = TestResult.SKIP
                test.message = "Neither HTTP nor local modules available"

        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)

        test.duration_ms = (time.time() - start) * 1000
        return test

    # =========================================================================
    # Command Registry API Tests
    # =========================================================================

    def test_command_list(self) -> TestCase:
        """Test listing registered commands"""
        test = TestCase(name="Command API - List Commands", description="Should list all registered voice commands")
        start = time.time()

        try:
            if self.use_http and HAS_HTTPX:
                response = httpx.get(f"{self.service_url}/commands", timeout=5.0)
                commands = response.json()

                if isinstance(commands, list) and len(commands) >= 1:
                    has_lights = any(c["command_id"] == "lights_off" for c in commands)
                    if has_lights:
                        test.result = TestResult.PASS
                        test.message = f"Found {len(commands)} command(s) including 'lights_off'"
                    else:
                        test.result = TestResult.FAIL
                        test.message = "Default 'lights_off' command not found"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"Unexpected response: {commands}"

            elif HAS_LOCAL:
                registry = get_command_registry()
                commands = registry.list_all()

                has_lights = any(c.command_id == "lights_off" for c in commands)
                if has_lights:
                    test.result = TestResult.PASS
                    test.message = f"Found {len(commands)} command(s) including 'lights_off'"
                else:
                    test.result = TestResult.FAIL
                    test.message = "Default 'lights_off' command not found"
            else:
                test.result = TestResult.SKIP
                test.message = "Neither HTTP nor local modules available"

        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)

        test.duration_ms = (time.time() - start) * 1000
        return test

    def test_command_register_and_delete(self) -> TestCase:
        """Test registering and deleting a custom command"""
        test = TestCase(
            name="Command API - Register & Delete", description="Should be able to register and delete custom commands"
        )
        start = time.time()

        test_cmd = {
            "command_id": "test_custom_cmd",
            "pattern": r"test.*custom.*pattern",
            "description": "Test custom command",
            "n8n_action": "test_action",
            "enabled": True,
        }

        try:
            if self.use_http and HAS_HTTPX:
                # Register
                reg_response = httpx.post(f"{self.service_url}/commands", json=test_cmd, timeout=5.0)

                if reg_response.status_code not in (200, 201):
                    test.result = TestResult.FAIL
                    test.message = f"Registration failed: {reg_response.status_code}"
                else:
                    # Verify it's there
                    list_response = httpx.get(f"{self.service_url}/commands", timeout=5.0)
                    commands = list_response.json()
                    found = any(c["command_id"] == "test_custom_cmd" for c in commands)

                    if not found:
                        test.result = TestResult.FAIL
                        test.message = "Command not found after registration"
                    else:
                        # Delete it
                        del_response = httpx.delete(f"{self.service_url}/commands/test_custom_cmd", timeout=5.0)

                        # Verify it's gone
                        list_response = httpx.get(f"{self.service_url}/commands", timeout=5.0)
                        commands = list_response.json()
                        still_there = any(c["command_id"] == "test_custom_cmd" for c in commands)

                        if not still_there:
                            test.result = TestResult.PASS
                            test.message = "Successfully registered and deleted custom command"
                        else:
                            test.result = TestResult.FAIL
                            test.message = "Command still present after deletion"

            elif HAS_LOCAL:
                registry = CommandRegistry()  # Fresh instance for test

                # Register
                cmd = registry.register(**test_cmd)

                # Verify
                found = registry.get("test_custom_cmd")
                if not found:
                    test.result = TestResult.FAIL
                    test.message = "Command not found after registration"
                else:
                    # Delete
                    registry.unregister("test_custom_cmd")

                    # Verify gone
                    still_there = registry.get("test_custom_cmd")
                    if not still_there:
                        test.result = TestResult.PASS
                        test.message = "Successfully registered and deleted custom command"
                    else:
                        test.result = TestResult.FAIL
                        test.message = "Command still present after deletion"
            else:
                test.result = TestResult.SKIP
                test.message = "Neither HTTP nor local modules available"

        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)

        test.duration_ms = (time.time() - start) * 1000
        return test

    # =========================================================================
    # Health & Integration Tests
    # =========================================================================

    def test_health_endpoint(self) -> TestCase:
        """Test service health endpoint"""
        test = TestCase(name="Health Check - Basic", description="Service should respond to health check")
        start = time.time()

        try:
            if self.use_http and HAS_HTTPX:
                response = httpx.get(f"{self.service_url}/health", timeout=5.0)

                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        test.result = TestResult.PASS
                        test.message = f"Service healthy, commands={data.get('commands_loaded')}"
                    else:
                        test.result = TestResult.FAIL
                        test.message = f"Unhealthy status: {data}"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"HTTP {response.status_code}"
            else:
                test.result = TestResult.SKIP
                test.message = "HTTP client required for health check"

        except httpx.ConnectError:
            test.result = TestResult.FAIL
            test.message = f"Cannot connect to {self.service_url}"
        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)

        test.duration_ms = (time.time() - start) * 1000
        return test

    def test_end_to_end_transcription_flow(self) -> TestCase:
        """Test complete transcription → n8n flow with voice command"""
        test = TestCase(
            name="End-to-End - Voice Command Flow", description="Simulate full transcription segment processing"
        )
        start = time.time()

        try:
            if self.use_http and HAS_HTTPX:
                # Simulate what transcription service sends
                transcription_payload = {
                    "job_id": "e2e-test-001",
                    "session_id": "cli-test",
                    "segments": [
                        {
                            "text": "Hello how are you",
                            "speaker": "speaker_0",
                            "verified_speaker": "pruitt",
                            "start_time": 0.0,
                            "end_time": 1.5,
                            "emotion": "neutral",
                            "emotion_confidence": 0.9,
                        },
                        {
                            "text": "Honey, can you turn off the lights please",
                            "speaker": "speaker_0",
                            "verified_speaker": "pruitt",
                            "start_time": 1.5,
                            "end_time": 4.2,
                            "emotion": "neutral",
                            "emotion_confidence": 0.85,
                        },
                        {
                            "text": "Thank you",
                            "speaker": "speaker_1",
                            "verified_speaker": "ericah",
                            "start_time": 4.5,
                            "end_time": 5.0,
                            "emotion": "joy",
                            "emotion_confidence": 0.92,
                        },
                    ],
                }

                response = httpx.post(f"{self.service_url}/process", json=transcription_payload, timeout=10.0)

                if response.status_code == 200:
                    result = response.json()
                    if result.get("voice_commands_triggered", 0) == 1:
                        test.result = TestResult.PASS
                        test.message = (
                            f"Processed {result['processed_segments']} segments, triggered lights_off command"
                        )
                    else:
                        test.result = TestResult.FAIL
                        test.message = f"Expected 1 voice command, got {result}"
                else:
                    test.result = TestResult.FAIL
                    test.message = f"HTTP {response.status_code}: {response.text}"
            else:
                test.result = TestResult.SKIP
                test.message = "HTTP client required for E2E test"

        except Exception as e:
            test.result = TestResult.ERROR
            test.message = str(e)

        test.duration_ms = (time.time() - start) * 1000
        return test

    # =========================================================================
    # Test Runner
    # =========================================================================

    def run_all(self) -> tuple[int, int, int]:
        """Run all tests and return (passed, failed, skipped)"""
        print("\n" + "=" * 60)
        print("n8n Integration Service - CLI Test Suite")
        print("=" * 60)
        print(f"Service URL: {self.service_url}")
        print(f"Local modules: {'available' if HAS_LOCAL else 'not available'}")
        print(f"HTTP client: {'available' if HAS_HTTPX else 'not available'}")
        print("=" * 60 + "\n")

        # Health check first
        print("📡 Service Health\n" + "-" * 40)
        self.add_result(self.test_health_endpoint())

        # Voice command tests
        print("\n🎤 Voice Command Tests\n" + "-" * 40)
        self.add_result(self.test_voice_command_match_positive())
        self.add_result(self.test_voice_command_match_negative())

        # Emotion alert tests
        print("\n😠 Emotion Alert Tests\n" + "-" * 40)
        self.add_result(self.test_emotion_alert_threshold())
        self.add_result(self.test_emotion_alert_reset())
        self.add_result(self.test_emotion_untracked_speaker())

        # Command API tests
        print("\n📝 Command Registry API Tests\n" + "-" * 40)
        self.add_result(self.test_command_list())
        self.add_result(self.test_command_register_and_delete())

        # End-to-end tests
        print("\n🔄 End-to-End Tests\n" + "-" * 40)
        self.add_result(self.test_end_to_end_transcription_flow())

        # Summary
        passed = sum(1 for t in self.tests if t.result == TestResult.PASS)
        failed = sum(1 for t in self.tests if t.result == TestResult.FAIL)
        skipped = sum(1 for t in self.tests if t.result == TestResult.SKIP)
        errors = sum(1 for t in self.tests if t.result == TestResult.ERROR)

        print("\n" + "=" * 60)
        print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped, {errors} errors")
        print(f"Total time: {sum(t.duration_ms for t in self.tests):.1f}ms")
        print("=" * 60 + "\n")

        return passed, failed, skipped

    def run_specific(self, test_name: str) -> bool:
        """Run a specific test by name"""
        test_map = {
            "voice_command": [self.test_voice_command_match_positive, self.test_voice_command_match_negative],
            "emotion_alert": [
                self.test_emotion_alert_threshold,
                self.test_emotion_alert_reset,
                self.test_emotion_untracked_speaker,
            ],
            "command_api": [self.test_command_list, self.test_command_register_and_delete],
            "health": [self.test_health_endpoint],
            "e2e": [self.test_end_to_end_transcription_flow],
        }

        if test_name not in test_map:
            print(f"Unknown test: {test_name}")
            print(f"Available: {', '.join(test_map.keys())}")
            return False

        print(f"\nRunning tests: {test_name}\n" + "-" * 40)
        for test_fn in test_map[test_name]:
            self.add_result(test_fn())

        failed = sum(1 for t in self.tests if t.result == TestResult.FAIL)
        return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="CLI Test Suite for n8n Integration Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_n8n_cli.py                    # Run all tests locally
  python test_n8n_cli.py --url http://localhost:8011  # Test against running service
  python test_n8n_cli.py --test voice_command  # Run only voice command tests
  python test_n8n_cli.py --test emotion_alert  # Run only emotion alert tests
  python test_n8n_cli.py -v                    # Verbose output
        """,
    )
    parser.add_argument(
        "--url", "-u", default="http://localhost:8011", help="n8n service URL (default: http://localhost:8011)"
    )
    parser.add_argument(
        "--test",
        "-t",
        choices=["voice_command", "emotion_alert", "command_api", "health", "e2e"],
        help="Run specific test category",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--local", "-l", action="store_true", help="Force local testing (no HTTP)")

    args = parser.parse_args()

    url = "local://" if args.local else args.url
    runner = TestRunner(service_url=url, verbose=args.verbose)

    if args.test:
        success = runner.run_specific(args.test)
        sys.exit(0 if success else 1)
    else:
        passed, failed, skipped = runner.run_all()
        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
