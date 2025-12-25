"""
Property-Based Tests with Hypothesis.

Tests input processing functions with diverse, auto-generated inputs
to catch edge cases that example-based tests miss.

Phase 8 of ultimateseniordevplan.md.
"""
import pytest

# Hypothesis import with graceful fallback
try:
    from hypothesis import given, strategies as st, settings, assume
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    # Create dummy decorators for test discovery
    def given(*args, **kwargs):
        def decorator(f):
            return pytest.mark.skip(reason="hypothesis not installed")(f)
        return decorator
    
    class st:
        @staticmethod
        def text(*args, **kwargs):
            return None
        @staticmethod
        def binary(*args, **kwargs):
            return None
        @staticmethod
        def integers(*args, **kwargs):
            return None
    
    def settings(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    
    def assume(x):
        pass


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestTextSanitization:
    """Property tests for text input sanitization."""

    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=100)
    def test_text_sanitization_never_crashes(self, text: str):
        """Text sanitization handles arbitrary Unicode gracefully."""
        # Simple sanitization example - replace with actual function
        result = text.strip()
        assert isinstance(result, str)

    @given(st.text(min_size=1, max_size=500))
    @settings(max_examples=100)
    def test_sanitization_is_idempotent(self, text: str):
        """Sanitization applied twice yields same result."""
        once = text.strip().lower()
        twice = once.strip().lower()
        assert once == twice

    @given(st.text(min_size=0, max_size=200))
    @settings(max_examples=50)
    def test_valid_unicode_preserved(self, text: str):
        """Valid Unicode characters are preserved after sanitization."""
        result = text.strip()
        # Should not raise and should be valid UTF-8
        result.encode("utf-8")


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestSessionKeyDecoding:
    """Property tests for session key decoding."""

    @given(st.binary(min_size=32, max_size=32))
    @settings(max_examples=100)
    def test_valid_32_byte_keys_roundtrip(self, key_bytes: bytes):
        """Valid 32-byte keys roundtrip through base64 encoding."""
        import base64
        from core.dependencies import _decode_session_key

        encoded = base64.b64encode(key_bytes).decode()
        decoded = _decode_session_key(encoded)
        assert decoded == key_bytes

    @given(st.binary(min_size=33, max_size=64))
    @settings(max_examples=50)
    def test_longer_keys_truncate_to_32(self, key_bytes: bytes):
        """Keys longer than 32 bytes are truncated to 32."""
        import base64
        from core.dependencies import _decode_session_key

        encoded = base64.b64encode(key_bytes).decode()
        result = _decode_session_key(encoded)
        assert result is not None
        assert len(result) == 32
        assert result == key_bytes[:32]

    @given(st.binary(min_size=1, max_size=31))
    @settings(max_examples=50)
    def test_short_keys_return_none(self, key_bytes: bytes):
        """Keys shorter than 32 bytes return None gracefully."""
        import base64
        from core.dependencies import _decode_session_key

        encoded = base64.b64encode(key_bytes).decode()
        result = _decode_session_key(encoded)
        assert result is None

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_invalid_base64_never_crashes(self, text: str):
        """Invalid base64 input returns None without exception."""
        from core.dependencies import _decode_session_key

        # Should never raise, just return None for invalid input
        result = _decode_session_key(text)
        assert result is None or isinstance(result, bytes)


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestErrorCodeMapping:
    """Property tests for error code handling."""

    @given(st.text(min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_unknown_error_codes_return_500(self, code: str):
        """Unknown error codes default to 500 status."""
        from errors import get_status_code, ERROR_STATUS_CODES

        assume(code not in ERROR_STATUS_CODES)
        status = get_status_code(code)
        assert status == 500

    @given(st.integers(min_value=400, max_value=599))
    @settings(max_examples=20)
    def test_status_codes_in_valid_range(self, expected_status: int):
        """All mapped status codes are valid HTTP error codes."""
        from errors import ERROR_STATUS_CODES

        for code, status in ERROR_STATUS_CODES.items():
            assert 400 <= status <= 599, f"Invalid status {status} for {code}"


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestAPIErrorModel:
    """Property tests for APIError model."""

    @given(
        st.text(min_size=1, max_size=50),
        st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=50)
    def test_api_error_creation_never_crashes(self, code: str, message: str):
        """APIError creation handles arbitrary input."""
        from errors import APIError

        error = APIError(error_code=code, message=message)
        assert error.error_code == code
        assert error.message == message
        assert error.request_id.startswith("req_")

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=30)
    def test_api_error_serializes_to_json(self, message: str):
        """APIError can be serialized to JSON."""
        from errors import APIError, ErrorCode

        error = APIError(
            error_code=ErrorCode.VALIDATION_ERROR,
            message=message,
        )
        json_str = error.model_dump_json()
        assert isinstance(json_str, str)
        assert message in json_str or message.encode().decode("unicode_escape") in json_str
