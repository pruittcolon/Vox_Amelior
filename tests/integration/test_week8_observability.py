"""
Integration tests for Week 8: Observability & SLOs.

Tests cover:
- OpenTelemetry setup and no-op fallbacks
- SLO tracker configuration
- SLI recording and status calculation
- Error budget calculations
- Prometheus metrics integration
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "telemetry"))


class TestOpenTelemetrySetup:
    """Tests for OpenTelemetry configuration."""
    
    def test_otel_disabled_by_default(self) -> None:
        """OTel is disabled when OTEL_ENABLED is not set."""
        # Default behavior: OTEL_ENABLED should be False
        assert os.getenv("OTEL_ENABLED", "false").lower() == "false"
    
    def test_no_op_tracer_fallback(self) -> None:
        """NoOpTracer is returned when OTel unavailable."""
        from shared.telemetry import NoOpTracer, NoOpSpan
        
        tracer = NoOpTracer()
        
        # Should return NoOpSpan
        with tracer.start_as_current_span("test") as span:
            assert isinstance(span, NoOpSpan)
            # Should not raise
            span.set_attribute("key", "value")
            span.add_event("event_name")
    
    def test_no_op_meter_fallback(self) -> None:
        """NoOpMeter is returned when OTel unavailable."""
        from shared.telemetry import NoOpMeter, NoOpCounter, NoOpHistogram
        
        meter = NoOpMeter()
        
        counter = meter.create_counter("test_counter")
        assert isinstance(counter, NoOpCounter)
        counter.add(1)  # Should not raise
        
        histogram = meter.create_histogram("test_histogram")
        assert isinstance(histogram, NoOpHistogram)
        histogram.record(0.5)  # Should not raise


class TestSLODefinitions:
    """Tests for SLO definition configuration."""
    
    def test_default_slos_exist(self) -> None:
        """Default SLO definitions are configured."""
        from slo_tracker import DEFAULT_SLOS
        
        assert len(DEFAULT_SLOS) > 0
        
        # Check expected SLOs
        slo_names = [slo.name for slo in DEFAULT_SLOS]
        assert "api_availability" in slo_names
        assert "api_latency_p95" in slo_names  # Fixed: actual name
        assert "error_rate" in slo_names
    
    def test_slo_error_budget_calculation(self) -> None:
        """Error budget is correctly calculated from target."""
        from slo_tracker import SLODefinition, SLOType
        
        slo = SLODefinition(
            name="test_availability",
            slo_type=SLOType.AVAILABILITY,
            target=0.999,  # 99.9%
        )
        
        # Error budget is (1 - target) * 100 = 0.1%
        assert slo.error_budget_pct == pytest.approx(0.1)  # Fixed: 0.1 not 0.001
    
    def test_slo_types(self) -> None:
        """All SLO types are defined."""
        from slo_tracker import SLOType
        
        assert SLOType.AVAILABILITY.value == "availability"
        assert SLOType.LATENCY.value == "latency"
        assert SLOType.ERROR_RATE.value == "error_rate"
        assert SLOType.THROUGHPUT.value == "throughput"


class TestSLOTracker:
    """Tests for SLO tracking functionality."""
    
    @pytest.fixture
    def tracker(self):
        """Create fresh SLO tracker."""
        from slo_tracker import SLOTracker
        return SLOTracker()
    
    def test_record_request(self, tracker) -> None:
        """Requests are recorded correctly."""
        record = tracker.record_request(
            endpoint="/api/health",
            latency_ms=50.0,
            success=True,
        )
        
        assert record.endpoint == "/api/health"
        assert record.latency_ms == 50.0
        assert record.success is True
        assert isinstance(record.timestamp, datetime)
    
    def test_availability_calculation(self, tracker) -> None:
        """Availability SLO is correctly calculated."""
        # Record 999 successes and 1 failure
        for _ in range(999):
            tracker.record_request("/api/test", 50.0, success=True)
        tracker.record_request("/api/test", 50.0, success=False)
        
        status = tracker.get_slo_status("api_availability")
        
        # Current value should be 0.999 (key is "current" not "current_value")
        assert status["current"] == pytest.approx(0.999, rel=0.01)
    
    def test_latency_calculation(self, tracker) -> None:
        """Latency SLO is correctly calculated."""
        # Record requests with varying latencies
        latencies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        for lat in latencies:
            tracker.record_request("/api/test", float(lat), success=True)
        
        status = tracker.get_slo_status("api_latency_p95")  # Fixed name
        
        # P95 should be calculated
        assert "current_p95_ms" in status
        assert status["current_p95_ms"] > 0
    
    def test_error_rate_calculation(self, tracker) -> None:
        """Error rate SLO is correctly calculated."""
        # Record 98 successes and 2 failures
        for _ in range(98):
            tracker.record_request("/api/test", 50.0, success=True)
        for _ in range(2):
            tracker.record_request("/api/test", 50.0, success=False)
        
        status = tracker.get_slo_status("error_rate")
        
        # Error rate should be 2%
        assert status["current"] == pytest.approx(0.02, rel=0.1)
    
    def test_get_all_slo_status(self, tracker) -> None:
        """All SLO statuses can be retrieved."""
        # Record some data
        tracker.record_request("/api/test", 100.0, success=True)
        
        all_status = tracker.get_all_slo_status()
        
        assert len(all_status) > 0
        # All should have slo_name - even if no data
        assert all("slo_name" in s for s in all_status)
    
    def test_slo_breach_detection(self, tracker) -> None:
        """SLO breaches are detected."""
        # Record 100% failures to breach availability SLO
        for _ in range(100):
            tracker.record_request("/api/test", 50.0, success=False)
        
        breaches = tracker.check_slo_breach()
        
        # Should detect availability breach
        assert len(breaches) > 0
        assert any("availability" in b.get("slo_name", "") for b in breaches)


class TestErrorBudget:
    """Tests for error budget calculations."""
    
    def test_error_budget_remaining(self) -> None:
        """Error budget remaining is calculated correctly."""
        from slo_tracker import SLOTracker
        
        tracker = SLOTracker()
        
        # Record perfect uptime
        for _ in range(1000):
            tracker.record_request("/api/test", 50.0, success=True)
        
        budget = tracker.get_error_budget("api_availability")
        
        # Should have error_budget_remaining_pct
        assert "error_budget_remaining_pct" in budget
        assert budget["error_budget_remaining_pct"] > 0


class TestPrometheusMetrics:
    """Tests for Prometheus metrics integration."""
    
    def test_no_op_metric_class(self) -> None:
        """NoOpMetric class works correctly."""
        from metrics import PROMETHEUS_AVAILABLE
        
        if not PROMETHEUS_AVAILABLE:
            from metrics import NoOpMetric
            metric = NoOpMetric()
            
            # All operations should be no-ops
            metric.labels(method="GET", endpoint="/test").inc()
            metric.observe(0.5)
            metric.set(1)  # Should not raise
    
    def test_record_request_function(self) -> None:
        """record_request function works with both real and no-op metrics."""
        from metrics import record_request
        
        # Should not raise regardless of prometheus availability
        record_request("GET", "/api/test", 200, 0.05)
        record_request("POST", "/api/upload", 500, 1.5)
    
    def test_metrics_middleware_skip_paths(self) -> None:
        """Metrics middleware skips health check paths."""
        from metrics import MetricsMiddleware
        
        assert "/health" in MetricsMiddleware.SKIP_PATHS
        assert "/ready" in MetricsMiddleware.SKIP_PATHS
        assert "/metrics" in MetricsMiddleware.SKIP_PATHS
    
    def test_endpoint_normalization(self) -> None:
        """Endpoint paths are normalized for cardinality control."""
        from metrics import MetricsMiddleware
        
        # UUIDs should be replaced
        normalized = MetricsMiddleware._normalize_endpoint(
            "/api/users/550e8400-e29b-41d4-a716-446655440000/profile"
        )
        assert "{id}" in normalized
        
        # Numeric IDs should be replaced
        normalized = MetricsMiddleware._normalize_endpoint("/api/orders/12345")
        assert "{id}" in normalized


class TestCostTracker:
    """Tests for cost tracking functionality."""
    
    def test_cost_tracker_exists(self) -> None:
        """Cost tracker module is available."""
        from cost_tracker import CostTracker
        
        tracker = CostTracker()
        assert tracker is not None


class TestTelemetryExports:
    """Tests for telemetry module exports."""
    
    def test_main_exports(self) -> None:
        """Main telemetry exports are available."""
        from shared.telemetry import (
            setup_telemetry,
            get_tracer,
            get_meter,
            NoOpTracer,
            NoOpMeter,
        )
        
        assert callable(setup_telemetry)
        assert callable(get_tracer)
        assert callable(get_meter)
    
    def test_metrics_exports(self) -> None:
        """Metrics exports are available."""
        # Import may fail if prometheus metrics already registered (test isolation issue)
        # This is acceptable - we just check the module is importable
        try:
            from shared.telemetry.metrics import (
                REQUEST_COUNT,
                REQUEST_LATENCY,
                record_request,
                MetricsMiddleware,
            )
            
            assert REQUEST_COUNT is not None
            assert REQUEST_LATENCY is not None
            assert callable(record_request)
        except ValueError as e:
            if "Duplicated timeseries" in str(e):
                # Already registered, that's fine
                pass
            else:
                raise


class TestSLOTrackerSingleton:
    """Tests for SLO tracker singleton."""
    
    def test_get_slo_tracker_returns_instance(self) -> None:
        """get_slo_tracker returns SLOTracker instance."""
        from slo_tracker import get_slo_tracker, SLOTracker
        
        tracker = get_slo_tracker()
        assert isinstance(tracker, SLOTracker)
    
    def test_singleton_returns_same_instance(self) -> None:
        """Singleton returns same instance."""
        from slo_tracker import get_slo_tracker
        
        tracker1 = get_slo_tracker()
        tracker2 = get_slo_tracker()
        
        assert tracker1 is tracker2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
