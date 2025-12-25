"""
OpenTelemetry Setup for Nemo Services
2025 Standard: OTel for traces/metrics, Prometheus 3.0 for storage

Usage:
    from shared.telemetry import setup_telemetry, get_tracer, get_meter

    # In service startup
    setup_telemetry("api-gateway")

    # Create spans
    tracer = get_tracer()
    with tracer.start_as_current_span("my_operation"):
        # do work
        pass
"""

import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# Configuration
OTEL_ENABLED = os.environ.get("OTEL_ENABLED", "false").lower() == "true"
OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "nemo-service")

# Lazy imports for optional dependencies
_tracer = None
_meter = None
_initialized = False


def setup_telemetry(service_name: str = None) -> bool:
    """
    Initialize OpenTelemetry for the service.
    Returns True if successful, False if OTel is disabled or unavailable.
    """
    global _tracer, _meter, _initialized

    if _initialized:
        return True

    if not OTEL_ENABLED:
        logger.info("OpenTelemetry disabled (set OTEL_ENABLED=true to enable)")
        _initialized = True
        return False

    try:
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import SERVICE_NAME as RESOURCE_SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        name = service_name or SERVICE_NAME

        # Create resource with service name
        resource = Resource.create({RESOURCE_SERVICE_NAME: name})

        # Setup tracing
        tracer_provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)
        _tracer = trace.get_tracer(name)

        # Setup metrics (Prometheus 3.0 native OTLP)
        metric_exporter = OTLPMetricExporter(endpoint=OTEL_ENDPOINT, insecure=True)
        metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=30000)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        _meter = metrics.get_meter(name)

        logger.info(f"OpenTelemetry initialized for {name} -> {OTEL_ENDPOINT}")
        _initialized = True
        return True

    except ImportError as e:
        logger.warning(f"OpenTelemetry dependencies not installed: {e}")
        _initialized = True
        return False
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}")
        _initialized = True
        return False


def get_tracer():
    """Get the configured tracer, or a no-op tracer if OTel is disabled."""
    global _tracer

    if _tracer is not None:
        return _tracer

    # Return no-op tracer
    try:
        from opentelemetry import trace

        return trace.get_tracer("nemo-noop")
    except ImportError:
        return NoOpTracer()


def get_meter():
    """Get the configured meter, or a no-op meter if OTel is disabled."""
    global _meter

    if _meter is not None:
        return _meter

    # Return no-op meter
    try:
        from opentelemetry import metrics

        return metrics.get_meter("nemo-noop")
    except ImportError:
        return NoOpMeter()


# ============================================================
# No-op implementations for when OTel is not available
# ============================================================


class NoOpSpan:
    """No-op span for when OTel is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key, value):
        pass

    def add_event(self, name, attributes=None):
        pass


class NoOpTracer:
    """No-op tracer for when OTel is disabled."""

    def start_as_current_span(self, name, **kwargs):
        return NoOpSpan()

    def start_span(self, name, **kwargs):
        return NoOpSpan()


class NoOpMeter:
    """No-op meter for when OTel is disabled."""

    def create_counter(self, name, **kwargs):
        return NoOpCounter()

    def create_histogram(self, name, **kwargs):
        return NoOpHistogram()

    def create_gauge(self, name, **kwargs):
        return NoOpGauge()


class NoOpCounter:
    def add(self, amount, attributes=None):
        pass


class NoOpHistogram:
    def record(self, value, attributes=None):
        pass


class NoOpGauge:
    def set(self, value, attributes=None):
        pass
