import json
import logging
from datetime import UTC, datetime


class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings for structured logging systems (ELK, Datadog, etc.)
    """

    def format(self, record: logging.LogRecord) -> str:
        # Create a dictionary for the log record
        log_obj = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread,
        }

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            # formatted exception might be multiline; detailed parsers might prefer structured exception object
            # but string is safer for generic JSON parsers

        # Add extra fields if passed via 'extra'
        if hasattr(record, "service"):
            log_obj["service"] = record.service
        if hasattr(record, "user_id"):
            log_obj["user_id"] = record.user_id
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id

        return json.dumps(log_obj)


def setup_structured_logging(service_name: str, level: str = "INFO"):
    """
    Configure the root logger to use JSON formatting
    """
    # Create handler
    handler = logging.StreamHandler()
    formatter = JSONFormatter()
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates (e.g. from uvicorn default config)
    # But keep them if we want dual logging? Usually replace.
    if root_logger.handlers:
        root_logger.handlers = []

    root_logger.addHandler(handler)

    # Set service name for all records (hacky via factory or adapter,
    # simpler to just trust caller to bind it or rely on env vars)
    # A cleaner way is using a LoggerAdapter, but this global setup is a good start.
