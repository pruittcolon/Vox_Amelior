"""
Shared Errors Module - Standardized API Error Responses.

This module provides consistent error response formats across all services,
following RFC 7807 Problem Details principles. All API errors should use
these structured response models.

Phase 11 of ultimateseniordevplan.md.
"""

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    """Standardized error codes for API responses.

    Error codes are prefixed by category:
    - VALIDATION_*: Input validation errors
    - AUTH_*: Authentication/authorization errors
    - RESOURCE_*: Resource-related errors
    - SERVICE_*: Service-level errors
    - GPU_*: GPU-related errors
    """

    # Validation errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    VALIDATION_MISSING_FIELD = "VALIDATION_MISSING_FIELD"
    VALIDATION_INVALID_FORMAT = "VALIDATION_INVALID_FORMAT"

    # Authentication errors
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    AUTHENTICATION_EXPIRED = "AUTHENTICATION_EXPIRED"
    AUTHORIZATION_FAILED = "AUTHORIZATION_FAILED"
    CSRF_TOKEN_INVALID = "CSRF_TOKEN_INVALID"

    # Resource errors
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"

    # Rate limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # Service errors
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    SERVICE_TIMEOUT = "SERVICE_TIMEOUT"
    SERVICE_UPSTREAM_ERROR = "SERVICE_UPSTREAM_ERROR"

    # GPU errors
    GPU_TIMEOUT = "GPU_TIMEOUT"
    GPU_UNAVAILABLE = "GPU_UNAVAILABLE"
    GPU_OUT_OF_MEMORY = "GPU_OUT_OF_MEMORY"

    # Internal errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"


class APIError(BaseModel):
    """Standardized API error response model.

    Follows RFC 7807 Problem Details principles for consistent
    error responses across all API endpoints.

    Attributes:
        error_code: Machine-readable error code from ErrorCode enum.
        message: Human-readable error description.
        details: Optional additional context (field errors, etc.).
        request_id: Unique identifier for request tracing.
        service: Name of service that generated the error.

    Example:
        >>> error = APIError(
        ...     error_code=ErrorCode.VALIDATION_ERROR,
        ...     message="Invalid email format",
        ...     details={"field": "email", "value": "not-an-email"}
        ... )
    """

    error_code: str = Field(
        ...,
        description="Machine-readable error code",
        examples=["VALIDATION_ERROR", "GPU_TIMEOUT"],
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
        examples=["GPU lock could not be acquired within timeout"],
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional error context",
        examples=[{"wait_time_ms": 15000, "queue_position": 3}],
    )
    request_id: str = Field(
        default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}",
        description="Unique request identifier for tracing",
    )
    service: str = Field(
        default="api-gateway",
        description="Service that generated the error",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "error_code": "GPU_TIMEOUT",
                "message": "GPU lock could not be acquired",
                "details": {"wait_time_ms": 15000},
                "request_id": "req_abc123def456",
                "service": "api-gateway",
            }
        }
    }


class APIException(Exception):
    """Exception wrapper for APIError responses.

    This class inherits from Exception so it can be used with FastAPI's
    exception_handler decorator. It wraps an APIError model for structured
    error responses.

    Attributes:
        error: The APIError model containing error details.
        status_code: HTTP status code for the response.

    Example:
        >>> raise APIException(
        ...     error_code=ErrorCode.GPU_TIMEOUT,
        ...     message="GPU lock timed out"
        ... )
    """

    def __init__(
        self,
        error_code: str,
        message: str,
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
        service: str = "api-gateway",
    ):
        self.error = APIError(
            error_code=error_code if isinstance(error_code, str) else error_code.value,
            message=message,
            details=details,
            service=service,
        )
        self.status_code = status_code or get_status_code(self.error.error_code)
        super().__init__(message)

    def to_response(self) -> dict[str, Any]:
        """Convert to dictionary for JSONResponse."""
        return self.error.model_dump()


class ValidationErrorDetail(BaseModel):
    """Detail for field-level validation errors.

    Attributes:
        field: Name of the field that failed validation.
        message: Description of what went wrong.
        value: The rejected value (if safe to include).
    """

    field: str
    message: str
    value: Any | None = None


def create_validation_error(
    message: str,
    field_errors: list[ValidationErrorDetail] = None,
    request_id: str = None,
) -> APIError:
    """Factory for validation error responses.

    Args:
        message: Overall error message.
        field_errors: List of field-level errors.
        request_id: Optional request ID (auto-generated if not provided).

    Returns:
        APIError: Structured validation error response.

    Example:
        >>> error = create_validation_error(
        ...     "Request validation failed",
        ...     [ValidationErrorDetail(field="email", message="Invalid format")]
        ... )
    """
    details = None
    if field_errors:
        details = {"fields": [e.model_dump() for e in field_errors]}

    return APIError(
        error_code=ErrorCode.VALIDATION_ERROR,
        message=message,
        details=details,
        request_id=request_id or f"req_{uuid.uuid4().hex[:12]}",
    )


def create_service_error(
    error_code: ErrorCode,
    message: str,
    details: dict[str, Any] = None,
    request_id: str = None,
    service: str = "api-gateway",
) -> APIError:
    """Factory for service-level error responses.

    Args:
        error_code: Error code from ErrorCode enum.
        message: Human-readable error message.
        details: Additional context.
        request_id: Optional request ID.
        service: Service name.

    Returns:
        APIError: Structured error response.
    """
    return APIError(
        error_code=error_code.value if isinstance(error_code, ErrorCode) else error_code,
        message=message,
        details=details,
        request_id=request_id or f"req_{uuid.uuid4().hex[:12]}",
        service=service,
    )


# HTTP status code mappings
ERROR_STATUS_CODES: dict[str, int] = {
    ErrorCode.VALIDATION_ERROR.value: 400,
    ErrorCode.VALIDATION_MISSING_FIELD.value: 400,
    ErrorCode.VALIDATION_INVALID_FORMAT.value: 400,
    ErrorCode.AUTHENTICATION_FAILED.value: 401,
    ErrorCode.AUTHENTICATION_EXPIRED.value: 401,
    ErrorCode.AUTHORIZATION_FAILED.value: 403,
    ErrorCode.CSRF_TOKEN_INVALID.value: 403,
    ErrorCode.RESOURCE_NOT_FOUND.value: 404,
    ErrorCode.RESOURCE_ALREADY_EXISTS.value: 409,
    ErrorCode.RESOURCE_CONFLICT.value: 409,
    ErrorCode.RATE_LIMIT_EXCEEDED.value: 429,
    ErrorCode.SERVICE_UNAVAILABLE.value: 503,
    ErrorCode.SERVICE_TIMEOUT.value: 504,
    ErrorCode.GPU_TIMEOUT.value: 504,
    ErrorCode.GPU_UNAVAILABLE.value: 503,
    ErrorCode.INTERNAL_ERROR.value: 500,
}


def get_status_code(error_code: str) -> int:
    """Get HTTP status code for error code.

    Args:
        error_code: Error code string.

    Returns:
        int: Appropriate HTTP status code.
    """
    return ERROR_STATUS_CODES.get(error_code, 500)
