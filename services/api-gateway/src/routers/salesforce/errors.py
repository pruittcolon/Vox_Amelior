"""
Salesforce Custom Exceptions

Structured error handling for Salesforce API operations.
"""


class SalesforceError(Exception):
    """Base Salesforce error with status code support."""

    def __init__(self, message: str, status_code: int = 500, error_code: str = None, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict:
        """Convert error to dictionary for API responses."""
        return {
            "error": self.message,
            "error_code": self.error_code,
            "status_code": self.status_code,
            "details": self.details,
        }


class SalesforceAuthError(SalesforceError):
    """Authentication or authorization failed."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message=message, status_code=401, error_code="AUTH_FAILED", details=details)


class SalesforceRateLimitError(SalesforceError):
    """API rate limit exceeded."""

    def __init__(self, retry_after: int = 60, details: dict = None):
        super().__init__(
            message=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            status_code=429,
            error_code="RATE_LIMITED",
            details={"retry_after": retry_after, **(details or {})},
        )
        self.retry_after = retry_after


class SalesforceNotFoundError(SalesforceError):
    """Resource not found in Salesforce."""

    def __init__(self, object_type: str, record_id: str):
        super().__init__(
            message=f"{object_type} with ID '{record_id}' not found",
            status_code=404,
            error_code="NOT_FOUND",
            details={"object_type": object_type, "record_id": record_id},
        )


class SalesforceValidationError(SalesforceError):
    """Data validation failed."""

    def __init__(self, message: str, field_errors: dict = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details={"field_errors": field_errors or {}},
        )


class SalesforceConnectionError(SalesforceError):
    """Connection to Salesforce failed."""

    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(
            message=message,
            status_code=503,
            error_code="CONNECTION_FAILED",
            details={"original_error": str(original_error) if original_error else None},
        )
