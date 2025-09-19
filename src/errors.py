"""
errors.py
Standardized error handling for the Memorizer framework.
Provides consistent error responses, logging, and middleware.
"""

import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """Standardized error codes."""

    # Authentication & Authorization
    AUTHENTICATION_FAILED = "AUTH_001"
    AUTHORIZATION_DENIED = "AUTH_002"
    INVALID_TOKEN = "AUTH_003"
    TOKEN_EXPIRED = "AUTH_004"
    INSUFFICIENT_PERMISSIONS = "AUTH_005"

    # Validation Errors
    VALIDATION_ERROR = "VAL_001"
    INVALID_INPUT = "VAL_002"
    MISSING_REQUIRED_FIELD = "VAL_003"
    INVALID_FORMAT = "VAL_004"
    VALUE_TOO_LONG = "VAL_005"
    VALUE_TOO_SHORT = "VAL_006"

    # Database Errors
    DATABASE_CONNECTION_FAILED = "DB_001"
    DATABASE_QUERY_FAILED = "DB_002"
    DATABASE_TRANSACTION_FAILED = "DB_003"
    RECORD_NOT_FOUND = "DB_004"
    DUPLICATE_RECORD = "DB_005"
    CONSTRAINT_VIOLATION = "DB_006"

    # Memory Management Errors
    MEMORY_NOT_FOUND = "MEM_001"
    MEMORY_CREATION_FAILED = "MEM_002"
    MEMORY_UPDATE_FAILED = "MEM_003"
    MEMORY_DELETION_FAILED = "MEM_004"
    MEMORY_TIER_INVALID = "MEM_005"

    # Vector Database Errors
    VECTOR_DB_CONNECTION_FAILED = "VEC_001"
    VECTOR_DB_QUERY_FAILED = "VEC_002"
    EMBEDDING_GENERATION_FAILED = "VEC_003"
    VECTOR_DB_INDEX_ERROR = "VEC_004"

    # Cache Errors
    CACHE_CONNECTION_FAILED = "CACHE_001"
    CACHE_OPERATION_FAILED = "CACHE_002"
    CACHE_KEY_NOT_FOUND = "CACHE_003"

    # Rate Limiting
    RATE_LIMIT_EXCEEDED = "RATE_001"
    TOO_MANY_REQUESTS = "RATE_002"

    # Configuration Errors
    CONFIGURATION_ERROR = "CONFIG_001"
    MISSING_CONFIGURATION = "CONFIG_002"
    INVALID_CONFIGURATION = "CONFIG_003"

    # External Service Errors
    EXTERNAL_SERVICE_ERROR = "EXT_001"
    API_KEY_INVALID = "EXT_002"
    SERVICE_UNAVAILABLE = "EXT_003"
    TIMEOUT_ERROR = "EXT_004"

    # Internal Errors
    INTERNAL_ERROR = "INT_001"
    UNEXPECTED_ERROR = "INT_002"
    RESOURCE_EXHAUSTED = "INT_003"
    OPERATION_TIMEOUT = "INT_004"


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for errors."""

    user_id: Optional[str] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    timestamp: Optional[datetime] = None
    trace_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.trace_id is None:
            self.trace_id = str(uuid.uuid4())


@dataclass
class StandardError:
    """Standardized error response format."""

    error_code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    context: Optional[ErrorContext] = None
    timestamp: Optional[datetime] = None
    trace_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.trace_id is None:
            self.trace_id = str(uuid.uuid4())
        if self.context is None:
            self.context = ErrorContext()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        result = asdict(self)
        # Convert datetime to ISO string
        if result.get("timestamp"):
            result["timestamp"] = result["timestamp"].isoformat()
        if result.get("context", {}).get("timestamp"):
            result["context"]["timestamp"] = result["context"]["timestamp"].isoformat()
        return result


class MemorizerError(Exception):
    """Base exception class for Memorizer framework."""

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.severity = severity
        self.context = context or ErrorContext()
        self.original_exception = original_exception
        self.timestamp = datetime.now(timezone.utc)
        self.trace_id = str(uuid.uuid4())

    def to_standard_error(self) -> StandardError:
        """Convert to standardized error format."""
        return StandardError(
            error_code=self.error_code,
            message=self.message,
            details=self.details,
            severity=self.severity,
            context=self.context,
            timestamp=self.timestamp,
            trace_id=self.trace_id,
        )


class AuthenticationError(MemorizerError):
    """Authentication related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorCode.AUTHENTICATION_FAILED, message, details, ErrorSeverity.HIGH
        )


class AuthorizationError(MemorizerError):
    """Authorization related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorCode.AUTHORIZATION_DENIED, message, details, ErrorSeverity.HIGH
        )


class ValidationError(MemorizerError):
    """Validation related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorCode.VALIDATION_ERROR, message, details, ErrorSeverity.MEDIUM
        )


class DatabaseError(MemorizerError):
    """Database related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorCode.DATABASE_QUERY_FAILED, message, details, ErrorSeverity.HIGH
        )


class CacheError(MemorizerError):
    """Cache related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorCode.CACHE_OPERATION_FAILED, message, details, ErrorSeverity.MEDIUM
        )


class RateLimitError(MemorizerError):
    """Rate limiting related errors."""

    def __init__(
        self,
        message: str,
        retry_after: int = 60,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        details["retry_after"] = retry_after
        super().__init__(
            ErrorCode.RATE_LIMIT_EXCEEDED, message, details, ErrorSeverity.MEDIUM
        )


class ConfigurationError(MemorizerError):
    """Configuration related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorCode.CONFIGURATION_ERROR, message, details, ErrorSeverity.CRITICAL
        )


class ExternalServiceError(MemorizerError):
    """External service related errors."""

    def __init__(
        self, message: str, service: str, details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["service"] = service
        super().__init__(
            ErrorCode.EXTERNAL_SERVICE_ERROR, message, details, ErrorSeverity.HIGH
        )


class ErrorHandler:
    """Centralized error handling and logging."""

    @staticmethod
    def log_error(
        error: Union[Exception, MemorizerError],
        context: Optional[ErrorContext] = None,
        additional_details: Optional[Dict[str, Any]] = None,
    ) -> StandardError:
        """Log error and return standardized error format."""

        # Create standardized error
        if isinstance(error, MemorizerError):
            standard_error = error.to_standard_error()
        else:
            # Convert generic exception to standardized error
            standard_error = StandardError(
                error_code=ErrorCode.UNEXPECTED_ERROR,
                message=str(error),
                details=additional_details or {},
                severity=ErrorSeverity.HIGH,
                context=context or ErrorContext(),
            )

        # Add additional details
        if additional_details:
            standard_error.details.update(additional_details)

        # Log based on severity
        log_message = f"[{standard_error.error_code}] {standard_error.message}"
        log_details = {
            "error_code": standard_error.error_code,
            "severity": standard_error.severity,
            "trace_id": standard_error.trace_id,
            "context": (
                asdict(standard_error.context) if standard_error.context else None
            ),
            "details": standard_error.details,
        }

        if standard_error.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra=log_details)
        elif standard_error.severity == ErrorSeverity.HIGH:
            logger.error(log_message, extra=log_details)
        elif standard_error.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, extra=log_details)
        else:
            logger.info(log_message, extra=log_details)

        # Log stack trace for unexpected errors
        if not isinstance(error, MemorizerError):
            logger.debug(f"Stack trace for {standard_error.trace_id}:", exc_info=True)

        return standard_error

    @staticmethod
    def get_http_status_code(error_code: ErrorCode) -> int:
        """Get appropriate HTTP status code for error."""
        status_map = {
            # Authentication & Authorization
            ErrorCode.AUTHENTICATION_FAILED: 401,
            ErrorCode.AUTHORIZATION_DENIED: 403,
            ErrorCode.INVALID_TOKEN: 401,
            ErrorCode.TOKEN_EXPIRED: 401,
            ErrorCode.INSUFFICIENT_PERMISSIONS: 403,
            # Validation Errors
            ErrorCode.VALIDATION_ERROR: 400,
            ErrorCode.INVALID_INPUT: 400,
            ErrorCode.MISSING_REQUIRED_FIELD: 400,
            ErrorCode.INVALID_FORMAT: 400,
            ErrorCode.VALUE_TOO_LONG: 400,
            ErrorCode.VALUE_TOO_SHORT: 400,
            # Database Errors
            ErrorCode.DATABASE_CONNECTION_FAILED: 503,
            ErrorCode.DATABASE_QUERY_FAILED: 500,
            ErrorCode.DATABASE_TRANSACTION_FAILED: 500,
            ErrorCode.RECORD_NOT_FOUND: 404,
            ErrorCode.DUPLICATE_RECORD: 409,
            ErrorCode.CONSTRAINT_VIOLATION: 400,
            # Memory Management Errors
            ErrorCode.MEMORY_NOT_FOUND: 404,
            ErrorCode.MEMORY_CREATION_FAILED: 500,
            ErrorCode.MEMORY_UPDATE_FAILED: 500,
            ErrorCode.MEMORY_DELETION_FAILED: 500,
            ErrorCode.MEMORY_TIER_INVALID: 400,
            # Vector Database Errors
            ErrorCode.VECTOR_DB_CONNECTION_FAILED: 503,
            ErrorCode.VECTOR_DB_QUERY_FAILED: 500,
            ErrorCode.EMBEDDING_GENERATION_FAILED: 500,
            ErrorCode.VECTOR_DB_INDEX_ERROR: 500,
            # Cache Errors
            ErrorCode.CACHE_CONNECTION_FAILED: 503,
            ErrorCode.CACHE_OPERATION_FAILED: 500,
            ErrorCode.CACHE_KEY_NOT_FOUND: 404,
            # Rate Limiting
            ErrorCode.RATE_LIMIT_EXCEEDED: 429,
            ErrorCode.TOO_MANY_REQUESTS: 429,
            # Configuration Errors
            ErrorCode.CONFIGURATION_ERROR: 500,
            ErrorCode.MISSING_CONFIGURATION: 500,
            ErrorCode.INVALID_CONFIGURATION: 500,
            # External Service Errors
            ErrorCode.EXTERNAL_SERVICE_ERROR: 502,
            ErrorCode.API_KEY_INVALID: 401,
            ErrorCode.SERVICE_UNAVAILABLE: 503,
            ErrorCode.TIMEOUT_ERROR: 504,
            # Internal Errors
            ErrorCode.INTERNAL_ERROR: 500,
            ErrorCode.UNEXPECTED_ERROR: 500,
            ErrorCode.RESOURCE_EXHAUSTED: 503,
            ErrorCode.OPERATION_TIMEOUT: 504,
        }

        return status_map.get(error_code, 500)


# Global error handler instance
_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    return _error_handler


def handle_error(
    error: Union[Exception, MemorizerError],
    context: Optional[ErrorContext] = None,
    additional_details: Optional[Dict[str, Any]] = None,
) -> StandardError:
    """Handle error and return standardized format."""
    return _error_handler.log_error(error, context, additional_details)
