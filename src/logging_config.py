"""
logging_config.py
Comprehensive logging configuration for the Memorizer framework.
Provides structured logging, request tracing, and log aggregation.
"""

import json
import logging
import logging.config
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import threading

# Thread-local storage for request context
_request_context = threading.local()


@dataclass
class LogContext:
    """Structured log context for request tracing."""

    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: str = None
    correlation_id: Optional[str] = None
    operation: Optional[str] = None
    start_time: Optional[datetime] = None
    duration_ms: Optional[float] = None

    def __post_init__(self):
        if self.trace_id is None:
            self.trace_id = str(uuid.uuid4())
        if self.start_time is None:
            self.start_time = datetime.now(timezone.utc)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log structure
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request context if available
        if hasattr(_request_context, "context") and _request_context.context:
            context = _request_context.context
            log_entry["context"] = {
                "request_id": context.request_id,
                "user_id": context.user_id,
                "session_id": context.session_id,
                "trace_id": context.trace_id,
                "correlation_id": context.correlation_id,
                "operation": context.operation,
            }

            # Add duration if available
            if context.duration_ms is not None:
                log_entry["context"]["duration_ms"] = context.duration_ms

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
            }:
                log_entry[key] = value

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class RequestTracingFilter(logging.Filter):
    """Filter to add request tracing information to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request context to log record."""
        if hasattr(_request_context, "context") and _request_context.context:
            context = _request_context.context
            record.request_id = context.request_id
            record.user_id = context.user_id
            record.trace_id = context.trace_id
            record.operation = context.operation
        return True


def get_logging_config() -> Dict[str, Any]:
    """Get comprehensive logging configuration."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "json")  # json or text

    # Base configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": StructuredFormatter,
            },
            "text": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "filters": {
            "request_tracing": {
                "()": RequestTracingFilter,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": log_format,
                "filters": ["request_tracing"],
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": log_format,
                "filters": ["request_tracing"],
                "filename": os.getenv("LOG_FILE", "logs/memorizer.log"),
                "maxBytes": int(os.getenv("LOG_MAX_BYTES", "10485760")),  # 10MB
                "backupCount": int(os.getenv("LOG_BACKUP_COUNT", "5")),
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": log_format,
                "filters": ["request_tracing"],
                "filename": os.getenv("ERROR_LOG_FILE", "logs/memorizer_errors.log"),
                "maxBytes": int(os.getenv("LOG_MAX_BYTES", "10485760")),
                "backupCount": int(os.getenv("LOG_BACKUP_COUNT", "5")),
            },
        },
        "loggers": {
            "": {  # Root logger
                "level": log_level,
                "handlers": ["console", "file", "error_file"],
                "propagate": False,
            },
            "memorizer": {
                "level": log_level,
                "handlers": ["console", "file", "error_file"],
                "propagate": False,
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "fastapi": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "sqlalchemy": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "psycopg2": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
    }

    # Add external log aggregation if configured
    if os.getenv("LOG_AGGREGATION_ENABLED", "false").lower() == "true":
        config = _add_log_aggregation(config, log_level, log_format)

    return config


def _add_log_aggregation(
    config: Dict[str, Any], log_level: str, log_format: str
) -> Dict[str, Any]:
    """Add external log aggregation handlers."""
    aggregation_type = os.getenv("LOG_AGGREGATION_TYPE", "elasticsearch")

    if aggregation_type == "elasticsearch":
        config["handlers"]["elasticsearch"] = {
            "class": "elasticsearch_logging.handler.ElasticsearchHandler",
            "level": log_level,
            "formatter": log_format,
            "filters": ["request_tracing"],
            "hosts": os.getenv("ELASTICSEARCH_HOSTS", "localhost:9200").split(","),
            "index": os.getenv("ELASTICSEARCH_INDEX", "memorizer-logs"),
        }
        config["loggers"][""]["handlers"].append("elasticsearch")
        config["loggers"]["memorizer"]["handlers"].append("elasticsearch")

    elif aggregation_type == "fluentd":
        config["handlers"]["fluentd"] = {
            "class": "fluent.handler.FluentHandler",
            "level": log_level,
            "formatter": log_format,
            "filters": ["request_tracing"],
            "tag": os.getenv("FLUENTD_TAG", "memorizer"),
            "host": os.getenv("FLUENTD_HOST", "localhost"),
            "port": int(os.getenv("FLUENTD_PORT", "24224")),
        }
        config["loggers"][""]["handlers"].append("fluentd")
        config["loggers"]["memorizer"]["handlers"].append("fluentd")

    elif aggregation_type == "splunk":
        config["handlers"]["splunk"] = {
            "class": "splunk_hec_handler.SplunkHecHandler",
            "level": log_level,
            "formatter": log_format,
            "filters": ["request_tracing"],
            "host": os.getenv("SPLUNK_HOST", "localhost"),
            "port": int(os.getenv("SPLUNK_PORT", "8088")),
            "token": os.getenv("SPLUNK_TOKEN"),
            "index": os.getenv("SPLUNK_INDEX", "memorizer"),
        }
        config["loggers"][""]["handlers"].append("splunk")
        config["loggers"]["memorizer"]["handlers"].append("splunk")

    return config


def setup_logging() -> None:
    """Setup comprehensive logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(os.getenv("LOG_FILE", "logs/memorizer.log"))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Apply logging configuration
    config = get_logging_config()
    logging.config.dictConfig(config)

    # Set up root logger
    logger = logging.getLogger("memorizer")
    logger.info(
        "Logging system initialized",
        extra={
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "log_format": os.getenv("LOG_FORMAT", "json"),
            "log_aggregation": os.getenv("LOG_AGGREGATION_ENABLED", "false"),
        },
    )


@contextmanager
def request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    operation: Optional[str] = None,
    correlation_id: Optional[str] = None,
):
    """Context manager for request tracing."""
    if request_id is None:
        request_id = str(uuid.uuid4())

    context = LogContext(
        request_id=request_id,
        user_id=user_id,
        session_id=session_id,
        operation=operation,
        correlation_id=correlation_id,
    )

    # Store context in thread-local storage
    _request_context.context = context

    start_time = time.time()

    try:
        yield context
    finally:
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        context.duration_ms = duration_ms

        # Log request completion
        logger = logging.getLogger("memorizer.request")
        logger.info(
            f"Request {operation or 'completed'}",
            extra={
                "request_id": request_id,
                "user_id": user_id,
                "operation": operation,
                "duration_ms": duration_ms,
                "status": "completed",
            },
        )

        # Clear context
        _request_context.context = None


def get_request_context() -> Optional[LogContext]:
    """Get current request context."""
    return getattr(_request_context, "context", None)


def log_operation(
    operation: str, level: str = "INFO", message: str = None, **kwargs
) -> None:
    """Log an operation with current request context."""
    logger = logging.getLogger("memorizer.operation")

    if message is None:
        message = f"Operation: {operation}"

    log_data = {"operation": operation, **kwargs}

    # Add request context if available
    context = get_request_context()
    if context:
        log_data.update(
            {
                "request_id": context.request_id,
                "user_id": context.user_id,
                "trace_id": context.trace_id,
            }
        )

    getattr(logger, level.lower())(message, extra=log_data)


def log_performance(
    operation: str, duration_ms: float, success: bool = True, **kwargs
) -> None:
    """Log performance metrics."""
    logger = logging.getLogger("memorizer.performance")

    log_data = {
        "operation": operation,
        "duration_ms": duration_ms,
        "success": success,
        "performance_category": "operation_timing",
        **kwargs,
    }

    # Add request context if available
    context = get_request_context()
    if context:
        log_data.update(
            {
                "request_id": context.request_id,
                "user_id": context.user_id,
                "trace_id": context.trace_id,
            }
        )

    level = "INFO" if success else "WARNING"
    logger.log(
        getattr(logging, level),
        f"Performance: {operation} took {duration_ms:.2f}ms",
        extra=log_data,
    )


def log_security_event(
    event_type: str, user_id: Optional[str] = None, severity: str = "INFO", **kwargs
) -> None:
    """Log security-related events."""
    logger = logging.getLogger("memorizer.security")

    log_data = {
        "event_type": event_type,
        "user_id": user_id,
        "security_category": "security_event",
        "severity": severity,
        **kwargs,
    }

    # Add request context if available
    context = get_request_context()
    if context:
        log_data.update(
            {
                "request_id": context.request_id,
                "trace_id": context.trace_id,
            }
        )

    getattr(logger, severity.lower())(f"Security event: {event_type}", extra=log_data)


def log_business_event(
    event_type: str, user_id: Optional[str] = None, **kwargs
) -> None:
    """Log business-related events."""
    logger = logging.getLogger("memorizer.business")

    log_data = {
        "event_type": event_type,
        "user_id": user_id,
        "business_category": "business_event",
        **kwargs,
    }

    # Add request context if available
    context = get_request_context()
    if context:
        log_data.update(
            {
                "request_id": context.request_id,
                "trace_id": context.trace_id,
            }
        )

    logger.info(f"Business event: {event_type}", extra=log_data)


# Initialize logging on module import
setup_logging()
