"""
error_middleware.py
Error handling middleware for FastAPI applications.
Provides consistent error responses and logging.
"""

import logging
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .errors import ErrorCode, ErrorContext, MemorizerError, get_error_handler

logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling errors consistently across the application."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.error_handler = get_error_handler()

    async def dispatch(self, request: Request, call_next):
        """Process request and handle any errors."""
        # Create error context
        context = ErrorContext(
            endpoint=str(request.url.path),
            method=request.method,
            request_id=request.headers.get("x-request-id"),
            user_id=self._extract_user_id(request),
        )

        try:
            response = await call_next(request)
            return response

        except MemorizerError as e:
            # Handle framework-specific errors
            return await self._handle_memorizer_error(e, context, request)

        except Exception as e:
            # Handle unexpected errors
            return await self._handle_unexpected_error(e, context, request)

    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request headers or authentication."""
        # Try to get from authorization header
        auth_header = request.headers.get("authorization")
        if auth_header:
            # In production, decode JWT or API key to get user ID
            # For now, return a placeholder
            return "authenticated_user"

        # Try to get from custom header
        return request.headers.get("x-user-id")

    async def _handle_memorizer_error(
        self, error: MemorizerError, context: ErrorContext, request: Request
    ) -> JSONResponse:
        """Handle Memorizer framework errors."""
        # Log the error
        standard_error = self.error_handler.log_error(error, context)

        # Get HTTP status code
        status_code = self.error_handler.get_http_status_code(error.error_code)

        # Create error response
        error_response = standard_error.to_dict()

        # Add retry-after header for rate limiting errors
        headers = {}
        if error.error_code == ErrorCode.RATE_LIMIT_EXCEEDED:
            retry_after = error.details.get("retry_after", 60)
            headers["Retry-After"] = str(retry_after)

        return JSONResponse(
            status_code=status_code, content=error_response, headers=headers
        )

    async def _handle_unexpected_error(
        self, error: Exception, context: ErrorContext, request: Request
    ) -> JSONResponse:
        """Handle unexpected errors."""
        # Log the error with full stack trace
        standard_error = self.error_handler.log_error(error, context)

        # In production, don't expose internal error details
        if context.endpoint and "/health" in context.endpoint:
            # For health checks, return minimal error info
            error_response = {
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Internal server error",
                "trace_id": standard_error.trace_id,
                "timestamp": standard_error.timestamp.isoformat(),
            }
        else:
            # For other endpoints, return more details in development
            error_response = standard_error.to_dict()

            # Remove sensitive information in production
            if not self._is_development():
                error_response.pop("details", None)
                error_response.pop("context", None)

        return JSONResponse(status_code=500, content=error_response)

    def _is_development(self) -> bool:
        """Check if running in development mode."""
        import os

        return os.getenv("ENVIRONMENT", "development").lower() == "development"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = logging.getLogger("request_logger")

    async def dispatch(self, request: Request, call_next):
        """Log request and response details."""
        start_time = time.time()

        # Log request
        self.logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "user_agent": request.headers.get("user-agent"),
                "request_id": request.headers.get("x-request-id"),
            },
        )

        # Process request
        response = await call_next(request)

        # Log response
        process_time = time.time() - start_time
        self.logger.info(
            f"Response: {response.status_code} - {process_time:.3f}s",
            extra={
                "status_code": response.status_code,
                "process_time": process_time,
                "request_id": request.headers.get("x-request-id"),
            },
        )

        return response


# Import time for request logging
import time
