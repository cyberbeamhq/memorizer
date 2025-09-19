"""
tracing_middleware.py
Request tracing middleware for FastAPI applications.
Provides distributed tracing and request correlation.
"""
import time
import uuid
from typing import Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .logging_config import request_context, log_operation, log_performance

class TracingMiddleware(BaseHTTPMiddleware):
    """Middleware for request tracing and correlation."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        """Process request with tracing."""
        # Extract or generate request ID
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        
        # Extract user ID from headers or authentication
        user_id = self._extract_user_id(request)
        
        # Extract session ID
        session_id = request.headers.get("x-session-id")
        
        # Extract correlation ID
        correlation_id = request.headers.get("x-correlation-id")
        
        # Determine operation
        operation = f"{request.method} {request.url.path}"
        
        # Start request context
        with request_context(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            operation=operation,
            correlation_id=correlation_id,
        ):
            # Log request start
            log_operation(
                operation="request_start",
                message=f"Request started: {operation}",
                method=request.method,
                path=request.url.path,
                query_params=dict(request.query_params),
                user_agent=request.headers.get("user-agent"),
                content_length=request.headers.get("content-length"),
            )
            
            start_time = time.time()
            
            try:
                # Process request
                response = await call_next(request)
                
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Log request completion
                log_performance(
                    operation=operation,
                    duration_ms=duration_ms,
                    success=True,
                    status_code=response.status_code,
                    response_size=response.headers.get("content-length"),
                )
                
                # Add tracing headers to response
                response.headers["x-request-id"] = request_id
                if correlation_id:
                    response.headers["x-correlation-id"] = correlation_id
                
                return response
                
            except Exception as e:
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Log request failure
                log_performance(
                    operation=operation,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                
                # Re-raise the exception
                raise
    
    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request."""
        # Try to get from authorization header
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In production, decode JWT to get user ID
            # For now, return a placeholder
            return "authenticated_user"
        
        # Try to get from custom header
        return request.headers.get("x-user-id")

class DatabaseTracingMiddleware:
    """Middleware for database operation tracing."""
    
    @staticmethod
    def trace_database_operation(operation: str, query: str = None, **kwargs):
        """Trace database operations."""
        log_operation(
            operation=f"database_{operation}",
            message=f"Database operation: {operation}",
            query=query,
            **kwargs
        )
    
    @staticmethod
    def trace_database_performance(operation: str, duration_ms: float, success: bool = True, **kwargs):
        """Trace database performance."""
        log_performance(
            operation=f"database_{operation}",
            duration_ms=duration_ms,
            success=success,
            **kwargs
        )

class CacheTracingMiddleware:
    """Middleware for cache operation tracing."""
    
    @staticmethod
    def trace_cache_operation(operation: str, key: str = None, **kwargs):
        """Trace cache operations."""
        log_operation(
            operation=f"cache_{operation}",
            message=f"Cache operation: {operation}",
            cache_key=key,
            **kwargs
        )
    
    @staticmethod
    def trace_cache_performance(operation: str, duration_ms: float, hit: bool = None, **kwargs):
        """Trace cache performance."""
        log_performance(
            operation=f"cache_{operation}",
            duration_ms=duration_ms,
            success=True,
            cache_hit=hit,
            **kwargs
        )

class ExternalServiceTracingMiddleware:
    """Middleware for external service tracing."""
    
    @staticmethod
    def trace_external_service(service: str, operation: str, **kwargs):
        """Trace external service calls."""
        log_operation(
            operation=f"external_{service}_{operation}",
            message=f"External service call: {service}.{operation}",
            service=service,
            **kwargs
        )
    
    @staticmethod
    def trace_external_performance(service: str, operation: str, duration_ms: float, success: bool = True, **kwargs):
        """Trace external service performance."""
        log_performance(
            operation=f"external_{service}_{operation}",
            duration_ms=duration_ms,
            success=success,
            service=service,
            **kwargs
        )
