"""
api.py
FastAPI web interface for production deployment.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from . import auth, health, memory_manager, metrics, security
from .auth import AuthMethod
from .error_middleware import ErrorHandlingMiddleware, RequestLoggingMiddleware
from .errors import AuthorizationError, ErrorContext, RateLimitError, handle_error
from .rate_limiter import check_rate_limit, get_rate_limit_stats
from .logging_config import request_context, log_operation, log_performance
from .tracing_middleware import TracingMiddleware
from .performance_monitor import get_performance_monitor
from .health_monitor import get_health_monitor
from .automated_testing import get_automated_testing
from .dashboard import dashboard_router
from dataclasses import asdict

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/live", "/health/ready"]:
            return await call_next(request)

        # Get user ID from request (if authenticated)
        user_id = None
        try:
            # Try to get user ID from authorization header
            auth_header = request.headers.get("authorization")
            if auth_header:
                # Extract user ID from token (simplified)
                # In production, you'd decode the JWT or API key
                user_id = "authenticated_user"  # Simplified for demo
        except:
            pass

        # Check rate limit
        is_allowed, reason, retry_after = check_rate_limit(user_id)

        if not is_allowed:
            # Use standardized error handling
            error = RateLimitError(
                message=f"Rate limit exceeded: {reason}", retry_after=retry_after
            )
            context = ErrorContext(
                endpoint=request.url.path, method=request.method, user_id=user_id
            )
            standard_error = handle_error(error, context)

            return JSONResponse(
                status_code=429,
                content=standard_error.to_dict(),
                headers={"Retry-After": str(retry_after)},
            )

        response = await call_next(request)
        return response


# Create FastAPI app
app = FastAPI(
    title="Memorizer API",
    description="Production-ready memory lifecycle framework for AI assistants and agents",
    version="0.1.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
)

# Add error handling middleware (should be first)
app.add_middleware(ErrorHandlingMiddleware)

# Add request tracing middleware
app.add_middleware(TracingMiddleware)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Include dashboard router
app.include_router(dashboard_router)


# Request/Response models
class MemoryRequest(BaseModel):
    content: str = Field(..., description="Memory content", max_length=50000)
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )


class MemoryResponse(BaseModel):
    memory_id: str
    user_id: str
    content: str
    metadata: Dict[str, Any]
    tier: str
    created_at: datetime


class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query", max_length=1000)
    max_items: int = Field(
        default=5, ge=1, le=50, description="Maximum number of results"
    )


class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_found: int
    query_time_ms: float


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    checks: Dict[str, Any]
    summary: Dict[str, Any]


# Authentication dependency
async def get_current_user(
    authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """Get current authenticated user."""
    auth_manager = auth.get_auth_manager()

    # Try JWT authentication
    if authorization and authorization.startswith("Bearer "):
        try:
            token = authorization[7:]
            payload = auth_manager.authenticate_jwt(token)
            return {
                "user_id": payload["user_id"],
                "permissions": payload.get("permissions", []),
                "auth_method": AuthMethod.JWT,
            }
        except auth.AuthenticationError:
            pass

    # Try API key authentication
    if x_api_key:
        try:
            api_key_obj = auth_manager.authenticate_api_key(x_api_key)
            return {
                "user_id": api_key_obj.user_id,
                "permissions": api_key_obj.permissions,
                "auth_method": AuthMethod.API_KEY,
            }
        except auth.AuthenticationError:
            pass

    raise HTTPException(status_code=401, detail="Authentication required")


# Error handlers
# Exception handlers are now handled by ErrorHandlingMiddleware


# Health endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    health_monitor = get_health_monitor()
    health_status = health_monitor.get_health_status()
    return HealthResponse(**asdict(health_status))


@app.get("/health/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    health_monitor = get_health_monitor()
    health_status = health_monitor.get_health_status()

    if health_status.overall_status == "unhealthy":
        raise HTTPException(status_code=503, detail="Service not ready")

    return {"status": "ready"}


@app.get("/health/live")
async def liveness_check():
    """Liveness check for Kubernetes."""
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    from fastapi.responses import PlainTextResponse

    metrics_data = metrics.get_metrics()
    return PlainTextResponse(metrics_data, media_type="text/plain")


# Memory management endpoints
@app.post("/memories", response_model=MemoryResponse)
async def create_memory(
    memory_request: MemoryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Create a new memory."""
    try:
        # Check permissions
        if not auth.get_auth_manager().check_permission(
            current_user["user_id"], "write_memories"
        ):
            raise AuthorizationError("Write permission required")

        # Create memory
        memory_id = memory_manager.add_session(
            user_id=current_user["user_id"],
            content=memory_request.content,
            metadata=memory_request.metadata,
        )

        # Log operation
        security.audit_log(
            action="memory_created",
            user_id=current_user["user_id"],
            details={
                "memory_id": memory_id,
                "content_length": len(memory_request.content),
            },
            resource_id=memory_id,
        )

        # Track metrics
        metrics.track_user_operation(
            current_user["user_id"], "create_memory", "success"
        )

        return MemoryResponse(
            memory_id=memory_id,
            user_id=current_user["user_id"],
            content=memory_request.content,
            metadata=memory_request.metadata,
            tier="very_new",
            created_at=datetime.now(timezone.utc),
        )

    except Exception as e:
        logger.error(f"Failed to create memory: {e}")
        metrics.track_user_operation(current_user["user_id"], "create_memory", "error")
        raise HTTPException(status_code=500, detail="Failed to create memory")


@app.get("/memories", response_model=List[MemoryResponse])
async def list_memories(
    tier: Optional[str] = None,
    limit: int = 10,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List user's memories."""
    try:
        # Check permissions
        if not auth.get_auth_manager().check_permission(
            current_user["user_id"], "read_memories"
        ):
            raise AuthorizationError("Read permission required")

        # Get memories
        from . import db

        memories = db.fetch_memories(
            user_id=current_user["user_id"], tier=tier, limit=limit
        )

        # Track metrics
        metrics.track_user_operation(
            current_user["user_id"], "list_memories", "success"
        )

        return [
            MemoryResponse(
                memory_id=str(memory["id"]),
                user_id=memory["user_id"],
                content=memory["content"],
                metadata=memory["metadata"],
                tier=memory["tier"],
                created_at=memory["created_at"],
            )
            for memory in memories
        ]

    except Exception as e:
        logger.error(f"Failed to list memories: {e}")
        metrics.track_user_operation(current_user["user_id"], "list_memories", "error")
        raise HTTPException(status_code=500, detail="Failed to list memories")


@app.post("/query", response_model=QueryResponse)
async def query_memories(
    query_request: QueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Query user's memories."""
    try:
        # Check permissions
        if not auth.get_auth_manager().check_permission(
            current_user["user_id"], "read_memories"
        ):
            raise AuthorizationError("Read permission required")

        # Query memories
        start_time = datetime.now()
        results = memory_manager.get_context(
            user_id=current_user["user_id"],
            query=query_request.query,
            max_items=query_request.max_items,
        )
        query_time = (datetime.now() - start_time).total_seconds() * 1000

        # Track metrics
        metrics.track_user_operation(
            current_user["user_id"], "query_memories", "success"
        )
        metrics.update_retrieval_results("api", len(results))

        return QueryResponse(
            results=results,
            total_found=len(results),
            query_time_ms=round(query_time, 2),
        )

    except Exception as e:
        logger.error(f"Failed to query memories: {e}")
        metrics.track_user_operation(current_user["user_id"], "query_memories", "error")
        raise HTTPException(status_code=500, detail="Failed to query memories")


@app.get("/stats")
async def get_user_stats(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get user's memory statistics."""
    try:
        # Check permissions
        if not auth.get_auth_manager().check_permission(
            current_user["user_id"], "read_memories"
        ):
            raise AuthorizationError("Read permission required")

        stats = memory_manager.get_memory_stats(current_user["user_id"])

        # Track metrics
        metrics.track_user_operation(current_user["user_id"], "get_stats", "success")

        return stats

    except Exception as e:
        logger.error(f"Failed to get user stats: {e}")
        metrics.track_user_operation(current_user["user_id"], "get_stats", "error")
        raise HTTPException(status_code=500, detail="Failed to get user stats")


@app.get("/rate-limit/stats")
async def get_rate_limit_stats_endpoint(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get rate limiting statistics for the current user."""
    try:
        # Check permissions
        if not auth.get_auth_manager().check_permission(
            current_user["user_id"], "read_memories"
        ):
            raise AuthorizationError("Read permission required")

        user_stats = get_rate_limit_stats(current_user["user_id"])
        global_stats = get_rate_limit_stats()

        return {
            "user_stats": user_stats,
            "global_stats": global_stats,
            "user_id": current_user["user_id"],
        }

    except Exception as e:
        logger.error(f"Failed to get rate limit stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get rate limit stats")


# Admin endpoints
@app.post("/admin/move-memories")
async def move_memories_between_tiers(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Move memories between tiers (admin only)."""
    try:
        # Check admin permissions
        if not auth.get_auth_manager().check_permission(
            current_user["user_id"], "admin_access"
        ):
            raise AuthorizationError("Admin access required")

        # Move memories
        moved = memory_manager.move_memory_between_tiers(current_user["user_id"])

        # Log operation
        security.audit_log(
            action="admin_memory_move",
            user_id=current_user["user_id"],
            details=moved,
            security_level=security.SecurityLevel.HIGH,
        )

        return moved

    except Exception as e:
        logger.error(f"Failed to move memories: {e}")
        raise HTTPException(status_code=500, detail="Failed to move memories")


# Monitoring endpoints
@app.get("/monitoring/performance")
async def get_performance_metrics():
    """Get performance metrics."""
    monitor = get_performance_monitor()
    return monitor.get_metrics_summary()


@app.get("/monitoring/prometheus")
async def get_prometheus_metrics():
    """Get Prometheus-formatted metrics."""
    from fastapi.responses import PlainTextResponse

    monitor = get_performance_monitor()
    metrics = monitor.get_prometheus_metrics()
    return PlainTextResponse(metrics, media_type="text/plain")


@app.get("/monitoring/health/detailed")
async def get_detailed_health():
    """Get detailed health status."""
    health_monitor = get_health_monitor()
    return asdict(health_monitor.get_health_status())


@app.get("/monitoring/tests")
async def run_automated_tests():
    """Run automated tests."""
    testing_system = get_automated_testing()
    test_suite = testing_system.run_all_tests()
    return asdict(test_suite)


@app.get("/monitoring/tests/{component}")
async def run_component_tests(component: str):
    """Run tests for a specific component."""
    testing_system = get_automated_testing()
    test_suite = testing_system.run_component_tests(component)
    return asdict(test_suite)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Memorizer API starting up...")

    # Initialize system
    try:
        from . import db, vector_db

        db.initialize_db()
        vector_db.init_vector_db()
        logger.info("System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Memorizer API shutting down...")

    # Shutdown background workers
    try:
        memory_manager.shutdown_background_workers()
        logger.info("Background workers shut down successfully")
    except Exception as e:
        logger.error(f"Failed to shutdown background workers: {e}")


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "memorizer.api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") == "development",
    )
