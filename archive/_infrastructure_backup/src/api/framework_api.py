"""
framework_api.py
FastAPI integration with the Memorizer Framework.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from memorizer.core.framework import create_framework
from memorizer.core.config import FrameworkConfig, load_config
from memorizer.security.auth import AuthManager, AuthenticationError, AuthorizationError

logger = logging.getLogger(__name__)

# Global instances
_framework = None
_auth_manager = None

def get_framework():
    """Get or create the framework instance."""
    global _framework
    if _framework is None:
        try:
            # Try to load from config file first
            config_path = os.getenv("MEMORIZER_CONFIG_PATH", "memorizer.yaml")
            if os.path.exists(config_path):
                config = load_config(config_path)
            else:
                # Fall back to default configuration
                config = FrameworkConfig.create_default()
            _framework = create_framework(config)
        except Exception as e:
            logger.error(f"Failed to create framework: {e}")
            # Create minimal framework for API to work
            config = FrameworkConfig.create_default()
            _framework = create_framework(config)
    return _framework

def get_auth_manager():
    """Get or create the auth manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager

# Pydantic models
class MemoryCreate(BaseModel):
    user_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    tier: str = "very_new"

class MemoryUpdate(BaseModel):
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MemorySearch(BaseModel):
    user_id: str
    query: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None

class MemoryResponse(BaseModel):
    id: str
    user_id: str
    content: str
    metadata: Dict[str, Any]
    tier: str
    created_at: datetime
    updated_at: Optional[datetime] = None

class SearchResponse(BaseModel):
    memories: List[MemoryResponse]
    total_found: int
    retrieval_time: float
    source: str

class HealthResponse(BaseModel):
    framework: Dict[str, Any]
    components: Dict[str, Any]
    registry: Dict[str, Any]

class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class UserInfo(BaseModel):
    user_id: str
    permissions: List[str]
    auth_method: str

# Security scheme
security_scheme = HTTPBearer()

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
    """Get current authenticated user."""
    try:
        auth_manager = get_auth_manager()
        
        # Try JWT authentication first
        if credentials.credentials.startswith("eyJ"):
            user_info = auth_manager.jwt_auth.verify_token(credentials.credentials)
            return UserInfo(
                user_id=user_info["user_id"],
                permissions=user_info.get("permissions", []),
                auth_method="jwt"
            )
        else:
            # Try API key authentication
            user_info = auth_manager.authenticate_api_key(credentials.credentials)
            return UserInfo(
                user_id=user_info["user_id"],
                permissions=user_info.get("permissions", []),
                auth_method="api_key"
            )
    except (AuthenticationError, AuthorizationError) as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

# Optional authentication (for public endpoints)
async def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Security(security_scheme)):
    """Get current user if authenticated, otherwise return None."""
    if not credentials:
        return None
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None

# Create FastAPI app
app = FastAPI(
    title="Memorizer Framework API",
    description="Production-ready memory lifecycle framework for AI assistants and agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Memorizer Framework API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        framework = get_framework()
        health = framework.get_health_status()
        return HealthResponse(**health)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/login", response_model=AuthResponse)
async def login(user_id: str, password: str = None):
    """Login endpoint for JWT authentication."""
    try:
        auth_manager = get_auth_manager()
        
        # For development, accept any user_id
        # In production, validate against user database
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Generate JWT token
        token = auth_manager.jwt_auth.create_token(
            user_id=user_id,
            permissions=["read_memories", "write_memories"],
            expires_in=3600  # 1 hour
        )
        
        return AuthResponse(
            access_token=token,
            token_type="bearer",
            expires_in=3600
        )
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=401, detail="Login failed")

@app.get("/auth/me", response_model=UserInfo)
async def get_current_user_info(current_user: UserInfo = Depends(get_current_user)):
    """Get current user information."""
    return current_user

@app.post("/memories", response_model=Dict[str, str])
async def create_memory(memory: MemoryCreate, current_user: UserInfo = Depends(get_current_user)):
    """Create a new memory."""
    try:
        # Check permissions
        if "write_memories" not in current_user.permissions:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        framework = get_framework()
        memory_manager = framework.get_memory_manager()
        
        # Use authenticated user's ID
        memory_id = memory_manager.store_memory(
            user_id=current_user.user_id,
            content=memory.content,
            metadata=memory.metadata,
            tier=memory.tier
        )
        
        return {"memory_id": memory_id, "status": "created"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str, current_user: UserInfo = Depends(get_current_user)):
    """Get a specific memory."""
    try:
        # Check permissions
        if "read_memories" not in current_user.permissions:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        framework = get_framework()
        memory_manager = framework.get_memory_manager()
        
        memory = memory_manager.get_memory(memory_id, current_user.user_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        return MemoryResponse(
            id=memory.id,
            user_id=memory.user_id,
            content=memory.content,
            metadata=memory.metadata,
            tier=memory.tier,
            created_at=memory.created_at,
            updated_at=memory.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/memories/{memory_id}", response_model=Dict[str, str])
async def update_memory(
    memory_id: str, 
    memory_update: MemoryUpdate,
    current_user: UserInfo = Depends(get_current_user)
):
    """Update a memory."""
    try:
        # Check permissions
        if "write_memories" not in current_user.permissions:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        framework = get_framework()
        memory_manager = framework.get_memory_manager()
        
        success = memory_manager.update_memory(
            memory_id=memory_id,
            user_id=current_user.user_id,
            content=memory_update.content,
            metadata=memory_update.metadata
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found or update failed")
        
        return {"status": "updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memories/{memory_id}", response_model=Dict[str, str])
async def delete_memory(memory_id: str, current_user: UserInfo = Depends(get_current_user)):
    """Delete a memory."""
    try:
        # Check permissions
        if "delete_memories" not in current_user.permissions:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        framework = get_framework()
        memory_manager = framework.get_memory_manager()
        
        success = memory_manager.delete_memory(memory_id, current_user.user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        return {"status": "deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/search", response_model=SearchResponse)
async def search_memories(search: MemorySearch, current_user: UserInfo = Depends(get_current_user)):
    """Search for memories."""
    try:
        # Check permissions
        if "read_memories" not in current_user.permissions:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        framework = get_framework()
        memory_manager = framework.get_memory_manager()
        
        results = memory_manager.search_memories(
            user_id=current_user.user_id,  # Use authenticated user's ID
            query=search.query,
            limit=search.limit,
            filters=search.filters
        )
        
        memories = [
            MemoryResponse(
                id=memory.id,
                user_id=memory.user_id,
                content=memory.content,
                metadata=memory.metadata,
                tier=memory.tier,
                created_at=memory.created_at,
                updated_at=memory.updated_at
            )
            for memory in results.memories
        ]
        
        return SearchResponse(
            memories=memories,
            total_found=results.total_found,
            retrieval_time=results.retrieval_time,
            source=results.source
        )
        
    except Exception as e:
        logger.error(f"Failed to search memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/{memory_id}/promote", response_model=Dict[str, str])
async def promote_memory(
    memory_id: str,
    from_tier: str,
    to_tier: str,
    user_id: str = Header(..., alias="X-User-ID")
):
    """Promote a memory to a different tier."""
    try:
        framework = get_framework()
        memory_manager = framework.get_memory_manager()
        
        success = memory_manager.promote_memory(memory_id, from_tier, to_tier)
        if not success:
            raise HTTPException(status_code=400, detail="Promotion failed")
        
        return {"status": "promoted", "new_tier": to_tier}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to promote memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/stats", response_model=Dict[str, Any])
async def get_user_stats(user_id: str):
    """Get statistics for a user's memories."""
    try:
        framework = get_framework()
        memory_manager = framework.get_memory_manager()
        
        stats = memory_manager.get_user_stats(user_id)
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get user stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the server
    uvicorn.run(
        "framework_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
