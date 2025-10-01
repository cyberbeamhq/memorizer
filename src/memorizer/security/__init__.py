"""
Security components.
"""

from .pii_detection import PIIDetector, detect_pii, sanitize_text, has_pii

__all__ = [
    "PIIDetector",
    "detect_pii",
    "sanitize_text",
    "has_pii",
]

# Optional components - import only if available
try:
    from .auth import AuthManager, JWTAuth, APIKey
    __all__.extend(["AuthManager", "JWTAuth", "APIKey"])
except ImportError:
    pass

try:
    from .security import SecurityLevel, ActionType, Role, Permission
    __all__.extend(["SecurityLevel", "ActionType", "Role", "Permission"])
except ImportError:
    pass

try:
    from .rate_limiter import RateLimiter
    __all__.extend(["RateLimiter"])
except ImportError:
    pass
