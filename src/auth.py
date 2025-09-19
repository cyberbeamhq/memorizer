"""
auth.py
Production-ready authentication and authorization system.
"""

import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional

import jwt

from . import security

logger = logging.getLogger(__name__)


class AuthMethod(Enum):
    """Authentication methods."""

    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH = "oauth"


class APIKey:
    """API Key management."""

    def __init__(
        self,
        key_id: str,
        user_id: str,
        permissions: List[str],
        expires_at: Optional[datetime] = None,
        key_hash: str = None,
    ):
        self.key_id = key_id
        self.user_id = user_id
        self.permissions = permissions
        self.expires_at = expires_at
        self.created_at = datetime.now(timezone.utc)
        self.last_used = None
        self.key_hash = key_hash  # Store the hash of the actual key

    def is_valid(self) -> bool:
        """Check if API key is valid."""
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True

    def has_permission(self, permission: str) -> bool:
        """Check if API key has specific permission."""
        return permission in self.permissions


class JWTAuth:
    """JWT-based authentication."""

    def __init__(
        self, secret_key: str = None, algorithm: str = "HS256", expires_in: int = 3600
    ):
        self.secret_key = secret_key or os.getenv(
            "JWT_SECRET_KEY", secrets.token_urlsafe(32)
        )
        self.algorithm = algorithm
        self.expires_in = expires_in

    def create_token(
        self, user_id: str, permissions: List[str] = None, expires_in: int = None
    ) -> str:
        """Create JWT token."""
        if permissions is None:
            permissions = []

        if expires_in is None:
            expires_in = self.expires_in

        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(seconds=expires_in),
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")


class AuthenticationError(Exception):
    """Authentication error."""

    pass


class AuthorizationError(Exception):
    """Authorization error."""

    pass


class AuthManager:
    """Main authentication manager."""

    def __init__(self):
        self.jwt_auth = JWTAuth()
        self.api_keys: Dict[str, APIKey] = {}
        self._load_api_keys()

    def _load_api_keys(self):
        """Load API keys from database."""
        try:
            # In production, load from database
            # For development, create some default keys with proper hashing
            default_keys = [
                {
                    "key_id": "admin_key",
                    "user_id": "admin_user",
                    "permissions": [
                        "read_memories",
                        "write_memories",
                        "delete_memories",
                        "admin_access",
                    ],
                    "expires_at": None,
                    "key_hash": hashlib.sha256(
                        "dev_admin_key_12345".encode()
                    ).hexdigest(),
                },
                {
                    "key_id": "user_key",
                    "user_id": "regular_user",
                    "permissions": ["read_memories", "write_memories"],
                    "expires_at": None,
                    "key_hash": hashlib.sha256(
                        "dev_user_key_67890".encode()
                    ).hexdigest(),
                },
            ]

            for key_data in default_keys:
                api_key = APIKey(**key_data)
                self.api_keys[key_data["key_id"]] = api_key

            logger.warning(
                "Using default API keys for development. Change these in production!"
            )

        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")

    def authenticate_api_key(self, api_key: str) -> APIKey:
        """Authenticate using API key."""
        if not api_key or not isinstance(api_key, str):
            raise AuthenticationError("Invalid API key format")

        # Hash the provided key for lookup
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Look up by the actual key hash, not key_id hash
        for key_id, key_obj in self.api_keys.items():
            # Compare the provided key hash with stored key hash
            if hasattr(key_obj, "key_hash") and key_obj.key_hash == key_hash:
                if key_obj.is_valid():
                    key_obj.last_used = datetime.now(timezone.utc)
                    return key_obj
                else:
                    raise AuthenticationError("API key has expired")

        raise AuthenticationError("Invalid API key")

    def authenticate_jwt(self, token: str) -> Dict[str, Any]:
        """Authenticate using JWT token."""
        return self.jwt_auth.verify_token(token)

    def create_api_key(
        self, user_id: str, permissions: List[str], expires_in_days: int = None
    ) -> str:
        """Create new API key."""
        # Generate the actual API key (this is what the user will use)
        actual_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(actual_key.encode()).hexdigest()

        # Generate a separate key_id for internal tracking
        key_id = secrets.token_urlsafe(16)

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        api_key = APIKey(key_id, user_id, permissions, expires_at, key_hash)
        self.api_keys[key_id] = api_key

        # Log API key creation
        security.audit_log(
            action="api_key_created",
            user_id=user_id,
            details={
                "key_id": key_id,
                "permissions": permissions,
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
            security_level=security.SecurityLevel.HIGH,
        )

        # Return the actual key (not the key_id)
        return actual_key

    def revoke_api_key(self, key_id: str, user_id: str):
        """Revoke API key."""
        if key_id in self.api_keys:
            del self.api_keys[key_id]

            # Log API key revocation
            security.audit_log(
                action="api_key_revoked",
                user_id=user_id,
                details={"key_id": key_id},
                security_level=security.SecurityLevel.HIGH,
            )

    def check_permission(
        self, user_id: str, permission: str, auth_method: AuthMethod = None
    ) -> bool:
        """Check if user has permission."""
        try:
            # Get user role
            role = security.get_user_role(user_id)
            user_permissions = security.ROLE_PERMISSIONS.get(role, set())

            # Check permission
            has_permission = permission in user_permissions

            # Log permission check
            security.audit_log(
                action="permission_check",
                user_id=user_id,
                details={
                    "permission": permission,
                    "granted": has_permission,
                    "auth_method": auth_method.value if auth_method else None,
                },
                security_level=security.SecurityLevel.MEDIUM,
            )

            return has_permission

        except Exception as e:
            logger.error(f"Permission check failed for user {user_id}: {e}")
            return False


# Global auth manager instance
_auth_manager = AuthManager()


def require_auth(permission: str = None, auth_method: AuthMethod = None):
    """Decorator to require authentication."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract authentication info from request context
            # This would be implemented based on your web framework
            auth_header = kwargs.get("auth_header")
            api_key = kwargs.get("api_key")

            user_id = None
            authenticated_method = None

            # Try JWT authentication
            if auth_header and auth_header.startswith("Bearer "):
                try:
                    token = auth_header[7:]  # Remove 'Bearer ' prefix
                    payload = _auth_manager.authenticate_jwt(token)
                    user_id = payload["user_id"]
                    authenticated_method = AuthMethod.JWT
                except AuthenticationError:
                    pass

            # Try API key authentication
            if not user_id and api_key:
                try:
                    api_key_obj = _auth_manager.authenticate_api_key(api_key)
                    user_id = api_key_obj.user_id
                    authenticated_method = AuthMethod.API_KEY
                except AuthenticationError:
                    pass

            if not user_id:
                raise AuthenticationError("Authentication required")

            # Check permission if specified
            if permission:
                if not _auth_manager.check_permission(
                    user_id, permission, authenticated_method
                ):
                    raise AuthorizationError(f"Permission '{permission}' required")

            # Add user info to kwargs
            kwargs["user_id"] = user_id
            kwargs["auth_method"] = authenticated_method

            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_auth_manager() -> AuthManager:
    """Get the global auth manager instance."""
    return _auth_manager
