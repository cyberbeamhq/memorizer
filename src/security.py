"""
security.py
Basic security, audit logging, and RBAC hooks.
Optional but recommended for enterprise deployments.
"""

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------
# Security Configuration
# ---------------------------
class SecurityLevel(Enum):
    """Security levels for different operations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(Enum):
    """Types of actions that can be audited."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    COMPRESS = "compress"
    RETRIEVE = "retrieve"
    ADMIN = "admin"


# ---------------------------
# Audit Logging
# ---------------------------
def audit_log(
    action: str,
    user_id: str,
    details: Dict[str, Any] = None,
    security_level: SecurityLevel = SecurityLevel.MEDIUM,
    resource_id: str = None,
    success: bool = True,
) -> None:
    """
    Log security-relevant actions for auditing.

    Args:
        action: Action performed (e.g., 'read', 'write', 'delete')
        user_id: User who performed the action
        details: Additional details about the action
        security_level: Security level of the action
        resource_id: ID of the resource affected
        success: Whether the action was successful
    """
    if details is None:
        details = {}

    audit_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "user_id": user_id,
        "resource_id": resource_id,
        "security_level": security_level.value,
        "success": success,
        "details": details,
        "session_id": details.get("session_id"),
        "ip_address": details.get("ip_address"),
        "user_agent": details.get("user_agent"),
    }

    # Log to standard logger with appropriate level
    log_message = f"AUDIT: {action} by {user_id} on {resource_id or 'unknown'}"
    if not success:
        log_message += " [FAILED]"

    if security_level == SecurityLevel.CRITICAL:
        logger.critical(log_message, extra={"audit": audit_entry})
    elif security_level == SecurityLevel.HIGH:
        logger.error(log_message, extra={"audit": audit_entry})
    elif security_level == SecurityLevel.MEDIUM:
        logger.warning(log_message, extra={"audit": audit_entry})
    else:
        logger.info(log_message, extra={"audit": audit_entry})

    # In production, you might want to send to external audit system
    _send_to_audit_system(audit_entry)


def _send_to_audit_system(audit_entry: Dict[str, Any]) -> None:
    """
    Send audit entry to external audit system.
    This is a placeholder for integration with systems like Splunk, ELK, etc.
    """
    # In a real implementation, you would:
    # 1. Send to external audit service
    # 2. Store in dedicated audit database
    # 3. Send alerts for critical events

    audit_file = os.getenv("AUDIT_LOG_FILE", "/tmp/memorizer_audit.log")
    try:
        with open(audit_file, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write audit log: {e}")


# ---------------------------
# Role-Based Access Control
# ---------------------------
class Role(Enum):
    """User roles in the system."""

    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    GUEST = "guest"


class Permission(Enum):
    """Permissions that can be granted to roles."""

    READ_MEMORIES = "read_memories"
    WRITE_MEMORIES = "write_memories"
    DELETE_MEMORIES = "delete_memories"
    COMPRESS_MEMORIES = "compress_memories"
    ADMIN_ACCESS = "admin_access"
    AUDIT_ACCESS = "audit_access"


# Default role permissions
ROLE_PERMISSIONS = {
    Role.ADMIN: set(Permission),
    Role.USER: {
        Permission.READ_MEMORIES,
        Permission.WRITE_MEMORIES,
        Permission.DELETE_MEMORIES,
        Permission.COMPRESS_MEMORIES,
    },
    Role.READONLY: {Permission.READ_MEMORIES},
    Role.GUEST: set(),  # No permissions by default
}


def get_user_role(user_id: str) -> Role:
    """
    Get user role from user ID.
    In a real system, this would query a user database.

    Args:
        user_id: User identifier

    Returns:
        User's role
    """
    # Simple role mapping for demo purposes
    # In production, this would query a user management system
    if user_id.startswith("admin_"):
        return Role.ADMIN
    elif user_id.startswith("readonly_"):
        return Role.READONLY
    elif user_id.startswith("guest_"):
        return Role.GUEST
    else:
        return Role.USER


def check_access(user_id: str, permission: Permission, resource_id: str = None) -> bool:
    """
    Check if user has permission to perform an action.

    Args:
        user_id: User identifier
        permission: Permission to check
        resource_id: Optional resource ID for context

    Returns:
        True if user has permission, False otherwise
    """
    try:
        user_role = get_user_role(user_id)
        user_permissions = ROLE_PERMISSIONS.get(user_role, set())

        has_permission = permission in user_permissions

        # Log the access check
        audit_log(
            action="access_check",
            user_id=user_id,
            details={
                "permission": permission.value,
                "role": user_role.value,
                "resource_id": resource_id,
                "granted": has_permission,
            },
            security_level=SecurityLevel.MEDIUM,
            resource_id=resource_id,
            success=has_permission,
        )

        return has_permission

    except Exception as e:
        logger.error(f"Access check failed for user {user_id}: {e}")
        audit_log(
            action="access_check",
            user_id=user_id,
            details={"permission": permission.value, "error": str(e)},
            security_level=SecurityLevel.HIGH,
            success=False,
        )
        return False
