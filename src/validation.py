"""
validation.py
Comprehensive input validation for the Memorizer framework.
Provides security-focused validation for all user inputs.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation operation."""

    is_valid: bool
    errors: List[str]
    sanitized_value: Any = None


class InputValidator:
    """Comprehensive input validator."""

    # Constants for validation
    MAX_USER_ID_LENGTH = 255
    MAX_CONTENT_LENGTH = 100000  # 100KB
    MAX_METADATA_SIZE = 10000  # 10KB
    MAX_QUERY_LENGTH = 1000
    MAX_SESSION_ID_LENGTH = 255
    MAX_MEMORY_ID_LENGTH = 255

    # Regex patterns
    USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")
    SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")
    MEMORY_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")

    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",  # Script tags
        r"javascript:",  # JavaScript URLs
        r"data:text/html",  # Data URLs
        r"vbscript:",  # VBScript
        r"onload\s*=",  # Event handlers
        r"onerror\s*=",  # Event handlers
        r"onclick\s*=",  # Event handlers
        r"<iframe[^>]*>",  # Iframe tags
        r"<object[^>]*>",  # Object tags
        r"<embed[^>]*>",  # Embed tags
        r"<link[^>]*>",  # Link tags
        r"<meta[^>]*>",  # Meta tags
        r"<style[^>]*>.*?</style>",  # Style tags
    ]

    @classmethod
    def validate_user_id(cls, user_id: Any) -> ValidationResult:
        """Validate user ID."""
        errors = []

        if not user_id:
            errors.append("User ID cannot be empty")
            return ValidationResult(False, errors)

        if not isinstance(user_id, str):
            errors.append("User ID must be a string")
            return ValidationResult(False, errors)

        if len(user_id) > cls.MAX_USER_ID_LENGTH:
            errors.append(f"User ID too long (max {cls.MAX_USER_ID_LENGTH} characters)")

        if not cls.USER_ID_PATTERN.match(user_id):
            errors.append(
                "User ID contains invalid characters (only alphanumeric, dots, hyphens, underscores allowed)"
            )

        # Check for dangerous patterns
        if cls._contains_dangerous_patterns(user_id):
            errors.append("User ID contains potentially dangerous content")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_value=user_id.strip() if isinstance(user_id, str) else user_id,
        )

    @classmethod
    def validate_content(cls, content: Any) -> ValidationResult:
        """Validate content."""
        errors = []

        if content is None:
            errors.append("Content cannot be None")
            return ValidationResult(False, errors)

        if not isinstance(content, str):
            errors.append("Content must be a string")
            return ValidationResult(False, errors)

        if len(content) > cls.MAX_CONTENT_LENGTH:
            errors.append(f"Content too long (max {cls.MAX_CONTENT_LENGTH} characters)")

        # Check for dangerous patterns
        if cls._contains_dangerous_patterns(content):
            errors.append("Content contains potentially dangerous patterns")

        # Check for excessive whitespace (potential DoS)
        if len(content.strip()) == 0:
            errors.append("Content cannot be empty or only whitespace")

        sanitized_content = content.strip()

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, sanitized_value=sanitized_content
        )

    @classmethod
    def validate_metadata(cls, metadata: Any) -> ValidationResult:
        """Validate metadata."""
        errors = []

        if metadata is None:
            return ValidationResult(True, [], {})

        if not isinstance(metadata, dict):
            errors.append("Metadata must be a dictionary")
            return ValidationResult(False, errors)

        # Check metadata size
        try:
            import json

            metadata_str = json.dumps(metadata)
            if len(metadata_str) > cls.MAX_METADATA_SIZE:
                errors.append(f"Metadata too large (max {cls.MAX_METADATA_SIZE} bytes)")
        except (TypeError, ValueError) as e:
            errors.append(f"Metadata contains non-serializable data: {e}")

        # Validate metadata keys and values
        sanitized_metadata = {}
        for key, value in metadata.items():
            # Validate key
            if not isinstance(key, str):
                errors.append(f"Metadata key must be string, got {type(key)}")
                continue

            if len(key) > 100:  # Reasonable key length limit
                errors.append(f"Metadata key too long: {key}")
                continue

            if cls._contains_dangerous_patterns(key):
                errors.append(f"Metadata key contains dangerous patterns: {key}")
                continue

            # Validate value
            if isinstance(value, str):
                if len(value) > 1000:  # Reasonable value length limit
                    errors.append(f"Metadata value too long for key: {key}")
                    continue

                if cls._contains_dangerous_patterns(value):
                    errors.append(
                        f"Metadata value contains dangerous patterns for key: {key}"
                    )
                    continue

                sanitized_metadata[key] = value.strip()
            elif isinstance(value, (int, float, bool, type(None))):
                sanitized_metadata[key] = value
            elif isinstance(value, list):
                # Validate list values
                sanitized_list = []
                for item in value:
                    if isinstance(item, str):
                        if len(item) > 500:
                            errors.append(f"List item too long in metadata key: {key}")
                            continue
                        if cls._contains_dangerous_patterns(item):
                            errors.append(
                                f"List item contains dangerous patterns in metadata key: {key}"
                            )
                            continue
                        sanitized_list.append(item.strip())
                    elif isinstance(item, (int, float, bool, type(None))):
                        sanitized_list.append(item)
                    else:
                        errors.append(
                            f"Unsupported list item type in metadata key: {key}"
                        )
                sanitized_metadata[key] = sanitized_list
            else:
                errors.append(
                    f"Unsupported metadata value type for key {key}: {type(value)}"
                )

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, sanitized_value=sanitized_metadata
        )

    @classmethod
    def validate_query(cls, query: Any) -> ValidationResult:
        """Validate search query."""
        errors = []

        if not query:
            errors.append("Query cannot be empty")
            return ValidationResult(False, errors)

        if not isinstance(query, str):
            errors.append("Query must be a string")
            return ValidationResult(False, errors)

        if len(query) > cls.MAX_QUERY_LENGTH:
            errors.append(f"Query too long (max {cls.MAX_QUERY_LENGTH} characters)")

        # Check for dangerous patterns
        if cls._contains_dangerous_patterns(query):
            errors.append("Query contains potentially dangerous patterns")

        # Check for SQL injection patterns
        if cls._contains_sql_injection_patterns(query):
            errors.append("Query contains potential SQL injection patterns")

        sanitized_query = query.strip()

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, sanitized_value=sanitized_query
        )

    @classmethod
    def validate_session_id(cls, session_id: Any) -> ValidationResult:
        """Validate session ID."""
        errors = []

        if session_id is None:
            return ValidationResult(True, [], None)

        if not isinstance(session_id, str):
            errors.append("Session ID must be a string")
            return ValidationResult(False, errors)

        if len(session_id) > cls.MAX_SESSION_ID_LENGTH:
            errors.append(
                f"Session ID too long (max {cls.MAX_SESSION_ID_LENGTH} characters)"
            )

        if not cls.SESSION_ID_PATTERN.match(session_id):
            errors.append("Session ID contains invalid characters")

        if cls._contains_dangerous_patterns(session_id):
            errors.append("Session ID contains potentially dangerous content")

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, sanitized_value=session_id.strip()
        )

    @classmethod
    def validate_memory_id(cls, memory_id: Any) -> ValidationResult:
        """Validate memory ID."""
        errors = []

        if not memory_id:
            errors.append("Memory ID cannot be empty")
            return ValidationResult(False, errors)

        if not isinstance(memory_id, str):
            errors.append("Memory ID must be a string")
            return ValidationResult(False, errors)

        if len(memory_id) > cls.MAX_MEMORY_ID_LENGTH:
            errors.append(
                f"Memory ID too long (max {cls.MAX_MEMORY_ID_LENGTH} characters)"
            )

        if not cls.MEMORY_ID_PATTERN.match(memory_id):
            errors.append("Memory ID contains invalid characters")

        if cls._contains_dangerous_patterns(memory_id):
            errors.append("Memory ID contains potentially dangerous content")

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, sanitized_value=memory_id.strip()
        )

    @classmethod
    def validate_tier(cls, tier: Any) -> ValidationResult:
        """Validate memory tier."""
        errors = []

        if not tier:
            errors.append("Tier cannot be empty")
            return ValidationResult(False, errors)

        if not isinstance(tier, str):
            errors.append("Tier must be a string")
            return ValidationResult(False, errors)

        valid_tiers = ["very_new", "mid_term", "long_term"]
        if tier not in valid_tiers:
            errors.append(f"Invalid tier: {tier}. Must be one of: {valid_tiers}")

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, sanitized_value=tier
        )

    @classmethod
    def validate_limit(cls, limit: Any) -> ValidationResult:
        """Validate limit parameter."""
        errors = []

        if limit is None:
            return ValidationResult(True, [], None)

        if not isinstance(limit, int):
            try:
                limit = int(limit)
            except (ValueError, TypeError):
                errors.append("Limit must be an integer")
                return ValidationResult(False, errors)

        if limit < 0:
            errors.append("Limit cannot be negative")

        if limit > 1000:  # Reasonable upper limit
            errors.append("Limit too large (max 1000)")

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, sanitized_value=limit
        )

    @classmethod
    def validate_offset(cls, offset: Any) -> ValidationResult:
        """Validate offset parameter."""
        errors = []

        if offset is None:
            return ValidationResult(True, [], 0)

        if not isinstance(offset, int):
            try:
                offset = int(offset)
            except (ValueError, TypeError):
                errors.append("Offset must be an integer")
                return ValidationResult(False, errors)

        if offset < 0:
            errors.append("Offset cannot be negative")

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, sanitized_value=offset
        )

    @classmethod
    def _contains_dangerous_patterns(cls, text: str) -> bool:
        """Check if text contains dangerous patterns."""
        text_lower = text.lower()
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                return True
        return False

    @classmethod
    def _contains_sql_injection_patterns(cls, text: str) -> bool:
        """Check for potential SQL injection patterns."""
        sql_patterns = [
            r"(\bunion\b.*\bselect\b)",  # UNION SELECT
            r"(\bselect\b.*\bfrom\b)",  # SELECT FROM
            r"(\binsert\b.*\binto\b)",  # INSERT INTO
            r"(\bupdate\b.*\bset\b)",  # UPDATE SET
            r"(\bdelete\b.*\bfrom\b)",  # DELETE FROM
            r"(\bdrop\b.*\btable\b)",  # DROP TABLE
            r"(\balter\b.*\btable\b)",  # ALTER TABLE
            r"(\bcreate\b.*\btable\b)",  # CREATE TABLE
            r"(\bexec\b)",  # EXEC
            r"(\bexecute\b)",  # EXECUTE
            r"(\bsp_\w+)",  # Stored procedures
            r"(\bxp_\w+)",  # Extended procedures
            r"(\b--\b)",  # SQL comments
            r"(\b/\*.*\*/\b)",  # SQL block comments
            r"(\bwaitfor\b.*\bdelay\b)",  # WAITFOR DELAY
            r"(\bwaitfor\b.*\btime\b)",  # WAITFOR TIME
        ]

        text_lower = text.lower()
        for pattern in sql_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                return True
        return False


class ValidationError(Exception):
    """Validation error exception."""

    def __init__(self, message: str, errors: List[str] = None):
        super().__init__(message)
        self.errors = errors or []


def validate_memory_input(
    user_id: str, content: str, metadata: Dict[str, Any] = None, session_id: str = None
) -> Dict[str, Any]:
    """Validate memory input parameters."""
    errors = []
    validated_data = {}

    # Validate user_id
    user_result = InputValidator.validate_user_id(user_id)
    if not user_result.is_valid:
        errors.extend([f"User ID: {error}" for error in user_result.errors])
    else:
        validated_data["user_id"] = user_result.sanitized_value

    # Validate content
    content_result = InputValidator.validate_content(content)
    if not content_result.is_valid:
        errors.extend([f"Content: {error}" for error in content_result.errors])
    else:
        validated_data["content"] = content_result.sanitized_value

    # Validate metadata
    metadata_result = InputValidator.validate_metadata(metadata)
    if not metadata_result.is_valid:
        errors.extend([f"Metadata: {error}" for error in metadata_result.errors])
    else:
        validated_data["metadata"] = metadata_result.sanitized_value

    # Validate session_id
    if session_id is not None:
        session_result = InputValidator.validate_session_id(session_id)
        if not session_result.is_valid:
            errors.extend([f"Session ID: {error}" for error in session_result.errors])
        else:
            validated_data["session_id"] = session_result.sanitized_value

    if errors:
        raise ValidationError("Input validation failed", errors)

    return validated_data


def validate_query_input(
    user_id: str, query: str, limit: int = None, offset: int = None
) -> Dict[str, Any]:
    """Validate query input parameters."""
    errors = []
    validated_data = {}

    # Validate user_id
    user_result = InputValidator.validate_user_id(user_id)
    if not user_result.is_valid:
        errors.extend([f"User ID: {error}" for error in user_result.errors])
    else:
        validated_data["user_id"] = user_result.sanitized_value

    # Validate query
    query_result = InputValidator.validate_query(query)
    if not query_result.is_valid:
        errors.extend([f"Query: {error}" for error in query_result.errors])
    else:
        validated_data["query"] = query_result.sanitized_value

    # Validate limit
    if limit is not None:
        limit_result = InputValidator.validate_limit(limit)
        if not limit_result.is_valid:
            errors.extend([f"Limit: {error}" for error in limit_result.errors])
        else:
            validated_data["limit"] = limit_result.sanitized_value

    # Validate offset
    if offset is not None:
        offset_result = InputValidator.validate_offset(offset)
        if not offset_result.is_valid:
            errors.extend([f"Offset: {error}" for error in offset_result.errors])
        else:
            validated_data["offset"] = offset_result.sanitized_value

    if errors:
        raise ValidationError("Query validation failed", errors)

    return validated_data
