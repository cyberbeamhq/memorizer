"""
utils.py
Utility functions shared across the framework.
"""

import functools
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------
# Retry Logic
# ---------------------------
def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator for retrying function calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry on

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise e

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
                    logger.info(f"Retrying in {current_delay} seconds...")

                    time.sleep(current_delay)
                    current_delay *= backoff_factor

            # This should never be reached, but just in case
            raise last_exception

        return wrapper

    return decorator


# ---------------------------
# JSON Utilities
# ---------------------------
def safe_parse_json(raw: str, fallback: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Safely parse JSON string with fallback handling.

    Args:
        raw: JSON string to parse
        fallback: Fallback dictionary if parsing fails

    Returns:
        Parsed JSON dictionary or fallback
    """
    if fallback is None:
        fallback = {"error": "json_parse_failed"}

    if not raw or not raw.strip():
        logger.warning("Empty JSON string provided")
        return fallback

    try:
        # Try to extract JSON from response (handle cases where LLM adds extra text)
        raw = raw.strip()

        # Look for JSON block markers
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_text = raw[json_start:json_end]
        else:
            json_text = raw

        parsed = json.loads(json_text)

        # Ensure result is a dictionary
        if not isinstance(parsed, dict):
            logger.warning(f"JSON parsed to non-dict type: {type(parsed)}")
            return fallback

        return parsed

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"JSON parsing failed: {e}. Raw response: {raw[:100]}...")
        return fallback


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize object to JSON string.

    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps

    Returns:
        JSON string or error message
    """
    try:
        return json.dumps(obj, **kwargs)
    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization failed: {e}")
        return json.dumps({"error": "serialization_failed", "message": str(e)})


# ---------------------------
# Time Utilities
# ---------------------------
def now_ts() -> str:
    """
    Get current timestamp as ISO string.

    Returns:
        Current timestamp in ISO format
    """
    return datetime.now(timezone.utc).isoformat()


def utc_now() -> datetime:
    """
    Get current UTC datetime.

    Returns:
        Current UTC datetime
    """
    return datetime.now(timezone.utc)


def parse_iso_timestamp(iso_string: str) -> Optional[datetime]:
    """
    Parse ISO timestamp string to datetime object.

    Args:
        iso_string: ISO format timestamp string

    Returns:
        Datetime object or None if parsing fails
    """
    try:
        # Handle various ISO formats
        if iso_string.endswith("Z"):
            iso_string = iso_string[:-1] + "+00:00"

        return datetime.fromisoformat(iso_string)
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse timestamp '{iso_string}': {e}")
        return None


# ---------------------------
# String Utilities
# ---------------------------
def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum allowed length
        suffix: Suffix to add when truncating

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing/replacing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    import re

    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip("_.")

    return sanitized or "unnamed"


# ---------------------------
# Data Validation
# ---------------------------
def validate_user_id(user_id: str) -> bool:
    """
    Validate user ID format.

    Args:
        user_id: User ID to validate

    Returns:
        True if valid, False otherwise
    """
    if not user_id or not isinstance(user_id, str):
        return False

    # Basic validation: non-empty, reasonable length, no special chars
    if len(user_id) < 1 or len(user_id) > 255:
        return False

    # Allow alphanumeric, hyphens, underscores, dots
    import re

    if not re.match(r"^[a-zA-Z0-9._-]+$", user_id):
        return False

    return True


def validate_content(content: str, max_length: int = 50000) -> bool:
    """
    Validate content for storage.

    Args:
        content: Content to validate
        max_length: Maximum allowed length

    Returns:
        True if valid, False otherwise
    """
    if not content or not isinstance(content, str):
        return False

    if len(content) > max_length:
        return False

    return True


# ---------------------------
# Performance Utilities
# ---------------------------
def measure_time(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to measure function execution time.

    Args:
        func: Function to measure

    Returns:
        Decorated function with timing
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            execution_time = time.time() - start_time
            logger.debug(
                f"Function {func.__name__} executed in {execution_time:.3f} seconds"
            )

    return wrapper


# ---------------------------
# Error Handling
# ---------------------------
class MemorizerError(Exception):
    """Base exception for Memorizer framework."""

    pass


class DatabaseError(MemorizerError):
    """Database-related errors."""

    pass


class CompressionError(MemorizerError):
    """Compression-related errors."""

    pass


class RetrievalError(MemorizerError):
    """Retrieval-related errors."""

    pass


class VectorDBError(MemorizerError):
    """Vector database-related errors."""

    pass


# ---------------------------
# Configuration Utilities
# ---------------------------
def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get boolean value from environment variable.

    Args:
        key: Environment variable key
        default: Default value if not set

    Returns:
        Boolean value
    """
    import os

    value = os.getenv(key, "").lower()
    return value in ("true", "1", "yes", "on") if value else default


def get_env_int(key: str, default: int = 0) -> int:
    """
    Get integer value from environment variable.

    Args:
        key: Environment variable key
        default: Default value if not set

    Returns:
        Integer value
    """
    import os

    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return default


# ---------------------------
# Testing Utilities
# ---------------------------
def create_test_memory(
    user_id: str = "test_user", content: str = "Test content"
) -> Dict[str, Any]:
    """
    Create a test memory object for testing purposes.

    Args:
        user_id: Test user ID
        content: Test content

    Returns:
        Test memory dictionary
    """
    return {
        "id": "test_memory_1",
        "user_id": user_id,
        "content": content,
        "metadata": {"source": "test", "created_by": "test_utils"},
        "tier": "very_new",
        "created_at": now_ts(),
    }
