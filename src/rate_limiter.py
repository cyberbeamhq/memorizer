"""
rate_limiter.py
Rate limiting and request throttling for the Memorizer framework.
Provides protection against abuse and ensures fair resource usage.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10  # Max requests in a short burst
    burst_window: int = 10  # Burst window in seconds
    enable_user_limits: bool = True
    enable_global_limits: bool = True


class RateLimiter:
    """Rate limiter with sliding window and token bucket algorithms."""

    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self.lock = threading.RLock()

        # User-specific rate limiting
        self.user_requests = defaultdict(
            lambda: {
                "minute": deque(),
                "hour": deque(),
                "day": deque(),
                "burst": deque(),
            }
        )

        # Global rate limiting
        self.global_requests = {
            "minute": deque(),
            "hour": deque(),
            "day": deque(),
            "burst": deque(),
        }

        # Cleanup tracking
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes

    def _cleanup_old_entries(self):
        """Clean up old entries to prevent memory leaks."""
        current_time = time.time()

        # Only cleanup every 5 minutes
        if current_time - self.last_cleanup < self.cleanup_interval:
            return

        self.last_cleanup = current_time

        # Cleanup user entries
        users_to_remove = []
        for user_id, requests in self.user_requests.items():
            # Clean up old entries
            for window in ["minute", "hour", "day", "burst"]:
                cutoff_time = current_time - self._get_window_seconds(window)
                while requests[window] and requests[window][0] < cutoff_time:
                    requests[window].popleft()

            # Remove users with no recent activity
            if not any(
                requests[window] for window in ["minute", "hour", "day", "burst"]
            ):
                users_to_remove.append(user_id)

        for user_id in users_to_remove:
            del self.user_requests[user_id]

        # Cleanup global entries
        for window in ["minute", "hour", "day", "burst"]:
            cutoff_time = current_time - self._get_window_seconds(window)
            while (
                self.global_requests[window]
                and self.global_requests[window][0] < cutoff_time
            ):
                self.global_requests[window].popleft()

        logger.debug(
            f"Rate limiter cleanup completed. Active users: {len(self.user_requests)}"
        )

    def _get_window_seconds(self, window: str) -> int:
        """Get window size in seconds."""
        window_sizes = {
            "minute": 60,
            "hour": 3600,
            "day": 86400,
            "burst": self.config.burst_window,
        }
        return window_sizes[window]

    def _get_limit(self, window: str) -> int:
        """Get rate limit for window."""
        limits = {
            "minute": self.config.requests_per_minute,
            "hour": self.config.requests_per_hour,
            "day": self.config.requests_per_day,
            "burst": self.config.burst_limit,
        }
        return limits[window]

    def _check_rate_limit(
        self, requests: Dict[str, deque], user_id: str = None
    ) -> Tuple[bool, str, int]:
        """Check if request is within rate limits."""
        current_time = time.time()

        # Check each window
        for window in ["burst", "minute", "hour", "day"]:
            window_seconds = self._get_window_seconds(window)
            limit = self._get_limit(window)

            # Clean old entries for this window
            cutoff_time = current_time - window_seconds
            while requests[window] and requests[window][0] < cutoff_time:
                requests[window].popleft()

            # Check if limit exceeded
            if len(requests[window]) >= limit:
                remaining_time = int(
                    requests[window][0] + window_seconds - current_time
                )
                return False, f"Rate limit exceeded for {window} window", remaining_time

        return True, "", 0

    def is_allowed(self, user_id: str = None) -> Tuple[bool, str, int]:
        """
        Check if request is allowed.

        Returns:
            Tuple of (is_allowed, reason, retry_after_seconds)
        """
        with self.lock:
            # Cleanup old entries periodically
            self._cleanup_old_entries()

            current_time = time.time()

            # Check global rate limits first
            if self.config.enable_global_limits:
                is_allowed, reason, retry_after = self._check_rate_limit(
                    self.global_requests, "global"
                )
                if not is_allowed:
                    return False, f"Global {reason}", retry_after

            # Check user-specific rate limits
            if self.config.enable_user_limits and user_id:
                user_requests = self.user_requests[user_id]
                is_allowed, reason, retry_after = self._check_rate_limit(
                    user_requests, user_id
                )
                if not is_allowed:
                    return False, f"User {reason}", retry_after

            # Record the request
            if self.config.enable_global_limits:
                for window in ["minute", "hour", "day", "burst"]:
                    self.global_requests[window].append(current_time)

            if self.config.enable_user_limits and user_id:
                for window in ["minute", "hour", "day", "burst"]:
                    self.user_requests[user_id][window].append(current_time)

            return True, "", 0

    def get_user_stats(self, user_id: str) -> Dict[str, int]:
        """Get rate limiting statistics for a user."""
        with self.lock:
            if user_id not in self.user_requests:
                return {window: 0 for window in ["minute", "hour", "day", "burst"]}

            current_time = time.time()
            user_requests = self.user_requests[user_id]
            stats = {}

            for window in ["minute", "hour", "day", "burst"]:
                window_seconds = self._get_window_seconds(window)
                cutoff_time = current_time - window_seconds

                # Count recent requests
                count = sum(
                    1 for req_time in user_requests[window] if req_time > cutoff_time
                )
                stats[window] = count

            return stats

    def get_global_stats(self) -> Dict[str, int]:
        """Get global rate limiting statistics."""
        with self.lock:
            current_time = time.time()
            stats = {}

            for window in ["minute", "hour", "day", "burst"]:
                window_seconds = self._get_window_seconds(window)
                cutoff_time = current_time - window_seconds

                # Count recent requests
                count = sum(
                    1
                    for req_time in self.global_requests[window]
                    if req_time > cutoff_time
                )
                stats[window] = count

            return stats

    def reset_user_limits(self, user_id: str):
        """Reset rate limits for a specific user."""
        with self.lock:
            if user_id in self.user_requests:
                for window in ["minute", "hour", "day", "burst"]:
                    self.user_requests[user_id][window].clear()
                logger.info(f"Rate limits reset for user: {user_id}")

    def reset_global_limits(self):
        """Reset global rate limits."""
        with self.lock:
            for window in ["minute", "hour", "day", "burst"]:
                self.global_requests[window].clear()
            logger.info("Global rate limits reset")


class TokenBucket:
    """Token bucket rate limiter for more sophisticated rate limiting."""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket."""
        with self.lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def _refill(self):
        """Refill tokens based on elapsed time."""
        current_time = time.time()
        elapsed = current_time - self.last_refill

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = current_time

    def get_tokens(self) -> float:
        """Get current number of tokens."""
        with self.lock:
            self._refill()
            return self.tokens


# Global rate limiter instance
_rate_limiter = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def initialize_rate_limiter(config: RateLimitConfig = None):
    """Initialize global rate limiter."""
    global _rate_limiter
    _rate_limiter = RateLimiter(config)
    logger.info("Rate limiter initialized")


def check_rate_limit(user_id: str = None) -> Tuple[bool, str, int]:
    """Check if request is within rate limits."""
    limiter = get_rate_limiter()
    return limiter.is_allowed(user_id)


def get_rate_limit_stats(user_id: str = None) -> Dict[str, int]:
    """Get rate limiting statistics."""
    limiter = get_rate_limiter()
    if user_id:
        return limiter.get_user_stats(user_id)
    else:
        return limiter.get_global_stats()
