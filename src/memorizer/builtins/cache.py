"""
Built-in Cache Components
Provides default caching implementations for the Memorizer framework.
"""

import json
import logging
import threading
import time
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseCache(ABC):
    """Abstract base class for cache providers."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries."""
        pass

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the cache provider."""
        return {"status": "healthy", "type": self.__class__.__name__}


class MemoryCacheProvider(BaseCache):
    """In-memory cache provider."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600, **kwargs):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check if expired
            if entry["expires_at"] and time.time() > entry["expires_at"]:
                del self._cache[key]
                return None

            # Update access time for LRU
            entry["accessed_at"] = time.time()
            return entry["value"]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        with self._lock:
            try:
                # Evict if at max size and key doesn't exist
                if len(self._cache) >= self.max_size and key not in self._cache:
                    self._evict_lru()

                ttl_seconds = ttl if ttl is not None else self.default_ttl
                expires_at = time.time() + ttl_seconds if ttl_seconds > 0 else None

                self._cache[key] = {
                    "value": value,
                    "created_at": time.time(),
                    "accessed_at": time.time(),
                    "expires_at": expires_at
                }

                return True

            except Exception as e:
                logger.error(f"Failed to set cache key {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> bool:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            return True

    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k]["accessed_at"]
        )
        del self._cache[lru_key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_entries = len(self._cache)
            expired_count = 0
            current_time = time.time()

            for entry in self._cache.values():
                if entry["expires_at"] and current_time > entry["expires_at"]:
                    expired_count += 1

            return {
                "total_entries": total_entries,
                "expired_entries": expired_count,
                "max_size": self.max_size,
                "utilization": total_entries / self.max_size if self.max_size > 0 else 0
            }


class FileCacheProvider(BaseCache):
    """File-based cache provider."""

    def __init__(self, cache_dir: str = "cache", default_ttl: int = 3600, **kwargs):
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.cache_dir.mkdir(exist_ok=True)
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        with self._lock:
            try:
                cache_file = self.cache_dir / f"{self._safe_key(key)}.json"

                if not cache_file.exists():
                    return None

                with open(cache_file, 'r') as f:
                    entry = json.load(f)

                # Check if expired
                if entry["expires_at"] and time.time() > entry["expires_at"]:
                    cache_file.unlink(missing_ok=True)
                    return None

                return entry["value"]

            except Exception as e:
                logger.error(f"Failed to get cache key {key}: {e}")
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in file cache."""
        with self._lock:
            try:
                ttl_seconds = ttl if ttl is not None else self.default_ttl
                expires_at = time.time() + ttl_seconds if ttl_seconds > 0 else None

                entry = {
                    "value": value,
                    "created_at": time.time(),
                    "expires_at": expires_at
                }

                cache_file = self.cache_dir / f"{self._safe_key(key)}.json"

                with open(cache_file, 'w') as f:
                    json.dump(entry, f)

                return True

            except Exception as e:
                logger.error(f"Failed to set cache key {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        """Delete value from file cache."""
        with self._lock:
            try:
                cache_file = self.cache_dir / f"{self._safe_key(key)}.json"
                if cache_file.exists():
                    cache_file.unlink()
                    return True
                return False

            except Exception as e:
                logger.error(f"Failed to delete cache key {key}: {e}")
                return False

    def clear(self) -> bool:
        """Clear all cache entries."""
        with self._lock:
            try:
                for cache_file in self.cache_dir.glob("*.json"):
                    cache_file.unlink()
                return True

            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
                return False

    def _safe_key(self, key: str) -> str:
        """Convert key to safe filename."""
        import hashlib
        return hashlib.md5(key.encode()).hexdigest()


class RedisCacheProvider(BaseCache):
    """Redis cache provider (mock implementation)."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, **kwargs):
        self.host = host
        self.port = port
        self.db = db
        logger.warning("Redis cache not fully implemented, using memory fallback")
        self._fallback = MemoryCacheProvider(**kwargs)

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis (fallback to memory)."""
        return self._fallback.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis (fallback to memory)."""
        return self._fallback.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        """Delete value from Redis (fallback to memory)."""
        return self._fallback.delete(key)

    def clear(self) -> bool:
        """Clear all cache entries from Redis (fallback to memory)."""
        return self._fallback.clear()


__all__ = [
    "BaseCache",
    "MemoryCacheProvider",
    "FileCacheProvider",
    "RedisCacheProvider",
]