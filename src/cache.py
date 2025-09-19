"""
cache.py
Redis caching layer for the Memorizer framework.
Provides high-performance caching for frequently accessed data.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import redis
from redis.exceptions import ConnectionError, RedisError, TimeoutError

logger = logging.getLogger(__name__)


class CacheConfig:
    """Cache configuration."""

    def __init__(self):
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", "6379"))
        self.db = int(os.getenv("REDIS_DB", "0"))
        self.password = os.getenv("REDIS_PASSWORD")
        self.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
        self.socket_timeout = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
        self.socket_connect_timeout = int(os.getenv("REDIS_CONNECT_TIMEOUT", "5"))
        self.retry_on_timeout = True
        self.health_check_interval = 30


class CacheManager:
    """Redis cache manager with connection pooling and error handling."""

    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self._pool = None
        self._client = None
        self._is_connected = False
        self._last_health_check = None

        # Cache key prefixes
        self.prefixes = {
            "memory": "mem:",
            "user_stats": "stats:",
            "embedding": "emb:",
            "query_result": "query:",
            "session": "session:",
            "lock": "lock:",
        }

        # Default TTL values (in seconds)
        self.default_ttl = {
            "memory": 3600,  # 1 hour
            "user_stats": 1800,  # 30 minutes
            "embedding": 7200,  # 2 hours
            "query_result": 900,  # 15 minutes
            "session": 1800,  # 30 minutes
            "lock": 60,  # 1 minute
        }

        self._initialize_connection()

    def _initialize_connection(self):
        """Initialize Redis connection pool."""
        try:
            # Create connection pool
            self._pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval,
            )

            # Create Redis client
            self._client = redis.Redis(connection_pool=self._pool)

            # Test connection
            self._test_connection()
            self._is_connected = True
            logger.info("Redis cache connection established")

        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self._is_connected = False
            # Don't raise - allow application to continue without cache

    def _test_connection(self):
        """Test Redis connection."""
        try:
            self._client.ping()
            self._last_health_check = datetime.now()
        except Exception as e:
            raise ConnectionError(f"Redis connection test failed: {e}")

    def _is_healthy(self) -> bool:
        """Check if cache is healthy."""
        if not self._is_connected or not self._client:
            return False

        # Check if we need to do a health check
        if (
            self._last_health_check is None
            or datetime.now() - self._last_health_check > timedelta(seconds=30)
        ):
            try:
                self._test_connection()
                return True
            except:
                self._is_connected = False
                return False

        return True

    def _get_key(self, prefix: str, key: str) -> str:
        """Generate cache key with prefix."""
        return f"{self.prefixes[prefix]}{key}"

    def _serialize(self, data: Any) -> str:
        """Serialize data for caching."""
        try:
            return json.dumps(data, default=str)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize data: {e}")
            raise

    def _deserialize(self, data: str) -> Any:
        """Deserialize cached data."""
        try:
            return json.loads(data)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to deserialize data: {e}")
            raise

    def get(self, prefix: str, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._is_healthy():
            return None

        try:
            cache_key = self._get_key(prefix, key)
            data = self._client.get(cache_key)

            if data is None:
                return None

            return self._deserialize(data.decode("utf-8"))

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected cache get error for key {key}: {e}")
            return None

    def set(self, prefix: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self._is_healthy():
            return False

        try:
            cache_key = self._get_key(prefix, key)
            serialized_value = self._serialize(value)

            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl.get(prefix, 3600)

            result = self._client.setex(cache_key, ttl, serialized_value)
            return bool(result)

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected cache set error for key {key}: {e}")
            return False

    def delete(self, prefix: str, key: str) -> bool:
        """Delete value from cache."""
        if not self._is_healthy():
            return False

        try:
            cache_key = self._get_key(prefix, key)
            result = self._client.delete(cache_key)
            return bool(result)

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected cache delete error for key {key}: {e}")
            return False

    def exists(self, prefix: str, key: str) -> bool:
        """Check if key exists in cache."""
        if not self._is_healthy():
            return False

        try:
            cache_key = self._get_key(prefix, key)
            return bool(self._client.exists(cache_key))

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Cache exists error for key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected cache exists error for key {key}: {e}")
            return False

    def get_many(self, prefix: str, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        if not self._is_healthy() or not keys:
            return {}

        try:
            cache_keys = [self._get_key(prefix, key) for key in keys]
            values = self._client.mget(cache_keys)

            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = self._deserialize(value.decode("utf-8"))
                    except Exception as e:
                        logger.warning(
                            f"Failed to deserialize cached value for key {key}: {e}"
                        )

            return result

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Cache get_many error: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected cache get_many error: {e}")
            return {}

    def set_many(
        self, prefix: str, data: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Set multiple values in cache."""
        if not self._is_healthy() or not data:
            return False

        try:
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl.get(prefix, 3600)

            # Prepare data for Redis
            redis_data = {}
            for key, value in data.items():
                cache_key = self._get_key(prefix, key)
                redis_data[cache_key] = self._serialize(value)

            # Use pipeline for better performance
            pipe = self._client.pipeline()
            for cache_key, serialized_value in redis_data.items():
                pipe.setex(cache_key, ttl, serialized_value)

            results = pipe.execute()
            return all(results)

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Cache set_many error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected cache set_many error: {e}")
            return False

    def delete_many(self, prefix: str, keys: List[str]) -> int:
        """Delete multiple values from cache."""
        if not self._is_healthy() or not keys:
            return 0

        try:
            cache_keys = [self._get_key(prefix, key) for key in keys]
            return self._client.delete(*cache_keys)

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Cache delete_many error: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected cache delete_many error: {e}")
            return 0

    def clear_prefix(self, prefix: str) -> int:
        """Clear all keys with a specific prefix."""
        if not self._is_healthy():
            return 0

        try:
            pattern = f"{self.prefixes[prefix]}*"
            keys = self._client.keys(pattern)

            if keys:
                return self._client.delete(*keys)
            return 0

        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Cache clear_prefix error for prefix {prefix}: {e}")
            return 0
        except Exception as e:
            logger.error(
                f"Unexpected cache clear_prefix error for prefix {prefix}: {e}"
            )
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._is_healthy():
            return {"status": "disconnected"}

        try:
            info = self._client.info()
            return {
                "status": "connected",
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0),
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"status": "error", "error": str(e)}

    def close(self):
        """Close cache connection."""
        try:
            if self._client:
                self._client.close()
            if self._pool:
                self._pool.disconnect()
            self._is_connected = False
            logger.info("Redis cache connection closed")
        except Exception as e:
            logger.error(f"Error closing cache connection: {e}")


# Global cache manager instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def initialize_cache():
    """Initialize global cache manager."""
    global _cache_manager
    _cache_manager = CacheManager()
    logger.info("Cache manager initialized")


# Cache decorators and utilities
def cache_result(
    prefix: str, ttl: Optional[int] = None, key_func: Optional[callable] = None
):
    """Decorator to cache function results."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_cache_manager()

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()

            # Try to get from cache
            cached_result = cache.get(prefix, cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(prefix, cache_key, result, ttl)

            return result

        return wrapper

    return decorator


def invalidate_cache(prefix: str, key: str):
    """Invalidate specific cache entry."""
    cache = get_cache_manager()
    cache.delete(prefix, key)


def invalidate_user_cache(user_id: str):
    """Invalidate all cache entries for a user."""
    cache = get_cache_manager()
    cache.clear_prefix("memory")
    cache.clear_prefix("user_stats")
    cache.clear_prefix("query_result")
    cache.clear_prefix("session")
