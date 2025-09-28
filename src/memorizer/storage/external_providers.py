"""
External Database Providers
Support for third-party database services like Supabase, PlanetScale, Railway, etc.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

try:
    from .postgres_storage import PostgresStorage
except ImportError:
    # Fallback - create a minimal PostgresStorage class
    from ..core.interfaces import Storage, Memory
    import logging

    class PostgresStorage(Storage):
        def __init__(self, connection_string: str = "", **kwargs):
            self.connection_string = connection_string

        def store(self, memory: Memory) -> str:
            return memory.id

        def get(self, memory_id: str, user_id: str) -> Optional[Memory]:
            return None

        def search(self, query, limit: int = 10, offset: int = 0):
            return []

        def update(self, memory: Memory) -> bool:
            return True

        def delete(self, memory_id: str, user_id: str) -> bool:
            return True

        def get_health_status(self):
            return {"status": "fallback", "type": "PostgresStorage"}

logger = logging.getLogger(__name__)


class ExternalDatabaseProvider(ABC):
    """Abstract base class for external database providers."""

    @abstractmethod
    def get_connection_string(self) -> str:
        """Get the connection string for the external provider."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name."""
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate the connection to the external provider."""
        pass

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the external provider."""
        try:
            is_valid = self.validate_connection()
            return {
                "status": "healthy" if is_valid else "unhealthy",
                "provider": self.get_provider_name(),
                "connection_valid": is_valid
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.get_provider_name(),
                "error": str(e)
            }


class SupabaseProvider(ExternalDatabaseProvider):
    """Supabase database provider."""

    def __init__(
        self,
        project_url: str = "",
        anon_key: str = "",
        service_role_key: str = "",
        database_password: str = "",
        **kwargs
    ):
        self.project_url = project_url or os.getenv("SUPABASE_URL", "")
        self.anon_key = anon_key or os.getenv("SUPABASE_ANON_KEY", "")
        self.service_role_key = service_role_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        self.database_password = database_password or os.getenv("SUPABASE_DB_PASSWORD", "")

        if not self.project_url:
            raise ValueError("Supabase project URL is required")

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "Supabase"

    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string for Supabase."""
        if not self.project_url:
            raise ValueError("Supabase project URL not configured")

        # Extract project ID from URL
        project_id = self.project_url.replace("https://", "").replace(".supabase.co", "")

        # Construct PostgreSQL connection string
        connection_string = f"postgresql://postgres:{self.database_password}@db.{project_id}.supabase.co:5432/postgres"

        return connection_string

    def validate_connection(self) -> bool:
        """Validate connection to Supabase."""
        try:
            # Try to create a connection using the connection string
            import psycopg2
            conn_str = self.get_connection_string()
            with psycopg2.connect(conn_str) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return True
        except ImportError:
            logger.warning("psycopg2 not installed, cannot validate Supabase connection")
            return False
        except Exception as e:
            logger.error(f"Supabase connection validation failed: {e}")
            return False

    def get_supabase_client(self):
        """Get Supabase client for additional operations."""
        try:
            import supabase

            key = self.service_role_key or self.anon_key
            if not key:
                raise ValueError("Supabase API key not configured")

            client = supabase.create_client(self.project_url, key)
            return client
        except ImportError:
            logger.warning("supabase package not installed. Install with: pip install supabase")
            return None
        except Exception as e:
            logger.error(f"Failed to create Supabase client: {e}")
            return None


class PlanetScaleProvider(ExternalDatabaseProvider):
    """PlanetScale database provider."""

    def __init__(
        self,
        host: str = "",
        username: str = "",
        password: str = "",
        database: str = "",
        **kwargs
    ):
        self.host = host or os.getenv("PLANETSCALE_HOST", "")
        self.username = username or os.getenv("PLANETSCALE_USERNAME", "")
        self.password = password or os.getenv("PLANETSCALE_PASSWORD", "")
        self.database = database or os.getenv("PLANETSCALE_DATABASE", "")

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "PlanetScale"

    def get_connection_string(self) -> str:
        """Get MySQL connection string for PlanetScale."""
        if not all([self.host, self.username, self.password, self.database]):
            raise ValueError("PlanetScale configuration incomplete")

        # PlanetScale uses MySQL protocol
        connection_string = f"mysql://{self.username}:{self.password}@{self.host}/{self.database}?ssl=true"
        return connection_string

    def validate_connection(self) -> bool:
        """Validate connection to PlanetScale."""
        try:
            import pymysql

            connection = pymysql.connect(
                host=self.host,
                user=self.username,
                password=self.password,
                database=self.database,
                ssl={'ssl_disabled': False}
            )

            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                return True

        except ImportError:
            logger.warning("PyMySQL not installed, cannot validate PlanetScale connection")
            return False
        except Exception as e:
            logger.error(f"PlanetScale connection validation failed: {e}")
            return False


class RailwayProvider(ExternalDatabaseProvider):
    """Railway database provider."""

    def __init__(
        self,
        database_url: str = "",
        database_host: str = "",
        database_port: str = "",
        database_user: str = "",
        database_password: str = "",
        database_name: str = "",
        **kwargs
    ):
        # Railway typically provides a DATABASE_URL
        self.database_url = database_url or os.getenv("DATABASE_URL", "")

        # Individual components as fallback
        self.database_host = database_host or os.getenv("PGHOST", "")
        self.database_port = database_port or os.getenv("PGPORT", "5432")
        self.database_user = database_user or os.getenv("PGUSER", "")
        self.database_password = database_password or os.getenv("PGPASSWORD", "")
        self.database_name = database_name or os.getenv("PGDATABASE", "")

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "Railway"

    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string for Railway."""
        if self.database_url:
            return self.database_url

        if not all([self.database_host, self.database_user, self.database_password, self.database_name]):
            raise ValueError("Railway database configuration incomplete")

        connection_string = f"postgresql://{self.database_user}:{self.database_password}@{self.database_host}:{self.database_port}/{self.database_name}"
        return connection_string

    def validate_connection(self) -> bool:
        """Validate connection to Railway."""
        try:
            import psycopg2
            conn_str = self.get_connection_string()
            with psycopg2.connect(conn_str) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return True
        except ImportError:
            logger.warning("psycopg2 not installed, cannot validate Railway connection")
            return False
        except Exception as e:
            logger.error(f"Railway connection validation failed: {e}")
            return False


class NeonProvider(ExternalDatabaseProvider):
    """Neon database provider."""

    def __init__(
        self,
        connection_string: str = "",
        host: str = "",
        database: str = "",
        username: str = "",
        password: str = "",
        **kwargs
    ):
        self.connection_string = connection_string or os.getenv("NEON_DATABASE_URL", "")
        self.host = host or os.getenv("NEON_HOST", "")
        self.database = database or os.getenv("NEON_DATABASE", "")
        self.username = username or os.getenv("NEON_USERNAME", "")
        self.password = password or os.getenv("NEON_PASSWORD", "")

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "Neon"

    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string for Neon."""
        if self.connection_string:
            return self.connection_string

        if not all([self.host, self.username, self.password, self.database]):
            raise ValueError("Neon database configuration incomplete")

        connection_string = f"postgresql://{self.username}:{self.password}@{self.host}/{self.database}?sslmode=require"
        return connection_string

    def validate_connection(self) -> bool:
        """Validate connection to Neon."""
        try:
            import psycopg2
            conn_str = self.get_connection_string()
            with psycopg2.connect(conn_str) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return True
        except ImportError:
            logger.warning("psycopg2 not installed, cannot validate Neon connection")
            return False
        except Exception as e:
            logger.error(f"Neon connection validation failed: {e}")
            return False


class CockroachDBProvider(ExternalDatabaseProvider):
    """CockroachDB Cloud provider."""

    def __init__(
        self,
        connection_string: str = "",
        host: str = "",
        port: str = "",
        database: str = "",
        username: str = "",
        password: str = "",
        cluster_id: str = "",
        **kwargs
    ):
        self.connection_string = connection_string or os.getenv("COCKROACH_DATABASE_URL", "")
        self.host = host or os.getenv("COCKROACH_HOST", "")
        self.port = port or os.getenv("COCKROACH_PORT", "26257")
        self.database = database or os.getenv("COCKROACH_DATABASE", "")
        self.username = username or os.getenv("COCKROACH_USERNAME", "")
        self.password = password or os.getenv("COCKROACH_PASSWORD", "")
        self.cluster_id = cluster_id or os.getenv("COCKROACH_CLUSTER_ID", "")

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "CockroachDB"

    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string for CockroachDB."""
        if self.connection_string:
            return self.connection_string

        if not all([self.host, self.username, self.password, self.database]):
            raise ValueError("CockroachDB configuration incomplete")

        connection_string = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode=require"
        return connection_string

    def validate_connection(self) -> bool:
        """Validate connection to CockroachDB."""
        try:
            import psycopg2
            conn_str = self.get_connection_string()
            with psycopg2.connect(conn_str) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return True
        except ImportError:
            logger.warning("psycopg2 not installed, cannot validate CockroachDB connection")
            return False
        except Exception as e:
            logger.error(f"CockroachDB connection validation failed: {e}")
            return False


class ExternalDatabaseStorage(PostgresStorage):
    """
    External database storage that wraps PostgresStorage
    with external provider connection management.
    """

    def __init__(self, provider: ExternalDatabaseProvider, **kwargs):
        self.provider = provider

        # Get connection string from provider
        connection_string = provider.get_connection_string()

        # Initialize parent PostgresStorage with the connection string
        super().__init__(connection_string=connection_string, **kwargs)

        logger.info(f"External database storage initialized with {provider.get_provider_name()}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status including provider information."""
        base_status = super().get_health_status()
        provider_status = self.provider.get_health_status()

        return {
            **base_status,
            "provider": provider_status,
            "type": f"ExternalDatabaseStorage({self.provider.get_provider_name()})"
        }


# Provider factory
def create_external_provider(provider_type: str, **config) -> ExternalDatabaseProvider:
    """Create an external database provider."""
    providers = {
        "supabase": SupabaseProvider,
        "planetscale": PlanetScaleProvider,
        "railway": RailwayProvider,
        "neon": NeonProvider,
        "cockroachdb": CockroachDBProvider,
    }

    if provider_type.lower() not in providers:
        raise ValueError(f"Unknown provider type: {provider_type}. Available: {list(providers.keys())}")

    provider_class = providers[provider_type.lower()]
    return provider_class(**config)


def create_external_storage(provider_type: str, **config) -> ExternalDatabaseStorage:
    """Create external database storage with the specified provider."""
    provider = create_external_provider(provider_type, **config)
    return ExternalDatabaseStorage(provider, **config)


__all__ = [
    "ExternalDatabaseProvider",
    "SupabaseProvider",
    "PlanetScaleProvider",
    "RailwayProvider",
    "NeonProvider",
    "CockroachDBProvider",
    "ExternalDatabaseStorage",
    "create_external_provider",
    "create_external_storage",
]