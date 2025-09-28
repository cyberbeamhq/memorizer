"""
Storage components.
"""

try:
    from .external_providers import (
        ExternalDatabaseProvider,
        SupabaseProvider,
        RailwayProvider,
        NeonProvider,
        CockroachDBProvider,
        ExternalDatabaseStorage,
        create_external_provider,
        create_external_storage
    )

    __all__ = [
        "ExternalDatabaseProvider",
        "SupabaseProvider",
        "RailwayProvider",
        "NeonProvider",
        "CockroachDBProvider",
        "ExternalDatabaseStorage",
        "create_external_provider",
        "create_external_storage"
    ]

    # Try to import other components if available
    try:
        from .vector_db import VectorDBProvider
        __all__.append("VectorDBProvider")
    except ImportError:
        pass

except ImportError as e:
    # Basic fallback
    __all__ = []
