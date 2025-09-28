"""
Memorizer - Intelligent Memory Management Library

A focused library for adding intelligent memory capabilities to AI applications.
Provides memory storage, retrieval, lifecycle management, and external provider integrations.
"""

__version__ = "1.0.0"

# Core memory management imports
try:
    # Import what's actually available
    from .core.simple_config import MemoryConfig, create_default_config
    from .storage.external_providers import (
        create_external_provider,
        create_external_storage,
        SupabaseProvider,
        RailwayProvider,
        NeonProvider,
    )
    from .builtins.vector_stores import (
        PineconeVectorStore,
        WeaviateVectorStore,
        MemoryVectorStore,
        SQLiteVectorStore,
    )

    # Try to import core components
    try:
        from .memory.simple_manager import MemoryManager, Memory, MemoryStats, RetrievalResult
        # Define MemoryTier since it doesn't exist in interfaces
        class MemoryTier:
            VERY_NEW = "very_new"
            MID_TERM = "mid_term"
            LONG_TERM = "long_term"
    except ImportError as e2:
        # Try alternative import for framework memory manager
        try:
            from .memory.memory_manager import FrameworkMemoryManager as MemoryManager
            from .core.interfaces import Memory
            class MemoryTier:
                VERY_NEW = "very_new"
                MID_TERM = "mid_term"
                LONG_TERM = "long_term"
        except ImportError as e3:
            # Create mock classes for demo
            import logging
            logging.warning(f"Core components not available, using mocks: {e3}")

            class MemoryManager:
                def __init__(self, config):
                    self.config = config
                    self._memories = {}

                def store_memory(self, user_id, content, metadata=None):
                    memory_id = f"mem_{len(self._memories)}"
                    self._memories[memory_id] = {
                        "user_id": user_id,
                        "content": content,
                        "metadata": metadata or {}
                    }
                    return memory_id

                def search_memories(self, user_id, query, limit=10):
                    # Simple mock search
                    class MockResult:
                        def __init__(self):
                            self.memories = []
                            self.total_found = 0

                    return MockResult()

            class Memory:
                pass

            class MemoryTier:
                pass

except ImportError as e:
    # Graceful fallback if some components aren't available
    import logging
    logging.warning(f"Some memorizer components not available: {e}")

    # Define minimal fallbacks
    MemoryConfig = None
    create_default_config = None

# Simple factory function for easy setup
def create_memory_manager(
    storage_provider="memory",
    vector_store="memory",
    llm_provider="mock",
    **config
):
    """
    Create a memory manager instance with simple configuration.

    Args:
        storage_provider: Database provider ("memory", "supabase", "railway", "neon")
        vector_store: Vector store provider ("memory", "pinecone", "weaviate", "sqlite")
        llm_provider: LLM provider for summarization ("openai", "anthropic", "mock")
        **config: Additional configuration parameters

    Returns:
        MemoryManager: Configured memory manager instance

    Example:
        # Simple setup with defaults
        memory = create_memory_manager()

        # With external providers
        memory = create_memory_manager(
            storage_provider="supabase",
            vector_store="pinecone",
            llm_provider="openai",
            supabase_url="https://your-project.supabase.co",
            supabase_password="your-password",
            pinecone_api_key="your-key"
        )
    """
    try:
        from .core.simple_config import MemoryConfig

        # Create configuration
        memory_config = MemoryConfig(
            storage_provider=storage_provider,
            vector_store=vector_store,
            llm_provider=llm_provider,
            **config
        )

        return MemoryManager(memory_config)
    except ImportError as e:
        # Fallback to the original demo behavior
        print(f"Note: Using demo mode due to import issues: {e}")
        return None


def create_memory():
    """Create an in-memory manager for testing/development."""
    return create_memory_manager(
        storage_provider="memory",
        vector_store="memory",
        llm_provider="mock"
    )


# Export main classes
__all__ = [
    # Main API
    "create_memory_manager",
    "create_memory",

    # Core classes (if available)
    "MemoryManager",
    "Memory",
    "MemoryTier",

    # External providers (if available)
    "create_external_provider",
    "create_external_storage",
    "SupabaseProvider",
    "RailwayProvider",
    "NeonProvider",

    # Vector stores (if available)
    "PineconeVectorStore",
    "WeaviateVectorStore",
    "MemoryVectorStore",
    "SQLiteVectorStore",
]
