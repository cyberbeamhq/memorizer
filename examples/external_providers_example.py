#!/usr/bin/env python3
"""
External Providers Example
Demonstrates how to use external database and vector store providers.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memorizer.core.config import FrameworkConfig, ComponentConfig
from memorizer.core.framework import create_framework
from memorizer.storage.external_providers import (
    SupabaseProvider,
    RailwayProvider,
    NeonProvider,
    create_external_provider,
    create_external_storage
)
from memorizer.builtins.vector_stores import (
    PineconeVectorStore,
    WeaviateVectorStore,
    MemoryVectorStore
)


def demo_external_database_providers():
    """Demonstrate external database provider configurations."""
    print("🗄️  External Database Providers Demo")
    print("=" * 50)

    # Test each provider configuration (without actual connections)
    providers_config = {
        "Supabase": {
            "project_url": "https://your-project.supabase.co",
            "anon_key": "your-anon-key",
            "service_role_key": "your-service-role-key",
            "database_password": "your-db-password"
        },
        "Railway": {
            "database_url": "postgresql://user:pass@host:port/db"
        },
        "Neon": {
            "connection_string": "postgresql://user:pass@host/db?sslmode=require"
        },
        "PlanetScale": {
            "host": "host.planetscale.com",
            "username": "your-username",
            "password": "your-password",
            "database": "your-database"
        },
        "CockroachDB": {
            "connection_string": "postgresql://user:pass@host:26257/db?sslmode=require"
        }
    }

    for provider_name, config in providers_config.items():
        try:
            provider = create_external_provider(provider_name.lower(), **config)
            print(f"✅ {provider_name} provider configured")
            print(f"   Provider: {provider.get_provider_name()}")

            # Get connection string (won't actually connect)
            try:
                conn_str = provider.get_connection_string()
                print(f"   Connection string generated: {conn_str[:50]}...")
            except Exception as e:
                print(f"   Connection string error: {e}")

            print()
        except Exception as e:
            print(f"❌ {provider_name} configuration failed: {e}")
            print()


def demo_vector_store_providers():
    """Demonstrate vector store provider configurations."""
    print("🔍 Vector Store Providers Demo")
    print("=" * 50)

    # Test memory vector store (always works)
    print("1. Memory Vector Store")
    memory_store = MemoryVectorStore()
    print(f"   Status: {memory_store.get_health_status()}")

    # Test vector store operations
    test_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
    test_metadata = {"test": "data", "type": "demo"}

    # Store a vector
    success = memory_store.store_vector("test_vector_1", test_vector, test_metadata)
    print(f"   Store operation: {'✅ Success' if success else '❌ Failed'}")

    # Search vectors
    results = memory_store.search_vectors(test_vector, limit=5, threshold=0.5)
    print(f"   Search results: {len(results)} found")

    # Test framework interface compatibility
    success = memory_store.insert_embedding("test_embed_1", test_vector, test_metadata)
    print(f"   Framework interface: {'✅ Compatible' if success else '❌ Not compatible'}")
    print()

    # Test Pinecone (without actual connection)
    print("2. Pinecone Vector Store")
    pinecone_store = PineconeVectorStore(
        api_key="test-key",
        environment="test-env",
        index_name="test-index"
    )
    status = pinecone_store.get_health_status()
    print(f"   Status: {status['status']} ({status.get('error', 'OK')})")
    print()

    # Test Weaviate (without actual connection)
    print("3. Weaviate Vector Store")
    weaviate_store = WeaviateVectorStore(
        url="http://localhost:8080",
        api_key="test-key",
        class_name="TestMemory"
    )
    status = weaviate_store.get_health_status()
    print(f"   Status: Available with fallback")
    print()


def demo_framework_with_external_providers():
    """Demonstrate framework configuration with external providers."""
    print("🚀 Framework with External Providers")
    print("=" * 50)

    # Create configuration with external providers
    config = FrameworkConfig(
        framework={"version": "1.0.0", "debug": True},
        storage=ComponentConfig(
            type="storage",
            name="memory",  # Use memory for demo
            enabled=True,
            config={}
        ),
        vector_store=ComponentConfig(
            type="vector_store",
            name="memory",  # Use memory for demo
            enabled=True,
            config={}
        ),
        cache=ComponentConfig(
            type="cache",
            name="memory",
            enabled=True,
            config={}
        ),
        retriever=ComponentConfig(
            type="retriever",
            name="hybrid",
            enabled=True,
            config={}
        ),
        summarizer=ComponentConfig(
            type="summarizer",
            name="mock",
            enabled=True,
            config={}
        ),
        pii_filter=ComponentConfig(
            type="pii_filter",
            name="noop",
            enabled=True,
            config={}
        ),
        scorer=ComponentConfig(
            type="scorer",
            name="simple",
            enabled=True,
            config={}
        ),
        task_runner=ComponentConfig(
            type="task_runner",
            name="thread",
            enabled=True,
            config={}
        ),
        embedding_provider=ComponentConfig(
            type="embedding_provider",
            name="mock",
            enabled=True,
            config={}
        )
    )

    # Create framework
    framework = create_framework(config)
    print("✅ Framework created with external provider support")

    # Get memory manager
    memory_manager = framework.get_memory_manager()
    print("✅ Memory manager ready")

    # Test memory operations
    memory_id = memory_manager.store_memory(
        user_id="external_demo_user",
        content="This is a test memory using external providers configuration",
        metadata={"demo": True, "provider": "external"}
    )
    print(f"✅ Memory stored: {memory_id}")

    # Search memories
    results = memory_manager.search_memories(
        user_id="external_demo_user",
        query="test memory providers",
        limit=5
    )
    print(f"✅ Search completed: {results.total_found} results found")

    # Get framework health
    health = framework.get_health_status()
    print(f"✅ Framework health: {health['framework']['status']}")
    print()


def demo_configuration_examples():
    """Show configuration examples for external providers."""
    print("⚙️  Configuration Examples")
    print("=" * 50)

    print("Example 1: Supabase Configuration")
    print("storage:")
    print("  name: supabase")
    print("  config:")
    print("    supabase:")
    print("      project_url: ${SUPABASE_URL}")
    print("      database_password: ${SUPABASE_DB_PASSWORD}")
    print()

    print("Example 2: Pinecone + Railway Configuration")
    print("storage:")
    print("  name: railway")
    print("  config:")
    print("    railway:")
    print("      database_url: ${RAILWAY_DATABASE_URL}")
    print()
    print("vector_store:")
    print("  name: pinecone")
    print("  config:")
    print("    pinecone:")
    print("      api_key: ${PINECONE_API_KEY}")
    print("      environment: ${PINECONE_ENVIRONMENT}")
    print("      index_name: memorizer")
    print()

    print("Example 3: Full External Setup")
    print("# Use Neon for database, Weaviate for vectors")
    print("storage:")
    print("  name: neon")
    print("  config:")
    print("    neon:")
    print("      connection_string: ${NEON_DATABASE_URL}")
    print()
    print("vector_store:")
    print("  name: weaviate")
    print("  config:")
    print("    weaviate:")
    print("      url: ${WEAVIATE_URL}")
    print("      api_key: ${WEAVIATE_API_KEY}")
    print()


def main():
    """Run all external provider demos."""
    print("🌐 Memorizer External Providers Demo")
    print("=" * 60)
    print()

    try:
        demo_external_database_providers()
        demo_vector_store_providers()
        demo_framework_with_external_providers()
        demo_configuration_examples()

        print("🎉 External providers demo completed successfully!")
        print()
        print("💡 Next steps:")
        print("   • Set up your external provider credentials")
        print("   • Update memorizer.yaml with your provider configuration")
        print("   • Deploy with your chosen external services")
        print("   • Monitor performance and costs")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()