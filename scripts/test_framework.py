#!/usr/bin/env python3
"""
Test Framework Script
Simple script to test the Memorizer Framework components.
"""

import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from framework.factory import create_framework
from framework.core.config import FrameworkConfig
from framework.core.interfaces import Memory, Query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_framework():
    """Test the framework with minimal configuration."""
    print("🧪 Testing Memorizer Framework")
    print("=" * 40)
    
    # Create minimal configuration
    config = FrameworkConfig.from_dict({
        "storage": {
            "type": "sqlite",
            "name": "sqlite",
            "config": {"db_path": ":memory:"}
        },
        "cache": {
            "type": "memory",
            "name": "memory",
            "config": {"max_size": 100}
        },
        "summarizer": {
            "type": "mock",
            "name": "mock",
            "config": {}
        },
        "retriever": {
            "type": "keyword",
            "name": "keyword",
            "config": {}
        },
        "scorer": {
            "type": "tfidf",
            "name": "tfidf",
            "config": {}
        },
        "task_runner": {
            "type": "thread",
            "name": "thread",
            "config": {"max_workers": 2}
        },
        "embedding_provider": {
            "type": "openai",
            "name": "openai",
            "config": {"api_key": "test-key", "model": "text-embedding-3-small"}
        },
        "vector_store": {
            "type": "weaviate",
            "name": "weaviate",
            "config": {}
        },
        "pii_filter": {
            "type": "basic",
            "name": "basic",
            "config": {}
        }
    })
    
    try:
        # Create framework
        print("1. Creating framework...")
        print(f"   Config: {config.summarizer}")
        
        # Check registry before creating framework
        from framework.core.registry import get_registry
        registry = get_registry()
        print(f"   Available summarizers: {registry.list_available('summarizer')}")
        
        framework = create_framework(config)
        print("   ✅ Framework created successfully!")
        
        # Test storage
        print("\n2. Testing storage...")
        memory = Memory(
            id="test_memory_1",
            user_id="test_user",
            content="This is a test memory for the framework.",
            metadata={"test": True},
            tier="very_new",
            created_at=datetime.now()
        )
        
        memory_id = framework.storage.store(memory)
        print(f"   ✅ Stored memory: {memory_id}")
        
        retrieved = framework.storage.get(memory_id, "test_user")
        if retrieved:
            print(f"   ✅ Retrieved memory: {retrieved.content[:50]}...")
        else:
            print("   ❌ Failed to retrieve memory")
        
        # Test cache
        print("\n3. Testing cache...")
        framework.cache.set("test_key", {"message": "Hello, World!"}, ttl=60)
        cached_value = framework.cache.get("test_key")
        if cached_value:
            print(f"   ✅ Cache working: {cached_value}")
        else:
            print("   ❌ Cache failed")
        
        # Test PII filter
        print("\n4. Testing PII filter...")
        pii_content = "My email is test@example.com and my phone is 555-123-4567."
        sanitized, pii_data = framework.pii_filter.filter(pii_content)
        print(f"   Original: {pii_content}")
        print(f"   Sanitized: {sanitized}")
        print(f"   Detected PII: {len(pii_data.get('detected_pii', []))} items")
        
        # Test summarizer
        print("\n5. Testing summarizer...")
        try:
            summary = framework.summarizer.summarize(
                "This is a long text that needs to be summarized for testing purposes.",
                {"test": True},
                "mid_term"
            )
            print(f"   ✅ Summary: {summary}")
        except Exception as e:
            print(f"   ❌ Summarizer failed: {e}")
        
        # Test retriever
        print("\n6. Testing retriever...")
        try:
            query = Query(text="test memory", user_id="test_user")
            results = framework.retriever.retrieve(query)
            print(f"   ✅ Found {len(results.memories)} memories")
        except Exception as e:
            print(f"   ❌ Retriever failed: {e}")
        
        # Test task runner
        print("\n7. Testing task runner...")
        try:
            def background_task(message: str):
                return f"Background task completed: {message}"
            
            task = framework.task_runner.submit(background_task, "Hello from background!")
            result = framework.task_runner.get_result(task)
            print(f"   ✅ Task result: {result}")
        except Exception as e:
            print(f"   ❌ Task runner failed: {e}")
        
        # Test embedding provider
        print("\n8. Testing embedding provider...")
        try:
            embedding = framework.embedding_provider.get_embedding("Test text for embedding")
            print(f"   ✅ Generated embedding with {len(embedding)} dimensions")
        except Exception as e:
            print(f"   ❌ Embedding provider failed: {e}")
        
        # Test vector store
        print("\n9. Testing vector store...")
        try:
            test_embedding = [0.1] * 10  # Mock embedding
            framework.vector_store.insert_embedding(
                "test_memory",
                "test_user",
                "Test content",
                test_embedding,
                {"test": True}
            )
            print("   ✅ Inserted test embedding")
            
            results = framework.vector_store.search_embeddings(
                test_embedding,
                "test_user",
                limit=5
            )
            print(f"   ✅ Found {len(results)} similar embeddings")
        except Exception as e:
            print(f"   ❌ Vector store failed: {e}")
        
        # Test health checks
        print("\n10. Testing health checks...")
        components = [
            ("Storage", framework.storage),
            ("Cache", framework.cache),
            ("Task Runner", framework.task_runner),
            ("Embedding Provider", framework.embedding_provider),
            ("Vector Store", framework.vector_store)
        ]
        
        for name, component in components:
            try:
                if hasattr(component, 'get_health_status'):
                    health = component.get_health_status()
                    status = "✅" if health.get("status") == "healthy" else "❌"
                    print(f"   {status} {name}: {health.get('status', 'unknown')}")
                else:
                    print(f"   ⚠️  {name}: No health check available")
            except Exception as e:
                print(f"   ❌ {name}: Health check failed - {e}")
        
        print("\n🎉 Framework test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Framework test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_framework()
