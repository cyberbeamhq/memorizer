#!/usr/bin/env python3
"""
Complete Framework Example
Demonstrates how to use the Memorizer Framework with all components.
"""

import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from memorizer.core.framework import MemorizerFramework, create_framework
from memorizer.core.config import FrameworkConfig
from memorizer.core.interfaces import Memory, Query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    print("🚀 Memorizer Framework Complete Example")
    print("=" * 50)
    
    # 1. Create framework configuration
    config = FrameworkConfig.from_dict({
        "storage": {
            "type": "postgres",
            "connection": "postgresql://user:password@localhost:5432/memorizer"
        },
        "cache": {
            "type": "memory",
            "max_size": 1000,
            "default_ttl": 3600
        },
        "summarizer": {
            "type": "openai",
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "retriever": {
            "type": "hybrid",
            "keyword_weight": 0.4,
            "vector_weight": 0.6
        },
        "scorer": {
            "type": "hybrid",
            "keyword_weight": 0.4,
            "vector_weight": 0.6
        },
        "task_runner": {
            "type": "thread",
            "max_workers": 4
        },
        "embedding_provider": {
            "type": "openai",
            "model": "text-embedding-3-small",
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "vector_store": {
            "type": "pinecone",
            "api_key": os.getenv("PINECONE_API_KEY"),
            "environment": "us-west1-gcp",
            "index_name": "memorizer"
        },
        "pii_filter": {
            "type": "memorizer",
            "sensitivity_level": "medium"
        }
    })
    
    # 2. Create framework instance
    try:
        framework = create_framework(config)
        print("✅ Framework created successfully!")
    except Exception as e:
        print(f"❌ Failed to create framework: {e}")
        return
    
    # 3. Store some memories
    print("\n📝 Storing memories...")
    
    memories = [
        {
            "user_id": "user123",
            "content": "I love working with AI and machine learning. Today I learned about transformer architectures.",
            "metadata": {"topic": "AI", "sentiment": "positive"},
            "tier": "very_new",
            "created_at": datetime.now()
        },
        {
            "user_id": "user123", 
            "content": "The weather is beautiful today. I went for a walk in the park and saw many birds.",
            "metadata": {"topic": "weather", "sentiment": "positive"},
            "tier": "very_new",
            "created_at": datetime.now()
        },
        {
            "user_id": "user123",
            "content": "I need to remember to call my doctor tomorrow about my appointment.",
            "metadata": {"topic": "health", "priority": "high"},
            "tier": "very_new",
            "created_at": datetime.now()
        }
    ]
    
    stored_memories = []
    for memory_data in memories:
        try:
            memory = Memory(**memory_data)
            memory_id = framework.storage.store(memory)
            stored_memories.append(memory_id)
            print(f"  ✅ Stored memory: {memory_id}")
        except Exception as e:
            print(f"  ❌ Failed to store memory: {e}")
    
    # 4. Search for memories
    print("\n🔍 Searching for memories...")
    
    queries = [
        "AI and machine learning",
        "weather and nature",
        "doctor appointment"
    ]
    
    for query_text in queries:
        try:
            query = Query(text=query_text, user_id="user123")
            results = framework.retriever.retrieve(query)
            
            print(f"\n  Query: '{query_text}'")
            print(f"  Found {len(results.memories)} memories:")
            
            for i, memory in enumerate(results.memories[:3], 1):
                print(f"    {i}. {memory.content[:100]}...")
                print(f"       Score: {memory.score:.3f}")
                
        except Exception as e:
            print(f"  ❌ Search failed for '{query_text}': {e}")
    
    # 5. Test PII filtering
    print("\n🔒 Testing PII filtering...")
    
    pii_content = "My email is john.doe@example.com and my phone is 555-123-4567. My SSN is 123-45-6789."
    
    try:
        sanitized_content, pii_data = framework.pii_filter.filter(pii_content)
        print(f"  Original: {pii_content}")
        print(f"  Sanitized: {sanitized_content}")
        print(f"  Detected PII: {len(pii_data.get('detected_pii', []))} items")
    except Exception as e:
        print(f"  ❌ PII filtering failed: {e}")
    
    # 6. Test caching
    print("\n💾 Testing caching...")
    
    try:
        # Store in cache
        framework.cache.set("test_key", {"message": "Hello, World!"}, ttl=60)
        
        # Retrieve from cache
        cached_value = framework.cache.get("test_key")
        print(f"  Cached value: {cached_value}")
        
        # Clear cache
        framework.cache.clear()
        print("  ✅ Cache cleared")
        
    except Exception as e:
        print(f"  ❌ Caching failed: {e}")
    
    # 7. Test task runner
    print("\n⚡ Testing task runner...")
    
    def background_task(message: str, delay: float = 1.0):
        """A simple background task."""
        import time
        time.sleep(delay)
        return f"Task completed: {message}"
    
    try:
        # Submit background task
        task_result = framework.task_runner.submit(background_task, "Hello from background!", 0.5)
        print(f"  ✅ Task submitted: {task_result}")
        
        # Get result (this will block until task completes)
        result = framework.task_runner.get_result(task_result)
        print(f"  ✅ Task result: {result}")
        
    except Exception as e:
        print(f"  ❌ Task runner failed: {e}")
    
    # 8. Test embedding generation
    print("\n🧠 Testing embedding generation...")
    
    try:
        text = "This is a test sentence for embedding generation."
        embedding = framework.embedding_provider.get_embedding(text)
        print(f"  ✅ Generated embedding with {len(embedding)} dimensions")
        
        # Test batch embeddings
        texts = ["First text", "Second text", "Third text"]
        batch_embeddings = framework.embedding_provider.batch_get_embeddings(texts)
        print(f"  ✅ Generated {len(batch_embeddings)} batch embeddings")
        
    except Exception as e:
        print(f"  ❌ Embedding generation failed: {e}")
    
    # 9. Test vector store
    print("\n🗄️ Testing vector store...")
    
    try:
        # Insert test embedding
        test_embedding = [0.1] * 1536  # Example embedding
        framework.vector_store.insert_embedding(
            memory_id="test_memory",
            user_id="user123",
            content="Test content",
            embedding=test_embedding,
            metadata={"test": True}
        )
        print("  ✅ Inserted test embedding")
        
        # Search for similar embeddings
        search_results = framework.vector_store.search_embeddings(
            query_embedding=test_embedding,
            user_id="user123",
            limit=5
        )
        print(f"  ✅ Found {len(search_results)} similar embeddings")
        
    except Exception as e:
        print(f"  ❌ Vector store failed: {e}")
    
    # 10. Test health checks
    print("\n🏥 Testing health checks...")
    
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
                print(f"  {status} {name}: {health.get('status', 'unknown')}")
            else:
                print(f"  ⚠️  {name}: No health check available")
        except Exception as e:
            print(f"  ❌ {name}: Health check failed - {e}")
    
    # 11. Test memory lifecycle
    print("\n🔄 Testing memory lifecycle...")
    
    try:
        # Process a new memory
        new_memory_data = {
            "user_id": "user123",
            "content": "This is a test memory for lifecycle management.",
            "metadata": {"test": True},
            "tier": "very_new",
            "created_at": datetime.now()
        }
        
        memory = Memory(**new_memory_data)
        memory_id = framework.lifecycle.process_new_memory(memory)
        print(f"  ✅ Processed new memory: {memory_id}")
        
        # Get memory status
        status = framework.lifecycle.get_memory_status(memory_id, "user123")
        print(f"  ✅ Memory status: {status}")
        
    except Exception as e:
        print(f"  ❌ Memory lifecycle failed: {e}")
    
    print("\n🎉 Framework example completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
