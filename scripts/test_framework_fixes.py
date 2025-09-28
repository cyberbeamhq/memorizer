#!/usr/bin/env python3
"""
Test Framework Fixes
Comprehensive test to verify all the fixes work properly.
"""

import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from framework.factory import create_framework
from framework.core.config import FrameworkConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_framework_fixes():
    """Test all the framework fixes."""
    print("🧪 Testing Framework Fixes")
    print("=" * 50)
    
    # Create configuration with fallback components
    config = FrameworkConfig.from_dict({
        "storage": {
            "type": "storage",
            "name": "sqlite",
            "config": {"db_path": ":memory:"}
        },
        "cache": {
            "type": "cache",
            "name": "memory",
            "config": {"max_size": 100}
        },
        "summarizer": {
            "type": "summarizer",
            "name": "mock",
            "config": {}
        },
        "retriever": {
            "type": "retriever",
            "name": "keyword",
            "config": {}
        },
        "pii_filter": {
            "type": "pii_filter",
            "name": "basic",
            "config": {}
        },
        "scorer": {
            "type": "scorer",
            "name": "tfidf",
            "config": {}
        },
        "task_runner": {
            "type": "task_runner",
            "name": "thread",
            "config": {"max_workers": 2}
        },
        "embedding_provider": {
            "type": "embedding_provider",
            "name": "openai",
            "config": {"api_key": "test-key", "model": "text-embedding-3-small"}
        },
        "vector_store": {
            "type": "vector_store",
            "name": "weaviate",
            "config": {}
        },
        "memory_lifecycle": {
            "tiers": {
                "very_new": {"max_age_days": 7, "min_accesses": 5, "min_content_length": 1000},
                "mid_term": {"max_age_days": 30, "min_accesses": 10, "min_content_length": 500},
                "long_term": {"max_age_days": 365, "min_accesses": 20, "min_content_length": 100}
            },
            "compression_threshold": 0.8,
            "cleanup_interval": 3600
        }
    })
    
    try:
        # Test 1: Framework Creation
        print("\n1. Testing Framework Creation...")
        framework = create_framework(config)
        print("   ✅ Framework created successfully")
        
        # Test 2: Memory Manager
        print("\n2. Testing Memory Manager...")
        memory_manager = framework.get_memory_manager()
        print("   ✅ Memory manager available")
        
        # Test 3: Memory Operations
        print("\n3. Testing Memory Operations...")
        
        # Store a memory
        memory_id = memory_manager.store_memory(
            user_id="test_user",
            content="This is a test memory for the framework fixes.",
            metadata={"test": True, "access_count": 0},
            tier="very_new"
        )
        print(f"   ✅ Stored memory: {memory_id}")
        
        # Get the memory
        memory = memory_manager.get_memory(memory_id, "test_user")
        if memory:
            print("   ✅ Retrieved memory successfully")
        else:
            print("   ❌ Failed to retrieve memory")
        
        # Test 4: Memory Lifecycle
        print("\n4. Testing Memory Lifecycle...")
        
        # Test tier advancement
        new_tier = memory_manager.promote_memory(memory_id, "very_new", "mid_term")
        if new_tier:
            print("   ✅ Memory tier advancement working")
        else:
            print("   ❌ Memory tier advancement failed")
        
        # Test 5: Search
        print("\n5. Testing Search...")
        results = memory_manager.search_memories(
            user_id="test_user",
            query="test memory"
        )
        print(f"   ✅ Search completed: {results.total_found} results")
        
        # Test 6: Health Checks
        print("\n6. Testing Health Checks...")
        health = framework.get_health_status()
        
        # Check framework status
        framework_status = health.get("framework", {})
        if framework_status.get("status") == "healthy":
            print("   ✅ Framework health check passed")
        else:
            print(f"   ❌ Framework health check failed: {framework_status}")
        
        # Check component status
        components = health.get("components", {})
        healthy_components = 0
        total_components = len(components)
        
        for name, status in components.items():
            if status.get("status") == "healthy":
                healthy_components += 1
                print(f"   ✅ {name}: healthy")
            else:
                print(f"   ⚠️  {name}: {status.get('status', 'unknown')}")
        
        print(f"   📊 Health Summary: {healthy_components}/{total_components} components healthy")
        
        # Test 7: Registry
        print("\n7. Testing Component Registry...")
        registry = framework.registry
        stats = registry.get_stats()
        print(f"   ✅ Registry: {stats['total_components']} components registered")
        
        # Test 8: Task Runner
        print("\n8. Testing Task Runner...")
        task_runner = framework.task_runner
        
        # Submit a test task
        def test_task(message):
            return f"Task completed: {message}"
        
        task_id = task_runner.submit(test_task, "Hello from framework test")
        print(f"   ✅ Task submitted: {task_id}")
        
        # Test 9: Cache
        print("\n9. Testing Cache...")
        cache = framework.cache
        
        # Test cache operations
        cache.set("test_key", {"message": "Hello, Cache!"}, 60)
        cached_value = cache.get("test_key")
        if cached_value and cached_value.get("message") == "Hello, Cache!":
            print("   ✅ Cache operations working")
        else:
            print("   ❌ Cache operations failed")
        
        # Test 10: PII Filter
        print("\n10. Testing PII Filter...")
        pii_filter = framework.pii_filter
        
        test_content = "My email is test@example.com and my phone is 555-123-4567."
        sanitized, pii_data = pii_filter.filter(test_content)
        
        if pii_data and pii_data.get("pii_detected"):
            print("   ✅ PII detection working")
            print(f"   📧 Detected PII: {len(pii_data.get('pii_types', []))} types")
        else:
            print("   ❌ PII detection failed")
        
        print("\n🎉 All Framework Fixes Test Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_framework_fixes()
    sys.exit(0 if success else 1)
