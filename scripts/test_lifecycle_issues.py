#!/usr/bin/env python3
"""
Test Memory Lifecycle Issues
Comprehensive test to identify issues in the memory lifecycle system.
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from framework.factory import create_framework
from framework.core.config import FrameworkConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_lifecycle_issues():
    """Test memory lifecycle for issues."""
    print("🔍 Testing Memory Lifecycle Issues")
    print("=" * 50)
    
    # Create test configuration
    config = FrameworkConfig.from_dict({
        'storage': {'type': 'storage', 'name': 'sqlite', 'config': {'db_path': ':memory:'}},
        'cache': {'type': 'cache', 'name': 'memory', 'config': {'max_size': 100}},
        'summarizer': {'type': 'summarizer', 'name': 'mock', 'config': {}},
        'retriever': {'type': 'retriever', 'name': 'keyword', 'config': {}},
        'pii_filter': {'type': 'pii_filter', 'name': 'basic', 'config': {}},
        'scorer': {'type': 'scorer', 'name': 'tfidf', 'config': {}},
        'task_runner': {'type': 'task_runner', 'name': 'thread', 'config': {'max_workers': 2}},
        'embedding_provider': {'type': 'embedding_provider', 'name': 'openai', 'config': {'api_key': 'test-key', 'model': 'text-embedding-3-small'}},
        'vector_store': {'type': 'vector_store', 'name': 'weaviate', 'config': {}},
        'memory_lifecycle': {
            'tiers': {
                'very_new': {'ttl_days': 7, 'max_items': 1000, 'max_age_days': 7, 'min_accesses': 5, 'min_content_length': 1000},
                'mid_term': {'ttl_days': 30, 'max_items': 5000, 'max_age_days': 30, 'min_accesses': 10, 'min_content_length': 500},
                'long_term': {'ttl_days': 365, 'max_items': 10000, 'max_age_days': 365, 'min_accesses': 20, 'min_content_length': 100}
            },
            'compression_threshold': 0.8,
            'cleanup_interval': 3600
        }
    })
    
    try:
        framework = create_framework(config)
        memory_manager = framework.get_memory_manager()
        
        print("\n1. Testing Basic Memory Operations...")
        
        # Store a memory
        memory_id = memory_manager.store_memory(
            user_id='test_user',
            content='This is a test memory for lifecycle investigation.',
            metadata={'test': True, 'access_count': 0},
            tier='very_new'
        )
        print(f"   ✅ Stored memory: {memory_id}")
        
        # Get the memory
        memory = memory_manager.get_memory(memory_id, 'test_user')
        if memory:
            print(f"   ✅ Retrieved memory: tier={memory.tier}, created_at={memory.created_at}")
        else:
            print("   ❌ Failed to retrieve memory")
            return
        
        print("\n2. Testing Manual Memory Promotion...")
        
        # Try to promote memory manually
        success = memory_manager.promote_memory(memory_id, 'very_new', 'mid_term', 'test_user')
        print(f"   Manual promotion result: {success}")
        
        # Check memory after promotion
        memory = memory_manager.get_memory(memory_id, 'test_user')
        if memory:
            print(f"   Memory tier after promotion: {memory.tier}")
        else:
            print("   ❌ Memory not found after promotion")
        
        print("\n3. Testing Automatic Tier Advancement...")
        
        # Reset memory to very_new for testing
        if memory:
            memory.tier = 'very_new'
            memory.metadata['access_count'] = 6  # Set to trigger advancement
            memory_manager.update_memory(memory_id, 'test_user', metadata=memory.metadata)
        
        # Try automatic advancement
        new_tier = memory_manager.advance_memory_tier(memory_id, 'test_user', 'very_new')
        print(f"   Automatic advancement result: {new_tier}")
        
        # Check memory after advancement
        memory = memory_manager.get_memory(memory_id, 'test_user')
        if memory:
            print(f"   Memory tier after advancement: {memory.tier}")
            print(f"   Memory metadata: {memory.metadata}")
        else:
            print("   ❌ Memory not found after advancement")
        
        print("\n4. Testing Edge Cases...")
        
        # Test invalid tier transition
        invalid_promotion = memory_manager.promote_memory(memory_id, 'very_new', 'invalid_tier', 'test_user')
        print(f"   Invalid tier transition result: {invalid_promotion}")
        
        # Test promotion to same tier
        same_tier_promotion = memory_manager.promote_memory(memory_id, 'very_new', 'very_new', 'test_user')
        print(f"   Same tier promotion result: {same_tier_promotion}")
        
        # Test promotion from terminal tier
        terminal_promotion = memory_manager.promote_memory(memory_id, 'long_term', 'very_new', 'test_user')
        print(f"   Terminal tier promotion result: {terminal_promotion}")
        
        print("\n5. Testing Cleanup Logic...")
        
        # Test cleanup with current memory
        cleaned = memory_manager.cleanup_expired_memories('test_user')
        print(f"   Cleanup result: {cleaned} memories cleaned")
        
        # Test cleanup with non-existent user
        cleaned_nonexistent = memory_manager.cleanup_expired_memories('nonexistent_user')
        print(f"   Cleanup nonexistent user result: {cleaned_nonexistent}")
        
        print("\n6. Testing Configuration Issues...")
        
        # Test with missing tier configuration
        lifecycle = framework.lifecycle
        print(f"   Tiers configured: {list(lifecycle.tiers.keys())}")
        print(f"   Compression threshold: {lifecycle.compression_threshold}")
        print(f"   Cleanup interval: {lifecycle.cleanup_interval}")
        
        print("\n7. Testing Memory Access Count Logic...")
        
        # Test memory with high access count
        high_access_memory_id = memory_manager.store_memory(
            user_id='test_user',
            content='High access memory',
            metadata={'access_count': 15},
            tier='very_new'
        )
        
        # Try advancement with high access count
        new_tier_high_access = memory_manager.advance_memory_tier(high_access_memory_id, 'test_user', 'very_new')
        print(f"   High access memory advancement: {new_tier_high_access}")
        
        print("\n8. Testing Content Length Logic...")
        
        # Test memory with long content
        long_content = "This is a very long memory content. " * 100  # ~3500 characters
        long_memory_id = memory_manager.store_memory(
            user_id='test_user',
            content=long_content,
            metadata={'access_count': 0},
            tier='very_new'
        )
        
        # Try advancement with long content
        new_tier_long_content = memory_manager.advance_memory_tier(long_memory_id, 'test_user', 'very_new')
        print(f"   Long content memory advancement: {new_tier_long_content}")
        
        print("\n9. Testing Age-based Logic...")
        
        # Create a memory with old timestamp
        old_memory_id = memory_manager.store_memory(
            user_id='test_user',
            content='Old memory',
            metadata={'access_count': 0},
            tier='very_new'
        )
        
        # Manually set old timestamp (simulate old memory)
        old_memory = memory_manager.get_memory(old_memory_id, 'test_user')
        if old_memory:
            old_memory.created_at = datetime.now() - timedelta(days=10)  # 10 days old
            memory_manager.update_memory(old_memory_id, 'test_user', metadata=old_memory.metadata)
        
        # Try advancement with old memory
        new_tier_old = memory_manager.advance_memory_tier(old_memory_id, 'test_user', 'very_new')
        print(f"   Old memory advancement: {new_tier_old}")
        
        print("\n🎯 Lifecycle Test Completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lifecycle_issues()
