#!/usr/bin/env python3
"""
Simple Memory Management Example
Demonstrates the core memory management functionality without infrastructure complexity.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import memorizer


def main():
    """Demonstrate memory management."""
    print("🧠 Memory Management Demo")
    print("=" * 40)

    # Example 1: Basic in-memory setup
    print("\n1. Basic In-Memory Setup")
    memory = memorizer.create_memory()
    print("✅ Created memory manager")

    # Store some memories
    user_id = "demo_user"

    memories = [
        "I love playing guitar and writing songs",
        "My favorite programming language is Python",
        "I'm working on a machine learning project about NLP",
        "I enjoy hiking and spending time in nature"
    ]

    memory_ids = []
    for content in memories:
        memory_id = memory.store_memory(user_id, content)
        memory_ids.append(memory_id)
        print(f"✅ Stored: {memory_id}")

    # Search memories
    print("\n2. Memory Search")
    results = memory.search_memories(user_id, "programming projects", limit=3)
    print(f"📝 Search for 'programming projects' found {len(results.memories)} results:")
    for result in results.memories:
        print(f"   - {result.content[:50]}... (score: {result.metadata.get('score', 'N/A')})")

    # Example 2: External providers (without actual connections)
    print("\n3. External Provider Configuration")

    # Supabase example
    try:
        supabase_memory = memorizer.create_memory_manager(
            storage_provider="supabase",
            vector_store="pinecone",
            llm_provider="openai",
            supabase_url="https://your-project.supabase.co",
            supabase_password="your-password",
            pinecone_api_key="your-api-key"
        )
        print("✅ Configured for Supabase + Pinecone (mock)")
    except Exception as e:
        print(f"⚠️  External providers not available: {e}")

    # Example 3: Memory lifecycle (if available)
    print("\n4. Memory Lifecycle")
    try:
        # Get memory details
        if memory_ids and hasattr(memory, 'get_memory'):
            memory_obj = memory.get_memory(memory_ids[0], user_id)
            if memory_obj:
                print(f"✅ Retrieved memory: {memory_obj.content[:30]}...")
                print(f"   Tier: {memory_obj.tier}")
                print(f"   Created: {memory_obj.created_at}")

        # Check memory statistics
        if hasattr(memory, 'get_stats'):
            stats = memory.get_stats(user_id)
            print(f"📊 Memory stats: {stats.total_memories} total, {stats.memory_by_tier}")
        else:
            print("📊 Memory statistics not available in simple mode")

    except Exception as e:
        print(f"⚠️  Lifecycle features not available: {e}")

    print("\n🎉 Memory demo completed!")


def advanced_example():
    """Show more advanced memory management features."""
    print("\n🚀 Advanced Memory Features")
    print("=" * 40)

    # Custom configuration
    print("\n1. Custom Configuration")
    config = memorizer.MemoryConfig(
        storage_provider="memory",
        vector_store="memory",
        llm_provider="mock",
        very_new_ttl_days=3,      # Shorter retention for demo
        mid_term_ttl_days=14,
        compression_threshold=0.5  # More aggressive compression
    )

    memory = memorizer.MemoryManager(config)
    print("✅ Created memory manager with custom configuration")

    # Batch operations
    print("\n2. Batch Memory Operations")
    user_id = "advanced_user"

    # Store multiple memories
    conversation_memories = [
        "User asked about Python best practices",
        "Discussed object-oriented programming concepts",
        "Explained the difference between lists and tuples",
        "Covered exception handling and try-catch blocks",
        "Reviewed code optimization techniques"
    ]

    batch_ids = []
    for i, content in enumerate(conversation_memories):
        memory_id = memory.store_memory(
            user_id,
            content,
            metadata={"conversation_turn": i, "topic": "python_programming"}
        )
        batch_ids.append(memory_id)

    print(f"✅ Stored {len(batch_ids)} conversation memories")

    # Search with different strategies
    print("\n3. Different Search Strategies")

    queries = [
        "object oriented programming",
        "error handling",
        "performance optimization"
    ]

    for query in queries:
        results = memory.search_memories(user_id, query, limit=2)
        print(f"🔍 '{query}' → {len(results.memories)} results")
        for result in results.memories:
            topic = result.metadata.get("topic", "unknown")
            turn = result.metadata.get("conversation_turn", "?")
            print(f"   Turn {turn}: {result.content[:40]}... (topic: {topic})")

    print("\n✨ Advanced demo completed!")


if __name__ == "__main__":
    try:
        main()
        advanced_example()

        print("\n💡 Next Steps:")
        print("  • Try with external providers (Supabase, Pinecone)")
        print("  • Integrate into your AI application")
        print("  • Experiment with different memory lifecycle settings")
        print("  • Check out other examples in this directory")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()