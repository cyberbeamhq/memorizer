"""
framework_example.py
Example demonstrating the Memorizer Framework usage.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memorizer.core.framework import create_framework
from memorizer.core.config import FrameworkConfig


def main():
    """Demonstrate framework usage."""
    print("🚀 Memorizer Framework Example")
    print("=" * 50)
    
    # Create a temporary config file for this example
    config_path = Path("temp_config.yaml")
    
    try:
        # Create default configuration
        create_default_config_file(config_path)
        print(f"📁 Created configuration: {config_path}")
        
        # Load framework with configuration
        print("🔧 Loading framework...")
        framework = create_framework(str(config_path))
        
        # Get memory manager
        memory_manager = framework.get_memory_manager()
        print("✅ Framework loaded successfully!")
        
        # Demonstrate memory operations
        print("\n📝 Memory Operations Demo:")
        print("-" * 30)
        
        # Store a memory
        print("1. Storing a memory...")
        memory_id = memory_manager.store_memory(
            user_id="demo_user",
            content="This is a demo memory about machine learning and AI frameworks. It contains some important information that should be remembered.",
            metadata={
                "category": "demo",
                "importance": "high",
                "tags": ["ai", "machine-learning", "framework"]
            },
            tier="very_new"
        )
        print(f"   ✅ Stored memory: {memory_id}")
        
        # Store another memory
        print("2. Storing another memory...")
        memory_id2 = memory_manager.store_memory(
            user_id="demo_user",
            content="Another memory about Python programming and data science. This contains contact information like john@example.com and phone (555) 123-4567.",
            metadata={
                "category": "programming",
                "importance": "medium",
                "tags": ["python", "data-science"]
            },
            tier="very_new"
        )
        print(f"   ✅ Stored memory: {memory_id2}")
        
        # Search memories
        print("3. Searching memories...")
        results = memory_manager.search_memories(
            user_id="demo_user",
            query="machine learning",
            limit=5
        )
        print(f"   🔍 Found {results.total_found} memories")
        print(f"   ⏱️  Search took {results.retrieval_time:.3f}s")
        print(f"   📊 Source: {results.source}")
        
        # Show search results
        for i, memory in enumerate(results.memories, 1):
            print(f"   {i}. {memory.content[:50]}... (score: {results.scores[i-1]:.3f})")
        
        # Get memory by ID
        print("4. Retrieving specific memory...")
        memory = memory_manager.get_memory(memory_id, "demo_user")
        if memory:
            print(f"   📄 Memory content: {memory.content[:100]}...")
            print(f"   🏷️  Tier: {memory.tier}")
            print(f"   📅 Created: {memory.created_at}")
            print(f"   🔒 PII Filtered: {memory.metadata.get('pii_filtered', False)}")
        
        # Update memory
        print("5. Updating memory...")
        success = memory_manager.update_memory(
            memory_id,
            "demo_user",
            content="Updated memory content with more detailed information about machine learning frameworks and their applications.",
            metadata={"updated": True, "version": 2}
        )
        print(f"   {'✅' if success else '❌'} Update {'successful' if success else 'failed'}")
        
        # Get user statistics
        print("6. Getting user statistics...")
        stats = memory_manager.get_user_stats("demo_user")
        print(f"   📊 User stats: {stats}")
        
        # Health check
        print("7. Checking framework health...")
        health = framework.get_health_status()
        print(f"   🏥 Framework status: {health['framework']['status']}")
        print(f"   🔧 Components: {len(health['components'])}")
        
        # Show component health
        for name, status in health['components'].items():
            status_icon = "✅" if status.get('status') == 'healthy' else "❌"
            print(f"      {status_icon} {name}: {status.get('status', 'unknown')}")
        
        print("\n🎉 Framework demonstration completed!")
        print("\n💡 Next steps:")
        print("   • Customize memorizer.yaml for your needs")
        print("   • Add your own components and plugins")
        print("   • Integrate with your application")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary config file
        if config_path.exists():
            config_path.unlink()
            print(f"\n🧹 Cleaned up temporary file: {config_path}")


if __name__ == "__main__":
    main()
