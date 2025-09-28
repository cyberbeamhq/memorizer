#!/usr/bin/env python3
"""
Memorizer Framework Demo
Quick demonstration script to show framework capabilities.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("🚀 Memorizer Framework Demo")
    print("=" * 40)

    try:
        from memorizer import MemorizerFramework, FrameworkConfig

        # Create framework
        print("Initializing framework...")
        config = FrameworkConfig.create_default()
        framework = MemorizerFramework(config)
        memory_manager = framework.get_memory_manager()

        # Store a memory
        print("Storing a memory...")
        memory_id = memory_manager.store_memory(
            user_id="demo_user",
            content="I love using AI frameworks for building intelligent applications!"
        )
        print(f"✅ Memory stored with ID: {memory_id}")

        # Search for the memory
        print("Searching for memories...")
        results = memory_manager.search_memories(
            user_id="demo_user",
            query="AI frameworks"
        )
        print(f"✅ Found {results.total_found} memories")

        for memory in results.memories:
            print(f"📝 {memory.content}")

        # Check health
        health = framework.get_health_status()
        print(f"📊 Framework status: {health['framework']['status']}")

        # Cleanup
        framework.shutdown()

        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Run examples/quickstart_example.py for a 5-minute tutorial")
        print("2. Run examples/comprehensive_framework_demo.py for full features")
        print("3. Start the API: python -m memorizer.api.framework_api")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install the framework first:")
        print("pip install -e .")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Check the logs for more details.")


if __name__ == "__main__":
    main()
