#!/usr/bin/env python3
"""
Memorizer Framework Quickstart Example
Get started with Memorizer in 5 minutes!
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memorizer import MemorizerFramework, FrameworkConfig, Memory, Query


def main():
    """5-minute quickstart example."""
    print("🚀 Memorizer Framework - 5 Minute Quickstart")
    print("=" * 50)

    # 1. Create and configure the framework
    print("1️⃣ Setting up Memorizer...")
    config = FrameworkConfig.create_default()
    framework = MemorizerFramework(config)
    memory_manager = framework.get_memory_manager()
    print("✅ Framework ready!")

    # 2. Store some memories
    print("\n2️⃣ Storing memories...")
    user_id = "quickstart_user"

    memories = [
        "I prefer Python for backend development",
        "I'm working on a chatbot project using OpenAI",
        "I had trouble with Docker container networking last week",
        "My favorite IDE is VS Code with Python extensions"
    ]

    for content in memories:
        memory = Memory(user_id=user_id, content=content)
        memory_id = memory_manager.store_memory(memory)
        print(f"✅ Stored: {content}")

    # 3. Search for relevant memories
    print("\n3️⃣ Searching memories...")
    query = Query(
        user_id=user_id,
        content="Python development tools"
    )

    results = memory_manager.search_memories(query)
    print(f"🔍 Found {len(results.memories)} relevant memories:")
    for memory in results.memories:
        print(f"   📝 {memory.content}")

    # 4. Use with LangChain (if available)
    print("\n4️⃣ Testing LangChain integration...")
    try:
        from memorizer.integrations.framework_integrations import get_langchain_memory

        langchain_memory = get_langchain_memory(
            framework,
            user_id="langchain_quickstart",
            session_id="session_001"
        )

        # Simulate a conversation
        langchain_memory.save_context(
            inputs={"input": "Hello, I need help with my Python project"},
            outputs={"output": "I'd be happy to help! What specifically do you need assistance with?"}
        )

        print("✅ LangChain integration working!")

    except ImportError:
        print("ℹ️ LangChain not installed - install with: pip install langchain")

    # 5. Check framework health
    print("\n5️⃣ Framework health check...")
    health = framework.get_health_status()
    print(f"📊 Framework status: {health['framework']['status']}")

    # 6. Cleanup
    framework.shutdown()
    print("\n🎉 Quickstart complete! Check out the comprehensive demo for more features.")


if __name__ == "__main__":
    main()