#!/usr/bin/env python3
"""
Comprehensive Memorizer Framework Demo
Demonstrates all major features and integrations of the Memorizer Framework.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memorizer.core.framework import create_framework
from memorizer.core.config import FrameworkConfig
from memorizer.core.interfaces import Memory, Query
from memorizer.integrations.framework_integrations import create_integration_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MemorizerDemo:
    """Comprehensive demo of Memorizer Framework capabilities."""

    def __init__(self):
        self.framework = None
        self.memory_manager = None
        self.integration_manager = None

    def setup_framework(self):
        """Set up the Memorizer framework with default configuration."""
        print("🚀 Setting up Memorizer Framework...")

        # Create configuration
        config = FrameworkConfig.create_default()

        # Override with environment variables if available
        if os.getenv("OPENAI_API_KEY"):
            config.summarizer.config["api_key"] = os.getenv("OPENAI_API_KEY")

        # Create framework
        self.framework = create_framework(config)
        self.memory_manager = self.framework.get_memory_manager()
        self.integration_manager = create_integration_manager(self.framework)

        print("✅ Framework initialized successfully!")
        return True

    def demonstrate_basic_memory_operations(self):
        """Demonstrate basic memory operations."""
        print("\n📝 Demonstrating Basic Memory Operations")
        print("=" * 50)

        user_id = "demo_user_001"

        # Store some memories
        memories_to_store = [
            {
                "content": "User prefers dark mode for all applications",
                "metadata": {"category": "preferences", "ui_setting": "dark_mode"}
            },
            {
                "content": "User is working on a Python project using FastAPI",
                "metadata": {"category": "projects", "technology": "python", "framework": "fastapi"}
            },
            {
                "content": "User had trouble with database connections last week",
                "metadata": {"category": "issues", "severity": "medium", "resolved": False}
            },
            {
                "content": "User successfully deployed their app to AWS yesterday",
                "metadata": {"category": "achievements", "platform": "aws", "date": "2024-01-10"}
            }
        ]

        stored_ids = []
        for mem_data in memories_to_store:
            memory = Memory(
                user_id=user_id,
                content=mem_data["content"],
                metadata=mem_data["metadata"]
            )

            memory_id = self.memory_manager.store_memory(memory)
            stored_ids.append(memory_id)
            print(f"✅ Stored memory: {mem_data['content'][:50]}...")

        # Search for memories
        print("\n🔍 Searching for memories...")

        # Search by content
        search_query = Query(
            user_id=user_id,
            content="Python development",
            metadata={}
        )

        results = self.memory_manager.search_memories(search_query)
        print(f"Found {len(results.memories)} memories for 'Python development':")
        for memory in results.memories:
            print(f"  - {memory.content}")

        # Search by metadata
        filter_query = Query(
            user_id=user_id,
            content="",
            metadata={"category": "preferences"}
        )

        filtered_results = self.memory_manager.search_memories(filter_query)
        print(f"\nFound {len(filtered_results.memories)} preference memories:")
        for memory in filtered_results.memories:
            print(f"  - {memory.content}")

        return stored_ids

    def demonstrate_ai_integrations(self):
        """Demonstrate AI framework integrations."""
        print("\n🤖 Demonstrating AI Framework Integrations")
        print("=" * 50)

        # Check available integrations
        available_frameworks = self.integration_manager.list_available_frameworks()
        print("Available framework integrations:")
        for framework, info in available_frameworks.items():
            status = "✅" if info["available"] else "❌"
            print(f"  {status} {framework.upper()}")

        # Demonstrate LangChain integration
        self.demonstrate_langchain_integration()

        # Demonstrate AI SDK integration
        self.demonstrate_ai_sdk_integration()

    def demonstrate_langchain_integration(self):
        """Demonstrate LangChain integration."""
        print("\n🔗 LangChain Integration Demo")
        try:
            # Create LangChain memory
            langchain_memory = self.integration_manager.create_langchain_memory(
                user_id="langchain_user_001",
                session_id="demo_session_001"
            )

            # Simulate a conversation
            conversation_data = [
                {"input": "Hello, I'm working on a chatbot project", "output": "That's great! What framework are you using?"},
                {"input": "I'm using LangChain with OpenAI", "output": "Excellent choice! LangChain is very powerful for building AI applications."},
                {"input": "Can you help me with memory management?", "output": "Absolutely! Memorizer Framework provides excellent memory management for LangChain applications."}
            ]

            for turn in conversation_data:
                # Save the conversation turn
                langchain_memory.save_context(
                    inputs={"input": turn["input"]},
                    outputs={"output": turn["output"]}
                )

            # Load memory variables
            memory_vars = langchain_memory.load_memory_variables({"input": "Tell me about our previous conversation"})
            print(f"✅ LangChain memory loaded {len(memory_vars.get('history', []))} messages")

        except ImportError:
            print("❌ LangChain not available - install with: pip install langchain")
        except Exception as e:
            print(f"❌ LangChain demo failed: {e}")

    def demonstrate_ai_sdk_integration(self):
        """Demonstrate AI SDK integration."""
        print("\n🌐 AI SDK Integration Demo")
        try:
            # Create AI SDK storage
            ai_storage = self.integration_manager.create_ai_sdk_storage(
                user_id="ai_sdk_user_001",
                conversation_id="ai_conversation_001"
            )

            # Simulate saving messages
            from memorizer.integrations.ai_sdk_integration import AIMessage

            messages = [
                AIMessage(role="user", content="What's the weather like today?"),
                AIMessage(role="assistant", content="I don't have access to real-time weather data, but I can help you find weather information."),
                AIMessage(role="user", content="Can you help me write a Python function?"),
                AIMessage(role="assistant", content="Of course! What kind of function do you need help with?")
            ]

            # Save messages
            for message in messages:
                asyncio.run(ai_storage.save_message(message))

            # Load messages back
            loaded_messages = asyncio.run(ai_storage.load_messages())
            print(f"✅ AI SDK storage saved and loaded {len(loaded_messages)} messages")

        except Exception as e:
            print(f"❌ AI SDK demo failed: {e}")

    def demonstrate_memory_lifecycle(self):
        """Demonstrate memory lifecycle management."""
        print("\n🔄 Demonstrating Memory Lifecycle")
        print("=" * 50)

        user_id = "lifecycle_demo_user"

        # Create memories with different ages (simulated)
        lifecycle_memories = [
            {
                "content": "Very recent conversation about API design",
                "tier": "very_new",
                "age_days": 1
            },
            {
                "content": "Discussion about database optimization from last month",
                "tier": "mid_term",
                "age_days": 30
            },
            {
                "content": "Project planning session from 6 months ago",
                "tier": "long_term",
                "age_days": 180
            }
        ]

        for mem_data in lifecycle_memories:
            memory = Memory(
                user_id=user_id,
                content=mem_data["content"],
                metadata={
                    "tier": mem_data["tier"],
                    "simulated_age_days": mem_data["age_days"]
                }
            )

            memory_id = self.memory_manager.store_memory(memory)
            print(f"✅ Stored {mem_data['tier']} memory: {mem_data['content'][:40]}...")

        # Get lifecycle manager
        lifecycle_manager = self.framework.get_lifecycle_manager()

        # Demonstrate memory promotion (this would normally happen automatically)
        print("\n📈 Memory lifecycle management configured")
        print("   - Very new memories: Raw storage (< 10 days)")
        print("   - Mid-term memories: Compressed summaries (< 12 months)")
        print("   - Long-term memories: Highly compressed insights (> 12 months)")

    def demonstrate_health_monitoring(self):
        """Demonstrate health monitoring and metrics."""
        print("\n🏥 Demonstrating Health Monitoring")
        print("=" * 50)

        # Get health status
        health = self.framework.get_health_status()

        print("Framework Health Status:")
        print(f"  Overall Status: {health['framework']['status']}")
        print(f"  Version: {health['framework']['version']}")

        print("\nComponent Health:")
        for component, status in health['components'].items():
            component_status = status.get('status', 'unknown')
            status_icon = "✅" if component_status == 'healthy' else "❓"
            print(f"  {status_icon} {component}: {component_status}")

        print(f"\nRegistry Stats:")
        registry_stats = health.get('registry', {})
        for stat_name, stat_value in registry_stats.items():
            print(f"  - {stat_name}: {stat_value}")

    def demonstrate_advanced_search(self):
        """Demonstrate advanced search capabilities."""
        print("\n🔍 Demonstrating Advanced Search")
        print("=" * 50)

        user_id = "search_demo_user"

        # Store diverse memories for search demo
        search_memories = [
            {
                "content": "Implemented user authentication with JWT tokens",
                "metadata": {"type": "implementation", "technology": "jwt", "feature": "auth"}
            },
            {
                "content": "Fixed bug in payment processing system",
                "metadata": {"type": "bugfix", "system": "payments", "priority": "high"}
            },
            {
                "content": "Optimized database queries for better performance",
                "metadata": {"type": "optimization", "component": "database", "impact": "performance"}
            },
            {
                "content": "Added new API endpoints for user management",
                "metadata": {"type": "feature", "component": "api", "area": "user_management"}
            }
        ]

        # Store memories
        for mem_data in search_memories:
            memory = Memory(
                user_id=user_id,
                content=mem_data["content"],
                metadata=mem_data["metadata"]
            )
            self.memory_manager.store_memory(memory)

        # Demonstrate different search types
        search_scenarios = [
            {
                "name": "Content-based search",
                "query": Query(user_id=user_id, content="authentication JWT", metadata={}),
                "description": "Search for content containing 'authentication JWT'"
            },
            {
                "name": "Metadata filter search",
                "query": Query(user_id=user_id, content="", metadata={"type": "bugfix"}),
                "description": "Search for all bug fix memories"
            },
            {
                "name": "Combined search",
                "query": Query(user_id=user_id, content="performance", metadata={"component": "database"}),
                "description": "Search for database-related performance memories"
            }
        ]

        for scenario in search_scenarios:
            print(f"\n{scenario['name']}:")
            print(f"  Description: {scenario['description']}")

            results = self.memory_manager.search_memories(scenario['query'])
            print(f"  Results: {len(results.memories)} memories found")

            for i, memory in enumerate(results.memories, 1):
                print(f"    {i}. {memory.content}")

    async def run_comprehensive_demo(self):
        """Run the complete demonstration."""
        print("🎯 Memorizer Framework Comprehensive Demo")
        print("=" * 60)

        try:
            # Setup
            if not self.setup_framework():
                print("❌ Failed to setup framework")
                return

            # Run demonstrations
            self.demonstrate_basic_memory_operations()
            self.demonstrate_memory_lifecycle()
            self.demonstrate_advanced_search()
            self.demonstrate_ai_integrations()
            self.demonstrate_health_monitoring()

            print("\n🎉 Demo completed successfully!")
            print("\nNext steps:")
            print("1. 📖 Check the documentation for detailed API reference")
            print("2. 🛠️  Explore the integrations/ directory for framework-specific examples")
            print("3. 🚀 Start the REST API with: python -m memorizer.api.framework_api")
            print("4. 🧪 Run tests with: pytest tests/")
            print("5. 📊 Set up monitoring with the configurations in monitoring/")

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print(f"❌ Demo failed: {e}")

        finally:
            # Cleanup
            if self.framework:
                self.framework.shutdown()
                print("\n🛑 Framework shutdown complete")


async def main():
    """Main demo function."""
    demo = MemorizerDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())