#!/usr/bin/env python3
"""
Real-World Chatbot Example with Memorizer Framework
Demonstrates building a production-ready chatbot with memory management.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memorizer import MemorizerFramework, FrameworkConfig, Memory, Query


class ChatbotMemory:
    """Enhanced chatbot with memory capabilities."""

    def __init__(self, memorizer_framework: MemorizerFramework):
        self.framework = memorizer_framework
        self.memory_manager = memorizer_framework.get_memory_manager()

    def store_conversation_turn(
        self,
        user_id: str,
        user_message: str,
        bot_response: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store a complete conversation turn."""
        session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        base_metadata = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "source": "chatbot"
        }

        if metadata:
            base_metadata.update(metadata)

        # Store user message
        user_memory = Memory(
            user_id=user_id,
            content=user_message,
            metadata={**base_metadata, "message_type": "user"}
        )
        self.memory_manager.store_memory(user_memory)

        # Store bot response
        bot_memory = Memory(
            user_id=user_id,
            content=bot_response,
            metadata={**base_metadata, "message_type": "assistant"}
        )
        self.memory_manager.store_memory(bot_memory)

    def get_conversation_context(
        self,
        user_id: str,
        current_message: str,
        max_history: int = 5,
        session_id: Optional[str] = None
    ) -> str:
        """Get relevant conversation context for the current message."""
        # Search for relevant memories
        query = Query(
            user_id=user_id,
            content=current_message,
            metadata={"session_id": session_id} if session_id else {}
        )

        results = self.memory_manager.search_memories(query, limit=max_history)

        if not results.memories:
            return ""

        # Format context
        context_lines = ["Previous relevant conversation:"]
        for memory in results.memories:
            role = "User" if memory.metadata.get("message_type") == "user" else "Assistant"
            context_lines.append(f"{role}: {memory.content}")

        return "\n".join(context_lines)

    def store_user_preference(
        self,
        user_id: str,
        preference_type: str,
        preference_value: str,
        context: Optional[str] = None
    ):
        """Store user preferences separately."""
        content = f"User preference: {preference_type} = {preference_value}"
        if context:
            content += f" (Context: {context})"

        memory = Memory(
            user_id=user_id,
            content=content,
            metadata={
                "type": "preference",
                "preference_type": preference_type,
                "preference_value": preference_value,
                "source": "chatbot",
                "timestamp": datetime.now().isoformat()
            }
        )

        self.memory_manager.store_memory(memory)

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get all user preferences."""
        query = Query(
            user_id=user_id,
            content="",
            metadata={"type": "preference"}
        )

        results = self.memory_manager.search_memories(query, limit=50)
        preferences = {}

        for memory in results.memories:
            pref_type = memory.metadata.get("preference_type")
            pref_value = memory.metadata.get("preference_value")
            if pref_type and pref_value:
                preferences[pref_type] = pref_value

        return preferences


class MockLLMChatbot:
    """Mock LLM chatbot for demonstration."""

    def __init__(self, chatbot_memory: ChatbotMemory):
        self.memory = chatbot_memory

    async def generate_response(
        self,
        user_id: str,
        message: str,
        session_id: Optional[str] = None
    ) -> str:
        """Generate a response using memory context."""
        # Get context from memory
        context = self.memory.get_conversation_context(
            user_id=user_id,
            current_message=message,
            session_id=session_id
        )

        # Get user preferences
        preferences = self.memory.get_user_preferences(user_id)

        # Simple rule-based responses (in real implementation, this would be LLM)
        response = self._generate_mock_response(message, context, preferences)

        # Store the conversation turn
        self.memory.store_conversation_turn(
            user_id=user_id,
            user_message=message,
            bot_response=response,
            session_id=session_id
        )

        return response

    def _generate_mock_response(
        self,
        message: str,
        context: str,
        preferences: Dict[str, Any]
    ) -> str:
        """Generate mock responses based on simple rules."""
        message_lower = message.lower()

        # Handle preferences
        if "prefer" in message_lower or "like" in message_lower:
            if "python" in message_lower:
                self.memory.store_user_preference("demo_user", "language", "python", message)
                return "I'll remember that you prefer Python! It's a great choice for many projects."
            elif "dark mode" in message_lower:
                self.memory.store_user_preference("demo_user", "ui_theme", "dark", message)
                return "Got it! I'll remember you prefer dark mode interfaces."

        # Use context in responses
        if context and "python" in context.lower():
            return "I see we've discussed Python before. What specific Python topic can I help you with today?"

        # Apply preferences in responses
        if preferences.get("language") == "python":
            if "programming" in message_lower or "code" in message_lower:
                return "Since you prefer Python, I'd recommend using Python for this task. Here's how you could approach it..."

        # Default responses
        if "hello" in message_lower or "hi" in message_lower:
            return "Hello! How can I help you today?"
        elif "weather" in message_lower:
            return "I don't have access to real-time weather data, but I can help you find weather APIs or services."
        elif "help" in message_lower:
            return "I'm here to help! I can remember our conversations and your preferences. What do you need assistance with?"
        else:
            return "That's interesting! Can you tell me more about what you'd like to know or do?"


async def chatbot_demo():
    """Demonstrate the chatbot with memory."""
    print("🤖 Real-World Chatbot with Memory Demo")
    print("=" * 50)

    # Setup
    config = FrameworkConfig.create_default()
    framework = MemorizerFramework(config)
    chatbot_memory = ChatbotMemory(framework)
    chatbot = MockLLMChatbot(chatbot_memory)

    user_id = "demo_user"
    session_id = "demo_session_001"

    # Simulate a conversation
    conversation = [
        "Hello there!",
        "I'm working on a programming project",
        "I prefer using Python for backend development",
        "Can you help me with API design?",
        "I like dark mode interfaces",
        "What's the weather like today?",
        "I need help with the same programming project we discussed",
        "Can you recommend some Python frameworks?",
    ]

    print("💬 Starting conversation simulation...\n")

    for i, user_message in enumerate(conversation, 1):
        print(f"👤 User: {user_message}")

        # Generate response
        bot_response = await chatbot.generate_response(
            user_id=user_id,
            message=user_message,
            session_id=session_id
        )

        print(f"🤖 Bot: {bot_response}")
        print()

        # Add delay for realism
        await asyncio.sleep(0.5)

    # Show memory stats
    print("📊 Memory Analysis")
    print("-" * 30)

    # Get all memories for this user
    all_memories_query = Query(user_id=user_id, content="")
    all_results = chatbot_memory.memory_manager.search_memories(all_memories_query, limit=100)

    conversation_count = len([m for m in all_results.memories if m.metadata.get("message_type")])
    preference_count = len([m for m in all_results.memories if m.metadata.get("type") == "preference"])

    print(f"💭 Total memories stored: {len(all_results.memories)}")
    print(f"💬 Conversation messages: {conversation_count}")
    print(f"⚙️ User preferences: {preference_count}")

    # Show user preferences
    preferences = chatbot_memory.get_user_preferences(user_id)
    if preferences:
        print(f"\n👤 User preferences learned:")
        for pref_type, pref_value in preferences.items():
            print(f"   • {pref_type}: {pref_value}")

    # Demonstrate context retrieval
    print(f"\n🔍 Context retrieval example:")
    context = chatbot_memory.get_conversation_context(
        user_id=user_id,
        current_message="programming help",
        session_id=session_id
    )
    if context:
        print("Context found for 'programming help':")
        print(context[:200] + "..." if len(context) > 200 else context)

    # Cleanup
    framework.shutdown()
    print("\n✅ Demo completed!")


def production_tips():
    """Show production deployment tips."""
    print("\n🚀 Production Deployment Tips")
    print("=" * 40)

    tips = [
        "🔒 Set up proper authentication and rate limiting",
        "📊 Configure monitoring and health checks",
        "🗄️ Use PostgreSQL for production database",
        "⚡ Set up Redis for caching and job queues",
        "🐳 Use Docker for containerized deployment",
        "📈 Implement proper logging and metrics",
        "🔐 Secure API keys in environment variables",
        "🔄 Set up automated backup for memories",
        "⚖️ Configure load balancing for high availability",
        "🧪 Implement comprehensive testing"
    ]

    for tip in tips:
        print(f"  {tip}")

    print("\n📖 See INSTALLATION.md and docker-compose.yml for setup instructions!")


async def main():
    """Main demo function."""
    await chatbot_demo()
    production_tips()


if __name__ == "__main__":
    asyncio.run(main())