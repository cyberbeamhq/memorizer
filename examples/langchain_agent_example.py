"""
LangChain Agent with Memorizer Integration - Complete Example
Shows how to connect Memorizer to a LangChain agent for intelligent memory management.
"""

import os
from datetime import datetime

# Step 1: Initialize Memorizer
from memorizer import create_memory_manager
from memorizer.core.simple_config import MemoryConfig
from memorizer.integrations.langchain_integration import (
    LangChainMemorizerMemory,
    LangChainMemorizerCallback,
    create_langchain_integration
)

# Step 2: Initialize LangChain (requires: pip install langchain langchain-openai)
try:
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain.tools import Tool
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.schema import SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain not installed. Install with: pip install langchain langchain-openai")


def main():
    """Example: LangChain agent with Memorizer for persistent memory."""

    if not LANGCHAIN_AVAILABLE:
        print("\n❌ This example requires LangChain. Install with:")
        print("   pip install langchain langchain-openai")
        return

    print("=" * 70)
    print("LangChain Agent + Memorizer Integration Example")
    print("=" * 70)

    # ========================================
    # 1. Initialize Memorizer
    # ========================================
    print("\n1. Initializing Memorizer...")
    config = MemoryConfig()
    memory_manager = create_memory_manager(config)
    print("   ✓ Memory manager created")

    # ========================================
    # 2. Create LangChain Memory Integration
    # ========================================
    print("\n2. Setting up LangChain integration...")

    user_id = "agent_user_123"
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create Memorizer-backed LangChain memory
    langchain_memory = LangChainMemorizerMemory(
        memorizer_framework=None,  # Will use simple manager
        user_id=user_id,
        session_id=session_id,
        memory_key="chat_history",
        return_messages=True
    )

    # Override memory manager with simple manager
    langchain_memory.memory_manager = memory_manager

    print(f"   ✓ Memory integration created")
    print(f"   ✓ User ID: {user_id}")
    print(f"   ✓ Session ID: {session_id}")

    # ========================================
    # 3. Create LangChain Agent
    # ========================================
    print("\n3. Creating LangChain agent...")

    # Define some simple tools
    def calculator_tool(query: str) -> str:
        """Simple calculator that evaluates math expressions."""
        try:
            # Safe eval for simple math
            result = eval(query, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except:
            return "Error: Invalid math expression"

    def memory_search_tool(query: str) -> str:
        """Search through agent's memory."""
        results = memory_manager.search_memories(user_id, query, limit=3)
        if results.total_found > 0:
            memories = "\n".join([f"- {m.content}" for m in results.memories])
            return f"Found {results.total_found} relevant memories:\n{memories}"
        return "No relevant memories found"

    tools = [
        Tool(
            name="calculator",
            func=calculator_tool,
            description="Useful for math calculations. Input should be a math expression."
        ),
        Tool(
            name="memory_search",
            func=memory_search_tool,
            description="Search through past conversations and memories. Input should be a search query."
        )
    ]

    # Create LLM (you need OPENAI_API_KEY set, or use a different LLM)
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo"
        )
        print("   ✓ Using OpenAI GPT-3.5-turbo")
    except Exception as e:
        print(f"   ⚠️  OpenAI not configured: {e}")
        print("   💡 Set OPENAI_API_KEY environment variable")
        # Use mock for demo
        print("   ℹ️  Using mock mode for demonstration")
        llm = None
        return

    # Create prompt with memory placeholder
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a helpful AI assistant with access to your memory.
You can remember past conversations and use tools to help users.
Always check your memory for relevant context before answering."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)

    # Create callback to automatically save memories
    memorizer_callback = LangChainMemorizerCallback(
        memorizer_framework=None,
        user_id=user_id,
        session_id=session_id,
        capture_tool_calls=True
    )
    memorizer_callback.memory_manager = memory_manager

    # Create agent executor with memory
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=langchain_memory,
        callbacks=[memorizer_callback],
        verbose=True
    )

    print("   ✓ Agent created with Memorizer integration")

    # ========================================
    # 4. Interact with Agent
    # ========================================
    print("\n4. Testing agent with memory...")
    print("-" * 70)

    test_queries = [
        "My name is Alice and I love machine learning.",
        "What's 25 + 17?",
        "Do you remember my name?",
        "Search your memory for what I love.",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}] {query}")
        try:
            response = agent_executor.invoke({"input": query})
            print(f"[Response] {response['output']}\n")
        except Exception as e:
            print(f"[Error] {e}\n")

    # ========================================
    # 5. Show Memory Statistics
    # ========================================
    print("\n" + "=" * 70)
    print("Memory Statistics")
    print("=" * 70)

    stats = memory_manager.get_stats(user_id)
    print(f"Total memories stored: {stats.total_memories}")
    print(f"Memories by tier: {stats.memory_by_tier}")

    # Show all stored memories
    all_memories = memory_manager.search_memories(user_id, "", limit=100)
    print(f"\nAll conversation memories:")
    for mem in all_memories.memories:
        msg_type = mem.metadata.get('message_type', 'unknown')
        print(f"  [{msg_type}] {mem.content[:80]}...")

    print("\n" + "=" * 70)
    print("✅ LangChain + Memorizer Integration Complete!")
    print("=" * 70)
    print("\nKey Benefits:")
    print("  ✓ Automatic memory storage from conversations")
    print("  ✓ Intelligent retrieval based on relevance")
    print("  ✓ Tiered memory lifecycle (very_new → mid_term → long_term)")
    print("  ✓ Compression policies for long-term storage")
    print("  ✓ Multi-session memory with searchability")
    print("  ✓ Tool call tracking and error capture")


def simple_example():
    """Minimal example without LangChain dependencies."""
    print("\n" + "=" * 70)
    print("Simple Memorizer Example (No LangChain Required)")
    print("=" * 70)

    # Initialize Memorizer
    config = MemoryConfig()
    manager = create_memory_manager(config)

    user_id = "simple_user"

    # Store some memories
    print("\nStoring memories...")
    manager.store_memory(user_id, "User said: My favorite color is blue", {"type": "preference"})
    manager.store_memory(user_id, "User asked about Python programming", {"type": "question"})
    manager.store_memory(user_id, "User mentioned living in San Francisco", {"type": "personal_info"})

    # Search memories
    print("\nSearching for 'favorite'...")
    results = manager.search_memories(user_id, "favorite", limit=5)
    print(f"Found {results.total_found} results:")
    for mem in results.memories:
        print(f"  - {mem.content}")

    # Get stats
    stats = manager.get_stats(user_id)
    print(f"\nMemory stats: {stats.total_memories} total memories")
    print(f"By tier: {stats.memory_by_tier}")

    print("\n✅ Simple example complete!")


if __name__ == "__main__":
    # Run simple example first (always works)
    simple_example()

    # Then try LangChain if available
    if LANGCHAIN_AVAILABLE:
        print("\n\n")
        main()
    else:
        print("\n" + "=" * 70)
        print("LangChain Example Skipped")
        print("=" * 70)
        print("\nTo run the full LangChain example:")
        print("  1. pip install langchain langchain-openai")
        print("  2. export OPENAI_API_KEY='your-api-key'")
        print("  3. python examples/langchain_agent_example.py")
