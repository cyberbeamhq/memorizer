#!/usr/bin/env python3
"""
AI Integration Example
Practical example showing how to use Memorizer with different AI frameworks.
"""

import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def langchain_example():
    """Example of using Memorizer with LangChain."""
    print("🔗 LangChain Integration Example")
    print("=" * 40)
    
    try:
        from memorizer.core.framework import create_framework
        from framework.core.config import load_config
        from agent_integrations import AgentIntegrationManager, AgentMemory, AgentContext
        
        # Initialize framework
        config = load_config("memorizer.yaml")
        framework = create_framework(config)
        memory_manager = framework.get_memory_manager()
        integration_manager = AgentIntegrationManager(memory_manager)
        
        # Get LangChain integration
        langchain = integration_manager.get_integration("langchain")
        
        # Simulate a conversation with a LangChain agent
        agent_id = "customer_service_bot"
        session_id = "session_001"
        
        # Store conversation memories
        conversation_memories = [
            AgentMemory(
                agent_id=agent_id,
                session_id=session_id,
                content="User: Hi, I need help with my order",
                metadata={"speaker": "user", "intent": "order_help"},
                timestamp=datetime.now(),
                memory_type="conversation"
            ),
            AgentMemory(
                agent_id=agent_id,
                session_id=session_id,
                content="Agent: I'd be happy to help with your order. What's your order number?",
                metadata={"speaker": "agent", "intent": "request_info"},
                timestamp=datetime.now(),
                memory_type="conversation"
            ),
            AgentMemory(
                agent_id=agent_id,
                session_id=session_id,
                content="User: My order number is ORD-12345",
                metadata={"speaker": "user", "order_number": "ORD-12345"},
                timestamp=datetime.now(),
                memory_type="conversation"
            ),
            AgentMemory(
                agent_id=agent_id,
                session_id=session_id,
                content="Agent: I found your order. It's currently being processed and will ship tomorrow.",
                metadata={"speaker": "agent", "order_status": "processing", "ship_date": "tomorrow"},
                timestamp=datetime.now(),
                memory_type="conversation"
            )
        ]
        
        # Store all memories
        for memory in conversation_memories:
            memory_id = langchain.store_memory(memory)
            print(f"   ✅ Stored memory: {memory.content[:50]}...")
        
        # Retrieve relevant memories for context
        context = AgentContext(
            agent_id=agent_id,
            session_id=session_id,
            query="order status ORD-12345",
            max_memories=10
        )
        
        retrieved_memories = langchain.retrieve_memories(context)
        print(f"\n   📝 Retrieved {len(retrieved_memories)} relevant memories:")
        for i, mem in enumerate(retrieved_memories, 1):
            print(f"      {i}. [{mem.memory_type}] {mem.content}")
        
        # Get memory statistics
        stats = langchain.get_memory_stats(agent_id)
        print(f"\n   📊 Memory Statistics:")
        print(f"      Total memories: {stats['total_memories']}")
        print(f"      Memory types: {stats['memory_types']}")
        print(f"      Unique sessions: {stats['unique_sessions']}")
        
    except Exception as e:
        print(f"   ❌ LangChain example failed: {e}")
        import traceback
        traceback.print_exc()


def llamaindex_example():
    """Example of using Memorizer with LlamaIndex."""
    print("\n🦙 LlamaIndex Integration Example")
    print("=" * 40)
    
    try:
        from memorizer.core.framework import create_framework
        from framework.core.config import load_config
        from agent_integrations import AgentIntegrationManager, AgentMemory, AgentContext
        
        # Initialize framework
        config = load_config("memorizer.yaml")
        framework = create_framework(config)
        memory_manager = framework.get_memory_manager()
        integration_manager = AgentIntegrationManager(memory_manager)
        
        # Get LlamaIndex integration
        llamaindex = integration_manager.get_integration("llamaindex")
        
        # Simulate document processing with LlamaIndex
        agent_id = "document_processor"
        session_id = "doc_session_001"
        
        # Store document processing memories
        document_memories = [
            AgentMemory(
                agent_id=agent_id,
                session_id=session_id,
                content="Processed document: AI Research Paper - 'Attention Is All You Need'",
                metadata={
                    "document_type": "research_paper",
                    "title": "Attention Is All You Need",
                    "authors": ["Vaswani et al."],
                    "pages": 15,
                    "topics": ["transformer", "attention", "neural_networks"]
                },
                timestamp=datetime.now(),
                memory_type="document_processing"
            ),
            AgentMemory(
                agent_id=agent_id,
                session_id=session_id,
                content="Extracted key concepts: self-attention, multi-head attention, positional encoding",
                metadata={
                    "extraction_type": "key_concepts",
                    "concepts": ["self-attention", "multi-head attention", "positional encoding"]
                },
                timestamp=datetime.now(),
                memory_type="concept_extraction"
            ),
            AgentMemory(
                agent_id=agent_id,
                session_id=session_id,
                content="Generated summary: Paper introduces Transformer architecture based on attention mechanisms",
                metadata={
                    "summary_length": "short",
                    "main_topic": "transformer_architecture"
                },
                timestamp=datetime.now(),
                memory_type="summary_generation"
            )
        ]
        
        # Store all memories
        for memory in document_memories:
            memory_id = llamaindex.store_memory(memory)
            print(f"   ✅ Stored memory: {memory.content[:50]}...")
        
        # Retrieve relevant memories for context
        context = AgentContext(
            agent_id=agent_id,
            session_id=session_id,
            query="transformer attention mechanisms",
            max_memories=10
        )
        
        retrieved_memories = llamaindex.retrieve_memories(context)
        print(f"\n   📝 Retrieved {len(retrieved_memories)} relevant memories:")
        for i, mem in enumerate(retrieved_memories, 1):
            print(f"      {i}. [{mem.memory_type}] {mem.content}")
        
        # Get memory statistics
        stats = llamaindex.get_memory_stats(agent_id)
        print(f"\n   📊 Memory Statistics:")
        print(f"      Total memories: {stats['total_memories']}")
        print(f"      Memory types: {stats['memory_types']}")
        print(f"      Unique sessions: {stats['unique_sessions']}")
        
    except Exception as e:
        print(f"   ❌ LlamaIndex example failed: {e}")
        import traceback
        traceback.print_exc()


def autogpt_example():
    """Example of using Memorizer with AutoGPT."""
    print("\n🤖 AutoGPT Integration Example")
    print("=" * 40)
    
    try:
        from memorizer.core.framework import create_framework
        from framework.core.config import load_config
        from agent_integrations import AgentIntegrationManager, AgentMemory, AgentContext
        
        # Initialize framework
        config = load_config("memorizer.yaml")
        framework = create_framework(config)
        memory_manager = framework.get_memory_manager()
        integration_manager = AgentIntegrationManager(memory_manager)
        
        # Get AutoGPT integration
        autogpt = integration_manager.get_integration("autogpt")
        
        # Simulate AutoGPT task execution
        agent_id = "autogpt_assistant"
        session_id = "task_session_001"
        
        # Store task execution memories
        task_memories = [
            AgentMemory(
                agent_id=agent_id,
                session_id=session_id,
                content="Task: Research latest AI developments",
                metadata={
                    "task_type": "research",
                    "priority": "high",
                    "deadline": "2024-01-15"
                },
                timestamp=datetime.now(),
                memory_type="task_assignment"
            ),
            AgentMemory(
                agent_id=agent_id,
                session_id=session_id,
                content="Action: Searched Google for 'AI developments 2024'",
                metadata={
                    "action_type": "web_search",
                    "query": "AI developments 2024",
                    "results_count": 50
                },
                timestamp=datetime.now(),
                memory_type="action_execution"
            ),
            AgentMemory(
                agent_id=agent_id,
                session_id=session_id,
                content="Result: Found 3 relevant articles about GPT-4, DALL-E 3, and ChatGPT updates",
                metadata={
                    "result_type": "research_findings",
                    "articles_found": 3,
                    "topics": ["GPT-4", "DALL-E 3", "ChatGPT"]
                },
                timestamp=datetime.now(),
                memory_type="task_result"
            ),
            AgentMemory(
                agent_id=agent_id,
                session_id=session_id,
                content="Decision: Compile findings into a summary report",
                metadata={
                    "decision_type": "next_action",
                    "reasoning": "User requested research summary"
                },
                timestamp=datetime.now(),
                memory_type="decision_making"
            )
        ]
        
        # Store all memories
        for memory in task_memories:
            memory_id = autogpt.store_memory(memory)
            print(f"   ✅ Stored memory: {memory.content[:50]}...")
        
        # Retrieve relevant memories for context
        context = AgentContext(
            agent_id=agent_id,
            session_id=session_id,
            query="AI developments research findings",
            max_memories=10
        )
        
        retrieved_memories = autogpt.retrieve_memories(context)
        print(f"\n   📝 Retrieved {len(retrieved_memories)} relevant memories:")
        for i, mem in enumerate(retrieved_memories, 1):
            print(f"      {i}. [{mem.memory_type}] {mem.content}")
        
        # Get memory statistics
        stats = autogpt.get_memory_stats(agent_id)
        print(f"\n   📊 Memory Statistics:")
        print(f"      Total memories: {stats['total_memories']}")
        print(f"      Memory types: {stats['memory_types']}")
        print(f"      Unique sessions: {stats['unique_sessions']}")
        
    except Exception as e:
        print(f"   ❌ AutoGPT example failed: {e}")
        import traceback
        traceback.print_exc()


def crewai_example():
    """Example of using Memorizer with CrewAI."""
    print("\n👥 CrewAI Integration Example")
    print("=" * 40)
    
    try:
        from memorizer.core.framework import create_framework
        from framework.core.config import load_config
        from agent_integrations import AgentIntegrationManager, AgentMemory, AgentContext
        
        # Initialize framework
        config = load_config("memorizer.yaml")
        framework = create_framework(config)
        memory_manager = framework.get_memory_manager()
        integration_manager = AgentIntegrationManager(memory_manager)
        
        # Get CrewAI integration
        crewai = integration_manager.get_integration("crewai")
        
        # Simulate CrewAI multi-agent collaboration
        agent_id = "marketing_crew"
        session_id = "campaign_session_001"
        
        # Store crew collaboration memories
        crew_memories = [
            AgentMemory(
                agent_id=agent_id,
                session_id=session_id,
                content="Research Agent: Analyzed market trends for Q1 2024",
                metadata={
                    "crew_role": "research_agent",
                    "task": "market_analysis",
                    "findings": "Growing interest in AI-powered tools"
                },
                timestamp=datetime.now(),
                memory_type="crew_task"
            ),
            AgentMemory(
                agent_id=agent_id,
                session_id=session_id,
                content="Content Agent: Created 5 blog post ideas based on research",
                metadata={
                    "crew_role": "content_agent",
                    "deliverable": "blog_ideas",
                    "count": 5
                },
                timestamp=datetime.now(),
                memory_type="crew_task"
            ),
            AgentMemory(
                agent_id=agent_id,
                session_id=session_id,
                content="Strategy Agent: Recommended social media campaign for AI tools",
                metadata={
                    "crew_role": "strategy_agent",
                    "recommendation": "social_media_campaign",
                    "platforms": ["LinkedIn", "Twitter", "Facebook"]
                },
                timestamp=datetime.now(),
                memory_type="crew_task"
            ),
            AgentMemory(
                agent_id=agent_id,
                session_id=session_id,
                content="Crew Coordinator: All agents completed their tasks successfully",
                metadata={
                    "crew_role": "coordinator",
                    "status": "completed",
                    "next_steps": "review_and_approve"
                },
                timestamp=datetime.now(),
                memory_type="crew_coordination"
            )
        ]
        
        # Store all memories
        for memory in crew_memories:
            memory_id = crewai.store_memory(memory)
            print(f"   ✅ Stored memory: {memory.content[:50]}...")
        
        # Retrieve relevant memories for context
        context = AgentContext(
            agent_id=agent_id,
            session_id=session_id,
            query="AI tools marketing campaign",
            max_memories=10
        )
        
        retrieved_memories = crewai.retrieve_memories(context)
        print(f"\n   📝 Retrieved {len(retrieved_memories)} relevant memories:")
        for i, mem in enumerate(retrieved_memories, 1):
            print(f"      {i}. [{mem.memory_type}] {mem.content}")
        
        # Get memory statistics
        stats = crewai.get_memory_stats(agent_id)
        print(f"\n   📊 Memory Statistics:")
        print(f"      Total memories: {stats['total_memories']}")
        print(f"      Memory types: {stats['memory_types']}")
        print(f"      Unique sessions: {stats['unique_sessions']}")
        
    except Exception as e:
        print(f"   ❌ CrewAI example failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all AI integration examples."""
    print("🚀 AI Framework Integration Examples")
    print("=" * 60)
    
    # Run all examples
    langchain_example()
    llamaindex_example()
    autogpt_example()
    crewai_example()
    
    print("\n🎯 All AI Integration Examples Completed!")
    print("\n📚 Key Features Demonstrated:")
    print("   ✅ Framework-specific memory storage")
    print("   ✅ Rich metadata support")
    print("   ✅ Memory type classification")
    print("   ✅ Session-based organization")
    print("   ✅ Contextual memory retrieval")
    print("   ✅ Memory statistics and analytics")
    print("   ✅ Integration with memory lifecycle")
    
    print("\n🔧 Usage Patterns:")
    print("   - LangChain: Conversation and chat memory")
    print("   - LlamaIndex: Document processing and knowledge extraction")
    print("   - AutoGPT: Task execution and decision tracking")
    print("   - CrewAI: Multi-agent collaboration and coordination")


if __name__ == "__main__":
    main()
