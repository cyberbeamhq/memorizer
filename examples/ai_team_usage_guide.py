#!/usr/bin/env python3
"""
AI Team Usage Guide - How to integrate Memorizer into your AI applications
Shows common patterns that AI teams use when adopting open source libraries.
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import memorizer
import json
from typing import Dict, List, Any


def example_1_quick_start():
    """Example 1: 30-second integration - like adding any npm/pip package"""
    print("=" * 60)
    print("🚀 EXAMPLE 1: Quick Start (30 seconds)")
    print("=" * 60)

    # Just like using requests, pandas, or any other library
    memory = memorizer.create_memory()

    # Store user preferences and context
    user_id = "user_123"
    memory.store_memory(user_id, "User prefers detailed explanations")
    memory.store_memory(user_id, "Working on a React Native mobile app")
    memory.store_memory(user_id, "Uses TypeScript and prefers functional programming")

    # Retrieve relevant context for AI responses
    context = memory.search_memories(user_id, "programming preferences")
    print(f"✅ Found {len(context.memories)} relevant memories for context")

    for mem in context.memories:
        print(f"   📝 {mem.content}")


def example_2_chatbot_integration():
    """Example 2: Chatbot/Assistant Integration - common AI use case"""
    print("\n" + "=" * 60)
    print("🤖 EXAMPLE 2: AI Chatbot Integration")
    print("=" * 60)

    class AIAssistant:
        def __init__(self, user_id: str):
            self.user_id = user_id
            # Initialize memory just like you'd initialize any library
            self.memory = memorizer.create_memory()

        def chat(self, user_message: str) -> str:
            # 1. Get relevant context from memory
            context = self.memory.search_memories(self.user_id, user_message, limit=3)
            context_str = "\n".join([mem.content for mem in context.memories])

            # 2. Generate response (your AI logic here)
            response = self._generate_ai_response(user_message, context_str)

            # 3. Store the conversation for future context
            conversation = f"User: {user_message}\nAssistant: {response}"
            self.memory.store_memory(self.user_id, conversation)

            return response

        def _generate_ai_response(self, message: str, context: str) -> str:
            # Placeholder for your LLM integration (OpenAI, Anthropic, etc.)
            if context:
                return f"Based on our previous conversations about {context[:50]}..., here's my response to: {message}"
            return f"Here's my response to: {message}"

    # Usage - just like instantiating any class
    assistant = AIAssistant("user_456")

    # Simulate conversation
    response1 = assistant.chat("I'm learning Python")
    print(f"👤 User: I'm learning Python")
    print(f"🤖 AI: {response1}")

    response2 = assistant.chat("What's the best way to handle errors?")
    print(f"\n👤 User: What's the best way to handle errors?")
    print(f"🤖 AI: {response2}")


def example_3_production_setup():
    """Example 3: Production Setup with External Providers"""
    print("\n" + "=" * 60)
    print("🏢 EXAMPLE 3: Production Setup (External Providers)")
    print("=" * 60)

    # Production configuration - like configuring any database/service
    try:
        production_memory = memorizer.create_memory_manager(
            storage_provider="supabase",
            vector_store="pinecone",
            llm_provider="openai",
            # Configuration (would come from environment variables)
            supabase_url="https://your-project.supabase.co",
            supabase_password="your-password",
            pinecone_api_key="your-pinecone-key",
            openai_api_key="your-openai-key"
        )
        print("✅ Production memory manager configured")
        print("   📊 Storage: Supabase (PostgreSQL)")
        print("   🔍 Vectors: Pinecone")
        print("   🧠 LLM: OpenAI")
    except Exception as e:
        print(f"📝 Production setup example (requires actual credentials): {e}")


def example_4_rag_application():
    """Example 4: RAG Application - popular AI pattern"""
    print("\n" + "=" * 60)
    print("📚 EXAMPLE 4: RAG Application Integration")
    print("=" * 60)

    class DocumentRAG:
        def __init__(self):
            self.memory = memorizer.create_memory()
            self.system_user = "rag_system"

        def ingest_document(self, doc_content: str, doc_metadata: Dict[str, Any]):
            """Ingest documents into memory - like adding to any vector store"""
            # Split document into chunks (your chunking logic)
            chunks = self._chunk_document(doc_content)

            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **doc_metadata,
                    "chunk_index": i,
                    "doc_type": "knowledge_base"
                }
                memory_id = self.memory.store_memory(
                    self.system_user,
                    chunk,
                    metadata=chunk_metadata
                )
                print(f"   📄 Stored chunk {i}: {memory_id}")

        def query(self, question: str) -> Dict[str, Any]:
            """Query the knowledge base - like querying any search engine"""
            results = self.memory.search_memories(self.system_user, question, limit=5)

            # Format response with sources
            sources = []
            context = []
            for mem in results.memories:
                context.append(mem.content)
                sources.append({
                    "content": mem.content[:100] + "...",
                    "metadata": mem.metadata
                })

            return {
                "context": "\n".join(context),
                "sources": sources,
                "answer": f"Based on {len(sources)} sources: [Your LLM would generate answer here]"
            }

        def _chunk_document(self, content: str) -> List[str]:
            # Simple chunking - you'd use proper chunking logic
            words = content.split()
            chunk_size = 50
            return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    # Usage example
    rag = DocumentRAG()

    # Ingest some knowledge
    rag.ingest_document(
        "Python is a versatile programming language. It's great for data science, web development, and AI applications. Python has excellent libraries like pandas, numpy, and scikit-learn.",
        {"source": "python_guide.md", "category": "programming"}
    )

    # Query the knowledge base
    result = rag.query("What is Python good for?")
    print(f"❓ Query: What is Python good for?")
    print(f"💡 Answer: {result['answer']}")
    print(f"📚 Sources: {len(result['sources'])} documents")


def example_5_langchain_integration():
    """Example 5: Integration with existing AI frameworks"""
    print("\n" + "=" * 60)
    print("🔗 EXAMPLE 5: Integration with AI Frameworks (LangChain style)")
    print("=" * 60)

    # This shows how it would integrate with popular frameworks
    class MemorizerLangChainMemory:
        """Drop-in replacement for LangChain memory classes"""

        def __init__(self, user_id: str):
            self.user_id = user_id
            self.memory_manager = memorizer.create_memory()

        def save_context(self, inputs: dict, outputs: dict):
            """Save conversation context - LangChain interface"""
            conversation = f"Human: {inputs.get('input', '')}\nAI: {outputs.get('output', '')}"
            self.memory_manager.store_memory(self.user_id, conversation)

        def load_memory_variables(self, inputs: dict) -> dict:
            """Load relevant memory - LangChain interface"""
            query = inputs.get('input', '')
            results = self.memory_manager.search_memories(self.user_id, query, limit=3)

            history = []
            for mem in results.memories:
                history.append(mem.content)

            return {"history": "\n".join(history)}

    # Usage - drop-in replacement
    memory = MemorizerLangChainMemory("user_789")

    # Simulate LangChain-style usage
    inputs = {"input": "How do I deploy a Python app?"}
    outputs = {"output": "You can deploy Python apps using Docker, Heroku, or cloud services..."}

    memory.save_context(inputs, outputs)
    context = memory.load_memory_variables({"input": "deployment strategies"})

    print("✅ LangChain-style integration working")
    print(f"📝 Loaded context: {len(context['history'])} characters")


def example_6_environment_setup():
    """Example 6: How teams typically set up the library in their projects"""
    print("\n" + "=" * 60)
    print("⚙️  EXAMPLE 6: Team Setup & Configuration")
    print("=" * 60)

    print("1. Install the library:")
    print("   pip install memorizer")
    print("   # or for development:")
    print("   git clone https://github.com/your-org/memorizer")
    print("   pip install -e .")

    print("\n2. Environment configuration (.env file):")
    env_example = """
# Basic setup - works out of the box
MEMORIZER_STORAGE=memory
MEMORIZER_VECTOR_STORE=memory

# Production setup
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_DB_PASSWORD=your_password
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=us-west1-gcp
OPENAI_API_KEY=your_openai_key
"""
    print(env_example)

    print("3. Team configuration (memorizer.yaml):")
    config_example = {
        "development": {
            "storage_provider": "memory",
            "vector_store": "memory",
            "llm_provider": "mock"
        },
        "staging": {
            "storage_provider": "supabase",
            "vector_store": "pinecone",
            "llm_provider": "openai"
        },
        "production": {
            "storage_provider": "supabase",
            "vector_store": "pinecone",
            "llm_provider": "openai",
            "very_new_ttl_days": 7,
            "mid_term_ttl_days": 30,
            "compression_threshold": 0.8
        }
    }
    print(json.dumps(config_example, indent=2))


if __name__ == "__main__":
    print("🧠 MEMORIZER - AI Team Usage Guide")
    print("How to integrate Memorizer like any other open source library")

    # Run all examples
    example_1_quick_start()
    example_2_chatbot_integration()
    example_3_production_setup()
    example_4_rag_application()
    example_5_langchain_integration()
    example_6_environment_setup()

    print("\n" + "=" * 60)
    print("🎉 READY FOR AI TEAMS!")
    print("=" * 60)
    print("✅ Simple pip install")
    print("✅ 3-line integration")
    print("✅ Works with existing AI frameworks")
    print("✅ Scales from development to production")
    print("✅ Standard configuration patterns")
    print("✅ Drop-in memory for any AI application")