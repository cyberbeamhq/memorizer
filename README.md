# Memorizer - Intelligent Memory Management Library

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

**A focused Python library for adding intelligent memory capabilities to AI applications.**

Memorizer provides memory storage, retrieval, lifecycle management, and external provider integrations without the complexity of application infrastructure.

## 🎯 Core Focus

**Pure Memory Management** - No web servers, no containers, no infrastructure complexity. Just intelligent memory for your AI applications.

## ✨ Key Features

- **🧠 Intelligent Memory Lifecycle**: Three-tier aging system (very_new → mid_term → long_term)
- **🔍 Hybrid Search**: Combines keyword and vector search for optimal retrieval
- **🤖 LLM Integration**: Works with OpenAI, Anthropic, Groq, and custom providers
- **🗄️ External Providers**: Easy integration with Supabase, Railway, Neon, Pinecone, Weaviate
- **🛡️ PII Protection**: Built-in personally identifiable information filtering
- **⚡ Simple API**: Get started in 3 lines of code

## 🚀 Quick Start

### Installation

```bash
pip install memorizer

# Optional: Install external provider dependencies
pip install pinecone-client weaviate-client supabase
```

### Basic Usage

```python
import memorizer

# Create a memory manager
memory = memorizer.create_memory()

# Store memories
user_id = "user123"
memory_id = memory.store_memory(user_id, "I love playing guitar")

# Search memories
results = memory.search_memories(user_id, "music hobbies")
print(f"Found {len(results.memories)} relevant memories")
```

### With External Providers

```python
import memorizer

# Use Supabase for storage and Pinecone for vectors
memory = memorizer.create_memory_manager(
    storage_provider="supabase",
    vector_store="pinecone",
    llm_provider="openai",
    supabase_url="https://your-project.supabase.co",
    supabase_password="your-password",
    pinecone_api_key="your-api-key"
)

# Same API
memory.store_memory("user123", "Working on a React project")
results = memory.search_memories("user123", "frontend development")
```

## 🏗️ Architecture

### Memory Lifecycle

```
Input Memory
     ↓
[very_new] → [mid_term] → [long_term]
   7 days      30 days      365 days
     ↓           ↓            ↓
  Raw text → Summarized → Compressed
```

### Core Components

```
memorizer/
├── memory/           # Memory manager and lifecycle
├── core/            # Interfaces and configuration
├── builtins/        # Storage, retrievers, summarizers
├── storage/         # Database and external providers
├── retrieval/       # Hybrid search algorithms
├── security/        # PII detection and filtering
└── integrations/    # LLM and agent integrations
```

## 📖 Supported Providers

### Database Providers
- **Memory**: In-memory storage (development)
- **Supabase**: PostgreSQL with real-time features
- **Railway**: Simple PostgreSQL hosting
- **Neon**: Serverless PostgreSQL
- **PostgreSQL**: Direct PostgreSQL connection

### Vector Stores
- **Memory**: In-memory vectors (development)
- **SQLite**: Local vector storage
- **Pinecone**: Managed vector database
- **Weaviate**: Open-source vector search
- **Chroma**: Embedding database

### LLM Providers
- **OpenAI**: GPT models for summarization
- **Anthropic**: Claude models
- **Groq**: Fast inference
- **Mock**: For testing/development

## 🔧 External Provider Setup

### Supabase Configuration

```python
import memorizer

memory = memorizer.create_memory_manager(
    storage_provider="supabase",
    supabase_url="https://your-project.supabase.co",
    supabase_password="your-database-password"
)
```

**Required Environment Variables:**
```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_DB_PASSWORD=your_database_password
```

### Pinecone Configuration

```python
import memorizer

memory = memorizer.create_memory_manager(
    vector_store="pinecone",
    pinecone_api_key="your-api-key",
    pinecone_environment="us-west1-gcp"
)
```

**Required Environment Variables:**
```bash
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=us-west1-gcp
```

### Complete External Setup

```python
import memorizer

# Production setup with Supabase + Pinecone + OpenAI
memory = memorizer.create_memory_manager(
    storage_provider="supabase",
    vector_store="pinecone",
    llm_provider="openai",
    supabase_url="https://your-project.supabase.co",
    supabase_password="your-password",
    pinecone_api_key="your-pinecone-key",
    openai_api_key="your-openai-key"
)
```

## 🧪 Examples

### Basic Memory Operations

```python
import memorizer

memory = memorizer.create_memory()
user_id = "demo_user"

# Store different types of memories
memories = [
    "I prefer morning workouts at 6 AM",
    "My favorite cuisine is Italian food",
    "Working on a Python ML project",
    "Planning a trip to Japan next year"
]

for content in memories:
    memory.store_memory(user_id, content)

# Search with different queries
queries = ["exercise habits", "food preferences", "programming work"]

for query in queries:
    results = memory.search_memories(user_id, query, limit=2)
    print(f"'{query}' found {len(results.memories)} memories")
```

### Integration with LangChain

```python
import memorizer
from langchain.memory import ConversationBufferMemory

# Create memory manager
memory_manager = memorizer.create_memory_manager(
    storage_provider="supabase",
    llm_provider="openai"
)

# Custom LangChain memory class
class MemorizerMemory(ConversationBufferMemory):
    def __init__(self, user_id: str, **kwargs):
        super().__init__(**kwargs)
        self.user_id = user_id
        self.memory_manager = memory_manager

    def save_context(self, inputs: dict, outputs: dict):
        # Save to both LangChain and Memorizer
        super().save_context(inputs, outputs)

        # Store in Memorizer for long-term memory
        conversation = f"User: {inputs.get('input', '')}\nAI: {outputs.get('output', '')}"
        self.memory_manager.store_memory(self.user_id, conversation)
```

### Custom AI Application

```python
import memorizer

class AIAssistant:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory = memorizer.create_memory_manager(
            storage_provider="supabase",
            vector_store="pinecone"
        )

    def process_message(self, message: str) -> str:
        # Retrieve relevant context
        context = self.memory.search_memories(
            self.user_id,
            message,
            limit=5
        )

        # Generate response (your LLM logic here)
        response = self.generate_response(message, context)

        # Store the interaction
        interaction = f"User: {message}\nAssistant: {response}"
        self.memory.store_memory(self.user_id, interaction)

        return response
```

## 📁 Project Structure

```
memorizer/
├── src/memorizer/          # Main library code
├── examples/               # Example scripts and tutorials
├── tests/                  # Test suite
├── docs/                   # Documentation
├── archive/                # Archived/legacy components
├── README.md              # This file
├── requirements.txt       # Core dependencies
├── .env.example          # Environment configuration template
├── memorizer.yaml.example # Advanced configuration template
└── setup.py              # Package setup
```

## 🎯 Perfect For

- 🤖 **AI Chatbots** with persistent memory
- 🔍 **RAG Applications** with intelligent context
- 📝 **AI Writing Assistants** that remember user preferences
- 🎮 **AI Game NPCs** with evolving personalities
- 📊 **Analytics Tools** that learn from interactions

## 🆚 Why Memorizer?

### vs. Building Your Own
- ✅ Proven memory lifecycle algorithms
- ✅ Battle-tested external provider integrations
- ✅ PII protection out of the box
- ✅ Optimized search algorithms

### vs. Other Memory Libraries
- ✅ **Focused**: Pure memory management, no infrastructure bloat
- ✅ **External Providers**: Easy integration with Supabase, Pinecone, etc.
- ✅ **Lifecycle Management**: Intelligent aging and compression
- ✅ **Simple API**: Get started in minutes, not hours

### vs. Vector Databases Directly
- ✅ **Higher Level**: Memory lifecycle, not just vector storage
- ✅ **Hybrid Search**: Combines keyword + vector search
- ✅ **LLM Integration**: Built-in summarization and compression
- ✅ **Multi-Provider**: Switch between Pinecone, Weaviate, etc. easily

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Get started in 30 seconds:**

```bash
pip install memorizer
```

```python
import memorizer
memory = memorizer.create_memory()
memory.store_memory("user1", "I love coffee")
results = memory.search_memories("user1", "beverages")
print(results.memories[0].content)  # "I love coffee"
```