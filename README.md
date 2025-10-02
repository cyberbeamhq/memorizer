# Memorizer - Intelligent Memory Management Framework

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

**A production-ready Python framework for intelligent memory management in AI applications.**

Memorizer provides advanced memory lifecycle management, intelligent compression, hybrid retrieval, and seamless integration with LangChain, Supabase, and major vector databases.

## 🎯 Core Philosophy

**Pure Memory Management** - Focus exclusively on intelligent memory lifecycle, compression policies, and retrieval. No web servers, no containers—just powerful memory management for your AI applications.

## ✨ Key Features

### Memory Lifecycle Management
- **🔄 Three-Tier Aging System**: Automatic transitions from `very_new` → `mid_term` → `long_term`
- **🗜️ Intelligent Compression**: Age-based, size-based, access-based, and tier-based policies
- **⚙️ Adaptive Algorithms**: GZIP, ZLIB, Semantic, and Adaptive compression
- **📊 Lifecycle Automation**: Background compression with retry logic and monitoring

### Search & Retrieval
- **🔍 Hybrid Retrieval**: TF-IDF + keyword matching + phrase detection + semantic search
- **📈 Relevance Scoring**: Multi-factor ranking with recency, frequency, and context
- **🎯 Context-Aware**: Session-based and agent-based memory filtering
- **⚡ Optimized Performance**: Caching with TTL using `cachetools`

### Security & Privacy
- **🛡️ PII Detection**: Microsoft Presidio integration for production-grade detection
- **🔒 Multi-Tenant Isolation**: Row-level security (RLS) with Supabase auth
- **🔐 Secure Storage**: Encrypted connections and credential management
- **✅ Input Validation**: Comprehensive validation for all user inputs

### Integrations
- **🤖 LangChain**: Three integration methods (Memory, ChatHistory, Callback)
- **🗄️ Supabase**: PostgreSQL + pgvector + Edge Functions + RLS
- **🧠 LLM Providers**: OpenAI, Anthropic, Groq with provider abstraction
- **📊 Vector Stores**: Official SDKs for Pinecone, Weaviate, ChromaDB

## 🚀 Quick Start

### Installation

```bash
# Core framework
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt

# Optional: Vector database SDKs
pip install pinecone-client>=2.2.0 weaviate-client>=3.0.0 chromadb>=0.4.0

# Optional: Enhanced security
pip install presidio-analyzer>=2.2.0 presidio-anonymizer>=2.2.0
```

### Basic Usage

```python
from memorizer.core.framework import MemorizerFramework
from memorizer.core.interfaces import Memory, Query

# Initialize framework
framework = MemorizerFramework()
memory_manager = framework.get_memory_manager()

# Store a memory
memory = Memory(
    user_id="user_123",
    content="I love playing guitar on weekends",
    metadata={"session_id": "session_456", "source": "chat"}
)
memory_id = memory_manager.store_memory(memory)

# Search memories
query = Query(
    user_id="user_123",
    content="music hobbies",
    metadata={"session_id": "session_456"}
)
results = memory_manager.search_memories(query, limit=5)

for memory in results.memories:
    print(f"Score: {memory.relevance_score:.2f} - {memory.content}")
```

### LangChain Integration

Memorizer provides three methods to integrate with LangChain:

#### Method 1: Memory Interface

```python
from memorizer.integrations.langchain_integration import LangChainMemorizerMemory
from langchain.agents import AgentExecutor

# Create LangChain memory backed by Memorizer
langchain_memory = LangChainMemorizerMemory(
    memorizer_framework=framework,
    user_id="user_123",
    session_id="session_456",
    memory_key="chat_history",
    return_messages=True
)

# Use in your agent
agent_executor = AgentExecutor(
    agent=your_agent,
    memory=langchain_memory,
    verbose=True
)
```

#### Method 2: Chat History

```python
from memorizer.integrations.langchain_integration import LangChainMemorizerChatHistory
from langchain.memory import ConversationBufferMemory

# Create chat history
chat_history = LangChainMemorizerChatHistory(
    memorizer_framework=framework,
    user_id="user_123",
    session_id="session_456"
)

# Use with conversation memory
memory = ConversationBufferMemory(
    chat_memory=chat_history,
    return_messages=True
)
```

#### Method 3: Callback Handler

```python
from memorizer.integrations.langchain_integration import LangChainMemorizerCallback

# Create callback for automatic memory capture
callback = LangChainMemorizerCallback(
    memorizer_framework=framework,
    user_id="user_123",
    session_id="session_456",
    capture_tool_calls=True,
    capture_errors=True
)

# Use in LLM calls
response = llm.invoke(prompt, callbacks=[callback])
```

See [examples/langchain_agent_example.py](examples/langchain_agent_example.py) and [examples/quick_start_langchain.md](examples/quick_start_langchain.md) for complete examples.

### Supabase Integration

Memorizer includes production-ready Supabase integration with RLS, pgvector, and Edge Functions.

```python
from memorizer.integrations.supabase_client import SupabaseMemoryManager

# Initialize with Supabase
memory_manager = SupabaseMemoryManager(
    supabase_url="https://your-project.supabase.co",
    supabase_key="your-anon-key",
    user_token="user-jwt-token"  # From Supabase Auth
)

# Store memory (RLS automatically enforces user_id)
memory_id = memory_manager.store_memory(
    content="Working on a React project",
    metadata={"project": "frontend", "priority": "high"},
    tier="very_new"
)

# Search memories
results = memory_manager.search_memories(
    query="frontend development",
    limit=10
)
```

**Setup Instructions**: See [SUPABASE_SETUP.md](SUPABASE_SETUP.md) for complete migration scripts, RLS policies, and Edge Function deployment.

## 🏗️ Architecture

### Memory Lifecycle

```
Input Memory
     ↓
┌─────────────┐
│  very_new   │  Age: 0-7 days
│  Raw text   │  Compression: None
└──────┬──────┘
       ↓ Transition Policy: Age > 7 days OR Size > threshold
┌─────────────┐
│  mid_term   │  Age: 7-30 days
│ Summarized  │  Compression: GZIP/Semantic
└──────┬──────┘
       ↓ Transition Policy: Age > 30 days OR Access < threshold
┌─────────────┐
│  long_term  │  Age: 30+ days
│ Compressed  │  Compression: Adaptive/Semantic
└─────────────┘
```

### Compression Policies

- **Age-Based**: Compress memories older than threshold
- **Size-Based**: Compress when total size exceeds limit
- **Access-Based**: Compress rarely accessed memories
- **Tier-Based**: Different compression algorithms per tier

### Hybrid Retrieval System

```python
# Combines multiple retrieval strategies
1. TF-IDF Scoring: Term frequency-inverse document frequency
2. Keyword Matching: Exact and partial keyword overlap
3. Phrase Detection: Multi-word phrase matching
4. Vector Similarity: Semantic search via embeddings
5. Recency Weighting: Boost recent memories
6. Context Filtering: Session/agent-based filtering
```

## 📖 Core Components

```
src/memorizer/
├── core/
│   ├── framework.py           # Main framework orchestration
│   ├── interfaces.py          # Core data models (Memory, Query, etc.)
│   └── simple_config.py       # Configuration management
├── memory/
│   ├── manager.py             # Memory lifecycle manager
│   └── compression_agent.py   # LLM-based compression
├── lifecycle/
│   ├── compression_policies.py # Policy engine for compression
│   ├── lifecycle_manager.py    # Tier transition management
│   └── tier_management.py      # Memory tier operations
├── retrieval/
│   └── retrieval.py           # Hybrid search implementation
├── storage/
│   ├── db.py                  # PostgreSQL storage with thread safety
│   └── vector_db.py           # Vector storage abstraction
├── security/
│   └── pii_detection.py       # Presidio-based PII filtering
├── integrations/
│   ├── langchain_integration.py  # LangChain adapters
│   ├── supabase_client.py        # Supabase integration
│   ├── llm_providers.py          # LLM provider abstraction
│   └── embeddings.py             # Embedding generation with caching
└── utils/
    ├── validation.py          # Input validation
    ├── errors.py              # Error handling framework
    └── utils.py               # Utility functions
```

## 🔧 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/memorizer
REDIS_URL=redis://localhost:6379/0  # Optional: for caching

# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...

# Vector Stores
PINECONE_API_KEY=your-api-key
PINECONE_ENVIRONMENT=us-west1-gcp
WEAVIATE_URL=http://localhost:8080
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key  # For Edge Functions

# Security
JWT_SECRET_KEY=your-secret-key-at-least-32-chars
EMBEDDING_PROVIDER=openai  # or anthropic, mock
VECTOR_DB_PROVIDER=pinecone  # or weaviate, chroma, mock
ENVIRONMENT=production  # or development, test
```

### YAML Configuration

Create `memorizer.yaml` for advanced configuration:

```yaml
memory:
  tiers:
    very_new:
      max_age_days: 7
      compression: none
    mid_term:
      max_age_days: 30
      compression: gzip
    long_term:
      max_age_days: 365
      compression: adaptive

compression:
  policies:
    - type: age_based
      threshold_days: 7
      target_tier: mid_term
    - type: size_based
      threshold_mb: 100
      compression_algorithm: gzip

retrieval:
  hybrid:
    tfidf_weight: 0.3
    keyword_weight: 0.2
    vector_weight: 0.5
  max_results: 50
  min_relevance_score: 0.1
```

## 🧪 Testing

```bash
# Run basic tests (no external dependencies)
pytest tests/ -v -m "not slow"

# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_basic.py -v

# Run with different Python versions
tox
```

## 📊 Supported Providers

### LLM Providers
- **OpenAI**: GPT-3.5, GPT-4, GPT-4-turbo for semantic compression
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku) for summarization
- **Groq**: Fast inference with Llama, Mixtral models
- **Mock**: Testing and development without API costs

### Vector Databases
- **Pinecone**: Managed vector database with official SDK
- **Weaviate**: Open-source vector search with official client
- **ChromaDB**: Embedding database with local/cloud options
- **PostgreSQL + pgvector**: Self-hosted with 1536-dimension embeddings

### Storage Backends
- **Supabase**: PostgreSQL + Auth + RLS + Edge Functions
- **PostgreSQL**: Direct connection with thread-safe pooling
- **SQLite**: Local development and testing
- **Mock**: In-memory for testing

## 🎯 Use Cases

### AI Chatbots with Long-Term Memory
```python
# Store conversation context with automatic lifecycle
memory_manager.store_memory(Memory(
    user_id="user_123",
    content="User prefers morning meetings at 9 AM",
    metadata={"category": "preferences", "session_id": "chat_001"}
))

# Later, retrieve context for personalization
query = Query(user_id="user_123", content="schedule meeting")
context = memory_manager.search_memories(query, limit=5)
```

### RAG Applications with Intelligent Context
```python
# Store document chunks with metadata
for chunk in document_chunks:
    memory_manager.store_memory(Memory(
        user_id="doc_processor",
        content=chunk.text,
        metadata={"document_id": doc.id, "page": chunk.page}
    ))

# Retrieve relevant chunks for generation
query = Query(user_id="doc_processor", content=user_question)
relevant_chunks = memory_manager.search_memories(query, limit=10)
```

### Multi-Agent Systems
```python
# Each agent has isolated memory with shared access
agent_memory = Memory(
    user_id="agent_researcher",
    content="Found relevant paper on transformers",
    metadata={"agent_type": "researcher", "shared": True}
)

# Query across agent memories
query = Query(
    user_id="agent_writer",
    content="recent research findings",
    metadata={"shared": True}  # Access shared memories
)
```

## 🆚 Why Memorizer?

### vs. Building Your Own
✅ **Proven Lifecycle Algorithms**: Battle-tested three-tier aging system
✅ **Production Libraries**: Presidio (PII), cachetools (LRU), official SDKs
✅ **Thread-Safe**: Race condition protection in connection pooling
✅ **Hybrid Retrieval**: TF-IDF + keyword + vector + recency scoring

### vs. LangChain Memory Alone
✅ **Intelligent Lifecycle**: Automatic tier transitions and compression
✅ **Production Storage**: Supabase with RLS and pgvector integration
✅ **Advanced Retrieval**: Multi-factor relevance scoring
✅ **Drop-In Replacement**: Compatible with LangChain Memory API

### vs. Vector Databases Directly
✅ **Higher Abstraction**: Memory lifecycle, not just vector storage
✅ **Hybrid Search**: Combines keyword + semantic search
✅ **Automatic Compression**: LLM-based summarization and compression
✅ **Multi-Provider**: Easy switching between Pinecone, Weaviate, ChromaDB

## 📁 Project Structure

```
memorizer/
├── src/memorizer/          # Main framework code
├── examples/               # Usage examples and tutorials
│   ├── langchain_agent_example.py
│   └── quick_start_langchain.md
├── tests/                  # Test suite
│   ├── test_basic.py
│   └── test_integration.py
├── supabase/              # Supabase integration
│   ├── migrations/        # Database migrations
│   └── functions/         # Edge Functions
├── docs/                  # Documentation
├── .github/workflows/     # CI/CD pipeline
├── requirements.txt       # Core dependencies
├── requirements-dev.txt   # Development dependencies
├── SUPABASE_SETUP.md     # Supabase setup guide
├── INTEGRATION_GUIDE.md  # Integration documentation
└── README.md             # This file
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/memorizer.git
cd memorizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run linting
flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
black --check src/
mypy src/ --ignore-missing-imports
```

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## 🔗 Resources

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Supabase Setup**: [SUPABASE_SETUP.md](SUPABASE_SETUP.md)
- **Integration Guide**: [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- **Issue Tracker**: [GitHub Issues](https://github.com/yourusername/memorizer/issues)

---

**Get started in 30 seconds:**

```bash
pip install -r requirements.txt
```

```python
from memorizer.core.framework import MemorizerFramework
from memorizer.core.interfaces import Memory, Query

framework = MemorizerFramework()
manager = framework.get_memory_manager()

# Store a memory
memory_id = manager.store_memory(Memory(
    user_id="user_123",
    content="I love coffee in the morning"
))

# Search memories
results = manager.search_memories(Query(
    user_id="user_123",
    content="beverages"
))

print(results.memories[0].content)  # "I love coffee in the morning"
```

**Built with ❤️ for the AI community**
