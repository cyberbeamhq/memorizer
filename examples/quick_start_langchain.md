# Quick Start: LangChain + Memorizer Integration

Connect your LangChain agents to Memorizer for intelligent, persistent memory management.

## 🚀 Quick Example (3 Steps)

### 1. Install Dependencies

```bash
pip install memorizer langchain langchain-openai
```

### 2. Basic Integration

```python
from memorizer import create_memory_manager
from memorizer.core.simple_config import MemoryConfig
from memorizer.integrations.langchain_integration import LangChainMemorizerMemory

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Initialize Memorizer
config = MemoryConfig()
memory_manager = create_memory_manager(config)

# Create LangChain memory backed by Memorizer
langchain_memory = LangChainMemorizerMemory(
    memorizer_framework=None,  # Using simple manager
    user_id="user_123",
    session_id="session_456",
    memory_key="chat_history",
    return_messages=True
)
langchain_memory.memory_manager = memory_manager

# Create your LangChain agent
llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with memory."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools=[], prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[],
    memory=langchain_memory,
    verbose=True
)

# Use the agent - memory is automatically managed!
response = agent_executor.invoke({"input": "My name is Alice"})
response = agent_executor.invoke({"input": "What's my name?"})
# Agent remembers: "Your name is Alice!"
```

### 3. Run Example

```bash
export OPENAI_API_KEY='your-api-key'
python examples/langchain_agent_example.py
```

---

## 🎯 Three Integration Methods

### Method 1: Memory Interface (Recommended)

Use `LangChainMemorizerMemory` for automatic context management:

```python
from memorizer.integrations.langchain_integration import LangChainMemorizerMemory

memory = LangChainMemorizerMemory(
    memorizer_framework=your_framework,
    user_id="user_123",
    session_id="session_456",
    return_messages=True  # Return as LangChain messages
)

# Add to agent
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)
```

**Benefits:**
- ✅ Automatic memory loading/saving
- ✅ Works with any LangChain chain/agent
- ✅ Retrieves relevant context automatically

### Method 2: Chat History

Use `LangChainMemorizerChatHistory` for chat-based applications:

```python
from memorizer.integrations.langchain_integration import LangChainMemorizerChatHistory

chat_history = LangChainMemorizerChatHistory(
    memorizer_framework=your_framework,
    user_id="user_123",
    session_id="chat_session"
)

# Use with conversation memory
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    chat_memory=chat_history,
    return_messages=True
)
```

**Benefits:**
- ✅ Perfect for chatbots
- ✅ Persistent across sessions
- ✅ Searchable conversation history

### Method 3: Callback Handler

Use `LangChainMemorizerCallback` for automatic event capture:

```python
from memorizer.integrations.langchain_integration import LangChainMemorizerCallback

callback = LangChainMemorizerCallback(
    memorizer_framework=your_framework,
    user_id="user_123",
    session_id="session_456",
    capture_tool_calls=True,  # Capture tool usage
    capture_errors=True       # Capture errors
)

# Add to agent
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[callback]
)
```

**Benefits:**
- ✅ Captures all LLM calls
- ✅ Tracks tool usage
- ✅ Logs errors
- ✅ Great for debugging/monitoring

---

## 🔥 Advanced Features

### Tiered Memory Lifecycle

Memorizer automatically manages memory tiers:

```python
# New memories start as "very_new"
memory_manager.store_memory(user_id, "Recent conversation", tier="very_new")

# Older memories transition to "mid_term" (compressed)
# Eventually move to "long_term" (highly compressed)

# Retrieval automatically searches across all tiers
results = memory_manager.search_memories(user_id, "conversation")
```

### Intelligent Compression

```python
from memorizer.lifecycle.compression_policies import CompressionPolicyManager

# Compression happens automatically based on policies:
# - Age-based: Compress memories older than 1 week
# - Size-based: Compress large memories (>10KB)
# - Access-based: Compress rarely accessed memories
# - Tier-based: Different compression for each tier

policy_manager = CompressionPolicyManager()
stats = policy_manager.get_compression_stats()
print(f"Active policies: {stats['total_policies']}")
```

### Multi-Session Search

```python
# Search across all sessions for a user
all_results = memory_manager.search_memories(
    user_id="user_123",
    query="machine learning",
    limit=10
)

# Filter by session
session_results = [
    m for m in all_results.memories
    if m.metadata.get('session_id') == 'session_456'
]
```

---

## 📊 What You Get

| Feature | Memorizer | Standard LangChain Memory |
|---------|-----------|---------------------------|
| **Persistent storage** | ✅ PostgreSQL/SQLite | ❌ In-memory only |
| **Tiered lifecycle** | ✅ very_new → mid_term → long_term | ❌ No tiers |
| **Automatic compression** | ✅ Multiple algorithms | ❌ No compression |
| **Semantic search** | ✅ Vector + keyword + TF-IDF | ⚠️ Simple only |
| **Multi-session** | ✅ Search across sessions | ❌ Session-scoped |
| **PII detection** | ✅ Presidio integration | ❌ No built-in |
| **Monitoring** | ✅ Metrics + callbacks | ⚠️ Basic logging |

---

## 🛠️ Configuration

### Simple Config (In-Memory)

```python
from memorizer.core.simple_config import MemoryConfig

config = MemoryConfig()  # Uses in-memory storage
manager = create_memory_manager(config)
```

### Production Config (PostgreSQL)

```python
from memorizer.core.config import ConfigManager

config = ConfigManager()
config.set_database_config({
    "url": "postgresql://user:pass@localhost/memorizer",
    "pool_size": 10
})
config.set_vector_store_config({
    "provider": "pinecone",
    "api_key": "your-pinecone-key",
    "environment": "us-west1-gcp"
})

framework = create_memorizer_framework(config)
memory_manager = framework.get_memory_manager()
```

---

## 🧪 Testing Your Integration

```python
# 1. Store some memories
manager.store_memory("user1", "I love Python programming", {"type": "preference"})
manager.store_memory("user1", "My name is Alice", {"type": "personal"})

# 2. Search memories
results = manager.search_memories("user1", "Python", limit=5)
assert results.total_found > 0

# 3. Check stats
stats = manager.get_stats("user1")
assert stats.total_memories == 2

# 4. Use with LangChain
response = agent_executor.invoke({"input": "What do I love?"})
assert "Python" in response['output']
```

---

## 📚 More Examples

- **Full example**: `examples/langchain_agent_example.py`
- **CrewAI integration**: `examples/crewai_integration_example.py`
- **LlamaIndex integration**: `examples/llamaindex_integration_example.py`

---

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'langchain'"

```bash
pip install langchain langchain-openai
```

### "Memory not persisting between runs"

Make sure you're using the same `user_id` and `session_id`:

```python
# ✅ Correct
memory = LangChainMemorizerMemory(
    user_id="user_123",  # Same user
    session_id="session_456"  # Same session
)

# ❌ Wrong - creates new session each time
memory = LangChainMemorizerMemory(
    user_id="user_123",
    session_id=f"session_{datetime.now()}"  # Different each run!
)
```

### "No memories retrieved"

Check that memories are actually stored:

```python
stats = memory_manager.get_stats("user_123")
print(f"Total memories: {stats.total_memories}")

# List all memories
all_mems = memory_manager.search_memories("user_123", "", limit=100)
for m in all_mems.memories:
    print(f"- {m.content}")
```

---

## 🎉 Ready to Build!

You now have:
- ✅ Persistent memory across sessions
- ✅ Intelligent tiered lifecycle
- ✅ Automatic compression
- ✅ Semantic search
- ✅ PII detection
- ✅ Production-ready storage

Start building agents with real memory! 🚀
