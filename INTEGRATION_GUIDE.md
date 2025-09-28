# 🚀 AI Team Integration Guide

**How to use Memorizer like any other open source library**

## Quick Start (30 seconds)

```bash
# 1. Install (like any Python package)
pip install memorizer

# 2. Use in your code (3 lines)
import memorizer
memory = memorizer.create_memory()
memory.store_memory("user123", "I prefer detailed explanations")
```

## 🎯 Common AI Team Use Cases

### 1. **AI Chatbot/Assistant**
```python
import memorizer

class MyChatbot:
    def __init__(self, user_id):
        self.user_id = user_id
        self.memory = memorizer.create_memory()

    def chat(self, message):
        # Get relevant context
        context = self.memory.search_memories(self.user_id, message)

        # Your AI logic here (OpenAI, Anthropic, etc.)
        response = generate_ai_response(message, context)

        # Remember the conversation
        self.memory.store_memory(self.user_id, f"User: {message}\nBot: {response}")
        return response

# Usage
bot = MyChatbot("user123")
response = bot.chat("Tell me about Python")
```

### 2. **RAG Application**
```python
import memorizer

class DocumentRAG:
    def __init__(self):
        self.memory = memorizer.create_memory()

    def add_documents(self, docs):
        for doc in docs:
            # Store document chunks
            chunks = chunk_document(doc)
            for chunk in chunks:
                self.memory.store_memory("knowledge_base", chunk,
                                       metadata={"source": doc.source})

    def query(self, question):
        # Find relevant context
        results = self.memory.search_memories("knowledge_base", question)
        context = "\n".join([r.content for r in results.memories])

        # Generate answer with your LLM
        return generate_answer(question, context)

# Usage
rag = DocumentRAG()
rag.add_documents(my_docs)
answer = rag.query("How does authentication work?")
```

### 3. **LangChain Integration**
```python
import memorizer
from langchain.memory import BaseMemory

class MemorizerLangChainMemory(BaseMemory):
    def __init__(self, user_id):
        self.user_id = user_id
        self.memory = memorizer.create_memory()

    def save_context(self, inputs, outputs):
        conversation = f"Human: {inputs['input']}\nAI: {outputs['output']}"
        self.memory.store_memory(self.user_id, conversation)

    def load_memory_variables(self, inputs):
        results = self.memory.search_memories(self.user_id, inputs['input'])
        return {"history": "\n".join([r.content for r in results.memories])}

# Drop-in replacement for LangChain memory
chain_memory = MemorizerLangChainMemory("user123")
```

### 4. **AI Agent with Memory**
```python
import memorizer

class AIAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.memory = memorizer.create_memory()

    def learn(self, experience):
        """Store new experiences"""
        self.memory.store_memory(self.agent_id, experience,
                               metadata={"type": "experience"})

    def recall(self, situation):
        """Recall relevant past experiences"""
        results = self.memory.search_memories(self.agent_id, situation)
        return [r.content for r in results.memories]

    def act(self, situation):
        past_experiences = self.recall(situation)
        # Use experiences to inform decision making
        return make_decision(situation, past_experiences)

# Usage
agent = AIAgent("sales_agent")
agent.learn("Customer interested in pricing, closed deal with 20% discount")
action = agent.act("Customer asking about pricing")
```

## 🏢 Production Setup

### Environment Configuration
```bash
# .env file
MEMORIZER_STORAGE=supabase
MEMORIZER_VECTOR_STORE=pinecone
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_DB_PASSWORD=your_password
PINECONE_API_KEY=your_api_key
OPENAI_API_KEY=your_openai_key
```

### Production Code
```python
import memorizer
import os

# Automatic configuration from environment
memory = memorizer.create_memory_manager(
    storage_provider=os.getenv("MEMORIZER_STORAGE", "memory"),
    vector_store=os.getenv("MEMORIZER_VECTOR_STORE", "memory"),
    llm_provider="openai"
)

# Same API, scales to production
memory.store_memory("user123", "Customer prefers email communication")
results = memory.search_memories("user123", "communication preferences")
```

## 🔧 Configuration Patterns

### 1. **Simple Development**
```python
# Works out of the box, no setup required
memory = memorizer.create_memory()
```

### 2. **Team Development**
```python
# Shared configuration file
config = {
    "storage_provider": "supabase",
    "supabase_url": "https://team-project.supabase.co",
    "supabase_password": os.getenv("SUPABASE_PASSWORD")
}
memory = memorizer.create_memory_manager(**config)
```

### 3. **Enterprise Production**
```python
# Full external provider setup
memory = memorizer.create_memory_manager(
    storage_provider="supabase",      # Team database
    vector_store="pinecone",          # Enterprise vector search
    llm_provider="openai",            # LLM for summarization
    very_new_ttl_days=7,              # Memory lifecycle
    mid_term_ttl_days=30,
    compression_threshold=0.8
)
```

## 📦 Installation Options

### Option 1: PyPI (Recommended)
```bash
pip install memorizer
```

### Option 2: Development Install
```bash
git clone https://github.com/your-org/memorizer
cd memorizer
pip install -e .
```

### Option 3: With External Providers
```bash
pip install memorizer[all]  # Includes Pinecone, Weaviate, Supabase clients
```

## 🔄 Migration Patterns

### From In-Memory to Production
```python
# Development
memory = memorizer.create_memory()

# Production (same API)
memory = memorizer.create_memory_manager(
    storage_provider="supabase",
    vector_store="pinecone"
)
```

### From Other Memory Solutions
```python
# Replace existing memory systems
class YourAIApp:
    def __init__(self):
        # Old: self.memory = SomeOtherMemorySystem()
        # New:
        self.memory = memorizer.create_memory()

    # Keep existing methods, just change backend
    def remember(self, user_id, content):
        return self.memory.store_memory(user_id, content)

    def recall(self, user_id, query):
        results = self.memory.search_memories(user_id, query)
        return [r.content for r in results.memories]
```

## 🧪 Testing Patterns

### Unit Tests
```python
import memorizer
import pytest

class TestMyAIApp:
    def setup_method(self):
        # Use simple memory for tests
        self.memory = memorizer.create_memory()
        self.app = MyAIApp(memory=self.memory)

    def test_memory_storage(self):
        memory_id = self.memory.store_memory("test_user", "test content")
        assert memory_id is not None

        results = self.memory.search_memories("test_user", "test")
        assert len(results.memories) == 1
```

### Integration Tests
```python
def test_with_external_providers():
    # Test with real external services
    memory = memorizer.create_memory_manager(
        storage_provider="supabase",
        supabase_url=test_config.SUPABASE_URL
    )
    # ... test logic
```

## 📊 Monitoring & Observability

```python
import memorizer

# Get memory statistics
memory = memorizer.create_memory()
stats = memory.get_stats("user123")

print(f"Total memories: {stats.total_memories}")
print(f"By tier: {stats.memory_by_tier}")

# Log memory operations
import logging
logging.basicConfig(level=logging.INFO)

# Memory operations will be logged automatically
memory.store_memory("user123", "content")  # Logs: "Stored memory mem_123"
```

## 🚀 Deployment Examples

### Docker
```dockerfile
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV MEMORIZER_STORAGE=supabase
CMD ["python", "app.py"]
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-app
spec:
  template:
    spec:
      containers:
      - name: ai-app
        image: your-ai-app:latest
        env:
        - name: MEMORIZER_STORAGE
          value: "supabase"
        - name: SUPABASE_URL
          valueFrom:
            secretKeyRef:
              name: memorizer-secrets
              key: supabase-url
```

## 🤝 Common Integration Patterns

### 1. **Framework Agnostic**
Works with any AI framework:
- ✅ LangChain
- ✅ LlamaIndex
- ✅ CrewAI
- ✅ AutoGen
- ✅ Custom solutions

### 2. **Provider Agnostic**
Switch between providers easily:
- 🗄️ **Storage**: Memory → SQLite → Supabase → PostgreSQL
- 🔍 **Vectors**: Memory → SQLite → Pinecone → Weaviate
- 🧠 **LLM**: Mock → OpenAI → Anthropic → Groq

### 3. **Deployment Agnostic**
Deploy anywhere:
- 💻 Local development
- ☁️ Cloud (AWS, GCP, Azure)
- 🐳 Containers (Docker, K8s)
- 🚀 Serverless (Lambda, Vercel)

## 💡 Best Practices

1. **Start Simple**: Use `create_memory()` for development
2. **Environment Config**: Use environment variables for credentials
3. **Graceful Degradation**: Handle external provider failures
4. **Memory Lifecycle**: Configure TTL based on your use case
5. **Search Strategy**: Use specific queries for better results
6. **Metadata**: Store relevant metadata for filtering

---

## 🎉 Ready to Integrate!

Memorizer is designed to work exactly like any other Python library your team already uses. No complex setup, no infrastructure requirements, just intelligent memory for your AI applications.

**Get Started:**
```bash
pip install memorizer
```

**5-minute integration:**
```python
import memorizer
memory = memorizer.create_memory()
# Your AI app now has memory! 🧠
```