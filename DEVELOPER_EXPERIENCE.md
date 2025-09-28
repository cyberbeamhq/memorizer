# 🧠 Memorizer - Developer Experience

**How AI teams use Memorizer just like any other open source library**

## 📦 Installation (Like any Python package)

```bash
# Standard installation
pip install memorizer

# With all external providers
pip install memorizer[all]

# Development installation
git clone https://github.com/your-org/memorizer
cd memorizer
pip install -e .
```

## ⚡ 30-Second Integration

```python
# 1. Import (like requests, pandas, etc.)
import memorizer

# 2. Initialize (like any library)
memory = memorizer.create_memory()

# 3. Use immediately
memory_id = memory.store_memory("user123", "I prefer detailed code examples")
results = memory.search_memories("user123", "coding preferences")

# ✅ Your AI app now has memory!
```

## 🎯 Real-World Usage Patterns

### Pattern 1: AI Chatbot Memory
```python
import memorizer

class ChatBot:
    def __init__(self, user_id):
        self.user_id = user_id
        # Just like initializing any service
        self.memory = memorizer.create_memory()

    def chat(self, message):
        # Get context from memory
        context = self.memory.search_memories(self.user_id, message, limit=3)

        # Your AI logic (OpenAI, Anthropic, etc.)
        response = your_llm_call(message, context)

        # Remember this conversation
        self.memory.store_memory(
            self.user_id,
            f"User: {message}\nBot: {response}"
        )
        return response

# Usage exactly like any other class
bot = ChatBot("user_456")
response = bot.chat("Help me with Python")
```

### Pattern 2: RAG Document System
```python
import memorizer

class DocumentRAG:
    def __init__(self):
        # Initialize like any vector store
        self.memory = memorizer.create_memory()

    def add_document(self, content, metadata=None):
        # Store documents like adding to any database
        return self.memory.store_memory(
            "knowledge_base",
            content,
            metadata=metadata
        )

    def search(self, query, limit=5):
        # Search like querying any search engine
        results = self.memory.search_memories("knowledge_base", query, limit)
        return [r.content for r in results.memories]

# Standard usage pattern
rag = DocumentRAG()
rag.add_document("Python is great for AI development", {"topic": "python"})
relevant_docs = rag.search("AI development languages")
```

### Pattern 3: Drop-in Framework Integration
```python
import memorizer
from langchain.memory import BaseMemory

class MemorizerLangChain(BaseMemory):
    """Drop-in replacement for LangChain memory"""

    def __init__(self, user_id):
        self.user_id = user_id
        self.memory = memorizer.create_memory()

    def save_context(self, inputs, outputs):
        conversation = f"Human: {inputs['input']}\nAI: {outputs['output']}"
        self.memory.store_memory(self.user_id, conversation)

    def load_memory_variables(self, inputs):
        results = self.memory.search_memories(self.user_id, inputs['input'])
        return {"history": "\n".join([r.content for r in results.memories])}

# Works with existing LangChain code
memory = MemorizerLangChain("user123")
# Use in your existing LangChain chains...
```

## 🏗️ Team Development Workflow

### 1. Local Development
```python
# No configuration needed - works immediately
import memorizer
memory = memorizer.create_memory()
```

### 2. Team Staging
```python
# Environment-based configuration
import memorizer
import os

memory = memorizer.create_memory_manager(
    storage_provider=os.getenv("MEMORY_STORAGE", "memory"),
    vector_store=os.getenv("MEMORY_VECTORS", "memory")
)
```

### 3. Production Deployment
```python
# Full external provider configuration
import memorizer

memory = memorizer.create_memory_manager(
    storage_provider="supabase",
    vector_store="pinecone",
    llm_provider="openai",
    # Auto-loads from environment variables
)
```

## 🔧 Configuration Management

### Environment Variables (.env)
```bash
# Development (optional - has defaults)
MEMORIZER_STORAGE=memory
MEMORIZER_VECTORS=memory

# Production
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_DB_PASSWORD=your_password
PINECONE_API_KEY=your_key
PINECONE_ENVIRONMENT=us-west1-gcp
OPENAI_API_KEY=your_openai_key
```

### Team Configuration (config.yaml)
```yaml
development:
  storage_provider: memory
  vector_store: memory

staging:
  storage_provider: supabase
  vector_store: pinecone

production:
  storage_provider: supabase
  vector_store: pinecone
  llm_provider: openai
  very_new_ttl_days: 7
  mid_term_ttl_days: 30
```

## 🧪 Testing Patterns

### Unit Tests
```python
import memorizer
import pytest

class TestAIApp:
    def setup_method(self):
        # Use memory for isolated tests
        self.memory = memorizer.create_memory()
        self.app = YourAIApp(memory=self.memory)

    def test_memory_functionality(self):
        # Test like any other component
        memory_id = self.memory.store_memory("test_user", "test content")
        assert memory_id is not None

        results = self.memory.search_memories("test_user", "test")
        assert len(results.memories) == 1
        assert results.memories[0].content == "test content"
```

### Integration Tests
```python
def test_with_real_providers():
    # Test with actual external services
    memory = memorizer.create_memory_manager(
        storage_provider="supabase",
        supabase_url="https://test-project.supabase.co"
    )
    # Test real functionality...
```

## 📊 Monitoring & Debugging

### Built-in Statistics
```python
import memorizer

memory = memorizer.create_memory()

# Store some memories...
memory.store_memory("user123", "Python programming question")
memory.store_memory("user123", "Database design discussion")

# Get insights
stats = memory.get_stats("user123")
print(f"Total memories: {stats.total_memories}")
print(f"Memory distribution: {stats.memory_by_tier}")
```

### Logging Integration
```python
import logging
import memorizer

# Configure logging (standard Python logging)
logging.basicConfig(level=logging.INFO)

memory = memorizer.create_memory()

# All operations are automatically logged
memory.store_memory("user123", "content")  # Logs: "Stored memory mem_123"
memory.search_memories("user123", "query")  # Logs: "Searched memories for user123"
```

## 🚀 Deployment Examples

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Environment configuration
ENV MEMORIZER_STORAGE=supabase
ENV MEMORIZER_VECTORS=pinecone

CMD ["python", "main.py"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  ai-app:
    build: .
    environment:
      - MEMORIZER_STORAGE=supabase
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_DB_PASSWORD=${SUPABASE_PASSWORD}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
    ports:
      - "8000:8000"
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-chatbot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-chatbot
  template:
    metadata:
      labels:
        app: ai-chatbot
    spec:
      containers:
      - name: chatbot
        image: your-org/ai-chatbot:latest
        env:
        - name: MEMORIZER_STORAGE
          value: "supabase"
        - name: SUPABASE_URL
          valueFrom:
            secretKeyRef:
              name: memorizer-secrets
              key: supabase-url
```

## 🔄 Migration & Scaling

### From Simple to Production
```python
# Development - starts simple
memory = memorizer.create_memory()

# Production - same API, different backend
memory = memorizer.create_memory_manager(
    storage_provider="supabase",
    vector_store="pinecone"
)

# No code changes needed! 🎉
```

### Provider Switching
```python
# Switch storage providers easily
configs = {
    "sqlite": {"storage_provider": "sqlite"},
    "supabase": {"storage_provider": "supabase"},
    "railway": {"storage_provider": "railway"},
}

memory = memorizer.create_memory_manager(**configs["supabase"])
```

## 💡 Best Practices

### 1. **Start Simple**
```python
# Begin with in-memory for development
memory = memorizer.create_memory()
```

### 2. **Use Environment Config**
```python
# Environment-driven configuration
memory = memorizer.create_memory_manager(
    storage_provider=os.getenv("MEMORY_STORAGE", "memory")
)
```

### 3. **Handle Graceful Degradation**
```python
try:
    memory = memorizer.create_memory_manager(storage_provider="supabase")
except Exception:
    # Fallback to simple memory
    memory = memorizer.create_memory()
```

### 4. **Structure Memory Content**
```python
# Use metadata for better organization
memory.store_memory(
    user_id,
    content,
    metadata={
        "conversation_id": "conv_123",
        "intent": "coding_help",
        "timestamp": datetime.now().isoformat()
    }
)
```

## 🎉 Why Teams Love Memorizer

✅ **Familiar**: Works exactly like libraries you already use
✅ **Simple**: 3-line integration, no complex setup
✅ **Scalable**: Memory → SQLite → Supabase → Enterprise
✅ **Flexible**: Works with any AI framework
✅ **Reliable**: Production-ready with proper error handling
✅ **Testable**: Easy to unit test and mock

## 🚀 Ready to Ship!

Memorizer is designed to work exactly like any other Python library your team already uses. No learning curve, no infrastructure complexity, just intelligent memory for your AI applications.

**Start building:**
```bash
pip install memorizer
```

**5 minutes later:**
```python
import memorizer
memory = memorizer.create_memory()
# Your AI app now remembers everything! 🧠✨
```