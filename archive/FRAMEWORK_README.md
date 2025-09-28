# Memorizer Framework

A comprehensive, extensible framework for AI agent memory management with pluggable components and enterprise-grade features.

## 🚀 Quick Start

```python
from framework.factory import create_framework
from framework.core.config import FrameworkConfig

# Create configuration
config = FrameworkConfig.from_dict({
    "storage": {"type": "postgres", "connection": "postgresql://..."},
    "cache": {"type": "redis", "host": "localhost", "port": 6379},
    "summarizer": {"type": "openai", "model": "gpt-4o-mini"},
    "retriever": {"type": "hybrid"},
    "embedding_provider": {"type": "openai", "model": "text-embedding-3-small"},
    "vector_store": {"type": "pinecone", "api_key": "...", "index_name": "memorizer"}
})

# Create framework instance
framework = create_framework(config)

# Store a memory
memory = Memory(
    user_id="user123",
    content="I love working with AI and machine learning.",
    metadata={"topic": "AI", "sentiment": "positive"},
    tier="very_new"
)
memory_id = framework.storage.store(memory)

# Search for memories
query = Query(text="AI and machine learning", user_id="user123")
results = framework.retriever.retrieve(query)
```

## 🏗️ Architecture

The Memorizer Framework is built on a modular, plugin-based architecture:

### Core Components

- **Interfaces**: Abstract base classes defining contracts for all components
- **Registry**: Component discovery and management system
- **Configuration**: YAML/JSON/ENV-based configuration management
- **Lifecycle**: End-to-end memory lifecycle management

### Builtin Components

- **Storage**: PostgreSQL, MongoDB, SQLite
- **Cache**: Redis, Memory, File-based
- **Summarizers**: OpenAI, Anthropic, Groq, Mock
- **Retrievers**: Hybrid, Keyword, Vector
- **Scorers**: TF-IDF, Cosine, Hybrid
- **Task Runners**: Celery, RQ, Thread-based
- **Embedding Providers**: OpenAI, Cohere, HuggingFace
- **Vector Stores**: Pinecone, Weaviate, Chroma, PostgreSQL
- **PII Filters**: Memorizer, Basic regex-based

## 🔧 Configuration

The framework supports multiple configuration sources:

### YAML Configuration

```yaml
# memorizer.yaml
storage:
  type: postgres
  connection: postgresql://user:password@localhost:5432/memorizer
  pool_size: 10

cache:
  type: redis
  host: localhost
  port: 6379
  db: 0
  default_ttl: 3600

summarizer:
  type: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
  temperature: 0.3

retriever:
  type: hybrid
  keyword_weight: 0.4
  vector_weight: 0.6

scorer:
  type: hybrid
  keyword_weight: 0.4
  vector_weight: 0.6

task_runner:
  type: thread
  max_workers: 4

embedding_provider:
  type: openai
  model: text-embedding-3-small
  api_key: ${OPENAI_API_KEY}

vector_store:
  type: pinecone
  api_key: ${PINECONE_API_KEY}
  environment: us-west1-gcp
  index_name: memorizer

pii_filter:
  type: memorizer
  sensitivity_level: medium
  replace_with: "[REDACTED]"
```

### Environment Variables

```bash
# Storage
MEMORIZER_STORAGE_TYPE=postgres
MEMORIZER_STORAGE_CONNECTION=postgresql://user:password@localhost:5432/memorizer

# Cache
MEMORIZER_CACHE_TYPE=redis
MEMORIZER_CACHE_HOST=localhost
MEMORIZER_CACHE_PORT=6379

# Summarizer
MEMORIZER_SUMMARIZER_TYPE=openai
MEMORIZER_SUMMARIZER_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_api_key_here

# Vector Store
MEMORIZER_VECTOR_STORE_TYPE=pinecone
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENVIRONMENT=us-west1-gcp
```

## 🔌 Plugin System

The framework supports custom components through a plugin system:

### Creating Custom Components

```python
from framework.core.interfaces import Summarizer
from framework.core.registry import register_component

@register_component("summarizer", "custom")
class CustomSummarizer(Summarizer):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def summarize(self, content: str, metadata: Dict[str, Any], compression_type: str) -> Dict[str, Any]:
        # Your custom summarization logic
        return {"summary": "Custom summary", "confidence": 0.95}
```

### Using Custom Components

```python
config = FrameworkConfig.from_dict({
    "summarizer": {
        "type": "custom",
        "custom_param": "value"
    }
})
```

## 📊 Memory Lifecycle

The framework manages a three-tier memory lifecycle:

1. **Very New**: Raw memories, immediate access
2. **Mid-term**: Compressed summaries, moderate access
3. **Long-term**: Highly compressed, archival storage

### Lifecycle Management

```python
# Process new memory
memory = Memory(user_id="user123", content="...", tier="very_new")
memory_id = framework.lifecycle.process_new_memory(memory)

# Advance memory tier
new_tier = framework.lifecycle.advance_memory_tier(memory_id, "user123", "very_new")

# Get memory status
status = framework.lifecycle.get_memory_status(memory_id, "user123")
```

## 🔍 Retrieval System

The framework supports multiple retrieval strategies:

### Hybrid Retrieval

```python
# Configure hybrid retriever
config = {
    "retriever": {
        "type": "hybrid",
        "keyword_weight": 0.4,
        "vector_weight": 0.6
    }
}

# Search
query = Query(text="AI and machine learning", user_id="user123")
results = framework.retriever.retrieve(query)
```

### Vector Search

```python
# Configure vector retriever
config = {
    "retriever": {
        "type": "vector",
        "similarity_threshold": 0.7
    }
}
```

## 🛡️ Security Features

### PII Detection and Filtering

```python
# Configure PII filter
config = {
    "pii_filter": {
        "type": "memorizer",
        "sensitivity_level": "high",
        "replace_with": "[REDACTED]"
    }
}

# Filter content
sanitized_content, pii_data = framework.pii_filter.filter(content)
```

### Access Control

```python
# Configure RBAC
config = {
    "security": {
        "rbac_enabled": True,
        "default_role": "user",
        "roles": {
            "admin": ["read", "write", "delete"],
            "user": ["read", "write"],
            "viewer": ["read"]
        }
    }
}
```

## ⚡ Performance Features

### Caching

```python
# Configure caching
config = {
    "cache": {
        "type": "redis",
        "host": "localhost",
        "port": 6379,
        "default_ttl": 3600
    }
}

# Use cache
framework.cache.set("key", {"data": "value"}, ttl=60)
value = framework.cache.get("key")
```

### Background Processing

```python
# Configure task runner
config = {
    "task_runner": {
        "type": "celery",
        "broker_url": "redis://localhost:6379/0"
    }
}

# Submit background task
task = framework.task_runner.submit(process_memory, memory_id)
result = framework.task_runner.get_result(task)
```

## 📈 Monitoring and Observability

### Health Checks

```python
# Check component health
health = framework.storage.get_health_status()
print(f"Storage status: {health['status']}")

# Check all components
for component in [framework.storage, framework.cache, framework.vector_store]:
    health = component.get_health_status()
    print(f"{component.__class__.__name__}: {health['status']}")
```

### Metrics Collection

```python
# Configure metrics
config = {
    "monitoring": {
        "metrics_enabled": True,
        "prometheus_port": 8000
    }
}
```

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "-m", "framework.cli.main", "run"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memorizer-framework
spec:
  replicas: 3
  selector:
    matchLabels:
      app: memorizer-framework
  template:
    metadata:
      labels:
        app: memorizer-framework
    spec:
      containers:
      - name: memorizer
        image: memorizer:latest
        ports:
        - containerPort: 8000
        env:
        - name: MEMORIZER_STORAGE_TYPE
          value: "postgres"
        - name: MEMORIZER_CACHE_TYPE
          value: "redis"
```

## 🧪 Testing

### Unit Tests

```python
import pytest
from framework.factory import create_framework
from framework.core.config import FrameworkConfig

def test_memory_storage():
    config = FrameworkConfig.from_dict({
        "storage": {"type": "memory"},
        "cache": {"type": "memory"}
    })
    
    framework = create_framework(config)
    
    memory = Memory(
        user_id="test_user",
        content="Test content",
        tier="very_new"
    )
    
    memory_id = framework.storage.store(memory)
    assert memory_id is not None
    
    retrieved = framework.storage.get(memory_id, "test_user")
    assert retrieved.content == "Test content"
```

### Integration Tests

```python
def test_end_to_end_workflow():
    # Test complete memory lifecycle
    pass
```

## 📚 Examples

See the `examples/` directory for comprehensive examples:

- `framework_complete_example.py` - Complete framework usage
- `custom_components_example.py` - Custom component development
- `deployment_example.py` - Deployment configurations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Documentation: [Framework Guide](FRAMEWORK_GUIDE.md)
- Issues: [GitHub Issues](https://github.com/your-org/memorizer/issues)
- Discussions: [GitHub Discussions](https://github.com/your-org/memorizer/discussions)

## 🎯 Roadmap

- [ ] GraphQL API
- [ ] WebSocket support
- [ ] Advanced analytics
- [ ] Multi-tenant support
- [ ] Plugin marketplace
- [ ] Visual configuration editor
