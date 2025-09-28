# Memorizer Framework Guide

## 🎯 What is the Memorizer Framework?

The Memorizer Framework is a **true framework** for building memory management systems for AI agents. Unlike the previous service-oriented approach, the framework provides:

- **Extensible Architecture**: Plugin system for custom components
- **Configuration-Driven**: YAML-based configuration, no code changes needed
- **Language-Agnostic**: REST/GraphQL APIs for any language
- **Modular Design**: Use only what you need
- **Developer-Friendly**: CLI tools, examples, and comprehensive documentation

## 🏗️ Framework Architecture

### **Core Components**

```
┌─────────────────────────────────────────────────────────────┐
│                    Memorizer Framework                      │
├─────────────────────────────────────────────────────────────┤
│  Core Interfaces (ABC)                                     │
│  ├── Summarizer    ├── Retriever    ├── Storage           │
│  ├── PIIFilter     ├── Scorer       ├── TaskRunner        │
│  ├── EmbeddingProvider ├── VectorStore ├── CacheProvider  │
│  └── MemoryLifecycle                                        │
├─────────────────────────────────────────────────────────────┤
│  Component Registry & Plugin System                        │
├─────────────────────────────────────────────────────────────┤
│  Configuration Management (YAML/JSON/ENV)                  │
├─────────────────────────────────────────────────────────────┤
│  Built-in Implementations                                  │
│  ├── OpenAI/Anthropic/Groq Summarizers                    │
│  ├── Hybrid/Keyword/Vector Retrievers                     │
│  ├── PostgreSQL/MongoDB Storage                           │
│  └── Redis/Memory Cache                                   │
├─────────────────────────────────────────────────────────────┤
│  Extensions (Community Plugins)                           │
│  ├── Custom Summarizers   ├── Custom Retrievers          │
│  ├── Custom Storage       ├── Custom Filters             │
│  └── Custom Task Runners                                  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### **1. Installation**

```bash
# Install the framework
pip install memorizer

# Or install from source
git clone https://github.com/your-org/memorizer
cd memorizer
pip install -e .
```

### **2. Initialize a Project**

```bash
# Create a new project
memorizer init my-memory-app

# This creates:
# ├── memorizer.yaml          # Configuration file
# ├── example.py              # Usage example
# └── README.md               # Project documentation
```

### **3. Configure Your Components**

Edit `memorizer.yaml`:

```yaml
# Core configuration
summarizer:
  type: "summarizer"
  name: "openai"
  config:
    model: "gpt-4o-mini"
    api_key: "${OPENAI_API_KEY}"

storage:
  type: "storage"
  name: "postgres"
  config:
    connection: "${DATABASE_URL}"

retriever:
  type: "retriever"
  name: "hybrid"
  config:
    keyword_weight: 0.4
    vector_weight: 0.6
```

### **4. Use the Framework**

```python
from memorizer import create_framework

# Load framework
framework = create_framework("memorizer.yaml")

# Get memory manager
memory_manager = framework.get_memory_manager()

# Store a memory
memory_id = memory_manager.store_memory(
    user_id="user123",
    content="Important information to remember",
    metadata={"category": "important"}
)

# Search memories
results = memory_manager.search_memories(
    user_id="user123",
    query="important information"
)

print(f"Found {results.total_found} memories")
```

## 🔌 Plugin System

### **Creating Custom Components**

```python
from memorizer.framework import register_component, Summarizer

@register_component("summarizer", "my_custom")
class MyCustomSummarizer(Summarizer):
    def summarize(self, content: str, metadata: Dict, compression_type: str = "mid_term"):
        # Your custom summarization logic
        return {
            "summary": "Custom summary",
            "metadata": {"custom": True}
        }
    
    def get_supported_types(self):
        return ["mid_term", "long_term"]
    
    def get_health_status(self):
        return {"status": "healthy", "provider": "custom"}
```

### **Using Custom Components**

```yaml
# memorizer.yaml
summarizer:
  type: "summarizer"
  name: "my_custom"  # Your registered component
  config:
    custom_param: "value"
```

## ⚙️ Configuration

### **Configuration Sources (Priority Order)**

1. **YAML/JSON File**: `memorizer.yaml`
2. **Environment Variables**: `MEMORIZER_*`
3. **Default Values**: Built-in defaults

### **Environment Variables**

```bash
# Core settings
export MEMORIZER_DEBUG=false
export MEMORIZER_LOG_LEVEL=INFO

# Component settings
export MEMORIZER_SUMMARIZER=openai
export MEMORIZER_STORAGE=postgres
export MEMORIZER_CACHE=redis

# API keys
export OPENAI_API_KEY=your_key
export DATABASE_URL=postgresql://...
export REDIS_URL=redis://...
```

### **Configuration Schema**

```yaml
framework:
  version: "1.0.0"
  debug: false
  log_level: "INFO"

# Component configurations
summarizer:
  type: "summarizer"
  name: "openai"  # or "anthropic", "groq", "mock"
  enabled: true
  config:
    model: "gpt-4o-mini"
    api_key: "${OPENAI_API_KEY}"
    temperature: 0.3

storage:
  type: "storage"
  name: "postgres"  # or "mongodb", "sqlite"
  enabled: true
  config:
    connection: "${DATABASE_URL}"
    pool_size: 10

# Memory lifecycle
memory_lifecycle:
  tiers:
    very_new:
      ttl_days: 7
      max_items: 1000
    mid_term:
      ttl_days: 30
      max_items: 5000
    long_term:
      ttl_days: 365
      max_items: 10000
```

## 🛠️ CLI Tools

### **Available Commands**

```bash
# Initialize a new project
memorizer init my-project

# Run the framework
memorizer run --config memorizer.yaml

# List available plugins
memorizer plugins

# Check framework health
memorizer health --config memorizer.yaml
```

### **Command Examples**

```bash
# Initialize with custom directory
memorizer init /path/to/project --force

# Run in daemon mode
memorizer run --config config.yaml --daemon

# Check health with specific config
memorizer health --config production.yaml
```

## 📚 API Reference

### **Framework Factory**

```python
from memorizer import create_framework

# From config file
framework = create_framework("config.yaml")

# From environment
framework = create_framework()

# From config object
from memorizer.framework.core.config import FrameworkConfig
config = FrameworkConfig()
framework = create_framework(config=config)
```

### **Memory Manager**

```python
memory_manager = framework.get_memory_manager()

# Store memory
memory_id = memory_manager.store_memory(
    user_id: str,
    content: str,
    metadata: Optional[Dict] = None,
    tier: str = "very_new"
) -> str

# Get memory
memory = memory_manager.get_memory(memory_id: str, user_id: str) -> Optional[Memory]

# Search memories
results = memory_manager.search_memories(
    user_id: str,
    query: str,
    limit: int = 10,
    filters: Optional[Dict] = None
) -> RetrievalResult

# Update memory
success = memory_manager.update_memory(
    memory_id: str,
    user_id: str,
    content: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> bool

# Delete memory
success = memory_manager.delete_memory(memory_id: str, user_id: str) -> bool
```

### **Component Interfaces**

All components implement standard interfaces:

```python
# Summarizer
class Summarizer(ABC):
    def summarize(self, content: str, metadata: Dict, compression_type: str) -> Dict
    def get_supported_types(self) -> List[str]
    def get_health_status(self) -> Dict[str, Any]

# Retriever
class Retriever(ABC):
    def retrieve(self, query: Query) -> RetrievalResult
    def get_health_status(self) -> Dict[str, Any]

# Storage
class Storage(ABC):
    def store(self, memory: Memory) -> str
    def get(self, memory_id: str, user_id: str) -> Optional[Memory]
    def update(self, memory: Memory) -> bool
    def delete(self, memory_id: str, user_id: str) -> bool
    def search(self, query: Query, limit: int, offset: int) -> List[Memory]
```

## 🔧 Built-in Components

### **Summarizers**

- **OpenAI**: GPT models (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
- **Anthropic**: Claude models (claude-3-sonnet, claude-3-opus, claude-3-haiku)
- **Groq**: Fast inference (llama3-8b-8192, mixtral-8x7b-32768)
- **Mock**: For testing and development

### **Retrievers**

- **Hybrid**: Combines keyword and vector search
- **Keyword**: Full-text search based
- **Vector**: Embedding-based similarity search

### **Storage**

- **PostgreSQL**: Production-ready with JSONB support
- **MongoDB**: Document-based storage
- **SQLite**: Lightweight local storage

### **Cache**

- **Redis**: High-performance distributed cache
- **Memory**: In-process cache
- **File**: File-based cache

## 🚀 Migration from Service to Framework

### **Before (Service Approach)**

```python
# Old way - tightly coupled
from memorizer import MemoryManager
manager = MemoryManager()  # Fixed stack
```

### **After (Framework Approach)**

```python
# New way - configurable and extensible
from memorizer import create_framework
framework = create_framework("config.yaml")
manager = framework.get_memory_manager()
```

### **Migration Steps**

1. **Install Framework**: `pip install memorizer`
2. **Create Config**: `memorizer init my-project`
3. **Update Code**: Replace direct imports with framework factory
4. **Test**: Run with new configuration
5. **Deploy**: Use configuration-driven deployment

## 🎯 Framework Benefits

### **For Developers**

- **Extensibility**: Easy to add custom components
- **Flexibility**: Choose your own infrastructure stack
- **Modularity**: Use only what you need
- **Testing**: Framework provides testing utilities

### **For the Project**

- **Community Growth**: Others can contribute plugins
- **Adoption**: Easier to integrate into existing systems
- **Maintenance**: Core framework separated from implementations
- **Ecosystem**: Plugin marketplace and community contributions

## 📈 Next Steps

1. **Explore Examples**: Check `examples/framework_example.py`
2. **Create Custom Components**: Build your own summarizers, retrievers, etc.
3. **Contribute Plugins**: Share your components with the community
4. **Integrate**: Use the framework in your AI applications

## 🤝 Contributing

The framework is designed for community contributions:

1. **Create Plugins**: Build custom components
2. **Improve Core**: Enhance the framework itself
3. **Add Examples**: Share usage patterns
4. **Write Documentation**: Help others learn

## 📞 Support

- **Documentation**: [Framework Guide](FRAMEWORK_GUIDE.md)
- **Examples**: [examples/](examples/)
- **Issues**: [GitHub Issues](https://github.com/your-org/memorizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/memorizer/discussions)

---

**The Memorizer Framework: From Service to True Framework** 🚀
