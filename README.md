# Memorizer

[![Build](https://img.shields.io/github/actions/workflow/status/cyberbeamhq/memorizer/ci-cd.yml?branch=main)](https://github.com/cyberbeamhq/memorizer/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/LLM-OpenAI%20gpt--4o--mini-green)](https://platform.openai.com/)
[![Production Ready](https://img.shields.io/badge/status-Production%20Ready-green)](https://docs.memorizer.dev)

**Memorizer** is a production-ready, open-source memory lifecycle framework for AI assistants and agents.  
It helps LLM-based systems **forget smartly, remember what matters, and reduce token usage** while keeping historical context accessible.

> 📚 **For detailed documentation, API reference, and advanced usage examples, visit our [Documentation Website](https://docs.memorizer.dev)**

---

## ✨ Key Features

### 🧠 **Intelligent Memory Management**
- **Three-tier memory lifecycle**
  - **Very new** → recent sessions (raw, full text, up to 10 days / N sessions)
  - **Mid-term** → compressed summaries with unnecessary words removed (last 12 months)
  - **Long-term** → highly aggregated, <1000-character briefs with sentiment, preferences, and key metrics
- **Smart compression** with LLM-powered summarization (OpenAI `gpt-4o-mini` by default)
- **Hybrid retrieval** combining keyword relevance scoring with vector DB fallback

### 🏗️ **Production-Ready Infrastructure**
- **DB-first design** with PostgreSQL + JSONB for structured queries and analytics
- **RESTful API** with FastAPI, authentication, rate limiting, and comprehensive error handling
- **Background job processing** with Celery for embedding generation and memory compression
- **Redis caching** with LRU eviction and TTL for optimal performance
- **Docker support** with development and production configurations
- **Kubernetes ready** with Helm charts and deployment manifests

### 🔌 **Extensive Integrations**
- **Vector databases**: Pinecone, Weaviate, Chroma, pgvector
- **AI frameworks**: LangChain, LlamaIndex, AutoGPT, CrewAI
- **Embedding providers**: OpenAI, Cohere, HuggingFace, local models
- **Cloud providers**: AWS, Azure, Google Cloud
- **Monitoring**: Prometheus, Grafana, Sentry integration

### 🛡️ **Enterprise Security & Compliance**
- **Authentication & Authorization** with JWT and API keys
- **Role-based access control (RBAC)** with granular permissions
- **Input validation & sanitization** with XSS and SQL injection protection
- **Audit logging** for compliance and security monitoring
- **Rate limiting** with sliding window algorithm
- **Comprehensive error handling** with structured logging

### 📊 **Monitoring & Observability**
- **Structured logging** with request tracing and correlation IDs
- **Performance monitoring** with Prometheus metrics
- **Health checks** for all system components
- **Automated testing** with comprehensive test suites
- **Real-time dashboards** for system monitoring

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Memorizer Framework                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Very New    │  │ Mid-Term    │  │ Long-Term   │            │
│  │ Memory      │  │ Memory      │  │ Memory      │            │
│  │ (Raw, 10d)  │  │ (Summary,   │  │ (Brief,     │            │
│  │             │  │ 12 months)  │  │ <1k chars)  │            │
│  └─────┬───────┘  └─────┬───────┘  └─────┬───────┘            │
│        │ move/compress   │ aggregate      │ fallback          │
│        ▼                ▼                ▼                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              PostgreSQL + JSONB                        │  │
│  │         (Structured queries & analytics)               │  │
│  └─────────────────────┬───────────────────────────────────┘  │
│                        │                                      │
│  ┌─────────────────────▼───────────────────────────────────┐  │
│  │              Vector DB (Optional)                      │  │
│  │    (Pinecone / Weaviate / Chroma / pgvector)          │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Production Infrastructure                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ FastAPI     │  │ Celery      │  │ Redis       │            │
│  │ (REST API)  │  │ (Background │  │ (Caching &  │            │
│  │             │  │ Jobs)       │  │ Queues)     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Auth &      │  │ Monitoring  │  │ Docker &    │            │
│  │ Security    │  │ (Prometheus │  │ Kubernetes  │            │
│  │ (JWT/RBAC)  │  │ Grafana)    │  │ Ready)      │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏆 Competitive Advantage

### **How Memorizer Stands Out**

| Feature | **Memorizer** | Zep | Letta (MemGPT) | Mem0 | Redis-based |
|---------|---------------|-----|----------------|------|-------------|
| **Memory Lifecycle** | ✅ **3-tier intelligent** | ❌ Temporal graph only | ❌ Infinite context | ❌ Basic persistence | ❌ Manual decay |
| **Production Ready** | ✅ **Enterprise-grade** | ✅ SOC 2 compliant | ❌ Development focus | ❌ Lightweight only | ❌ Manual setup |
| **Cost Optimization** | ✅ **Smart compression** | ❌ No compression | ❌ Infinite context | ✅ Basic filtering | ❌ No optimization |
| **Developer Experience** | ✅ **5-minute setup** | ✅ Good tooling | ✅ Strong tooling | ✅ Easy to use | ❌ Manual coding |
| **Scalability** | ✅ **K8s + Cloud** | ✅ High availability | ✅ Scalable | ❌ Limited | ✅ Redis scaling |

### **Key Differentiators**

- **🧠 Intelligent Memory Lifecycle**: Only solution with automatic 3-tier memory evolution (raw → compressed → aggregated)
- **💰 Cost Optimization**: 60-80% reduction in token usage through smart compression
- **🏗️ Production-First**: Enterprise-grade security, monitoring, and Kubernetes-native deployment
- **🔌 Framework Agnostic**: Works with any AI framework while providing deep integrations
- **⚡ Developer Experience**: 5-minute setup vs. hours of configuration for competitors

---

## 🎯 Use Cases

### **Customer Service & Support**
- **Chatbots** that remember user preferences and conversation history
- **Support agents** with context-aware responses across multiple sessions
- **FAQ systems** that learn from user interactions and improve over time
- **Ticket routing** based on user history and preferences

### **E-commerce & Personalization**
- **Product recommendations** based on browsing and purchase history
- **Shopping assistants** that remember size preferences, brands, and budgets
- **Price tracking** and alert systems with user-specific criteria
- **Inventory management** with demand prediction based on user behavior

### **Healthcare & Wellness**
- **Patient management** systems that track symptoms and treatment history
- **Telemedicine** platforms with comprehensive patient context
- **Medication reminders** with personalized scheduling
- **Health monitoring** with trend analysis and alert systems

### **Education & Training**
- **Personalized learning** paths based on student progress and preferences
- **Tutoring systems** that adapt to individual learning styles
- **Skill assessment** with continuous improvement tracking
- **Course recommendations** based on career goals and interests

### **Financial Services**
- **Investment advisors** with personalized portfolio recommendations
- **Fraud detection** systems with user behavior patterns
- **Credit scoring** with comprehensive financial history
- **Budgeting tools** that learn spending patterns and suggest optimizations

### **Content & Media**
- **Content curation** based on reading/viewing history and preferences
- **News aggregation** with personalized filtering and relevance scoring
- **Social media** feeds with intelligent content prioritization
- **Streaming services** with advanced recommendation algorithms

### **Enterprise & Business**
- **CRM systems** with comprehensive customer interaction history
- **Sales automation** with lead scoring and follow-up optimization
- **Project management** with team collaboration and knowledge retention
- **Knowledge bases** that evolve based on user queries and feedback

---

## 🤖 Compression Agent

### **What is the Compression Agent?**

The Compression Agent is Memorizer's **intelligent memory optimization engine** that automatically transforms raw conversations into compressed, meaningful summaries while preserving critical information.

### **How It Works**

```python
# Memory Lifecycle Flow
Raw Conversation (Very New)
    ↓ (after 10 days or N sessions)
Compression Agent Processing
    ↓
Compressed Summary (Mid-term)
    ↓ (after 12 months)
Aggregated Brief (Long-term)
```

### **Compression Process**

1. **Content Analysis**: Identifies key topics, sentiment, and important facts
2. **Redundancy Removal**: Eliminates repetitive or unnecessary information
3. **Context Preservation**: Maintains user preferences, decisions, and critical details
4. **Summary Generation**: Creates concise summaries using LLM-powered compression
5. **Quality Validation**: Ensures compressed content retains essential meaning

### **Key Features**

- **🎯 Smart Compression**: Reduces content by 60-80% while preserving meaning
- **🧠 Multi-LLM Support**: Supports OpenAI, Anthropic, Groq, OpenRouter, Ollama, and custom models
- **⚡ Background Processing**: Non-blocking compression via Celery workers
- **🔄 Configurable**: Customizable compression policies and thresholds
- **📊 Analytics**: Tracks compression effectiveness and quality metrics

### **Compression Examples**

#### **Before Compression (Raw)**
```
User: "Hi, I'm looking for a laptop for programming. I need something with at least 16GB RAM, 
good battery life, and a comfortable keyboard. My budget is around $1500. I also need it to 
be portable since I travel frequently. I've been looking at MacBooks but they're expensive. 
What would you recommend?"

Agent: "Based on your requirements, I'd recommend the Dell XPS 13 or ThinkPad X1 Carbon. 
Both offer excellent keyboards, good battery life, and are highly portable. The XPS 13 
starts around $1200 with 16GB RAM, while the ThinkPad X1 Carbon is around $1400. Both 
are great for programming and much more affordable than MacBooks."
```

#### **After Compression (Mid-term)**
```
User seeking programming laptop: 16GB RAM, good battery, comfortable keyboard, $1500 budget, 
portable for travel. Considering MacBooks but finds them expensive.

Recommended: Dell XPS 13 ($1200) or ThinkPad X1 Carbon ($1400). Both have excellent keyboards, 
good battery life, portability, and are programming-friendly. More affordable than MacBooks.
```

#### **After Aggregation (Long-term)**
```
User preferences: Programming laptop, 16GB RAM, portable, $1500 budget, prefers value over 
premium brands. Recommended Dell XPS 13 and ThinkPad X1 Carbon. Budget-conscious, travels frequently.
```

### **Benefits**

- **💰 Cost Reduction**: 60-80% fewer tokens for LLM processing
- **⚡ Performance**: Faster retrieval and processing
- **🧠 Memory Efficiency**: More memories stored in same space
- **📈 Scalability**: Better performance at scale
- **🎯 Relevance**: Preserves important information while removing noise

---

## 🤖 LLM Providers

### **Supported Providers**

Memorizer supports multiple LLM providers for maximum flexibility and cost optimization:

| Provider | Description | Best For | Models |
|----------|-------------|----------|---------|
| **OpenAI** | GPT models | General use, high quality | `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo` |
| **Anthropic** | Claude models | Complex reasoning, long context | `claude-3-sonnet`, `claude-3-opus`, `claude-3-haiku` |
| **Groq** | Fast inference | Speed-critical applications | `llama3-8b-8192`, `mixtral-8x7b-32768` |
| **OpenRouter** | Multiple models | Cost optimization, model variety | `anthropic/claude-3-sonnet`, `openai/gpt-4o-mini` |
| **Ollama** | Local models | Privacy, offline use | `llama3:8b`, `mistral:7b`, `codellama:7b` |
| **Custom** | Enterprise APIs | Private models, custom endpoints | Any OpenAI-compatible API |

### **Model Recommendations by Use Case**

```bash
# General purpose (balanced quality/speed/cost)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Fast inference (speed priority)
LLM_PROVIDER=groq
LLM_MODEL=llama3-8b-8192

# Cost optimization (cheap models)
LLM_PROVIDER=openrouter
LLM_MODEL=openai/gpt-3.5-turbo

# High quality (best results)
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-opus-20240229

# Local deployment (privacy)
LLM_PROVIDER=ollama
LLM_MODEL=llama3:8b

# Coding tasks
LLM_PROVIDER=groq
LLM_MODEL=deepseek-coder-6.7b-instruct
```

### **Discovery Utility**

Use the built-in discovery utility to explore available models:

```bash
# List all providers
python scripts/llm_discovery.py list

# Get detailed info about a provider
python scripts/llm_discovery.py info groq

# See model recommendations
python scripts/llm_discovery.py recommend

# Test a provider
python scripts/llm_discovery.py test mock

# Validate a model
python scripts/llm_discovery.py validate groq llama3-8b-8192
```

### **Configuration**

Set your preferred provider and model in `.env`:

```bash
# Primary LLM provider
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Provider-specific settings
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
OLLAMA_BASE_URL=http://localhost:11434
```

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/cyberbeamhq/memorizer.git
cd memorizer
```

### 2. Set up environment
Copy the example .env file and update it with your credentials:

```bash
cp .env.example .env
```

You'll need:
- `OPENAI_API_KEY` (or other LLM provider key)
- `DATABASE_URL` (Postgres connection string)
- Optional vector DB API keys (e.g. Pinecone)

### 3. Install dependencies

#### For Production:
```bash
pip install -r requirements.txt
```

#### For Development:
```bash
pip install -r requirements-dev.txt
```

#### With Optional Features:
```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

#### Using pyproject.toml (Modern Python):
```bash
# Production
pip install .

# Development
pip install .[dev]

# With specific optional features
pip install .[dev,pinecone,weaviate,monitoring]

# All optional features
pip install .[dev,all]
```

### 4. Run database migrations
```bash
python scripts/init_db.py
```

### 5. Try a local demo
```bash
python demo.py
```

### 6. Docker Deployment

#### Development:
```bash
docker-compose up -d
```

#### Production:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

The Docker setup includes:
- PostgreSQL database
- Redis for caching and job queues
- Memorizer API with auto-reload (dev) or production optimizations
- Celery workers for background tasks
- Prometheus and Grafana for monitoring

---

## 🔧 Example: E-commerce AI Assistant

Memorizer can back an e-commerce assistant:

- **Very new memory**: Last 5 support chats ("Where is my order?")
- **Mid-term memory**: Summarized chat history ("Customer had 12 refund requests in 2024")
- **Long-term memory**: Aggregated insights ("Customer prefers express shipping, positive sentiment about product quality, negative about delivery speed")

When the customer chats again:
1. Assistant retrieves relevant context from Memorizer.
2. Uses hybrid retrieval: keyword search for "refund", vector fallback for older "delivery delay" issues.
3. Responds with awareness of customer history, without blowing up tokens.

---

## 🛠️ Tech Stack

### **Core Technologies**
- **Language**: Python 3.8+ (3.10+ recommended)
- **Database**: PostgreSQL 12+ with JSONB support
- **Cache & Queues**: Redis 6+
- **Web Framework**: FastAPI with async support
- **Background Jobs**: Celery with Redis broker

### **AI & ML**
- **LLM Providers**: OpenAI, Anthropic, Groq, OpenRouter, Ollama, Custom APIs
- **Vector Databases**: Pinecone, Weaviate, Chroma, pgvector
- **Embedding Providers**: OpenAI, Cohere, HuggingFace, sentence-transformers
- **AI Frameworks**: LangChain, LlamaIndex, AutoGPT, CrewAI

### **Infrastructure & DevOps**
- **Containerization**: Docker with multi-stage builds
- **Orchestration**: Kubernetes with Helm charts
- **Monitoring**: Prometheus, Grafana, Sentry
- **CI/CD**: GitHub Actions with comprehensive testing
- **Cloud**: AWS, Azure, Google Cloud ready

### **Security & Compliance**
- **Authentication**: JWT tokens, API keys
- **Authorization**: Role-based access control (RBAC)
- **Security**: Input validation, XSS/SQL injection protection
- **Compliance**: Audit logging, data encryption

---

## 📂 Repository Structure

```
memorizer/
├── src/                          # Core framework modules
│   ├── memory_manager.py         # Memory lifecycle orchestration
│   ├── db.py                     # Database schema & queries
│   ├── compression_agent.py      # LLM-powered summarization
│   ├── llm_providers.py          # Multi-provider LLM support
│   ├── retrieval.py              # Hybrid context retrieval
│   ├── vector_db.py              # Vector database abstraction
│   ├── embeddings.py             # Embedding providers
│   ├── api.py                    # FastAPI REST interface
│   ├── auth.py                   # Authentication & authorization
│   ├── security.py               # Security & RBAC
│   ├── config.py                 # Configuration management
│   ├── validation.py             # Input validation & sanitization
│   ├── cache.py                  # Redis caching layer
│   ├── rate_limiter.py           # API rate limiting
│   ├── errors.py                 # Error handling framework
│   ├── agent_integrations.py     # AI framework integrations
│   ├── agent_interface.py        # Standardized agent interface
│   ├── memory_templates.py       # Memory structure templates
│   ├── agent_profiles.py         # Agent-specific configurations
│   ├── logging_config.py         # Structured logging
│   ├── tracing_middleware.py     # Request tracing
│   ├── performance_monitor.py    # Performance metrics
│   ├── health_monitor.py         # Health checks
│   ├── automated_testing.py      # Automated testing
│   ├── dashboard.py              # Monitoring dashboard
│   ├── type_checking.py          # Runtime type validation
│   ├── utils.py                  # Utility functions
│   └── tasks/                    # Celery background tasks
│       └── embedding_tasks.py    # Embedding generation tasks
├── examples/                     # Usage examples
│   └── agent_memory_example.py   # Comprehensive example
├── scripts/                      # Database & deployment scripts
│   ├── init_db.py               # Database initialization
│   ├── migrate.py               # Database migrations
│   └── llm_discovery.py         # LLM provider discovery utility
├── k8s/                         # Kubernetes manifests
│   ├── deployment.yaml          # K8s deployment
│   ├── configmap.yaml           # Configuration
│   └── namespace.yaml           # Namespace
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development dependencies
├── requirements-optional.txt    # Optional integrations
├── pyproject.toml              # Project metadata & dependencies
├── Dockerfile                  # Production Docker image
├── Dockerfile.dev              # Development Docker image
├── docker-compose.yml          # Development environment
├── docker-compose.prod.yml     # Production environment
├── .github/workflows/ci.yml    # CI/CD pipeline
├── README.md                   # This file
├── INSTALLATION.md             # Detailed installation guide
├── USAGE.md                    # Usage documentation
├── MONITORING.md               # Monitoring setup guide
├── .env.example                # Environment variables
├── monitoring.env.example      # Monitoring configuration
└── demo.py                     # Quick demo script
```

---

## 📊 Roadmap

### ✅ **Completed Features**
- [x] **Core Memory Management** - Three-tier memory lifecycle with intelligent compression
- [x] **Production Infrastructure** - FastAPI, Celery, Redis, PostgreSQL
- [x] **Security & Authentication** - JWT, API keys, RBAC, input validation
- [x] **Monitoring & Observability** - Prometheus, Grafana, structured logging
- [x] **AI Framework Integrations** - LangChain, LlamaIndex, AutoGPT, CrewAI
- [x] **Vector Database Support** - Pinecone, Weaviate, Chroma, pgvector
- [x] **Docker & Kubernetes** - Production-ready containerization
- [x] **Comprehensive Testing** - Unit, integration, and performance tests

### 🚧 **In Progress**
- [ ] **Advanced Analytics** - Memory usage patterns and optimization insights
- [ ] **Multi-tenant Support** - Isolated memory spaces for different organizations
- [ ] **GraphQL API** - Alternative to REST API for complex queries

### 🔮 **Future Features**
- [ ] **Federated Learning** - Distributed memory learning across agents
- [ ] **Memory Provenance** - Detailed tracking of why memories were kept/removed
- [ ] **Advanced Compression** - Custom compression policies and rules
- [ ] **Memory Visualization** - Interactive dashboards for memory exploration
- [ ] **Edge Computing** - Lightweight version for edge deployments
- [ ] **Memory Marketplace** - Sharing and trading memory insights

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### **Ways to Contribute**
- 🐛 **Report bugs** and suggest features via [GitHub Issues](https://github.com/cyberbeamhq/memorizer/issues)
- 🔧 **Submit pull requests** for bug fixes and new features
- 📚 **Improve documentation** and add usage examples
- 🧪 **Add tests** to improve code coverage
- 🌍 **Add translations** for international users

### **Development Setup**
1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/memorizer.git`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Create a feature branch: `git checkout -b feature/your-feature`
5. Make your changes and add tests
6. Run tests: `pytest`
7. Submit a pull request

### **Code Standards**
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for all public APIs
- Write tests for new functionality
- Update documentation as needed

### **Community**
- 💬 Join our [Discord Community](https://discord.gg/memorizer)
- 📧 Contact us at [team@memorizer.dev](mailto:team@memorizer.dev)
- 🐦 Follow us on [Twitter](https://twitter.com/memorizer_dev)

---

## 📚 Documentation & Support

### **Documentation**
- 📖 **[Full Documentation](https://docs.memorizer.dev)** - Complete API reference and guides
- 🚀 **[Installation Guide](INSTALLATION.md)** - Detailed setup instructions
- 📋 **[Usage Examples](USAGE.md)** - Practical usage scenarios
- 📊 **[Monitoring Guide](MONITORING.md)** - Observability setup

### **Support**
- 🐛 **[Bug Reports](https://github.com/cyberbeamhq/memorizer/issues)** - Report issues
- 💡 **[Feature Requests](https://github.com/cyberbeamhq/memorizer/discussions)** - Suggest new features
- 💬 **[Community Support](https://discord.gg/memorizer)** - Get help from the community
- 📧 **[Enterprise Support](mailto:support@memorizer.dev)** - Commercial support options

---

## 📜 License

MIT License.  
See [LICENSE](./LICENSE) for details.

---

## 🙏 Acknowledgments

- **OpenAI** for providing the GPT models that power our compression
- **FastAPI** team for the excellent web framework
- **PostgreSQL** community for the robust database platform
- **All contributors** who help make Memorizer better

---

<div align="center">

**Made with ❤️ by the Memorizer Team**

[Website](https://memorizer.dev) • [Documentation](https://docs.memorizer.dev) • [GitHub](https://github.com/cyberbeamhq/memorizer) • [Discord](https://discord.gg/memorizer)

</div>
