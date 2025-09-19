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
- **LLM Providers**: OpenAI (gpt-4o-mini), Cohere, HuggingFace, local models
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
│   └── migrate.py               # Database migrations
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
