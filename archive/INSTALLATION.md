# Installation Guide

This guide covers all the different ways to install and set up Memorizer for various use cases.

## 📋 Prerequisites

- Python 3.8+ (3.10+ recommended)
- PostgreSQL 12+ (for database)
- Redis 6+ (for caching and job queues)
- Git (for cloning the repository)

## 🚀 Installation Methods

### 1. From Source (Recommended for Development)

#### Clone the Repository
```bash
git clone https://github.com/cyberbeamhq/memorizer.git
cd memorizer
```

#### Install Dependencies

**Production Installation:**
```bash
pip install -r requirements.txt
```

**Development Installation:**
```bash
pip install -r requirements-dev.txt
```

**With Optional Features:**
```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

### 2. Using pyproject.toml (Modern Python)

**Production:**
```bash
pip install .
```

**Development:**
```bash
pip install .[dev]
```

**With Specific Optional Features:**
```bash
# Vector databases
pip install .[pinecone,weaviate,chroma,pgvector]

# Monitoring and logging
pip install .[monitoring]

# Async support
pip install .[async]

# Data processing
pip install .[data]

# Web framework integrations
pip install .[web]

# Authentication providers
pip install .[auth]

# Cloud providers
pip install .[cloud]

# Message queues
pip install .[queues]

# AI/ML frameworks
pip install .[ai]

# Deployment tools
pip install .[deploy]

# All optional features
pip install .[all]
```

**Development with All Features:**
```bash
pip install .[dev,all]
```

### 3. Docker Installation

#### Development Environment
```bash
# Clone the repository
git clone https://github.com/cyberbeamhq/memorizer.git
cd memorizer

# Start development environment
docker-compose up -d

# Check logs
docker-compose logs -f api
```

#### Production Environment
```bash
# Clone the repository
git clone https://github.com/cyberbeamhq/memorizer.git
cd memorizer

# Set environment variables
cp .env.example .env
# Edit .env with your production settings

# Start production environment
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose -f docker-compose.prod.yml logs -f api
```

### 4. Virtual Environment Setup

#### Using venv
```bash
# Create virtual environment
python -m venv memorizer-env

# Activate virtual environment
# On Windows:
memorizer-env\Scripts\activate
# On macOS/Linux:
source memorizer-env/bin/activate

# Install dependencies
pip install -r requirements-dev.txt
```

#### Using conda
```bash
# Create conda environment
conda create -n memorizer python=3.11

# Activate environment
conda activate memorizer

# Install dependencies
pip install -r requirements-dev.txt
```

## 🔧 Configuration

### 1. Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your configuration:
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/memorizer

# Redis
REDIS_URL=redis://localhost:6379/0

# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Optional: Vector Database
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment

# Optional: Monitoring
SENTRY_DSN=your_sentry_dsn
PROMETHEUS_ENABLED=true
```

### 2. Database Setup

#### Initialize Database
```bash
python scripts/init_db.py
```

#### Run Migrations
```bash
python scripts/migrate.py
```

### 3. Optional Services Setup

#### Vector Database Setup

**Pinecone:**
1. Create account at [pinecone.io](https://pinecone.io)
2. Create a new project and index
3. Add API key to `.env`

**Weaviate:**
1. Install Weaviate locally or use cloud service
2. Update `WEAVIATE_URL` in `.env`

**Chroma:**
1. Install ChromaDB: `pip install chromadb`
2. ChromaDB runs locally by default

**pgvector:**
1. Install pgvector extension in PostgreSQL
2. Update `DATABASE_URL` to include pgvector

#### Monitoring Setup

**Prometheus & Grafana:**
```bash
# Start monitoring stack
docker-compose up -d prometheus grafana

# Access Grafana at http://localhost:3000
# Default credentials: admin/admin
```

**Sentry:**
1. Create account at [sentry.io](https://sentry.io)
2. Add DSN to `.env`

## 🧪 Verification

### 1. Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m slow
```

### 2. Run Demo
```bash
python demo.py
```

### 3. Check API Health
```bash
# Start the API
python -m uvicorn src.api:app --reload

# Check health endpoint
curl http://localhost:8000/health
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Database Connection Issues
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Check connection string format
# Should be: postgresql://user:password@host:port/database
```

#### 2. Redis Connection Issues
```bash
# Check Redis is running
redis-cli ping

# Should return: PONG
```

#### 3. Import Errors
```bash
# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or install in development mode
pip install -e .
```

#### 4. Permission Issues
```bash
# Check file permissions
ls -la src/

# Fix permissions if needed
chmod -R 755 src/
```

### Getting Help

1. Check the [Issues](https://github.com/cyberbeamhq/memorizer/issues) page
2. Review the [Documentation](https://docs.memorizer.dev)
3. Join our [Discord Community](https://discord.gg/memorizer)

## 📦 Package Management

### Updating Dependencies

#### Update requirements files
```bash
# Update all dependencies
pip-compile requirements.in
pip-compile requirements-dev.in
pip-compile requirements-optional.in
```

#### Update pyproject.toml
```bash
# Update dependencies in pyproject.toml
pip install --upgrade pip-tools
pip-compile pyproject.toml
```

### Security Updates
```bash
# Check for security vulnerabilities
safety check

# Update vulnerable packages
pip install --upgrade package-name
```

## 🚀 Next Steps

After installation:

1. **Read the [Quickstart Guide](README.md#quickstart)**
2. **Explore [Examples](examples/)**
3. **Check [API Documentation](docs/api.md)**
4. **Join the [Community](https://discord.gg/memorizer)**

## 📚 Additional Resources

- [Configuration Guide](docs/configuration.md)
- [API Reference](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
