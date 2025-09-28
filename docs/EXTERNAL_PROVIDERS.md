# External Providers Guide

This guide explains how to use external database and vector store providers with the Memorizer framework. These integrations allow you to easily connect to third-party services like Supabase, Railway, Pinecone, and more.

## 🗄️ Database Providers

### Supabase

[Supabase](https://supabase.io) is an open-source Firebase alternative with PostgreSQL backend.

#### Configuration

```yaml
storage:
  name: supabase
  config:
    supabase:
      project_url: "${SUPABASE_URL}"
      anon_key: "${SUPABASE_ANON_KEY}"
      service_role_key: "${SUPABASE_SERVICE_ROLE_KEY}"
      database_password: "${SUPABASE_DB_PASSWORD}"
```

#### Environment Variables

```bash
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_DB_PASSWORD=your_database_password

# Optional (for API access)
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

#### Setup Steps

1. Create a Supabase project at [supabase.io](https://supabase.io)
2. Get your project URL from the dashboard
3. Set a strong database password
4. Copy the environment variables to your `.env` file

### Railway

[Railway](https://railway.app) provides simple PostgreSQL hosting.

#### Configuration

```yaml
storage:
  name: railway
  config:
    railway:
      database_url: "${RAILWAY_DATABASE_URL}"
      # Or individual components:
      database_host: "${RAILWAY_PGHOST}"
      database_port: "${RAILWAY_PGPORT}"
      database_user: "${RAILWAY_PGUSER}"
      database_password: "${RAILWAY_PGPASSWORD}"
      database_name: "${RAILWAY_PGDATABASE}"
```

#### Environment Variables

```bash
# Option 1: Full connection string
RAILWAY_DATABASE_URL=postgresql://user:password@host:port/database

# Option 2: Individual components
RAILWAY_PGHOST=containers-us-west-123.railway.app
RAILWAY_PGPORT=5432
RAILWAY_PGUSER=postgres
RAILWAY_PGPASSWORD=your_password
RAILWAY_PGDATABASE=railway
```

### Neon

[Neon](https://neon.tech) is a serverless PostgreSQL platform.

#### Configuration

```yaml
storage:
  name: neon
  config:
    neon:
      connection_string: "${NEON_DATABASE_URL}"
```

#### Environment Variables

```bash
NEON_DATABASE_URL=postgresql://user:password@ep-xxx.us-east-1.aws.neon.tech/neondb?sslmode=require
```

### PlanetScale

[PlanetScale](https://planetscale.com) provides serverless MySQL (Note: Use with caution as framework is optimized for PostgreSQL).

#### Configuration

```yaml
storage:
  name: planetscale
  config:
    planetscale:
      host: "${PLANETSCALE_HOST}"
      username: "${PLANETSCALE_USERNAME}"
      password: "${PLANETSCALE_PASSWORD}"
      database: "${PLANETSCALE_DATABASE}"
```

#### Environment Variables

```bash
PLANETSCALE_HOST=your-host.planetscale.com
PLANETSCALE_USERNAME=your_username
PLANETSCALE_PASSWORD=your_password
PLANETSCALE_DATABASE=your_database
```

### CockroachDB

[CockroachDB](https://cockroachlabs.com) is a distributed SQL database.

#### Configuration

```yaml
storage:
  name: cockroachdb
  config:
    cockroachdb:
      connection_string: "${COCKROACH_DATABASE_URL}"
```

#### Environment Variables

```bash
COCKROACH_DATABASE_URL=postgresql://user:password@host:26257/database?sslmode=require
```

## 🔍 Vector Store Providers

### Pinecone

[Pinecone](https://pinecone.io) is a managed vector database service.

#### Configuration

```yaml
vector_store:
  name: pinecone
  config:
    pinecone:
      api_key: "${PINECONE_API_KEY}"
      environment: "${PINECONE_ENVIRONMENT}"
      index_name: "memorizer"
```

#### Environment Variables

```bash
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=us-west1-gcp  # or your preferred environment
```

#### Setup Steps

1. Sign up at [pinecone.io](https://pinecone.io)
2. Create a new index with dimension 1536 (for OpenAI embeddings)
3. Copy your API key and environment
4. The framework will automatically create the index if it doesn't exist

### Weaviate

[Weaviate](https://weaviate.io) is an open-source vector search engine.

#### Configuration

```yaml
vector_store:
  name: weaviate
  config:
    weaviate:
      url: "${WEAVIATE_URL}"
      api_key: "${WEAVIATE_API_KEY}"  # Optional for cloud instances
      class_name: "MemorizerMemory"
```

#### Environment Variables

```bash
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your_api_key  # For Weaviate Cloud
```

#### Setup Steps

1. Use Weaviate Cloud or self-hosted instance
2. For cloud: Create cluster at [console.weaviate.cloud](https://console.weaviate.cloud)
3. Get your cluster URL and API key
4. The framework will automatically create the required schema

### Chroma

[Chroma](https://www.trychroma.com) is an open-source embedding database.

#### Configuration

```yaml
vector_store:
  name: chroma
  config:
    chroma:
      persist_directory: "./chroma_db"
      collection_name: "memorizer_memories"
```

#### Setup Steps

1. Install ChromaDB: `pip install chromadb`
2. Choose a persistence directory
3. The framework will create the collection automatically

## 📋 Complete Example Configurations

### Option 1: Supabase + Pinecone (Recommended)

```yaml
# memorizer.yaml
storage:
  name: supabase
  config:
    supabase:
      project_url: "${SUPABASE_URL}"
      database_password: "${SUPABASE_DB_PASSWORD}"

vector_store:
  name: pinecone
  config:
    pinecone:
      api_key: "${PINECONE_API_KEY}"
      environment: "${PINECONE_ENVIRONMENT}"
      index_name: "memorizer"
```

```bash
# .env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_DB_PASSWORD=your_secure_password
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-west1-gcp
```

### Option 2: Railway + Weaviate

```yaml
# memorizer.yaml
storage:
  name: railway
  config:
    railway:
      database_url: "${RAILWAY_DATABASE_URL}"

vector_store:
  name: weaviate
  config:
    weaviate:
      url: "${WEAVIATE_URL}"
      api_key: "${WEAVIATE_API_KEY}"
```

### Option 3: Neon + Local Chroma

```yaml
# memorizer.yaml
storage:
  name: neon
  config:
    neon:
      connection_string: "${NEON_DATABASE_URL}"

vector_store:
  name: chroma
  config:
    chroma:
      persist_directory: "./data/chroma"
```

## 🚀 Deployment Guide

### 1. Development Setup

```bash
# Copy example configuration
cp memorizer.yaml.example memorizer.yaml

# Edit configuration to use your preferred providers
vim memorizer.yaml

# Set environment variables
cp .env.example .env
vim .env

# Install optional dependencies
pip install pinecone-client weaviate-client chromadb supabase
```

### 2. Production Deployment

#### Docker with External Providers

```dockerfile
# Dockerfile
FROM python:3.11

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install external provider dependencies
RUN pip install pinecone-client weaviate-client supabase psycopg2-binary

COPY . .
EXPOSE 8000
CMD ["uvicorn", "memorizer.api.framework_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Environment Variables for Production

```bash
# Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_DB_PASSWORD=your_secure_password

# Vector Store
PINECONE_API_KEY=your_production_api_key
PINECONE_ENVIRONMENT=us-west1-gcp

# LLM Provider
OPENAI_API_KEY=your_openai_key

# Security
JWT_SECRET_KEY=your_jwt_secret
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Install missing dependencies
pip install pinecone-client  # For Pinecone
pip install weaviate-client  # For Weaviate
pip install chromadb        # For Chroma
pip install supabase        # For Supabase API access
```

#### 2. Connection Errors

- **Database**: Check connection string format and credentials
- **Vector Store**: Verify API keys and service availability
- **Network**: Ensure firewall allows outbound connections

#### 3. Performance Issues

- **Database**: Check connection pooling settings
- **Vector Store**: Monitor API rate limits
- **Memory**: Increase memory allocation for large embeddings

### Health Checks

```python
# Check provider health
from memorizer import create_framework, FrameworkConfig

config = FrameworkConfig.from_file("memorizer.yaml")
framework = create_framework(config)

# Get comprehensive health status
health = framework.get_health_status()
print(health)
```

## 📊 Cost Optimization

### Database Providers

- **Supabase**: Free tier includes 500MB storage + 2GB bandwidth
- **Railway**: Usage-based pricing, ~$5/month for small apps
- **Neon**: Free tier with 0.5GB storage, good for development
- **PlanetScale**: Free tier with 1 billion row reads/month

### Vector Store Providers

- **Pinecone**: Free tier with 1M vectors, paid plans start at $70/month
- **Weaviate Cloud**: Pay-as-you-go pricing
- **Chroma**: Self-hosted = free, cloud pricing varies

### Recommendations

1. **Development**: Neon (DB) + Chroma (Vector)
2. **Small Production**: Supabase (DB) + Pinecone (Vector)
3. **Large Scale**: Railway/CockroachDB (DB) + Weaviate (Vector)

## 🛡️ Security Best Practices

1. **Environment Variables**: Never commit API keys to version control
2. **Network Security**: Use SSL/TLS for all connections
3. **Access Control**: Use service accounts with minimal permissions
4. **Monitoring**: Set up alerts for unusual usage patterns
5. **Backup**: Regular backups of both database and vector data

## 📚 Additional Resources

- [Supabase Documentation](https://supabase.io/docs)
- [Railway Documentation](https://docs.railway.app)
- [Pinecone Documentation](https://docs.pinecone.io)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Memorizer Framework Docs](https://docs.memorizer.ai)

For more examples, see `examples/external_providers_example.py` in the repository.