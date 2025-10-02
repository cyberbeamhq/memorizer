# Supabase Setup Guide for Memorizer

Complete guide to setting up Memorizer with Supabase for production deployment with auth, database, and Edge Functions.

## 🚀 Quick Start

### 1. Prerequisites

- Supabase account ([https://supabase.com](https://supabase.com))
- Supabase CLI installed: `npm install -g supabase`
- Python 3.9+ with pip

### 2. Install Dependencies

```bash
pip install supabase
```

### 3. Create Supabase Project

1. Go to [https://app.supabase.com](https://app.supabase.com)
2. Click "New Project"
3. Fill in project details
4. Save your project URL and keys:
   - `SUPABASE_URL`: Your project URL
   - `SUPABASE_ANON_KEY`: Anon/public key (for client-side)
   - `SUPABASE_SERVICE_ROLE_KEY`: Service role key (for server-side, bypasses RLS)

---

## 📊 Database Setup

### Option 1: Using Supabase Dashboard (Recommended)

1. Go to your project's SQL Editor
2. Copy the entire content from `supabase/migrations/001_initial_schema.sql`
3. Paste and run it

### Option 2: Using Supabase CLI

```bash
# Login
supabase login

# Link to your project
supabase link --project-ref your-project-ref

# Run migrations
supabase db push
```

### What Gets Created:

**Tables:**
- `memories` - Main memory storage with tiered lifecycle
- `memory_embeddings` - Vector embeddings for semantic search
- `compression_jobs` - Background compression tasks
- `user_memory_stats` - Aggregated user statistics
- `agent_sessions` - Agent conversation tracking

**Features:**
- ✅ Row Level Security (RLS) policies
- ✅ Full-text search with PostgreSQL
- ✅ Vector search with pgvector
- ✅ Automatic triggers for stats
- ✅ Indexes for performance
- ✅ Composite indexes for common queries

---

## 🔐 Authentication Setup

Memorizer uses Supabase Auth for user management with RLS.

### Enable Auth Providers

1. Go to Authentication > Providers
2. Enable providers you want:
   - Email/Password (default)
   - Google OAuth
   - GitHub OAuth
   - Magic Link
   - etc.

### Python Integration

```python
from memorizer.integrations.supabase_client import create_supabase_memory_manager

# Initialize manager
manager = create_supabase_memory_manager(
    supabase_url="https://your-project.supabase.co",
    supabase_key="your-anon-key"
)

# Login user (example with email/password)
auth_response = manager.client.auth.sign_in_with_password({
    "email": "user@example.com",
    "password": "password123"
})

# Set auth token for subsequent requests
manager.set_auth_token(auth_response.session.access_token)

# Now all operations respect RLS - user can only access their own data
memory_id = manager.store_memory(
    content="My private memory",
    metadata={"category": "personal"}
)
```

---

## 🤖 LangChain Integration with Supabase

```python
from memorizer.integrations.supabase_client import create_supabase_memory_manager
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory

# Initialize Supabase manager
supabase_manager = create_supabase_memory_manager()

# Authenticate user
auth_response = supabase_manager.client.auth.sign_in_with_password({
    "email": "user@example.com",
    "password": "password"
})
supabase_manager.set_auth_token(auth_response.session.access_token)

# Create agent session
session_id = supabase_manager.create_agent_session(
    session_id="chat_123",
    agent_id="langchain_agent_1",
    agent_type="langchain"
)

# Store memories through agent
memory_id = supabase_manager.store_memory(
    content="User asked about Python",
    session_id="chat_123",
    agent_id="langchain_agent_1",
    source="langchain",
    metadata={"message_type": "human"}
)

# Search memories for context
memories = supabase_manager.search_memories(
    query="Python programming",
    session_id="chat_123",
    limit=5
)

# Get session history
sessions = supabase_manager.get_agent_sessions(is_active=True)
```

---

## ⚡ Edge Functions Setup

### 1. Deploy Compression Edge Function

```bash
# Navigate to your project
cd /path/to/memorizer

# Deploy the function
supabase functions deploy compress-memories

# Set environment variables
supabase secrets set OPENAI_API_KEY=your-openai-key
```

### 2. Schedule Automatic Compression

Run this SQL in Supabase SQL Editor to schedule hourly compression:

```sql
-- Enable pg_cron extension
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Schedule compression to run every hour
SELECT cron.schedule(
    'compress-old-memories',
    '0 * * * *',  -- Every hour
    $$
    SELECT net.http_post(
        url := 'https://your-project.supabase.co/functions/v1/compress-memories',
        headers := jsonb_build_object(
            'Authorization', 'Bearer YOUR_ANON_KEY',
            'Content-Type', 'application/json'
        )
    );
    $$
);

-- Check scheduled jobs
SELECT * FROM cron.job;
```

### 3. Manual Trigger

```bash
# Trigger compression manually
curl -X POST \
  'https://your-project.supabase.co/functions/v1/compress-memories' \
  -H 'Authorization: Bearer YOUR_ANON_KEY' \
  -H 'Content-Type: application/json'
```

---

## 🔍 Vector Search Setup

### 1. Enable pgvector Extension

```sql
-- Run in Supabase SQL Editor
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Store Embeddings

```python
import openai
from memorizer.integrations.supabase_client import create_supabase_memory_manager

manager = create_supabase_memory_manager()

# Store a memory
memory_id = manager.store_memory(content="Python is great for AI")

# Generate embedding (using OpenAI)
openai.api_key = "your-openai-key"
response = openai.Embedding.create(
    input="Python is great for AI",
    model="text-embedding-ada-002"
)
embedding = response['data'][0]['embedding']

# Store embedding in Supabase
manager.client.table("memory_embeddings").insert({
    "memory_id": memory_id,
    "embedding": embedding,
    "model_name": "text-embedding-ada-002"
}).execute()
```

### 3. Semantic Search

```python
# Generate query embedding
query_response = openai.Embedding.create(
    input="AI programming",
    model="text-embedding-ada-002"
)
query_embedding = query_response['data'][0]['embedding']

# Search with vector similarity
results = manager.client.rpc(
    "match_memories",
    {
        "query_embedding": query_embedding,
        "match_threshold": 0.7,
        "match_count": 5
    }
).execute()

# Returns memories ranked by similarity
```

---

## 📈 Row Level Security (RLS) Explained

All tables have RLS enabled. Users can only access their own data.

### How It Works:

```sql
-- When a user is authenticated, Supabase automatically:
-- 1. Sets auth.uid() to the user's ID
-- 2. Applies RLS policies

-- Example: User can only see their own memories
CREATE POLICY "Users can view their own memories"
    ON memories FOR SELECT
    USING (auth.uid() = user_id);
```

### Testing RLS:

```python
# User 1 logs in
manager.client.auth.sign_in_with_password({
    "email": "user1@example.com",
    "password": "pass1"
})

# User 1 stores memory
mem1 = manager.store_memory(content="User 1's private data")

# User 2 logs in
manager.client.auth.sign_in_with_password({
    "email": "user2@example.com",
    "password": "pass2"
})

# User 2 tries to access User 1's memory
memory = manager.get_memory(mem1)  # Returns None - RLS blocks it!
```

---

## 🛡️ Security Best Practices

### 1. Environment Variables

Never commit keys! Use environment variables:

```bash
# .env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key  # Keep secret!
```

### 2. Service Role Key Usage

**NEVER** expose service role key to client-side code!

```python
# ❌ WRONG - Client-side
manager = create_supabase_memory_manager(
    service_role_key="your-service-key"  # Never in browser!
)

# ✅ CORRECT - Server-side only
# Use service role only for:
# - Edge Functions
# - Background jobs
# - Admin operations
```

### 3. RLS Policies

Always keep RLS enabled on tables with user data:

```sql
-- Check RLS status
SELECT tablename, rowsecurity
FROM pg_tables
WHERE schemaname = 'public';

-- Should show rowsecurity = true for all user tables
```

---

## 📊 Monitoring & Analytics

### 1. View Memory Statistics

```python
stats = manager.get_user_stats()
print(f"Total memories: {stats['total_memories']}")
print(f"By tier: {stats['very_new_count']}, {stats['mid_term_count']}, {stats['long_term_count']}")
```

### 2. Query Recent Activity

```sql
-- In Supabase SQL Editor
SELECT * FROM recent_memories LIMIT 10;
```

### 3. Compression Job Monitoring

```sql
SELECT
    status,
    COUNT(*) as count,
    AVG(compression_ratio) as avg_ratio
FROM compression_jobs
GROUP BY status;
```

---

## 🚀 Production Deployment

### 1. Connection Pooling

Supabase provides connection pooling by default. For high-traffic apps:

```python
from memorizer.integrations.supabase_client import create_supabase_memory_manager

# Use connection pooling endpoint
manager = create_supabase_memory_manager(
    supabase_url="https://your-project.supabase.co"  # Automatically pooled
)
```

### 2. Database Backups

Supabase Pro+ includes:
- Daily automatic backups
- Point-in-time recovery
- Manual backup download

### 3. Performance Optimization

```sql
-- Add additional indexes for your query patterns
CREATE INDEX idx_memories_custom
ON memories(user_id, session_id, created_at DESC)
WHERE session_id IS NOT NULL;

-- Analyze query performance
EXPLAIN ANALYZE
SELECT * FROM memories
WHERE user_id = 'user-uuid'
AND session_id = 'session-123'
ORDER BY created_at DESC
LIMIT 10;
```

---

## 🧪 Testing

```python
# test_supabase_integration.py
import pytest
from memorizer.integrations.supabase_client import create_supabase_memory_manager

def test_memory_storage():
    manager = create_supabase_memory_manager()

    # Authenticate test user
    manager.client.auth.sign_in_with_password({
        "email": "test@example.com",
        "password": "testpass123"
    })

    # Store memory
    memory_id = manager.store_memory(
        content="Test memory",
        metadata={"test": True}
    )

    assert memory_id is not None

    # Retrieve memory
    memory = manager.get_memory(memory_id)
    assert memory["content"] == "Test memory"

    # Clean up
    manager.delete_memory(memory_id)
```

---

## 📚 Additional Resources

- [Supabase Documentation](https://supabase.com/docs)
- [pgvector Guide](https://github.com/pgvector/pgvector)
- [Edge Functions Guide](https://supabase.com/docs/guides/functions)
- [RLS Deep Dive](https://supabase.com/docs/guides/auth/row-level-security)

---

## 🆘 Troubleshooting

### "Row Level Security policy violation"

**Problem**: User can't access their own data.

**Solution**: Make sure user is authenticated:

```python
# Check auth status
user = manager.client.auth.get_user()
print(f"Authenticated as: {user.user.email if user.user else 'Not logged in'}")
```

### "Column 'user_id' violates not-null constraint"

**Problem**: RLS not setting user_id automatically.

**Solution**: User must be authenticated before inserting:

```python
# ❌ Wrong - no auth
manager.store_memory(content="test")

# ✅ Correct - authenticated first
manager.client.auth.sign_in_with_password({...})
manager.store_memory(content="test")
```

### "Function search_memories does not exist"

**Problem**: Migration not run completely.

**Solution**: Re-run migration SQL in Supabase dashboard.

---

## ✅ Complete Setup Checklist

- [ ] Create Supabase project
- [ ] Save URL and keys to `.env`
- [ ] Run database migration
- [ ] Enable pgvector extension
- [ ] Configure auth providers
- [ ] Test memory storage with auth
- [ ] Deploy Edge Functions
- [ ] Schedule compression jobs
- [ ] Set up monitoring
- [ ] Configure backups
- [ ] Test RLS policies

🎉 **You're ready to use Memorizer with Supabase in production!**
