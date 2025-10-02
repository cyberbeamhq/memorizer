-- Memorizer Framework - Supabase Migration
-- Initial schema with RLS policies and optimizations

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text search
CREATE EXTENSION IF NOT EXISTS "vector";   -- For pgvector (if using embeddings)

-- ============================================================================
-- Main Memories Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    -- Content
    content TEXT NOT NULL,
    compressed_content TEXT,  -- For compressed versions

    -- Metadata
    metadata JSONB DEFAULT '{}',

    -- Memory lifecycle
    tier TEXT NOT NULL DEFAULT 'very_new' CHECK (tier IN ('very_new', 'mid_term', 'long_term')),
    compression_ratio FLOAT,
    compression_algorithm TEXT,

    -- Session tracking for agents
    session_id TEXT,
    agent_id TEXT,

    -- Source tracking
    source TEXT DEFAULT 'manual',  -- 'langchain', 'manual', 'crewai', etc.
    source_memory_ids UUID[] DEFAULT ARRAY[]::UUID[],

    -- Access tracking
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Full-text search
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- Indexes for performance
CREATE INDEX idx_memories_user_id ON public.memories(user_id);
CREATE INDEX idx_memories_user_tier ON public.memories(user_id, tier, created_at DESC);
CREATE INDEX idx_memories_session ON public.memories(user_id, session_id) WHERE session_id IS NOT NULL;
CREATE INDEX idx_memories_agent ON public.memories(user_id, agent_id) WHERE agent_id IS NOT NULL;
CREATE INDEX idx_memories_created_at ON public.memories(created_at DESC);
CREATE INDEX idx_memories_metadata ON public.memories USING GIN(metadata);
CREATE INDEX idx_memories_content_tsv ON public.memories USING GIN(content_tsv);
CREATE INDEX idx_memories_tier_created ON public.memories(tier, created_at) WHERE tier IN ('mid_term', 'long_term');

-- Composite index for common queries
CREATE INDEX idx_memories_user_created_tier ON public.memories(user_id, created_at DESC, tier);

-- ============================================================================
-- Memory Embeddings Table (for vector search)
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.memory_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    memory_id UUID NOT NULL REFERENCES public.memories(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    -- Vector embedding (adjust dimension based on your model)
    embedding VECTOR(1536),  -- OpenAI ada-002 uses 1536 dimensions

    -- Metadata
    model_name TEXT NOT NULL DEFAULT 'text-embedding-ada-002',
    embedding_metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Ensure one embedding per memory
    UNIQUE(memory_id, model_name)
);

-- Indexes for vector search
CREATE INDEX idx_embeddings_memory_id ON public.memory_embeddings(memory_id);
CREATE INDEX idx_embeddings_user_id ON public.memory_embeddings(user_id);
CREATE INDEX idx_embeddings_vector ON public.memory_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ============================================================================
-- Compression Jobs Table (for background compression)
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.compression_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    memory_id UUID NOT NULL REFERENCES public.memories(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    -- Job details
    policy_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),

    -- Retry logic
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    next_retry_at TIMESTAMPTZ DEFAULT NOW(),

    -- Results
    original_size INTEGER,
    compressed_size INTEGER,
    compression_ratio FLOAT,
    error_message TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    -- Prevent duplicate jobs
    UNIQUE(memory_id, policy_name)
);

CREATE INDEX idx_compression_jobs_status ON public.compression_jobs(status, next_retry_at);
CREATE INDEX idx_compression_jobs_user ON public.compression_jobs(user_id, status);

-- ============================================================================
-- User Memory Statistics (materialized view for performance)
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.user_memory_stats (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,

    total_memories INTEGER DEFAULT 0,
    very_new_count INTEGER DEFAULT 0,
    mid_term_count INTEGER DEFAULT 0,
    long_term_count INTEGER DEFAULT 0,

    total_size_bytes BIGINT DEFAULT 0,
    compressed_size_bytes BIGINT DEFAULT 0,

    last_updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Agent Sessions Table (for tracking agent conversations)
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.agent_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    session_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    agent_type TEXT,  -- 'langchain', 'crewai', etc.

    -- Session metadata
    metadata JSONB DEFAULT '{}',

    -- Statistics
    message_count INTEGER DEFAULT 0,
    first_message_at TIMESTAMPTZ,
    last_message_at TIMESTAMPTZ,

    -- Status
    is_active BOOLEAN DEFAULT true,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Unique session per user and agent
    UNIQUE(user_id, session_id, agent_id)
);

CREATE INDEX idx_agent_sessions_user ON public.agent_sessions(user_id, is_active);
CREATE INDEX idx_agent_sessions_active ON public.agent_sessions(is_active, last_message_at DESC);

-- ============================================================================
-- Functions and Triggers
-- ============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_memories_updated_at
    BEFORE UPDATE ON public.memories
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_agent_sessions_updated_at
    BEFORE UPDATE ON public.agent_sessions
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

-- Auto-update user statistics
CREATE OR REPLACE FUNCTION public.update_user_memory_stats()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_memory_stats (user_id, total_memories, very_new_count, mid_term_count, long_term_count)
    SELECT
        user_id,
        COUNT(*) as total_memories,
        COUNT(*) FILTER (WHERE tier = 'very_new') as very_new_count,
        COUNT(*) FILTER (WHERE tier = 'mid_term') as mid_term_count,
        COUNT(*) FILTER (WHERE tier = 'long_term') as long_term_count
    FROM public.memories
    WHERE user_id = COALESCE(NEW.user_id, OLD.user_id)
    GROUP BY user_id
    ON CONFLICT (user_id) DO UPDATE SET
        total_memories = EXCLUDED.total_memories,
        very_new_count = EXCLUDED.very_new_count,
        mid_term_count = EXCLUDED.mid_term_count,
        long_term_count = EXCLUDED.long_term_count,
        last_updated_at = NOW();

    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_user_stats_insert
    AFTER INSERT ON public.memories
    FOR EACH ROW
    EXECUTE FUNCTION public.update_user_memory_stats();

CREATE TRIGGER trigger_update_user_stats_update
    AFTER UPDATE ON public.memories
    FOR EACH ROW
    WHEN (OLD.tier IS DISTINCT FROM NEW.tier)
    EXECUTE FUNCTION public.update_user_memory_stats();

CREATE TRIGGER trigger_update_user_stats_delete
    AFTER DELETE ON public.memories
    FOR EACH ROW
    EXECUTE FUNCTION public.update_user_memory_stats();

-- Function to search memories with full-text search
CREATE OR REPLACE FUNCTION public.search_memories(
    p_user_id UUID,
    p_query TEXT,
    p_limit INTEGER DEFAULT 10,
    p_session_id TEXT DEFAULT NULL,
    p_tier TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    metadata JSONB,
    tier TEXT,
    session_id TEXT,
    created_at TIMESTAMPTZ,
    relevance FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.content,
        m.metadata,
        m.tier,
        m.session_id,
        m.created_at,
        ts_rank(m.content_tsv, websearch_to_tsquery('english', p_query)) as relevance
    FROM public.memories m
    WHERE m.user_id = p_user_id
        AND (p_session_id IS NULL OR m.session_id = p_session_id)
        AND (p_tier IS NULL OR m.tier = p_tier)
        AND (p_query = '' OR m.content_tsv @@ websearch_to_tsquery('english', p_query))
    ORDER BY
        CASE
            WHEN p_query = '' THEN m.created_at
            ELSE NULL
        END DESC,
        CASE
            WHEN p_query != '' THEN ts_rank(m.content_tsv, websearch_to_tsquery('english', p_query))
            ELSE NULL
        END DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE SECURITY DEFINER;

-- ============================================================================
-- Row Level Security (RLS) Policies
-- ============================================================================

-- Enable RLS
ALTER TABLE public.memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.memory_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.compression_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_memory_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_sessions ENABLE ROW LEVEL SECURITY;

-- Memories policies
CREATE POLICY "Users can view their own memories"
    ON public.memories FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own memories"
    ON public.memories FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own memories"
    ON public.memories FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own memories"
    ON public.memories FOR DELETE
    USING (auth.uid() = user_id);

-- Memory embeddings policies
CREATE POLICY "Users can view their own embeddings"
    ON public.memory_embeddings FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own embeddings"
    ON public.memory_embeddings FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own embeddings"
    ON public.memory_embeddings FOR DELETE
    USING (auth.uid() = user_id);

-- Compression jobs policies
CREATE POLICY "Users can view their own compression jobs"
    ON public.compression_jobs FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Service role can manage compression jobs"
    ON public.compression_jobs FOR ALL
    USING (auth.jwt()->>'role' = 'service_role');

-- User stats policies
CREATE POLICY "Users can view their own stats"
    ON public.user_memory_stats FOR SELECT
    USING (auth.uid() = user_id);

-- Agent sessions policies
CREATE POLICY "Users can view their own sessions"
    ON public.agent_sessions FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can manage their own sessions"
    ON public.agent_sessions FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- ============================================================================
-- Grant permissions
-- ============================================================================

-- Grant access to authenticated users
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT ALL ON public.memories TO authenticated;
GRANT ALL ON public.memory_embeddings TO authenticated;
GRANT SELECT ON public.compression_jobs TO authenticated;
GRANT SELECT ON public.user_memory_stats TO authenticated;
GRANT ALL ON public.agent_sessions TO authenticated;

-- Grant access to service role for background jobs
GRANT ALL ON ALL TABLES IN SCHEMA public TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO service_role;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO service_role;

-- ============================================================================
-- Useful Views
-- ============================================================================

-- Recent memories view
CREATE OR REPLACE VIEW public.recent_memories AS
SELECT
    m.id,
    m.user_id,
    m.content,
    m.tier,
    m.session_id,
    m.agent_id,
    m.created_at,
    m.access_count
FROM public.memories m
WHERE m.created_at > NOW() - INTERVAL '7 days'
ORDER BY m.created_at DESC;

-- Memory statistics by tier
CREATE OR REPLACE VIEW public.memory_tier_stats AS
SELECT
    user_id,
    tier,
    COUNT(*) as count,
    AVG(LENGTH(content)) as avg_content_length,
    SUM(access_count) as total_accesses
FROM public.memories
GROUP BY user_id, tier;

COMMENT ON TABLE public.memories IS 'Main table storing all user memories with tiered lifecycle management';
COMMENT ON TABLE public.memory_embeddings IS 'Vector embeddings for semantic search';
COMMENT ON TABLE public.compression_jobs IS 'Background jobs for memory compression';
COMMENT ON TABLE public.user_memory_stats IS 'Aggregated statistics per user';
COMMENT ON TABLE public.agent_sessions IS 'Tracking agent conversation sessions';
