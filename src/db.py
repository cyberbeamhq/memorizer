"""
db.py
Database schema and query helpers for Memorizer.
Handles connections, inserts, updates, and lifecycle transitions.
"""
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv

# Load environment variables (DATABASE_URL must be set in .env)
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/memorizer")

logger = logging.getLogger(__name__)

# Connection pool for better performance and connection management
_connection_pool = None

# ---------------------------
# Connection Helpers
# ---------------------------
def init_connection_pool(min_conn: int = 1, max_conn: int = 10):
    """Initialize connection pool."""
    global _connection_pool
    try:
        _connection_pool = SimpleConnectionPool(
            min_conn, max_conn, DATABASE_URL, cursor_factory=RealDictCursor
        )
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {e}")
        raise

@contextmanager
def get_connection():
    """Get a connection from the pool with proper cleanup."""
    if _connection_pool is None:
        init_connection_pool()
    
    conn = None
    try:
        conn = _connection_pool.getconn()
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            _connection_pool.putconn(conn)

# ---------------------------
# Schema Setup
# ---------------------------
def init_schema():
    """Initialize the database schema if not exists."""
    # Use a unified table with tier column instead of separate tables
    # This is more maintainable and allows for easier queries across tiers
    schema_queries = [
        """
        CREATE TABLE IF NOT EXISTS memories (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            tier TEXT NOT NULL CHECK (tier IN ('very_new', 'mid_term', 'long_term')),
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            source_memory_ids INTEGER[] DEFAULT ARRAY[]::INTEGER[]
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memories_user_tier 
        ON memories(user_id, tier, created_at DESC);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memories_user_created 
        ON memories(user_id, created_at DESC);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memories_metadata 
        ON memories USING GIN(metadata);
        """,
        """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        """,
        """
        DROP TRIGGER IF EXISTS update_memories_updated_at ON memories;
        CREATE TRIGGER update_memories_updated_at 
        BEFORE UPDATE ON memories 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """
    ]
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for query in schema_queries:
                    cur.execute(query)
                conn.commit()
        logger.info("Database schema initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize schema: {e}")
        raise

def migrate_from_old_schema():
    """Migrate from old separate table schema to unified schema."""
    migration_queries = [
        """
        INSERT INTO memories (user_id, content, metadata, tier, created_at)
        SELECT user_id, content, metadata, 'very_new', created_at 
        FROM very_new 
        WHERE NOT EXISTS (
            SELECT 1 FROM memories 
            WHERE memories.user_id = very_new.user_id 
            AND memories.content = very_new.content 
            AND memories.tier = 'very_new'
        );
        """,
        """
        INSERT INTO memories (user_id, content, metadata, tier, created_at)
        SELECT user_id, content, metadata, 'mid_term', created_at 
        FROM mid_term 
        WHERE NOT EXISTS (
            SELECT 1 FROM memories 
            WHERE memories.user_id = mid_term.user_id 
            AND memories.content = mid_term.content 
            AND memories.tier = 'mid_term'
        );
        """,
        """
        INSERT INTO memories (user_id, content, metadata, tier, created_at)
        SELECT user_id, content, metadata, 'long_term', created_at 
        FROM long_term 
        WHERE NOT EXISTS (
            SELECT 1 FROM memories 
            WHERE memories.user_id = long_term.user_id 
            AND memories.content = long_term.content 
            AND memories.tier = 'long_term'
        );
        """,
        "DROP TABLE IF EXISTS very_new;",
        "DROP TABLE IF EXISTS mid_term;",
        "DROP TABLE IF EXISTS long_term;"
    ]
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for query in migration_queries:
                    cur.execute(query)
                conn.commit()
        logger.info("Migration from old schema completed")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

# ---------------------------
# Insert Helpers
# ---------------------------
def insert_session(
    user_id: str, 
    content: str, 
    metadata: Dict[str, Any] = None, 
    tier: str = "very_new",
    source_memory_ids: List[int] = None
) -> str:
    """
    Insert a new session into the specified tier.
    
    Args:
        user_id: User identifier
        content: Session content
        metadata: Additional metadata
        tier: Memory tier (very_new, mid_term, long_term)
        source_memory_ids: List of memory IDs that were compressed to create this
        
    Returns:
        The ID of the inserted memory
    """
    if metadata is None:
        metadata = {}
    if source_memory_ids is None:
        source_memory_ids = []
        
    # Validate tier
    if tier not in ['very_new', 'mid_term', 'long_term']:
        raise ValueError(f"Invalid tier: {tier}")
    
    query = """
        INSERT INTO memories (user_id, content, metadata, tier, source_memory_ids)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
    """
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (user_id, content, Json(metadata), tier, source_memory_ids))
                row = cur.fetchone()
                conn.commit()
                return str(row["id"])
    except Exception as e:
        logger.error(f"Failed to insert session for user {user_id}: {e}")
        raise

# ---------------------------
# Fetch Helpers
# ---------------------------
def fetch_memories(
    user_id: str, 
    tier: str = None, 
    limit: int = None,
    offset: int = 0,
    order_by: str = "created_at DESC"
) -> List[Dict[str, Any]]:
    """
    Fetch memories from specified tier(s).
    
    Args:
        user_id: User identifier
        tier: Specific tier to fetch from (None for all tiers)
        limit: Maximum number of records to return
        offset: Number of records to skip
        order_by: Order clause
        
    Returns:
        List of memory dictionaries
    """
    conditions = ["user_id = %s"]
    params = [user_id]
    
    if tier:
        if tier not in ['very_new', 'mid_term', 'long_term']:
            raise ValueError(f"Invalid tier: {tier}")
        conditions.append("tier = %s")
        params.append(tier)
    
    query = f"""
        SELECT id, user_id, content, metadata, tier, created_at, updated_at, source_memory_ids
        FROM memories
        WHERE {' AND '.join(conditions)}
        ORDER BY {order_by}
    """
    
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)
    
    if offset > 0:
        query += " OFFSET %s"
        params.append(offset)
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Failed to fetch memories for user {user_id}: {e}")
        raise

def search_memories(
    user_id: str,
    search_term: str,
    tier: str = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search memories by content using full-text search.
    
    Args:
        user_id: User identifier
        search_term: Text to search for
        tier: Specific tier to search in
        limit: Maximum results to return
        
    Returns:
        List of matching memories with relevance scores
    """
    conditions = ["user_id = %s"]
    params = [user_id]
    
    if tier:
        conditions.append("tier = %s")
        params.append(tier)
    
    # Use PostgreSQL full-text search
    conditions.append("to_tsvector('english', content) @@ plainto_tsquery('english', %s)")
    params.append(search_term)
    
    query = f"""
        SELECT *, 
               ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) as relevance_score
        FROM memories
        WHERE {' AND '.join(conditions)}
        ORDER BY relevance_score DESC, created_at DESC
        LIMIT %s;
    """
    params.append(search_term)  # for ts_rank
    params.append(limit)
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Failed to search memories for user {user_id}: {e}")
        return []

# ---------------------------
# Move Between Tiers
# ---------------------------
def move_memory(
    user_id: str, 
    source_tier: str, 
    target_tier: str, 
    memory_id: int, 
    new_content: str, 
    new_metadata: Dict[str, Any] = None
) -> str:
    """
    Move memory from one tier to another, updating content and metadata.
    
    Args:
        user_id: User identifier
        source_tier: Source tier name
        target_tier: Target tier name
        memory_id: ID of memory to move
        new_content: Updated content (compressed)
        new_metadata: Updated metadata
        
    Returns:
        ID of the updated memory record
    """
    if new_metadata is None:
        new_metadata = {}
        
    # Validate tiers
    valid_tiers = ['very_new', 'mid_term', 'long_term']
    if source_tier not in valid_tiers or target_tier not in valid_tiers:
        raise ValueError("Invalid tier specified")
    
    query = """
        UPDATE memories 
        SET tier = %s, content = %s, metadata = %s, updated_at = NOW()
        WHERE id = %s AND user_id = %s AND tier = %s
        RETURNING id;
    """
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (target_tier, new_content, Json(new_metadata), memory_id, user_id, source_tier))
                row = cur.fetchone()
                if not row:
                    raise ValueError(f"Memory {memory_id} not found in {source_tier} for user {user_id}")
                conn.commit()
                return str(row["id"])
    except Exception as e:
        logger.error(f"Failed to move memory {memory_id}: {e}")
        raise

def delete_memory(user_id: str, memory_id: int) -> bool:
    """Delete a specific memory."""
    query = "DELETE FROM memories WHERE id = %s AND user_id = %s;"
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (memory_id, user_id))
                deleted = cur.rowcount > 0
                conn.commit()
                return deleted
    except Exception as e:
        logger.error(f"Failed to delete memory {memory_id}: {e}")
        raise

def delete_old_memories(user_id: str, cutoff_date: datetime) -> int:
    """Delete memories older than the cutoff date."""
    query = "DELETE FROM memories WHERE user_id = %s AND created_at < %s;"
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (user_id, cutoff_date))
                deleted_count = cur.rowcount
                conn.commit()
                return deleted_count
    except Exception as e:
        logger.error(f"Failed to delete old memories for user {user_id}: {e}")
        raise

# ---------------------------
# Analytics and Stats
# ---------------------------
def get_user_memory_stats(user_id: str) -> Dict[str, Any]:
    """Get comprehensive memory statistics for a user."""
    query = """
        SELECT 
            tier,
            COUNT(*) as count,
            AVG(LENGTH(content)) as avg_content_length,
            MIN(created_at) as oldest_memory,
            MAX(created_at) as newest_memory
        FROM memories 
        WHERE user_id = %s 
        GROUP BY tier;
    """
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (user_id,))
                rows = cur.fetchall()
                
                stats = {
                    'very_new': {'count': 0, 'avg_length': 0},
                    'mid_term': {'count': 0, 'avg_length': 0},
                    'long_term': {'count': 0, 'avg_length': 0},
                    'total_memories': 0
                }
                
                for row in rows:
                    tier_stats = {
                        'count': row['count'],
                        'avg_length': int(row['avg_content_length'] or 0),
                        'oldest': row['oldest_memory'],
                        'newest': row['newest_memory']
                    }
                    stats[row['tier']] = tier_stats
                    stats['total_memories'] += row['count']
                
                return stats
    except Exception as e:
        logger.error(f"Failed to get stats for user {user_id}: {e}")
        return {}

# ---------------------------
# Initialization
# ---------------------------
def initialize_db():
    """Initialize the database with schema and connection pool."""
    try:
        init_connection_pool()
        init_schema()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
