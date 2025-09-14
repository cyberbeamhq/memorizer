"""
db.py
Database schema and query helpers for Memorizer.
Handles connections, inserts, updates, and lifecycle transitions.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv

# Load environment variables (DATABASE_URL must be set in .env)
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/memorizer")

# ---------------------------
# Connection Helpers
# ---------------------------

def get_connection():
    """Get a new DB connection."""
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

# ---------------------------
# Schema Setup
# ---------------------------

def init_schema():
    """Initialize the database schema if not exists."""
    queries = [
        """
        CREATE TABLE IF NOT EXISTS very_new (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT NOW()
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS mid_term (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT NOW()
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS long_term (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
    ]
    with get_connection() as conn, conn.cursor() as cur:
        for q in queries:
            cur.execute(q)
        conn.commit()

# ---------------------------
# Insert Helpers
# ---------------------------

def insert_session(user_id: str, content: str, metadata: dict = None, tier: str = "very_new"):
    """Insert a new session into the specified tier."""
    if metadata is None:
        metadata = {}
    query = f"""
        INSERT INTO {tier} (user_id, content, metadata)
        VALUES (%s, %s, %s)
        RETURNING id;
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, (user_id, content, Json(metadata)))
        row = cur.fetchone()
        conn.commit()
        return row["id"]

# ---------------------------
# Fetch Helpers
# ---------------------------

def fetch_memories(user_id: str, tier: str, limit: int = 10):
    """Fetch most recent memories from a tier."""
    query = f"""
        SELECT * FROM {tier}
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT %s;
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, (user_id, limit))
        return cur.fetchall()

# ---------------------------
# Move Between Tiers
# ---------------------------

def move_memory(user_id: str, source: str, target: str, memory_id: int, new_content: str, new_metadata: dict = None):
    """Move memory from one tier to another, compressing content if needed."""
    if new_metadata is None:
        new_metadata = {}

    # Fetch memory from source
    fetch_q = f"SELECT * FROM {source} WHERE id = %s AND user_id = %s;"
    delete_q = f"DELETE FROM {source} WHERE id = %s AND user_id = %s;"
    insert_q = f"""
        INSERT INTO {target} (user_id, content, metadata)
        VALUES (%s, %s, %s)
        RETURNING id;
    """

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(fetch_q, (memory_id, user_id))
        row = cur.fetchone()
        if not row:
            return None  # Memory not found

        # Remove from source
        cur.execute(delete_q, (memory_id, user_id))

        # Insert into target
        cur.execute(insert_q, (user_id, new_content, Json(new_metadata)))
        new_row = cur.fetchone()
        conn.commit()
        return new_row["id"]
