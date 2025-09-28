"""
Built-in Storage Components
Provides default storage implementations for the Memorizer framework.
"""

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

from ..core.interfaces import Memory, RetrievalResult

logger = logging.getLogger(__name__)


class BaseStorage(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def store_memory(self, memory: Memory) -> str:
        """Store a memory and return its ID."""
        pass

    @abstractmethod
    def get_memory(self, memory_id: str, user_id: str) -> Optional[Memory]:
        """Retrieve a specific memory by ID."""
        pass

    @abstractmethod
    def search_memories(
        self,
        user_id: str,
        query: str = "",
        limit: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Memory]:
        """Search for memories matching criteria."""
        pass

    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        user_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing memory."""
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a memory."""
        pass

    @abstractmethod
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a user's memories."""
        pass

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the storage backend."""
        return {"status": "healthy", "type": self.__class__.__name__}


class MemoryStorage(BaseStorage):
    """In-memory storage backend for development and testing."""

    def __init__(self, **kwargs):
        self._memories: Dict[str, Memory] = {}
        self._user_memories: Dict[str, List[str]] = {}
        self._lock = threading.RLock()

    def store_memory(self, memory: Memory) -> str:
        """Store a memory in memory."""
        with self._lock:
            memory_id = memory.id or str(uuid.uuid4())

            # Create a new memory with the generated ID
            stored_memory = Memory(
                id=memory_id,
                user_id=memory.user_id,
                content=memory.content,
                metadata=memory.metadata or {},
                created_at=memory.created_at or datetime.now().timestamp(),
                updated_at=datetime.now().timestamp()
            )

            # Store the memory
            self._memories[memory_id] = stored_memory

            # Track user memories
            if memory.user_id not in self._user_memories:
                self._user_memories[memory.user_id] = []
            self._user_memories[memory.user_id].append(memory_id)

            logger.debug(f"Stored memory {memory_id} for user {memory.user_id}")
            return memory_id

    def get_memory(self, memory_id: str, user_id: str) -> Optional[Memory]:
        """Retrieve a specific memory by ID."""
        with self._lock:
            memory = self._memories.get(memory_id)
            if memory and memory.user_id == user_id:
                return memory
            return None

    def search_memories(
        self,
        user_id: str,
        query: str = "",
        limit: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Memory]:
        """Search for memories matching criteria."""
        with self._lock:
            user_memory_ids = self._user_memories.get(user_id, [])
            matching_memories = []

            for memory_id in user_memory_ids:
                memory = self._memories.get(memory_id)
                if not memory:
                    continue

                # Apply metadata filters
                if metadata_filters:
                    match = True
                    for key, value in metadata_filters.items():
                        if memory.metadata.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue

                # Apply content search
                if query:
                    query_lower = query.lower()
                    if query_lower in memory.content.lower():
                        matching_memories.append(memory)
                else:
                    matching_memories.append(memory)

            # Sort by creation time (newest first) and limit
            matching_memories.sort(key=lambda m: m.created_at, reverse=True)
            return matching_memories[:limit]

    def update_memory(
        self,
        memory_id: str,
        user_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing memory."""
        with self._lock:
            memory = self._memories.get(memory_id)
            if not memory or memory.user_id != user_id:
                return False

            # Update content if provided
            if content is not None:
                memory.content = content

            # Update metadata if provided
            if metadata is not None:
                memory.metadata.update(metadata)

            # Update timestamp
            memory.updated_at = datetime.now()

            logger.debug(f"Updated memory {memory_id} for user {user_id}")
            return True

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a memory."""
        with self._lock:
            memory = self._memories.get(memory_id)
            if not memory or memory.user_id != user_id:
                return False

            # Remove from memories
            del self._memories[memory_id]

            # Remove from user memories
            if user_id in self._user_memories:
                try:
                    self._user_memories[user_id].remove(memory_id)
                except ValueError:
                    pass

            logger.debug(f"Deleted memory {memory_id} for user {user_id}")
            return True

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a user's memories."""
        with self._lock:
            user_memory_ids = self._user_memories.get(user_id, [])
            total_memories = len(user_memory_ids)

            if total_memories == 0:
                return {
                    "total_memories": 0,
                    "total_content_length": 0,
                    "average_content_length": 0,
                    "oldest_memory": None,
                    "newest_memory": None
                }

            memories = [self._memories[mid] for mid in user_memory_ids if mid in self._memories]
            total_content_length = sum(len(m.content) for m in memories)

            memories_by_date = sorted(memories, key=lambda m: m.created_at)

            return {
                "total_memories": total_memories,
                "total_content_length": total_content_length,
                "average_content_length": total_content_length // total_memories,
                "oldest_memory": memories_by_date[0].created_at.isoformat() if memories_by_date else None,
                "newest_memory": memories_by_date[-1].created_at.isoformat() if memories_by_date else None
            }


class SQLiteStorage(BaseStorage):
    """SQLite storage backend for persistent storage."""

    def __init__(self, db_path: str = "memories.db", **kwargs):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_user_id
                ON memories (user_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_created_at
                ON memories (created_at)
            """)

    def store_memory(self, memory: Memory) -> str:
        """Store a memory in SQLite."""
        with self._lock:
            memory_id = memory.id or str(uuid.uuid4())
            created_at = (memory.created_at or datetime.now()).isoformat()
            updated_at = datetime.now().isoformat()

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO memories (id, user_id, content, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        memory.user_id,
                        memory.content,
                        json.dumps(memory.metadata or {}),
                        created_at,
                        updated_at
                    )
                )

            logger.debug(f"Stored memory {memory_id} for user {memory.user_id}")
            return memory_id

    def get_memory(self, memory_id: str, user_id: str) -> Optional[Memory]:
        """Retrieve a specific memory by ID."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM memories WHERE id = ? AND user_id = ?",
                    (memory_id, user_id)
                )
                row = cursor.fetchone()

                if row:
                    return Memory(
                        id=row["id"],
                        user_id=row["user_id"],
                        content=row["content"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
                    )
                return None

    def search_memories(
        self,
        user_id: str,
        query: str = "",
        limit: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Memory]:
        """Search for memories matching criteria."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                sql = "SELECT * FROM memories WHERE user_id = ?"
                params = [user_id]

                if query:
                    sql += " AND content LIKE ?"
                    params.append(f"%{query}%")

                sql += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)

                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()

                memories = []
                for row in rows:
                    memory = Memory(
                        id=row["id"],
                        user_id=row["user_id"],
                        content=row["content"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
                    )

                    # Apply metadata filters if provided
                    if metadata_filters:
                        match = True
                        for key, value in metadata_filters.items():
                            if memory.metadata.get(key) != value:
                                match = False
                                break
                        if match:
                            memories.append(memory)
                    else:
                        memories.append(memory)

                return memories

    def update_memory(
        self,
        memory_id: str,
        user_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing memory."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # First, get the current memory
                cursor = conn.execute(
                    "SELECT * FROM memories WHERE id = ? AND user_id = ?",
                    (memory_id, user_id)
                )
                row = cursor.fetchone()

                if not row:
                    return False

                # Prepare updates
                updated_at = datetime.now().isoformat()
                new_content = content if content is not None else row[2]  # row[2] is content

                current_metadata = json.loads(row[3]) if row[3] else {}  # row[3] is metadata
                if metadata:
                    current_metadata.update(metadata)
                new_metadata = json.dumps(current_metadata)

                # Update the memory
                conn.execute(
                    """
                    UPDATE memories
                    SET content = ?, metadata = ?, updated_at = ?
                    WHERE id = ? AND user_id = ?
                    """,
                    (new_content, new_metadata, updated_at, memory_id, user_id)
                )

                logger.debug(f"Updated memory {memory_id} for user {user_id}")
                return True

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a memory."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM memories WHERE id = ? AND user_id = ?",
                    (memory_id, user_id)
                )

                if cursor.rowcount > 0:
                    logger.debug(f"Deleted memory {memory_id} for user {user_id}")
                    return True
                return False

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a user's memories."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_memories,
                        SUM(LENGTH(content)) as total_content_length,
                        MIN(created_at) as oldest_memory,
                        MAX(created_at) as newest_memory
                    FROM memories
                    WHERE user_id = ?
                    """,
                    (user_id,)
                )
                row = cursor.fetchone()

                total_memories = row[0] or 0
                total_content_length = row[1] or 0
                average_content_length = total_content_length // total_memories if total_memories > 0 else 0

                return {
                    "total_memories": total_memories,
                    "total_content_length": total_content_length,
                    "average_content_length": average_content_length,
                    "oldest_memory": row[2],
                    "newest_memory": row[3]
                }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the SQLite storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM memories")
                total_memories = cursor.fetchone()[0]

                return {
                    "status": "healthy",
                    "type": "SQLiteStorage",
                    "total_memories": total_memories,
                    "db_path": self.db_path
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "type": "SQLiteStorage",
                "error": str(e)
            }


# Mock PostgreSQL storage for when psycopg2 is not available
class PostgreSQLStorage(BaseStorage):
    """PostgreSQL storage backend (mock implementation)."""

    def __init__(self, connection_string: str = "", **kwargs):
        self.connection_string = connection_string
        logger.warning("PostgreSQL storage is not fully implemented yet. Using SQLite fallback.")
        self._fallback = SQLiteStorage(**kwargs)

    def store_memory(self, memory: Memory) -> str:
        return self._fallback.store_memory(memory)

    def get_memory(self, memory_id: str, user_id: str) -> Optional[Memory]:
        return self._fallback.get_memory(memory_id, user_id)

    def search_memories(
        self,
        user_id: str,
        query: str = "",
        limit: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Memory]:
        return self._fallback.search_memories(user_id, query, limit, metadata_filters)

    def update_memory(
        self,
        memory_id: str,
        user_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        return self._fallback.update_memory(memory_id, user_id, content, metadata)

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        return self._fallback.delete_memory(memory_id, user_id)

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        return self._fallback.get_user_stats(user_id)


# Mock MongoDB storage for when pymongo is not available
class MongoDBStorage(BaseStorage):
    """MongoDB storage backend (mock implementation)."""

    def __init__(self, connection_string: str = "", **kwargs):
        self.connection_string = connection_string
        logger.warning("MongoDB storage is not fully implemented yet. Using SQLite fallback.")
        self._fallback = SQLiteStorage(**kwargs)

    def store_memory(self, memory: Memory) -> str:
        return self._fallback.store_memory(memory)

    def get_memory(self, memory_id: str, user_id: str) -> Optional[Memory]:
        return self._fallback.get_memory(memory_id, user_id)

    def search_memories(
        self,
        user_id: str,
        query: str = "",
        limit: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Memory]:
        return self._fallback.search_memories(user_id, query, limit, metadata_filters)

    def update_memory(
        self,
        memory_id: str,
        user_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        return self._fallback.update_memory(memory_id, user_id, content, metadata)

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        return self._fallback.delete_memory(memory_id, user_id)

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        return self._fallback.get_user_stats(user_id)


__all__ = [
    "BaseStorage",
    "MemoryStorage",
    "SQLiteStorage",
    "PostgreSQLStorage",
    "MongoDBStorage",
]