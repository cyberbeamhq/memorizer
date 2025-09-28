"""
Memory Manager
A memory manager focused purely on memory management without infrastructure complexity.
"""

import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """Simple memory object."""
    id: str
    user_id: str
    content: str
    metadata: Dict[str, Any]
    tier: str = "very_new"
    created_at: float = None
    updated_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class MemoryStats:
    """Memory statistics."""
    total_memories: int
    memory_by_tier: Dict[str, int]
    user_id: str


@dataclass
class RetrievalResult:
    """Search result."""
    memories: List[Memory]
    total_found: int


class MemoryManagerCore:
    """Memory manager focused on core memory management."""

    def __init__(self, config):
        """Initialize the memory manager."""
        self.config = config
        self._storage = {}  # Simple in-memory storage
        self._memory_counter = 0

        logger.info("Memory manager initialized")

    def store_memory(
        self,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tier: str = "very_new"
    ) -> str:
        """
        Store a new memory.

        Args:
            user_id: User ID
            content: Memory content
            metadata: Optional metadata
            tier: Memory tier (very_new, mid_term, long_term)

        Returns:
            Memory ID
        """
        try:
            # Generate unique memory ID
            self._memory_counter += 1
            memory_id = f"mem_{self._memory_counter}_{int(time.time())}"

            # Create memory object
            memory = Memory(
                id=memory_id,
                user_id=user_id,
                content=content,
                metadata=metadata or {},
                tier=tier
            )

            # Store in memory storage
            if user_id not in self._storage:
                self._storage[user_id] = {}

            self._storage[user_id][memory_id] = memory

            logger.debug(f"Stored memory {memory_id} for user {user_id}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise

    def get_memory(self, memory_id: str, user_id: str) -> Optional[Memory]:
        """
        Get a memory by ID.

        Args:
            memory_id: Memory ID
            user_id: User ID

        Returns:
            Memory object or None
        """
        try:
            user_memories = self._storage.get(user_id, {})
            return user_memories.get(memory_id)
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None

    def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Search for memories using text matching.

        Args:
            user_id: User ID
            query: Search query
            limit: Maximum number of results
            filters: Additional filters (not implemented in basic version)

        Returns:
            RetrievalResult with matching memories
        """
        try:
            user_memories = self._storage.get(user_id, {})
            matching_memories = []

            # Simple keyword search - check for any word matches
            query_words = query.lower().split()
            for memory in user_memories.values():
                content_lower = memory.content.lower()
                # Check if any query word appears in content
                if any(word in content_lower for word in query_words):
                    matching_memories.append(memory)
                    if len(matching_memories) >= limit:
                        break

            return RetrievalResult(
                memories=matching_memories,
                total_found=len(matching_memories)
            )

        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return RetrievalResult(
                memories=[],
                total_found=0
            )

    def get_stats(self, user_id: str) -> MemoryStats:
        """
        Get memory statistics for a user.

        Args:
            user_id: User ID

        Returns:
            MemoryStats object
        """
        try:
            user_memories = self._storage.get(user_id, {})
            tier_counts = {}

            for memory in user_memories.values():
                tier = memory.tier
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

            return MemoryStats(
                total_memories=len(user_memories),
                memory_by_tier=tier_counts,
                user_id=user_id
            )
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return MemoryStats(
                total_memories=0,
                memory_by_tier={},
                user_id=user_id
            )

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory ID
            user_id: User ID

        Returns:
            Success status
        """
        try:
            user_memories = self._storage.get(user_id, {})
            if memory_id in user_memories:
                del user_memories[memory_id]
                logger.debug(f"Deleted memory {memory_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    def update_memory(
        self,
        memory_id: str,
        user_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None
    ) -> bool:
        """
        Update a memory.

        Args:
            memory_id: Memory ID
            user_id: User ID
            content: New content (optional)
            metadata: New metadata (optional)
            tier: New tier (optional)

        Returns:
            Success status
        """
        try:
            memory = self.get_memory(memory_id, user_id)
            if not memory:
                return False

            # Update fields
            if content is not None:
                memory.content = content
            if metadata is not None:
                memory.metadata.update(metadata)
            if tier is not None:
                memory.tier = tier

            memory.updated_at = time.time()

            logger.debug(f"Updated memory {memory_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False


# Alias for consistency with expected naming
MemoryManager = MemoryManagerCore