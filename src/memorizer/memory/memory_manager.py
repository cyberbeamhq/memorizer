"""
Framework Memory Manager
Main memory management class for the Memorizer framework.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.interfaces import Memory, Query, RetrievalResult

logger = logging.getLogger(__name__)


class FrameworkMemoryManager:
    """Main memory manager for the Memorizer framework."""
    
    def __init__(self, framework):
        """Initialize the memory manager."""
        self.framework = framework
        self.storage = framework.storage
        self.retriever = framework.retriever
        self.summarizer = framework.summarizer
        self.vector_store = framework.vector_store
        self.embedding_provider = framework.embedding_provider
        self.cache = framework.cache
        
        logger.info("Framework memory manager initialized")
    
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
            # Create memory object
            memory = Memory(
                id=f"mem_{int(time.time() * 1000)}_{hash(content) % 10000}",
                user_id=user_id,
                content=content,
                metadata=metadata or {},
                tier=tier
            )
            
            # Store in storage
            memory_id = self.storage.store_memory(memory)
            
            # Generate embedding if vector store is available
            if self.vector_store and self.embedding_provider:
                try:
                    embedding = self.embedding_provider.get_embedding(content)
                    self.vector_store.insert_embedding(
                        memory_id=memory_id,
                        embedding=embedding,
                        metadata=memory.metadata
                    )
                except Exception as e:
                    logger.warning(f"Failed to create embedding for memory {memory_id}: {e}")
            
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
            return self.storage.get_memory(memory_id, user_id)
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None
    
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
            # Get existing memory
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
            
            # Update in storage
            success = self.storage.update_memory(memory.id, memory.user_id, memory.content, memory.metadata)
            if success:
                logger.debug(f"Updated memory {memory_id}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
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
            # Delete from vector store if available
            if self.vector_store:
                try:
                    self.vector_store.delete_embedding(memory_id)
                except Exception as e:
                    logger.warning(f"Failed to delete embedding for memory {memory_id}: {e}")
            
            # Delete from storage
            success = self.storage.delete_memory(memory_id, user_id)
            if success:
                logger.debug(f"Deleted memory {memory_id}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Search for memories.
        
        Args:
            user_id: User ID
            query: Search query
            limit: Maximum number of results
            filters: Additional filters
            
        Returns:
            RetrievalResult with matching memories
        """
        try:
            search_query = Query(
                text=query,
                user_id=user_id,
                limit=limit,
                filters=filters
            )
            
            return self.retriever.retrieve(search_query)
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return RetrievalResult(
                memories=[],
                scores=[],
                total_found=0,
                retrieval_time=0.0,
                source="error"
            )
    
    def promote_memory(
        self,
        memory_id: str,
        user_id: str,
        new_tier: str
    ) -> bool:
        """
        Promote memory to a new tier.
        
        Args:
            memory_id: Memory ID
            user_id: User ID
            new_tier: New tier
            
        Returns:
            Success status
        """
        try:
            return self.update_memory(memory_id, user_id, tier=new_tier)
        except Exception as e:
            logger.error(f"Failed to promote memory {memory_id}: {e}")
            return False
    
    def archive_memory(self, memory_id: str, user_id: str) -> bool:
        """
        Archive a memory.
        
        Args:
            memory_id: Memory ID
            user_id: User ID
            
        Returns:
            Success status
        """
        try:
            return self.update_memory(memory_id, user_id, tier="archived")
        except Exception as e:
            logger.error(f"Failed to archive memory {memory_id}: {e}")
            return False
    
    def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get memory statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Statistics dictionary
        """
        try:
            # This would need to be implemented in the storage layer
            # For now, return basic stats
            return {
                "user_id": user_id,
                "total_memories": 0,
                "tiers": {
                    "very_new": 0,
                    "mid_term": 0,
                    "long_term": 0
                }
            }
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the memory manager."""
        try:
            return {
                "status": "healthy",
                "components": {
                    "storage": self.storage.get_health_status() if hasattr(self.storage, 'get_health_status') else {"status": "unknown"},
                    "retriever": self.retriever.get_health_status() if hasattr(self.retriever, 'get_health_status') else {"status": "unknown"},
                    "vector_store": self.vector_store.get_health_status() if hasattr(self.vector_store, 'get_health_status') else {"status": "unknown"},
                    "embedding_provider": self.embedding_provider.get_health_status() if hasattr(self.embedding_provider, 'get_health_status') else {"status": "unknown"},
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }