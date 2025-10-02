"""
Supabase Client Integration for Memorizer
Handles Supabase-specific database operations with RLS and auth.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.warning("Supabase client not available. Install with: pip install supabase")


class SupabaseMemoryManager:
    """Memory manager using Supabase as backend with RLS."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        service_role_key: Optional[str] = None
    ):
        """
        Initialize Supabase memory manager.

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase anon/public key (for user operations with RLS)
            service_role_key: Service role key (for admin operations, bypasses RLS)
        """
        if not SUPABASE_AVAILABLE:
            raise ImportError("Supabase not installed. Install with: pip install supabase")

        # Get credentials from environment or parameters
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_ANON_KEY")
        self.service_role_key = service_role_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be provided")

        # Create client for user operations (respects RLS)
        self.client: Client = create_client(self.supabase_url, self.supabase_key)

        # Create admin client for background jobs (bypasses RLS)
        if self.service_role_key:
            self.admin_client: Client = create_client(self.supabase_url, self.service_role_key)
        else:
            self.admin_client = None

        logger.info("Supabase memory manager initialized")

    def set_auth_token(self, access_token: str):
        """Set auth token for RLS context (call this after user login)."""
        self.client.auth.set_session(access_token, None)

    def store_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tier: str = "very_new",
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        source: str = "manual"
    ) -> str:
        """
        Store a new memory (user must be authenticated).

        Args:
            content: Memory content
            metadata: Additional metadata
            tier: Memory tier
            session_id: Optional session ID for agent tracking
            agent_id: Optional agent ID
            source: Source of the memory

        Returns:
            Memory ID
        """
        try:
            data = {
                "content": content,
                "metadata": metadata or {},
                "tier": tier,
                "session_id": session_id,
                "agent_id": agent_id,
                "source": source
            }

            # RLS automatically sets user_id from auth.uid()
            result = self.client.table("memories").insert(data).execute()

            if result.data and len(result.data) > 0:
                memory_id = result.data[0]["id"]
                logger.debug(f"Stored memory: {memory_id}")
                return memory_id
            else:
                raise Exception("Failed to store memory - no data returned")

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a memory by ID (respects RLS - only returns if user owns it).

        Args:
            memory_id: Memory ID

        Returns:
            Memory data or None
        """
        try:
            result = self.client.table("memories")\
                .select("*")\
                .eq("id", memory_id)\
                .maybe_single()\
                .execute()

            return result.data if result.data else None

        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None

    def search_memories(
        self,
        query: str = "",
        limit: int = 10,
        session_id: Optional[str] = None,
        tier: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search memories using full-text search (respects RLS).

        Args:
            query: Search query (empty string returns all)
            limit: Maximum results
            session_id: Filter by session
            tier: Filter by tier
            agent_id: Filter by agent

        Returns:
            List of matching memories
        """
        try:
            # Build query
            q = self.client.table("memories").select("*")

            # Apply filters
            if session_id:
                q = q.eq("session_id", session_id)
            if tier:
                q = q.eq("tier", tier)
            if agent_id:
                q = q.eq("agent_id", agent_id)

            # Full-text search or simple filter
            if query:
                q = q.text_search("content", query)

            # Order and limit
            q = q.order("created_at", desc=True).limit(limit)

            result = q.execute()
            return result.data or []

        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []

    def search_memories_rpc(
        self,
        query: str = "",
        limit: int = 10,
        session_id: Optional[str] = None,
        tier: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using the RPC function for better full-text search.

        Args:
            query: Search query
            limit: Maximum results
            session_id: Filter by session
            tier: Filter by tier

        Returns:
            List of memories with relevance scores
        """
        try:
            # Get current user ID from auth
            user = self.client.auth.get_user()
            if not user or not user.user:
                raise Exception("User not authenticated")

            result = self.client.rpc(
                "search_memories",
                {
                    "p_user_id": user.user.id,
                    "p_query": query,
                    "p_limit": limit,
                    "p_session_id": session_id,
                    "p_tier": tier
                }
            ).execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Failed to search memories (RPC): {e}")
            return []

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tier: Optional[str] = None,
        compressed_content: Optional[str] = None,
        compression_ratio: Optional[float] = None
    ) -> bool:
        """
        Update a memory (respects RLS).

        Args:
            memory_id: Memory ID
            content: New content
            metadata: Metadata updates
            tier: New tier
            compressed_content: Compressed version
            compression_ratio: Compression ratio

        Returns:
            Success status
        """
        try:
            updates = {}
            if content is not None:
                updates["content"] = content
            if metadata is not None:
                updates["metadata"] = metadata
            if tier is not None:
                updates["tier"] = tier
            if compressed_content is not None:
                updates["compressed_content"] = compressed_content
            if compression_ratio is not None:
                updates["compression_ratio"] = compression_ratio

            if not updates:
                return True

            result = self.client.table("memories")\
                .update(updates)\
                .eq("id", memory_id)\
                .execute()

            return bool(result.data)

        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory (respects RLS).

        Args:
            memory_id: Memory ID

        Returns:
            Success status
        """
        try:
            result = self.client.table("memories")\
                .delete()\
                .eq("id", memory_id)\
                .execute()

            return bool(result.data)

        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    def get_user_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get memory statistics for current user.

        Returns:
            Statistics dictionary
        """
        try:
            result = self.client.table("user_memory_stats")\
                .select("*")\
                .maybe_single()\
                .execute()

            return result.data if result.data else None

        except Exception as e:
            logger.error(f"Failed to get user stats: {e}")
            return None

    def create_agent_session(
        self,
        session_id: str,
        agent_id: str,
        agent_type: str = "langchain",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create or update an agent session.

        Args:
            session_id: Session ID
            agent_id: Agent ID
            agent_type: Type of agent
            metadata: Session metadata

        Returns:
            Session UUID
        """
        try:
            data = {
                "session_id": session_id,
                "agent_id": agent_id,
                "agent_type": agent_type,
                "metadata": metadata or {},
                "is_active": True
            }

            result = self.client.table("agent_sessions")\
                .upsert(data, on_conflict="user_id,session_id,agent_id")\
                .execute()

            if result.data and len(result.data) > 0:
                return result.data[0]["id"]
            else:
                raise Exception("Failed to create session")

        except Exception as e:
            logger.error(f"Failed to create agent session: {e}")
            raise

    def get_agent_sessions(
        self,
        is_active: Optional[bool] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get agent sessions for current user.

        Args:
            is_active: Filter by active status
            limit: Maximum results

        Returns:
            List of sessions
        """
        try:
            q = self.client.table("agent_sessions").select("*")

            if is_active is not None:
                q = q.eq("is_active", is_active)

            result = q.order("last_message_at", desc=True)\
                .limit(limit)\
                .execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Failed to get agent sessions: {e}")
            return []

    def increment_access_count(self, memory_id: str) -> bool:
        """
        Increment access count for a memory.

        Args:
            memory_id: Memory ID

        Returns:
            Success status
        """
        try:
            # Use RPC or direct update with increment
            result = self.client.rpc(
                "increment_memory_access",
                {"memory_id": memory_id}
            ).execute()

            return True

        except Exception as e:
            # Fallback to manual update
            try:
                memory = self.get_memory(memory_id)
                if memory:
                    new_count = memory.get("access_count", 0) + 1
                    return self.update_memory(
                        memory_id,
                        metadata={
                            **memory.get("metadata", {}),
                            "access_count": new_count,
                            "last_accessed_at": datetime.now().isoformat()
                        }
                    )
            except:
                pass
            logger.error(f"Failed to increment access count: {e}")
            return False

    # Admin functions (require service role key)

    def admin_get_compression_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get compression jobs (admin only).

        Args:
            status: Filter by status
            limit: Maximum results

        Returns:
            List of compression jobs
        """
        if not self.admin_client:
            raise Exception("Service role key required for admin operations")

        try:
            q = self.admin_client.table("compression_jobs").select("*")

            if status:
                q = q.eq("status", status)

            result = q.order("created_at", desc=True)\
                .limit(limit)\
                .execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Failed to get compression jobs: {e}")
            return []

    def admin_create_compression_job(
        self,
        memory_id: str,
        user_id: str,
        policy_name: str
    ) -> str:
        """
        Create a compression job (admin only).

        Args:
            memory_id: Memory to compress
            user_id: User ID
            policy_name: Compression policy name

        Returns:
            Job ID
        """
        if not self.admin_client:
            raise Exception("Service role key required for admin operations")

        try:
            data = {
                "memory_id": memory_id,
                "user_id": user_id,
                "policy_name": policy_name,
                "status": "pending"
            }

            result = self.admin_client.table("compression_jobs")\
                .insert(data)\
                .execute()

            if result.data and len(result.data) > 0:
                return result.data[0]["id"]
            else:
                raise Exception("Failed to create compression job")

        except Exception as e:
            logger.error(f"Failed to create compression job: {e}")
            raise


def create_supabase_memory_manager(
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    service_role_key: Optional[str] = None
) -> SupabaseMemoryManager:
    """
    Factory function to create Supabase memory manager.

    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase anon key
        service_role_key: Service role key (optional)

    Returns:
        SupabaseMemoryManager instance
    """
    return SupabaseMemoryManager(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        service_role_key=service_role_key
    )
