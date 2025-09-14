"""
memory_manager.py
High-level orchestration of memory lifecycle.
Coordinates between db.py, compression_agent.py, retrieval.py, and vector_db.py.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from . import db, compression_agent, retrieval, vector_db

logger = logging.getLogger(__name__)

# ---------------------------
# Add New Session
# ---------------------------
def add_session(user_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Add a new interaction (goes into very_new tier).
    Also pushes embedding into vector DB for future retrieval.
    
    Args:
        user_id: Unique identifier for the user
        content: The session content/conversation
        metadata: Optional metadata dictionary
        
    Returns:
        memory_id: The ID of the created memory record
    """
    try:
        # Insert into DB
        memory_id = db.insert_session(user_id, content, metadata or {}, tier="very_new")
        
        # Push to vector DB (optional, should be async in production)
        try:
            vector_db.insert_embedding(memory_id, user_id, content, metadata or {})
        except Exception as e:
            logger.warning(f"Failed to insert embedding for memory {memory_id}: {e}")
            
        return memory_id
    except Exception as e:
        logger.error(f"Failed to add session for user {user_id}: {e}")
        raise

# ---------------------------
# Move Memories Between Tiers
# ---------------------------
def move_memory_between_tiers(
    user_id: str, 
    very_new_limit: int = 20, 
    mid_term_limit: int = 200,
    very_new_days: int = 10,
    mid_term_months: int = 12
) -> Dict[str, List[str]]:
    """
    Periodic job: compress/move old memories into mid_term or long_term.
    - very_new → mid_term after limit or time (10 days default)
    - mid_term → long_term after limit or time (12 months default)
    
    Args:
        user_id: User to process
        very_new_limit: Max number of very_new memories
        mid_term_limit: Max number of mid_term memories
        very_new_days: Days after which very_new becomes mid_term
        mid_term_months: Months after which mid_term becomes long_term
    """
    moved = {"to_mid_term": [], "to_long_term": []}
    
    try:
        # Step 1: Move old very_new memories to mid_term
        cutoff_date = datetime.now() - timedelta(days=very_new_days)
        very_new_memories = db.fetch_memories(user_id, "very_new", limit=None)
        
        memories_to_compress = []
        for mem in very_new_memories:
            # Move if over limit or over time threshold
            if (len(very_new_memories) > very_new_limit and 
                very_new_memories.index(mem) >= very_new_limit) or \
               (mem.get("created_at") and mem["created_at"] < cutoff_date):
                memories_to_compress.append(mem)
        
        # Compress and move to mid_term
        for mem in memories_to_compress:
            try:
                compressed = compression_agent.compress_to_mid_term(mem["content"], mem.get("metadata", {}))
                new_id = db.move_memory(
                    user_id=user_id,
                    source_tier="very_new",
                    target_tier="mid_term",
                    memory_id=mem["id"],
                    new_content=compressed.get("summary", ""),
                    new_metadata=compressed.get("metadata", {}),
                )
                moved["to_mid_term"].append(new_id)
                logger.info(f"Moved memory {mem['id']} from very_new to mid_term as {new_id}")
            except Exception as e:
                logger.error(f"Failed to compress memory {mem['id']}: {e}")
        
        # Step 2: Move old mid_term memories to long_term
        cutoff_date = datetime.now() - timedelta(days=mid_term_months * 30)  # Approximate months
        mid_term_memories = db.fetch_memories(user_id, "mid_term", limit=None)
        
        memories_to_aggregate = []
        for mem in mid_term_memories:
            if (len(mid_term_memories) > mid_term_limit and 
                mid_term_memories.index(mem) >= mid_term_limit) or \
               (mem.get("created_at") and mem["created_at"] < cutoff_date):
                memories_to_aggregate.append(mem)
        
        # Aggregate multiple mid_term memories into long_term insights
        if memories_to_aggregate:
            try:
                contents = [m["content"] for m in memories_to_aggregate]
                metadata_list = [m.get("metadata", {}) for m in memories_to_aggregate]
                
                aggregated = compression_agent.compress_to_long_term(contents, metadata_list)
                
                # Create new long_term memory
                new_id = db.insert_session(
                    user_id=user_id,
                    content=aggregated.get("summary", ""),
                    metadata=aggregated.get("metadata", {}),
                    tier="long_term",
                )
                
                # Remove old mid_term memories
                for mem in memories_to_aggregate:
                    db.delete_memory(user_id, mem["id"])
                
                moved["to_long_term"].append(new_id)
                logger.info(f"Aggregated {len(memories_to_aggregate)} mid_term memories into long_term {new_id}")
                
            except Exception as e:
                logger.error(f"Failed to aggregate mid_term memories: {e}")
        
    except Exception as e:
        logger.error(f"Error in move_memory_between_tiers for user {user_id}: {e}")
        
    return moved

# ---------------------------
# Retrieve Context
# ---------------------------
def get_context(user_id: str, query: str, max_items: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve context for an active session using hybrid retrieval:
    1. First, fetch from very_new / mid_term / long_term (DB-based)
    2. Use keyword relevance scoring
    3. If insufficient results, fallback to vector DB semantic search
    
    Args:
        user_id: User to retrieve context for
        query: Search query/context
        max_items: Maximum number of items to return
        
    Returns:
        List of memory dictionaries with relevance scores
    """
    try:
        # Step 1: Use hybrid retrieval (keyword + relevance scoring first)
        context = retrieval.retrieve_context(user_id, query, max_items=max_items)
        
        # Step 2: If we have sufficient high-quality results, return them
        if len(context) >= min(max_items, 3) and any(item.get("score", 0) > 0.7 for item in context):
            logger.debug(f"Retrieved {len(context)} items via hybrid retrieval for user {user_id}")
            return context[:max_items]
        
        # Step 3: Fallback to vector DB for semantic search
        logger.debug(f"Falling back to vector DB for user {user_id}")
        vector_results = vector_db.query_embeddings(user_id, query, top_k=max_items)
        
        # Combine results, preferring DB results for recent/high-relevance items
        combined_results = []
        seen_ids = set()
        
        # Add high-scoring DB results first
        for item in context:
            if item.get("score", 0) > 0.5 and item["id"] not in seen_ids:
                combined_results.append(item)
                seen_ids.add(item["id"])
        
        # Fill remaining slots with vector results
        for item in vector_results:
            if len(combined_results) >= max_items:
                break
            if item["id"] not in seen_ids:
                combined_results.append(item)
                seen_ids.add(item["id"])
        
        return combined_results[:max_items]
        
    except Exception as e:
        logger.error(f"Error retrieving context for user {user_id}: {e}")
        return []

# ---------------------------
# Utility Functions
# ---------------------------
def get_memory_stats(user_id: str) -> Dict[str, int]:
    """Get memory statistics for a user."""
    try:
        stats = {}
        for tier in ["very_new", "mid_term", "long_term"]:
            memories = db.fetch_memories(user_id, tier, limit=None)
            stats[tier] = len(memories) if memories else 0
        return stats
    except Exception as e:
        logger.error(f"Error getting memory stats for user {user_id}: {e}")
        return {"very_new": 0, "mid_term": 0, "long_term": 0}

def cleanup_old_memories(user_id: str, max_age_days: int = 365 * 2) -> int:
    """Clean up very old memories beyond the retention period."""
    try:
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        return db.delete_old_memories(user_id, cutoff_date)
    except Exception as e:
        logger.error(f"Error cleaning up old memories for user {user_id}: {e}")
        return 0
