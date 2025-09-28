"""
Memory Lifecycle Management
Manages memory tier advancement, compression, and cleanup.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .interfaces import Memory, Storage, Summarizer, TaskRunner

logger = logging.getLogger(__name__)


class MemoryLifecycleManager:
    """Manages memory lifecycle including tier advancement and compression."""
    
    def __init__(
        self,
        storage: Storage,
        summarizer: Summarizer,
        task_runner: TaskRunner,
        config: Dict[str, Any]
    ):
        """Initialize memory lifecycle manager."""
        self.storage = storage
        self.summarizer = summarizer
        self.task_runner = task_runner
        self.config = config
        self.enabled = config.get("enabled", True)
        self.compression_threshold = config.get("compression_threshold", 0.7)
        self.cleanup_interval = config.get("cleanup_interval_seconds", 3600)
        self.tiers = config.get("tiers", {})
        
        logger.info("Memory lifecycle manager initialized")
    
    def should_advance_tier(self, memory: Memory) -> bool:
        """Check if memory should advance to next tier."""
        if not self.enabled:
            return False
        
        current_tier = memory.tier
        tier_config = self.tiers.get(current_tier, {})
        
        # Check age
        if "ttl_days" in tier_config:
            age_days = (time.time() - memory.created_at) / (24 * 3600)
            if age_days > tier_config["ttl_days"]:
                return True
        
        # Check access count
        if "min_accesses" in tier_config:
            if memory.access_count >= tier_config["min_accesses"]:
                return True
        
        # Check content length
        if "min_content_length" in tier_config:
            if len(memory.content) >= tier_config["min_content_length"]:
                return True
        
        return False
    
    def get_next_tier(self, current_tier: str) -> Optional[str]:
        """Get the next tier in the lifecycle."""
        tier_order = ["very_new", "mid_term", "long_term"]
        try:
            current_index = tier_order.index(current_tier)
            if current_index < len(tier_order) - 1:
                return tier_order[current_index + 1]
        except ValueError:
            pass
        return None
    
    def advance_memory_tier(self, memory: Memory) -> Optional[str]:
        """Advance memory to next tier."""
        if not self.should_advance_tier(memory):
            return None
        
        next_tier = self.get_next_tier(memory.tier)
        if not next_tier:
            return None
        
        # Update memory tier
        memory.tier = next_tier
        memory.updated_at = time.time()
        
        # Update in storage
        success = self.storage.update(memory)
        if success:
            logger.info(f"Memory {memory.id} advanced to tier {next_tier}")
            return next_tier
        else:
            logger.error(f"Failed to advance memory {memory.id} to tier {next_tier}")
            return None
    
    def should_compress(self, memory: Memory) -> bool:
        """Check if memory should be compressed."""
        if not self.enabled:
            return False
        
        # Check if memory is in long-term tier
        if memory.tier != "long_term":
            return False
        
        # Check content length
        if len(memory.content) < 1000:  # Don't compress short content
            return False
        
        # Check if already compressed
        if memory.metadata.get("compressed", False):
            return False
        
        return True
    
    def compress_memory(self, memory: Memory) -> bool:
        """Compress memory content."""
        if not self.should_compress(memory):
            return False
        
        try:
            # Create compressed version
            compressed_content = self.summarizer.summarize(
                memory.content,
                max_length=int(len(memory.content) * self.compression_threshold)
            )
            
            # Update memory
            original_content = memory.content
            memory.content = compressed_content
            memory.metadata["compressed"] = True
            memory.metadata["original_length"] = len(original_content)
            memory.metadata["compressed_length"] = len(compressed_content)
            memory.metadata["compression_ratio"] = len(compressed_content) / len(original_content)
            memory.updated_at = time.time()
            
            # Update in storage
            success = self.storage.update(memory)
            if success:
                logger.info(f"Memory {memory.id} compressed: {len(original_content)} -> {len(compressed_content)} chars")
                return True
            else:
                logger.error(f"Failed to compress memory {memory.id}")
                return False
                
        except Exception as e:
            logger.error(f"Error compressing memory {memory.id}: {e}")
            return False
    
    def cleanup_expired_memories(self) -> int:
        """Clean up expired memories."""
        if not self.enabled:
            return 0
        
        cleaned_count = 0
        
        try:
            # Get all memories (this is a simplified implementation)
            # In a real implementation, you'd query by tier and age
            memories_to_cleanup = []
            
            # Check each tier for cleanup
            for tier_name, tier_config in self.tiers.items():
                if "ttl_days" not in tier_config:
                    continue
                
                # Calculate cutoff time
                cutoff_time = time.time() - (tier_config["ttl_days"] * 24 * 3600)
                
                # Find memories older than cutoff
                # This would need to be implemented in the storage layer
                # For now, we'll just log the intention
                logger.debug(f"Checking tier {tier_name} for cleanup (cutoff: {cutoff_time})")
            
            # Clean up memories
            for memory in memories_to_cleanup:
                success = self.storage.delete(memory.id, memory.user_id)
                if success:
                    cleaned_count += 1
                    logger.info(f"Cleaned up expired memory {memory.id}")
            
            logger.info(f"Cleaned up {cleaned_count} expired memories")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        return cleaned_count
    
    def run_lifecycle_cycle(self) -> Dict[str, Any]:
        """Run a complete lifecycle cycle."""
        if not self.enabled:
            return {"status": "disabled"}
        
        results = {
            "status": "completed",
            "advanced": 0,
            "compressed": 0,
            "cleaned": 0,
            "errors": 0
        }
        
        try:
            # Advance memory tiers
            # This would need to be implemented with proper memory querying
            logger.debug("Running memory tier advancement")
            
            # Compress memories
            logger.debug("Running memory compression")
            
            # Cleanup expired memories
            cleaned = self.cleanup_expired_memories()
            results["cleaned"] = cleaned
            
            logger.info(f"Lifecycle cycle completed: {results}")
            
        except Exception as e:
            logger.error(f"Error in lifecycle cycle: {e}")
            results["status"] = "error"
            results["error"] = str(e)
            results["errors"] += 1
        
        return results
    
    def start_background_processing(self):
        """Start background lifecycle processing."""
        if not self.enabled:
            logger.info("Lifecycle processing disabled")
            return
        
        def lifecycle_worker():
            while True:
                try:
                    self.run_lifecycle_cycle()
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    logger.error(f"Error in lifecycle worker: {e}")
                    time.sleep(60)  # Wait before retrying
        
        # Start background task
        self.task_runner.run_async_task(lifecycle_worker)
        logger.info("Background lifecycle processing started")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get lifecycle statistics."""
        return {
            "enabled": self.enabled,
            "compression_threshold": self.compression_threshold,
            "cleanup_interval": self.cleanup_interval,
            "tiers": self.tiers,
            "last_cleanup": time.time()  # This would be tracked in a real implementation
        }
