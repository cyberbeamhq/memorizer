"""
agent_interface.py
Standardized interface for AI agent memory management.
Provides a unified API for different types of AI agents to interact with memory.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of AI agents."""

    CONVERSATIONAL = "conversational"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    TASK_ORIENTED = "task_oriented"
    RESEARCH = "research"
    CUSTOMER_SERVICE = "customer_service"
    ECOMMERCE = "ecommerce"
    GENERAL = "general"


class MemoryType(Enum):
    """Types of memories."""

    CONVERSATION = "conversation"
    DECISION = "decision"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    SUCCESS = "success"
    CONTEXT = "context"
    PREFERENCE = "preference"
    FACT = "fact"
    GOAL = "goal"
    TASK = "task"


@dataclass
class AgentConfig:
    """Configuration for an AI agent."""

    agent_id: str
    agent_type: AgentType
    framework: str = "general"
    context_window: int = 5
    max_tokens: int = 4000
    memory_ttl_days: int = 30
    compression_enabled: bool = True
    retrieval_enabled: bool = True
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryRequest:
    """Request to store a memory."""

    content: str
    memory_type: MemoryType
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=low, 2=medium, 3=high
    tags: List[str] = field(default_factory=list)


@dataclass
class MemoryResponse:
    """Response from memory operations."""

    memory_id: str
    success: bool
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalRequest:
    """Request to retrieve memories."""

    query: str
    max_memories: int = 5
    memory_types: Optional[List[MemoryType]] = None
    session_id: Optional[str] = None
    time_range: Optional[Dict[str, datetime]] = None
    tags: Optional[List[str]] = None
    min_relevance: float = 0.1


@dataclass
class RetrievalResponse:
    """Response from memory retrieval."""

    memories: List[Dict[str, Any]]
    total_found: int
    query: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentMemoryInterface(ABC):
    """Abstract interface for agent memory management."""

    @abstractmethod
    def store_memory(self, request: MemoryRequest) -> MemoryResponse:
        """Store a memory for the agent."""
        pass

    @abstractmethod
    def retrieve_memories(self, request: RetrievalRequest) -> RetrievalResponse:
        """Retrieve relevant memories for the agent."""
        pass

    @abstractmethod
    def clear_memories(self, session_id: Optional[str] = None) -> int:
        """Clear memories for the agent."""
        pass

    @abstractmethod
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics for the agent."""
        pass

    @abstractmethod
    def update_agent_config(self, config: AgentConfig) -> bool:
        """Update agent configuration."""
        pass


class MemorizerAgentInterface(AgentMemoryInterface):
    """Implementation of agent memory interface using Memorizer framework."""

    def __init__(self, config: AgentConfig, memory_manager, integration_manager=None):
        self.config = config
        self.memory_manager = memory_manager
        self.integration_manager = integration_manager
        self.user_id = f"{config.framework}_{config.agent_id}"
        logger.info(
            f"Initialized agent interface for {config.agent_id} ({config.agent_type.value})"
        )

    def store_memory(self, request: MemoryRequest) -> MemoryResponse:
        """Store a memory for the agent."""
        try:
            # Prepare content with memory type prefix
            content = f"[{request.memory_type.value}] {request.content}"

            # Prepare metadata
            metadata = {
                **request.metadata,
                "agent_id": self.config.agent_id,
                "agent_type": self.config.agent_type.value,
                "framework": self.config.framework,
                "memory_type": request.memory_type.value,
                "session_id": request.session_id,
                "priority": request.priority,
                "tags": request.tags,
                "timestamp": datetime.now().isoformat(),
            }

            # Store using memory manager
            memory_id = self.memory_manager.add_session(
                user_id=self.user_id, content=content, metadata=metadata
            )

            logger.debug(f"Stored memory {memory_id} for agent {self.config.agent_id}")

            return MemoryResponse(
                memory_id=memory_id,
                success=True,
                message="Memory stored successfully",
                metadata={"memory_id": memory_id},
            )

        except Exception as e:
            logger.error(
                f"Failed to store memory for agent {self.config.agent_id}: {e}"
            )
            return MemoryResponse(
                memory_id="", success=False, message=f"Failed to store memory: {str(e)}"
            )

    def retrieve_memories(self, request: RetrievalRequest) -> RetrievalResponse:
        """Retrieve relevant memories for the agent."""
        try:
            # Build query with filters
            query_parts = [request.query]

            if request.memory_types:
                type_filter = " OR ".join(
                    [f"type:{mt.value}" for mt in request.memory_types]
                )
                query_parts.append(f"({type_filter})")

            if request.tags:
                tag_filter = " OR ".join([f"tag:{tag}" for tag in request.tags])
                query_parts.append(f"({tag_filter})")

            if request.session_id:
                query_parts.append(f"session:{request.session_id}")

            full_query = " ".join(query_parts)

            # Retrieve memories
            memories = self.memory_manager.get_context(
                user_id=self.user_id, query=full_query, max_items=request.max_memories
            )

            # Filter by time range if specified
            if request.time_range:
                filtered_memories = []
                for memory in memories:
                    memory_time = datetime.fromisoformat(
                        memory.get("metadata", {}).get(
                            "timestamp", datetime.now().isoformat()
                        )
                    )
                    if (
                        request.time_range.get("start", datetime.min)
                        <= memory_time
                        <= request.time_range.get("end", datetime.max)
                    ):
                        filtered_memories.append(memory)
                memories = filtered_memories

            # Filter by minimum relevance
            if request.min_relevance > 0:
                memories = [
                    m
                    for m in memories
                    if m.get("relevance_score", 0) >= request.min_relevance
                ]

            logger.debug(
                f"Retrieved {len(memories)} memories for agent {self.config.agent_id}"
            )

            return RetrievalResponse(
                memories=memories,
                total_found=len(memories),
                query=request.query,
                metadata={
                    "agent_id": self.config.agent_id,
                    "filters_applied": len(query_parts) > 1,
                },
            )

        except Exception as e:
            logger.error(
                f"Failed to retrieve memories for agent {self.config.agent_id}: {e}"
            )
            return RetrievalResponse(
                memories=[],
                total_found=0,
                query=request.query,
                metadata={"error": str(e)},
            )

    def clear_memories(self, session_id: Optional[str] = None) -> int:
        """Clear memories for the agent."""
        try:
            # This would need to be implemented in memory_manager
            # For now, return 0 as placeholder
            logger.info(
                f"Cleared memories for agent {self.config.agent_id}, session: {session_id}"
            )
            return 0
        except Exception as e:
            logger.error(
                f"Failed to clear memories for agent {self.config.agent_id}: {e}"
            )
            return 0

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics for the agent."""
        try:
            stats = self.memory_manager.get_memory_stats(self.user_id)
            return {
                "agent_id": self.config.agent_id,
                "agent_type": self.config.agent_type.value,
                "framework": self.config.framework,
                "memory_stats": stats,
                "total_memories": sum(stats.values()),
                "config": {
                    "context_window": self.config.context_window,
                    "max_tokens": self.config.max_tokens,
                    "memory_ttl_days": self.config.memory_ttl_days,
                },
            }
        except Exception as e:
            logger.error(
                f"Failed to get memory stats for agent {self.config.agent_id}: {e}"
            )
            return {"agent_id": self.config.agent_id, "error": str(e)}

    def update_agent_config(self, config: AgentConfig) -> bool:
        """Update agent configuration."""
        try:
            self.config = config
            self.user_id = f"{config.framework}_{config.agent_id}"
            logger.info(f"Updated config for agent {config.agent_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update config for agent {config.agent_id}: {e}")
            return False


class AgentMemoryManager:
    """Manager for multiple agent memory interfaces."""

    def __init__(self, memory_manager, integration_manager=None):
        self.memory_manager = memory_manager
        self.integration_manager = integration_manager
        self.agents = {}
        logger.info("Agent memory manager initialized")

    def register_agent(self, config: AgentConfig) -> AgentMemoryInterface:
        """Register a new agent."""
        try:
            agent_interface = MemorizerAgentInterface(
                config=config,
                memory_manager=self.memory_manager,
                integration_manager=self.integration_manager,
            )

            self.agents[config.agent_id] = agent_interface
            logger.info(
                f"Registered agent: {config.agent_id} ({config.agent_type.value})"
            )

            return agent_interface

        except Exception as e:
            logger.error(f"Failed to register agent {config.agent_id}: {e}")
            raise

    def get_agent(self, agent_id: str) -> Optional[AgentMemoryInterface]:
        """Get agent interface by ID."""
        return self.agents.get(agent_id)

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        try:
            if agent_id in self.agents:
                del self.agents[agent_id]
                logger.info(f"Unregistered agent: {agent_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False

    def list_agents(self) -> List[str]:
        """List all registered agent IDs."""
        return list(self.agents.keys())

    def get_all_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get memory statistics for all agents."""
        stats = {}
        for agent_id, agent_interface in self.agents.items():
            try:
                stats[agent_id] = agent_interface.get_memory_stats()
            except Exception as e:
                stats[agent_id] = {"error": str(e)}
        return stats

    def cleanup_old_memories(self, max_age_days: int = 30) -> Dict[str, int]:
        """Clean up old memories for all agents."""
        results = {}
        for agent_id, agent_interface in self.agents.items():
            try:
                # This would need to be implemented in memory_manager
                results[agent_id] = 0
            except Exception as e:
                logger.error(f"Failed to cleanup memories for agent {agent_id}: {e}")
                results[agent_id] = 0
        return results


# Convenience functions for common operations
def create_agent_config(
    agent_id: str,
    agent_type: Union[str, AgentType],
    framework: str = "general",
    **kwargs,
) -> AgentConfig:
    """Create an agent configuration."""
    if isinstance(agent_type, str):
        agent_type = AgentType(agent_type)

    return AgentConfig(
        agent_id=agent_id, agent_type=agent_type, framework=framework, **kwargs
    )


def create_memory_request(
    content: str, memory_type: Union[str, MemoryType], **kwargs
) -> MemoryRequest:
    """Create a memory request."""
    if isinstance(memory_type, str):
        memory_type = MemoryType(memory_type)

    return MemoryRequest(content=content, memory_type=memory_type, **kwargs)


def create_retrieval_request(query: str, **kwargs) -> RetrievalRequest:
    """Create a retrieval request."""
    return RetrievalRequest(query=query, **kwargs)


# Global agent memory manager instance
_agent_memory_manager = None


def get_agent_memory_manager(
    memory_manager=None, integration_manager=None
) -> AgentMemoryManager:
    """Get global agent memory manager instance."""
    global _agent_memory_manager
    if _agent_memory_manager is None and memory_manager is not None:
        _agent_memory_manager = AgentMemoryManager(memory_manager, integration_manager)
    return _agent_memory_manager


def initialize_agent_memory_manager(memory_manager, integration_manager=None):
    """Initialize global agent memory manager."""
    global _agent_memory_manager
    _agent_memory_manager = AgentMemoryManager(memory_manager, integration_manager)
    logger.info("Agent memory manager initialized")
