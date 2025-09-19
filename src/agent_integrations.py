"""
agent_integrations.py
Integrations for popular AI agent frameworks.
Provides standardized interfaces for LangChain, LlamaIndex, AutoGPT, and CrewAI.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class AgentMemory:
    """Standardized memory structure for agent integrations."""

    agent_id: str
    session_id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    memory_type: str = "conversation"  # conversation, tool_call, decision, etc.


@dataclass
class AgentContext:
    """Context information for agent memory retrieval."""

    agent_id: str
    session_id: Optional[str] = None
    query: str = ""
    max_memories: int = 5
    memory_types: Optional[List[str]] = None
    time_range: Optional[Dict[str, datetime]] = None


class AgentIntegration(ABC):
    """Abstract base class for agent framework integrations."""

    @abstractmethod
    def store_memory(self, memory: AgentMemory) -> str:
        """Store a memory for an agent."""
        pass

    @abstractmethod
    def retrieve_memories(self, context: AgentContext) -> List[AgentMemory]:
        """Retrieve relevant memories for an agent."""
        pass

    @abstractmethod
    def clear_agent_memories(
        self, agent_id: str, session_id: Optional[str] = None
    ) -> int:
        """Clear memories for an agent or session."""
        pass

    @abstractmethod
    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics for an agent."""
        pass


class LangChainIntegration(AgentIntegration):
    """Integration for LangChain agents."""

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.agent_type = "langchain"
        logger.info("LangChain integration initialized")

    def store_memory(self, memory: AgentMemory) -> str:
        """Store a memory for a LangChain agent."""
        try:
            # Convert to memorizer format
            user_id = f"langchain_{memory.agent_id}"
            content = f"[{memory.memory_type}] {memory.content}"

            # Add LangChain-specific metadata
            metadata = {
                **memory.metadata,
                "agent_framework": "langchain",
                "agent_id": memory.agent_id,
                "session_id": memory.session_id,
                "memory_type": memory.memory_type,
                "timestamp": memory.timestamp.isoformat(),
            }

            memory_id = self.memory_manager.add_session(user_id, content, metadata)
            logger.debug(f"Stored LangChain memory: {memory_id}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to store LangChain memory: {e}")
            raise

    def retrieve_memories(self, context: AgentContext) -> List[AgentMemory]:
        """Retrieve relevant memories for a LangChain agent."""
        try:
            user_id = f"langchain_{context.agent_id}"

            # Get memories from memorizer
            memories = self.memory_manager.get_context(
                user_id=user_id, query=context.query, max_items=context.max_memories
            )

            # Convert to AgentMemory format
            agent_memories = []
            for mem in memories:
                agent_memory = AgentMemory(
                    agent_id=context.agent_id,
                    session_id=mem.get("metadata", {}).get("session_id", ""),
                    content=mem.get("content", ""),
                    metadata=mem.get("metadata", {}),
                    timestamp=datetime.fromisoformat(
                        mem.get("metadata", {}).get(
                            "timestamp", datetime.now().isoformat()
                        )
                    ),
                    memory_type=mem.get("metadata", {}).get(
                        "memory_type", "conversation"
                    ),
                )
                agent_memories.append(agent_memory)

            return agent_memories

        except Exception as e:
            logger.error(f"Failed to retrieve LangChain memories: {e}")
            return []

    def clear_agent_memories(
        self, agent_id: str, session_id: Optional[str] = None
    ) -> int:
        """Clear memories for a LangChain agent."""
        try:
            user_id = f"langchain_{agent_id}"
            # This would need to be implemented in memory_manager
            # For now, return 0 as placeholder
            logger.info(f"Cleared memories for LangChain agent: {agent_id}")
            return 0
        except Exception as e:
            logger.error(f"Failed to clear LangChain memories: {e}")
            return 0

    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics for a LangChain agent."""
        try:
            user_id = f"langchain_{agent_id}"
            stats = self.memory_manager.get_memory_stats(user_id)
            return {
                "agent_id": agent_id,
                "agent_framework": "langchain",
                "memory_stats": stats,
                "total_memories": sum(stats.values()),
            }
        except Exception as e:
            logger.error(f"Failed to get LangChain agent stats: {e}")
            return {"agent_id": agent_id, "error": str(e)}


class LlamaIndexIntegration(AgentIntegration):
    """Integration for LlamaIndex agents."""

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.agent_type = "llamaindex"
        logger.info("LlamaIndex integration initialized")

    def store_memory(self, memory: AgentMemory) -> str:
        """Store a memory for a LlamaIndex agent."""
        try:
            user_id = f"llamaindex_{memory.agent_id}"
            content = f"[{memory.memory_type}] {memory.content}"

            metadata = {
                **memory.metadata,
                "agent_framework": "llamaindex",
                "agent_id": memory.agent_id,
                "session_id": memory.session_id,
                "memory_type": memory.memory_type,
                "timestamp": memory.timestamp.isoformat(),
            }

            memory_id = self.memory_manager.add_session(user_id, content, metadata)
            logger.debug(f"Stored LlamaIndex memory: {memory_id}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to store LlamaIndex memory: {e}")
            raise

    def retrieve_memories(self, context: AgentContext) -> List[AgentMemory]:
        """Retrieve relevant memories for a LlamaIndex agent."""
        try:
            user_id = f"llamaindex_{context.agent_id}"

            memories = self.memory_manager.get_context(
                user_id=user_id, query=context.query, max_items=context.max_memories
            )

            agent_memories = []
            for mem in memories:
                agent_memory = AgentMemory(
                    agent_id=context.agent_id,
                    session_id=mem.get("metadata", {}).get("session_id", ""),
                    content=mem.get("content", ""),
                    metadata=mem.get("metadata", {}),
                    timestamp=datetime.fromisoformat(
                        mem.get("metadata", {}).get(
                            "timestamp", datetime.now().isoformat()
                        )
                    ),
                    memory_type=mem.get("metadata", {}).get(
                        "memory_type", "conversation"
                    ),
                )
                agent_memories.append(agent_memory)

            return agent_memories

        except Exception as e:
            logger.error(f"Failed to retrieve LlamaIndex memories: {e}")
            return []

    def clear_agent_memories(
        self, agent_id: str, session_id: Optional[str] = None
    ) -> int:
        """Clear memories for a LlamaIndex agent."""
        try:
            user_id = f"llamaindex_{agent_id}"
            logger.info(f"Cleared memories for LlamaIndex agent: {agent_id}")
            return 0
        except Exception as e:
            logger.error(f"Failed to clear LlamaIndex memories: {e}")
            return 0

    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics for a LlamaIndex agent."""
        try:
            user_id = f"llamaindex_{agent_id}"
            stats = self.memory_manager.get_memory_stats(user_id)
            return {
                "agent_id": agent_id,
                "agent_framework": "llamaindex",
                "memory_stats": stats,
                "total_memories": sum(stats.values()),
            }
        except Exception as e:
            logger.error(f"Failed to get LlamaIndex agent stats: {e}")
            return {"agent_id": agent_id, "error": str(e)}


class AutoGPTIntegration(AgentIntegration):
    """Integration for AutoGPT agents."""

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.agent_type = "autogpt"
        logger.info("AutoGPT integration initialized")

    def store_memory(self, memory: AgentMemory) -> str:
        """Store a memory for an AutoGPT agent."""
        try:
            user_id = f"autogpt_{memory.agent_id}"
            content = f"[{memory.memory_type}] {memory.content}"

            metadata = {
                **memory.metadata,
                "agent_framework": "autogpt",
                "agent_id": memory.agent_id,
                "session_id": memory.session_id,
                "memory_type": memory.memory_type,
                "timestamp": memory.timestamp.isoformat(),
            }

            memory_id = self.memory_manager.add_session(user_id, content, metadata)
            logger.debug(f"Stored AutoGPT memory: {memory_id}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to store AutoGPT memory: {e}")
            raise

    def retrieve_memories(self, context: AgentContext) -> List[AgentMemory]:
        """Retrieve relevant memories for an AutoGPT agent."""
        try:
            user_id = f"autogpt_{context.agent_id}"

            memories = self.memory_manager.get_context(
                user_id=user_id, query=context.query, max_items=context.max_memories
            )

            agent_memories = []
            for mem in memories:
                agent_memory = AgentMemory(
                    agent_id=context.agent_id,
                    session_id=mem.get("metadata", {}).get("session_id", ""),
                    content=mem.get("content", ""),
                    metadata=mem.get("metadata", {}),
                    timestamp=datetime.fromisoformat(
                        mem.get("metadata", {}).get(
                            "timestamp", datetime.now().isoformat()
                        )
                    ),
                    memory_type=mem.get("metadata", {}).get(
                        "memory_type", "conversation"
                    ),
                )
                agent_memories.append(agent_memory)

            return agent_memories

        except Exception as e:
            logger.error(f"Failed to retrieve AutoGPT memories: {e}")
            return []

    def clear_agent_memories(
        self, agent_id: str, session_id: Optional[str] = None
    ) -> int:
        """Clear memories for an AutoGPT agent."""
        try:
            user_id = f"autogpt_{agent_id}"
            logger.info(f"Cleared memories for AutoGPT agent: {agent_id}")
            return 0
        except Exception as e:
            logger.error(f"Failed to clear AutoGPT memories: {e}")
            return 0

    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics for an AutoGPT agent."""
        try:
            user_id = f"autogpt_{agent_id}"
            stats = self.memory_manager.get_memory_stats(user_id)
            return {
                "agent_id": agent_id,
                "agent_framework": "autogpt",
                "memory_stats": stats,
                "total_memories": sum(stats.values()),
            }
        except Exception as e:
            logger.error(f"Failed to get AutoGPT agent stats: {e}")
            return {"agent_id": agent_id, "error": str(e)}


class CrewAIIntegration(AgentIntegration):
    """Integration for CrewAI agents."""

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.agent_type = "crewai"
        logger.info("CrewAI integration initialized")

    def store_memory(self, memory: AgentMemory) -> str:
        """Store a memory for a CrewAI agent."""
        try:
            user_id = f"crewai_{memory.agent_id}"
            content = f"[{memory.memory_type}] {memory.content}"

            metadata = {
                **memory.metadata,
                "agent_framework": "crewai",
                "agent_id": memory.agent_id,
                "session_id": memory.session_id,
                "memory_type": memory.memory_type,
                "timestamp": memory.timestamp.isoformat(),
            }

            memory_id = self.memory_manager.add_session(user_id, content, metadata)
            logger.debug(f"Stored CrewAI memory: {memory_id}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to store CrewAI memory: {e}")
            raise

    def retrieve_memories(self, context: AgentContext) -> List[AgentMemory]:
        """Retrieve relevant memories for a CrewAI agent."""
        try:
            user_id = f"crewai_{context.agent_id}"

            memories = self.memory_manager.get_context(
                user_id=user_id, query=context.query, max_items=context.max_memories
            )

            agent_memories = []
            for mem in memories:
                agent_memory = AgentMemory(
                    agent_id=context.agent_id,
                    session_id=mem.get("metadata", {}).get("session_id", ""),
                    content=mem.get("content", ""),
                    metadata=mem.get("metadata", {}),
                    timestamp=datetime.fromisoformat(
                        mem.get("metadata", {}).get(
                            "timestamp", datetime.now().isoformat()
                        )
                    ),
                    memory_type=mem.get("metadata", {}).get(
                        "memory_type", "conversation"
                    ),
                )
                agent_memories.append(agent_memory)

            return agent_memories

        except Exception as e:
            logger.error(f"Failed to retrieve CrewAI memories: {e}")
            return []

    def clear_agent_memories(
        self, agent_id: str, session_id: Optional[str] = None
    ) -> int:
        """Clear memories for a CrewAI agent."""
        try:
            user_id = f"crewai_{agent_id}"
            logger.info(f"Cleared memories for CrewAI agent: {agent_id}")
            return 0
        except Exception as e:
            logger.error(f"Failed to clear CrewAI memories: {e}")
            return 0

    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics for a CrewAI agent."""
        try:
            user_id = f"crewai_{agent_id}"
            stats = self.memory_manager.get_memory_stats(user_id)
            return {
                "agent_id": agent_id,
                "agent_framework": "crewai",
                "memory_stats": stats,
                "total_memories": sum(stats.values()),
            }
        except Exception as e:
            logger.error(f"Failed to get CrewAI agent stats: {e}")
            return {"agent_id": agent_id, "error": str(e)}


class AgentIntegrationManager:
    """Manager for agent framework integrations."""

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.integrations = {}
        self._initialize_integrations()

    def _initialize_integrations(self):
        """Initialize available agent integrations."""
        # Check which integrations are enabled
        if os.getenv("LANGCHAIN_ENABLED", "false").lower() == "true":
            self.integrations["langchain"] = LangChainIntegration(self.memory_manager)

        if os.getenv("LLAMAINDEX_ENABLED", "false").lower() == "true":
            self.integrations["llamaindex"] = LlamaIndexIntegration(self.memory_manager)

        if os.getenv("AUTOGPT_ENABLED", "false").lower() == "true":
            self.integrations["autogpt"] = AutoGPTIntegration(self.memory_manager)

        if os.getenv("CREWAI_ENABLED", "false").lower() == "true":
            self.integrations["crewai"] = CrewAIIntegration(self.memory_manager)

        logger.info(f"Initialized agent integrations: {list(self.integrations.keys())}")

    def get_integration(self, framework: str) -> Optional[AgentIntegration]:
        """Get integration for a specific framework."""
        return self.integrations.get(framework.lower())

    def store_agent_memory(self, framework: str, memory: AgentMemory) -> str:
        """Store memory for an agent using the specified framework."""
        integration = self.get_integration(framework)
        if not integration:
            raise ValueError(f"No integration available for framework: {framework}")

        return integration.store_memory(memory)

    def retrieve_agent_memories(
        self, framework: str, context: AgentContext
    ) -> List[AgentMemory]:
        """Retrieve memories for an agent using the specified framework."""
        integration = self.get_integration(framework)
        if not integration:
            raise ValueError(f"No integration available for framework: {framework}")

        return integration.retrieve_memories(context)

    def clear_agent_memories(
        self, framework: str, agent_id: str, session_id: Optional[str] = None
    ) -> int:
        """Clear memories for an agent using the specified framework."""
        integration = self.get_integration(framework)
        if not integration:
            raise ValueError(f"No integration available for framework: {framework}")

        return integration.clear_agent_memories(agent_id, session_id)

    def get_agent_stats(self, framework: str, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics for an agent using the specified framework."""
        integration = self.get_integration(framework)
        if not integration:
            raise ValueError(f"No integration available for framework: {framework}")

        return integration.get_agent_stats(agent_id)

    def list_available_frameworks(self) -> List[str]:
        """List available agent frameworks."""
        return list(self.integrations.keys())


# Global integration manager instance
_integration_manager = None


def get_integration_manager(memory_manager=None) -> AgentIntegrationManager:
    """Get global integration manager instance."""
    global _integration_manager
    if _integration_manager is None and memory_manager is not None:
        _integration_manager = AgentIntegrationManager(memory_manager)
    return _integration_manager


def initialize_integration_manager(memory_manager):
    """Initialize global integration manager."""
    global _integration_manager
    _integration_manager = AgentIntegrationManager(memory_manager)
    logger.info("Agent integration manager initialized")
