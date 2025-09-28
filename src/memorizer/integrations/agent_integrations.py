"""
agent_integrations_fixed.py
Fixed integrations for popular AI agent frameworks.
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
    def clear_memories(self, agent_id: str, session_id: Optional[str] = None) -> bool:
        """Clear memories for an agent or session."""
        pass

    @abstractmethod
    def get_memory_stats(self, agent_id: str) -> Dict[str, Any]:
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

            memory_id = self.memory_manager.store_memory(
                user_id=user_id,
                content=content,
                metadata=metadata,
                tier="very_new"
            )
            logger.debug(f"Stored LangChain memory: {memory_id}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to store LangChain memory: {e}")
            raise

    def retrieve_memories(self, context: AgentContext) -> List[AgentMemory]:
        """Retrieve relevant memories for a LangChain agent."""
        try:
            user_id = f"langchain_{context.agent_id}"

            # Get memories from memorizer using search
            search_results = self.memory_manager.search_memories(
                user_id=user_id,
                query=context.query,
                limit=context.max_memories
            )

            # Convert to AgentMemory format
            agent_memories = []
            for mem in search_results.memories:
                agent_memory = AgentMemory(
                    agent_id=context.agent_id,
                    session_id=mem.metadata.get("session_id", ""),
                    content=mem.content,
                    metadata=mem.metadata,
                    timestamp=datetime.fromisoformat(
                        mem.metadata.get("timestamp", datetime.now().isoformat())
                    ),
                    memory_type=mem.metadata.get("memory_type", "conversation"),
                )
                agent_memories.append(agent_memory)

            logger.debug(f"Retrieved {len(agent_memories)} LangChain memories")
            return agent_memories

        except Exception as e:
            logger.error(f"Failed to retrieve LangChain memories: {e}")
            return []

    def clear_memories(self, agent_id: str, session_id: Optional[str] = None) -> bool:
        """Clear memories for a LangChain agent or session."""
        try:
            user_id = f"langchain_{agent_id}"
            
            # Search for memories to delete
            search_results = self.memory_manager.search_memories(
                user_id=user_id,
                query="",  # Empty query to get all memories
                limit=1000  # Large limit to get all memories
            )
            
            deleted_count = 0
            for mem in search_results.memories:
                if session_id is None or mem.metadata.get("session_id") == session_id:
                    success = self.memory_manager.delete_memory(mem.id, user_id)
                    if success:
                        deleted_count += 1
            
            logger.info(f"Cleared {deleted_count} LangChain memories for agent {agent_id}")
            return deleted_count > 0

        except Exception as e:
            logger.error(f"Failed to clear LangChain memories: {e}")
            return False

    def get_memory_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics for a LangChain agent."""
        try:
            user_id = f"langchain_{agent_id}"
            
            # Search for all memories
            search_results = self.memory_manager.search_memories(
                user_id=user_id,
                query="",
                limit=1000
            )
            
            # Calculate statistics
            total_memories = len(search_results.memories)
            memory_types = {}
            sessions = set()
            
            for mem in search_results.memories:
                memory_type = mem.metadata.get("memory_type", "conversation")
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
                sessions.add(mem.metadata.get("session_id", "unknown"))
            
            return {
                "agent_id": agent_id,
                "agent_framework": "langchain",
                "total_memories": total_memories,
                "memory_types": memory_types,
                "unique_sessions": len(sessions),
                "tiers": {
                    "very_new": sum(1 for m in search_results.memories if m.tier == "very_new"),
                    "mid_term": sum(1 for m in search_results.memories if m.tier == "mid_term"),
                    "long_term": sum(1 for m in search_results.memories if m.tier == "long_term"),
                }
            }

        except Exception as e:
            logger.error(f"Failed to get LangChain memory stats: {e}")
            return {"error": str(e)}


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

            memory_id = self.memory_manager.store_memory(
                user_id=user_id,
                content=content,
                metadata=metadata,
                tier="very_new"
            )
            logger.debug(f"Stored LlamaIndex memory: {memory_id}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to store LlamaIndex memory: {e}")
            raise

    def retrieve_memories(self, context: AgentContext) -> List[AgentMemory]:
        """Retrieve relevant memories for a LlamaIndex agent."""
        try:
            user_id = f"llamaindex_{context.agent_id}"

            search_results = self.memory_manager.search_memories(
                user_id=user_id,
                query=context.query,
                limit=context.max_memories
            )

            agent_memories = []
            for mem in search_results.memories:
                agent_memory = AgentMemory(
                    agent_id=context.agent_id,
                    session_id=mem.metadata.get("session_id", ""),
                    content=mem.content,
                    metadata=mem.metadata,
                    timestamp=datetime.fromisoformat(
                        mem.metadata.get("timestamp", datetime.now().isoformat())
                    ),
                    memory_type=mem.metadata.get("memory_type", "conversation"),
                )
                agent_memories.append(agent_memory)

            logger.debug(f"Retrieved {len(agent_memories)} LlamaIndex memories")
            return agent_memories

        except Exception as e:
            logger.error(f"Failed to retrieve LlamaIndex memories: {e}")
            return []

    def clear_memories(self, agent_id: str, session_id: Optional[str] = None) -> bool:
        """Clear memories for a LlamaIndex agent or session."""
        try:
            user_id = f"llamaindex_{agent_id}"
            
            search_results = self.memory_manager.search_memories(
                user_id=user_id,
                query="",
                limit=1000
            )
            
            deleted_count = 0
            for mem in search_results.memories:
                if session_id is None or mem.metadata.get("session_id") == session_id:
                    success = self.memory_manager.delete_memory(mem.id, user_id)
                    if success:
                        deleted_count += 1
            
            logger.info(f"Cleared {deleted_count} LlamaIndex memories for agent {agent_id}")
            return deleted_count > 0

        except Exception as e:
            logger.error(f"Failed to clear LlamaIndex memories: {e}")
            return False

    def get_memory_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics for a LlamaIndex agent."""
        try:
            user_id = f"llamaindex_{agent_id}"
            
            search_results = self.memory_manager.search_memories(
                user_id=user_id,
                query="",
                limit=1000
            )
            
            total_memories = len(search_results.memories)
            memory_types = {}
            sessions = set()
            
            for mem in search_results.memories:
                memory_type = mem.metadata.get("memory_type", "conversation")
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
                sessions.add(mem.metadata.get("session_id", "unknown"))
            
            return {
                "agent_id": agent_id,
                "agent_framework": "llamaindex",
                "total_memories": total_memories,
                "memory_types": memory_types,
                "unique_sessions": len(sessions),
                "tiers": {
                    "very_new": sum(1 for m in search_results.memories if m.tier == "very_new"),
                    "mid_term": sum(1 for m in search_results.memories if m.tier == "mid_term"),
                    "long_term": sum(1 for m in search_results.memories if m.tier == "long_term"),
                }
            }

        except Exception as e:
            logger.error(f"Failed to get LlamaIndex memory stats: {e}")
            return {"error": str(e)}


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

            memory_id = self.memory_manager.store_memory(
                user_id=user_id,
                content=content,
                metadata=metadata,
                tier="very_new"
            )
            logger.debug(f"Stored AutoGPT memory: {memory_id}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to store AutoGPT memory: {e}")
            raise

    def retrieve_memories(self, context: AgentContext) -> List[AgentMemory]:
        """Retrieve relevant memories for an AutoGPT agent."""
        try:
            user_id = f"autogpt_{context.agent_id}"

            search_results = self.memory_manager.search_memories(
                user_id=user_id,
                query=context.query,
                limit=context.max_memories
            )

            agent_memories = []
            for mem in search_results.memories:
                agent_memory = AgentMemory(
                    agent_id=context.agent_id,
                    session_id=mem.metadata.get("session_id", ""),
                    content=mem.content,
                    metadata=mem.metadata,
                    timestamp=datetime.fromisoformat(
                        mem.metadata.get("timestamp", datetime.now().isoformat())
                    ),
                    memory_type=mem.metadata.get("memory_type", "conversation"),
                )
                agent_memories.append(agent_memory)

            logger.debug(f"Retrieved {len(agent_memories)} AutoGPT memories")
            return agent_memories

        except Exception as e:
            logger.error(f"Failed to retrieve AutoGPT memories: {e}")
            return []

    def clear_memories(self, agent_id: str, session_id: Optional[str] = None) -> bool:
        """Clear memories for an AutoGPT agent or session."""
        try:
            user_id = f"autogpt_{agent_id}"
            
            search_results = self.memory_manager.search_memories(
                user_id=user_id,
                query="",
                limit=1000
            )
            
            deleted_count = 0
            for mem in search_results.memories:
                if session_id is None or mem.metadata.get("session_id") == session_id:
                    success = self.memory_manager.delete_memory(mem.id, user_id)
                    if success:
                        deleted_count += 1
            
            logger.info(f"Cleared {deleted_count} AutoGPT memories for agent {agent_id}")
            return deleted_count > 0

        except Exception as e:
            logger.error(f"Failed to clear AutoGPT memories: {e}")
            return False

    def get_memory_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics for an AutoGPT agent."""
        try:
            user_id = f"autogpt_{agent_id}"
            
            search_results = self.memory_manager.search_memories(
                user_id=user_id,
                query="",
                limit=1000
            )
            
            total_memories = len(search_results.memories)
            memory_types = {}
            sessions = set()
            
            for mem in search_results.memories:
                memory_type = mem.metadata.get("memory_type", "conversation")
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
                sessions.add(mem.metadata.get("session_id", "unknown"))
            
            return {
                "agent_id": agent_id,
                "agent_framework": "autogpt",
                "total_memories": total_memories,
                "memory_types": memory_types,
                "unique_sessions": len(sessions),
                "tiers": {
                    "very_new": sum(1 for m in search_results.memories if m.tier == "very_new"),
                    "mid_term": sum(1 for m in search_results.memories if m.tier == "mid_term"),
                    "long_term": sum(1 for m in search_results.memories if m.tier == "long_term"),
                }
            }

        except Exception as e:
            logger.error(f"Failed to get AutoGPT memory stats: {e}")
            return {"error": str(e)}


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

            memory_id = self.memory_manager.store_memory(
                user_id=user_id,
                content=content,
                metadata=metadata,
                tier="very_new"
            )
            logger.debug(f"Stored CrewAI memory: {memory_id}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to store CrewAI memory: {e}")
            raise

    def retrieve_memories(self, context: AgentContext) -> List[AgentMemory]:
        """Retrieve relevant memories for a CrewAI agent."""
        try:
            user_id = f"crewai_{context.agent_id}"

            search_results = self.memory_manager.search_memories(
                user_id=user_id,
                query=context.query,
                limit=context.max_memories
            )

            agent_memories = []
            for mem in search_results.memories:
                agent_memory = AgentMemory(
                    agent_id=context.agent_id,
                    session_id=mem.metadata.get("session_id", ""),
                    content=mem.content,
                    metadata=mem.metadata,
                    timestamp=datetime.fromisoformat(
                        mem.metadata.get("timestamp", datetime.now().isoformat())
                    ),
                    memory_type=mem.metadata.get("memory_type", "conversation"),
                )
                agent_memories.append(agent_memory)

            logger.debug(f"Retrieved {len(agent_memories)} CrewAI memories")
            return agent_memories

        except Exception as e:
            logger.error(f"Failed to retrieve CrewAI memories: {e}")
            return []

    def clear_memories(self, agent_id: str, session_id: Optional[str] = None) -> bool:
        """Clear memories for a CrewAI agent or session."""
        try:
            user_id = f"crewai_{agent_id}"
            
            search_results = self.memory_manager.search_memories(
                user_id=user_id,
                query="",
                limit=1000
            )
            
            deleted_count = 0
            for mem in search_results.memories:
                if session_id is None or mem.metadata.get("session_id") == session_id:
                    success = self.memory_manager.delete_memory(mem.id, user_id)
                    if success:
                        deleted_count += 1
            
            logger.info(f"Cleared {deleted_count} CrewAI memories for agent {agent_id}")
            return deleted_count > 0

        except Exception as e:
            logger.error(f"Failed to clear CrewAI memories: {e}")
            return False

    def get_memory_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics for a CrewAI agent."""
        try:
            user_id = f"crewai_{agent_id}"
            
            search_results = self.memory_manager.search_memories(
                user_id=user_id,
                query="",
                limit=1000
            )
            
            total_memories = len(search_results.memories)
            memory_types = {}
            sessions = set()
            
            for mem in search_results.memories:
                memory_type = mem.metadata.get("memory_type", "conversation")
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
                sessions.add(mem.metadata.get("session_id", "unknown"))
            
            return {
                "agent_id": agent_id,
                "agent_framework": "crewai",
                "total_memories": total_memories,
                "memory_types": memory_types,
                "unique_sessions": len(sessions),
                "tiers": {
                    "very_new": sum(1 for m in search_results.memories if m.tier == "very_new"),
                    "mid_term": sum(1 for m in search_results.memories if m.tier == "mid_term"),
                    "long_term": sum(1 for m in search_results.memories if m.tier == "long_term"),
                }
            }

        except Exception as e:
            logger.error(f"Failed to get CrewAI memory stats: {e}")
            return {"error": str(e)}


class AgentIntegrationManager:
    """Manager for all agent integrations."""

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.integrations = {
            "langchain": LangChainIntegration(memory_manager),
            "llamaindex": LlamaIndexIntegration(memory_manager),
            "autogpt": AutoGPTIntegration(memory_manager),
            "crewai": CrewAIIntegration(memory_manager),
        }
        logger.info("Agent integration manager initialized")

    def get_integration(self, framework: str) -> Optional[AgentIntegration]:
        """Get integration for a specific framework."""
        return self.integrations.get(framework.lower())

    def store_memory(self, framework: str, memory: AgentMemory) -> str:
        """Store a memory using the specified framework integration."""
        integration = self.get_integration(framework)
        if not integration:
            raise ValueError(f"Unsupported framework: {framework}")
        return integration.store_memory(memory)

    def retrieve_memories(self, framework: str, context: AgentContext) -> List[AgentMemory]:
        """Retrieve memories using the specified framework integration."""
        integration = self.get_integration(framework)
        if not integration:
            raise ValueError(f"Unsupported framework: {framework}")
        return integration.retrieve_memories(context)

    def clear_memories(self, framework: str, agent_id: str, session_id: Optional[str] = None) -> bool:
        """Clear memories using the specified framework integration."""
        integration = self.get_integration(framework)
        if not integration:
            raise ValueError(f"Unsupported framework: {framework}")
        return integration.clear_memories(agent_id, session_id)

    def get_memory_stats(self, framework: str, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics using the specified framework integration."""
        integration = self.get_integration(framework)
        if not integration:
            raise ValueError(f"Unsupported framework: {framework}")
        return integration.get_memory_stats(agent_id)

    def list_frameworks(self) -> List[str]:
        """List all supported frameworks."""
        return list(self.integrations.keys())

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get memory statistics for all frameworks."""
        stats = {}
        for framework, integration in self.integrations.items():
            try:
                # Get stats for all agents in this framework
                # This is a simplified version - in practice you'd need to track agent IDs
                stats[framework] = {
                    "framework": framework,
                    "status": "active",
                    "integrations": 1
                }
            except Exception as e:
                stats[framework] = {
                    "framework": framework,
                    "status": "error",
                    "error": str(e)
                }
        return stats
