"""
CrewAI Integration for Memorizer Framework
Provides seamless integration with CrewAI for multi-agent memory management.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from crewai import Agent, Task, Crew
    from crewai.memory import ShortTermMemory, LongTermMemory
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    Agent = object
    Task = object
    Crew = object
    ShortTermMemory = object
    LongTermMemory = object
    BaseTool = object
    logger.warning("CrewAI not available. Install with: pip install crewai")

from ..core.framework import MemorizerFramework
from ..core.interfaces import Memory, Query
from .agent_integrations import AgentMemory, AgentContext, AgentIntegration


@dataclass
class CrewMemoryItem:
    """Memory item for CrewAI integration."""

    crew_id: str
    agent_id: str
    task_id: Optional[str]
    content: str
    memory_type: str  # "conversation", "task_result", "tool_usage", "decision"
    metadata: Dict[str, Any]
    timestamp: datetime


class MemorizerCrewMemory:
    """Memorizer-backed memory system for CrewAI."""

    def __init__(
        self,
        memorizer_framework: MemorizerFramework,
        crew_id: str,
        enable_short_term: bool = True,
        enable_long_term: bool = True
    ):
        self.memorizer = memorizer_framework
        self.memory_manager = memorizer_framework.get_memory_manager()
        self.crew_id = crew_id
        self.enable_short_term = enable_short_term
        self.enable_long_term = enable_long_term

    def store_agent_memory(
        self,
        agent_id: str,
        content: str,
        memory_type: str = "conversation",
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store memory for a specific agent."""
        try:
            memory_metadata = {
                "crew_id": self.crew_id,
                "agent_id": agent_id,
                "memory_type": memory_type,
                "source": "crewai",
                "timestamp": datetime.now().isoformat()
            }

            if task_id:
                memory_metadata["task_id"] = task_id

            if metadata:
                memory_metadata.update(metadata)

            memory = Memory(
                user_id=f"crew_{self.crew_id}_agent_{agent_id}",
                content=content,
                metadata=memory_metadata
            )

            return self.memory_manager.store_memory(memory)

        except Exception as e:
            logger.error(f"Failed to store agent memory: {e}")
            return ""

    def retrieve_agent_memories(
        self,
        agent_id: str,
        query: str = "",
        limit: int = 10,
        memory_types: Optional[List[str]] = None,
        task_id: Optional[str] = None
    ) -> List[CrewMemoryItem]:
        """Retrieve memories for a specific agent."""
        try:
            query_metadata = {"crew_id": self.crew_id, "agent_id": agent_id}

            if memory_types:
                # For simplicity, we'll handle this in post-processing
                pass

            if task_id:
                query_metadata["task_id"] = task_id

            memory_query = Query(
                user_id=f"crew_{self.crew_id}_agent_{agent_id}",
                content=query,
                metadata=query_metadata
            )

            results = self.memory_manager.search_memories(memory_query, limit=limit)

            crew_memories = []
            for memory in results.memories:
                if memory_types and memory.metadata.get("memory_type") not in memory_types:
                    continue

                crew_memory = CrewMemoryItem(
                    crew_id=self.crew_id,
                    agent_id=agent_id,
                    task_id=memory.metadata.get("task_id"),
                    content=memory.content,
                    memory_type=memory.metadata.get("memory_type", "conversation"),
                    metadata=memory.metadata,
                    timestamp=memory.created_at
                )
                crew_memories.append(crew_memory)

            return crew_memories

        except Exception as e:
            logger.error(f"Failed to retrieve agent memories: {e}")
            return []

    def store_crew_memory(
        self,
        content: str,
        memory_type: str = "crew_decision",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store memory at the crew level."""
        try:
            memory_metadata = {
                "crew_id": self.crew_id,
                "memory_type": memory_type,
                "source": "crewai_crew",
                "timestamp": datetime.now().isoformat()
            }

            if metadata:
                memory_metadata.update(metadata)

            memory = Memory(
                user_id=f"crew_{self.crew_id}",
                content=content,
                metadata=memory_metadata
            )

            return self.memory_manager.store_memory(memory)

        except Exception as e:
            logger.error(f"Failed to store crew memory: {e}")
            return ""

    def retrieve_crew_memories(
        self,
        query: str = "",
        limit: int = 10,
        memory_types: Optional[List[str]] = None
    ) -> List[CrewMemoryItem]:
        """Retrieve memories at the crew level."""
        try:
            query_metadata = {"crew_id": self.crew_id}

            memory_query = Query(
                user_id=f"crew_{self.crew_id}",
                content=query,
                metadata=query_metadata
            )

            results = self.memory_manager.search_memories(memory_query, limit=limit)

            crew_memories = []
            for memory in results.memories:
                if memory_types and memory.metadata.get("memory_type") not in memory_types:
                    continue

                crew_memory = CrewMemoryItem(
                    crew_id=self.crew_id,
                    agent_id="crew",
                    task_id=memory.metadata.get("task_id"),
                    content=memory.content,
                    memory_type=memory.metadata.get("memory_type", "crew_decision"),
                    metadata=memory.metadata,
                    timestamp=memory.created_at
                )
                crew_memories.append(crew_memory)

            return crew_memories

        except Exception as e:
            logger.error(f"Failed to retrieve crew memories: {e}")
            return []

    def get_context_for_agent(
        self,
        agent_id: str,
        current_task: Optional[str] = None,
        max_context_length: int = 2000
    ) -> str:
        """Get relevant context for an agent."""
        try:
            # Get recent memories
            recent_memories = self.retrieve_agent_memories(
                agent_id=agent_id,
                query=current_task or "",
                limit=5
            )

            # Get crew-level decisions
            crew_memories = self.retrieve_crew_memories(
                query=current_task or "",
                limit=3,
                memory_types=["crew_decision", "task_result"]
            )

            # Combine and format context
            context_parts = []

            if crew_memories:
                context_parts.append("Crew Context:")
                for memory in crew_memories:
                    context_parts.append(f"- {memory.content}")
                context_parts.append("")

            if recent_memories:
                context_parts.append("Agent Memory:")
                for memory in recent_memories:
                    context_parts.append(f"- {memory.content}")

            context = "\n".join(context_parts)

            # Truncate if too long
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."

            return context

        except Exception as e:
            logger.error(f"Failed to get context for agent: {e}")
            return ""


class MemorizerCrewAgent(Agent if CREWAI_AVAILABLE else object):
    """Enhanced CrewAI Agent with Memorizer integration."""

    def __init__(
        self,
        *args,
        memorizer_framework: MemorizerFramework,
        crew_id: str,
        enable_memory: bool = True,
        **kwargs
    ):
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI is not available")

        super().__init__(*args, **kwargs)

        self.memorizer_framework = memorizer_framework
        self.crew_id = crew_id
        self.enable_memory = enable_memory

        if enable_memory:
            self.crew_memory = MemorizerCrewMemory(memorizer_framework, crew_id)

    def execute_task(self, task, **kwargs):
        """Execute task with memory integration."""
        if not self.enable_memory:
            return super().execute_task(task, **kwargs)

        try:
            # Get context before task execution
            context = self.crew_memory.get_context_for_agent(
                agent_id=str(self.id),
                current_task=str(task.description) if hasattr(task, 'description') else str(task)
            )

            # Store task start
            self.crew_memory.store_agent_memory(
                agent_id=str(self.id),
                content=f"Starting task: {task.description if hasattr(task, 'description') else str(task)}",
                memory_type="task_start",
                task_id=str(task.id) if hasattr(task, 'id') else None,
                metadata={"context_provided": bool(context)}
            )

            # Execute the task
            result = super().execute_task(task, **kwargs)

            # Store task result
            self.crew_memory.store_agent_memory(
                agent_id=str(self.id),
                content=f"Task completed: {str(result)}",
                memory_type="task_result",
                task_id=str(task.id) if hasattr(task, 'id') else None,
                metadata={"success": True}
            )

            return result

        except Exception as e:
            if self.enable_memory:
                # Store task failure
                self.crew_memory.store_agent_memory(
                    agent_id=str(self.id),
                    content=f"Task failed: {str(e)}",
                    memory_type="task_error",
                    task_id=str(task.id) if hasattr(task, 'id') else None,
                    metadata={"success": False, "error": str(e)}
                )
            raise


class MemorizerCrewTool(BaseTool if CREWAI_AVAILABLE else object):
    """Base tool class with memory integration for CrewAI."""

    def __init__(
        self,
        memorizer_framework: MemorizerFramework,
        crew_id: str,
        agent_id: str,
        **kwargs
    ):
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI is not available")

        super().__init__(**kwargs)
        self.memorizer_framework = memorizer_framework
        self.crew_memory = MemorizerCrewMemory(memorizer_framework, crew_id)
        self.crew_id = crew_id
        self.agent_id = agent_id

    def _run(self, *args, **kwargs):
        """Run tool with memory tracking."""
        try:
            # Store tool usage start
            self.crew_memory.store_agent_memory(
                agent_id=self.agent_id,
                content=f"Using tool {self.name} with args: {args}, kwargs: {kwargs}",
                memory_type="tool_usage",
                metadata={"tool_name": self.name, "phase": "start"}
            )

            # Execute the tool
            result = self._execute(*args, **kwargs)

            # Store tool result
            self.crew_memory.store_agent_memory(
                agent_id=self.agent_id,
                content=f"Tool {self.name} result: {str(result)}",
                memory_type="tool_result",
                metadata={"tool_name": self.name, "phase": "complete", "success": True}
            )

            return result

        except Exception as e:
            # Store tool error
            self.crew_memory.store_agent_memory(
                agent_id=self.agent_id,
                content=f"Tool {self.name} failed: {str(e)}",
                memory_type="tool_error",
                metadata={"tool_name": self.name, "phase": "error", "success": False}
            )
            raise

    def _execute(self, *args, **kwargs):
        """Override this method in subclasses."""
        raise NotImplementedError("Subclasses must implement _execute method")


class CrewAIIntegration(AgentIntegration):
    """Complete CrewAI integration for Memorizer."""

    def __init__(self, memorizer_framework: MemorizerFramework):
        self.memorizer = memorizer_framework
        self.memory_manager = memorizer_framework.get_memory_manager()

    def create_crew_memory(self, crew_id: str) -> MemorizerCrewMemory:
        """Create a crew memory instance."""
        return MemorizerCrewMemory(self.memorizer, crew_id)

    def create_agent(
        self,
        crew_id: str,
        *args,
        enable_memory: bool = True,
        **kwargs
    ) -> MemorizerCrewAgent:
        """Create a CrewAI agent with memory integration."""
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI is not available")

        return MemorizerCrewAgent(
            *args,
            memorizer_framework=self.memorizer,
            crew_id=crew_id,
            enable_memory=enable_memory,
            **kwargs
        )

    def store_memory(self, memory: AgentMemory) -> str:
        """Store memory using CrewAI format."""
        memorizer_memory = Memory(
            user_id=f"crew_{memory.session_id}_agent_{memory.agent_id}",
            content=memory.content,
            metadata={
                "crew_id": memory.session_id,  # Using session_id as crew_id
                "agent_id": memory.agent_id,
                "memory_type": memory.memory_type,
                "source": "crewai",
                "timestamp": memory.timestamp.isoformat(),
                **memory.metadata
            }
        )
        return self.memory_manager.store_memory(memorizer_memory)

    def retrieve_memories(self, context: AgentContext) -> List[AgentMemory]:
        """Retrieve memories in CrewAI format."""
        query = Query(
            user_id=f"crew_{context.session_id}_agent_{context.agent_id}" if context.session_id else context.agent_id,
            content=context.query,
            metadata={"crew_id": context.session_id} if context.session_id else {}
        )

        results = self.memory_manager.search_memories(query, limit=context.max_memories)

        return [
            AgentMemory(
                agent_id=memory.metadata.get("agent_id", context.agent_id),
                session_id=memory.metadata.get("crew_id", ""),
                content=memory.content,
                metadata=memory.metadata,
                timestamp=memory.created_at,
                memory_type=memory.metadata.get("memory_type", "conversation")
            )
            for memory in results.memories
        ]


def create_crewai_integration(memorizer_framework: MemorizerFramework) -> CrewAIIntegration:
    """Factory function to create CrewAI integration."""
    if not CREWAI_AVAILABLE:
        raise ImportError("CrewAI is not available. Install with: pip install crewai")

    return CrewAIIntegration(memorizer_framework)