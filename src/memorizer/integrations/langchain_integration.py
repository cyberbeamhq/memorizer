"""
LangChain Integration for Memorizer Framework
Provides seamless integration with LangChain for memory management.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

try:
    from langchain.memory import BaseMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.schema.memory import BaseChatMessageHistory
    from langchain.callbacks.base import BaseCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseMemory = object
    BaseMessage = object
    BaseChatMessageHistory = object
    BaseCallbackHandler = object
    logger.warning("LangChain not available. Install with: pip install langchain")

from ..core.framework import MemorizerFramework
from ..core.interfaces import Memory, Query
from .agent_integrations import AgentMemory, AgentContext, AgentIntegration


class LangChainMemorizerMemory(BaseMemory):
    """LangChain memory interface for Memorizer Framework."""

    def __init__(
        self,
        memorizer_framework: MemorizerFramework,
        user_id: str,
        session_id: Optional[str] = None,
        memory_key: str = "history",
        return_messages: bool = True,
        human_prefix: str = "Human",
        ai_prefix: str = "Assistant"
    ):
        super().__init__()
        self.memorizer = memorizer_framework
        self.memory_manager = memorizer_framework.get_memory_manager()
        self.user_id = user_id
        self.session_id = session_id or f"langchain_{datetime.now().isoformat()}"
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables from Memorizer."""
        try:
            # Build query from inputs
            current_input = inputs.get("input", "")
            query = Query(
                user_id=self.user_id,
                content=current_input,
                metadata={"session_id": self.session_id}
            )

            # Retrieve relevant memories
            results = self.memory_manager.search_memories(query)

            if self.return_messages:
                messages = self._convert_memories_to_messages(results.memories)
                return {self.memory_key: messages}
            else:
                # Return as string
                history = self._convert_memories_to_string(results.memories)
                return {self.memory_key: history}

        except Exception as e:
            logger.error(f"Failed to load memory variables: {e}")
            return {self.memory_key: [] if self.return_messages else ""}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context to Memorizer."""
        try:
            # Save human input
            human_input = inputs.get("input", "")
            if human_input:
                human_memory = Memory(
                    user_id=self.user_id,
                    content=human_input,
                    metadata={
                        "session_id": self.session_id,
                        "message_type": "human",
                        "source": "langchain",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                self.memory_manager.store_memory(human_memory)

            # Save AI output
            ai_output = outputs.get("output", "")
            if ai_output:
                ai_memory = Memory(
                    user_id=self.user_id,
                    content=ai_output,
                    metadata={
                        "session_id": self.session_id,
                        "message_type": "ai",
                        "source": "langchain",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                self.memory_manager.store_memory(ai_memory)

        except Exception as e:
            logger.error(f"Failed to save context: {e}")

    def clear(self) -> None:
        """Clear memory for this session."""
        try:
            # Note: This could be enhanced to actually delete memories
            # For now, we'll just log the clear request
            logger.info(f"Clear requested for session {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")

    def _convert_memories_to_messages(self, memories: List[Memory]) -> List[BaseMessage]:
        """Convert Memorizer memories to LangChain messages."""
        if not LANGCHAIN_AVAILABLE:
            return []

        messages = []
        for memory in memories:
            message_type = memory.metadata.get("message_type", "human")

            if message_type == "human":
                messages.append(HumanMessage(content=memory.content))
            elif message_type == "ai":
                messages.append(AIMessage(content=memory.content))
            else:
                messages.append(SystemMessage(content=memory.content))

        return messages

    def _convert_memories_to_string(self, memories: List[Memory]) -> str:
        """Convert memories to string format."""
        history_parts = []
        for memory in memories:
            message_type = memory.metadata.get("message_type", "human")
            prefix = self.human_prefix if message_type == "human" else self.ai_prefix
            history_parts.append(f"{prefix}: {memory.content}")

        return "\n".join(history_parts)


class LangChainMemorizerChatHistory(BaseChatMessageHistory):
    """LangChain chat message history backed by Memorizer."""

    def __init__(
        self,
        memorizer_framework: MemorizerFramework,
        user_id: str,
        session_id: Optional[str] = None
    ):
        super().__init__()
        self.memorizer = memorizer_framework
        self.memory_manager = memorizer_framework.get_memory_manager()
        self.user_id = user_id
        self.session_id = session_id or f"chat_{datetime.now().isoformat()}"

    @property
    def messages(self) -> List[BaseMessage]:
        """Get messages from Memorizer."""
        if not LANGCHAIN_AVAILABLE:
            return []

        try:
            query = Query(
                user_id=self.user_id,
                content="",  # Empty query to get all messages
                metadata={"session_id": self.session_id}
            )

            results = self.memory_manager.search_memories(query)
            return self._convert_memories_to_messages(results.memories)

        except Exception as e:
            logger.error(f"Failed to get messages: {e}")
            return []

    def add_message(self, message: BaseMessage) -> None:
        """Add message to Memorizer."""
        try:
            message_type = "human" if isinstance(message, HumanMessage) else "ai"

            memory = Memory(
                user_id=self.user_id,
                content=message.content,
                metadata={
                    "session_id": self.session_id,
                    "message_type": message_type,
                    "source": "langchain_chat",
                    "timestamp": datetime.now().isoformat()
                }
            )

            self.memory_manager.store_memory(memory)

        except Exception as e:
            logger.error(f"Failed to add message: {e}")

    def clear(self) -> None:
        """Clear messages for this session."""
        try:
            logger.info(f"Clear requested for chat session {self.session_id}")
            # Could implement actual deletion here
        except Exception as e:
            logger.error(f"Failed to clear messages: {e}")

    def _convert_memories_to_messages(self, memories: List[Memory]) -> List[BaseMessage]:
        """Convert memories to LangChain messages."""
        if not LANGCHAIN_AVAILABLE:
            return []

        messages = []
        for memory in memories:
            message_type = memory.metadata.get("message_type", "human")

            if message_type == "human":
                messages.append(HumanMessage(content=memory.content))
            elif message_type == "ai":
                messages.append(AIMessage(content=memory.content))
            else:
                messages.append(SystemMessage(content=memory.content))

        return messages


class LangChainMemorizerCallback(BaseCallbackHandler):
    """LangChain callback handler for automatic memory capture."""

    def __init__(
        self,
        memorizer_framework: MemorizerFramework,
        user_id: str,
        session_id: Optional[str] = None,
        capture_tool_calls: bool = True,
        capture_errors: bool = True
    ):
        super().__init__()
        self.memorizer = memorizer_framework
        self.memory_manager = memorizer_framework.get_memory_manager()
        self.user_id = user_id
        self.session_id = session_id or f"callback_{datetime.now().isoformat()}"
        self.capture_tool_calls = capture_tool_calls
        self.capture_errors = capture_errors

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts."""
        try:
            if prompts:
                memory = Memory(
                    user_id=self.user_id,
                    content=prompts[0],
                    metadata={
                        "session_id": self.session_id,
                        "event_type": "llm_start",
                        "model": serialized.get("name", "unknown"),
                        "source": "langchain_callback",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                self.memory_manager.store_memory(memory)
        except Exception as e:
            logger.error(f"Failed to capture LLM start: {e}")

    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM ends."""
        try:
            if hasattr(response, 'generations') and response.generations:
                content = response.generations[0][0].text if response.generations[0] else ""

                memory = Memory(
                    user_id=self.user_id,
                    content=content,
                    metadata={
                        "session_id": self.session_id,
                        "event_type": "llm_end",
                        "source": "langchain_callback",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                self.memory_manager.store_memory(memory)
        except Exception as e:
            logger.error(f"Failed to capture LLM end: {e}")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when tool starts."""
        if not self.capture_tool_calls:
            return

        try:
            memory = Memory(
                user_id=self.user_id,
                content=f"Tool called: {serialized.get('name', 'unknown')} with input: {input_str}",
                metadata={
                    "session_id": self.session_id,
                    "event_type": "tool_start",
                    "tool_name": serialized.get("name", "unknown"),
                    "source": "langchain_callback",
                    "timestamp": datetime.now().isoformat()
                }
            )
            self.memory_manager.store_memory(memory)
        except Exception as e:
            logger.error(f"Failed to capture tool start: {e}")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when tool ends."""
        if not self.capture_tool_calls:
            return

        try:
            memory = Memory(
                user_id=self.user_id,
                content=f"Tool output: {output}",
                metadata={
                    "session_id": self.session_id,
                    "event_type": "tool_end",
                    "source": "langchain_callback",
                    "timestamp": datetime.now().isoformat()
                }
            )
            self.memory_manager.store_memory(memory)
        except Exception as e:
            logger.error(f"Failed to capture tool end: {e}")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Called when LLM encounters an error."""
        if not self.capture_errors:
            return

        try:
            memory = Memory(
                user_id=self.user_id,
                content=f"LLM Error: {str(error)}",
                metadata={
                    "session_id": self.session_id,
                    "event_type": "llm_error",
                    "error_type": type(error).__name__,
                    "source": "langchain_callback",
                    "timestamp": datetime.now().isoformat()
                }
            )
            self.memory_manager.store_memory(memory)
        except Exception as e:
            logger.error(f"Failed to capture LLM error: {e}")


class LangChainIntegration(AgentIntegration):
    """Complete LangChain integration for Memorizer."""

    def __init__(self, memorizer_framework: MemorizerFramework):
        self.memorizer = memorizer_framework
        self.memory_manager = memorizer_framework.get_memory_manager()

    def create_memory(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        **kwargs
    ) -> LangChainMemorizerMemory:
        """Create a LangChain memory instance."""
        return LangChainMemorizerMemory(
            memorizer_framework=self.memorizer,
            user_id=user_id,
            session_id=session_id,
            **kwargs
        )

    def create_chat_history(
        self,
        user_id: str,
        session_id: Optional[str] = None
    ) -> LangChainMemorizerChatHistory:
        """Create a LangChain chat history instance."""
        return LangChainMemorizerChatHistory(
            memorizer_framework=self.memorizer,
            user_id=user_id,
            session_id=session_id
        )

    def create_callback(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        **kwargs
    ) -> LangChainMemorizerCallback:
        """Create a LangChain callback handler."""
        return LangChainMemorizerCallback(
            memorizer_framework=self.memorizer,
            user_id=user_id,
            session_id=session_id,
            **kwargs
        )

    def store_memory(self, memory: AgentMemory) -> str:
        """Store memory using LangChain format."""
        memorizer_memory = Memory(
            user_id=memory.agent_id,  # Use agent_id as user_id
            content=memory.content,
            metadata={
                "session_id": memory.session_id,
                "memory_type": memory.memory_type,
                "source": "langchain",
                "timestamp": memory.timestamp.isoformat()
            }
        )
        return self.memory_manager.store_memory(memorizer_memory)

    def retrieve_memories(self, context: AgentContext) -> List[AgentMemory]:
        """Retrieve memories in LangChain format."""
        query = Query(
            user_id=context.agent_id,
            content=context.query,
            metadata={"session_id": context.session_id} if context.session_id else {}
        )

        results = self.memory_manager.search_memories(query, limit=context.max_memories)

        return [
            AgentMemory(
                agent_id=memory.user_id,
                session_id=memory.metadata.get("session_id", ""),
                content=memory.content,
                metadata=memory.metadata,
                timestamp=memory.created_at,
                memory_type=memory.metadata.get("memory_type", "conversation")
            )
            for memory in results.memories
        ]


def create_langchain_integration(memorizer_framework: MemorizerFramework) -> LangChainIntegration:
    """Factory function to create LangChain integration."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is not available. Install with: pip install langchain")

    return LangChainIntegration(memorizer_framework)