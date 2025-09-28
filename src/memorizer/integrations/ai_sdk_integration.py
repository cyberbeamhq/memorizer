"""
AI SDK Integration for Memorizer Framework
Provides seamless integration with Vercel AI SDK and similar frameworks.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

from ..core.framework import MemorizerFramework
from ..core.interfaces import Memory, Query
from .agent_integrations import AgentMemory, AgentContext, AgentIntegration


@dataclass
class AIMessage:
    """Standardized AI message format."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "role": self.role,
            "content": self.content
        }

        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.metadata:
            result["metadata"] = self.metadata
        if self.timestamp:
            result["timestamp"] = self.timestamp.isoformat()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIMessage":
        """Create from dictionary format."""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])

        return cls(
            role=data["role"],
            content=data["content"],
            name=data.get("name"),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            metadata=data.get("metadata"),
            timestamp=timestamp
        )


@dataclass
class Conversation:
    """Represents a conversation with messages and metadata."""

    id: str
    user_id: str
    messages: List[AIMessage]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class MemorizerChatStorage:
    """Chat storage backend using Memorizer Framework."""

    def __init__(
        self,
        memorizer_framework: MemorizerFramework,
        user_id: str,
        conversation_id: Optional[str] = None
    ):
        self.memorizer = memorizer_framework
        self.memory_manager = memorizer_framework.get_memory_manager()
        self.user_id = user_id
        self.conversation_id = conversation_id or f"chat_{datetime.now().isoformat()}"

    async def save_message(
        self,
        message: AIMessage,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a message to the conversation."""
        try:
            message_metadata = {
                "conversation_id": self.conversation_id,
                "role": message.role,
                "source": "ai_sdk",
                "timestamp": (message.timestamp or datetime.now()).isoformat()
            }

            if message.name:
                message_metadata["message_name"] = message.name
            if message.tool_calls:
                message_metadata["tool_calls"] = message.tool_calls
            if message.tool_call_id:
                message_metadata["tool_call_id"] = message.tool_call_id
            if message.metadata:
                message_metadata.update(message.metadata)
            if metadata:
                message_metadata.update(metadata)

            memory = Memory(
                user_id=self.user_id,
                content=message.content,
                metadata=message_metadata
            )

            return self.memory_manager.store_memory(memory)

        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            return ""

    async def load_messages(
        self,
        limit: int = 50,
        before_message_id: Optional[str] = None
    ) -> List[AIMessage]:
        """Load messages from the conversation."""
        try:
            query = Query(
                user_id=self.user_id,
                content="",  # Empty query to get all messages
                metadata={"conversation_id": self.conversation_id}
            )

            results = self.memory_manager.search_memories(query, limit=limit)

            messages = []
            for memory in results.memories:
                message = AIMessage(
                    role=memory.metadata.get("role", "user"),
                    content=memory.content,
                    name=memory.metadata.get("message_name"),
                    tool_calls=memory.metadata.get("tool_calls"),
                    tool_call_id=memory.metadata.get("tool_call_id"),
                    metadata=memory.metadata,
                    timestamp=memory.created_at
                )
                messages.append(message)

            # Sort by timestamp
            messages.sort(key=lambda m: m.timestamp or datetime.min)

            return messages

        except Exception as e:
            logger.error(f"Failed to load messages: {e}")
            return []

    async def save_conversation(self, conversation: Conversation) -> str:
        """Save an entire conversation."""
        try:
            # Save conversation metadata
            conv_metadata = {
                "conversation_id": conversation.id,
                "memory_type": "conversation_metadata",
                "source": "ai_sdk",
                "message_count": len(conversation.messages),
                "created_at": conversation.created_at.isoformat(),
                "updated_at": (conversation.updated_at or datetime.now()).isoformat(),
                **conversation.metadata
            }

            conv_memory = Memory(
                user_id=conversation.user_id,
                content=f"Conversation {conversation.id} with {len(conversation.messages)} messages",
                metadata=conv_metadata
            )

            conv_id = self.memory_manager.store_memory(conv_memory)

            # Save all messages
            for message in conversation.messages:
                await self.save_message(message)

            return conv_id

        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            return ""

    async def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Load a complete conversation."""
        try:
            # Load conversation metadata
            meta_query = Query(
                user_id=self.user_id,
                content="",
                metadata={
                    "conversation_id": conversation_id,
                    "memory_type": "conversation_metadata"
                }
            )

            meta_results = self.memory_manager.search_memories(meta_query, limit=1)
            if not meta_results.memories:
                return None

            meta_memory = meta_results.memories[0]

            # Load messages
            messages = await self.load_messages()

            return Conversation(
                id=conversation_id,
                user_id=self.user_id,
                messages=messages,
                metadata=meta_memory.metadata,
                created_at=datetime.fromisoformat(meta_memory.metadata["created_at"]),
                updated_at=datetime.fromisoformat(meta_memory.metadata["updated_at"]) if meta_memory.metadata.get("updated_at") else None
            )

        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return None

    async def search_conversations(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search across conversations."""
        try:
            search_query = Query(
                user_id=self.user_id,
                content=query,
                metadata={"source": "ai_sdk"}
            )

            results = self.memory_manager.search_memories(search_query, limit=limit)

            # Group by conversation
            conversations = {}
            for memory in results.memories:
                conv_id = memory.metadata.get("conversation_id")
                if conv_id:
                    if conv_id not in conversations:
                        conversations[conv_id] = {
                            "conversation_id": conv_id,
                            "messages": [],
                            "latest_timestamp": memory.created_at
                        }

                    conversations[conv_id]["messages"].append({
                        "content": memory.content,
                        "role": memory.metadata.get("role", "user"),
                        "timestamp": memory.created_at.isoformat()
                    })

            return list(conversations.values())

        except Exception as e:
            logger.error(f"Failed to search conversations: {e}")
            return []


class MemorizerChatMiddleware:
    """Middleware for AI SDK chat completion with memory integration."""

    def __init__(
        self,
        memorizer_framework: MemorizerFramework,
        user_id: str,
        conversation_id: Optional[str] = None,
        auto_save: bool = True,
        context_window: int = 10
    ):
        self.memorizer = memorizer_framework
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.auto_save = auto_save
        self.context_window = context_window
        self.chat_storage = MemorizerChatStorage(memorizer_framework, user_id, conversation_id)

    async def get_context_messages(
        self,
        current_message: str,
        include_system: bool = True
    ) -> List[AIMessage]:
        """Get relevant context messages for the current conversation."""
        try:
            # Get recent messages from this conversation
            recent_messages = await self.chat_storage.load_messages(limit=self.context_window)

            # If we have a current message, search for relevant historical context
            relevant_messages = []
            if current_message:
                search_query = Query(
                    user_id=self.user_id,
                    content=current_message,
                    metadata={"source": "ai_sdk"}
                )

                search_results = self.memory_manager.search_memories(search_query, limit=5)

                for memory in search_results.memories:
                    # Skip if it's already in recent messages
                    if memory.metadata.get("conversation_id") == self.conversation_id:
                        continue

                    relevant_message = AIMessage(
                        role=memory.metadata.get("role", "assistant"),
                        content=memory.content,
                        metadata=memory.metadata,
                        timestamp=memory.created_at
                    )
                    relevant_messages.append(relevant_message)

            # Combine and sort messages
            all_messages = recent_messages + relevant_messages
            all_messages.sort(key=lambda m: m.timestamp or datetime.min)

            return all_messages

        except Exception as e:
            logger.error(f"Failed to get context messages: {e}")
            return []

    async def process_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        response_handler: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Process chat completion with memory integration."""
        try:
            # Convert messages to AIMessage format
            ai_messages = [AIMessage.from_dict(msg) for msg in messages]

            # Save user messages if auto_save is enabled
            if self.auto_save:
                for message in ai_messages:
                    if message.role == "user":
                        await self.chat_storage.save_message(message)

            # Get relevant context
            user_message = next((msg for msg in ai_messages if msg.role == "user"), None)
            context_messages = []

            if user_message:
                context_messages = await self.get_context_messages(user_message.content)

            # Prepare enhanced messages with context
            enhanced_messages = messages.copy()

            # Add context as system message if available
            if context_messages:
                context_content = self._format_context_messages(context_messages)
                if context_content:
                    context_msg = {
                        "role": "system",
                        "content": f"Relevant context from previous conversations:\n{context_content}"
                    }
                    enhanced_messages.insert(0, context_msg)

            return {
                "messages": enhanced_messages,
                "context_added": len(context_messages) > 0,
                "context_count": len(context_messages)
            }

        except Exception as e:
            logger.error(f"Failed to process chat completion: {e}")
            return {"messages": messages, "context_added": False, "context_count": 0}

    async def save_response(
        self,
        response_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save AI response to memory."""
        if not self.auto_save:
            return ""

        try:
            response_message = AIMessage(
                role="assistant",
                content=response_content,
                metadata=metadata,
                timestamp=datetime.now()
            )

            return await self.chat_storage.save_message(response_message, metadata)

        except Exception as e:
            logger.error(f"Failed to save response: {e}")
            return ""

    def _format_context_messages(self, messages: List[AIMessage]) -> str:
        """Format context messages for inclusion in system prompt."""
        if not messages:
            return ""

        context_parts = []
        for msg in messages[-5:]:  # Last 5 context messages
            role_prefix = "User" if msg.role == "user" else "Assistant"
            timestamp = msg.timestamp.strftime("%Y-%m-%d") if msg.timestamp else ""
            context_parts.append(f"[{timestamp}] {role_prefix}: {msg.content}")

        return "\n".join(context_parts)


class AISdkIntegration(AgentIntegration):
    """Complete AI SDK integration for Memorizer."""

    def __init__(self, memorizer_framework: MemorizerFramework):
        self.memorizer = memorizer_framework
        self.memory_manager = memorizer_framework.get_memory_manager()

    def create_chat_storage(
        self,
        user_id: str,
        conversation_id: Optional[str] = None
    ) -> MemorizerChatStorage:
        """Create a chat storage instance."""
        return MemorizerChatStorage(self.memorizer, user_id, conversation_id)

    def create_middleware(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        **kwargs
    ) -> MemorizerChatMiddleware:
        """Create a chat middleware instance."""
        return MemorizerChatMiddleware(
            self.memorizer,
            user_id,
            conversation_id,
            **kwargs
        )

    def store_memory(self, memory: AgentMemory) -> str:
        """Store memory using AI SDK format."""
        memorizer_memory = Memory(
            user_id=memory.agent_id,
            content=memory.content,
            metadata={
                "conversation_id": memory.session_id,
                "memory_type": memory.memory_type,
                "source": "ai_sdk",
                "timestamp": memory.timestamp.isoformat(),
                **memory.metadata
            }
        )
        return self.memory_manager.store_memory(memorizer_memory)

    def retrieve_memories(self, context: AgentContext) -> List[AgentMemory]:
        """Retrieve memories in AI SDK format."""
        query = Query(
            user_id=context.agent_id,
            content=context.query,
            metadata={"conversation_id": context.session_id} if context.session_id else {}
        )

        results = self.memory_manager.search_memories(query, limit=context.max_memories)

        return [
            AgentMemory(
                agent_id=memory.user_id,
                session_id=memory.metadata.get("conversation_id", ""),
                content=memory.content,
                metadata=memory.metadata,
                timestamp=memory.created_at,
                memory_type=memory.metadata.get("memory_type", "conversation")
            )
            for memory in results.memories
        ]


def create_ai_sdk_integration(memorizer_framework: MemorizerFramework) -> AISdkIntegration:
    """Factory function to create AI SDK integration."""
    return AISdkIntegration(memorizer_framework)