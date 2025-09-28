"""
Unified Framework Integrations
Provides a single entry point for all AI framework integrations.
"""

import logging
from typing import Any, Dict, Optional, Type, Union
from enum import Enum

from ..core.framework import MemorizerFramework

logger = logging.getLogger(__name__)


class SupportedFramework(Enum):
    """Supported AI frameworks."""

    LANGCHAIN = "langchain"
    CREWAI = "crewai"
    AI_SDK = "ai_sdk"
    LLAMAINDEX = "llamaindex"
    AUTOGPT = "autogpt"


class FrameworkIntegrationManager:
    """Manages all AI framework integrations."""

    def __init__(self, memorizer_framework: MemorizerFramework):
        self.memorizer = memorizer_framework
        self._integrations: Dict[str, Any] = {}

    def get_integration(self, framework: Union[str, SupportedFramework]) -> Any:
        """Get integration for a specific framework."""
        framework_name = framework.value if isinstance(framework, SupportedFramework) else framework

        if framework_name in self._integrations:
            return self._integrations[framework_name]

        # Lazy load integration
        integration = self._create_integration(framework_name)
        if integration:
            self._integrations[framework_name] = integration

        return integration

    def _create_integration(self, framework_name: str) -> Optional[Any]:
        """Create integration for a specific framework."""
        try:
            if framework_name == SupportedFramework.LANGCHAIN.value:
                from .langchain_integration import create_langchain_integration
                return create_langchain_integration(self.memorizer)

            elif framework_name == SupportedFramework.CREWAI.value:
                from .crewai_integration import create_crewai_integration
                return create_crewai_integration(self.memorizer)

            elif framework_name == SupportedFramework.AI_SDK.value:
                from .ai_sdk_integration import create_ai_sdk_integration
                return create_ai_sdk_integration(self.memorizer)

            else:
                logger.warning(f"Unknown framework: {framework_name}")
                return None

        except ImportError as e:
            logger.error(f"Failed to import {framework_name} integration: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create {framework_name} integration: {e}")
            return None

    def list_available_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """List all available framework integrations."""
        frameworks = {}

        for framework in SupportedFramework:
            try:
                integration = self.get_integration(framework)
                frameworks[framework.value] = {
                    "available": integration is not None,
                    "integration_type": type(integration).__name__ if integration else None
                }
            except Exception as e:
                frameworks[framework.value] = {
                    "available": False,
                    "error": str(e)
                }

        return frameworks

    def create_langchain_memory(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        **kwargs
    ):
        """Create LangChain memory integration."""
        integration = self.get_integration(SupportedFramework.LANGCHAIN)
        if not integration:
            raise ImportError("LangChain integration not available")
        return integration.create_memory(user_id, session_id, **kwargs)

    def create_crewai_memory(self, crew_id: str):
        """Create CrewAI memory integration."""
        integration = self.get_integration(SupportedFramework.CREWAI)
        if not integration:
            raise ImportError("CrewAI integration not available")
        return integration.create_crew_memory(crew_id)

    def create_ai_sdk_storage(
        self,
        user_id: str,
        conversation_id: Optional[str] = None
    ):
        """Create AI SDK storage integration."""
        integration = self.get_integration(SupportedFramework.AI_SDK)
        if not integration:
            raise ImportError("AI SDK integration not available")
        return integration.create_chat_storage(user_id, conversation_id)


# Convenience functions for direct use
def create_integration_manager(memorizer_framework: MemorizerFramework) -> FrameworkIntegrationManager:
    """Create a framework integration manager."""
    return FrameworkIntegrationManager(memorizer_framework)


def get_langchain_memory(
    memorizer_framework: MemorizerFramework,
    user_id: str,
    session_id: Optional[str] = None,
    **kwargs
):
    """Quick function to get LangChain memory."""
    manager = FrameworkIntegrationManager(memorizer_framework)
    return manager.create_langchain_memory(user_id, session_id, **kwargs)


def get_crewai_memory(
    memorizer_framework: MemorizerFramework,
    crew_id: str
):
    """Quick function to get CrewAI memory."""
    manager = FrameworkIntegrationManager(memorizer_framework)
    return manager.create_crewai_memory(crew_id)


def get_ai_sdk_storage(
    memorizer_framework: MemorizerFramework,
    user_id: str,
    conversation_id: Optional[str] = None
):
    """Quick function to get AI SDK storage."""
    manager = FrameworkIntegrationManager(memorizer_framework)
    return manager.create_ai_sdk_storage(user_id, conversation_id)