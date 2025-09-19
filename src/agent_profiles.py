"""
agent_profiles.py
Agent-specific memory profiles and configurations.
Provides predefined configurations for different types of AI agents.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class MemoryRetentionPolicy(Enum):
    """Memory retention policies."""

    SHORT_TERM = "short_term"  # 7-30 days
    MEDIUM_TERM = "medium_term"  # 30-90 days
    LONG_TERM = "long_term"  # 90+ days
    PERMANENT = "permanent"  # Never delete


class CompressionStrategy(Enum):
    """Compression strategies."""

    AGGRESSIVE = "aggressive"  # High compression, less detail
    BALANCED = "balanced"  # Moderate compression
    CONSERVATIVE = "conservative"  # Low compression, preserve detail
    NONE = "none"  # No compression


class RetrievalStrategy(Enum):
    """Retrieval strategies."""

    KEYWORD_FIRST = "keyword_first"  # Prefer keyword search
    SEMANTIC_FIRST = "semantic_first"  # Prefer semantic search
    HYBRID = "hybrid"  # Balanced approach
    CONTEXT_AWARE = "context_aware"  # Context-dependent


@dataclass
class MemoryProfile:
    """Memory profile for an agent type."""

    profile_name: str
    agent_type: str
    description: str

    # Memory lifecycle settings
    very_new_limit: int = 20
    mid_term_limit: int = 200
    long_term_limit: int = 1000

    # Time-based retention
    very_new_days: int = 10
    mid_term_days: int = 365
    long_term_days: int = 1095

    # Compression settings
    compression_strategy: CompressionStrategy = CompressionStrategy.BALANCED
    compression_enabled: bool = True
    compression_batch_size: int = 5

    # Retrieval settings
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    context_window: int = 5
    max_retrieval_items: int = 10
    min_relevance_score: float = 0.1

    # Memory types to prioritize
    priority_memory_types: List[str] = field(default_factory=list)

    # Custom metadata
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class AgentProfileManager:
    """Manager for agent memory profiles."""

    def __init__(self):
        self.profiles = {}
        self._initialize_default_profiles()
        logger.info("Agent profile manager initialized")

    def _initialize_default_profiles(self):
        """Initialize default agent profiles."""

        # Conversational Agent Profile
        self.profiles["conversational"] = MemoryProfile(
            profile_name="Conversational Agent",
            agent_type="conversational",
            description="Profile for conversational AI agents that engage in dialogue",
            very_new_limit=30,
            mid_term_limit=300,
            long_term_limit=1500,
            very_new_days=7,
            mid_term_days=180,
            long_term_days=730,
            compression_strategy=CompressionStrategy.CONSERVATIVE,
            context_window=10,
            max_retrieval_items=15,
            min_relevance_score=0.05,
            priority_memory_types=["conversation", "user_preference", "context"],
            custom_settings={
                "sentiment_analysis": True,
                "intent_tracking": True,
                "conversation_flow": True,
            },
        )

        # Task-Oriented Agent Profile
        self.profiles["task_oriented"] = MemoryProfile(
            profile_name="Task-Oriented Agent",
            agent_type="task_oriented",
            description="Profile for agents that execute specific tasks and workflows",
            very_new_limit=15,
            mid_term_limit=150,
            long_term_limit=800,
            very_new_days=14,
            mid_term_days=90,
            long_term_days=365,
            compression_strategy=CompressionStrategy.BALANCED,
            context_window=8,
            max_retrieval_items=12,
            min_relevance_score=0.15,
            priority_memory_types=["task_execution", "decision", "tool_call", "goal"],
            custom_settings={
                "task_tracking": True,
                "workflow_optimization": True,
                "error_analysis": True,
            },
        )

        # Analytical Agent Profile
        self.profiles["analytical"] = MemoryProfile(
            profile_name="Analytical Agent",
            agent_type="analytical",
            description="Profile for agents that perform data analysis and research",
            very_new_limit=25,
            mid_term_limit=500,
            long_term_limit=2000,
            very_new_days=21,
            mid_term_days=180,
            long_term_days=1095,
            compression_strategy=CompressionStrategy.CONSERVATIVE,
            context_window=15,
            max_retrieval_items=20,
            min_relevance_score=0.08,
            priority_memory_types=["decision", "learning", "tool_call", "fact"],
            custom_settings={
                "data_retention": True,
                "pattern_recognition": True,
                "hypothesis_tracking": True,
            },
        )

        # Creative Agent Profile
        self.profiles["creative"] = MemoryProfile(
            profile_name="Creative Agent",
            agent_type="creative",
            description="Profile for agents that generate creative content",
            very_new_limit=20,
            mid_term_limit=250,
            long_term_limit=1200,
            very_new_days=10,
            mid_term_days=120,
            long_term_days=730,
            compression_strategy=CompressionStrategy.CONSERVATIVE,
            context_window=12,
            max_retrieval_items=18,
            min_relevance_score=0.06,
            priority_memory_types=["conversation", "learning", "context", "preference"],
            custom_settings={
                "inspiration_tracking": True,
                "style_consistency": True,
                "creative_process": True,
            },
        )

        # Customer Service Agent Profile
        self.profiles["customer_service"] = MemoryProfile(
            profile_name="Customer Service Agent",
            agent_type="customer_service",
            description="Profile for customer service and support agents",
            very_new_limit=40,
            mid_term_limit=400,
            long_term_limit=1800,
            very_new_days=5,
            mid_term_days=90,
            long_term_days=365,
            compression_strategy=CompressionStrategy.BALANCED,
            context_window=8,
            max_retrieval_items=12,
            min_relevance_score=0.12,
            priority_memory_types=[
                "conversation",
                "user_preference",
                "error",
                "decision",
            ],
            custom_settings={
                "customer_history": True,
                "issue_resolution": True,
                "satisfaction_tracking": True,
                "escalation_patterns": True,
            },
        )

        # E-commerce Agent Profile
        self.profiles["ecommerce"] = MemoryProfile(
            profile_name="E-commerce Agent",
            agent_type="ecommerce",
            description="Profile for e-commerce and shopping assistant agents",
            very_new_limit=35,
            mid_term_limit=350,
            long_term_limit=1600,
            very_new_days=7,
            mid_term_days=60,
            long_term_days=180,
            compression_strategy=CompressionStrategy.BALANCED,
            context_window=10,
            max_retrieval_items=15,
            min_relevance_score=0.10,
            priority_memory_types=[
                "conversation",
                "user_preference",
                "decision",
                "context",
            ],
            custom_settings={
                "purchase_history": True,
                "preference_learning": True,
                "recommendation_engine": True,
                "cart_behavior": True,
            },
        )

        # Research Agent Profile
        self.profiles["research"] = MemoryProfile(
            profile_name="Research Agent",
            agent_type="research",
            description="Profile for research and information gathering agents",
            very_new_limit=30,
            mid_term_limit=600,
            long_term_limit=2500,
            very_new_days=30,
            mid_term_days=365,
            long_term_days=1825,  # 5 years
            compression_strategy=CompressionStrategy.CONSERVATIVE,
            context_window=20,
            max_retrieval_items=25,
            min_relevance_score=0.05,
            priority_memory_types=["learning", "fact", "decision", "tool_call"],
            custom_settings={
                "source_tracking": True,
                "fact_verification": True,
                "knowledge_graph": True,
                "citation_management": True,
            },
        )

        # General Agent Profile
        self.profiles["general"] = MemoryProfile(
            profile_name="General Agent",
            agent_type="general",
            description="Default profile for general-purpose agents",
            very_new_limit=20,
            mid_term_limit=200,
            long_term_limit=1000,
            very_new_days=10,
            mid_term_days=365,
            long_term_days=1095,
            compression_strategy=CompressionStrategy.BALANCED,
            context_window=5,
            max_retrieval_items=10,
            min_relevance_score=0.1,
            priority_memory_types=["conversation", "decision", "task_execution"],
            custom_settings={},
        )

        logger.info(f"Initialized {len(self.profiles)} default agent profiles")

    def get_profile(self, agent_type: str) -> Optional[MemoryProfile]:
        """Get a memory profile by agent type."""
        return self.profiles.get(agent_type)

    def list_profiles(self) -> List[str]:
        """List all available profile names."""
        return list(self.profiles.keys())

    def create_custom_profile(self, profile: MemoryProfile) -> bool:
        """Create a custom memory profile."""
        try:
            self.profiles[profile.agent_type] = profile
            logger.info(f"Created custom profile: {profile.profile_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create custom profile: {e}")
            return False

    def update_profile(self, agent_type: str, updates: Dict[str, Any]) -> bool:
        """Update an existing profile."""
        try:
            if agent_type not in self.profiles:
                return False

            profile = self.profiles[agent_type]
            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)

            logger.info(f"Updated profile: {agent_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to update profile {agent_type}: {e}")
            return False

    def delete_profile(self, agent_type: str) -> bool:
        """Delete a profile (except default ones)."""
        try:
            if agent_type in self.profiles and agent_type != "general":
                del self.profiles[agent_type]
                logger.info(f"Deleted profile: {agent_type}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete profile {agent_type}: {e}")
            return False

    def get_profile_config(self, agent_type: str) -> Dict[str, Any]:
        """Get profile configuration as dictionary."""
        profile = self.get_profile(agent_type)
        if not profile:
            return {}

        return {
            "profile_name": profile.profile_name,
            "agent_type": profile.agent_type,
            "description": profile.description,
            "memory_limits": {
                "very_new": profile.very_new_limit,
                "mid_term": profile.mid_term_limit,
                "long_term": profile.long_term_limit,
            },
            "retention_days": {
                "very_new": profile.very_new_days,
                "mid_term": profile.mid_term_days,
                "long_term": profile.long_term_days,
            },
            "compression": {
                "strategy": profile.compression_strategy.value,
                "enabled": profile.compression_enabled,
                "batch_size": profile.compression_batch_size,
            },
            "retrieval": {
                "strategy": profile.retrieval_strategy.value,
                "context_window": profile.context_window,
                "max_items": profile.max_retrieval_items,
                "min_relevance": profile.min_relevance_score,
            },
            "priority_memory_types": profile.priority_memory_types,
            "custom_settings": profile.custom_settings,
        }

    def apply_profile_to_agent(
        self, agent_type: str, agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply a profile to an agent configuration."""
        profile = self.get_profile(agent_type)
        if not profile:
            logger.warning(f"No profile found for agent type: {agent_type}")
            return agent_config

        # Apply profile settings to agent config
        updated_config = agent_config.copy()

        # Memory lifecycle settings
        updated_config.update(
            {
                "very_new_limit": profile.very_new_limit,
                "mid_term_limit": profile.mid_term_limit,
                "long_term_limit": profile.long_term_limit,
                "very_new_days": profile.very_new_days,
                "mid_term_days": profile.mid_term_days,
                "long_term_days": profile.long_term_days,
                "compression_enabled": profile.compression_enabled,
                "compression_batch_size": profile.compression_batch_size,
                "context_window": profile.context_window,
                "max_tokens": profile.max_retrieval_items * 200,  # Estimate
                "memory_ttl_days": profile.mid_term_days,
            }
        )

        # Add custom settings
        if profile.custom_settings:
            updated_config["custom_settings"] = profile.custom_settings

        logger.info(f"Applied profile '{profile.profile_name}' to agent configuration")
        return updated_config


class AgentProfileFactory:
    """Factory for creating agent configurations with profiles."""

    def __init__(self, profile_manager: AgentProfileManager):
        self.profile_manager = profile_manager

    def create_agent_config(
        self,
        agent_id: str,
        agent_type: str,
        framework: str = "general",
        custom_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create an agent configuration with profile applied."""

        # Start with base configuration
        base_config = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "framework": framework,
            "created_at": datetime.now().isoformat(),
        }

        # Apply profile
        config_with_profile = self.profile_manager.apply_profile_to_agent(
            agent_type, base_config
        )

        # Apply custom overrides
        if custom_overrides:
            config_with_profile.update(custom_overrides)

        return config_with_profile

    def create_conversational_agent(
        self, agent_id: str, framework: str = "general", **kwargs
    ) -> Dict[str, Any]:
        """Create a conversational agent configuration."""
        return self.create_agent_config(agent_id, "conversational", framework, kwargs)

    def create_task_oriented_agent(
        self, agent_id: str, framework: str = "general", **kwargs
    ) -> Dict[str, Any]:
        """Create a task-oriented agent configuration."""
        return self.create_agent_config(agent_id, "task_oriented", framework, kwargs)

    def create_analytical_agent(
        self, agent_id: str, framework: str = "general", **kwargs
    ) -> Dict[str, Any]:
        """Create an analytical agent configuration."""
        return self.create_agent_config(agent_id, "analytical", framework, kwargs)

    def create_creative_agent(
        self, agent_id: str, framework: str = "general", **kwargs
    ) -> Dict[str, Any]:
        """Create a creative agent configuration."""
        return self.create_agent_config(agent_id, "creative", framework, kwargs)

    def create_customer_service_agent(
        self, agent_id: str, framework: str = "general", **kwargs
    ) -> Dict[str, Any]:
        """Create a customer service agent configuration."""
        return self.create_agent_config(agent_id, "customer_service", framework, kwargs)

    def create_ecommerce_agent(
        self, agent_id: str, framework: str = "general", **kwargs
    ) -> Dict[str, Any]:
        """Create an e-commerce agent configuration."""
        return self.create_agent_config(agent_id, "ecommerce", framework, kwargs)

    def create_research_agent(
        self, agent_id: str, framework: str = "general", **kwargs
    ) -> Dict[str, Any]:
        """Create a research agent configuration."""
        return self.create_agent_config(agent_id, "research", framework, kwargs)


# Global profile manager instance
_profile_manager = None


def get_profile_manager() -> AgentProfileManager:
    """Get global profile manager instance."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = AgentProfileManager()
    return _profile_manager


def initialize_profile_manager():
    """Initialize global profile manager."""
    global _profile_manager
    _profile_manager = AgentProfileManager()
    logger.info("Agent profile manager initialized")


# Convenience functions
def get_agent_profile(agent_type: str) -> Optional[MemoryProfile]:
    """Get an agent profile by type."""
    return get_profile_manager().get_profile(agent_type)


def create_agent_with_profile(
    agent_id: str, agent_type: str, framework: str = "general", **kwargs
) -> Dict[str, Any]:
    """Create an agent configuration with profile applied."""
    factory = AgentProfileFactory(get_profile_manager())
    return factory.create_agent_config(agent_id, agent_type, framework, kwargs)


def list_available_profiles() -> List[str]:
    """List all available agent profiles."""
    return get_profile_manager().list_profiles()
