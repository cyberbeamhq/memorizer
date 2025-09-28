"""
memory_templates.py
Memory templates for different AI agent use cases.
Provides structured templates for common agent scenarios.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class TemplateType(Enum):
    """Types of memory templates."""

    CONVERSATION = "conversation"
    TASK_EXECUTION = "task_execution"
    DECISION_MAKING = "decision_making"
    ERROR_HANDLING = "error_handling"
    USER_PREFERENCE = "user_preference"
    CONTEXT_SWITCH = "context_switch"
    TOOL_USAGE = "tool_usage"
    GOAL_TRACKING = "goal_tracking"
    LEARNING = "learning"
    CUSTOM = "custom"


@dataclass
class MemoryTemplate:
    """Template for creating structured memories."""

    template_type: TemplateType
    name: str
    description: str
    required_fields: List[str]
    optional_fields: List[str] = field(default_factory=list)
    default_metadata: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)


class MemoryTemplateManager:
    """Manager for memory templates."""

    def __init__(self):
        self.templates = {}
        self._initialize_default_templates()
        logger.info("Memory template manager initialized")

    def _initialize_default_templates(self):
        """Initialize default memory templates."""

        # Conversation Template
        self.templates["conversation"] = MemoryTemplate(
            template_type=TemplateType.CONVERSATION,
            name="Conversation Memory",
            description="Template for storing conversation interactions",
            required_fields=["user_message", "agent_response"],
            optional_fields=["session_id", "conversation_turn", "sentiment", "intent"],
            default_metadata={"memory_type": "conversation", "priority": 2},
            validation_rules={
                "user_message": {"min_length": 1, "max_length": 10000},
                "agent_response": {"min_length": 1, "max_length": 10000},
            },
        )

        # Task Execution Template
        self.templates["task_execution"] = MemoryTemplate(
            template_type=TemplateType.TASK_EXECUTION,
            name="Task Execution Memory",
            description="Template for storing task execution details",
            required_fields=["task_description", "execution_status"],
            optional_fields=[
                "task_id",
                "execution_time",
                "tools_used",
                "output",
                "errors",
            ],
            default_metadata={"memory_type": "task_execution", "priority": 3},
            validation_rules={
                "task_description": {"min_length": 1, "max_length": 5000},
                "execution_status": {
                    "allowed_values": [
                        "pending",
                        "in_progress",
                        "completed",
                        "failed",
                        "cancelled",
                    ]
                },
            },
        )

        # Decision Making Template
        self.templates["decision_making"] = MemoryTemplate(
            template_type=TemplateType.DECISION_MAKING,
            name="Decision Making Memory",
            description="Template for storing decision-making processes",
            required_fields=["decision_context", "decision_made", "reasoning"],
            optional_fields=[
                "alternatives_considered",
                "confidence_score",
                "decision_factors",
            ],
            default_metadata={"memory_type": "decision", "priority": 3},
            validation_rules={
                "decision_context": {"min_length": 1, "max_length": 5000},
                "decision_made": {"min_length": 1, "max_length": 1000},
                "reasoning": {"min_length": 1, "max_length": 5000},
            },
        )

        # Error Handling Template
        self.templates["error_handling"] = MemoryTemplate(
            template_type=TemplateType.ERROR_HANDLING,
            name="Error Handling Memory",
            description="Template for storing error handling information",
            required_fields=["error_type", "error_message", "resolution"],
            optional_fields=[
                "error_code",
                "stack_trace",
                "recovery_action",
                "prevention_measures",
            ],
            default_metadata={"memory_type": "error", "priority": 3},
            validation_rules={
                "error_type": {"min_length": 1, "max_length": 100},
                "error_message": {"min_length": 1, "max_length": 2000},
                "resolution": {"min_length": 1, "max_length": 2000},
            },
        )

        # User Preference Template
        self.templates["user_preference"] = MemoryTemplate(
            template_type=TemplateType.USER_PREFERENCE,
            name="User Preference Memory",
            description="Template for storing user preferences and settings",
            required_fields=["preference_type", "preference_value"],
            optional_fields=["user_id", "context", "confidence", "source"],
            default_metadata={"memory_type": "preference", "priority": 2},
            validation_rules={
                "preference_type": {"min_length": 1, "max_length": 100},
                "preference_value": {"min_length": 1, "max_length": 1000},
            },
        )

        # Context Switch Template
        self.templates["context_switch"] = MemoryTemplate(
            template_type=TemplateType.CONTEXT_SWITCH,
            name="Context Switch Memory",
            description="Template for storing context switching information",
            required_fields=["from_context", "to_context", "switch_reason"],
            optional_fields=["context_data", "preserved_information", "switch_time"],
            default_metadata={"memory_type": "context", "priority": 2},
            validation_rules={
                "from_context": {"min_length": 1, "max_length": 200},
                "to_context": {"min_length": 1, "max_length": 200},
                "switch_reason": {"min_length": 1, "max_length": 1000},
            },
        )

        # Tool Usage Template
        self.templates["tool_usage"] = MemoryTemplate(
            template_type=TemplateType.TOOL_USAGE,
            name="Tool Usage Memory",
            description="Template for storing tool usage information",
            required_fields=["tool_name", "tool_input", "tool_output"],
            optional_fields=[
                "execution_time",
                "success",
                "error_message",
                "tool_version",
            ],
            default_metadata={"memory_type": "tool_call", "priority": 2},
            validation_rules={
                "tool_name": {"min_length": 1, "max_length": 100},
                "tool_input": {"min_length": 1, "max_length": 5000},
                "tool_output": {"min_length": 1, "max_length": 10000},
            },
        )

        # Goal Tracking Template
        self.templates["goal_tracking"] = MemoryTemplate(
            template_type=TemplateType.GOAL_TRACKING,
            name="Goal Tracking Memory",
            description="Template for storing goal tracking information",
            required_fields=["goal_description", "progress_status"],
            optional_fields=[
                "goal_id",
                "target_date",
                "progress_percentage",
                "milestones",
                "obstacles",
            ],
            default_metadata={"memory_type": "goal", "priority": 3},
            validation_rules={
                "goal_description": {"min_length": 1, "max_length": 2000},
                "progress_status": {
                    "allowed_values": [
                        "not_started",
                        "in_progress",
                        "completed",
                        "paused",
                        "cancelled",
                    ]
                },
            },
        )

        # Learning Template
        self.templates["learning"] = MemoryTemplate(
            template_type=TemplateType.LEARNING,
            name="Learning Memory",
            description="Template for storing learning and knowledge acquisition",
            required_fields=["learning_topic", "knowledge_gained"],
            optional_fields=[
                "source",
                "confidence_level",
                "application_context",
                "related_concepts",
            ],
            default_metadata={"memory_type": "learning", "priority": 2},
            validation_rules={
                "learning_topic": {"min_length": 1, "max_length": 200},
                "knowledge_gained": {"min_length": 1, "max_length": 5000},
            },
        )

        logger.info(f"Initialized {len(self.templates)} default memory templates")

    def get_template(self, template_name: str) -> Optional[MemoryTemplate]:
        """Get a memory template by name."""
        return self.templates.get(template_name)

    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys())

    def create_memory_from_template(
        self,
        template_name: str,
        data: Dict[str, Any],
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a memory using a template."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        # Validate required fields
        for field in template.required_fields:
            if field not in data:
                raise ValueError(
                    f"Required field '{field}' missing for template '{template_name}'"
                )

        # Validate field values
        for field, value in data.items():
            if field in template.validation_rules:
                self._validate_field(field, value, template.validation_rules[field])

        # Create memory content
        content = self._format_template_content(template, data)

        # Create metadata
        metadata = {
            **template.default_metadata,
            **data,
            "template_name": template_name,
            "template_type": template.template_type.value,
            "created_at": datetime.now().isoformat(),
        }

        if custom_metadata:
            metadata.update(custom_metadata)

        return {"content": content, "metadata": metadata}

    def _validate_field(self, field_name: str, value: Any, rules: Dict[str, Any]):
        """Validate a field value against rules."""
        if "min_length" in rules and len(str(value)) < rules["min_length"]:
            raise ValueError(
                f"Field '{field_name}' too short (min: {rules['min_length']})"
            )

        if "max_length" in rules and len(str(value)) > rules["max_length"]:
            raise ValueError(
                f"Field '{field_name}' too long (max: {rules['max_length']})"
            )

        if "allowed_values" in rules and value not in rules["allowed_values"]:
            raise ValueError(
                f"Field '{field_name}' value '{value}' not in allowed values: {rules['allowed_values']}"
            )

    def _format_template_content(
        self, template: MemoryTemplate, data: Dict[str, Any]
    ) -> str:
        """Format template content based on template type."""
        if template.template_type == TemplateType.CONVERSATION:
            return f"User: {data['user_message']}\nAgent: {data['agent_response']}"

        elif template.template_type == TemplateType.TASK_EXECUTION:
            return (
                f"Task: {data['task_description']}\nStatus: {data['execution_status']}"
            )

        elif template.template_type == TemplateType.DECISION_MAKING:
            return f"Context: {data['decision_context']}\nDecision: {data['decision_made']}\nReasoning: {data['reasoning']}"

        elif template.template_type == TemplateType.ERROR_HANDLING:
            return f"Error: {data['error_type']} - {data['error_message']}\nResolution: {data['resolution']}"

        elif template.template_type == TemplateType.USER_PREFERENCE:
            return f"Preference: {data['preference_type']} = {data['preference_value']}"

        elif template.template_type == TemplateType.CONTEXT_SWITCH:
            return f"Context switch from '{data['from_context']}' to '{data['to_context']}': {data['switch_reason']}"

        elif template.template_type == TemplateType.TOOL_USAGE:
            return f"Tool: {data['tool_name']}\nInput: {data['tool_input']}\nOutput: {data['tool_output']}"

        elif template.template_type == TemplateType.GOAL_TRACKING:
            return (
                f"Goal: {data['goal_description']}\nStatus: {data['progress_status']}"
            )

        elif template.template_type == TemplateType.LEARNING:
            return f"Topic: {data['learning_topic']}\nKnowledge: {data['knowledge_gained']}"

        else:
            # Default formatting
            content_parts = []
            for field in template.required_fields:
                if field in data:
                    content_parts.append(f"{field.title()}: {data[field]}")
            return "\n".join(content_parts)

    def add_custom_template(self, template: MemoryTemplate) -> bool:
        """Add a custom template."""
        try:
            self.templates[template.name.lower().replace(" ", "_")] = template
            logger.info(f"Added custom template: {template.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add custom template: {e}")
            return False

    def remove_template(self, template_name: str) -> bool:
        """Remove a template."""
        try:
            if template_name in self.templates:
                del self.templates[template_name]
                logger.info(f"Removed template: {template_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove template: {e}")
            return False


# Agent-specific template collections
class AgentTemplateCollections:
    """Collections of templates for specific agent types."""

    @staticmethod
    def get_conversational_agent_templates() -> List[str]:
        """Get templates for conversational agents."""
        return ["conversation", "user_preference", "context_switch", "learning"]

    @staticmethod
    def get_task_oriented_agent_templates() -> List[str]:
        """Get templates for task-oriented agents."""
        return [
            "task_execution",
            "decision_making",
            "tool_usage",
            "goal_tracking",
            "error_handling",
        ]

    @staticmethod
    def get_analytical_agent_templates() -> List[str]:
        """Get templates for analytical agents."""
        return ["decision_making", "learning", "tool_usage", "error_handling"]

    @staticmethod
    def get_creative_agent_templates() -> List[str]:
        """Get templates for creative agents."""
        return ["conversation", "learning", "context_switch", "goal_tracking"]

    @staticmethod
    def get_customer_service_agent_templates() -> List[str]:
        """Get templates for customer service agents."""
        return [
            "conversation",
            "user_preference",
            "error_handling",
            "decision_making",
            "context_switch",
        ]

    @staticmethod
    def get_ecommerce_agent_templates() -> List[str]:
        """Get templates for e-commerce agents."""
        return [
            "conversation",
            "user_preference",
            "decision_making",
            "task_execution",
            "context_switch",
        ]


# Global template manager instance
_template_manager = None


def get_template_manager() -> MemoryTemplateManager:
    """Get global template manager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = MemoryTemplateManager()
    return _template_manager


def initialize_template_manager():
    """Initialize global template manager."""
    global _template_manager
    _template_manager = MemoryTemplateManager()
    logger.info("Memory template manager initialized")


# Convenience functions
def create_memory_from_template(
    template_name: str,
    data: Dict[str, Any],
    custom_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a memory using a template."""
    return get_template_manager().create_memory_from_template(
        template_name, data, custom_metadata
    )


def get_agent_templates(agent_type: str) -> List[str]:
    """Get recommended templates for an agent type."""
    collections = AgentTemplateCollections()

    if agent_type == "conversational":
        return collections.get_conversational_agent_templates()
    elif agent_type == "task_oriented":
        return collections.get_task_oriented_agent_templates()
    elif agent_type == "analytical":
        return collections.get_analytical_agent_templates()
    elif agent_type == "creative":
        return collections.get_creative_agent_templates()
    elif agent_type == "customer_service":
        return collections.get_customer_service_agent_templates()
    elif agent_type == "ecommerce":
        return collections.get_ecommerce_agent_templates()
    else:
        return get_template_manager().list_templates()
