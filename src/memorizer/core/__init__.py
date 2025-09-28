"""
Core framework components.
"""

from .framework import MemorizerFramework, create_framework
from .config import FrameworkConfig, load_config
from .interfaces import Memory, Query, RetrievalResult, ComponentConfig
from .lifecycle import MemoryLifecycleManager
from .registry import ComponentRegistry

__all__ = [
    "MemorizerFramework",
    "create_framework",
    "FrameworkConfig",
    "load_config", 
    "Memory",
    "Query",
    "RetrievalResult",
    "ComponentConfig",
    "MemoryLifecycleManager",
    "ComponentRegistry",
]