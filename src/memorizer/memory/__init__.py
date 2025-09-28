"""
Memory management components.
"""

from .memory_manager import FrameworkMemoryManager
from .memory_templates import MemoryTemplate, MemoryTemplateManager, get_template_manager
from .compression_agent import CompressionAgent
from .compression_metrics import CompressionMetrics

__all__ = [
    "FrameworkMemoryManager",
    "MemoryTemplate",
    "MemoryTemplateManager",
    "get_template_manager",
    "CompressionAgent", 
    "CompressionMetrics",
]
