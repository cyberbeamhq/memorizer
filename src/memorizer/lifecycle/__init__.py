"""
Memory Lifecycle Module
Advanced memory lifecycle management with compression policies and automation.
"""

from .compression_policies import CompressionPolicyManager, CompressionPolicy

__all__ = [
    "CompressionPolicyManager",
    "CompressionPolicy",
]

# Optional components
try:
    from .lifecycle_manager import MemoryLifecycleManager, LifecycleRule
    __all__.extend(["MemoryLifecycleManager", "LifecycleRule"])
except ImportError:
    pass

try:
    from .tier_management import TierManager, MemoryTier
    __all__.extend(["TierManager", "MemoryTier"])
except ImportError:
    pass