"""
memorizer.framework.factory
Framework factory for creating configured instances.
"""

import logging
from typing import Any, Dict, Optional

from .config import FrameworkConfig, load_config
from .registry import get_registry

logger = logging.getLogger(__name__)


class MemorizerFramework:
    """Main framework class that orchestrates all components."""
    
    def __init__(self, config: FrameworkConfig):
        """Initialize the framework with configuration."""
        self.config = config
        self.registry = get_registry()
        
        # Register built-in components
        register_builtin_components()
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Memorizer framework initialized")
    
    def _initialize_components(self):
        """Initialize all configured components."""
        # Initialize core components
        self.storage = self._get_component("storage", self.config.storage)
        self.retriever = self._get_component("retriever", self.config.retriever)
        self.summarizer = self._get_component("summarizer", self.config.summarizer)
        self.pii_filter = self._get_component("pii_filter", self.config.pii_filter)
        self.scorer = self._get_component("scorer", self.config.scorer)
        self.task_runner = self._get_component("task_runner", self.config.task_runner)
        self.embedding_provider = self._get_component("embedding_provider", self.config.embedding_provider)
        self.vector_store = self._get_component("vector_store", self.config.vector_store)
        self.cache = self._get_component("cache", self.config.cache)
        
        # Initialize lifecycle manager
        from .core.lifecycle import MemoryLifecycleManager
        self.lifecycle = MemoryLifecycleManager(
            storage=self.storage,
            summarizer=self.summarizer,
            task_runner=self.task_runner,
            config=self.config.memory_lifecycle
        )
    
    def _get_component(self, component_type: str, component_config) -> Any:
        """Get a component instance from the registry."""
        print(f"DEBUG: Getting component {component_type} with name {component_config.name}")
        
        if not component_config.enabled:
            raise ValueError(f"Component {component_type}:{component_config.name} is disabled")
        
        # Special handling for retriever to pass storage component
        if component_type == "retriever" and hasattr(self, 'storage'):
            component = self.registry.get_instance(
                component_type, 
                component_config.name, 
                component_config.config,
                storage=self.storage
            )
        else:
            component = self.registry.get_instance(
                component_type, 
                component_config.name, 
                component_config.config
            )
        
        if not component:
            raise ValueError(f"Failed to initialize component {component_type}:{component_config.name}")
        
        return component
    
    def get_memory_manager(self):
        """Get the memory manager for this framework."""
        from .memory_manager import FrameworkMemoryManager
        return FrameworkMemoryManager(self)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the framework."""
        try:
            return {
                "framework": {
                    "version": self.config.version,
                    "status": "healthy",
                    "debug": self.config.debug
                },
                "components": {
                    "storage": self.storage.get_health_status(),
                    "retriever": self.retriever.get_health_status(),
                    "summarizer": self.summarizer.get_health_status(),
                    "pii_filter": self.pii_filter.get_health_status(),
                    "scorer": self.scorer.get_health_status(),
                    "task_runner": self.task_runner.get_health_status(),
                    "embedding_provider": self.embedding_provider.get_health_status(),
                    "vector_store": self.vector_store.get_health_status(),
                    "cache": self.cache.get_health_status(),
                    "lifecycle": self.lifecycle.get_health_status()
                },
                "registry": self.registry.get_stats()
            }
        except Exception as e:
            logger.error(f"Failed to get framework health status: {e}")
            return {
                "framework": {
                    "status": "unhealthy",
                    "error": str(e)
                }
            }


def create_framework(
    config: Optional[FrameworkConfig] = None,
    config_path: Optional[str] = None,
    use_env: bool = True
) -> MemorizerFramework:
    """
    Create a configured Memorizer framework instance.
    
    Args:
        config: FrameworkConfig object (if provided, config_path is ignored)
        config_path: Path to configuration file
        use_env: Whether to use environment variables as fallback
        
    Returns:
        Configured MemorizerFramework instance
    """
    if config is not None:
        framework_config = config
    else:
        framework_config = load_config(config_path, use_env)
    
    return MemorizerFramework(framework_config)


def create_framework_from_dict(config_dict: Dict[str, Any]) -> MemorizerFramework:
    """
    Create a framework from a configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configured MemorizerFramework instance
    """
    from .core.config import ConfigLoader
    config = ConfigLoader._dict_to_config(config_dict)
    return MemorizerFramework(config)
