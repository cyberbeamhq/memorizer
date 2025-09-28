"""
Memorizer Framework Core
Main framework class that orchestrates all components.
"""

import logging
from typing import Any, Dict, Optional

from .config import FrameworkConfig
from .registry import ComponentRegistry
from .lifecycle import MemoryLifecycleManager
from ..memory.memory_manager import FrameworkMemoryManager
from ..builtins import *

logger = logging.getLogger(__name__)


class MemorizerFramework:
    """Main Memorizer framework class."""
    
    def __init__(self, config: FrameworkConfig):
        """Initialize the framework with configuration."""
        self.config = config
        self.registry = ComponentRegistry()
        self._initialize_components()
        self._memory_manager = None
        self._lifecycle_manager = None
        
    def _initialize_components(self):
        """Initialize all framework components."""
        try:
            # Initialize storage
            self.storage = self._get_component("storage", self.config.storage)
            logger.info(f"Storage initialized: {self.config.storage.name}")
            
            # Initialize cache
            self.cache = self._get_component("cache", self.config.cache)
            logger.info(f"Cache initialized: {self.config.cache.name}")
            
            # Initialize retriever
            self.retriever = self._get_component("retriever", self.config.retriever)
            logger.info(f"Retriever initialized: {self.config.retriever.name}")
            
            # Initialize summarizer
            self.summarizer = self._get_component("summarizer", self.config.summarizer)
            logger.info(f"Summarizer initialized: {self.config.summarizer.name}")
            
            # Initialize PII filter
            self.pii_filter = self._get_component("pii_filter", self.config.pii_filter)
            logger.info(f"PII filter initialized: {self.config.pii_filter.name}")
            
            # Initialize scorer
            self.scorer = self._get_component("scorer", self.config.scorer)
            logger.info(f"Scorer initialized: {self.config.scorer.name}")
            
            # Initialize task runner
            self.task_runner = self._get_component("task_runner", self.config.task_runner)
            logger.info(f"Task runner initialized: {self.config.task_runner.name}")
            
            # Initialize embedding provider
            self.embedding_provider = self._get_component("embedding_provider", self.config.embedding_provider)
            logger.info(f"Embedding provider initialized: {self.config.embedding_provider.name}")
            
            # Initialize vector store
            self.vector_store = self._get_component("vector_store", self.config.vector_store)
            logger.info(f"Vector store initialized: {self.config.vector_store.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _get_component(self, component_type: str, component_config):
        """Get a component instance."""
        logger.debug(f"Getting component {component_type} with name {component_config.name}")
        
        if not component_config.enabled:
            raise ValueError(f"Component {component_type}:{component_config.name} is disabled")
        
        # Special handling for retriever to pass storage component
        if component_type == "retriever" and hasattr(self, 'storage'):
            component = self.registry.get_instance(
                component_type,
                component_config.name,
                storage=self.storage,
                **component_config.config
            )
        else:
            component = self.registry.get_instance(
                component_type,
                component_config.name,
                **component_config.config
            )
        
        if not component:
            raise ValueError(f"Failed to initialize component {component_type}:{component_config.name}")
        
        return component
    
    def get_memory_manager(self) -> FrameworkMemoryManager:
        """Get the memory manager for this framework."""
        if self._memory_manager is None:
            self._memory_manager = FrameworkMemoryManager(self)
        return self._memory_manager
    
    def get_lifecycle_manager(self) -> MemoryLifecycleManager:
        """Get the lifecycle manager for this framework."""
        if self._lifecycle_manager is None:
            self._lifecycle_manager = MemoryLifecycleManager(
                storage=self.storage,
                summarizer=self.summarizer,
                task_runner=self.task_runner,
                config=self.config.memory_lifecycle
            )
        return self._lifecycle_manager
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the framework."""
        try:
            health_status = {
                "framework": {
                    "status": "healthy",
                    "version": "1.0.0",
                    "components_initialized": True
                },
                "components": {},
                "registry": self.registry.get_stats()
            }
            
            # Check component health
            components = [
                ("storage", self.storage),
                ("cache", self.cache),
                ("retriever", self.retriever),
                ("summarizer", self.summarizer),
                ("pii_filter", self.pii_filter),
                ("scorer", self.scorer),
                ("task_runner", self.task_runner),
                ("embedding_provider", self.embedding_provider),
                ("vector_store", self.vector_store),
            ]
            
            for name, component in components:
                try:
                    if hasattr(component, 'get_health_status'):
                        health_status["components"][name] = component.get_health_status()
                    else:
                        health_status["components"][name] = {"status": "unknown"}
                except Exception as e:
                    health_status["components"][name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "framework": {
                    "status": "unhealthy",
                    "error": str(e)
                },
                "components": {},
                "registry": {}
            }
    
    def shutdown(self):
        """Shutdown the framework gracefully."""
        try:
            logger.info("Shutting down Memorizer framework...")
            
            # Shutdown task runner
            if hasattr(self.task_runner, 'shutdown'):
                self.task_runner.shutdown()
            
            # Close storage connections
            if hasattr(self.storage, 'close'):
                self.storage.close()
            
            logger.info("Framework shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during framework shutdown: {e}")


def create_framework(config: FrameworkConfig) -> MemorizerFramework:
    """Create a new Memorizer framework instance."""
    return MemorizerFramework(config)
