"""
Component Registry
Registry for managing framework components.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Callable

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """Registry for managing framework components."""
    
    def __init__(self):
        """Initialize component registry."""
        self._classes: Dict[str, Dict[str, Type]] = {}
        self._instances: Dict[str, Dict[str, Any]] = {}
        self._factories: Dict[str, Dict[str, Callable]] = {}
        
        # Register built-in components
        self._register_builtins()
        
        logger.info("Component registry initialized")
    
    def _register_builtins(self):
        """Register built-in components."""
        # Import built-in components
        try:
            from ..builtins.storage import MemoryStorage, PostgreSQLStorage, MongoDBStorage, SQLiteStorage
            from ..builtins.retrievers import KeywordRetriever, VectorRetriever, HybridRetriever, SemanticRetriever
            from ..builtins.summarizers import OpenAISummarizer, AnthropicSummarizer, GroqSummarizer, MockSummarizer, ExtractiveSummarizer, AbstractiveSummarizer
            from ..builtins.vector_stores import MemoryVectorStore, PineconeVectorStore, WeaviateVectorStore, ChromaVectorStore, PostgreSQLVectorStore, SQLiteVectorStore
            from ..builtins.embedding_providers import OpenAIEmbeddingProvider, CohereEmbeddingProvider, HuggingFaceEmbeddingProvider, MockEmbeddingProvider, LocalEmbeddingProvider
            from ..builtins.cache import RedisCacheProvider, MemoryCacheProvider, FileCacheProvider
            from ..builtins.task_runners import CeleryTaskRunner, RQTaskRunner, ThreadTaskRunner, QueueTaskRunner
            from ..builtins.filters import BasicPIIFilter, AdvancedPIIFilter, MemorizerPIIFilter, ProfanityFilter, NoOpFilter
            from ..builtins.scorers import TFIDFScorer, BM25Scorer, SemanticScorer, CosineScorer, HybridScorer, SimpleScorer
            
            # Register storage components
            self.register_class("storage", "memory", MemoryStorage)
            self.register_class("storage", "postgres", PostgreSQLStorage)
            self.register_class("storage", "postgresql", PostgreSQLStorage)
            self.register_class("storage", "mongodb", MongoDBStorage)
            self.register_class("storage", "sqlite", SQLiteStorage)

            # Register external storage providers
            try:
                from ..storage.external_providers import create_external_storage
                self.register_factory("storage", "supabase", lambda **config: create_external_storage("supabase", **config))
                self.register_factory("storage", "railway", lambda **config: create_external_storage("railway", **config))
                self.register_factory("storage", "neon", lambda **config: create_external_storage("neon", **config))
                self.register_factory("storage", "planetscale", lambda **config: create_external_storage("planetscale", **config))
                self.register_factory("storage", "cockroachdb", lambda **config: create_external_storage("cockroachdb", **config))
            except ImportError as e:
                logger.warning(f"External storage providers not available: {e}")

            # Register retriever components
            self.register_class("retriever", "keyword", KeywordRetriever)
            self.register_class("retriever", "vector", VectorRetriever)
            self.register_class("retriever", "hybrid", HybridRetriever)
            self.register_class("retriever", "semantic", SemanticRetriever)

            # Register summarizer components
            self.register_class("summarizer", "openai", OpenAISummarizer)
            self.register_class("summarizer", "anthropic", AnthropicSummarizer)
            self.register_class("summarizer", "groq", GroqSummarizer)
            self.register_class("summarizer", "mock", MockSummarizer)
            self.register_class("summarizer", "extractive", ExtractiveSummarizer)
            self.register_class("summarizer", "abstractive", AbstractiveSummarizer)

            # Register vector store components
            self.register_class("vector_store", "memory", MemoryVectorStore)
            self.register_class("vector_store", "pinecone", PineconeVectorStore)
            self.register_class("vector_store", "weaviate", WeaviateVectorStore)
            self.register_class("vector_store", "chroma", ChromaVectorStore)
            self.register_class("vector_store", "postgresql", PostgreSQLVectorStore)
            self.register_class("vector_store", "sqlite", SQLiteVectorStore)

            # Register embedding provider components
            self.register_class("embedding_provider", "openai", OpenAIEmbeddingProvider)
            self.register_class("embedding_provider", "cohere", CohereEmbeddingProvider)
            self.register_class("embedding_provider", "huggingface", HuggingFaceEmbeddingProvider)
            self.register_class("embedding_provider", "mock", MockEmbeddingProvider)
            self.register_class("embedding_provider", "local", LocalEmbeddingProvider)

            # Register cache components
            self.register_class("cache", "redis", RedisCacheProvider)
            self.register_class("cache", "memory", MemoryCacheProvider)
            self.register_class("cache", "file", FileCacheProvider)

            # Register task runner components
            self.register_class("task_runner", "celery", CeleryTaskRunner)
            self.register_class("task_runner", "rq", RQTaskRunner)
            self.register_class("task_runner", "thread", ThreadTaskRunner)
            self.register_class("task_runner", "queue", QueueTaskRunner)

            # Register filter components
            self.register_class("pii_filter", "basic", BasicPIIFilter)
            self.register_class("pii_filter", "advanced", AdvancedPIIFilter)
            self.register_class("pii_filter", "memorizer", MemorizerPIIFilter)
            self.register_class("pii_filter", "profanity", ProfanityFilter)
            self.register_class("pii_filter", "noop", NoOpFilter)

            # Register scorer components
            self.register_class("scorer", "tfidf", TFIDFScorer)
            self.register_class("scorer", "bm25", BM25Scorer)
            self.register_class("scorer", "semantic", SemanticScorer)
            self.register_class("scorer", "cosine", CosineScorer)
            self.register_class("scorer", "hybrid", HybridScorer)
            self.register_class("scorer", "simple", SimpleScorer)
            
            logger.info("Built-in components registered")
            
        except ImportError as e:
            logger.warning(f"Failed to register some built-in components: {e}")
    
    def register_class(self, component_type: str, name: str, component_class: Type):
        """Register a component class."""
        if component_type not in self._classes:
            self._classes[component_type] = {}
        
        self._classes[component_type][name] = component_class
        logger.debug(f"Registered {component_type}:{name}")
    
    def register_factory(self, component_type: str, name: str, factory_func: Callable):
        """Register a component factory function."""
        if component_type not in self._factories:
            self._factories[component_type] = {}
        
        self._factories[component_type][name] = factory_func
        logger.debug(f"Registered factory {component_type}:{name}")
    
    def get_class(self, component_type: str, name: str) -> Optional[Type]:
        """Get a component class."""
        return self._classes.get(component_type, {}).get(name)
    
    def get_instance(
        self, 
        component_type: str, 
        name: str, 
        *args, 
        **kwargs
    ) -> Optional[Any]:
        """
        Get a component instance.
        
        Args:
            component_type: Type of component
            name: Name of component
            *args: Positional arguments for constructor
            **kwargs: Keyword arguments for constructor
            
        Returns:
            Component instance if found, None otherwise
        """
        logger.debug(f"Getting instance of {component_type}:{name}")
        
        # Check if instance already exists
        if (component_type in self._instances and 
            name in self._instances[component_type]):
            return self._instances[component_type][name]
        
        # Get the class
        component_class = self.get_class(component_type, name)
        if not component_class:
            logger.error(f"Component not found: {component_type}:{name}")
            logger.error(f"Available components: {self.list_available(component_type)}")
            return None
        
        try:
            # Use factory if available
            if (component_type in self._factories and 
                name in self._factories[component_type]):
                instance = self._factories[component_type][name](*args, **kwargs)
            else:
                instance = component_class(*args, **kwargs)
            
            # Cache the instance
            if component_type not in self._instances:
                self._instances[component_type] = {}
            self._instances[component_type][name] = instance
            
            logger.debug(f"Created instance of {component_type}:{name}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create instance of {component_type}:{name}: {e}")
            return None
    
    def list_available(self, component_type: str) -> List[str]:
        """List available components of a type."""
        return list(self._classes.get(component_type, {}).keys())
    
    def list_types(self) -> List[str]:
        """List available component types."""
        return list(self._classes.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        stats = {
            "types": {},
            "total_instances": 0
        }
        
        for component_type, components in self._classes.items():
            stats["types"][component_type] = {
                "available": len(components),
                "instances": len(self._instances.get(component_type, {}))
            }
            stats["total_instances"] += len(self._instances.get(component_type, {}))
        
        return stats
    
    def clear_instances(self, component_type: Optional[str] = None):
        """Clear cached instances."""
        if component_type:
            if component_type in self._instances:
                del self._instances[component_type]
                logger.debug(f"Cleared instances for {component_type}")
        else:
            self._instances.clear()
            logger.debug("Cleared all instances")
    
    def unregister(self, component_type: str, name: str):
        """Unregister a component."""
        if component_type in self._classes and name in self._classes[component_type]:
            del self._classes[component_type][name]
            logger.debug(f"Unregistered {component_type}:{name}")
        
        if component_type in self._instances and name in self._instances[component_type]:
            del self._instances[component_type][name]
            logger.debug(f"Cleared instance of {component_type}:{name}")


# Global registry instance
_registry = None


def get_registry() -> ComponentRegistry:
    """Get the global component registry."""
    global _registry
    if _registry is None:
        _registry = ComponentRegistry()
    return _registry


def register_component(component_type: str, name: str, component_class: Type):
    """Register a component with the global registry."""
    registry = get_registry()
    registry.register_class(component_type, name, component_class)


def get_component(component_type: str, name: str, *args, **kwargs) -> Optional[Any]:
    """Get a component instance from the global registry."""
    registry = get_registry()
    return registry.get_instance(component_type, name, *args, **kwargs)
