"""
Built-in implementations of core interfaces.
"""

import logging

logger = logging.getLogger(__name__)

# Import built-in components with error handling
try:
    from .storage import *
except ImportError as e:
    logger.warning(f"Failed to import storage components: {e}")

try:
    from .retrievers import *
except ImportError as e:
    logger.warning(f"Failed to import retriever components: {e}")

try:
    from .summarizers import *
except ImportError as e:
    logger.warning(f"Failed to import summarizer components: {e}")

try:
    from .vector_stores import *
except ImportError as e:
    logger.warning(f"Failed to import vector store components: {e}")

try:
    from .embedding_providers import *
except ImportError as e:
    logger.warning(f"Failed to import embedding provider components: {e}")

try:
    from .cache import *
except ImportError as e:
    logger.warning(f"Failed to import cache components: {e}")

try:
    from .task_runners import *
except ImportError as e:
    logger.warning(f"Failed to import task runner components: {e}")

try:
    from .filters import *
except ImportError as e:
    logger.warning(f"Failed to import filter components: {e}")

try:
    from .scorers import *
except ImportError as e:
    logger.warning(f"Failed to import scorer components: {e}")

__all__ = [
    # Storage implementations
    "PostgreSQLStorage",
    "MongoDBStorage", 
    "SQLiteStorage",
    
    # Retriever implementations
    "KeywordRetriever",
    "VectorRetriever",
    "HybridRetriever",
    
    # Summarizer implementations
    "OpenAISummarizer",
    "AnthropicSummarizer",
    "MockSummarizer",
    
    # Vector store implementations
    "PineconeVectorStore",
    "WeaviateVectorStore",
    "ChromaVectorStore",
    "PostgreSQLVectorStore",
    
    # Embedding provider implementations
    "OpenAIEmbeddingProvider",
    "CohereEmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
    "MockEmbeddingProvider",
    
    # Cache implementations
    "RedisCacheProvider",
    "MemoryCacheProvider",
    "FileCacheProvider",
    
    # Task runner implementations
    "CeleryTaskRunner",
    "RQTaskRunner",
    "ThreadTaskRunner",
    
    # Filter implementations
    "BasicPIIFilter",
    "AdvancedPIIFilter",
    
    # Scorer implementations
    "TFIDFScorer",
    "BM25Scorer",
    "SemanticScorer",
]
