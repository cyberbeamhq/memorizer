"""
Memorizer Framework Interfaces
Core interfaces and data models for the Memorizer framework.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class Memory(BaseModel):
    """Memory data model."""
    
    id: str
    user_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tier: str = "very_new"
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    access_count: int = 0
    last_accessed: Optional[float] = None


class Query(BaseModel):
    """Query data model for memory retrieval."""
    
    text: str
    user_id: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None


class RetrievalResult(BaseModel):
    """Result of memory retrieval operation."""
    
    memories: List[Memory] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)
    total_found: int = 0
    retrieval_time: float = 0.0
    source: str = "unknown"


class ComponentConfig(BaseModel):
    """Configuration for a framework component."""
    
    type: str
    name: str
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


# Abstract Base Classes for Components

class Storage(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def store(self, memory: Memory) -> str:
        """Store a memory."""
        pass
    
    @abstractmethod
    def get(self, memory_id: str, user_id: str) -> Optional[Memory]:
        """Get a memory by ID."""
        pass
    
    @abstractmethod
    def search(self, query: Query, limit: int = 10, offset: int = 0) -> List[Memory]:
        """Search memories."""
        pass
    
    @abstractmethod
    def update(self, memory: Memory) -> bool:
        """Update a memory."""
        pass
    
    @abstractmethod
    def delete(self, memory_id: str, user_id: str) -> bool:
        """Delete a memory."""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        pass


class Retriever(ABC):
    """Abstract base class for memory retrievers."""
    
    @abstractmethod
    def retrieve(self, query: Query) -> RetrievalResult:
        """Retrieve memories based on query."""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        pass


class Summarizer(ABC):
    """Abstract base class for memory summarizers."""
    
    @abstractmethod
    def summarize(self, content: str, max_length: int = 1000) -> str:
        """Summarize content."""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        pass


class PIIFilter(ABC):
    """Abstract base class for PII filters."""
    
    @abstractmethod
    def filter(self, content: str) -> str:
        """Filter PII from content."""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        pass


class Scorer(ABC):
    """Abstract base class for relevance scorers."""
    
    @abstractmethod
    def score(self, query: str, content: str) -> float:
        """Score relevance of content to query."""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        pass


class TaskRunner(ABC):
    """Abstract base class for task runners."""
    
    @abstractmethod
    def run_task(self, task_func, *args, **kwargs):
        """Run a task."""
        pass
    
    @abstractmethod
    def run_async_task(self, task_func, *args, **kwargs):
        """Run an async task."""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        pass


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        pass


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def insert_embedding(self, memory_id: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """Insert an embedding."""
        pass
    
    @abstractmethod
    def batch_insert_embeddings(self, embeddings: List[Dict[str, Any]]) -> bool:
        """Insert multiple embeddings."""
        pass
    
    @abstractmethod
    def search_embeddings(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        pass
    
    @abstractmethod
    def delete_embedding(self, memory_id: str) -> bool:
        """Delete an embedding."""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        pass


class CacheProvider(ABC):
    """Abstract base class for cache providers."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache."""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        pass
