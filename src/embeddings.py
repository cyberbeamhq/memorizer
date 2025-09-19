"""
embeddings.py
Real embedding providers for the Memorizer framework.
Supports OpenAI, Cohere, HuggingFace, and other embedding services.
"""

import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""

    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_retries: int = 3
    timeout: int = 30
    batch_size: int = 100


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            import openai

            self.client = openai.OpenAI(
                api_key=self.config.api_key or os.getenv("OPENAI_API_KEY"),
                base_url=self.config.base_url,
            )
            logger.info("OpenAI embedding provider initialized")
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                input=text, model=self.config.model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = self.client.embeddings.create(
                input=texts, model=self.config.model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension based on model."""
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return model_dimensions.get(self.config.model, 1536)


class CohereEmbeddingProvider(EmbeddingProvider):
    """Cohere embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Cohere client."""
        try:
            import cohere

            self.client = cohere.Client(
                api_key=self.config.api_key or os.getenv("COHERE_API_KEY")
            )
            logger.info("Cohere embedding provider initialized")
        except ImportError:
            raise ImportError(
                "Cohere package not installed. Install with: pip install cohere"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = self.client.embed(texts=[text], model=self.config.model)
            return response.embeddings[0]
        except Exception as e:
            logger.error(f"Cohere embedding failed: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = self.client.embed(texts=texts, model=self.config.model)
            return response.embeddings
        except Exception as e:
            logger.error(f"Cohere batch embedding failed: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension based on model."""
        model_dimensions = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384,
        }
        return model_dimensions.get(self.config.model, 1024)


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize HuggingFace model."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            model_name = self.config.model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

            # Set to evaluation mode
            self.model.eval()

            logger.info(f"HuggingFace embedding provider initialized with {model_name}")
        except ImportError:
            raise ImportError(
                "Transformers package not installed. Install with: pip install transformers torch"
            )
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace model: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            import torch

            # Tokenize and encode
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=512
            )

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

            return embeddings.tolist()
        except Exception as e:
            logger.error(f"HuggingFace embedding failed: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            import torch

            # Tokenize and encode batch
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

            return embeddings.tolist()
        except Exception as e:
            logger.error(f"HuggingFace batch embedding failed: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension from model config."""
        try:
            return self.model.config.hidden_size
        except:
            # Fallback for common models
            model_dimensions = {
                "sentence-transformers/all-MiniLM-L6-v2": 384,
                "sentence-transformers/all-mpnet-base-v2": 768,
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
            }
            return model_dimensions.get(self.config.model, 768)


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for development and testing."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.dimension = 384  # Standard dimension for mock embeddings
        logger.info("Mock embedding provider initialized")

    def embed_text(self, text: str) -> List[float]:
        """Generate mock embedding for a single text."""
        return self._generate_mock_embedding(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for multiple texts."""
        return [self._generate_mock_embedding(text) for text in texts]

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate a mock embedding using hash-based approach."""
        # Create a hash of the text
        hash_obj = hashlib.md5(text.encode("utf-8"))
        hash_bytes = hash_obj.digest()

        # Convert to embedding vector
        embedding = []
        for i in range(self.dimension):
            byte_idx = i % len(hash_bytes)
            # Normalize to [-1, 1] range
            embedding.append((hash_bytes[byte_idx] - 128) / 128.0)

        return embedding

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension


class LRUCache:
    """LRU cache with TTL support to prevent memory leaks."""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl

    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key
            for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)

    def _evict_lru(self):
        """Evict least recently used entry."""
        if self.cache:
            key, _ = self.cache.popitem(last=False)
            self.timestamps.pop(key, None)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache or self._is_expired(key):
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
            return None

        # Move to end (most recently used)
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def set(self, key: str, value: Any):
        """Set value in cache."""
        # Clean up expired entries first
        self._cleanup_expired()

        # Remove existing entry if present
        if key in self.cache:
            self.cache.pop(key)
            self.timestamps.pop(key, None)

        # Evict LRU if at capacity
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        # Add new entry
        self.cache[key] = value
        self.timestamps[key] = time.time()

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.timestamps.clear()

    def size(self) -> int:
        """Get current cache size."""
        self._cleanup_expired()
        return len(self.cache)


class EmbeddingManager:
    """Manager for embedding providers with caching and retry logic."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.provider = self._create_provider()
        # Use LRU cache with size limit and TTL to prevent memory leaks
        self.cache = LRUCache(max_size=1000, ttl=3600)  # 1000 entries, 1 hour TTL

    def _create_provider(self) -> EmbeddingProvider:
        """Create embedding provider based on configuration."""
        provider_map = {
            "openai": OpenAIEmbeddingProvider,
            "cohere": CohereEmbeddingProvider,
            "huggingface": HuggingFaceEmbeddingProvider,
            "mock": MockEmbeddingProvider,
        }

        provider_class = provider_map.get(self.config.provider.lower())
        if not provider_class:
            raise ValueError(f"Unsupported embedding provider: {self.config.provider}")

        return provider_class(self.config)

    def get_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """Get embedding for text with optional caching."""
        if use_cache:
            cache_key = self._get_cache_key(text)
            cached_embedding = self.cache.get(cache_key)
            if cached_embedding is not None:
                return cached_embedding

        try:
            embedding = self.provider.embed_text(text)

            if use_cache:
                self.cache.set(cache_key, embedding)

            return embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise

    def get_embeddings_batch(
        self, texts: List[str], use_cache: bool = True
    ) -> List[List[float]]:
        """Get embeddings for multiple texts with optional caching."""
        if not texts:
            return []

        embeddings = []
        uncached_texts = []
        uncached_indices = []

        if use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                cached_embedding = self.cache.get(cache_key)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                else:
                    embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            embeddings = [None] * len(texts)

        # Get embeddings for uncached texts
        if uncached_texts:
            try:
                uncached_embeddings = self.provider.embed_batch(uncached_texts)

                if use_cache:
                    for i, embedding in enumerate(uncached_embeddings):
                        cache_key = self._get_cache_key(uncached_texts[i])
                        self.cache.set(cache_key, embedding)
                        embeddings[uncached_indices[i]] = embedding
                else:
                    for i, embedding in enumerate(uncached_embeddings):
                        embeddings[uncached_indices[i]] = embedding

            except Exception as e:
                logger.error(f"Failed to get batch embeddings: {e}")
                raise

        return embeddings

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(
            f"{self.config.provider}:{self.config.model}:{text}".encode()
        ).hexdigest()

    def clear_cache(self):
        """Clear embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.provider.get_embedding_dimension()


# Global embedding manager instance
_embedding_manager = None


def get_embedding_manager() -> EmbeddingManager:
    """Get global embedding manager instance."""
    global _embedding_manager
    if _embedding_manager is None:
        config = EmbeddingConfig(
            provider=os.getenv("EMBEDDING_PROVIDER", "mock"),
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        _embedding_manager = EmbeddingManager(config)
    return _embedding_manager


def initialize_embedding_manager(config: EmbeddingConfig):
    """Initialize global embedding manager with custom config."""
    global _embedding_manager
    _embedding_manager = EmbeddingManager(config)
    logger.info(f"Embedding manager initialized with provider: {config.provider}")


# Convenience functions
def embed_text(text: str) -> List[float]:
    """Get embedding for text using global manager."""
    return get_embedding_manager().get_embedding(text)


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts using global manager."""
    return get_embedding_manager().get_embeddings_batch(texts)


def get_embedding_dimension() -> int:
    """Get embedding dimension using global manager."""
    return get_embedding_manager().get_embedding_dimension()


def get_cache_stats() -> Dict[str, Any]:
    """Get embedding cache statistics."""
    manager = get_embedding_manager()
    return {
        "cache_size": manager.cache.size(),
        "max_cache_size": manager.cache.max_size,
        "cache_ttl": manager.cache.ttl,
        "provider": manager.config.provider,
        "model": manager.config.model,
    }


def clear_embedding_cache():
    """Clear the embedding cache."""
    manager = get_embedding_manager()
    manager.cache.clear()
    logger.info("Embedding cache cleared")
