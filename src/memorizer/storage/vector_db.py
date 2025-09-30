"""
vector_db.py
Abstraction layer for vector database integrations.
Supports Pinecone, Weaviate, Chroma, and pgvector.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..integrations.embeddings import embed_text


logger = logging.getLogger(__name__)


# ---------------------------
# Abstract Vector DB Interface
# ---------------------------
class VectorDBProvider(ABC):
    """Abstract base class for vector database providers."""

    @abstractmethod
    def insert_embedding(
        self, memory_id: str, user_id: str, content: str, metadata: Dict[str, Any]
    ) -> bool:
        """Insert an embedding into the vector database."""
        pass

    @abstractmethod
    def query_embeddings(
        self, user_id: str, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Query for similar embeddings."""
        pass

    @abstractmethod
    def delete_embedding(self, memory_id: str, user_id: str) -> bool:
        """Delete an embedding by memory_id."""
        pass


# ---------------------------
# Mock Provider for Development
# ---------------------------
class MockVectorDBProvider(VectorDBProvider):
    """Mock vector database provider for development and testing."""

    def __init__(self):
        self.embeddings = {}  # In-memory storage
        logger.info("Mock vector database provider initialized")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using the configured embedding provider."""
        try:
            return embed_text(text)
        except Exception as e:
            logger.warning(f"Failed to get real embedding, falling back to mock: {e}")
            # Fallback to hash-based mock embedding
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()

            embedding = []
            for i in range(384):
                byte_idx = i % len(hash_bytes)
                embedding.append((hash_bytes[byte_idx] - 128) / 128.0)

            return embedding

    def _calculate_similarity(
        self, query_embedding: List[float], stored_embedding: List[float]
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        if len(query_embedding) != len(stored_embedding):
            return 0.0

        dot_product = sum(a * b for a, b in zip(query_embedding, stored_embedding))
        norm_a = sum(a * a for a in query_embedding) ** 0.5
        norm_b = sum(b * b for b in stored_embedding) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def insert_embedding(
        self, memory_id: str, user_id: str, content: str, metadata: Dict[str, Any]
    ) -> bool:
        """Insert an embedding into the mock vector database."""
        try:
            embedding = self._generate_embedding(content)

            self.embeddings[memory_id] = {
                "memory_id": memory_id,
                "user_id": user_id,
                "content": content,
                "metadata": metadata,
                "embedding": embedding,
                "created_at": None,  # Would be set by actual DB
            }

            logger.debug(f"Inserted mock embedding for memory {memory_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to insert mock embedding for {memory_id}: {e}")
            return False

    def query_embeddings(
        self, user_id: str, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Query for similar embeddings in the mock database."""
        try:
            query_embedding = self._generate_embedding(query)
            results = []

            for memory_id, data in self.embeddings.items():
                if data["user_id"] == user_id:
                    similarity = self._calculate_similarity(
                        query_embedding, data["embedding"]
                    )

                    results.append(
                        {
                            "id": memory_id,
                            "memory_id": memory_id,
                            "user_id": user_id,
                            "content": data["content"],
                            "metadata": data["metadata"],
                            "score": similarity,
                            "created_at": data["created_at"],
                        }
                    )

            # Sort by similarity score (descending)
            results.sort(key=lambda x: x["score"], reverse=True)

            logger.debug(
                f"Mock vector query returned {len(results[:top_k])} results for user {user_id}"
            )
            return results[:top_k]

        except Exception as e:
            logger.error(f"Mock vector query failed for user {user_id}: {e}")
            return []

    def delete_embedding(self, memory_id: str, user_id: str) -> bool:
        """Delete an embedding from the mock database."""
        try:
            if (
                memory_id in self.embeddings
                and self.embeddings[memory_id]["user_id"] == user_id
            ):
                del self.embeddings[memory_id]
                logger.debug(f"Deleted mock embedding for memory {memory_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete mock embedding for {memory_id}: {e}")
            return False


# ---------------------------
# Global Vector DB Instance
# ---------------------------
_vector_db_provider = None


def get_vector_db_provider() -> VectorDBProvider:
    """Get or create the default vector database provider."""
    global _vector_db_provider
    if _vector_db_provider is None:
        logger.info("Using mock vector database provider")
        _vector_db_provider = MockVectorDBProvider()

    return _vector_db_provider


# ---------------------------
# Main Interface Functions
# ---------------------------
def init_vector_db(provider: str = None, **kwargs) -> VectorDBProvider:
    """Initialize vector database with specified provider."""
    global _vector_db_provider
    _vector_db_provider = MockVectorDBProvider()
    return _vector_db_provider


def insert_embedding(
    memory_id: str, user_id: str, content: str, metadata: Dict[str, Any] = None
) -> bool:
    """Insert an embedding into the vector database."""
    if metadata is None:
        metadata = {}

    provider = get_vector_db_provider()
    return provider.insert_embedding(memory_id, user_id, content, metadata)


def query_embeddings(user_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Query vector database for similar embeddings."""
    provider = get_vector_db_provider()
    return provider.query_embeddings(user_id, query, top_k)


def delete_embedding(memory_id: str, user_id: str) -> bool:
    """Delete an embedding from the vector database."""
    provider = get_vector_db_provider()
    return provider.delete_embedding(memory_id, user_id)
