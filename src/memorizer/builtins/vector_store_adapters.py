"""
Vector Store Adapters
Thin wrappers around official vector database SDKs.
Provides a unified interface while using production-tested implementations.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def store_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a vector with metadata."""
        pass

    @abstractmethod
    def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector."""
        pass

    @abstractmethod
    def get_vector_count(self) -> int:
        """Get total number of vectors."""
        pass

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the vector store."""
        return {"status": "healthy", "type": self.__class__.__name__}

    # Framework interface compatibility methods
    def insert_embedding(self, memory_id: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """Insert an embedding (framework interface compatibility)."""
        return self.store_vector(memory_id, embedding, metadata)

    def search_embeddings(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar embeddings (framework interface compatibility)."""
        results = self.search_vectors(query_embedding, limit)
        return [
            {
                "memory_id": memory_id,
                "score": score,
                "metadata": metadata
            }
            for memory_id, score, metadata in results
        ]

    def delete_embedding(self, memory_id: str) -> bool:
        """Delete an embedding (framework interface compatibility)."""
        return self.delete_vector(memory_id)

    def batch_insert_embeddings(self, embeddings: List[Dict[str, Any]]) -> bool:
        """Insert multiple embeddings in batch."""
        try:
            for item in embeddings:
                memory_id = item.get("memory_id")
                embedding = item.get("embedding")
                metadata = item.get("metadata", {})
                if not self.insert_embedding(memory_id, embedding, metadata):
                    return False
            return True
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            return False


class PineconeVectorStore(BaseVectorStore):
    """Adapter for Pinecone using official SDK."""

    def __init__(self, api_key: str, environment: str, index_name: str, namespace: str = "", **kwargs):
        """
        Initialize Pinecone vector store using official SDK.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the index
            namespace: Optional namespace for isolation
        """
        try:
            import pinecone
            from pinecone import Pinecone as PineconeClient

            # Initialize Pinecone client
            self.pc = PineconeClient(api_key=api_key)
            self.index = self.pc.Index(index_name)
            self.namespace = namespace

            logger.info(f"Pinecone vector store initialized: {index_name}")
        except ImportError:
            raise ImportError("pinecone-client not installed. Install with: pip install pinecone-client")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise

    def store_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store vector in Pinecone."""
        try:
            self.index.upsert(
                vectors=[(vector_id, vector, metadata or {})],
                namespace=self.namespace
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store vector in Pinecone: {e}")
            return False

    def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search vectors in Pinecone."""
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=limit,
                include_metadata=True,
                namespace=self.namespace
            )

            matches = []
            for match in results.matches:
                if match.score >= threshold:
                    matches.append((
                        match.id,
                        match.score,
                        match.metadata or {}
                    ))

            return matches
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []

    def delete_vector(self, vector_id: str) -> bool:
        """Delete vector from Pinecone."""
        try:
            self.index.delete(ids=[vector_id], namespace=self.namespace)
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector from Pinecone: {e}")
            return False

    def get_vector_count(self) -> int:
        """Get vector count from Pinecone."""
        try:
            stats = self.index.describe_index_stats()
            if self.namespace:
                return stats.namespaces.get(self.namespace, {}).get('vector_count', 0)
            return stats.total_vector_count
        except Exception as e:
            logger.error(f"Failed to get Pinecone stats: {e}")
            return 0


class WeaviateVectorStore(BaseVectorStore):
    """Adapter for Weaviate using official SDK."""

    def __init__(self, url: str, api_key: Optional[str] = None, class_name: str = "Memory", **kwargs):
        """
        Initialize Weaviate vector store using official SDK.

        Args:
            url: Weaviate instance URL
            api_key: Optional API key
            class_name: Weaviate class name for memories
        """
        try:
            import weaviate
            from weaviate.auth import AuthApiKey

            # Initialize Weaviate client
            if api_key:
                auth_config = AuthApiKey(api_key=api_key)
                self.client = weaviate.Client(url=url, auth_client_secret=auth_config)
            else:
                self.client = weaviate.Client(url=url)

            self.class_name = class_name
            self._ensure_schema()

            logger.info(f"Weaviate vector store initialized: {url}")
        except ImportError:
            raise ImportError("weaviate-client not installed. Install with: pip install weaviate-client")
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {e}")
            raise

    def _ensure_schema(self):
        """Ensure the Weaviate schema exists."""
        try:
            if not self.client.schema.exists(self.class_name):
                schema = {
                    "class": self.class_name,
                    "vectorizer": "none",  # We provide our own vectors
                    "properties": [
                        {"name": "memory_id", "dataType": ["string"]},
                        {"name": "content", "dataType": ["text"]},
                        {"name": "metadata", "dataType": ["object"]},
                    ]
                }
                self.client.schema.create_class(schema)
                logger.info(f"Created Weaviate schema: {self.class_name}")
        except Exception as e:
            logger.warning(f"Schema check failed: {e}")

    def store_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store vector in Weaviate."""
        try:
            data_object = {
                "memory_id": vector_id,
                "metadata": metadata or {}
            }

            self.client.data_object.create(
                data_object=data_object,
                class_name=self.class_name,
                vector=vector,
                uuid=vector_id
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store vector in Weaviate: {e}")
            return False

    def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search vectors in Weaviate."""
        try:
            result = (
                self.client.query
                .get(self.class_name, ["memory_id", "metadata"])
                .with_near_vector({"vector": query_vector})
                .with_limit(limit)
                .with_additional(["distance", "certainty"])
                .do()
            )

            matches = []
            if "data" in result and "Get" in result["data"]:
                objects = result["data"]["Get"][self.class_name]
                for obj in objects:
                    certainty = obj.get("_additional", {}).get("certainty", 0)
                    if certainty >= threshold:
                        matches.append((
                            obj["memory_id"],
                            certainty,
                            obj.get("metadata", {})
                        ))

            return matches
        except Exception as e:
            logger.error(f"Weaviate search failed: {e}")
            return []

    def delete_vector(self, vector_id: str) -> bool:
        """Delete vector from Weaviate."""
        try:
            self.client.data_object.delete(uuid=vector_id, class_name=self.class_name)
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector from Weaviate: {e}")
            return False

    def get_vector_count(self) -> int:
        """Get vector count from Weaviate."""
        try:
            result = (
                self.client.query
                .aggregate(self.class_name)
                .with_meta_count()
                .do()
            )
            return result["data"]["Aggregate"][self.class_name][0]["meta"]["count"]
        except Exception as e:
            logger.error(f"Failed to get Weaviate count: {e}")
            return 0


class ChromaVectorStore(BaseVectorStore):
    """Adapter for ChromaDB using official SDK."""

    def __init__(self, collection_name: str = "memories", persist_directory: Optional[str] = None, **kwargs):
        """
        Initialize ChromaDB vector store using official SDK.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistence (None = in-memory)
        """
        try:
            import chromadb

            # Initialize ChromaDB client
            if persist_directory:
                self.client = chromadb.PersistentClient(path=persist_directory)
            else:
                self.client = chromadb.Client()

            self.collection = self.client.get_or_create_collection(name=collection_name)

            logger.info(f"ChromaDB vector store initialized: {collection_name}")
        except ImportError:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def store_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store vector in ChromaDB."""
        try:
            self.collection.add(
                ids=[vector_id],
                embeddings=[vector],
                metadatas=[metadata or {}]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store vector in ChromaDB: {e}")
            return False

    def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search vectors in ChromaDB."""
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=limit
            )

            matches = []
            if results and results["ids"]:
                for i, (vector_id, distance, metadata) in enumerate(zip(
                    results["ids"][0],
                    results["distances"][0],
                    results["metadatas"][0]
                )):
                    # ChromaDB returns distance, convert to similarity score
                    similarity = 1 / (1 + distance)
                    if similarity >= threshold:
                        matches.append((vector_id, similarity, metadata or {}))

            return matches
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []

    def delete_vector(self, vector_id: str) -> bool:
        """Delete vector from ChromaDB."""
        try:
            self.collection.delete(ids=[vector_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector from ChromaDB: {e}")
            return False

    def get_vector_count(self) -> int:
        """Get vector count from ChromaDB."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get ChromaDB count: {e}")
            return 0


# Factory function
def create_vector_store(provider: str, **config) -> BaseVectorStore:
    """
    Create a vector store instance using official SDKs.

    Args:
        provider: Vector store provider ("pinecone", "weaviate", "chroma")
        **config: Provider-specific configuration

    Returns:
        BaseVectorStore instance
    """
    providers = {
        "pinecone": PineconeVectorStore,
        "weaviate": WeaviateVectorStore,
        "chroma": ChromaVectorStore,
        "chromadb": ChromaVectorStore,
    }

    provider_class = providers.get(provider.lower())
    if not provider_class:
        raise ValueError(f"Unsupported vector store provider: {provider}. Choose from: {list(providers.keys())}")

    return provider_class(**config)
