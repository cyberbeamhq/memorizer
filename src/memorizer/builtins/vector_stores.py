"""
Built-in Vector Store Components
Provides default vector storage implementations for the Memorizer framework.
"""

import json
import logging
import sqlite3
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np

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


class MemoryVectorStore(BaseVectorStore):
    """In-memory vector store for development and testing."""

    def __init__(self, **kwargs):
        self._vectors: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}
        self._lock = threading.RLock()

    def store_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a vector with metadata in memory."""
        with self._lock:
            try:
                self._vectors[vector_id] = (vector, metadata or {})
                logger.debug(f"Stored vector {vector_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to store vector {vector_id}: {e}")
                return False

    def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors using cosine similarity."""
        with self._lock:
            try:
                if not query_vector:
                    return []

                results = []
                query_np = np.array(query_vector)
                query_norm = np.linalg.norm(query_np)

                if query_norm == 0:
                    return []

                for vector_id, (stored_vector, metadata) in self._vectors.items():
                    try:
                        stored_np = np.array(stored_vector)
                        stored_norm = np.linalg.norm(stored_np)

                        if stored_norm == 0:
                            continue

                        # Calculate cosine similarity
                        similarity = np.dot(query_np, stored_np) / (query_norm * stored_norm)

                        if similarity >= threshold:
                            results.append((vector_id, float(similarity), metadata))

                    except Exception as e:
                        logger.warning(f"Error calculating similarity for vector {vector_id}: {e}")
                        continue

                # Sort by similarity (highest first) and limit results
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:limit]

            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                return []

    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector from memory."""
        with self._lock:
            try:
                if vector_id in self._vectors:
                    del self._vectors[vector_id]
                    logger.debug(f"Deleted vector {vector_id}")
                    return True
                return False
            except Exception as e:
                logger.error(f"Failed to delete vector {vector_id}: {e}")
                return False

    def get_vector_count(self) -> int:
        """Get total number of vectors."""
        with self._lock:
            return len(self._vectors)

    def clear_all_vectors(self) -> int:
        """Clear all vectors. Returns count of deleted vectors."""
        with self._lock:
            count = len(self._vectors)
            self._vectors.clear()
            logger.info(f"Cleared {count} vectors")
            return count


class SQLiteVectorStore(BaseVectorStore):
    """SQLite-based vector store for persistent storage."""

    def __init__(self, db_path: str = "vectors.db", **kwargs):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    vector TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vectors_created_at
                ON vectors (created_at)
            """)

    def store_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a vector with metadata in SQLite."""
        with self._lock:
            try:
                from datetime import datetime

                vector_json = json.dumps(vector)
                metadata_json = json.dumps(metadata or {})
                created_at = datetime.now().isoformat()

                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO vectors (id, vector, metadata, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (vector_id, vector_json, metadata_json, created_at)
                    )

                logger.debug(f"Stored vector {vector_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to store vector {vector_id}: {e}")
                return False

    def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors using cosine similarity."""
        with self._lock:
            try:
                if not query_vector:
                    return []

                results = []
                query_np = np.array(query_vector)
                query_norm = np.linalg.norm(query_np)

                if query_norm == 0:
                    return []

                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute("SELECT id, vector, metadata FROM vectors")

                    for row in cursor:
                        try:
                            stored_vector = json.loads(row["vector"])
                            metadata = json.loads(row["metadata"]) if row["metadata"] else {}

                            stored_np = np.array(stored_vector)
                            stored_norm = np.linalg.norm(stored_np)

                            if stored_norm == 0:
                                continue

                            # Calculate cosine similarity
                            similarity = np.dot(query_np, stored_np) / (query_norm * stored_norm)

                            if similarity >= threshold:
                                results.append((row["id"], float(similarity), metadata))

                        except Exception as e:
                            logger.warning(f"Error processing vector {row['id']}: {e}")
                            continue

                # Sort by similarity (highest first) and limit results
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:limit]

            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                return []

    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector from SQLite."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("DELETE FROM vectors WHERE id = ?", (vector_id,))
                    if cursor.rowcount > 0:
                        logger.debug(f"Deleted vector {vector_id}")
                        return True
                    return False

            except Exception as e:
                logger.error(f"Failed to delete vector {vector_id}: {e}")
                return False

    def get_vector_count(self) -> int:
        """Get total number of vectors."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM vectors")
                    return cursor.fetchone()[0]

            except Exception as e:
                logger.error(f"Failed to get vector count: {e}")
                return 0

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the SQLite vector store."""
        try:
            count = self.get_vector_count()
            return {
                "status": "healthy",
                "type": "SQLiteVectorStore",
                "vector_count": count,
                "db_path": self.db_path
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "type": "SQLiteVectorStore",
                "error": str(e)
            }


class PineconeVectorStore(BaseVectorStore):
    """Pinecone vector store with real integration."""

    def __init__(self, api_key: str = "", environment: str = "", index_name: str = "", **kwargs):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self._client = None
        self._index = None
        self._fallback = SQLiteVectorStore(**kwargs)

        # Try to initialize Pinecone
        self._initialize_pinecone()

    def _initialize_pinecone(self):
        """Initialize Pinecone client if available."""
        try:
            if not self.api_key:
                logger.warning("No Pinecone API key provided, using fallback")
                return

            # Try to import and initialize Pinecone
            import pinecone

            # Initialize Pinecone
            pinecone.init(api_key=self.api_key, environment=self.environment)

            # Check if index exists
            if self.index_name not in pinecone.list_indexes():
                logger.info(f"Creating Pinecone index: {self.index_name}")
                # Create index with default dimensions (1536 for OpenAI embeddings)
                pinecone.create_index(
                    name=self.index_name,
                    dimension=1536,  # Default for OpenAI text-embedding-ada-002
                    metric="cosine"
                )

            self._index = pinecone.Index(self.index_name)
            logger.info(f"Pinecone vector store initialized with index: {self.index_name}")

        except ImportError:
            logger.warning("pinecone-client not installed, using fallback. Install with: pip install pinecone-client")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}, using fallback")

    def store_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store vector using Pinecone."""
        if self._index is None:
            return self._fallback.store_vector(vector_id, vector, metadata)

        try:
            # Prepare metadata for Pinecone (must be strings, numbers, or booleans)
            pinecone_metadata = {}
            if metadata:
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        pinecone_metadata[k] = v
                    else:
                        pinecone_metadata[k] = str(v)

            # Upsert to Pinecone
            self._index.upsert(vectors=[(vector_id, vector, pinecone_metadata)])
            logger.debug(f"Stored vector {vector_id} in Pinecone")
            return True

        except Exception as e:
            logger.error(f"Failed to store vector in Pinecone: {e}, falling back")
            return self._fallback.store_vector(vector_id, vector, metadata)

    def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search vectors using Pinecone."""
        if self._index is None:
            return self._fallback.search_vectors(query_vector, limit, threshold)

        try:
            # Query Pinecone
            results = self._index.query(
                vector=query_vector,
                top_k=limit,
                include_metadata=True
            )

            # Convert to expected format
            formatted_results = []
            for match in results.matches:
                if match.score >= threshold:
                    formatted_results.append((
                        match.id,
                        float(match.score),
                        match.metadata or {}
                    ))

            logger.debug(f"Pinecone search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search vectors in Pinecone: {e}, falling back")
            return self._fallback.search_vectors(query_vector, limit, threshold)

    def delete_vector(self, vector_id: str) -> bool:
        """Delete vector from Pinecone."""
        if self._index is None:
            return self._fallback.delete_vector(vector_id)

        try:
            self._index.delete(ids=[vector_id])
            logger.debug(f"Deleted vector {vector_id} from Pinecone")
            return True

        except Exception as e:
            logger.error(f"Failed to delete vector from Pinecone: {e}, falling back")
            return self._fallback.delete_vector(vector_id)

    def get_vector_count(self) -> int:
        """Get vector count from Pinecone."""
        if self._index is None:
            return self._fallback.get_vector_count()

        try:
            stats = self._index.describe_index_stats()
            return stats.total_vector_count
        except Exception as e:
            logger.error(f"Failed to get vector count from Pinecone: {e}")
            return self._fallback.get_vector_count()

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of Pinecone vector store."""
        if self._index is None:
            return {
                "status": "degraded",
                "type": "PineconeVectorStore",
                "error": "Pinecone not available, using fallback",
                "fallback_status": self._fallback.get_health_status()
            }

        try:
            stats = self._index.describe_index_stats()
            return {
                "status": "healthy",
                "type": "PineconeVectorStore",
                "index_name": self.index_name,
                "vector_count": stats.total_vector_count,
                "environment": self.environment
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "type": "PineconeVectorStore",
                "error": str(e)
            }


class WeaviateVectorStore(BaseVectorStore):
    """Weaviate vector store with real integration."""

    def __init__(self, url: str = "", api_key: str = "", class_name: str = "MemorizerMemory", **kwargs):
        self.url = url
        self.api_key = api_key
        self.class_name = class_name
        self._client = None
        self._fallback = SQLiteVectorStore(**kwargs)

        # Try to initialize Weaviate
        self._initialize_weaviate()

    def _initialize_weaviate(self):
        """Initialize Weaviate client if available."""
        try:
            if not self.url:
                logger.warning("No Weaviate URL provided, using fallback")
                return

            # Try to import and initialize Weaviate
            import weaviate

            # Create auth config if API key is provided
            auth_config = None
            if self.api_key:
                auth_config = weaviate.AuthApiKey(api_key=self.api_key)

            # Initialize client
            self._client = weaviate.Client(
                url=self.url,
                auth_client_secret=auth_config
            )

            # Check if schema exists, create if not
            self._ensure_schema_exists()
            logger.info(f"Weaviate vector store initialized with class: {self.class_name}")

        except ImportError:
            logger.warning("weaviate-client not installed, using fallback. Install with: pip install weaviate-client")
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {e}, using fallback")

    def _ensure_schema_exists(self):
        """Ensure the required class schema exists in Weaviate."""
        try:
            # Check if class exists
            schema = self._client.schema.get()
            existing_classes = [cls["class"] for cls in schema.get("classes", [])]

            if self.class_name not in existing_classes:
                # Create the class schema
                class_schema = {
                    "class": self.class_name,
                    "description": "Memory objects for Memorizer framework",
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "Memory content"
                        },
                        {
                            "name": "user_id",
                            "dataType": ["string"],
                            "description": "User ID"
                        },
                        {
                            "name": "tier",
                            "dataType": ["string"],
                            "description": "Memory tier"
                        },
                        {
                            "name": "created_at",
                            "dataType": ["string"],
                            "description": "Creation timestamp"
                        }
                    ]
                }
                self._client.schema.create_class(class_schema)
                logger.info(f"Created Weaviate class: {self.class_name}")

        except Exception as e:
            logger.error(f"Failed to ensure schema exists: {e}")

    def store_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store vector using Weaviate."""
        if self._client is None:
            return self._fallback.store_vector(vector_id, vector, metadata)

        try:
            # Prepare data object
            data_object = metadata.copy() if metadata else {}
            data_object["memory_id"] = vector_id

            # Create object with vector
            self._client.data_object.create(
                data_object=data_object,
                class_name=self.class_name,
                uuid=vector_id,
                vector=vector
            )

            logger.debug(f"Stored vector {vector_id} in Weaviate")
            return True

        except Exception as e:
            logger.error(f"Failed to store vector in Weaviate: {e}, falling back")
            return self._fallback.store_vector(vector_id, vector, metadata)

    def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search vectors using Weaviate."""
        if self._client is None:
            return self._fallback.search_vectors(query_vector, limit, threshold)

        try:
            # Perform vector search
            result = (
                self._client.query
                .get(self.class_name, ["memory_id"])
                .with_near_vector({"vector": query_vector})
                .with_limit(limit)
                .with_additional(["certainty"])
                .do()
            )

            # Convert results
            formatted_results = []
            if "data" in result and "Get" in result["data"]:
                objects = result["data"]["Get"].get(self.class_name, [])
                for obj in objects:
                    certainty = obj["_additional"]["certainty"]
                    if certainty >= threshold:
                        memory_id = obj.get("memory_id", obj.get("uuid", ""))
                        metadata = {k: v for k, v in obj.items() if not k.startswith("_")}
                        formatted_results.append((memory_id, float(certainty), metadata))

            logger.debug(f"Weaviate search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search vectors in Weaviate: {e}, falling back")
            return self._fallback.search_vectors(query_vector, limit, threshold)

    def delete_vector(self, vector_id: str) -> bool:
        """Delete vector from Weaviate."""
        if self._client is None:
            return self._fallback.delete_vector(vector_id)

        try:
            self._client.data_object.delete(uuid=vector_id)
            logger.debug(f"Deleted vector {vector_id} from Weaviate")
            return True

        except Exception as e:
            logger.error(f"Failed to delete vector from Weaviate: {e}, falling back")
            return self._fallback.delete_vector(vector_id)

    def get_vector_count(self) -> int:
        """Get vector count from Weaviate."""
        if self._client is None:
            return self._fallback.get_vector_count()

        try:
            result = (
                self._client.query
                .aggregate(self.class_name)
                .with_meta_count()
                .do()
            )

            if "data" in result and "Aggregate" in result["data"]:
                aggregate_data = result["data"]["Aggregate"].get(self.class_name, [])
                if aggregate_data:
                    return aggregate_data[0]["meta"]["count"]

            return 0

        except Exception as e:
            logger.error(f"Failed to get vector count from Weaviate: {e}")
            return self._fallback.get_vector_count()


class ChromaVectorStore(BaseVectorStore):
    """Chroma vector store (mock implementation)."""

    def __init__(self, persist_directory: str = "", **kwargs):
        self.persist_directory = persist_directory
        logger.warning("Chroma vector store not fully implemented, using SQLite fallback")
        self._fallback = SQLiteVectorStore(**kwargs)

    def store_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store vector using Chroma (fallback to SQLite)."""
        return self._fallback.store_vector(vector_id, vector, metadata)

    def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search vectors using Chroma (fallback to SQLite)."""
        return self._fallback.search_vectors(query_vector, limit, threshold)

    def delete_vector(self, vector_id: str) -> bool:
        """Delete vector from Chroma (fallback to SQLite)."""
        return self._fallback.delete_vector(vector_id)

    def get_vector_count(self) -> int:
        """Get vector count from Chroma (fallback to SQLite)."""
        return self._fallback.get_vector_count()


class PostgreSQLVectorStore(BaseVectorStore):
    """PostgreSQL vector store with pgvector (mock implementation)."""

    def __init__(self, connection_string: str = "", **kwargs):
        self.connection_string = connection_string
        logger.warning("PostgreSQL vector store not fully implemented, using SQLite fallback")
        self._fallback = SQLiteVectorStore(**kwargs)

    def store_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store vector using PostgreSQL with pgvector (fallback to SQLite)."""
        return self._fallback.store_vector(vector_id, vector, metadata)

    def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search vectors using PostgreSQL with pgvector (fallback to SQLite)."""
        return self._fallback.search_vectors(query_vector, limit, threshold)

    def delete_vector(self, vector_id: str) -> bool:
        """Delete vector from PostgreSQL (fallback to SQLite)."""
        return self._fallback.delete_vector(vector_id)

    def get_vector_count(self) -> int:
        """Get vector count from PostgreSQL (fallback to SQLite)."""
        return self._fallback.get_vector_count()


__all__ = [
    "BaseVectorStore",
    "MemoryVectorStore",
    "SQLiteVectorStore",
    "PineconeVectorStore",
    "WeaviateVectorStore",
    "ChromaVectorStore",
    "PostgreSQLVectorStore",
]