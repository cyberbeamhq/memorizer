"""
Built-in Embedding Provider Components
Provides default embedding implementations for the Memorizer framework.
"""

import logging
import hashlib
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        pass

    def get_embedding(self, text: str) -> List[float]:
        """Alias for embed_text for compatibility."""
        return self.embed_text(text)

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the embedding provider."""
        return {"status": "healthy", "type": self.__class__.__name__}


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Mock embedding provider for testing and development."""

    def __init__(self, dimension: int = 384, **kwargs):
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        """Generate mock embeddings based on text hash."""
        if not text.strip():
            return [0.0] * self.dimension

        # Use text hash to generate deterministic "embeddings"
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Convert hash to numbers
        embedding = []
        for i in range(0, min(len(text_hash), self.dimension * 2), 2):
            hex_pair = text_hash[i:i+2]
            value = int(hex_pair, 16) / 255.0  # Normalize to 0-1
            embedding.append(float(value))

        # Pad or truncate to desired dimension
        while len(embedding) < self.dimension:
            embedding.append(0.0)

        return embedding[:self.dimension]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for batch of texts."""
        return [self.embed_text(text) for text in texts]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, api_key: str = "", model: str = "text-embedding-3-small", **kwargs):
        self.api_key = api_key
        self.model = model
        self._mock = MockEmbeddingProvider()

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI API."""
        if not self.api_key:
            logger.warning("OpenAI API key not provided, using mock embeddings")
            return self._mock.embed_text(text)

        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)

            response = client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )

            return response.data[0].embedding

        except ImportError:
            logger.warning("OpenAI library not available, using mock embeddings")
            return self._mock.embed_text(text)
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return self._mock.embed_text(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts using OpenAI API."""
        if not self.api_key:
            logger.warning("OpenAI API key not provided, using mock embeddings")
            return self._mock.embed_batch(texts)

        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)

            response = client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )

            return [item.embedding for item in response.data]

        except ImportError:
            logger.warning("OpenAI library not available, using mock embeddings")
            return self._mock.embed_batch(texts)
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            return self._mock.embed_batch(texts)


class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """Cohere embedding provider."""

    def __init__(self, api_key: str = "", model: str = "embed-english-light-v3.0", **kwargs):
        self.api_key = api_key
        self.model = model
        self._mock = MockEmbeddingProvider()

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using Cohere API."""
        if not self.api_key:
            logger.warning("Cohere API key not provided, using mock embeddings")
            return self._mock.embed_text(text)

        try:
            import cohere

            client = cohere.Client(self.api_key)

            response = client.embed(
                texts=[text],
                model=self.model,
                input_type="search_document"
            )

            return response.embeddings[0]

        except ImportError:
            logger.warning("Cohere library not available, using mock embeddings")
            return self._mock.embed_text(text)
        except Exception as e:
            logger.error(f"Cohere embedding failed: {e}")
            return self._mock.embed_text(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts using Cohere API."""
        if not self.api_key:
            logger.warning("Cohere API key not provided, using mock embeddings")
            return self._mock.embed_batch(texts)

        try:
            import cohere

            client = cohere.Client(self.api_key)

            response = client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document"
            )

            return response.embeddings

        except ImportError:
            logger.warning("Cohere library not available, using mock embeddings")
            return self._mock.embed_batch(texts)
        except Exception as e:
            logger.error(f"Cohere batch embedding failed: {e}")
            return self._mock.embed_batch(texts)


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        self.model_name = model_name
        self._model = None
        self._mock = MockEmbeddingProvider()

    def _load_model(self):
        """Load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded HuggingFace model: {self.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not available, using mock embeddings")
                self._model = False
            except Exception as e:
                logger.error(f"Failed to load HuggingFace model: {e}")
                self._model = False

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using HuggingFace model."""
        self._load_model()

        if self._model is False:
            return self._mock.embed_text(text)

        try:
            embedding = self._model.encode([text])[0]
            return embedding.tolist()

        except Exception as e:
            logger.error(f"HuggingFace embedding failed: {e}")
            return self._mock.embed_text(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts using HuggingFace model."""
        self._load_model()

        if self._model is False:
            return self._mock.embed_batch(texts)

        try:
            embeddings = self._model.encode(texts)
            return [embedding.tolist() for embedding in embeddings]

        except Exception as e:
            logger.error(f"HuggingFace batch embedding failed: {e}")
            return self._mock.embed_batch(texts)


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local embedding provider using simple methods."""

    def __init__(self, method: str = "tfidf", **kwargs):
        self.method = method
        self._vocabulary = {}
        self._idf_scores = {}

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using local methods."""
        if self.method == "tfidf":
            return self._tfidf_embedding(text)
        elif self.method == "bag_of_words":
            return self._bow_embedding(text)
        else:
            # Fall back to mock
            mock = MockEmbeddingProvider()
            return mock.embed_text(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        if self.method == "tfidf":
            return self._tfidf_batch_embedding(texts)
        else:
            return [self.embed_text(text) for text in texts]

    def _tfidf_embedding(self, text: str, dimension: int = 100) -> List[float]:
        """Generate TF-IDF based embedding."""
        words = text.lower().split()
        word_freq = {}

        # Count word frequencies
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Simple TF-IDF calculation (without corpus IDF)
        embedding = [0.0] * dimension

        for i, word in enumerate(list(word_freq.keys())[:dimension]):
            tf = word_freq[word] / len(words) if words else 0
            # Use position-based hash for consistent placement
            pos = hash(word) % dimension
            embedding[pos] = tf

        return embedding

    def _bow_embedding(self, text: str, dimension: int = 100) -> List[float]:
        """Generate bag-of-words embedding."""
        words = text.lower().split()
        embedding = [0.0] * dimension

        for word in words:
            # Use hash to determine position
            pos = hash(word) % dimension
            embedding[pos] += 1.0

        # Normalize
        total = sum(embedding)
        if total > 0:
            embedding = [x / total for x in embedding]

        return embedding

    def _tfidf_batch_embedding(self, texts: List[str]) -> List[List[float]]:
        """Generate TF-IDF embeddings for batch with proper IDF calculation."""
        # Build vocabulary and document frequencies
        vocabulary = set()
        doc_word_sets = []

        for text in texts:
            words = set(text.lower().split())
            vocabulary.update(words)
            doc_word_sets.append(words)

        vocab_list = list(vocabulary)
        n_docs = len(texts)

        # Calculate IDF scores
        idf_scores = {}
        for word in vocab_list:
            doc_freq = sum(1 for word_set in doc_word_sets if word in word_set)
            idf_scores[word] = np.log(n_docs / (doc_freq + 1))

        # Generate embeddings
        embeddings = []
        dimension = min(100, len(vocab_list))

        for text in texts:
            words = text.lower().split()
            word_freq = {}

            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

            embedding = [0.0] * dimension

            for word, freq in word_freq.items():
                if word in vocab_list:
                    tf = freq / len(words) if words else 0
                    idf = idf_scores.get(word, 1.0)
                    tfidf = tf * idf

                    # Use consistent positioning
                    pos = vocab_list.index(word) % dimension
                    embedding[pos] = max(embedding[pos], tfidf)

            embeddings.append(embedding)

        return embeddings


__all__ = [
    "BaseEmbeddingProvider",
    "MockEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "CohereEmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
    "LocalEmbeddingProvider",
]