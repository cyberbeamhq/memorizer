"""
Built-in Retriever Components
Provides default retrieval implementations for the Memorizer framework.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime

from ..core.interfaces import Memory, Query, RetrievalResult

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for memory retrievers."""

    @abstractmethod
    def retrieve_memories(
        self,
        query: Query,
        limit: int = 10,
        storage=None
    ) -> RetrievalResult:
        """Retrieve memories based on query."""
        pass

    def retrieve(self, query: Query, storage=None) -> RetrievalResult:
        """Alias for retrieve_memories for compatibility."""
        return self.retrieve_memories(query, query.limit, storage)

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the retriever."""
        return {"status": "healthy", "type": self.__class__.__name__}


class KeywordRetriever(BaseRetriever):
    """Simple keyword-based retrieval using TF-IDF scoring."""

    def __init__(self, min_score: float = 0.1, **kwargs):
        self.min_score = min_score

    def retrieve_memories(
        self,
        query: Query,
        limit: int = 10,
        storage=None
    ) -> RetrievalResult:
        """Retrieve memories using keyword matching."""
        start_time = datetime.now()

        if not storage:
            return RetrievalResult(
                memories=[],
                total_found=0,
                retrieval_time=0.0,
                source="keyword",
                metadata={"error": "No storage provided"}
            )

        try:
            # Get all memories for the user
            all_memories = storage.search_memories(
                user_id=query.user_id,
                query="",  # Get all memories
                limit=1000,  # Large limit to get all
                metadata_filters=query.metadata
            )

            if not query.content.strip():
                # If no query content, return recent memories
                recent_memories = sorted(all_memories, key=lambda m: m.created_at, reverse=True)[:limit]
                retrieval_time = (datetime.now() - start_time).total_seconds()
                return RetrievalResult(
                    memories=recent_memories,
                    total_found=len(recent_memories),
                    retrieval_time=retrieval_time,
                    source="keyword",
                    metadata={"query_type": "recent"}
                )

            # Score memories based on keyword matching
            scored_memories = []
            query_terms = self._extract_terms(query.content)

            for memory in all_memories:
                score = self._calculate_keyword_score(query_terms, memory)
                if score >= self.min_score:
                    scored_memories.append((memory, score))

            # Sort by score and limit results
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            top_memories = [memory for memory, _ in scored_memories[:limit]]

            retrieval_time = (datetime.now() - start_time).total_seconds()

            return RetrievalResult(
                memories=top_memories,
                total_found=len(scored_memories),
                retrieval_time=retrieval_time,
                source="keyword",
                metadata={
                    "query_terms": query_terms,
                    "scores": [score for _, score in scored_memories[:limit]]
                }
            )

        except Exception as e:
            logger.error(f"Keyword retrieval failed: {e}")
            retrieval_time = (datetime.now() - start_time).total_seconds()
            return RetrievalResult(
                memories=[],
                total_found=0,
                retrieval_time=retrieval_time,
                source="keyword",
                metadata={"error": str(e)}
            )

    def _extract_terms(self, text: str) -> List[str]:
        """Extract terms from text."""
        # Simple tokenization - split on whitespace and punctuation
        terms = re.findall(r'\b\w+\b', text.lower())
        return [term for term in terms if len(term) > 2]  # Filter short terms

    def _calculate_keyword_score(self, query_terms: List[str], memory: Memory) -> float:
        """Calculate keyword-based score for a memory."""
        if not query_terms:
            return 0.0

        # Extract terms from memory content and metadata
        memory_text = memory.content.lower()
        if memory.metadata:
            # Include string values from metadata
            metadata_text = " ".join([
                str(v) for v in memory.metadata.values()
                if isinstance(v, str)
            ]).lower()
            memory_text += " " + metadata_text

        memory_terms = self._extract_terms(memory_text)
        memory_term_counts = Counter(memory_terms)

        # Calculate simple term frequency score
        score = 0.0
        total_query_terms = len(query_terms)

        for term in query_terms:
            if term in memory_term_counts:
                # Simple TF score with some normalization
                tf = memory_term_counts[term] / len(memory_terms) if memory_terms else 0
                score += tf

        # Normalize by query length
        return score / total_query_terms if total_query_terms > 0 else 0.0


class VectorRetriever(BaseRetriever):
    """Vector-based retrieval using embeddings (mock implementation)."""

    def __init__(self, similarity_threshold: float = 0.7, **kwargs):
        self.similarity_threshold = similarity_threshold

    def retrieve_memories(
        self,
        query: Query,
        limit: int = 10,
        storage=None
    ) -> RetrievalResult:
        """Retrieve memories using vector similarity (mock implementation)."""
        start_time = datetime.now()

        if not storage:
            return RetrievalResult(
                memories=[],
                total_found=0,
                retrieval_time=0.0,
                source="vector",
                metadata={"error": "No storage provided"}
            )

        try:
            # For now, fall back to keyword retrieval
            # In a real implementation, this would use vector embeddings
            logger.info("Vector retrieval not fully implemented, falling back to keyword search")

            # Get memories using simple content search
            memories = storage.search_memories(
                user_id=query.user_id,
                query=query.content,
                limit=limit,
                metadata_filters=query.metadata
            )

            retrieval_time = (datetime.now() - start_time).total_seconds()

            return RetrievalResult(
                memories=memories,
                total_found=len(memories),
                retrieval_time=retrieval_time,
                source="vector",
                metadata={
                    "note": "Vector retrieval not fully implemented",
                    "fallback": "keyword_search"
                }
            )

        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            retrieval_time = (datetime.now() - start_time).total_seconds()
            return RetrievalResult(
                memories=[],
                total_found=0,
                retrieval_time=retrieval_time,
                source="vector",
                metadata={"error": str(e)}
            )


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval combining keyword and vector approaches."""

    def __init__(
        self,
        keyword_weight: float = 0.4,
        vector_weight: float = 0.6,
        **kwargs
    ):
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.keyword_retriever = KeywordRetriever(**kwargs)
        self.vector_retriever = VectorRetriever(**kwargs)

    def retrieve_memories(
        self,
        query: Query,
        limit: int = 10,
        storage=None
    ) -> RetrievalResult:
        """Retrieve memories using hybrid approach."""
        start_time = datetime.now()

        if not storage:
            return RetrievalResult(
                memories=[],
                total_found=0,
                retrieval_time=0.0,
                source="hybrid",
                metadata={"error": "No storage provided"}
            )

        try:
            # Get results from both retrievers
            keyword_results = self.keyword_retriever.retrieve_memories(
                query, limit * 2, storage
            )
            vector_results = self.vector_retriever.retrieve_memories(
                query, limit * 2, storage
            )

            # Combine and score results
            memory_scores = {}

            # Add keyword scores
            for i, memory in enumerate(keyword_results.memories):
                # Give higher scores to higher-ranked memories
                base_score = (len(keyword_results.memories) - i) / len(keyword_results.memories)
                memory_scores[memory.id] = {
                    "memory": memory,
                    "keyword_score": base_score * self.keyword_weight,
                    "vector_score": 0.0
                }

            # Add vector scores
            for i, memory in enumerate(vector_results.memories):
                base_score = (len(vector_results.memories) - i) / len(vector_results.memories)
                if memory.id in memory_scores:
                    memory_scores[memory.id]["vector_score"] = base_score * self.vector_weight
                else:
                    memory_scores[memory.id] = {
                        "memory": memory,
                        "keyword_score": 0.0,
                        "vector_score": base_score * self.vector_weight
                    }

            # Calculate combined scores and sort
            scored_memories = []
            for memory_id, scores in memory_scores.items():
                combined_score = scores["keyword_score"] + scores["vector_score"]
                scored_memories.append((scores["memory"], combined_score))

            # Sort by combined score and limit
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            top_memories = [memory for memory, _ in scored_memories[:limit]]

            retrieval_time = (datetime.now() - start_time).total_seconds()

            return RetrievalResult(
                memories=top_memories,
                total_found=len(scored_memories),
                retrieval_time=retrieval_time,
                source="hybrid",
                metadata={
                    "keyword_weight": self.keyword_weight,
                    "vector_weight": self.vector_weight,
                    "keyword_results": len(keyword_results.memories),
                    "vector_results": len(vector_results.memories),
                    "combined_scores": [score for _, score in scored_memories[:limit]]
                }
            )

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            retrieval_time = (datetime.now() - start_time).total_seconds()
            return RetrievalResult(
                memories=[],
                total_found=0,
                retrieval_time=retrieval_time,
                source="hybrid",
                metadata={"error": str(e)}
            )


class SemanticRetriever(BaseRetriever):
    """Semantic retrieval using simple heuristics (mock implementation)."""

    def __init__(self, semantic_boost: float = 1.5, **kwargs):
        self.semantic_boost = semantic_boost
        self.keyword_retriever = KeywordRetriever(**kwargs)

    def retrieve_memories(
        self,
        query: Query,
        limit: int = 10,
        storage=None
    ) -> RetrievalResult:
        """Retrieve memories using semantic understanding (mock)."""
        start_time = datetime.now()

        if not storage:
            return RetrievalResult(
                memories=[],
                total_found=0,
                retrieval_time=0.0,
                source="semantic",
                metadata={"error": "No storage provided"}
            )

        try:
            # For now, use keyword retrieval with semantic boosting
            # In a real implementation, this would use NLP models
            keyword_results = self.keyword_retriever.retrieve_memories(
                query, limit * 2, storage
            )

            # Apply semantic boosting based on simple heuristics
            boosted_memories = []
            query_lower = query.content.lower()

            for memory in keyword_results.memories:
                base_score = 1.0
                memory_lower = memory.content.lower()

                # Boost if similar parts of speech or question types
                if self._has_similar_structure(query_lower, memory_lower):
                    base_score *= self.semantic_boost

                # Boost if contains similar entities (very simple)
                if self._has_similar_entities(query_lower, memory_lower):
                    base_score *= 1.2

                boosted_memories.append((memory, base_score))

            # Sort by boosted score and limit
            boosted_memories.sort(key=lambda x: x[1], reverse=True)
            top_memories = [memory for memory, _ in boosted_memories[:limit]]

            retrieval_time = (datetime.now() - start_time).total_seconds()

            return RetrievalResult(
                memories=top_memories,
                total_found=len(boosted_memories),
                retrieval_time=retrieval_time,
                source="semantic",
                metadata={
                    "semantic_boost": self.semantic_boost,
                    "boosted_scores": [score for _, score in boosted_memories[:limit]]
                }
            )

        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            retrieval_time = (datetime.now() - start_time).total_seconds()
            return RetrievalResult(
                memories=[],
                total_found=0,
                retrieval_time=retrieval_time,
                source="semantic",
                metadata={"error": str(e)}
            )

    def _has_similar_structure(self, query: str, memory: str) -> bool:
        """Check if query and memory have similar grammatical structure."""
        # Very simple heuristics
        query_words = query.split()
        memory_words = memory.split()

        # Check for question words
        question_words = {"what", "how", "when", "where", "why", "who", "which"}
        query_has_question = any(word in question_words for word in query_words)
        memory_has_question = any(word in question_words for word in memory_words)

        return query_has_question == memory_has_question

    def _has_similar_entities(self, query: str, memory: str) -> bool:
        """Check if query and memory reference similar entities."""
        # Very simple entity detection - look for capitalized words
        import re

        query_entities = set(re.findall(r'\b[A-Z][a-z]+\b', query))
        memory_entities = set(re.findall(r'\b[A-Z][a-z]+\b', memory))

        if not query_entities or not memory_entities:
            return False

        # Check for overlap
        overlap = len(query_entities.intersection(memory_entities))
        return overlap > 0


__all__ = [
    "BaseRetriever",
    "KeywordRetriever",
    "VectorRetriever",
    "HybridRetriever",
    "SemanticRetriever",
]