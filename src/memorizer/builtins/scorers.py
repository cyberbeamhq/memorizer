"""
Built-in Scorer Components
Provides default scoring implementations for the Memorizer framework.
"""

import logging
import math
from typing import Any, Dict, List, Tuple
from abc import ABC, abstractmethod
from collections import Counter
import re

logger = logging.getLogger(__name__)


class BaseScorer(ABC):
    """Abstract base class for memory scorers."""

    @abstractmethod
    def score_memory(self, query: str, memory_content: str, metadata: Dict[str, Any] = None) -> float:
        """Score a memory's relevance to a query."""
        pass

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the scorer."""
        return {"status": "healthy", "type": self.__class__.__name__}


class TFIDFScorer(BaseScorer):
    """TF-IDF based scorer."""

    def __init__(self, **kwargs):
        self.term_frequencies = {}
        self.document_frequencies = {}
        self.total_documents = 0

    def score_memory(self, query: str, memory_content: str, metadata: Dict[str, Any] = None) -> float:
        """Score memory using TF-IDF."""
        if not query.strip() or not memory_content.strip():
            return 0.0

        query_terms = self._extract_terms(query)
        memory_terms = self._extract_terms(memory_content)

        if not query_terms or not memory_terms:
            return 0.0

        # Calculate TF-IDF score
        score = 0.0
        memory_term_counts = Counter(memory_terms)
        total_memory_terms = len(memory_terms)

        for term in query_terms:
            if term in memory_term_counts:
                # Term frequency
                tf = memory_term_counts[term] / total_memory_terms

                # Inverse document frequency (simplified)
                # In a real implementation, this would use corpus statistics
                idf = math.log(100 / (1 + memory_term_counts[term]))

                score += tf * idf

        # Normalize by query length
        return score / len(query_terms)

    def _extract_terms(self, text: str) -> List[str]:
        """Extract terms from text."""
        # Simple tokenization
        terms = re.findall(r'\b\w+\b', text.lower())
        return [term for term in terms if len(term) > 2]


class BM25Scorer(BaseScorer):
    """BM25 scoring algorithm."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, **kwargs):
        self.k1 = k1
        self.b = b
        self.avg_doc_length = 100  # Assumed average document length

    def score_memory(self, query: str, memory_content: str, metadata: Dict[str, Any] = None) -> float:
        """Score memory using BM25."""
        if not query.strip() or not memory_content.strip():
            return 0.0

        query_terms = self._extract_terms(query)
        memory_terms = self._extract_terms(memory_content)

        if not query_terms or not memory_terms:
            return 0.0

        memory_term_counts = Counter(memory_terms)
        doc_length = len(memory_terms)

        score = 0.0

        for term in query_terms:
            if term in memory_term_counts:
                term_freq = memory_term_counts[term]

                # BM25 formula
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

                score += numerator / denominator

        return score

    def _extract_terms(self, text: str) -> List[str]:
        """Extract terms from text."""
        terms = re.findall(r'\b\w+\b', text.lower())
        return [term for term in terms if len(term) > 2]


class SemanticScorer(BaseScorer):
    """Semantic scoring using simple heuristics."""

    def __init__(self, **kwargs):
        pass

    def score_memory(self, query: str, memory_content: str, metadata: Dict[str, Any] = None) -> float:
        """Score memory using semantic similarity heuristics."""
        if not query.strip() or not memory_content.strip():
            return 0.0

        # Simple semantic scoring based on various factors
        score = 0.0

        # Exact phrase matching (highest weight)
        query_lower = query.lower()
        memory_lower = memory_content.lower()

        if query_lower in memory_lower:
            score += 1.0

        # Word overlap scoring
        query_words = set(self._extract_words(query_lower))
        memory_words = set(self._extract_words(memory_lower))

        if query_words and memory_words:
            overlap = len(query_words.intersection(memory_words))
            score += overlap / len(query_words)

        # Similar word structure (same length words, etc.)
        score += self._structural_similarity(query_lower, memory_lower)

        # Metadata boost
        if metadata:
            score += self._metadata_boost(query_lower, metadata)

        return score

    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text."""
        return re.findall(r'\b\w+\b', text)

    def _structural_similarity(self, query: str, memory: str) -> float:
        """Calculate structural similarity."""
        query_words = self._extract_words(query)
        memory_words = self._extract_words(memory)

        if not query_words or not memory_words:
            return 0.0

        # Average word length similarity
        avg_query_len = sum(len(word) for word in query_words) / len(query_words)
        avg_memory_len = sum(len(word) for word in memory_words) / len(memory_words)

        length_similarity = 1.0 - abs(avg_query_len - avg_memory_len) / max(avg_query_len, avg_memory_len)

        # Sentence structure similarity (very basic)
        query_sentences = len(re.split(r'[.!?]', query))
        memory_sentences = len(re.split(r'[.!?]', memory))

        if max(query_sentences, memory_sentences) > 0:
            structure_similarity = 1.0 - abs(query_sentences - memory_sentences) / max(query_sentences, memory_sentences)
        else:
            structure_similarity = 1.0

        return (length_similarity + structure_similarity) * 0.1  # Low weight

    def _metadata_boost(self, query: str, metadata: Dict[str, Any]) -> float:
        """Boost score based on metadata matches."""
        boost = 0.0

        for key, value in metadata.items():
            if isinstance(value, str) and value.lower() in query:
                boost += 0.2

        return min(boost, 0.5)  # Cap metadata boost


class CosineScorer(BaseScorer):
    """Cosine similarity scorer using term vectors."""

    def __init__(self, **kwargs):
        pass

    def score_memory(self, query: str, memory_content: str, metadata: Dict[str, Any] = None) -> float:
        """Score memory using cosine similarity."""
        if not query.strip() or not memory_content.strip():
            return 0.0

        query_vector = self._create_term_vector(query)
        memory_vector = self._create_term_vector(memory_content)

        return self._cosine_similarity(query_vector, memory_vector)

    def _create_term_vector(self, text: str) -> Dict[str, float]:
        """Create term frequency vector."""
        terms = re.findall(r'\b\w+\b', text.lower())
        terms = [term for term in terms if len(term) > 2]

        if not terms:
            return {}

        term_counts = Counter(terms)
        total_terms = len(terms)

        # Normalize by document length
        return {term: count / total_terms for term, count in term_counts.items()}

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0

        # Get common terms
        common_terms = set(vec1.keys()).intersection(set(vec2.keys()))

        if not common_terms:
            return 0.0

        # Calculate dot product
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)

        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        magnitude2 = math.sqrt(sum(val ** 2 for val in vec2.values()))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


class HybridScorer(BaseScorer):
    """Hybrid scorer combining multiple scoring methods."""

    def __init__(
        self,
        tfidf_weight: float = 0.3,
        bm25_weight: float = 0.4,
        semantic_weight: float = 0.2,
        cosine_weight: float = 0.1,
        **kwargs
    ):
        self.tfidf_weight = tfidf_weight
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.cosine_weight = cosine_weight

        # Initialize sub-scorers
        self.tfidf_scorer = TFIDFScorer()
        self.bm25_scorer = BM25Scorer()
        self.semantic_scorer = SemanticScorer()
        self.cosine_scorer = CosineScorer()

        # Normalize weights
        total_weight = self.tfidf_weight + self.bm25_weight + self.semantic_weight + self.cosine_weight
        if total_weight > 0:
            self.tfidf_weight /= total_weight
            self.bm25_weight /= total_weight
            self.semantic_weight /= total_weight
            self.cosine_weight /= total_weight

    def score_memory(self, query: str, memory_content: str, metadata: Dict[str, Any] = None) -> float:
        """Score memory using hybrid approach."""
        if not query.strip() or not memory_content.strip():
            return 0.0

        # Get scores from different methods
        tfidf_score = self.tfidf_scorer.score_memory(query, memory_content, metadata)
        bm25_score = self.bm25_scorer.score_memory(query, memory_content, metadata)
        semantic_score = self.semantic_scorer.score_memory(query, memory_content, metadata)
        cosine_score = self.cosine_scorer.score_memory(query, memory_content, metadata)

        # Combine scores with weights
        combined_score = (
            self.tfidf_weight * tfidf_score +
            self.bm25_weight * bm25_score +
            self.semantic_weight * semantic_score +
            self.cosine_weight * cosine_score
        )

        return combined_score


class SimpleScorer(BaseScorer):
    """Simple keyword-based scorer for basic matching."""

    def __init__(self, **kwargs):
        pass

    def score_memory(self, query: str, memory_content: str, metadata: Dict[str, Any] = None) -> float:
        """Score memory using simple keyword matching."""
        if not query.strip() or not memory_content.strip():
            return 0.0

        query_lower = query.lower()
        memory_lower = memory_content.lower()

        # Exact match gets highest score
        if query_lower == memory_lower:
            return 1.0

        # Substring match
        if query_lower in memory_lower:
            return 0.8

        # Word overlap
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        memory_words = set(re.findall(r'\b\w+\b', memory_lower))

        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(memory_words))
        return overlap / len(query_words)


__all__ = [
    "BaseScorer",
    "TFIDFScorer",
    "BM25Scorer",
    "SemanticScorer",
    "CosineScorer",
    "HybridScorer",
    "SimpleScorer",
]