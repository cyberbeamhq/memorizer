"""
retrieval.py
Retrieves relevant memory snippets for a given query.
Implements hybrid retrieval: keyword relevance scoring, semantic search, and vector fallback.
"""

import logging
import math
import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from ..storage import db, vector_db

logger = logging.getLogger(__name__)


# ---------------------------
# Text Processing Utilities
# ---------------------------
class TextProcessor:
    """Text processing utilities for relevance scoring."""

    def __init__(self):
        # Common English stop words that should be ignored in relevance scoring
        self.stop_words = {
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "a",
            "an",
            "the",
            "and",
            "but",
            "if",
            "or",
            "because",
            "as",
            "until",
            "while",
            "of",
            "at",
            "by",
            "for",
            "with",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "can",
            "will",
            "just",
            "don",
            "should",
            "now",
        }

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from text.
        Removes stop words and short words, handles basic stemming.
        """
        if not text:
            return []

        # Extract words, convert to lowercase
        words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())

        # Filter stop words and very short words
        meaningful_words = [
            w for w in words if w not in self.stop_words and len(w) >= 2
        ]

        # Basic stemming: remove common suffixes
        stemmed_words = []
        for word in meaningful_words:
            # Remove common suffixes
            for suffix in ["ing", "ed", "er", "est", "ly", "s"]:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    word = word[: -len(suffix)]
                    break
            stemmed_words.append(word)

        return stemmed_words

    def extract_phrases(self, text: str, max_phrase_length: int = 3) -> Set[str]:
        """Extract meaningful phrases (n-grams) from text."""
        words = self.extract_keywords(text)
        phrases = set()

        # Add individual words
        phrases.update(words)

        # Add n-grams
        for n in range(2, min(max_phrase_length + 1, len(words) + 1)):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])
                phrases.add(phrase)

        return phrases


# ---------------------------
# Relevance Scoring
# ---------------------------
class RelevanceScorer:
    """Advanced relevance scoring for memory retrieval."""

    def __init__(self):
        self.text_processor = TextProcessor()

    def tf_idf_score(
        self, query_terms: List[str], text_terms: List[str], corpus_size: int = 1000
    ) -> float:
        """
        Calculate TF-IDF based relevance score.

        Args:
            query_terms: Terms from the query
            text_terms: Terms from the text
            corpus_size: Estimated size of the corpus for IDF calculation
        """
        if not query_terms or not text_terms:
            return 0.0

        text_counter = Counter(text_terms)
        query_counter = Counter(query_terms)
        text_length = len(text_terms)

        score = 0.0
        for term in set(query_terms):
            if term in text_counter:
                # Term frequency
                tf = text_counter[term] / text_length

                # Inverse document frequency (simplified)
                # Assume rare terms are more important
                term_frequency_in_corpus = min(corpus_size // 10, len(term) * 10)
                idf = math.log(corpus_size / (1 + term_frequency_in_corpus))

                # Query term importance
                query_weight = query_counter[term] / len(query_terms)

                score += tf * idf * query_weight

        return min(score, 1.0)  # Normalize to [0, 1]

    def keyword_overlap_score(self, query: str, text: str) -> float:
        """
        Enhanced keyword overlap scoring.

        Args:
            query: Search query
            text: Text to score against

        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not text or not query:
            return 0.0

        query_terms = self.text_processor.extract_keywords(query)
        text_terms = self.text_processor.extract_keywords(text)

        if not query_terms:
            return 0.0

        # Calculate different types of matches
        exact_matches = set(query_terms).intersection(set(text_terms))

        # Phrase matching
        query_phrases = self.text_processor.extract_phrases(query, 2)
        text_phrases = self.text_processor.extract_phrases(text, 2)
        phrase_matches = query_phrases.intersection(text_phrases)

        # Scoring
        exact_score = len(exact_matches) / len(query_terms)
        phrase_bonus = len(phrase_matches) * 0.2  # Bonus for phrase matches

        # TF-IDF component
        tfidf_score = self.tf_idf_score(query_terms, text_terms)

        # Combine scores
        total_score = exact_score + phrase_bonus + (tfidf_score * 0.3)

        return min(total_score, 1.0)  # Cap at 1.0

    def semantic_distance_score(self, query: str, text: str) -> float:
        """
        Calculate semantic similarity (placeholder for future enhancement).
        Could integrate with sentence transformers or similar.
        """
        # For now, use enhanced keyword matching
        # TODO: Integrate sentence transformers for true semantic similarity
        return self.keyword_overlap_score(query, text)


# ---------------------------
# Memory Filtering and Ranking
# ---------------------------
class MemoryRanker:
    """Handles memory filtering, ranking, and selection."""

    def __init__(self, relevance_scorer: RelevanceScorer = None):
        self.scorer = relevance_scorer or RelevanceScorer()

    def calculate_recency_score(
        self, created_at: datetime, max_age_days: int = 365
    ) -> float:
        """Calculate recency score based on memory age."""
        if not created_at:
            return 0.5  # Default score for unknown dates

        age_days = (datetime.now() - created_at).days
        if age_days <= 0:
            return 1.0

        # Exponential decay with configurable max age
        decay_rate = 3.0 / max_age_days  # 95% decay over max_age_days
        return math.exp(-decay_rate * age_days)

    def calculate_tier_weight(self, tier: str) -> float:
        """Calculate importance weight based on memory tier."""
        tier_weights = {
            "very_new": 1.0,  # Most recent, full detail
            "mid_term": 0.8,  # Compressed but recent
            "long_term": 0.6,  # Highly compressed, older
            "vector_fallback": 0.4,  # Fallback results
        }
        return tier_weights.get(tier, 0.5)

    def calculate_composite_score(self, memory: Dict[str, Any], query: str) -> float:
        """
        Calculate composite relevance score combining multiple factors.

        Args:
            memory: Memory dictionary
            query: Search query

        Returns:
            Composite score between 0.0 and 1.0
        """
        # Base relevance score
        relevance = self.scorer.keyword_overlap_score(query, memory.get("content", ""))

        # Recency score
        created_at = memory.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except:
                created_at = None

        recency = self.calculate_recency_score(created_at) if created_at else 0.5

        # Tier importance
        tier_weight = self.calculate_tier_weight(memory.get("tier", "unknown"))

        # Metadata boosting (e.g., if memory was marked as important)
        metadata_boost = 1.0
        metadata = memory.get("metadata", {})
        if metadata.get("important"):
            metadata_boost = 1.2
        if metadata.get("user_feedback_positive"):
            metadata_boost *= 1.1

        # Combine scores (weighted average)
        composite = (
            relevance * 0.5  # Relevance is most important
            + recency * 0.2  # Recent memories get boost
            + tier_weight * 0.2  # Tier importance
            + (metadata_boost - 1.0) * 0.1  # Metadata bonus
        )

        return min(composite, 1.0)

    def filter_and_rank_memories(
        self,
        memories: List[Dict[str, Any]],
        query: str,
        min_score: float = 0.1,
        max_items: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Filter and rank memories by relevance.

        Args:
            memories: List of memory dictionaries
            query: Search query
            min_score: Minimum relevance score threshold
            max_items: Maximum number of items to return

        Returns:
            Filtered and ranked list of memories
        """
        scored_memories = []

        for memory in memories:
            score = self.calculate_composite_score(memory, query)
            if score >= min_score:
                memory_copy = memory.copy()
                memory_copy["relevance_score"] = score
                scored_memories.append(memory_copy)

        # Sort by relevance score (descending)
        scored_memories.sort(key=lambda m: m["relevance_score"], reverse=True)

        return scored_memories[:max_items]


# ---------------------------
# Main Retrieval Functions
# ---------------------------
class HybridRetriever:
    """Main hybrid retrieval engine."""

    def __init__(self):
        self.ranker = MemoryRanker()
        self.text_processor = TextProcessor()

    def retrieve_from_database(
        self,
        user_id: str,
        query: str,
        max_items: int = 10,
        tier_limits: Dict[str, int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and rank memories from database tiers.

        Args:
            user_id: User identifier
            query: Search query
            max_items: Maximum items to return
            tier_limits: Custom limits per tier

        Returns:
            Ranked list of relevant memories
        """
        if tier_limits is None:
            tier_limits = {"very_new": 50, "mid_term": 100, "long_term": 50}

        all_memories = []

        try:
            # First try database search if available
            search_memories = db.search_memories(user_id, query, limit=max_items * 2)
            if search_memories:
                logger.info(
                    f"Found {len(search_memories)} memories via database search"
                )
                all_memories.extend(search_memories)

            # Fetch from each tier if we need more results
            if len(all_memories) < max_items:
                for tier in ["very_new", "mid_term", "long_term"]:
                    tier_memories = db.fetch_memories(
                        user_id, tier=tier, limit=tier_limits.get(tier, 50)
                    )
                    for memory in tier_memories:
                        memory["tier"] = tier
                        all_memories.append(memory)

            logger.info(f"Retrieved {len(all_memories)} total memories for ranking")

        except Exception as e:
            logger.error(f"Error retrieving memories from database: {e}")
            return []

        # Remove duplicates (by ID)
        seen_ids = set()
        unique_memories = []
        for memory in all_memories:
            memory_id = memory.get("id")
            if memory_id not in seen_ids:
                seen_ids.add(memory_id)
                unique_memories.append(memory)

        # Filter and rank
        return self.ranker.filter_and_rank_memories(
            unique_memories, query, max_items=max_items
        )

    def retrieve_from_vector_db(
        self, user_id: str, query: str, max_items: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Fallback to vector database for semantic search.

        Args:
            user_id: User identifier
            query: Search query
            max_items: Maximum items to return

        Returns:
            List of semantically similar memories
        """
        try:
            logger.info(f"Falling back to vector database for user {user_id}")
            vector_results = vector_db.query_embeddings(user_id, query, top_k=max_items)

            # Normalize results to match database format
            normalized_results = []
            for result in vector_results:
                normalized = {
                    "id": result.get("id", "vector_unknown"),
                    "user_id": user_id,
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "tier": "vector_fallback",
                    "relevance_score": result.get("score", 0.0),
                    "created_at": result.get("created_at"),
                    "source": "vector_db",
                }
                normalized_results.append(normalized)

            logger.info(
                f"Retrieved {len(normalized_results)} results from vector database"
            )
            return normalized_results

        except Exception as e:
            logger.error(f"Vector database fallback failed: {e}")
            return []


# ---------------------------
# Main Retrieval Interface
# ---------------------------
def retrieve_context(
    user_id: str,
    query: str,
    max_items: int = 5,
    min_db_results: int = 2,
    fallback_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Main retrieval function implementing hybrid search strategy.

    Args:
        user_id: User to retrieve context for
        query: Search query
        max_items: Maximum number of items to return
        min_db_results: Minimum high-quality DB results before considering fallback
        fallback_threshold: Minimum score threshold for considering fallback

    Returns:
        List of relevant memory contexts ranked by relevance
    """
    retriever = HybridRetriever()

    # Step 1: Retrieve from database with keyword/relevance scoring
    db_results = retriever.retrieve_from_database(user_id, query, max_items)

    # Step 2: Evaluate if we need vector fallback
    high_quality_results = [
        r for r in db_results if r.get("relevance_score", 0) > fallback_threshold
    ]

    if len(high_quality_results) >= min_db_results:
        logger.info(
            f"Found {len(high_quality_results)} high-quality DB results, no fallback needed"
        )
        return db_results[:max_items]

    # Step 3: Use vector fallback for additional results
    vector_results = retriever.retrieve_from_vector_db(user_id, query, max_items)

    # Step 4: Combine and deduplicate results
    combined_results = []
    seen_ids = set()

    # Add high-scoring DB results first
    for result in db_results:
        if result["id"] not in seen_ids:
            combined_results.append(result)
            seen_ids.add(result["id"])

    # Fill remaining slots with vector results
    for result in vector_results:
        if len(combined_results) >= max_items:
            break
        if result["id"] not in seen_ids:
            combined_results.append(result)
            seen_ids.add(result["id"])

    logger.info(
        f"Combined retrieval: {len(db_results)} DB + {len(vector_results)} vector = {len(combined_results)} total"
    )
    return combined_results[:max_items]


# ---------------------------
# Convenience Functions and Legacy Support
# ---------------------------
def score_relevance(query: str, text: str) -> float:
    """
    Legacy function for backward compatibility.
    Simple relevance scoring between query and text.
    """
    scorer = RelevanceScorer()
    return scorer.keyword_overlap_score(query, text)


# ---------------------------
# Testing and Debugging
# ---------------------------
def test_retrieval_system():
    """Test the retrieval system with sample data."""
    # Mock some test data
    test_memories = [
        {
            "id": 1,
            "content": "User asked about refund policy for order",
            "tier": "very_new",
            "created_at": datetime.now(),
            "metadata": {},
        },
        {
            "id": 2,
            "content": "Customer complained about shipping delays",
            "tier": "mid_term",
            "created_at": datetime.now() - timedelta(days=5),
            "metadata": {},
        },
        {
            "id": 3,
            "content": "User praised fast delivery and good customer service",
            "tier": "long_term",
            "created_at": datetime.now() - timedelta(days=30),
            "metadata": {"important": True},
        },
    ]

    ranker = MemoryRanker()
    query = "shipping delivery problems"

    results = ranker.filter_and_rank_memories(test_memories, query)

    print("Test Results:")
    for i, result in enumerate(results):
        print(
            f"{i+1}. Score: {result.get('relevance_score', 0):.3f} - {result['content'][:50]}..."
        )


if __name__ == "__main__":
    test_retrieval_system()
