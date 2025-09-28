"""
Built-in Summarizer Components
Provides default summarization implementations for the Memorizer framework.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseSummarizer(ABC):
    """Abstract base class for summarizers."""

    @abstractmethod
    def summarize(self, content: str, max_length: int = 500) -> str:
        """Summarize the given content."""
        pass

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the summarizer."""
        return {"status": "healthy", "type": self.__class__.__name__}


class MockSummarizer(BaseSummarizer):
    """Mock summarizer for testing and development."""

    def __init__(self, **kwargs):
        pass

    def summarize(self, content: str, max_length: int = 500) -> str:
        """Create a simple extractive summary."""
        if not content.strip():
            return ""

        # Simple extractive summarization
        sentences = self._split_sentences(content)

        if len(sentences) <= 1:
            return content[:max_length]

        # Take first and last sentences if content is long
        if len(content) > max_length * 2:
            summary_parts = []

            # Always include first sentence
            summary_parts.append(sentences[0])

            # Add middle or last sentence if space allows
            if len(sentences) > 2:
                middle_idx = len(sentences) // 2
                summary_parts.append(sentences[middle_idx])

            # Add last sentence if different from first
            if len(sentences) > 1 and sentences[-1] != sentences[0]:
                summary_parts.append(sentences[-1])

            summary = " ".join(summary_parts)
        else:
            summary = content

        # Truncate if still too long
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(' ', 1)[0] + "..."

        return summary

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]


class OpenAISummarizer(BaseSummarizer):
    """OpenAI-based summarizer."""

    def __init__(self, api_key: str = "", model: str = "gpt-4o-mini", **kwargs):
        self.api_key = api_key
        self.model = model
        self._mock = MockSummarizer()

    def summarize(self, content: str, max_length: int = 500) -> str:
        """Summarize using OpenAI API."""
        if not self.api_key:
            logger.warning("OpenAI API key not provided, using mock summarizer")
            return self._mock.summarize(content, max_length)

        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)

            prompt = f"""Please provide a concise summary of the following text in approximately {max_length} characters or less:

{content}

Summary:"""

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length // 3,  # Rough estimate for token count
                temperature=0.3
            )

            summary = response.choices[0].message.content.strip()

            # Ensure it's not longer than max_length
            if len(summary) > max_length:
                summary = summary[:max_length].rsplit(' ', 1)[0] + "..."

            return summary

        except ImportError:
            logger.warning("OpenAI library not available, using mock summarizer")
            return self._mock.summarize(content, max_length)
        except Exception as e:
            logger.error(f"OpenAI summarization failed: {e}")
            return self._mock.summarize(content, max_length)


class AnthropicSummarizer(BaseSummarizer):
    """Anthropic Claude-based summarizer."""

    def __init__(self, api_key: str = "", model: str = "claude-3-haiku-20240307", **kwargs):
        self.api_key = api_key
        self.model = model
        self._mock = MockSummarizer()

    def summarize(self, content: str, max_length: int = 500) -> str:
        """Summarize using Anthropic API."""
        if not self.api_key:
            logger.warning("Anthropic API key not provided, using mock summarizer")
            return self._mock.summarize(content, max_length)

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            prompt = f"""Please provide a concise summary of the following text in approximately {max_length} characters or less:

{content}

Summary:"""

            response = client.messages.create(
                model=self.model,
                max_tokens=max_length // 3,  # Rough estimate
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            summary = response.content[0].text.strip()

            # Ensure it's not longer than max_length
            if len(summary) > max_length:
                summary = summary[:max_length].rsplit(' ', 1)[0] + "..."

            return summary

        except ImportError:
            logger.warning("Anthropic library not available, using mock summarizer")
            return self._mock.summarize(content, max_length)
        except Exception as e:
            logger.error(f"Anthropic summarization failed: {e}")
            return self._mock.summarize(content, max_length)


class GroqSummarizer(BaseSummarizer):
    """Groq-based summarizer."""

    def __init__(self, api_key: str = "", model: str = "llama3-8b-8192", **kwargs):
        self.api_key = api_key
        self.model = model
        self._mock = MockSummarizer()

    def summarize(self, content: str, max_length: int = 500) -> str:
        """Summarize using Groq API."""
        if not self.api_key:
            logger.warning("Groq API key not provided, using mock summarizer")
            return self._mock.summarize(content, max_length)

        try:
            import groq

            client = groq.Groq(api_key=self.api_key)

            prompt = f"""Please provide a concise summary of the following text in approximately {max_length} characters or less:

{content}

Summary:"""

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length // 3,  # Rough estimate
                temperature=0.3
            )

            summary = response.choices[0].message.content.strip()

            # Ensure it's not longer than max_length
            if len(summary) > max_length:
                summary = summary[:max_length].rsplit(' ', 1)[0] + "..."

            return summary

        except ImportError:
            logger.warning("Groq library not available, using mock summarizer")
            return self._mock.summarize(content, max_length)
        except Exception as e:
            logger.error(f"Groq summarization failed: {e}")
            return self._mock.summarize(content, max_length)


class ExtractiveSummarizer(BaseSummarizer):
    """Extractive summarizer using simple heuristics."""

    def __init__(self, sentence_count: int = 3, **kwargs):
        self.sentence_count = sentence_count

    def summarize(self, content: str, max_length: int = 500) -> str:
        """Create extractive summary by selecting key sentences."""
        if not content.strip():
            return ""

        sentences = self._split_sentences(content)

        if len(sentences) <= self.sentence_count:
            summary = content
        else:
            # Score sentences based on simple heuristics
            scored_sentences = []

            for i, sentence in enumerate(sentences):
                score = self._score_sentence(sentence, sentences, i)
                scored_sentences.append((sentence, score, i))

            # Sort by score and take top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = scored_sentences[:self.sentence_count]

            # Sort selected sentences by original order
            top_sentences.sort(key=lambda x: x[2])

            summary = " ".join([sentence for sentence, _, _ in top_sentences])

        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(' ', 1)[0] + "..."

        return summary

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _score_sentence(self, sentence: str, all_sentences: List[str], position: int) -> float:
        """Score a sentence based on various factors."""
        score = 0.0

        # Length factor - prefer medium-length sentences
        length = len(sentence.split())
        if 5 <= length <= 25:
            score += 1.0
        elif length > 25:
            score += 0.5

        # Position factor - prefer first and last sentences
        total_sentences = len(all_sentences)
        if position == 0 or position == total_sentences - 1:
            score += 1.5
        elif position < total_sentences * 0.3:  # First third
            score += 1.0

        # Keyword density - prefer sentences with many meaningful words
        words = sentence.lower().split()
        meaningful_words = [w for w in words if len(w) > 3 and w.isalpha()]
        if meaningful_words:
            score += len(meaningful_words) / len(words)

        # Capital letters - may indicate important proper nouns
        capital_count = sum(1 for c in sentence if c.isupper())
        if capital_count > 1:
            score += 0.5

        return score


class AbstractiveSummarizer(BaseSummarizer):
    """Abstractive summarizer using simple text manipulation."""

    def __init__(self, compression_ratio: float = 0.3, **kwargs):
        self.compression_ratio = compression_ratio

    def summarize(self, content: str, max_length: int = 500) -> str:
        """Create abstractive summary by rewriting content."""
        if not content.strip():
            return ""

        # Simple abstractive approach - extract key phrases and rewrite
        sentences = self._split_sentences(content)
        key_phrases = self._extract_key_phrases(content)

        if not key_phrases:
            # Fall back to extractive if no key phrases found
            extractor = ExtractiveSummarizer()
            return extractor.summarize(content, max_length)

        # Create summary from key phrases
        summary_parts = []

        # Add most important phrases
        for phrase in key_phrases[:5]:  # Top 5 phrases
            summary_parts.append(phrase)

        # Combine into coherent text
        if len(summary_parts) > 1:
            summary = "Key points: " + "; ".join(summary_parts) + "."
        else:
            summary = summary_parts[0] if summary_parts else content[:max_length]

        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(' ', 1)[0] + "..."

        return summary

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        # Simple phrase extraction - find noun phrases and important terms
        words = text.split()
        phrases = []

        # Look for phrases with important words
        important_words = set()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if len(clean_word) > 4 and clean_word.isalpha():
                important_words.add(clean_word)

        # Extract phrases containing important words
        text_lower = text.lower()
        for word in important_words:
            # Find context around important words
            word_positions = [m.start() for m in re.finditer(r'\b' + re.escape(word) + r'\b', text_lower)]
            for pos in word_positions:
                # Extract phrase around the word
                start = max(0, pos - 30)
                end = min(len(text), pos + 30)
                phrase = text[start:end].strip()

                # Clean up the phrase
                phrase = re.sub(r'^\W+|\W+$', '', phrase)
                if phrase and len(phrase) > 10:
                    phrases.append(phrase)

        # Remove duplicates and sort by length (prefer longer phrases)
        unique_phrases = list(set(phrases))
        unique_phrases.sort(key=len, reverse=True)

        return unique_phrases[:10]  # Return top 10 phrases


__all__ = [
    "BaseSummarizer",
    "MockSummarizer",
    "OpenAISummarizer",
    "AnthropicSummarizer",
    "GroqSummarizer",
    "ExtractiveSummarizer",
    "AbstractiveSummarizer",
]