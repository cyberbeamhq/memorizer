"""
Built-in Filter Components
Provides default filtering implementations for the Memorizer framework.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseFilter(ABC):
    """Abstract base class for content filters."""

    @abstractmethod
    def filter_content(self, content: str) -> str:
        """Filter content and return cleaned version."""
        pass

    @abstractmethod
    def detect_sensitive_data(self, content: str) -> List[Dict[str, Any]]:
        """Detect sensitive data in content."""
        pass

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the filter."""
        return {"status": "healthy", "type": self.__class__.__name__}


class BasicPIIFilter(BaseFilter):
    """Basic PII (Personally Identifiable Information) filter."""

    def __init__(self, **kwargs):
        # Common patterns for PII detection
        self.patterns = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            "ssn": re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            "credit_card": re.compile(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b'),
            "ip_address": re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
        }

    def filter_content(self, content: str) -> str:
        """Filter PII from content."""
        filtered_content = content

        # Replace emails
        filtered_content = self.patterns["email"].sub("[EMAIL]", filtered_content)

        # Replace phone numbers
        filtered_content = self.patterns["phone"].sub("[PHONE]", filtered_content)

        # Replace SSNs
        filtered_content = self.patterns["ssn"].sub("[SSN]", filtered_content)

        # Replace credit card numbers
        filtered_content = self.patterns["credit_card"].sub("[CREDIT_CARD]", filtered_content)

        # Replace IP addresses
        filtered_content = self.patterns["ip_address"].sub("[IP_ADDRESS]", filtered_content)

        return filtered_content

    def detect_sensitive_data(self, content: str) -> List[Dict[str, Any]]:
        """Detect sensitive data in content."""
        detections = []

        for pii_type, pattern in self.patterns.items():
            matches = pattern.finditer(content)
            for match in matches:
                detections.append({
                    "type": pii_type,
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9  # High confidence for regex matches
                })

        return detections


class AdvancedPIIFilter(BaseFilter):
    """Advanced PII filter with additional patterns."""

    def __init__(self, **kwargs):
        self.basic_filter = BasicPIIFilter()

        # Additional patterns
        self.additional_patterns = {
            "passport": re.compile(r'\b[A-Z]{1,2}[0-9]{6,9}\b'),
            "license_plate": re.compile(r'\b[A-Z0-9]{3,8}\b'),
            "api_key": re.compile(r'\b[A-Za-z0-9]{32,}\b'),
            "bitcoin_address": re.compile(r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b'),
        }

        # Combine patterns
        self.all_patterns = {**self.basic_filter.patterns, **self.additional_patterns}

    def filter_content(self, content: str) -> str:
        """Filter PII and additional sensitive content."""
        # Start with basic filtering
        filtered_content = self.basic_filter.filter_content(content)

        # Apply additional filters
        filtered_content = self.additional_patterns["passport"].sub("[PASSPORT]", filtered_content)
        filtered_content = self.additional_patterns["license_plate"].sub("[LICENSE_PLATE]", filtered_content)
        filtered_content = self.additional_patterns["api_key"].sub("[API_KEY]", filtered_content)
        filtered_content = self.additional_patterns["bitcoin_address"].sub("[BITCOIN_ADDRESS]", filtered_content)

        return filtered_content

    def detect_sensitive_data(self, content: str) -> List[Dict[str, Any]]:
        """Detect sensitive data including advanced patterns."""
        detections = []

        for pii_type, pattern in self.all_patterns.items():
            matches = pattern.finditer(content)
            for match in matches:
                # Adjust confidence based on pattern type
                confidence = 0.9 if pii_type in self.basic_filter.patterns else 0.7

                detections.append({
                    "type": pii_type,
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": confidence
                })

        return detections


class MemorizerPIIFilter(BaseFilter):
    """Memorizer's comprehensive PII filter."""

    def __init__(self, sensitivity_level: str = "medium", **kwargs):
        self.sensitivity_level = sensitivity_level
        self.basic_filter = BasicPIIFilter()
        self.advanced_filter = AdvancedPIIFilter()

        # Additional context-aware patterns
        self.context_patterns = {
            "name": re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),
            "address": re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b'),
            "date_of_birth": re.compile(r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}\b'),
        }

    def filter_content(self, content: str) -> str:
        """Filter content based on sensitivity level."""
        if self.sensitivity_level == "low":
            return self.basic_filter.filter_content(content)
        elif self.sensitivity_level == "high":
            filtered_content = self.advanced_filter.filter_content(content)
            return self._apply_context_filters(filtered_content)
        else:  # medium
            return self.advanced_filter.filter_content(content)

    def detect_sensitive_data(self, content: str) -> List[Dict[str, Any]]:
        """Detect sensitive data based on sensitivity level."""
        if self.sensitivity_level == "low":
            return self.basic_filter.detect_sensitive_data(content)
        elif self.sensitivity_level == "high":
            detections = self.advanced_filter.detect_sensitive_data(content)
            detections.extend(self._detect_context_sensitive(content))
            return detections
        else:  # medium
            return self.advanced_filter.detect_sensitive_data(content)

    def _apply_context_filters(self, content: str) -> str:
        """Apply context-aware filters."""
        filtered_content = content

        # Filter potential names (with lower confidence)
        filtered_content = self.context_patterns["name"].sub("[NAME]", filtered_content)

        # Filter addresses
        filtered_content = self.context_patterns["address"].sub("[ADDRESS]", filtered_content)

        # Filter dates that might be DOB
        filtered_content = self.context_patterns["date_of_birth"].sub("[DATE]", filtered_content)

        return filtered_content

    def _detect_context_sensitive(self, content: str) -> List[Dict[str, Any]]:
        """Detect context-sensitive PII."""
        detections = []

        for pii_type, pattern in self.context_patterns.items():
            matches = pattern.finditer(content)
            for match in matches:
                detections.append({
                    "type": pii_type,
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.6  # Lower confidence for context-based detection
                })

        return detections


class ProfanityFilter(BaseFilter):
    """Simple profanity filter."""

    def __init__(self, **kwargs):
        # Basic profanity list (expand as needed)
        self.profanity_words = {
            "damn", "hell", "shit", "fuck", "bitch", "ass", "crap"
        }

    def filter_content(self, content: str) -> str:
        """Filter profanity from content."""
        words = content.split()
        filtered_words = []

        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in self.profanity_words:
                filtered_words.append("***")
            else:
                filtered_words.append(word)

        return " ".join(filtered_words)

    def detect_sensitive_data(self, content: str) -> List[Dict[str, Any]]:
        """Detect profanity in content."""
        detections = []
        words = content.split()
        position = 0

        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in self.profanity_words:
                detections.append({
                    "type": "profanity",
                    "text": word,
                    "start": position,
                    "end": position + len(word),
                    "confidence": 0.8
                })
            position += len(word) + 1  # +1 for space

        return detections


class NoOpFilter(BaseFilter):
    """No-operation filter that doesn't modify content."""

    def filter_content(self, content: str) -> str:
        """Return content unchanged."""
        return content

    def detect_sensitive_data(self, content: str) -> List[Dict[str, Any]]:
        """Return no detections."""
        return []


__all__ = [
    "BaseFilter",
    "BasicPIIFilter",
    "AdvancedPIIFilter",
    "MemorizerPIIFilter",
    "ProfanityFilter",
    "NoOpFilter",
]