"""
pii_detection.py
PII (Personally Identifiable Information) detection and sanitization module.
Uses Microsoft Presidio for production-grade PII detection.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import presidio, fall back to basic regex if not available
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
    logger.info("Presidio PII detection available")
except ImportError:
    PRESIDIO_AVAILABLE = False
    logger.warning("Presidio not available, using fallback PII detection")
    import re


class PIIType(Enum):
    """Types of PII that can be detected."""
    EMAIL = "EMAIL_ADDRESS"
    PHONE = "PHONE_NUMBER"
    SSN = "US_SSN"
    CREDIT_CARD = "CREDIT_CARD"
    IP_ADDRESS = "IP_ADDRESS"
    NAME = "PERSON"
    ADDRESS = "LOCATION"
    DATE_OF_BIRTH = "DATE_TIME"
    PASSPORT = "US_PASSPORT"
    DRIVER_LICENSE = "US_DRIVER_LICENSE"
    IBAN = "IBAN_CODE"
    CRYPTO = "CRYPTO"
    MEDICAL_LICENSE = "MEDICAL_LICENSE"
    URL = "URL"


@dataclass
class PIIDetection:
    """Result of PII detection."""
    pii_type: str
    value: str
    start_pos: int
    end_pos: int
    confidence: float


class PIIDetector:
    """Detects and handles PII in text content using Presidio."""

    def __init__(self, language: str = "en"):
        """
        Initialize PII detector with Presidio engine.

        Args:
            language: Language code for NLP processing
        """
        self.language = language

        if PRESIDIO_AVAILABLE:
            # Initialize Presidio analyzer with NLP support
            try:
                provider = NlpEngineProvider()
                nlp_configuration = {
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": language, "model_name": "en_core_web_sm"}],
                }
                nlp_engine = provider.create_engine()
                self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
                self.anonymizer = AnonymizerEngine()
                logger.info("Presidio PII detector initialized with NLP support")
            except Exception as e:
                logger.warning(f"Failed to initialize Presidio with NLP, using basic: {e}")
                self.analyzer = AnalyzerEngine()
                self.anonymizer = AnonymizerEngine()
        else:
            # Fallback to basic regex patterns
            self.analyzer = None
            self.anonymizer = None
            self._fallback_patterns = self._compile_fallback_patterns()
            logger.info("Using fallback regex-based PII detection")

    def _compile_fallback_patterns(self) -> Dict[str, Any]:
        """Compile basic regex patterns for fallback detection."""
        return {
            "EMAIL_ADDRESS": re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                re.IGNORECASE
            ),
            "PHONE_NUMBER": re.compile(
                r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
            ),
            "US_SSN": re.compile(
                r'\b(?!000|666|9\d{2})\d{3}[-.\s]?(?!00)\d{2}[-.\s]?(?!0000)\d{4}\b'
            ),
            "CREDIT_CARD": re.compile(
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b'
            ),
            "IP_ADDRESS": re.compile(
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            ),
        }

    def detect_pii(self, text: str, entities: Optional[List[str]] = None) -> List[PIIDetection]:
        """
        Detect PII in the given text using Presidio.

        Args:
            text: Text content to analyze
            entities: Specific entity types to detect (None = all)

        Returns:
            List of detected PII instances
        """
        if not text or not text.strip():
            return []

        if PRESIDIO_AVAILABLE and self.analyzer:
            return self._detect_with_presidio(text, entities)
        else:
            return self._detect_with_fallback(text)

    def _detect_with_presidio(self, text: str, entities: Optional[List[str]] = None) -> List[PIIDetection]:
        """Use Presidio for PII detection."""
        try:
            # Analyze text for PII
            results = self.analyzer.analyze(
                text=text,
                language=self.language,
                entities=entities
            )

            # Convert to our PIIDetection format
            detections = []
            for result in results:
                detection = PIIDetection(
                    pii_type=result.entity_type,
                    value=text[result.start:result.end],
                    start_pos=result.start,
                    end_pos=result.end,
                    confidence=result.score
                )
                detections.append(detection)

            logger.debug(f"Detected {len(detections)} PII instances using Presidio")
            return detections

        except Exception as e:
            logger.error(f"Presidio PII detection failed: {e}")
            return []

    def _detect_with_fallback(self, text: str) -> List[PIIDetection]:
        """Use fallback regex detection when Presidio is unavailable."""
        detections = []

        for entity_type, pattern in self._fallback_patterns.items():
            for match in pattern.finditer(text):
                detection = PIIDetection(
                    pii_type=entity_type,
                    value=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.7  # Lower confidence for regex-only detection
                )
                detections.append(detection)

        detections.sort(key=lambda x: x.start_pos)
        logger.debug(f"Detected {len(detections)} PII instances using fallback")
        return detections

    def sanitize_text(
        self,
        text: str,
        anonymize_method: str = "replace",
        entities: Optional[List[str]] = None
    ) -> str:
        """
        Sanitize PII from text using Presidio anonymizer.

        Args:
            text: Text to sanitize
            anonymize_method: Method to use ('replace', 'mask', 'hash', 'redact')
            entities: Specific entity types to anonymize

        Returns:
            Sanitized text with PII removed/anonymized
        """
        if not text or not text.strip():
            return text

        if PRESIDIO_AVAILABLE and self.analyzer and self.anonymizer:
            return self._sanitize_with_presidio(text, anonymize_method, entities)
        else:
            return self._sanitize_with_fallback(text)

    def _sanitize_with_presidio(
        self,
        text: str,
        anonymize_method: str = "replace",
        entities: Optional[List[str]] = None
    ) -> str:
        """Use Presidio to sanitize text."""
        try:
            # First detect PII
            analyzer_results = self.analyzer.analyze(
                text=text,
                language=self.language,
                entities=entities
            )

            # Then anonymize
            operators = {}
            if anonymize_method == "replace":
                # Replace with entity type placeholder
                for result in analyzer_results:
                    operators[result.entity_type] = OperatorConfig("replace", {"new_value": f"<{result.entity_type}>"})
            elif anonymize_method == "mask":
                operators = {"DEFAULT": OperatorConfig("mask", {"chars_to_mask": 100, "masking_char": "*"})}
            elif anonymize_method == "hash":
                operators = {"DEFAULT": OperatorConfig("hash")}
            elif anonymize_method == "redact":
                operators = {"DEFAULT": OperatorConfig("redact")}

            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operators
            )

            return anonymized_result.text

        except Exception as e:
            logger.error(f"Presidio sanitization failed: {e}")
            return self._sanitize_with_fallback(text)

    def _sanitize_with_fallback(self, text: str) -> str:
        """Use fallback regex replacement for sanitization."""
        sanitized = text

        for entity_type, pattern in self._fallback_patterns.items():
            sanitized = pattern.sub(f"<{entity_type}>", sanitized)

        return sanitized

    def has_pii(self, text: str, threshold: float = 0.5) -> bool:
        """
        Check if text contains PII above confidence threshold.

        Args:
            text: Text to check
            threshold: Minimum confidence score to consider

        Returns:
            True if PII is detected above threshold
        """
        detections = self.detect_pii(text)
        return any(d.confidence >= threshold for d in detections)

    def get_pii_summary(self, text: str) -> Dict[str, int]:
        """
        Get summary of PII types found in text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary mapping PII types to counts
        """
        detections = self.detect_pii(text)
        summary = {}

        for detection in detections:
            pii_type = detection.pii_type
            summary[pii_type] = summary.get(pii_type, 0) + 1

        return summary


# Global PII detector instance
_pii_detector = None


def get_pii_detector() -> PIIDetector:
    """Get or create the global PII detector instance."""
    global _pii_detector
    if _pii_detector is None:
        _pii_detector = PIIDetector()
    return _pii_detector


def detect_pii(text: str) -> List[PIIDetection]:
    """Convenience function to detect PII in text."""
    detector = get_pii_detector()
    return detector.detect_pii(text)


def sanitize_text(text: str, method: str = "replace") -> str:
    """Convenience function to sanitize text."""
    detector = get_pii_detector()
    return detector.sanitize_text(text, anonymize_method=method)


def has_pii(text: str, threshold: float = 0.5) -> bool:
    """Convenience function to check if text has PII."""
    detector = get_pii_detector()
    return detector.has_pii(text, threshold=threshold)
