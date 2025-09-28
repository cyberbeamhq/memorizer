"""
pii_detection.py
PII (Personally Identifiable Information) detection and sanitization module.
Critical for production safety before sending data to external LLM providers.
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII that can be detected."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"


@dataclass
class PIIDetection:
    """Result of PII detection."""
    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float


class PIIDetector:
    """Detects and handles PII in text content."""
    
    def __init__(self):
        """Initialize PII detector with compiled regex patterns."""
        self.patterns = self._compile_patterns()
        self.sanitization_map = {}
        logger.info("PII detector initialized")
    
    def _compile_patterns(self) -> Dict[PIIType, re.Pattern]:
        """Compile regex patterns for PII detection."""
        patterns = {
            # Email addresses
            PIIType.EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                re.IGNORECASE
            ),
            
            # Phone numbers (US format)
            PIIType.PHONE: re.compile(
                r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
            ),
            
            # Social Security Numbers
            PIIType.SSN: re.compile(
                r'\b(?!000|666|9\d{2})\d{3}[-.\s]?(?!00)\d{2}[-.\s]?(?!0000)\d{4}\b'
            ),
            
            # Credit card numbers (basic pattern)
            PIIType.CREDIT_CARD: re.compile(
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
            ),
            
            # IP addresses
            PIIType.IP_ADDRESS: re.compile(
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            ),
            
            # Names (basic pattern - first letter capitalized words)
            PIIType.NAME: re.compile(
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
            ),
            
            # Addresses (basic pattern)
            PIIType.ADDRESS: re.compile(
                r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Place|Pl)\b',
                re.IGNORECASE
            ),
            
            # Date of birth (various formats)
            PIIType.DATE_OF_BIRTH: re.compile(
                r'\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b'
            ),
            
            # Passport numbers (basic pattern)
            PIIType.PASSPORT: re.compile(
                r'\b[A-Z]{1,2}\d{6,9}\b'
            ),
            
            # Driver's license (basic pattern)
            PIIType.DRIVER_LICENSE: re.compile(
                r'\b[A-Z]\d{7,8}\b'
            ),
        }
        return patterns
    
    def detect_pii(self, text: str) -> List[PIIDetection]:
        """
        Detect PII in the given text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of detected PII instances
        """
        if not text or not text.strip():
            return []
        
        detections = []
        
        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                # Calculate confidence based on pattern specificity
                confidence = self._calculate_confidence(pii_type, match.group())
                
                detection = PIIDetection(
                    pii_type=pii_type,
                    value=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence
                )
                detections.append(detection)
        
        # Sort by position for consistent processing
        detections.sort(key=lambda x: x.start_pos)
        
        logger.debug(f"Detected {len(detections)} PII instances in text")
        return detections
    
    def _calculate_confidence(self, pii_type: PIIType, value: str) -> float:
        """Calculate confidence score for PII detection."""
        base_confidence = {
            PIIType.EMAIL: 0.95,
            PIIType.PHONE: 0.90,
            PIIType.SSN: 0.95,
            PIIType.CREDIT_CARD: 0.85,
            PIIType.IP_ADDRESS: 0.80,
            PIIType.NAME: 0.60,  # Lower confidence for names
            PIIType.ADDRESS: 0.70,
            PIIType.DATE_OF_BIRTH: 0.75,
            PIIType.PASSPORT: 0.85,
            PIIType.DRIVER_LICENSE: 0.80,
        }
        
        confidence = base_confidence.get(pii_type, 0.50)
        
        # Adjust confidence based on value characteristics
        if pii_type == PIIType.EMAIL and '@' in value and '.' in value.split('@')[-1]:
            confidence = min(confidence + 0.05, 1.0)
        elif pii_type == PIIType.PHONE and len(re.sub(r'[^\d]', '', value)) == 10:
            confidence = min(confidence + 0.05, 1.0)
        elif pii_type == PIIType.CREDIT_CARD and self._validate_credit_card(value):
            confidence = min(confidence + 0.10, 1.0)
        
        return confidence
    
    def _validate_credit_card(self, number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        # Remove non-digits
        digits = re.sub(r'\D', '', number)
        
        if len(digits) < 13 or len(digits) > 19:
            return False
        
        # Luhn algorithm
        total = 0
        reverse_digits = digits[::-1]
        
        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n = n // 10 + n % 10
            total += n
        
        return total % 10 == 0
    
    def sanitize_text(self, text: str, replacement_strategy: str = "mask") -> Tuple[str, Dict[str, Any]]:
        """
        Sanitize text by removing or masking PII.
        
        Args:
            text: Text to sanitize
            replacement_strategy: Strategy for replacement ("mask", "remove", "hash")
            
        Returns:
            Tuple of (sanitized_text, sanitization_info)
        """
        if not text or not text.strip():
            return text, {}
        
        detections = self.detect_pii(text)
        if not detections:
            return text, {"pii_detected": False, "pii_count": 0}
        
        sanitized_text = text
        sanitization_info = {
            "pii_detected": True,
            "pii_count": len(detections),
            "pii_types": list(set(detection.pii_type.value for detection in detections)),
            "replacements": []
        }
        
        # Process detections in reverse order to maintain positions
        for detection in reversed(detections):
            replacement = self._get_replacement(detection, replacement_strategy)
            sanitized_text = (
                sanitized_text[:detection.start_pos] + 
                replacement + 
                sanitized_text[detection.end_pos:]
            )
            
            sanitization_info["replacements"].append({
                "pii_type": detection.pii_type.value,
                "original_value": detection.value,
                "replacement": replacement,
                "confidence": detection.confidence
            })
        
        logger.info(f"Sanitized {len(detections)} PII instances using {replacement_strategy} strategy")
        return sanitized_text, sanitization_info
    
    def _get_replacement(self, detection: PIIDetection, strategy: str) -> str:
        """Get replacement text for PII based on strategy."""
        if strategy == "mask":
            if detection.pii_type == PIIType.EMAIL:
                return "[EMAIL_REDACTED]"
            elif detection.pii_type == PIIType.PHONE:
                return "[PHONE_REDACTED]"
            elif detection.pii_type == PIIType.SSN:
                return "[SSN_REDACTED]"
            elif detection.pii_type == PIIType.CREDIT_CARD:
                return "[CARD_REDACTED]"
            elif detection.pii_type == PIIType.NAME:
                return "[NAME_REDACTED]"
            else:
                return f"[{detection.pii_type.value.upper()}_REDACTED]"
        
        elif strategy == "remove":
            return ""
        
        elif strategy == "hash":
            import hashlib
            hash_value = hashlib.sha256(detection.value.encode()).hexdigest()[:8]
            return f"[HASH_{hash_value}]"
        
        else:
            return "[PII_REDACTED]"
    
    def get_pii_summary(self, text: str) -> Dict[str, Any]:
        """Get summary of PII detection without sanitizing."""
        detections = self.detect_pii(text)
        
        if not detections:
            return {
                "pii_detected": False,
                "pii_count": 0,
                "pii_types": [],
                "risk_level": "low"
            }
        
        pii_types = [d.pii_type.value for d in detections]
        unique_types = list(set(pii_types))
        
        # Calculate risk level
        high_risk_types = {PIIType.SSN.value, PIIType.CREDIT_CARD.value, PIIType.PASSPORT.value}
        medium_risk_types = {PIIType.EMAIL.value, PIIType.PHONE.value, PIIType.DRIVER_LICENSE.value}
        
        if any(t in high_risk_types for t in unique_types):
            risk_level = "high"
        elif any(t in medium_risk_types for t in unique_types):
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "pii_detected": True,
            "pii_count": len(detections),
            "pii_types": unique_types,
            "risk_level": risk_level,
            "high_confidence_count": len([d for d in detections if d.confidence > 0.8])
        }


# Global PII detector instance
_pii_detector = None


def get_pii_detector() -> PIIDetector:
    """Get the global PII detector instance."""
    global _pii_detector
    if _pii_detector is None:
        _pii_detector = PIIDetector()
    return _pii_detector


def detect_pii_in_text(text: str) -> List[PIIDetection]:
    """Convenience function to detect PII in text."""
    detector = get_pii_detector()
    return detector.detect_pii(text)


def sanitize_text_for_llm(text: str) -> Tuple[str, Dict[str, Any]]:
    """Convenience function to sanitize text before sending to LLM."""
    detector = get_pii_detector()
    return detector.sanitize_text(text, replacement_strategy="mask")
