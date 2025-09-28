"""
compression_agent.py
Uses an LLM (supports multiple providers) to compress or summarize content.
Includes safe parsing and retry logic with exponential backoff.

PHASE 1 ENHANCEMENTS:
- PII detection before LLM processing
- Strict JSON schema validation
- Comprehensive metrics collection
- Enhanced error handling and retry logic
- Production-ready security and robustness
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from ..integrations.llm_providers import LLMProvider, LLMProviderFactory, get_llm_provider_from_config
# Enhanced compression agent removed to reduce complexity

class CompressionResult:
    """Simple compression result class."""
    def __init__(self, compressed_content: str, compression_ratio: float, original_length: int, compressed_length: int):
        self.compressed_content = compressed_content
        self.compression_ratio = compression_ratio
        self.original_length = original_length
        self.compressed_length = compressed_length

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CompressionAgent:
    """Agent that compresses memories using LLM providers.
    
    This is now a wrapper around the enhanced compression agent for backward compatibility.
    For new implementations, use EnhancedCompressionAgent directly.
    """

    def __init__(self, llm_provider: LLMProvider = None):
        """Initialize the compression agent with an LLM provider."""
        self.enhanced_agent = EnhancedCompressionAgent(llm_provider)
        logger.info(f"Compression agent initialized with {self.enhanced_agent.llm_provider.get_provider_name()}")
        logger.info("Using enhanced compression agent with PII detection, schema validation, and metrics")

    def compress_memory(
        self,
        content: str,
        compression_type: str = "mid_term",
        metadata: Dict[str, Any] = None,
        max_retries: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """
        Compress memory content using the enhanced compression agent.
        
        Args:
            content: The content to compress
            compression_type: Type of compression (mid_term, long_term)
            metadata: Additional metadata for the memory
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary containing compressed content and metadata
        """
        # Use enhanced compression agent with all safety features enabled
        result = self.enhanced_agent.compress_memory(
            content=content,
            compression_type=compression_type,
            metadata=metadata,
            max_retries=max_retries,
            enable_pii_detection=True,
            enable_schema_validation=True,
            enable_metrics=True
        )
        
        if result.success:
            return result.compressed_data
        else:
            logger.error(f"Compression failed: {result.error_message}")
            return None

    # Legacy methods removed - now using enhanced compression agent
    # All compression logic is handled by EnhancedCompressionAgent

    def batch_compress(
        self, 
        contents: List[str], 
        compression_type: str = "mid_term",
        batch_size: int = 5
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Compress multiple contents in batches using enhanced agent.
        
        Args:
            contents: List of content strings to compress
            compression_type: Type of compression to apply
            batch_size: Number of items to process in parallel
            
        Returns:
            List of compression results (None for failed compressions)
        """
        # Use enhanced batch compression
        results = self.enhanced_agent.batch_compress(
            contents=contents,
            compression_type=compression_type,
            batch_size=batch_size,
            enable_pii_detection=True,
            enable_schema_validation=True,
            enable_metrics=True
        )
        
        # Convert CompressionResult objects to dicts for backward compatibility
        return [result.compressed_data if result.success else None for result in results]

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the compression agent."""
        return self.enhanced_agent.get_compression_stats()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the compression agent."""
        return self.enhanced_agent.get_health_status()


# Global compression agent instance
_compression_agent = None


def get_compression_agent() -> CompressionAgent:
    """Get the global compression agent instance."""
    global _compression_agent
    if _compression_agent is None:
        _compression_agent = CompressionAgent()
    return _compression_agent


def initialize_compression_agent(llm_provider: LLMProvider = None) -> CompressionAgent:
    """Initialize the global compression agent with a specific provider."""
    global _compression_agent
    _compression_agent = CompressionAgent(llm_provider)
    return _compression_agent
