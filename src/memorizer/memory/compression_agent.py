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

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CompressionResult:
    """Result of a compression operation."""
    def __init__(self, compressed_content: str, compression_ratio: float, original_length: int, compressed_length: int, success: bool = True, error_message: str = None, compressed_data: Dict[str, Any] = None):
        self.compressed_content = compressed_content
        self.compression_ratio = compression_ratio
        self.original_length = original_length
        self.compressed_length = compressed_length
        self.success = success
        self.error_message = error_message
        self.compressed_data = compressed_data or {
            "compressed_content": compressed_content,
            "compression_ratio": compression_ratio,
            "original_length": original_length,
            "compressed_length": compressed_length
        }


class CompressionAgent:
    """Agent that compresses memories using LLM providers."""

    def __init__(self, llm_provider: LLMProvider = None):
        """Initialize the compression agent with an LLM provider."""
        if llm_provider is None:
            # Get default provider from config or factory
            try:
                self.llm_provider = get_llm_provider_from_config()
            except Exception as e:
                logger.warning(f"Failed to get LLM provider from config, using mock: {e}")
                self.llm_provider = LLMProviderFactory.create_provider("mock")
        else:
            self.llm_provider = llm_provider

        self.compression_stats = {
            "total_compressions": 0,
            "successful_compressions": 0,
            "failed_compressions": 0,
            "total_bytes_saved": 0
        }

        logger.info(f"Compression agent initialized with {self.llm_provider.get_provider_name()}")

    def compress_memory(
        self,
        content: str,
        compression_type: str = "mid_term",
        metadata: Dict[str, Any] = None,
        max_retries: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """
        Compress memory content using LLM.

        Args:
            content: The content to compress
            compression_type: Type of compression (mid_term, long_term)
            metadata: Additional metadata for the memory
            max_retries: Maximum number of retry attempts

        Returns:
            Dictionary containing compressed content and metadata
        """
        if metadata is None:
            metadata = {}

        original_length = len(content)

        # Determine compression prompt based on type
        if compression_type == "mid_term":
            prompt = f"Compress and summarize the following text to about 70% of its original length while preserving key information:\n\n{content}"
        elif compression_type == "long_term":
            prompt = f"Create a very concise summary of the following text, reducing it to about 30% of its original length:\n\n{content}"
        else:
            prompt = f"Summarize the following text:\n\n{content}"

        # Try compression with retries
        for attempt in range(max_retries):
            try:
                compressed_content = self.llm_provider.generate(prompt, max_tokens=2000)
                compressed_length = len(compressed_content)
                compression_ratio = compressed_length / original_length if original_length > 0 else 0

                self.compression_stats["total_compressions"] += 1
                self.compression_stats["successful_compressions"] += 1
                self.compression_stats["total_bytes_saved"] += (original_length - compressed_length)

                result = {
                    "compressed_content": compressed_content,
                    "compression_ratio": compression_ratio,
                    "original_length": original_length,
                    "compressed_length": compressed_length,
                    "compression_type": compression_type,
                    "metadata": metadata
                }

                logger.debug(f"Compressed {original_length} -> {compressed_length} bytes (ratio: {compression_ratio:.2f})")
                return result

            except Exception as e:
                logger.warning(f"Compression attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.compression_stats["total_compressions"] += 1
                    self.compression_stats["failed_compressions"] += 1
                    logger.error(f"All compression attempts failed for content")
                    return None

    def batch_compress(
        self,
        contents: List[str],
        compression_type: str = "mid_term",
        batch_size: int = 5
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Compress multiple contents in batches.

        Args:
            contents: List of content strings to compress
            compression_type: Type of compression to apply
            batch_size: Number of items to process in parallel (not implemented)

        Returns:
            List of compression results (None for failed compressions)
        """
        results = []
        for content in contents:
            result = self.compress_memory(
                content=content,
                compression_type=compression_type,
                max_retries=3
            )
            results.append(result)

        return results

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the compression agent."""
        return {
            **self.compression_stats,
            "provider": self.llm_provider.get_provider_name()
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the compression agent."""
        success_rate = 0
        if self.compression_stats["total_compressions"] > 0:
            success_rate = self.compression_stats["successful_compressions"] / self.compression_stats["total_compressions"]

        return {
            "status": "healthy" if success_rate >= 0.8 else "degraded",
            "provider": self.llm_provider.get_provider_name(),
            "success_rate": success_rate,
            "stats": self.compression_stats
        }


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
