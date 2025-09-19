"""
compression_agent.py
Uses an LLM (supports multiple providers) to compress or summarize content.
Includes safe parsing and retry logic with exponential backoff.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from .llm_providers import LLMProvider, LLMProviderFactory, get_llm_provider_from_config

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CompressionAgent:
    """Agent that compresses memories using LLM providers."""

    def __init__(self, llm_provider: LLMProvider = None):
        """Initialize the compression agent with an LLM provider."""
        self.llm_provider = llm_provider or get_llm_provider_from_config()
        logger.info(f"Compression agent initialized with {self.llm_provider.get_provider_name()}")

    def compress_memory(
        self,
        content: str,
        compression_type: str = "mid_term",
        metadata: Dict[str, Any] = None,
        max_retries: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """
        Compress memory content using the configured LLM provider.
        
        Args:
            content: The content to compress
            compression_type: Type of compression (mid_term, long_term)
            metadata: Additional metadata for the memory
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary containing compressed content and metadata
        """
        if not content or not content.strip():
            logger.warning("Empty content provided for compression")
            return None

        metadata = metadata or {}
        
        # Create compression prompt based on type
        if compression_type == "mid_term":
            prompt = self._create_mid_term_compression_prompt(content)
        elif compression_type == "long_term":
            prompt = self._create_long_term_compression_prompt(content)
        else:
            prompt = self._create_general_compression_prompt(content)

        system_prompt = self._get_compression_system_prompt(compression_type)

        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                logger.debug(f"Compression attempt {attempt + 1}/{max_retries}")
                
                response = self.llm_provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.3,  # Low temperature for consistent compression
                    max_tokens=1000
                )
                
                if not response:
                    logger.warning(f"Empty response from LLM provider on attempt {attempt + 1}")
                    continue

                # Parse the response
                compressed_data = self._parse_compression_response(response, compression_type)
                
                if compressed_data:
                    logger.info(f"Successfully compressed content using {self.llm_provider.get_provider_name()}")
                    return compressed_data
                else:
                    logger.warning(f"Failed to parse compression response on attempt {attempt + 1}")

            except Exception as e:
                logger.error(f"Compression attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} compression attempts failed")

        return None

    def _create_mid_term_compression_prompt(self, content: str) -> str:
        """Create prompt for mid-term compression."""
        return f"""
Please compress the following conversation/interaction into a concise summary while preserving key information:

CONTENT TO COMPRESS:
{content}

COMPRESSION REQUIREMENTS:
- Remove unnecessary words and filler content
- Preserve user preferences, decisions, and important facts
- Maintain context and relationships between topics
- Keep the summary under 200 words
- Focus on actionable insights and key takeaways

Please respond with a JSON object containing:
{{
    "summary": "compressed summary text",
    "key_points": ["point1", "point2", "point3"],
    "user_preferences": {{"preference_type": "value"}},
    "metadata": {{
        "compression_type": "mid_term",
        "original_length": {len(content)},
        "compression_ratio": "calculated ratio"
    }}
}}
"""

    def _create_long_term_compression_prompt(self, content: str) -> str:
        """Create prompt for long-term compression."""
        return f"""
Please create a highly aggregated brief from the following content, focusing on long-term insights and patterns:

CONTENT TO AGGREGATE:
{content}

AGGREGATION REQUIREMENTS:
- Extract key behavioral patterns and preferences
- Identify sentiment and emotional context
- Summarize into a brief under 1000 characters
- Focus on actionable insights for future interactions
- Preserve critical user information and preferences

Please respond with a JSON object containing:
{{
    "brief": "highly compressed brief text",
    "patterns": ["pattern1", "pattern2"],
    "sentiment": "positive/negative/neutral",
    "preferences": {{"category": "value"}},
    "metadata": {{
        "compression_type": "long_term",
        "original_length": {len(content)},
        "insights_count": "number of insights extracted"
    }}
}}
"""

    def _create_general_compression_prompt(self, content: str) -> str:
        """Create prompt for general compression."""
        return f"""
Please compress the following content while preserving essential information:

CONTENT TO COMPRESS:
{content}

Please respond with a JSON object containing:
{{
    "summary": "compressed summary",
    "metadata": {{
        "compression_type": "general",
        "original_length": {len(content)}
    }}
}}
"""

    def _get_compression_system_prompt(self, compression_type: str) -> str:
        """Get system prompt for compression."""
        base_prompt = """You are an expert content compression agent. Your task is to compress text while preserving essential information, user preferences, and actionable insights. Always respond with valid JSON format."""
        
        if compression_type == "mid_term":
            return base_prompt + " Focus on creating concise summaries that maintain context and relationships."
        elif compression_type == "long_term":
            return base_prompt + " Focus on extracting long-term patterns, preferences, and behavioral insights."
        else:
            return base_prompt

    def _parse_compression_response(self, response: str, compression_type: str) -> Optional[Dict[str, Any]]:
        """Parse and validate the compression response."""
        try:
            # Try to parse as JSON
            parsed = json.loads(response)
            
            # Validate required fields
            if "summary" not in parsed and "brief" not in parsed:
                logger.warning("Response missing summary/brief field")
                return None
            
            # Add compression metadata
            if "metadata" not in parsed:
                parsed["metadata"] = {}
            
            parsed["metadata"]["compression_timestamp"] = time.time()
            parsed["metadata"]["llm_provider"] = self.llm_provider.get_provider_name()
            parsed["metadata"]["llm_model"] = self.llm_provider.get_model_name()
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            
            # Fallback: create a simple summary
            return {
                "summary": response[:500] + "..." if len(response) > 500 else response,
                "metadata": {
                    "compression_type": compression_type,
                    "compression_timestamp": time.time(),
                    "llm_provider": self.llm_provider.get_provider_name(),
                    "llm_model": self.llm_provider.get_model_name(),
                    "parse_error": True
                }
            }
        
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
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
            batch_size: Number of items to process in parallel
            
        Returns:
            List of compression results (None for failed compressions)
        """
        results = []
        
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            logger.info(f"Processing compression batch {i//batch_size + 1}")
            
            batch_results = []
            for content in batch:
                result = self.compress_memory(content, compression_type)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(contents):
                time.sleep(1)
        
        return results

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get statistics about the compression agent."""
        return {
            "provider": self.llm_provider.get_provider_name(),
            "model": self.llm_provider.get_model_name(),
            "timestamp": time.time()
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
