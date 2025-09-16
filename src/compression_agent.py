"""
compression_agent.py
Uses an LLM (default: OpenAI gpt-4o-mini) to compress or summarize content.
Includes safe parsing and retry logic with exponential backoff.
"""
import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------
# Abstract LLM Provider Interface
# ---------------------------
class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> Optional[str]:
        """Generate text using the LLM."""
        pass

# ---------------------------
# OpenAI Provider Implementation
# ---------------------------
class OpenAIProvider(LLMProvider):
    """OpenAI implementation of LLM provider."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"OpenAI provider initialized with model: {model}")
    
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> Optional[str]:
        """Generate text using OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0),
                max_tokens=kwargs.get("max_tokens", 2000),
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

# ---------------------------
# Mock Provider for Testing
# ---------------------------
class MockProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> Optional[str]:
        """Return mock JSON responses."""
        if "mid-term" in prompt.lower():
            return json.dumps({
                "summary": "Mock mid-term summary of user interaction",
                "metadata": {"compression_type": "mid_term", "mock": True}
            })
        else:
            return json.dumps({
                "summary": "Mock long-term aggregated insights with preferences and sentiment",
                "metadata": {"compression_type": "long_term", "mock": True, "item_count": 5}
            })

# ---------------------------
# Compression Agent
# ---------------------------
class CompressionAgent:
    """
    Main compression agent that handles memory lifecycle compression.
    Uses configurable LLM providers with retry logic and validation.
    """
    
    def __init__(self, provider: LLMProvider = None, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize compression agent.
        
        Args:
            provider: LLM provider instance
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
        """
        self.provider = provider or self._get_default_provider()
        self.max_retries = max_retries
        self.base_delay = base_delay
        
    def _get_default_provider(self) -> LLMProvider:
        """Get default LLM provider based on available API keys."""
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if openai_key:
            return OpenAIProvider(api_key=openai_key)
        else:
            logger.warning("No LLM API keys found, using mock provider")
            return MockProvider()
    
    def _call_llm_with_retry(self, prompt: str, system_prompt: str = None, **kwargs) -> Optional[str]:
        """
        Call LLM with exponential backoff retry logic.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            **kwargs: Additional arguments for LLM
            
        Returns:
            LLM response or None if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                response = self.provider.generate(prompt, system_prompt, **kwargs)
                if response:
                    return response
                    
            except Exception as e:
                delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error("All retry attempts failed")
        
        return None
    
    def _safe_json_parse(self, text: str, fallback: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Parse JSON safely with comprehensive fallback handling.
        
        Args:
            text: Text to parse as JSON
            fallback: Fallback dictionary if parsing fails
            
        Returns:
            Parsed JSON or fallback dictionary
        """
        if fallback is None:
            fallback = {"summary": "Parsing failed", "metadata": {}}
        
        if not text or not text.strip():
            logger.warning("Empty response from LLM")
            return fallback
        
        # Try to extract JSON from response (handle cases where LLM adds extra text)
        text = text.strip()
        
        # Look for JSON block markers
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_text = text[json_start:json_end]
        else:
            json_text = text
        
        try:
            parsed = json.loads(json_text)
            
            # Validate required fields
            if not isinstance(parsed, dict):
                raise ValueError("Response is not a dictionary")
            
            if "summary" not in parsed:
                parsed["summary"] = fallback.get("summary", "No summary available")
            
            if "metadata" not in parsed:
                parsed["metadata"] = {}
            
            return parsed
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parsing failed: {e}. Raw response: {text[:100]}...")
            return fallback
    
    def _validate_compression_result(self, result: Dict[str, Any], max_length: int = None) -> Dict[str, Any]:
        """
        Validate and sanitize compression results.
        
        Args:
            result: Compression result dictionary
            max_length: Maximum allowed summary length
            
        Returns:
            Validated and sanitized result
        """
        # Ensure required fields exist
        if "summary" not in result or not result["summary"]:
            result["summary"] = "Summary unavailable"
        
        if "metadata" not in result:
            result["metadata"] = {}
        
        # Truncate summary if too long
        if max_length and len(result["summary"]) > max_length:
            logger.warning(f"Summary too long ({len(result['summary'])} chars), truncating to {max_length}")
            result["summary"] = result["summary"][:max_length - 3] + "..."
            result["metadata"]["truncated"] = True
        
        # Add compression metadata
        result["metadata"]["compressed_at"] = time.time()
        result["metadata"]["compression_agent"] = "memorizer_v1"
        
        return result
    
    def compress_to_mid_term(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Compress content into concise summary for mid_term storage.
        
        Args:
            content: Original content to compress
            metadata: Original metadata to consider
            
        Returns:
            Dictionary with compressed summary and metadata
        """
        if not content or not content.strip():
            logger.warning("Empty content provided for mid-term compression")
            return {"summary": "Empty content", "metadata": {"error": "no_content"}}
        
        # Prepare context from metadata
        context_info = ""
        if metadata:
            context_info = f"Original metadata: {json.dumps(metadata, indent=2)}\n\n"
        
        system_prompt = """You are a memory compression specialist. Your job is to create concise, informative summaries that preserve the most important information while reducing token usage.

RULES:
1. Output ONLY valid JSON with exactly these fields: {"summary": "...", "metadata": {...}}
2. Keep summaries concise but informative
3. Preserve key facts, actions, decisions, and outcomes
4. Remove filler words and redundant information
5. Include important context in metadata
6. Maximum 500 characters for summary"""
        
        user_prompt = f"""{context_info}Compress the following user interaction into a mid-term memory summary:

CONTENT TO COMPRESS:
{content}

Remember: Output only JSON with summary and metadata fields."""
        
        try:
            raw_response = self._call_llm_with_retry(user_prompt, system_prompt, max_tokens=800)
            
            if not raw_response:
                logger.error("No response from LLM for mid-term compression")
                return {
                    "summary": content[:200] + "..." if len(content) > 200 else content,
                    "metadata": {"error": "llm_failed", "original_length": len(content)}
                }
            
            result = self._safe_json_parse(raw_response, {
                "summary": content[:200] + "..." if len(content) > 200 else content,
                "metadata": {"error": "parse_failed"}
            })
            
            # Add original metadata to result
            if metadata:
                result["metadata"].update(metadata)
            
            return self._validate_compression_result(result, max_length=500)
            
        except Exception as e:
            logger.error(f"Mid-term compression failed: {e}")
            return {
                "summary": content[:200] + "..." if len(content) > 200 else content,
                "metadata": {"error": "compression_failed", "original_length": len(content)}
            }
    
    def compress_to_long_term(
        self, 
        content_list: List[str], 
        metadata_list: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate multiple mid-term items into a long-term insight.
        Creates comprehensive user profile with sentiment, preferences, and patterns.
        
        Args:
            content_list: List of mid-term content to aggregate
            metadata_list: List of corresponding metadata
            
        Returns:
            Dictionary with aggregated insights (must be <1000 characters)
        """
        if not content_list:
            logger.warning("Empty content list provided for long-term compression")
            return {"summary": "No content to aggregate", "metadata": {"error": "no_content"}}
        
        # Filter out empty content
        valid_content = [c.strip() for c in content_list if c and c.strip()]
        if not valid_content:
            return {"summary": "No valid content to aggregate", "metadata": {"error": "no_valid_content"}}
        
        # Prepare metadata context
        metadata_context = ""
        if metadata_list:
            metadata_context = f"Metadata context: {json.dumps(metadata_list[:5], indent=2)}\n\n"  # Limit to avoid token overflow
        
        system_prompt = f"""You are an expert at creating comprehensive user profiles from interaction history. Create a long-term memory insight that captures:

1. USER PATTERNS: Common behaviors, preferences, recurring themes
2. SENTIMENT: Overall emotional tone and attitude changes
3. KEY METRICS: Counts, frequencies, important statistics
4. PREFERENCES: Clear likes/dislikes, preferred approaches
5. INSIGHTS: What this tells us about the user

CRITICAL REQUIREMENTS:
- Output ONLY valid JSON: {{"summary": "...", "metadata": {{...}}}}
- Summary must be under 950 characters total
- Be concise but comprehensive
- Focus on actionable insights
- Include numerical data where relevant"""
        
        content_summary = f"Aggregating {len(valid_content)} interactions:\n\n"
        for i, content in enumerate(valid_content[:10]):  # Limit to avoid token overflow
            content_summary += f"{i+1}. {content[:200]}{'...' if len(content) > 200 else ''}\n"
        
        user_prompt = f"""{metadata_context}{content_summary}

Create a comprehensive long-term user profile from these interactions. Focus on patterns, preferences, sentiment, and actionable insights.

Remember: Output only JSON with summary under 950 characters and relevant metadata."""
        
        try:
            raw_response = self._call_llm_with_retry(user_prompt, system_prompt, max_tokens=1200)
            
            if not raw_response:
                logger.error("No response from LLM for long-term compression")
                return {
                    "summary": f"Aggregated {len(valid_content)} interactions. Analysis unavailable.",
                    "metadata": {"error": "llm_failed", "source_count": len(valid_content)}
                }
            
            result = self._safe_json_parse(raw_response, {
                "summary": f"Aggregated insights from {len(valid_content)} interactions.",
                "metadata": {"error": "parse_failed"}
            })
            
            # Add aggregation metadata
            result["metadata"].update({
                "source_count": len(valid_content),
                "aggregated_from": "mid_term",
                "total_original_length": sum(len(c) for c in valid_content)
            })
            
            return self._validate_compression_result(result, max_length=950)
            
        except Exception as e:
            logger.error(f"Long-term compression failed: {e}")
            return {
                "summary": f"Aggregated {len(valid_content)} interactions. Compression failed: {str(e)[:100]}",
                "metadata": {"error": "compression_failed", "source_count": len(valid_content)}
            }

# ---------------------------
# Global Instance and Convenience Functions
# ---------------------------
_default_agent = None

def get_compression_agent() -> CompressionAgent:
    """Get or create the default compression agent instance."""
    global _default_agent
    if _default_agent is None:
        _default_agent = CompressionAgent()
    return _default_agent

def compress_to_mid_term(content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function for mid-term compression."""
    return get_compression_agent().compress_to_mid_term(content, metadata)

def compress_to_long_term(
    content_list: List[str], 
    metadata_list: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function for long-term compression."""
    return get_compression_agent().compress_to_long_term(content_list, metadata_list)

# ---------------------------
# Configuration and Testing
# ---------------------------
def configure_agent(provider: LLMProvider = None, **kwargs) -> CompressionAgent:
    """Configure and return a new compression agent."""
    global _default_agent
    _default_agent = CompressionAgent(provider=provider, **kwargs)
    return _default_agent

def test_compression_agent():
    """Test the compression agent with sample data."""
    agent = get_compression_agent()
    
    # Test mid-term compression
    sample_content = "User asked about refund policy for their recent order #12345. Explained the 30-day return policy and provided instructions for initiating a return. User seemed satisfied with the response and thanked support."
    
    print("Testing mid-term compression...")
    mid_result = agent.compress_to_mid_term(sample_content)
    print(f"Mid-term result: {json.dumps(mid_result, indent=2)}")
    
    # Test long-term compression
    sample_contents = [
        "User inquired about shipping delays, expressed frustration",
        "User placed order for premium product, happy with fast checkout",
        "User contacted support about product defect, requested replacement",
        "User left positive review, mentioned excellent customer service",
        "User asked about bulk discounts for business account"
    ]
    
    print("\nTesting long-term compression...")
    long_result = agent.compress_to_long_term(sample_contents)
    print(f"Long-term result: {json.dumps(long_result, indent=2)}")

if __name__ == "__main__":
    test_compression_agent()
