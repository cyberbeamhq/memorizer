"""
llm_providers.py
Comprehensive LLM provider implementations for the Memorizer framework.
Supports OpenAI, Anthropic, Groq, OpenRouter, Ollama, and custom models.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_retries: int = 3
    timeout: int = 30
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    headers: Optional[Dict[str, str]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        system_prompt: str = None, 
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Optional[str]:
        """Generate text using the LLM."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of the provider."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI implementation of LLM provider."""

    def __init__(self, config: LLMConfig):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")

        self.config = config
        self.client = OpenAI(
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.base_url
        )
        
        if not self.client.api_key:
            raise ValueError("OpenAI API key is required")

    def generate(
        self, 
        prompt: str, 
        system_prompt: str = None, 
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Optional[str]:
        """Generate text using OpenAI API."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return None

    def get_provider_name(self) -> str:
        return "openai"

    def get_model_name(self) -> str:
        return self.config.model


class AnthropicProvider(LLMProvider):
    """Anthropic Claude implementation of LLM provider."""

    def __init__(self, config: LLMConfig):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")

        self.config = config
        self.client = anthropic.Anthropic(
            api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        
        if not self.client.api_key:
            raise ValueError("Anthropic API key is required")

    def generate(
        self, 
        prompt: str, 
        system_prompt: str = None, 
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Optional[str]:
        """Generate text using Anthropic Claude API."""
        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=max_tokens or self.config.max_tokens or 1000,
                temperature=temperature or self.config.temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            return None

    def get_provider_name(self) -> str:
        return "anthropic"

    def get_model_name(self) -> str:
        return self.config.model


class GroqProvider(LLMProvider):
    """Groq implementation of LLM provider."""

    def __init__(self, config: LLMConfig):
        try:
            from groq import Groq  # type: ignore
        except ImportError:
            raise ImportError("Groq library not installed. Run: pip install groq")

        self.config = config
        api_key = config.api_key or os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key to config.")
        
        self.client = Groq(api_key=api_key)

    def generate(
        self, 
        prompt: str, 
        system_prompt: str = None, 
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Optional[str]:
        """Generate text using Groq API."""
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided to Groq provider")
            return None
            
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                **kwargs
            )
            
            if not response.choices:
                logger.warning("No response choices returned from Groq")
                return None
                
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            return None

    def get_provider_name(self) -> str:
        return "groq"

    def get_model_name(self) -> str:
        return self.config.model


class OpenRouterProvider(LLMProvider):
    """OpenRouter implementation of LLM provider."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = config.base_url or "https://openrouter.ai/api/v1"
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")

    def generate(
        self, 
        prompt: str, 
        system_prompt: str = None, 
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Optional[str]:
        """Generate text using OpenRouter API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/cyberbeamhq/memorizer",
                "X-Title": "Memorizer Framework"
            }
            
            if self.config.headers:
                headers.update(self.config.headers)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                **kwargs
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"OpenRouter generation failed: {e}")
            return None

    def get_provider_name(self) -> str:
        return "openrouter"

    def get_model_name(self) -> str:
        return self.config.model


class OllamaProvider(LLMProvider):
    """Ollama implementation of LLM provider for local models."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}")
        except Exception as e:
            logger.warning(f"Ollama connection test failed: {e}")

    def generate(
        self, 
        prompt: str, 
        system_prompt: str = None, 
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Optional[str]:
        """Generate text using Ollama API."""
        try:
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature or self.config.temperature,
                    "num_predict": max_tokens or self.config.max_tokens,
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response")
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return None

    def get_provider_name(self) -> str:
        return "ollama"

    def get_model_name(self) -> str:
        return self.config.model


class CustomProvider(LLMProvider):
    """Custom model implementation for enterprise or custom APIs."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or os.getenv("CUSTOM_MODEL_BASE_URL")
        self.api_key = config.api_key or os.getenv("CUSTOM_MODEL_API_KEY")
        
        if not self.base_url:
            raise ValueError("Custom model base URL is required")

    def generate(
        self, 
        prompt: str, 
        system_prompt: str = None, 
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Optional[str]:
        """Generate text using custom API."""
        try:
            headers = {
                "Content-Type": "application/json",
            }
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            if self.config.headers:
                headers.update(self.config.headers)

            # Standard OpenAI-compatible format
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                **kwargs
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Custom model generation failed: {e}")
            return None

    def get_provider_name(self) -> str:
        return "custom"

    def get_model_name(self) -> str:
        return self.config.model


class MockProvider(LLMProvider):
    """Mock implementation for testing and development."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def generate(
        self, 
        prompt: str, 
        system_prompt: str = None, 
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Optional[str]:
        """Generate mock text for testing."""
        # Simulate processing time
        time.sleep(0.1)
        
        # Return a mock JSON response for compression
        if "compress" in prompt.lower() or "summary" in prompt.lower():
            return json.dumps({
                "summary": f"Mock compressed summary of: {prompt[:100]}...",
                "key_points": ["Mock point 1", "Mock point 2"],
                "user_preferences": {"preference_type": "mock_value"},
                "metadata": {
                    "compression_type": "mock",
                    "original_length": len(prompt),
                    "compression_ratio": "0.3",
                    "mock": True
                }
            })
        else:
            # Return a mock compressed response
            return f"[MOCK COMPRESSED] {prompt[:50]}... (compressed by {self.config.model})"

    def get_provider_name(self) -> str:
        return "mock"

    def get_model_name(self) -> str:
        return self.config.model


class LLMProviderFactory:
    """Factory class for creating LLM providers."""

    @staticmethod
    def create_provider(config: LLMConfig) -> LLMProvider:
        """Create an LLM provider based on configuration."""
        provider_name = config.provider.lower()
        
        # Validate provider
        if provider_name not in LLMProviderFactory.get_available_providers():
            raise ValueError(f"Unsupported LLM provider: {config.provider}. Available: {LLMProviderFactory.get_available_providers()}")
        
        # Validate model name for providers with known model lists
        if provider_name in ["openai", "anthropic"]:
            provider_info = LLMProviderFactory.get_provider_info(provider_name)
            if config.model not in provider_info.get("models", []):
                logger.warning(f"Model '{config.model}' may not be available for {provider_name}. Available: {provider_info.get('models', [])}")
        
        try:
            if provider_name == "openai":
                return OpenAIProvider(config)
            elif provider_name == "anthropic":
                return AnthropicProvider(config)
            elif provider_name == "groq":
                return GroqProvider(config)
            elif provider_name == "openrouter":
                return OpenRouterProvider(config)
            elif provider_name == "ollama":
                return OllamaProvider(config)
            elif provider_name == "custom":
                return CustomProvider(config)
            elif provider_name == "mock":
                return MockProvider(config)
        except ImportError as e:
            raise ImportError(f"Required library not installed for {provider_name}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to create {provider_name} provider: {e}")

    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available LLM providers."""
        return ["openai", "anthropic", "groq", "openrouter", "ollama", "custom", "mock"]

    @staticmethod
    def get_provider_info(provider_name: str) -> Dict[str, Any]:
        """Get information about a specific provider."""
        info = {
            "openai": {
                "name": "OpenAI",
                "description": "OpenAI GPT models (GPT-4, GPT-3.5, etc.)",
                "models": [
                    "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", 
                    "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
                ],
                "requires_api_key": True,
                "supports_streaming": True,
                "model_format": "model_name"
            },
            "anthropic": {
                "name": "Anthropic",
                "description": "Anthropic Claude models",
                "models": [
                    "claude-3-opus-20240229", "claude-3-sonnet-20240229", 
                    "claude-3-haiku-20240307", "claude-2.1", "claude-2.0"
                ],
                "requires_api_key": True,
                "supports_streaming": True,
                "model_format": "model_name"
            },
            "groq": {
                "name": "Groq",
                "description": "Fast inference with various open models (requires specific model names)",
                "models": [
                    # Llama models
                    "llama3-8b-8192", "llama3-70b-8192", "llama3.1-8b-instant-128k",
                    "llama3.1-70b-versatile-128k", "llama3.1-405b-instruct-128k",
                    # Mixtral models
                    "mixtral-8x7b-32768", "mixtral-8x22b-32768",
                    # Gemma models
                    "gemma-7b-it", "gemma-2-9b-it",
                    # Code models
                    "codegemma-7b-it", "codegemma-2-9b-it",
                    # Qwen models
                    "qwen2.5-7b-instruct", "qwen2.5-14b-instruct", "qwen2.5-32b-instruct",
                    # Other models
                    "deepseek-coder-6.7b-instruct", "llama-3.2-3b-instruct"
                ],
                "requires_api_key": True,
                "supports_streaming": True,
                "model_format": "exact_model_name",
                "note": "Check https://console.groq.com/docs/models for latest available models"
            },
            "openrouter": {
                "name": "OpenRouter",
                "description": "Access to multiple models through unified API (requires full model path)",
                "models": [
                    # Anthropic models
                    "anthropic/claude-3-opus-20240229", "anthropic/claude-3-sonnet-20240229",
                    "anthropic/claude-3-haiku-20240307", "anthropic/claude-2.1",
                    # OpenAI models
                    "openai/gpt-4o-mini", "openai/gpt-4o", "openai/gpt-4-turbo",
                    "openai/gpt-3.5-turbo", "openai/gpt-4",
                    # Meta models
                    "meta-llama/llama-3.1-8b-instruct", "meta-llama/llama-3.1-70b-instruct",
                    "meta-llama/llama-3.2-3b-instruct", "meta-llama/llama-3.2-11b-instruct",
                    # Google models
                    "google/gemini-pro-1.5", "google/gemini-flash-1.5",
                    # Mistral models
                    "mistralai/mistral-7b-instruct", "mistralai/mixtral-8x7b-instruct",
                    "mistralai/mixtral-8x22b-instruct", "mistralai/mistral-nemo-12b-2409",
                    # Cohere models
                    "cohere/command-r-plus", "cohere/command-r",
                    # Other models
                    "microsoft/phi-3-medium-128k-instruct", "deepseek/deepseek-coder-6.7b-instruct",
                    "qwen/qwen-2.5-7b-instruct", "qwen/qwen-2.5-14b-instruct"
                ],
                "requires_api_key": True,
                "supports_streaming": True,
                "model_format": "provider/model_name",
                "note": "Check https://openrouter.ai/models for latest available models and pricing"
            },
            "ollama": {
                "name": "Ollama",
                "description": "Local model deployment and inference (requires locally installed models)",
                "models": [
                    # Llama models
                    "llama2", "llama2:7b", "llama2:13b", "llama2:70b",
                    "llama3", "llama3:8b", "llama3:70b", "llama3.1:8b", "llama3.1:70b",
                    # Mistral models
                    "mistral", "mistral:7b", "mixtral", "mixtral:8x7b", "mixtral:8x22b",
                    # Code models
                    "codellama", "codellama:7b", "codellama:13b", "codellama:34b",
                    "codegemma", "codegemma:7b", "codegemma:2b",
                    # Other models
                    "gemma", "gemma:2b", "gemma:7b", "gemma:9b",
                    "phi3", "phi3:mini", "phi3:medium",
                    "qwen", "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b",
                    "deepseek-coder", "deepseek-coder:6.7b", "deepseek-coder:33b",
                    "neural-chat", "starling-lm", "orca-mini", "vicuna"
                ],
                "requires_api_key": False,
                "supports_streaming": True,
                "model_format": "model_name:tag",
                "note": "Models must be installed locally with 'ollama pull <model_name>'. Check https://ollama.ai/library for available models"
            },
            "custom": {
                "name": "Custom",
                "description": "Custom or enterprise model APIs (OpenAI-compatible endpoints)",
                "models": ["custom-model", "enterprise-model", "private-model"],
                "requires_api_key": True,
                "supports_streaming": False,
                "model_format": "your_model_name",
                "note": "Must be OpenAI-compatible API endpoint"
            },
            "mock": {
                "name": "Mock",
                "description": "Mock provider for testing and development",
                "models": ["mock-model", "test-model"],
                "requires_api_key": False,
                "supports_streaming": False,
                "model_format": "any_name"
            }
        }
        
        return info.get(provider_name.lower(), {})


def get_llm_provider_from_config() -> LLMProvider:
    """Get LLM provider from environment configuration."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    # Provider-specific configuration with proper model names
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
    elif provider == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL", os.getenv("LLM_MODEL", "claude-3-sonnet-20240229"))
        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    elif provider == "groq":
        model = os.getenv("GROQ_MODEL", os.getenv("LLM_MODEL", "llama3-8b-8192"))
        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=os.getenv("GROQ_API_KEY")
        )
    elif provider == "openrouter":
        model = os.getenv("OPENROUTER_MODEL", os.getenv("LLM_MODEL", "anthropic/claude-3-sonnet-20240229"))
        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        )
    elif provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", os.getenv("LLM_MODEL", "llama3:8b"))
        config = LLMConfig(
            provider=provider,
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
    elif provider == "custom":
        model = os.getenv("CUSTOM_MODEL_NAME", os.getenv("LLM_MODEL", "custom-model"))
        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=os.getenv("CUSTOM_MODEL_API_KEY"),
            base_url=os.getenv("CUSTOM_MODEL_BASE_URL"),
            headers=json.loads(os.getenv("CUSTOM_MODEL_HEADERS", "{}"))
        )
    elif provider == "mock":
        model = os.getenv("LLM_MODEL", "mock-model")
        config = LLMConfig(
            provider=provider,
            model=model
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    
    return LLMProviderFactory.create_provider(config)


def list_available_models(provider_name: str = None) -> Dict[str, Any]:
    """
    List available models for a specific provider or all providers.
    
    Args:
        provider_name: Specific provider to list models for, or None for all providers
        
    Returns:
        Dictionary with provider information and available models
    """
    if provider_name:
        provider_name = provider_name.lower()
        if provider_name not in LLMProviderFactory.get_available_providers():
            raise ValueError(f"Unknown provider: {provider_name}")
        
        return {provider_name: LLMProviderFactory.get_provider_info(provider_name)}
    else:
        # Return all providers
        result = {}
        for provider in LLMProviderFactory.get_available_providers():
            result[provider] = LLMProviderFactory.get_provider_info(provider)
        return result


def validate_model_for_provider(provider_name: str, model_name: str) -> bool:
    """
    Validate if a model is available for a specific provider.
    
    Args:
        provider_name: The provider name
        model_name: The model name to validate
        
    Returns:
        True if model is available, False otherwise
    """
    provider_info = LLMProviderFactory.get_provider_info(provider_name.lower())
    if not provider_info:
        return False
    
    # For providers with dynamic model lists (Groq, OpenRouter, Ollama),
    # we can't validate against a static list, so we return True
    if provider_name.lower() in ["groq", "openrouter", "ollama"]:
        return True
    
    # For providers with known model lists, check against the list
    return model_name in provider_info.get("models", [])


def get_model_recommendations(use_case: str = "general") -> Dict[str, str]:
    """
    Get model recommendations based on use case.
    
    Args:
        use_case: The use case (general, fast, cheap, high-quality, local, coding)
        
    Returns:
        Dictionary mapping providers to recommended models
    """
    recommendations = {
        "general": {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-sonnet-20240229",
            "groq": "llama3-8b-8192",
            "openrouter": "anthropic/claude-3-sonnet-20240229",
            "ollama": "llama3:8b"
        },
        "fast": {
            "groq": "llama3-8b-8192",
            "openai": "gpt-3.5-turbo",
            "anthropic": "claude-3-haiku-20240307",
            "openrouter": "openai/gpt-3.5-turbo",
            "ollama": "llama3:8b"
        },
        "cheap": {
            "openai": "gpt-3.5-turbo",
            "groq": "llama3-8b-8192",
            "openrouter": "openai/gpt-3.5-turbo",
            "ollama": "llama3:8b"
        },
        "high-quality": {
            "openai": "gpt-4o",
            "anthropic": "claude-3-opus-20240229",
            "openrouter": "anthropic/claude-3-opus-20240229"
        },
        "local": {
            "ollama": "llama3:8b"
        },
        "coding": {
            "openai": "gpt-4o-mini",
            "groq": "deepseek-coder-6.7b-instruct",
            "openrouter": "deepseek/deepseek-coder-6.7b-instruct",
            "ollama": "codellama:7b"
        }
    }
    
    return recommendations.get(use_case, recommendations["general"])


def validate_provider_config(provider: str, model: str, api_key: str = None) -> Dict[str, Any]:
    """
    Validate provider configuration and return validation results.
    
    Args:
        provider: Provider name
        model: Model name
        api_key: API key (optional)
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "provider_info": None
    }
    
    # Check if provider is supported
    if provider not in LLMProviderFactory.get_available_providers():
        result["valid"] = False
        result["errors"].append(f"Unsupported provider: {provider}")
        return result
    
    # Get provider info
    provider_info = LLMProviderFactory.get_provider_info(provider)
    result["provider_info"] = provider_info
    
    # Check API key requirement
    if provider_info.get("requires_api_key", False) and not api_key:
        result["warnings"].append(f"API key recommended for {provider}")
    
    # Validate model name
    if not validate_model_for_provider(provider, model):
        if provider in ["groq", "openrouter", "ollama"]:
            result["warnings"].append(f"Model '{model}' may not be available for {provider}")
        else:
            result["valid"] = False
            result["errors"].append(f"Model '{model}' not available for {provider}")
    
    return result


def get_provider_status(provider: str) -> Dict[str, Any]:
    """
    Get the current status and capabilities of a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        Dictionary with provider status information
    """
    provider_info = LLMProviderFactory.get_provider_info(provider)
    if not provider_info:
        return {"error": f"Unknown provider: {provider}"}
    
    status = {
        "provider": provider,
        "name": provider_info["name"],
        "description": provider_info["description"],
        "model_format": provider_info["model_format"],
        "requires_api_key": provider_info["requires_api_key"],
        "supports_streaming": provider_info["supports_streaming"],
        "available_models": len(provider_info["models"]),
        "note": provider_info.get("note", ""),
        "status": "available"
    }
    
    # Check if required libraries are installed
    try:
        if provider == "openai":
            import openai  # type: ignore
        elif provider == "anthropic":
            import anthropic  # type: ignore
        elif provider == "groq":
            import groq  # type: ignore
    except ImportError:
        status["status"] = "library_missing"
        status["error"] = f"Required library not installed for {provider}"
    
    return status
