#!/usr/bin/env python3
"""
LLM Providers Example
Demonstrates how to use different LLM providers with the Memorizer framework.
"""

import os
import sys
import json
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_providers import (
    LLMProviderFactory,
    LLMConfig,
    get_llm_provider_from_config,
    validate_provider_config,
    get_provider_status,
    get_model_recommendations
)


def demonstrate_provider_creation():
    """Demonstrate how to create different LLM providers."""
    print("🔧 Creating LLM Providers")
    print("=" * 50)
    
    # Example 1: Create OpenAI provider
    print("\n1. OpenAI Provider:")
    try:
        config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY", "test-key")
        )
        provider = LLMProviderFactory.create_provider(config)
        print(f"   ✅ Created: {provider.get_provider_name()} - {provider.get_model_name()}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Example 2: Create Groq provider
    print("\n2. Groq Provider:")
    try:
        config = LLMConfig(
            provider="groq",
            model="llama3-8b-8192",
            api_key=os.getenv("GROQ_API_KEY", "test-key")
        )
        provider = LLMProviderFactory.create_provider(config)
        print(f"   ✅ Created: {provider.get_provider_name()} - {provider.get_model_name()}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Example 3: Create Mock provider (always works)
    print("\n3. Mock Provider:")
    try:
        config = LLMConfig(
            provider="mock",
            model="test-model"
        )
        provider = LLMProviderFactory.create_provider(config)
        print(f"   ✅ Created: {provider.get_provider_name()} - {provider.get_model_name()}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")


def demonstrate_text_generation():
    """Demonstrate text generation with different providers."""
    print("\n\n🤖 Text Generation Examples")
    print("=" * 50)
    
    # Create a mock provider for demonstration
    config = LLMConfig(provider="mock", model="demo-model")
    provider = LLMProviderFactory.create_provider(config)
    
    test_prompts = [
        "Summarize this conversation: User asked about weather in New York.",
        "What are the key features of a good laptop?",
        "Explain quantum computing in simple terms."
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: {prompt[:50]}...")
        try:
            response = provider.generate(
                prompt=prompt,
                system_prompt="You are a helpful assistant.",
                temperature=0.7,
                max_tokens=100
            )
            print(f"   Response: {response[:100]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")


def demonstrate_configuration_validation():
    """Demonstrate configuration validation."""
    print("\n\n🔍 Configuration Validation")
    print("=" * 50)
    
    test_configs = [
        ("openai", "gpt-4o-mini", "test-key"),
        ("groq", "llama3-8b-8192", "test-key"),
        ("openrouter", "anthropic/claude-3-sonnet-20240229", "test-key"),
        ("ollama", "llama3:8b", None),
        ("invalid", "test-model", "test-key")
    ]
    
    for provider, model, api_key in test_configs:
        print(f"\nValidating {provider} with model {model}:")
        result = validate_provider_config(provider, model, api_key)
        
        if result["valid"]:
            print("   ✅ Valid configuration")
        else:
            print("   ❌ Invalid configuration")
            for error in result["errors"]:
                print(f"      • {error}")
        
        if result["warnings"]:
            print("   ⚠️ Warnings:")
            for warning in result["warnings"]:
                print(f"      • {warning}")


def demonstrate_provider_status():
    """Demonstrate provider status checking."""
    print("\n\n📊 Provider Status Check")
    print("=" * 50)
    
    providers = ["openai", "anthropic", "groq", "openrouter", "ollama", "mock"]
    
    for provider in providers:
        print(f"\n{provider.upper()}:")
        status = get_provider_status(provider)
        
        if "error" in status:
            print(f"   ❌ {status['error']}")
        else:
            print(f"   Status: {status['status']}")
            print(f"   Models: {status['available_models']}")
            print(f"   API Key Required: {'Yes' if status['requires_api_key'] else 'No'}")
            print(f"   Streaming: {'Yes' if status['supports_streaming'] else 'No'}")


def demonstrate_model_recommendations():
    """Demonstrate model recommendations by use case."""
    print("\n\n💡 Model Recommendations")
    print("=" * 50)
    
    use_cases = ["general", "fast", "cheap", "high-quality", "local", "coding"]
    
    for use_case in use_cases:
        print(f"\n{use_case.upper()}:")
        recommendations = get_model_recommendations(use_case)
        for provider, model in recommendations.items():
            print(f"   {provider}: {model}")


def demonstrate_environment_configuration():
    """Demonstrate how to use environment-based configuration."""
    print("\n\n⚙️ Environment Configuration")
    print("=" * 50)
    
    # Set up test environment
    os.environ.update({
        'LLM_PROVIDER': 'mock',
        'LLM_MODEL': 'test-model',
        'DATABASE_URL': 'postgresql://test:test@localhost:5432/test',
        'JWT_SECRET_KEY': 'test-secret-key-for-testing-only-32-chars',
        'ENVIRONMENT': 'test'
    })
    
    try:
        provider = get_llm_provider_from_config()
        print(f"✅ Loaded provider from environment: {provider.get_provider_name()}")
        print(f"   Model: {provider.get_model_name()}")
        
        # Test generation
        response = provider.generate("Test prompt for environment configuration")
        print(f"   Test response: {response[:50]}...")
        
    except Exception as e:
        print(f"❌ Failed to load from environment: {e}")


def demonstrate_compression_agent_integration():
    """Demonstrate integration with compression agent."""
    print("\n\n🗜️ Compression Agent Integration")
    print("=" * 50)
    
    try:
        from src.compression_agent import CompressionAgent, get_compression_agent
        
        # Create compression agent with mock provider
        config = LLMConfig(provider="mock", model="compression-model")
        provider = LLMProviderFactory.create_provider(config)
        agent = CompressionAgent(provider)
        
        print("✅ Compression agent created successfully")
        
        # Test compression
        test_content = """
        User: Hi, I'm looking for a laptop for programming. I need something with at least 16GB RAM, 
        good battery life, and a comfortable keyboard. My budget is around $1500. I also need it to 
        be portable since I travel frequently. I've been looking at MacBooks but they're expensive. 
        What would you recommend?
        
        Agent: Based on your requirements, I'd recommend the Dell XPS 13 or ThinkPad X1 Carbon. 
        Both offer excellent keyboards, good battery life, and are highly portable. The XPS 13 
        starts around $1200 with 16GB RAM, while the ThinkPad X1 Carbon is around $1400. Both 
        are great for programming and much more affordable than MacBooks.
        """
        
        result = agent.compress_memory(test_content, "mid_term")
        
        if result:
            print("✅ Compression successful")
            print(f"   Summary: {result.get('summary', 'N/A')[:100]}...")
            print(f"   Key Points: {result.get('key_points', [])}")
        else:
            print("❌ Compression failed")
            
    except Exception as e:
        print(f"❌ Compression agent integration failed: {e}")


def main():
    """Run all demonstrations."""
    print("🚀 LLM Providers Demonstration")
    print("=" * 60)
    
    demonstrate_provider_creation()
    demonstrate_text_generation()
    demonstrate_configuration_validation()
    demonstrate_provider_status()
    demonstrate_model_recommendations()
    demonstrate_environment_configuration()
    demonstrate_compression_agent_integration()
    
    print("\n\n✅ Demonstration completed!")
    print("\nTo use LLM providers in your code:")
    print("1. Set environment variables (see env.example)")
    print("2. Use get_llm_provider_from_config() for automatic configuration")
    print("3. Or create providers manually with LLMConfig and LLMProviderFactory")
    print("4. Use the discovery utility: python scripts/llm_discovery.py --help")


if __name__ == "__main__":
    main()
