#!/usr/bin/env python3
"""
LLM Discovery Utility
Helps users discover available models and test LLM providers.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_providers import (
    LLMProviderFactory, 
    list_available_models, 
    validate_model_for_provider,
    get_model_recommendations
)


def print_provider_info(provider_name: str):
    """Print detailed information about a provider."""
    info = LLMProviderFactory.get_provider_info(provider_name)
    if not info:
        print(f"❌ Unknown provider: {provider_name}")
        return
    
    print(f"\n🔌 {info['name']} ({provider_name})")
    print(f"   Description: {info['description']}")
    print(f"   Model Format: {info['model_format']}")
    print(f"   Requires API Key: {'Yes' if info['requires_api_key'] else 'No'}")
    print(f"   Supports Streaming: {'Yes' if info['supports_streaming'] else 'No'}")
    
    if 'note' in info:
        print(f"   Note: {info['note']}")
    
    print(f"\n   📋 Available Models:")
    for model in info['models'][:10]:  # Show first 10 models
        print(f"      • {model}")
    
    if len(info['models']) > 10:
        print(f"      ... and {len(info['models']) - 10} more models")
    
    print()


def list_all_providers():
    """List all available providers."""
    print("🚀 Available LLM Providers:")
    print("=" * 50)
    
    providers = LLMProviderFactory.get_available_providers()
    for provider in providers:
        info = LLMProviderFactory.get_provider_info(provider)
        print(f"• {provider}: {info['description']}")
    
    print()


def show_recommendations():
    """Show model recommendations for different use cases."""
    print("💡 Model Recommendations by Use Case:")
    print("=" * 50)
    
    use_cases = ["general", "fast", "cheap", "high-quality", "local", "coding"]
    
    for use_case in use_cases:
        print(f"\n🎯 {use_case.upper()}:")
        recommendations = get_model_recommendations(use_case)
        for provider, model in recommendations.items():
            print(f"   {provider}: {model}")


def test_provider(provider_name: str, model_name: str = None):
    """Test a provider with a specific model."""
    print(f"🧪 Testing {provider_name} provider...")
    
    try:
        # Set up test environment
        os.environ.update({
            'LLM_PROVIDER': provider_name,
            'LLM_MODEL': model_name or 'test-model',
            'DATABASE_URL': 'postgresql://test:test@localhost:5432/test',
            'JWT_SECRET_KEY': 'test-secret-key-for-testing-only-32-chars',
            'ENVIRONMENT': 'test'
        })
        
        # Create provider
        from llm_providers import LLMConfig, LLMProviderFactory
        
        config = LLMConfig(
            provider=provider_name,
            model=model_name or 'test-model',
            api_key="test-key" if provider_name not in ["mock", "ollama"] else None
        )
        
        provider = LLMProviderFactory.create_provider(config)
        print(f"✅ Provider created successfully")
        print(f"   Provider: {provider.get_provider_name()}")
        print(f"   Model: {provider.get_model_name()}")
        
        # Test generation (will fail for real providers without API keys)
        test_prompt = "Hello, this is a test message."
        try:
            response = provider.generate(test_prompt, max_tokens=50)
            if response:
                print(f"✅ Generation test passed")
                print(f"   Response: {response[:100]}...")
            else:
                print(f"⚠️ No response generated (may need API key)")
        except Exception as e:
            if "API key" in str(e) or "authentication" in str(e).lower():
                print(f"✅ Provider works (API key required for real usage)")
            else:
                print(f"❌ Generation test failed: {e}")
        
    except Exception as e:
        print(f"❌ Provider test failed: {e}")


def validate_model(provider_name: str, model_name: str):
    """Validate if a model is available for a provider."""
    is_valid = validate_model_for_provider(provider_name, model_name)
    
    if is_valid:
        print(f"✅ Model '{model_name}' is valid for {provider_name}")
    else:
        print(f"❌ Model '{model_name}' is not valid for {provider_name}")
        print(f"   Use 'python llm_discovery.py list {provider_name}' to see available models")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="LLM Discovery Utility")
    parser.add_argument("command", choices=["list", "info", "recommend", "test", "validate"], 
                       help="Command to execute")
    parser.add_argument("provider", nargs="?", help="Provider name")
    parser.add_argument("model", nargs="?", help="Model name")
    
    args = parser.parse_args()
    
    if args.command == "list":
        if args.provider:
            print_provider_info(args.provider)
        else:
            list_all_providers()
    
    elif args.command == "info":
        if not args.provider:
            print("❌ Provider name required for 'info' command")
            return
        print_provider_info(args.provider)
    
    elif args.command == "recommend":
        show_recommendations()
    
    elif args.command == "test":
        if not args.provider:
            print("❌ Provider name required for 'test' command")
            return
        test_provider(args.provider, args.model)
    
    elif args.command == "validate":
        if not args.provider or not args.model:
            print("❌ Both provider and model names required for 'validate' command")
            return
        validate_model(args.provider, args.model)


if __name__ == "__main__":
    main()
