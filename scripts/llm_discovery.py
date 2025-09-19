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

try:
    from llm_providers import (
        LLMProviderFactory, 
        list_available_models, 
        validate_model_for_provider,
        get_model_recommendations,
        validate_provider_config,
        get_provider_status
    )
except ImportError as e:
    print(f"❌ Failed to import llm_providers: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)


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
        try:
            from llm_providers import LLMConfig, LLMProviderFactory
        except ImportError as e:
            print(f"❌ Failed to import LLMConfig: {e}")
            return
        
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


def validate_config(provider_name: str, model_name: str, api_key: str = None):
    """Validate complete provider configuration."""
    print(f"🔍 Validating {provider_name} configuration...")
    
    result = validate_provider_config(provider_name, model_name, api_key)
    
    if result["valid"]:
        print(f"✅ Configuration is valid")
    else:
        print(f"❌ Configuration has errors:")
        for error in result["errors"]:
            print(f"   • {error}")
    
    if result["warnings"]:
        print(f"⚠️ Warnings:")
        for warning in result["warnings"]:
            print(f"   • {warning}")
    
    if result["provider_info"]:
        info = result["provider_info"]
        print(f"\n📋 Provider Info:")
        print(f"   Name: {info['name']}")
        print(f"   Model Format: {info['model_format']}")
        print(f"   Requires API Key: {'Yes' if info['requires_api_key'] else 'No'}")
        print(f"   Supports Streaming: {'Yes' if info['supports_streaming'] else 'No'}")


def check_status(provider_name: str):
    """Check the status of a provider."""
    print(f"🔍 Checking status of {provider_name}...")
    
    status = get_provider_status(provider_name)
    
    if "error" in status:
        print(f"❌ {status['error']}")
        return
    
    print(f"📊 Provider Status:")
    print(f"   Name: {status['name']}")
    print(f"   Status: {status['status']}")
    print(f"   Available Models: {status['available_models']}")
    print(f"   Model Format: {status['model_format']}")
    print(f"   Requires API Key: {'Yes' if status['requires_api_key'] else 'No'}")
    print(f"   Supports Streaming: {'Yes' if status['supports_streaming'] else 'No'}")
    
    if status.get("note"):
        print(f"   Note: {status['note']}")
    
    if "error" in status:
        print(f"   Error: {status['error']}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="LLM Discovery Utility")
    parser.add_argument("command", choices=["list", "info", "recommend", "test", "validate", "validate-config", "status"], 
                       help="Command to execute")
    parser.add_argument("provider", nargs="?", help="Provider name")
    parser.add_argument("model", nargs="?", help="Model name")
    parser.add_argument("--api-key", help="API key for validation")
    
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
    
    elif args.command == "validate-config":
        if not args.provider or not args.model:
            print("❌ Both provider and model names required for 'validate-config' command")
            return
        validate_config(args.provider, args.model, args.api_key)
    
    elif args.command == "status":
        if not args.provider:
            print("❌ Provider name required for 'status' command")
            return
        check_status(args.provider)


if __name__ == "__main__":
    main()
