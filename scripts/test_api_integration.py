#!/usr/bin/env python3
"""
Test API Integration
Test the integration between framework_api.py and the main framework.
"""

import os
import sys
import requests
import time
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_api_integration():
    """Test the API integration."""
    print("🚀 Testing API Integration")
    print("=" * 50)
    
    # Test 1: Direct framework creation
    print("\n1. Testing Direct Framework Creation...")
    try:
        from framework.factory import create_framework
        from framework.core.config import load_config
        
        # Load configuration
        config = load_config("memorizer.yaml")
        print(f"   ✅ Configuration loaded: {config.version}")
        
        # Create framework
        framework = create_framework(config)
        print(f"   ✅ Framework created successfully")
        
        # Test memory manager
        memory_manager = framework.get_memory_manager()
        print(f"   ✅ Memory manager available")
        
        # Test storing a memory
        memory_id = memory_manager.store_memory(
            user_id="test_user",
            content="Test memory for API integration",
            metadata={"test": True},
            tier="very_new"
        )
        print(f"   ✅ Memory stored: {memory_id}")
        
        # Test retrieving memory
        memory = memory_manager.get_memory(memory_id, "test_user")
        if memory:
            print(f"   ✅ Memory retrieved: {memory.content[:50]}...")
        else:
            print("   ❌ Failed to retrieve memory")
            
    except Exception as e:
        print(f"   ❌ Direct framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: API server startup
    print("\n2. Testing API Server Startup...")
    try:
        import subprocess
        import threading
        
        # Start the API server in a separate process
        def start_server():
            os.system("cd /Users/user/Desktop/memo-framework/memorizer && python -m uvicorn src.framework_api:app --host 0.0.0.0 --port 8000")
        
        # Start server in background
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        print("   ⏳ Waiting for server to start...")
        time.sleep(5)
        
        # Test API endpoints
        base_url = "http://localhost:8000"
        
        # Test root endpoint
        try:
            response = requests.get(f"{base_url}/", timeout=5)
            if response.status_code == 200:
                print(f"   ✅ Root endpoint working: {response.json()}")
            else:
                print(f"   ❌ Root endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Root endpoint error: {e}")
        
        # Test health endpoint
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"   ✅ Health endpoint working: {health_data.get('framework', {}).get('status', 'unknown')}")
            else:
                print(f"   ❌ Health endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Health endpoint error: {e}")
        
        # Test memory creation
        try:
            memory_data = {
                "user_id": "api_test_user",
                "content": "Test memory via API",
                "metadata": {"test": True, "api_test": True},
                "tier": "very_new"
            }
            response = requests.post(f"{base_url}/memories", json=memory_data, timeout=5)
            if response.status_code == 200:
                result = response.json()
                memory_id = result.get("memory_id")
                print(f"   ✅ Memory created via API: {memory_id}")
                
                # Test memory retrieval
                headers = {"X-User-ID": "api_test_user"}
                response = requests.get(f"{base_url}/memories/{memory_id}", headers=headers, timeout=5)
                if response.status_code == 200:
                    memory_data = response.json()
                    print(f"   ✅ Memory retrieved via API: {memory_data.get('content', '')[:50]}...")
                else:
                    print(f"   ❌ Memory retrieval failed: {response.status_code}")
            else:
                print(f"   ❌ Memory creation failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"   ❌ Memory API test error: {e}")
        
        # Test memory search
        try:
            search_data = {
                "user_id": "api_test_user",
                "query": "test memory",
                "limit": 10
            }
            response = requests.post(f"{base_url}/memories/search", json=search_data, timeout=5)
            if response.status_code == 200:
                search_results = response.json()
                print(f"   ✅ Memory search working: {len(search_results.get('memories', []))} results")
            else:
                print(f"   ❌ Memory search failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Memory search error: {e}")
        
    except Exception as e:
        print(f"   ❌ API server test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 API Integration Test Completed!")
    return True

if __name__ == "__main__":
    test_api_integration()
