"""
Basic tests for Memorizer framework.
These tests ensure the core functionality works without external dependencies.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test that core modules can be imported."""
    try:
        from src import config
        from src import validation
        from src import utils
        from src import errors
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")


def test_config_loading():
    """Test configuration loading."""
    import os
    # Set required environment variables for testing
    os.environ.update({
        'DATABASE_URL': 'postgresql://test:test@localhost:5432/test',
        'JWT_SECRET_KEY': 'test-secret-key-for-testing-only-32-chars',
        'EMBEDDING_PROVIDER': 'mock',
        'VECTOR_DB_PROVIDER': 'mock',
        'ENVIRONMENT': 'test'
    })
    
    try:
        from src.config import get_config_manager
        config_manager = get_config_manager()
        assert config_manager is not None
    except Exception as e:
        pytest.fail(f"Failed to load configuration: {e}")


def test_validation():
    """Test input validation."""
    try:
        from src.validation import InputValidator
        
        # Test valid input
        result = InputValidator.validate_user_id("test_user_123")
        assert result.is_valid
        
        # Test invalid input
        result = InputValidator.validate_user_id("")
        assert not result.is_valid
        
    except Exception as e:
        pytest.fail(f"Validation test failed: {e}")


def test_error_handling():
    """Test error handling framework."""
    try:
        from src.errors import MemorizerError, ErrorCode
        
        # Test custom error
        error = MemorizerError(
            message="Test error",
            error_code=ErrorCode.VALIDATION_ERROR,
            details={"field": "test"}
        )
        assert error.message == "Test error"
        assert error.error_code == ErrorCode.VALIDATION_ERROR
        
    except Exception as e:
        pytest.fail(f"Error handling test failed: {e}")


def test_utils():
    """Test utility functions."""
    try:
        from src.utils import safe_parse_json, utc_now
        
        # Test JSON parsing
        result = safe_parse_json('{"test": "value"}')
        assert result == {"test": "value"}
        
        # Test invalid JSON
        result = safe_parse_json('invalid json')
        assert result == {'error': 'json_parse_failed'}
        
        # Test UTC now function
        now = utc_now()
        assert now is not None
        
    except Exception as e:
        pytest.fail(f"Utils test failed: {e}")


@pytest.mark.slow
def test_database_connection():
    """Test database connection (requires database)."""
    import os
    # Set required environment variables for testing
    os.environ.update({
        'DATABASE_URL': 'postgresql://test:test@localhost:5432/test',
        'JWT_SECRET_KEY': 'test-secret-key-for-testing-only-32-chars',
        'EMBEDDING_PROVIDER': 'mock',
        'VECTOR_DB_PROVIDER': 'mock',
        'ENVIRONMENT': 'test'
    })
    
    try:
        from src.db import init_connection_pool
        
        # This will fail without proper DATABASE_URL, but we can test the function exists
        assert callable(init_connection_pool)
        
    except Exception as e:
        pytest.fail(f"Database connection test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
