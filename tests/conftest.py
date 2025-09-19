"""
Pytest configuration and fixtures for Memorizer tests.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set default environment variables for testing
os.environ.update({
    'DATABASE_URL': 'postgresql://test:test@localhost:5432/test',
    'JWT_SECRET_KEY': 'test-secret-key-for-testing-only-32-chars',
    'EMBEDDING_PROVIDER': 'mock',
    'VECTOR_DB_PROVIDER': 'mock',
    'ENVIRONMENT': 'test',
    'REDIS_URL': 'redis://localhost:6379/0',
    'OPENAI_API_KEY': 'test-key'
})


@pytest.fixture
def mock_database():
    """Mock database connection for tests."""
    with patch('src.db.init_connection_pool') as mock_db:
        mock_db.return_value = Mock()
        yield mock_db


@pytest.fixture
def mock_redis():
    """Mock Redis connection for tests."""
    with patch('src.cache.CacheManager') as mock_redis:
        mock_redis.return_value = Mock()
        yield mock_redis


@pytest.fixture
def mock_openai():
    """Mock OpenAI API for tests."""
    with patch('src.embeddings.openai') as mock_openai:
        mock_openai.Embedding.create.return_value = {
            'data': [{'embedding': [0.1] * 1536}]
        }
        yield mock_openai


@pytest.fixture
def sample_memory_data():
    """Sample memory data for testing."""
    return {
        "user_id": "test_user",
        "content": "This is a test memory",
        "metadata": {"type": "test", "importance": 0.8},
        "session_id": "test_session_123"
    }


@pytest.fixture
def sample_query_data():
    """Sample query data for testing."""
    return {
        "user_id": "test_user",
        "query": "test query",
        "limit": 10,
        "tier": "very_new"
    }


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit marker to tests that don't have any marker
        if not any(marker.name in ['slow', 'integration', 'unit'] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
