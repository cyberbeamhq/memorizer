"""
config.py
Configuration validation and management for the Memorizer framework.
Ensures all required settings are present and valid.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""

    url: str
    min_connections: int = 1
    max_connections: int = 10


@dataclass
class SecurityConfig:
    """Security configuration."""

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    enable_audit_logging: bool = True
    audit_log_file: str = "logs/audit.log"


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""

    provider: str
    model: str
    api_key: Optional[str] = None
    max_retries: int = 3
    timeout: int = 30


@dataclass
class VectorDBConfig:
    """Vector database configuration."""

    provider: str
    api_key: Optional[str] = None
    url: Optional[str] = None
    index_name: Optional[str] = None


@dataclass
class LLMConfig:
    """LLM configuration for compression and generation."""

    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_retries: int = 3
    timeout: int = 30
    temperature: float = 0.7
    max_tokens: Optional[int] = None


@dataclass
class AppConfig:
    """Application configuration."""

    environment: str
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4


class ConfigValidator:
    """Validates configuration settings."""

    @staticmethod
    def validate_database_url(url: str) -> bool:
        """Validate database URL format."""
        if not url:
            return False

        valid_prefixes = ("postgresql://", "postgres://")
        if not url.startswith(valid_prefixes):
            return False

        # Basic format validation
        try:
            # Check if it has the basic structure
            if "@" not in url or "://" not in url:
                return False
            return True
        except:
            return False

    @staticmethod
    def validate_jwt_secret(secret: str) -> bool:
        """Validate JWT secret key."""
        if not secret:
            return False

        # JWT secret should be at least 32 characters for security
        if len(secret) < 32:
            return False

        return True

    @staticmethod
    def validate_embedding_provider(provider: str) -> bool:
        """Validate embedding provider."""
        valid_providers = ["openai", "cohere", "huggingface", "mock"]
        return provider.lower() in valid_providers

    @staticmethod
    def validate_llm_provider(provider: str) -> bool:
        """Validate LLM provider."""
        valid_providers = ["openai", "anthropic", "groq", "openrouter", "ollama", "custom", "mock"]
        return provider.lower() in valid_providers

    @staticmethod
    def validate_vector_db_provider(provider: str) -> bool:
        """Validate vector database provider."""
        valid_providers = ["mock", "pinecone", "weaviate", "chroma", "pgvector"]
        return provider.lower() in valid_providers

    @staticmethod
    def validate_environment(env: str) -> bool:
        """Validate environment setting."""
        valid_environments = ["development", "staging", "production", "test"]
        return env.lower() in valid_environments


class ConfigManager:
    """Manages application configuration."""

    def __init__(self):
        self.config = {}
        self._load_and_validate()

    def _load_and_validate(self):
        """Load and validate configuration."""
        # Load environment variables
        load_dotenv()

        # Validate required settings
        self._validate_required_settings()

        # Load and validate configurations
        self.config = {
            "database": self._load_database_config(),
            "security": self._load_security_config(),
            "embedding": self._load_embedding_config(),
            "vector_db": self._load_vector_db_config(),
            "llm": self._load_llm_config(),
            "app": self._load_app_config(),
        }

        logger.info("Configuration loaded and validated successfully")

    def _validate_required_settings(self):
        """Validate that all required settings are present."""
        required_settings = {
            "DATABASE_URL": "Database connection string",
            "JWT_SECRET_KEY": "JWT secret key for authentication",
            "EMBEDDING_PROVIDER": "Embedding provider (openai, cohere, huggingface, mock)",
            "VECTOR_DB_PROVIDER": "Vector database provider",
            "ENVIRONMENT": "Application environment (development, staging, production)",
        }

        missing_settings = []
        for setting, description in required_settings.items():
            if not os.getenv(setting):
                missing_settings.append(f"{setting}: {description}")

        if missing_settings:
            error_msg = "Missing required environment variables:\n" + "\n".join(
                missing_settings
            )
            raise ValueError(error_msg)

    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration."""
        url = os.getenv("DATABASE_URL")

        if not ConfigValidator.validate_database_url(url):
            raise ValueError("Invalid DATABASE_URL format")

        return DatabaseConfig(
            url=url,
            min_connections=int(os.getenv("DB_MIN_CONNECTIONS", "1")),
            max_connections=int(os.getenv("DB_MAX_CONNECTIONS", "10")),
        )

    def _load_security_config(self) -> SecurityConfig:
        """Load security configuration."""
        jwt_secret = os.getenv("JWT_SECRET_KEY")

        if not ConfigValidator.validate_jwt_secret(jwt_secret):
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters long")

        return SecurityConfig(
            jwt_secret_key=jwt_secret,
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            jwt_expire_minutes=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")),
            enable_audit_logging=os.getenv("ENABLE_AUDIT_LOGGING", "true").lower()
            == "true",
            audit_log_file=os.getenv("AUDIT_LOG_FILE", "logs/audit.log"),
        )

    def _load_embedding_config(self) -> EmbeddingConfig:
        """Load embedding configuration."""
        provider = os.getenv("EMBEDDING_PROVIDER", "mock").lower()

        if not ConfigValidator.validate_embedding_provider(provider):
            raise ValueError(f"Invalid embedding provider: {provider}")

        # Get provider-specific settings
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        elif provider == "cohere":
            api_key = os.getenv("COHERE_API_KEY")
            model = os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
        elif provider == "huggingface":
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            model = os.getenv(
                "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
        else:  # mock
            api_key = None
            model = "mock"

        # Validate API key for non-mock providers
        if provider != "mock" and not api_key:
            raise ValueError(f"API key required for {provider} embedding provider")

        return EmbeddingConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            max_retries=int(os.getenv("COMPRESSION_MAX_RETRIES", "3")),
            timeout=int(os.getenv("EMBEDDING_TIMEOUT", "30")),
        )

    def _load_vector_db_config(self) -> VectorDBConfig:
        """Load vector database configuration."""
        provider = os.getenv("VECTOR_DB_PROVIDER", "mock").lower()

        if not ConfigValidator.validate_vector_db_provider(provider):
            raise ValueError(f"Invalid vector database provider: {provider}")

        # Get provider-specific settings
        if provider == "pinecone":
            api_key = os.getenv("PINECONE_API_KEY")
            url = os.getenv("PINECONE_ENVIRONMENT")
            index_name = os.getenv("PINECONE_INDEX_NAME", "memorizer")
        elif provider == "weaviate":
            api_key = os.getenv("WEAVIATE_API_KEY")
            url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
            index_name = os.getenv("WEAVIATE_CLASS_NAME", "Memory")
        elif provider == "chroma":
            api_key = None
            url = f"{os.getenv('CHROMA_HOST', 'localhost')}:{os.getenv('CHROMA_PORT', '8000')}"
            index_name = os.getenv("CHROMA_COLLECTION_NAME", "memories")
        else:  # mock or pgvector
            api_key = None
            url = None
            index_name = None

        # Validate API key for providers that require it
        if provider in ["pinecone", "weaviate"] and not api_key:
            raise ValueError(f"API key required for {provider} vector database")

        return VectorDBConfig(
            provider=provider, api_key=api_key, url=url, index_name=index_name
        )

    def _load_llm_config(self) -> LLMConfig:
        """Load LLM configuration."""
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        if not ConfigValidator.validate_llm_provider(provider):
            raise ValueError(f"Invalid LLM provider: {provider}")

        # Get provider-specific settings
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            base_url = None
        elif provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            base_url = None
        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        elif provider == "ollama":
            api_key = None
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        elif provider == "custom":
            api_key = os.getenv("CUSTOM_MODEL_API_KEY")
            base_url = os.getenv("CUSTOM_MODEL_BASE_URL")
        else:  # mock
            api_key = None
            base_url = None

        # Validate API key for providers that require it
        if provider in ["openai", "anthropic", "groq", "openrouter", "custom"] and not api_key:
            logger.warning(f"API key not provided for {provider} LLM provider")

        return LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            timeout=int(os.getenv("LLM_TIMEOUT", "30")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")) if os.getenv("LLM_MAX_TOKENS") else None,
        )

    def _load_app_config(self) -> AppConfig:
        """Load application configuration."""
        environment = os.getenv("ENVIRONMENT", "development").lower()

        if not ConfigValidator.validate_environment(environment):
            raise ValueError(f"Invalid environment: {environment}")

        return AppConfig(
            environment=environment,
            debug=os.getenv("DEBUG", "false").lower() == "true",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            workers=int(os.getenv("API_WORKERS", "4")),
        )

    def get_config(self, section: str) -> Dict[str, Any]:
        """Get configuration section."""
        return self.config.get(section, {})

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self.config["database"]

    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        return self.config["security"]

    def get_embedding_config(self) -> EmbeddingConfig:
        """Get embedding configuration."""
        return self.config["embedding"]

    def get_vector_db_config(self) -> VectorDBConfig:
        """Get vector database configuration."""
        return self.config["vector_db"]

    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        return self.config["llm"]

    def get_app_config(self) -> AppConfig:
        """Get application configuration."""
        return self.config["app"]

    def validate_production_ready(self) -> List[str]:
        """Validate that configuration is production-ready."""
        issues = []

        app_config = self.get_app_config()
        security_config = self.get_security_config()

        # Check environment
        if app_config.environment == "production":
            if app_config.debug:
                issues.append("Debug mode should be disabled in production")

            # Check for default JWT secret
            if (
                security_config.jwt_secret_key
                == "your_super_secret_jwt_key_change_this_in_production"
            ):
                issues.append("Default JWT secret key detected - change in production")

            # Check for development API keys
            if "dev_" in str(self.config):
                issues.append("Development API keys detected - change in production")

        return issues


# Global configuration instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def initialize_config():
    """Initialize global configuration manager."""
    global _config_manager
    _config_manager = ConfigManager()
    logger.info("Configuration manager initialized")
