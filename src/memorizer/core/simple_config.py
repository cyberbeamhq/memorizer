"""
Memory Configuration
Focus on core memory management configuration without infrastructure complexity.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class MemoryConfig:
    """Configuration for memory management."""

    # Core providers
    storage_provider: str = "memory"  # memory, supabase, railway, neon, postgres
    vector_store: str = "memory"      # memory, sqlite, pinecone, weaviate, chroma
    llm_provider: str = "mock"        # openai, anthropic, groq, mock

    # Memory lifecycle settings
    very_new_ttl_days: int = 7
    mid_term_ttl_days: int = 30
    long_term_ttl_days: int = 365
    compression_threshold: float = 0.8

    # Storage settings
    storage_config: Dict[str, Any] = field(default_factory=dict)
    vector_config: Dict[str, Any] = field(default_factory=dict)
    llm_config: Dict[str, Any] = field(default_factory=dict)

    # External provider settings
    supabase_url: Optional[str] = None
    supabase_password: Optional[str] = None
    supabase_anon_key: Optional[str] = None

    railway_database_url: Optional[str] = None
    neon_database_url: Optional[str] = None

    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "memorizer"

    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None

    def __post_init__(self):
        """Auto-configure from environment variables."""
        # Load from environment if not set
        if not self.supabase_url:
            self.supabase_url = os.getenv("SUPABASE_URL")
        if not self.supabase_password:
            self.supabase_password = os.getenv("SUPABASE_DB_PASSWORD")
        if not self.supabase_anon_key:
            self.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")

        if not self.railway_database_url:
            self.railway_database_url = os.getenv("RAILWAY_DATABASE_URL")
        if not self.neon_database_url:
            self.neon_database_url = os.getenv("NEON_DATABASE_URL")

        if not self.pinecone_api_key:
            self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_environment:
            self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

        if not self.weaviate_url:
            self.weaviate_url = os.getenv("WEAVIATE_URL")
        if not self.weaviate_api_key:
            self.weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.groq_api_key:
            self.groq_api_key = os.getenv("GROQ_API_KEY")

    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage-specific configuration."""
        base_config = self.storage_config.copy()

        if self.storage_provider == "supabase":
            base_config.update({
                "project_url": self.supabase_url,
                "database_password": self.supabase_password,
                "anon_key": self.supabase_anon_key,
            })
        elif self.storage_provider == "railway":
            base_config.update({
                "database_url": self.railway_database_url,
            })
        elif self.storage_provider == "neon":
            base_config.update({
                "connection_string": self.neon_database_url,
            })

        return base_config

    def get_vector_config(self) -> Dict[str, Any]:
        """Get vector store-specific configuration."""
        base_config = self.vector_config.copy()

        if self.vector_store == "pinecone":
            base_config.update({
                "api_key": self.pinecone_api_key,
                "environment": self.pinecone_environment,
                "index_name": self.pinecone_index_name,
            })
        elif self.vector_store == "weaviate":
            base_config.update({
                "url": self.weaviate_url,
                "api_key": self.weaviate_api_key,
            })

        return base_config

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM-specific configuration."""
        base_config = self.llm_config.copy()

        if self.llm_provider == "openai":
            base_config.update({
                "api_key": self.openai_api_key,
                "model": base_config.get("model", "gpt-4o-mini"),
            })
        elif self.llm_provider == "anthropic":
            base_config.update({
                "api_key": self.anthropic_api_key,
                "model": base_config.get("model", "claude-3-haiku-20240307"),
            })
        elif self.llm_provider == "groq":
            base_config.update({
                "api_key": self.groq_api_key,
                "model": base_config.get("model", "llama3-8b-8192"),
            })

        return base_config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MemoryConfig":
        """Create config from dictionary."""
        # Filter only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "storage_provider": self.storage_provider,
            "vector_store": self.vector_store,
            "llm_provider": self.llm_provider,
            "very_new_ttl_days": self.very_new_ttl_days,
            "mid_term_ttl_days": self.mid_term_ttl_days,
            "long_term_ttl_days": self.long_term_ttl_days,
            "compression_threshold": self.compression_threshold,
            "storage_config": self.get_storage_config(),
            "vector_config": self.get_vector_config(),
            "llm_config": self.get_llm_config(),
        }


def create_default_config() -> MemoryConfig:
    """Create a default configuration for development."""
    return MemoryConfig(
        storage_provider="memory",
        vector_store="memory",
        llm_provider="mock"
    )


def create_supabase_config(
    supabase_url: str,
    supabase_password: str,
    vector_store: str = "memory",
    llm_provider: str = "openai"
) -> MemoryConfig:
    """Create configuration for Supabase storage."""
    return MemoryConfig(
        storage_provider="supabase",
        vector_store=vector_store,
        llm_provider=llm_provider,
        supabase_url=supabase_url,
        supabase_password=supabase_password
    )


def create_pinecone_config(
    pinecone_api_key: str,
    pinecone_environment: str,
    storage_provider: str = "memory",
    llm_provider: str = "openai"
) -> MemoryConfig:
    """Create configuration for Pinecone vector store."""
    return MemoryConfig(
        storage_provider=storage_provider,
        vector_store="pinecone",
        llm_provider=llm_provider,
        pinecone_api_key=pinecone_api_key,
        pinecone_environment=pinecone_environment
    )


def create_external_config(
    storage_provider: str,
    vector_store: str,
    llm_provider: str = "openai",
    **kwargs
) -> MemoryConfig:
    """Create configuration for external providers."""
    return MemoryConfig(
        storage_provider=storage_provider,
        vector_store=vector_store,
        llm_provider=llm_provider,
        **kwargs
    )


__all__ = [
    "MemoryConfig",
    "create_default_config",
    "create_supabase_config",
    "create_pinecone_config",
    "create_external_config",
]