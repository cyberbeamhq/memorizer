"""
Memorizer Framework Configuration
Configuration management for the Memorizer framework.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ComponentConfig(BaseModel):
    """Configuration for a framework component."""
    
    type: str
    name: str
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class MemoryLifecycleConfig(BaseModel):
    """Configuration for memory lifecycle management."""
    
    enabled: bool = True
    compression_threshold: float = 0.7
    cleanup_interval_seconds: int = 3600
    tiers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class SecurityConfig(BaseModel):
    """Configuration for security settings."""
    
    jwt_secret: str = ""
    api_key_hash_salt: str = ""
    admin_api_key: str = ""
    user_api_key: str = ""


class PerformanceConfig(BaseModel):
    """Configuration for performance settings."""
    
    embedding_cache_size: int = 1000
    embedding_cache_ttl_seconds: int = 300
    rate_limit_per_minute: int = 60


class FrameworkConfig(BaseModel):
    """Main framework configuration."""

    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # Component configurations
    summarizer: ComponentConfig = Field(default_factory=lambda: ComponentConfig(
        type="summarizer",
        name="mock",
        config={}
    ))
    retriever: ComponentConfig = Field(default_factory=lambda: ComponentConfig(
        type="retriever",
        name="keyword",
        config={}
    ))
    storage: ComponentConfig = Field(default_factory=lambda: ComponentConfig(
        type="storage",
        name="memory",
        config={}
    ))
    pii_filter: ComponentConfig = Field(default_factory=lambda: ComponentConfig(
        type="pii_filter",
        name="basic",
        config={}
    ))
    scorer: ComponentConfig = Field(default_factory=lambda: ComponentConfig(
        type="scorer",
        name="simple",
        config={}
    ))
    task_runner: ComponentConfig = Field(default_factory=lambda: ComponentConfig(
        type="task_runner",
        name="thread",
        config={}
    ))
    embedding_provider: ComponentConfig = Field(default_factory=lambda: ComponentConfig(
        type="embedding_provider",
        name="mock",
        config={}
    ))
    vector_store: ComponentConfig = Field(default_factory=lambda: ComponentConfig(
        type="vector_store",
        name="memory",
        config={}
    ))
    cache: ComponentConfig = Field(default_factory=lambda: ComponentConfig(
        type="cache",
        name="memory",
        config={}
    ))

    # Additional configurations
    memory_lifecycle: MemoryLifecycleConfig = Field(default_factory=MemoryLifecycleConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    @classmethod
    def create_default(cls) -> "FrameworkConfig":
        """Create a default configuration for the framework."""
        return cls()


class ConfigLoader:
    """Configuration loader with support for multiple formats."""
    
    @staticmethod
    def load_from_file(file_path: Union[str, Path]) -> FrameworkConfig:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            FrameworkConfig object
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
        
        return ConfigLoader._dict_to_config(config_data)
    
    @staticmethod
    def _dict_to_config(data: Dict[str, Any]) -> FrameworkConfig:
        """Convert dictionary to FrameworkConfig object."""
        # Create base config
        config = FrameworkConfig()
        
        # Update basic settings
        if "version" in data:
            config.version = data["version"]
        if "debug" in data:
            config.debug = data["debug"]
        if "log_level" in data:
            config.log_level = data["log_level"]
        
        # Update component configurations
        component_fields = [
            "summarizer", "retriever", "storage", "pii_filter", "scorer",
            "task_runner", "embedding_provider", "vector_store", "cache"
        ]
        
        for field_name in component_fields:
            if field_name in data:
                component_data = data[field_name]
                component_config = ComponentConfig(
                    type=component_data.get("type", field_name),
                    name=component_data.get("name", "default"),
                    config=component_data.get("config", {}),
                    enabled=component_data.get("enabled", True)
                )
                setattr(config, field_name, component_config)
        
        # Update additional configurations
        if "memory_lifecycle" in data:
            config.memory_lifecycle.update(data["memory_lifecycle"])
        if "security" in data:
            config.security.update(data["security"])
        if "performance" in data:
            config.performance.update(data["performance"])
        
        return config


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    env_prefix: str = "MEMORIZER"
) -> FrameworkConfig:
    """
    Load configuration from file or environment variables.
    
    Args:
        config_path: Path to configuration file
        env_prefix: Environment variable prefix
        
    Returns:
        FrameworkConfig object
    """
    if config_path:
        try:
            return ConfigLoader.load_from_file(config_path)
        except FileNotFoundError:
            print(f"Configuration file not found: {config_path}")
            print("Using default configuration")
    
    # Load from environment variables
    config = FrameworkConfig()
    
    # Load basic settings from environment
    config.debug = os.getenv(f"{env_prefix}_DEBUG", "false").lower() == "true"
    config.log_level = os.getenv(f"{env_prefix}_LOG_LEVEL", "INFO")
    
    # Load component configurations from environment
    components = [
        "summarizer", "retriever", "storage", "pii_filter", "scorer",
        "task_runner", "embedding_provider", "vector_store", "cache"
    ]
    
    for component in components:
        env_name = f"{env_prefix}_{component.upper()}_TYPE"
        component_type = os.getenv(env_name)
        if component_type:
            getattr(config, component).type = component_type
        
        env_name = f"{env_prefix}_{component.upper()}_NAME"
        component_name = os.getenv(env_name)
        if component_name:
            getattr(config, component).name = component_name
    
    # Load security settings
    config.security.jwt_secret = os.getenv(f"{env_prefix}_JWT_SECRET", "")
    config.security.api_key_hash_salt = os.getenv(f"{env_prefix}_API_KEY_HASH_SALT", "")
    config.security.admin_api_key = os.getenv(f"{env_prefix}_ADMIN_API_KEY", "")
    config.security.user_api_key = os.getenv(f"{env_prefix}_USER_API_KEY", "")

    return config


def create_default_config_file(file_path: Union[str, Path], format: str = "yaml") -> None:
    """
    Create a default configuration file.

    Args:
        file_path: Path where to create the configuration file
        format: Configuration format ('yaml' or 'json')
    """
    file_path = Path(file_path)

    # Create default configuration
    config = FrameworkConfig.create_default()

    # Convert to dictionary
    config_dict = {
        "version": config.version,
        "debug": config.debug,
        "log_level": config.log_level,
        "summarizer": {
            "type": config.summarizer.type,
            "name": config.summarizer.name,
            "config": config.summarizer.config,
            "enabled": config.summarizer.enabled
        },
        "retriever": {
            "type": config.retriever.type,
            "name": config.retriever.name,
            "config": config.retriever.config,
            "enabled": config.retriever.enabled
        },
        "storage": {
            "type": config.storage.type,
            "name": config.storage.name,
            "config": config.storage.config,
            "enabled": config.storage.enabled
        },
        "pii_filter": {
            "type": config.pii_filter.type,
            "name": config.pii_filter.name,
            "config": config.pii_filter.config,
            "enabled": config.pii_filter.enabled
        },
        "scorer": {
            "type": config.scorer.type,
            "name": config.scorer.name,
            "config": config.scorer.config,
            "enabled": config.scorer.enabled
        },
        "task_runner": {
            "type": config.task_runner.type,
            "name": config.task_runner.name,
            "config": config.task_runner.config,
            "enabled": config.task_runner.enabled
        },
        "embedding_provider": {
            "type": config.embedding_provider.type,
            "name": config.embedding_provider.name,
            "config": config.embedding_provider.config,
            "enabled": config.embedding_provider.enabled
        },
        "vector_store": {
            "type": config.vector_store.type,
            "name": config.vector_store.name,
            "config": config.vector_store.config,
            "enabled": config.vector_store.enabled
        },
        "cache": {
            "type": config.cache.type,
            "name": config.cache.name,
            "config": config.cache.config,
            "enabled": config.cache.enabled
        },
        "memory_lifecycle": {
            "enabled": config.memory_lifecycle.enabled,
            "compression_threshold": config.memory_lifecycle.compression_threshold,
            "cleanup_interval_seconds": config.memory_lifecycle.cleanup_interval_seconds,
            "tiers": config.memory_lifecycle.tiers
        },
        "security": {
            "jwt_secret": config.security.jwt_secret,
            "api_key_hash_salt": config.security.api_key_hash_salt,
            "admin_api_key": config.security.admin_api_key,
            "user_api_key": config.security.user_api_key
        },
        "performance": {
            "embedding_cache_size": config.performance.embedding_cache_size,
            "embedding_cache_ttl_seconds": config.performance.embedding_cache_ttl_seconds,
            "rate_limit_per_minute": config.performance.rate_limit_per_minute
        }
    }

    # Write to file
    with open(file_path, 'w') as f:
        if format.lower() == "yaml":
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:  # json
            json.dump(config_dict, f, indent=2)

    print(f"Default configuration file created: {file_path}")
