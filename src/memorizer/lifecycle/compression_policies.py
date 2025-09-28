"""
Compression Policies Module
Advanced compression policies and management for memory optimization.
"""

import logging
import json
import gzip
import zlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CompressionAlgorithm(Enum):
    """Compression algorithm enumeration."""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    SEMANTIC = "semantic"
    ADAPTIVE = "adaptive"


class CompressionTrigger(Enum):
    """Compression trigger enumeration."""
    AGE_BASED = "age_based"
    SIZE_BASED = "size_based"
    ACCESS_BASED = "access_based"
    TIER_BASED = "tier_based"
    MANUAL = "manual"
    SCHEDULED = "scheduled"


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    success: bool
    original_size: int
    compressed_size: int
    compression_ratio: float
    algorithm_used: CompressionAlgorithm
    compression_time_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionPolicy:
    """Compression policy configuration."""
    name: str
    algorithm: CompressionAlgorithm
    trigger: CompressionTrigger
    enabled: bool = True
    priority: int = 0

    # Trigger conditions
    age_threshold_hours: Optional[int] = None
    size_threshold_bytes: Optional[int] = None
    access_threshold_count: Optional[int] = None
    tier_names: Optional[List[str]] = None

    # Algorithm parameters
    compression_level: int = 6
    chunk_size: int = 8192
    preserve_structure: bool = True

    # Quality controls
    min_compression_ratio: float = 0.1  # Minimum compression ratio to apply
    max_compression_time_ms: float = 5000  # Maximum time allowed for compression

    # Conditions
    content_type_filters: List[str] = field(default_factory=list)
    tag_filters: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class CompressionEngine(ABC):
    """Abstract base class for compression engines."""

    @abstractmethod
    def compress(self, data: str, **kwargs) -> CompressionResult:
        """Compress data using the engine's algorithm."""
        pass

    @abstractmethod
    def decompress(self, compressed_data: bytes, **kwargs) -> str:
        """Decompress data using the engine's algorithm."""
        pass

    @abstractmethod
    def estimate_compression_ratio(self, data: str) -> float:
        """Estimate compression ratio without actually compressing."""
        pass


class GzipCompressionEngine(CompressionEngine):
    """GZIP compression engine."""

    def compress(self, data: str, compression_level: int = 6, **kwargs) -> CompressionResult:
        """Compress data using GZIP."""
        start_time = datetime.now()
        original_size = len(data.encode('utf-8'))

        try:
            compressed_data = gzip.compress(data.encode('utf-8'), compresslevel=compression_level)
            compressed_size = len(compressed_data)
            compression_time = (datetime.now() - start_time).total_seconds() * 1000

            return CompressionResult(
                success=True,
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compressed_size / original_size,
                algorithm_used=CompressionAlgorithm.GZIP,
                compression_time_ms=compression_time,
                metadata={"compression_level": compression_level}
            )

        except Exception as e:
            return CompressionResult(
                success=False,
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=1.0,
                algorithm_used=CompressionAlgorithm.GZIP,
                compression_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                error_message=str(e)
            )

    def decompress(self, compressed_data: bytes, **kwargs) -> str:
        """Decompress GZIP data."""
        return gzip.decompress(compressed_data).decode('utf-8')

    def estimate_compression_ratio(self, data: str) -> float:
        """Estimate GZIP compression ratio."""
        # Simple heuristic based on data characteristics
        unique_chars = len(set(data))
        repetition_factor = len(data) / max(unique_chars, 1)

        if repetition_factor > 10:
            return 0.3  # High compression for repetitive data
        elif repetition_factor > 5:
            return 0.5  # Medium compression
        else:
            return 0.7  # Low compression for diverse data


class ZlibCompressionEngine(CompressionEngine):
    """ZLIB compression engine."""

    def compress(self, data: str, compression_level: int = 6, **kwargs) -> CompressionResult:
        """Compress data using ZLIB."""
        start_time = datetime.now()
        original_size = len(data.encode('utf-8'))

        try:
            compressed_data = zlib.compress(data.encode('utf-8'), level=compression_level)
            compressed_size = len(compressed_data)
            compression_time = (datetime.now() - start_time).total_seconds() * 1000

            return CompressionResult(
                success=True,
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compressed_size / original_size,
                algorithm_used=CompressionAlgorithm.ZLIB,
                compression_time_ms=compression_time,
                metadata={"compression_level": compression_level}
            )

        except Exception as e:
            return CompressionResult(
                success=False,
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=1.0,
                algorithm_used=CompressionAlgorithm.ZLIB,
                compression_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                error_message=str(e)
            )

    def decompress(self, compressed_data: bytes, **kwargs) -> str:
        """Decompress ZLIB data."""
        return zlib.decompress(compressed_data).decode('utf-8')

    def estimate_compression_ratio(self, data: str) -> float:
        """Estimate ZLIB compression ratio."""
        return max(0.4, 1.0 - (len(set(data)) / len(data)))


class SemanticCompressionEngine(CompressionEngine):
    """Semantic compression engine for text data."""

    def compress(self, data: str, **kwargs) -> CompressionResult:
        """Compress data using semantic techniques."""
        start_time = datetime.now()
        original_size = len(data.encode('utf-8'))

        try:
            # Simplified semantic compression
            # In production, this would use NLP techniques
            compressed_data = self._semantic_compress(data)
            compressed_size = len(compressed_data.encode('utf-8'))
            compression_time = (datetime.now() - start_time).total_seconds() * 1000

            return CompressionResult(
                success=True,
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compressed_size / original_size,
                algorithm_used=CompressionAlgorithm.SEMANTIC,
                compression_time_ms=compression_time,
                metadata={"semantic_features": self._extract_features(data)}
            )

        except Exception as e:
            return CompressionResult(
                success=False,
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=1.0,
                algorithm_used=CompressionAlgorithm.SEMANTIC,
                compression_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                error_message=str(e)
            )

    def decompress(self, compressed_data: bytes, **kwargs) -> str:
        """Decompress semantic data."""
        # In production, this would reconstruct from semantic representation
        return compressed_data.decode('utf-8')

    def estimate_compression_ratio(self, data: str) -> float:
        """Estimate semantic compression ratio."""
        # Estimate based on redundancy and structure
        word_count = len(data.split())
        unique_words = len(set(data.lower().split()))

        if word_count == 0:
            return 1.0

        redundancy = 1.0 - (unique_words / word_count)
        return 0.6 - (redundancy * 0.3)  # Higher redundancy = better compression

    def _semantic_compress(self, data: str) -> str:
        """Simplified semantic compression."""
        # Remove redundant words, abbreviate common phrases
        words = data.split()

        # Simple word frequency-based compression
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Replace frequent words with shorter tokens
        compressed_words = []
        for word in words:
            if word_freq[word] > 2 and len(word) > 4:
                # Use first 2 chars + frequency as token
                compressed_words.append(f"{word[:2]}{word_freq[word]}")
            else:
                compressed_words.append(word)

        return " ".join(compressed_words)

    def _extract_features(self, data: str) -> Dict[str, Any]:
        """Extract semantic features from data."""
        words = data.split()
        return {
            "word_count": len(words),
            "unique_words": len(set(words)),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "contains_code": any(char in data for char in "{}();"),
            "contains_json": data.strip().startswith(("{", "[")),
        }


class AdaptiveCompressionEngine(CompressionEngine):
    """Adaptive compression engine that chooses the best algorithm."""

    def __init__(self):
        """Initialize adaptive engine with available algorithms."""
        self.engines = {
            CompressionAlgorithm.GZIP: GzipCompressionEngine(),
            CompressionAlgorithm.ZLIB: ZlibCompressionEngine(),
            CompressionAlgorithm.SEMANTIC: SemanticCompressionEngine(),
        }

    def compress(self, data: str, **kwargs) -> CompressionResult:
        """Compress using the best available algorithm."""
        best_result = None
        best_ratio = 1.0

        # Test different algorithms and choose the best
        for algorithm, engine in self.engines.items():
            estimated_ratio = engine.estimate_compression_ratio(data)

            # Skip algorithms with poor estimated performance
            if estimated_ratio > 0.8:
                continue

            result = engine.compress(data, **kwargs)

            if result.success and result.compression_ratio < best_ratio:
                best_ratio = result.compression_ratio
                best_result = result

        if best_result:
            best_result.algorithm_used = CompressionAlgorithm.ADAPTIVE
            best_result.metadata["adaptive_choice"] = best_result.algorithm_used.value
            return best_result

        # Fallback to no compression
        return CompressionResult(
            success=True,
            original_size=len(data.encode('utf-8')),
            compressed_size=len(data.encode('utf-8')),
            compression_ratio=1.0,
            algorithm_used=CompressionAlgorithm.NONE,
            compression_time_ms=0.0
        )

    def decompress(self, compressed_data: bytes, algorithm_hint: CompressionAlgorithm = None, **kwargs) -> str:
        """Decompress using the specified algorithm."""
        if algorithm_hint and algorithm_hint in self.engines:
            return self.engines[algorithm_hint].decompress(compressed_data, **kwargs)

        # Try to auto-detect algorithm (simplified)
        try:
            return self.engines[CompressionAlgorithm.GZIP].decompress(compressed_data, **kwargs)
        except:
            try:
                return self.engines[CompressionAlgorithm.ZLIB].decompress(compressed_data, **kwargs)
            except:
                return compressed_data.decode('utf-8')

    def estimate_compression_ratio(self, data: str) -> float:
        """Estimate the best possible compression ratio."""
        best_ratio = 1.0

        for engine in self.engines.values():
            ratio = engine.estimate_compression_ratio(data)
            if ratio < best_ratio:
                best_ratio = ratio

        return best_ratio


class CompressionPolicyManager:
    """Manages compression policies and executes compression operations."""

    def __init__(self):
        """Initialize compression policy manager."""
        self.policies: List[CompressionPolicy] = []
        self.engines = {
            CompressionAlgorithm.GZIP: GzipCompressionEngine(),
            CompressionAlgorithm.ZLIB: ZlibCompressionEngine(),
            CompressionAlgorithm.SEMANTIC: SemanticCompressionEngine(),
            CompressionAlgorithm.ADAPTIVE: AdaptiveCompressionEngine(),
        }

        # Create default policies
        self._create_default_policies()

        logger.info("Compression policy manager initialized")

    def add_policy(self, policy: CompressionPolicy):
        """Add a compression policy."""
        policy.updated_at = datetime.now()
        self.policies.append(policy)

        # Sort by priority (higher priority first)
        self.policies.sort(key=lambda p: p.priority, reverse=True)

        logger.info(f"Added compression policy: {policy.name}")

    def remove_policy(self, policy_name: str) -> bool:
        """Remove a compression policy by name."""
        for i, policy in enumerate(self.policies):
            if policy.name == policy_name:
                del self.policies[i]
                logger.info(f"Removed compression policy: {policy_name}")
                return True
        return False

    def get_policy(self, policy_name: str) -> Optional[CompressionPolicy]:
        """Get a policy by name."""
        for policy in self.policies:
            if policy.name == policy_name:
                return policy
        return None

    def update_policy(self, policy_name: str, updates: Dict[str, Any]) -> bool:
        """Update a compression policy."""
        policy = self.get_policy(policy_name)
        if not policy:
            return False

        for key, value in updates.items():
            if hasattr(policy, key):
                setattr(policy, key, value)

        policy.updated_at = datetime.now()
        logger.info(f"Updated compression policy: {policy_name}")
        return True

    def evaluate_compression_policies(self, memory_data: Dict[str, Any]) -> List[CompressionPolicy]:
        """Evaluate which policies apply to a memory."""
        applicable_policies = []

        for policy in self.policies:
            if not policy.enabled:
                continue

            if self._policy_matches(policy, memory_data):
                applicable_policies.append(policy)

        return applicable_policies

    def compress_memory(self, memory_data: Dict[str, Any], policy_name: Optional[str] = None) -> CompressionResult:
        """Compress a memory using the best applicable policy."""
        if policy_name:
            policy = self.get_policy(policy_name)
            if not policy:
                return CompressionResult(
                    success=False,
                    original_size=0,
                    compressed_size=0,
                    compression_ratio=1.0,
                    algorithm_used=CompressionAlgorithm.NONE,
                    compression_time_ms=0.0,
                    error_message=f"Policy '{policy_name}' not found"
                )
            applicable_policies = [policy]
        else:
            applicable_policies = self.evaluate_compression_policies(memory_data)

        if not applicable_policies:
            return CompressionResult(
                success=False,
                original_size=len(memory_data.get("content", "").encode('utf-8')),
                compressed_size=len(memory_data.get("content", "").encode('utf-8')),
                compression_ratio=1.0,
                algorithm_used=CompressionAlgorithm.NONE,
                compression_time_ms=0.0,
                error_message="No applicable compression policies"
            )

        # Use the highest priority policy
        policy = applicable_policies[0]
        engine = self.engines.get(policy.algorithm)

        if not engine:
            return CompressionResult(
                success=False,
                original_size=len(memory_data.get("content", "").encode('utf-8')),
                compressed_size=len(memory_data.get("content", "").encode('utf-8')),
                compression_ratio=1.0,
                algorithm_used=policy.algorithm,
                compression_time_ms=0.0,
                error_message=f"Engine for {policy.algorithm.value} not available"
            )

        content = memory_data.get("content", "")
        result = engine.compress(
            content,
            compression_level=policy.compression_level,
            chunk_size=policy.chunk_size
        )

        # Check quality controls
        if result.success:
            if result.compression_ratio > policy.min_compression_ratio:
                result.success = False
                result.error_message = f"Compression ratio {result.compression_ratio:.2f} below threshold {policy.min_compression_ratio:.2f}"
            elif result.compression_time_ms > policy.max_compression_time_ms:
                result.success = False
                result.error_message = f"Compression time {result.compression_time_ms:.1f}ms exceeds limit {policy.max_compression_time_ms:.1f}ms"

        result.metadata["policy_used"] = policy.name
        return result

    def decompress_memory(self, compressed_data: bytes, algorithm: CompressionAlgorithm, **kwargs) -> str:
        """Decompress memory data."""
        engine = self.engines.get(algorithm)
        if not engine:
            raise ValueError(f"Engine for {algorithm.value} not available")

        return engine.decompress(compressed_data, **kwargs)

    def estimate_compression_benefit(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate compression benefit for a memory."""
        content = memory_data.get("content", "")
        applicable_policies = self.evaluate_compression_policies(memory_data)

        estimates = {}
        for policy in applicable_policies:
            engine = self.engines.get(policy.algorithm)
            if engine:
                estimated_ratio = engine.estimate_compression_ratio(content)
                original_size = len(content.encode('utf-8'))
                estimated_saved = original_size * (1 - estimated_ratio)

                estimates[policy.name] = {
                    "algorithm": policy.algorithm.value,
                    "estimated_ratio": estimated_ratio,
                    "estimated_saved_bytes": int(estimated_saved),
                    "estimated_saved_percentage": (1 - estimated_ratio) * 100
                }

        return {
            "memory_id": memory_data.get("id"),
            "original_size_bytes": len(content.encode('utf-8')),
            "estimates": estimates,
            "best_estimate": min(estimates.values(), key=lambda x: x["estimated_ratio"]) if estimates else None
        }

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        total_policies = len(self.policies)
        enabled_policies = len([p for p in self.policies if p.enabled])

        algorithm_counts = {}
        for policy in self.policies:
            alg = policy.algorithm.value
            algorithm_counts[alg] = algorithm_counts.get(alg, 0) + 1

        trigger_counts = {}
        for policy in self.policies:
            trigger = policy.trigger.value
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1

        return {
            "total_policies": total_policies,
            "enabled_policies": enabled_policies,
            "available_algorithms": list(self.engines.keys()),
            "algorithm_distribution": algorithm_counts,
            "trigger_distribution": trigger_counts,
            "policy_names": [p.name for p in self.policies]
        }

    def _create_default_policies(self):
        """Create default compression policies."""
        # Age-based GZIP for old memories
        self.add_policy(CompressionPolicy(
            name="age_based_gzip",
            algorithm=CompressionAlgorithm.GZIP,
            trigger=CompressionTrigger.AGE_BASED,
            age_threshold_hours=168,  # 1 week
            compression_level=6,
            priority=100
        ))

        # Size-based ZLIB for large memories
        self.add_policy(CompressionPolicy(
            name="size_based_zlib",
            algorithm=CompressionAlgorithm.ZLIB,
            trigger=CompressionTrigger.SIZE_BASED,
            size_threshold_bytes=10240,  # 10KB
            compression_level=6,
            priority=90
        ))

        # Tier-based adaptive compression
        self.add_policy(CompressionPolicy(
            name="tier_based_adaptive",
            algorithm=CompressionAlgorithm.ADAPTIVE,
            trigger=CompressionTrigger.TIER_BASED,
            tier_names=["long_term", "archived"],
            priority=80
        ))

        # Semantic compression for text content
        self.add_policy(CompressionPolicy(
            name="semantic_text",
            algorithm=CompressionAlgorithm.SEMANTIC,
            trigger=CompressionTrigger.ACCESS_BASED,
            access_threshold_count=1,  # Rarely accessed
            content_type_filters=["text", "markdown"],
            priority=70
        ))

    def _policy_matches(self, policy: CompressionPolicy, memory_data: Dict[str, Any]) -> bool:
        """Check if a policy matches a memory."""
        # Check content type filters
        if policy.content_type_filters:
            content_type = memory_data.get("content_type", "text")
            if content_type not in policy.content_type_filters:
                return False

        # Check tag filters
        if policy.tag_filters:
            memory_tags = set(memory_data.get("tags", []))
            policy_tags = set(policy.tag_filters)
            if not memory_tags.intersection(policy_tags):
                return False

        # Check exclude patterns
        if policy.exclude_patterns:
            content = memory_data.get("content", "")
            for pattern in policy.exclude_patterns:
                if pattern in content:
                    return False

        # Check trigger-specific conditions
        if policy.trigger == CompressionTrigger.AGE_BASED and policy.age_threshold_hours:
            created_at = memory_data.get("created_at")
            if created_at:
                try:
                    created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    age_hours = (datetime.now() - created_time).total_seconds() / 3600
                    if age_hours < policy.age_threshold_hours:
                        return False
                except:
                    return False

        elif policy.trigger == CompressionTrigger.SIZE_BASED and policy.size_threshold_bytes:
            size_bytes = memory_data.get("size_bytes", 0)
            if size_bytes < policy.size_threshold_bytes:
                return False

        elif policy.trigger == CompressionTrigger.ACCESS_BASED and policy.access_threshold_count is not None:
            access_count = memory_data.get("access_count", 0)
            if access_count > policy.access_threshold_count:
                return False

        elif policy.trigger == CompressionTrigger.TIER_BASED and policy.tier_names:
            memory_tier = memory_data.get("tier", "")
            if memory_tier not in policy.tier_names:
                return False

        return True

    def export_policies(self) -> List[Dict[str, Any]]:
        """Export all policies as dictionaries."""
        exported = []
        for policy in self.policies:
            policy_dict = {
                "name": policy.name,
                "algorithm": policy.algorithm.value,
                "trigger": policy.trigger.value,
                "enabled": policy.enabled,
                "priority": policy.priority,
                "age_threshold_hours": policy.age_threshold_hours,
                "size_threshold_bytes": policy.size_threshold_bytes,
                "access_threshold_count": policy.access_threshold_count,
                "tier_names": policy.tier_names,
                "compression_level": policy.compression_level,
                "chunk_size": policy.chunk_size,
                "preserve_structure": policy.preserve_structure,
                "min_compression_ratio": policy.min_compression_ratio,
                "max_compression_time_ms": policy.max_compression_time_ms,
                "content_type_filters": policy.content_type_filters,
                "tag_filters": policy.tag_filters,
                "exclude_patterns": policy.exclude_patterns,
                "created_at": policy.created_at.isoformat(),
                "updated_at": policy.updated_at.isoformat()
            }
            exported.append(policy_dict)
        return exported

    def import_policies(self, policies_data: List[Dict[str, Any]]) -> int:
        """Import policies from dictionaries."""
        imported_count = 0

        for policy_data in policies_data:
            try:
                policy = CompressionPolicy(
                    name=policy_data["name"],
                    algorithm=CompressionAlgorithm(policy_data["algorithm"]),
                    trigger=CompressionTrigger(policy_data["trigger"]),
                    enabled=policy_data.get("enabled", True),
                    priority=policy_data.get("priority", 0),
                    age_threshold_hours=policy_data.get("age_threshold_hours"),
                    size_threshold_bytes=policy_data.get("size_threshold_bytes"),
                    access_threshold_count=policy_data.get("access_threshold_count"),
                    tier_names=policy_data.get("tier_names"),
                    compression_level=policy_data.get("compression_level", 6),
                    chunk_size=policy_data.get("chunk_size", 8192),
                    preserve_structure=policy_data.get("preserve_structure", True),
                    min_compression_ratio=policy_data.get("min_compression_ratio", 0.1),
                    max_compression_time_ms=policy_data.get("max_compression_time_ms", 5000),
                    content_type_filters=policy_data.get("content_type_filters", []),
                    tag_filters=policy_data.get("tag_filters", []),
                    exclude_patterns=policy_data.get("exclude_patterns", [])
                )

                # Check if policy already exists
                existing = self.get_policy(policy.name)
                if existing:
                    self.remove_policy(policy.name)

                self.add_policy(policy)
                imported_count += 1

            except Exception as e:
                logger.error(f"Failed to import policy {policy_data.get('name', 'unknown')}: {e}")

        return imported_count