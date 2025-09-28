"""
compression_metrics.py
Metrics collection and analysis for compression operations.
Tracks token reduction, compression ratios, confidence scores, and performance.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class CompressionMetrics:
    """Metrics for a single compression operation."""
    timestamp: datetime
    compression_type: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    token_reduction: int
    processing_time: float
    llm_provider: str
    llm_model: str
    success: bool
    retry_count: int = 0
    confidence_score: float = 0.0
    pii_detected: bool = False
    pii_count: int = 0
    schema_validation_errors: int = 0
    error_message: Optional[str] = None


@dataclass
class CompressionStats:
    """Aggregated compression statistics."""
    total_compressions: int = 0
    successful_compressions: int = 0
    failed_compressions: int = 0
    average_compression_ratio: float = 0.0
    average_processing_time: float = 0.0
    average_confidence_score: float = 0.0
    total_token_reduction: int = 0
    pii_detection_rate: float = 0.0
    schema_validation_error_rate: float = 0.0
    retry_rate: float = 0.0
    provider_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    compression_type_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class CompressionMetricsCollector:
    """Collects and analyzes compression metrics."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector."""
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.stats_cache: Optional[CompressionStats] = None
        self.cache_timestamp: Optional[datetime] = None
        self.cache_ttl = timedelta(minutes=5)  # Cache TTL
        
        logger.info(f"Compression metrics collector initialized (max_history: {max_history})")
    
    def record_compression(
        self,
        compression_type: str,
        original_length: int,
        compressed_length: int,
        processing_time: float,
        llm_provider: str,
        llm_model: str,
        success: bool,
        retry_count: int = 0,
        confidence_score: float = 0.0,
        pii_detected: bool = False,
        pii_count: int = 0,
        schema_validation_errors: int = 0,
        error_message: Optional[str] = None
    ) -> CompressionMetrics:
        """Record a compression operation."""
        
        # Calculate derived metrics
        compression_ratio = compressed_length / original_length if original_length > 0 else 0.0
        token_reduction = original_length - compressed_length
        
        metrics = CompressionMetrics(
            timestamp=datetime.now(),
            compression_type=compression_type,
            original_length=original_length,
            compressed_length=compressed_length,
            compression_ratio=compression_ratio,
            token_reduction=token_reduction,
            processing_time=processing_time,
            llm_provider=llm_provider,
            llm_model=llm_model,
            success=success,
            retry_count=retry_count,
            confidence_score=confidence_score,
            pii_detected=pii_detected,
            pii_count=pii_count,
            schema_validation_errors=schema_validation_errors,
            error_message=error_message
        )
        
        self.metrics_history.append(metrics)
        self._invalidate_cache()
        
        logger.debug(f"Recorded compression metrics: {compression_type}, ratio: {compression_ratio:.2f}")
        return metrics
    
    def get_stats(self, time_window: Optional[timedelta] = None) -> CompressionStats:
        """Get aggregated compression statistics."""
        # Check cache
        if (self.stats_cache and self.cache_timestamp and 
            datetime.now() - self.cache_timestamp < self.cache_ttl):
            return self.stats_cache
        
        # Filter metrics by time window
        if time_window:
            cutoff_time = datetime.now() - time_window
            metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        else:
            metrics = list(self.metrics_history)
        
        if not metrics:
            return CompressionStats()
        
        # Calculate basic stats
        total_compressions = len(metrics)
        successful_compressions = len([m for m in metrics if m.success])
        failed_compressions = total_compressions - successful_compressions
        
        # Calculate averages
        successful_metrics = [m for m in metrics if m.success]
        
        if successful_metrics:
            avg_compression_ratio = statistics.mean([m.compression_ratio for m in successful_metrics])
            avg_processing_time = statistics.mean([m.processing_time for m in successful_metrics])
            avg_confidence_score = statistics.mean([m.confidence_score for m in successful_metrics])
            total_token_reduction = sum([m.token_reduction for m in successful_metrics])
        else:
            avg_compression_ratio = 0.0
            avg_processing_time = 0.0
            avg_confidence_score = 0.0
            total_token_reduction = 0
        
        # Calculate rates
        pii_detection_rate = len([m for m in metrics if m.pii_detected]) / total_compressions
        schema_error_rate = len([m for m in metrics if m.schema_validation_errors > 0]) / total_compressions
        retry_rate = len([m for m in metrics if m.retry_count > 0]) / total_compressions
        
        # Provider performance
        provider_performance = self._calculate_provider_performance(successful_metrics)
        
        # Compression type stats
        compression_type_stats = self._calculate_compression_type_stats(successful_metrics)
        
        stats = CompressionStats(
            total_compressions=total_compressions,
            successful_compressions=successful_compressions,
            failed_compressions=failed_compressions,
            average_compression_ratio=avg_compression_ratio,
            average_processing_time=avg_processing_time,
            average_confidence_score=avg_confidence_score,
            total_token_reduction=total_token_reduction,
            pii_detection_rate=pii_detection_rate,
            schema_validation_error_rate=schema_error_rate,
            retry_rate=retry_rate,
            provider_performance=provider_performance,
            compression_type_stats=compression_type_stats
        )
        
        # Update cache
        self.stats_cache = stats
        self.cache_timestamp = datetime.now()
        
        return stats
    
    def _calculate_provider_performance(self, metrics: List[CompressionMetrics]) -> Dict[str, Dict[str, Any]]:
        """Calculate performance metrics by provider."""
        provider_stats = defaultdict(lambda: {
            'count': 0,
            'avg_compression_ratio': 0.0,
            'avg_processing_time': 0.0,
            'avg_confidence_score': 0.0,
            'total_token_reduction': 0,
            'models': set()
        })
        
        for metric in metrics:
            provider = metric.llm_provider
            provider_stats[provider]['count'] += 1
            provider_stats[provider]['avg_compression_ratio'] += metric.compression_ratio
            provider_stats[provider]['avg_processing_time'] += metric.processing_time
            provider_stats[provider]['avg_confidence_score'] += metric.confidence_score
            provider_stats[provider]['total_token_reduction'] += metric.token_reduction
            provider_stats[provider]['models'].add(metric.llm_model)
        
        # Calculate averages
        for provider, stats in provider_stats.items():
            if stats['count'] > 0:
                stats['avg_compression_ratio'] /= stats['count']
                stats['avg_processing_time'] /= stats['count']
                stats['avg_confidence_score'] /= stats['count']
                stats['models'] = list(stats['models'])
        
        return dict(provider_stats)
    
    def _calculate_compression_type_stats(self, metrics: List[CompressionMetrics]) -> Dict[str, Dict[str, Any]]:
        """Calculate performance metrics by compression type."""
        type_stats = defaultdict(lambda: {
            'count': 0,
            'avg_compression_ratio': 0.0,
            'avg_processing_time': 0.0,
            'avg_confidence_score': 0.0,
            'total_token_reduction': 0
        })
        
        for metric in metrics:
            comp_type = metric.compression_type
            type_stats[comp_type]['count'] += 1
            type_stats[comp_type]['avg_compression_ratio'] += metric.compression_ratio
            type_stats[comp_type]['avg_processing_time'] += metric.processing_time
            type_stats[comp_type]['avg_confidence_score'] += metric.confidence_score
            type_stats[comp_type]['total_token_reduction'] += metric.token_reduction
        
        # Calculate averages
        for comp_type, stats in type_stats.items():
            if stats['count'] > 0:
                stats['avg_compression_ratio'] /= stats['count']
                stats['avg_processing_time'] /= stats['count']
                stats['avg_confidence_score'] /= stats['count']
        
        return dict(type_stats)
    
    def _invalidate_cache(self):
        """Invalidate the stats cache."""
        self.stats_cache = None
        self.cache_timestamp = None
    
    def get_recent_metrics(self, count: int = 100) -> List[CompressionMetrics]:
        """Get recent compression metrics."""
        return list(self.metrics_history)[-count:]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of compression metrics."""
        stats = self.get_stats()
        
        return {
            "overview": {
                "total_compressions": stats.total_compressions,
                "success_rate": stats.successful_compressions / max(stats.total_compressions, 1),
                "average_compression_ratio": stats.average_compression_ratio,
                "total_token_reduction": stats.total_token_reduction
            },
            "performance": {
                "average_processing_time": stats.average_processing_time,
                "average_confidence_score": stats.average_confidence_score,
                "retry_rate": stats.retry_rate
            },
            "quality": {
                "pii_detection_rate": stats.pii_detection_rate,
                "schema_validation_error_rate": stats.schema_validation_error_rate
            },
            "providers": stats.provider_performance,
            "compression_types": stats.compression_type_stats
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        if format == "json":
            import json
            metrics_data = []
            for metric in self.metrics_history:
                metrics_data.append({
                    "timestamp": metric.timestamp.isoformat(),
                    "compression_type": metric.compression_type,
                    "original_length": metric.original_length,
                    "compressed_length": metric.compressed_length,
                    "compression_ratio": metric.compression_ratio,
                    "token_reduction": metric.token_reduction,
                    "processing_time": metric.processing_time,
                    "llm_provider": metric.llm_provider,
                    "llm_model": metric.llm_model,
                    "success": metric.success,
                    "retry_count": metric.retry_count,
                    "confidence_score": metric.confidence_score,
                    "pii_detected": metric.pii_detected,
                    "pii_count": metric.pii_count,
                    "schema_validation_errors": metric.schema_validation_errors,
                    "error_message": metric.error_message
                })
            return json.dumps(metrics_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global metrics collector instance
_metrics_collector = None


def get_metrics_collector() -> CompressionMetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = CompressionMetricsCollector()
    return _metrics_collector


def record_compression_metrics(
    compression_type: str,
    original_length: int,
    compressed_length: int,
    processing_time: float,
    llm_provider: str,
    llm_model: str,
    success: bool,
    **kwargs
) -> CompressionMetrics:
    """Convenience function to record compression metrics."""
    collector = get_metrics_collector()
    return collector.record_compression(
        compression_type=compression_type,
        original_length=original_length,
        compressed_length=compressed_length,
        processing_time=processing_time,
        llm_provider=llm_provider,
        llm_model=llm_model,
        success=success,
        **kwargs
    )
