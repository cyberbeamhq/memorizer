"""
Memory Analytics Module
Advanced analytics and insights for memory usage patterns.
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Types of memory insights."""
    OPTIMIZATION = "optimization"
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    RECOMMENDATION = "recommendation"


@dataclass
class MemoryInsight:
    """Represents a memory usage insight."""
    type: InsightType
    title: str
    description: str
    impact: str
    confidence: float
    data: Dict[str, Any]
    timestamp: datetime
    actionable: bool = True


@dataclass
class MemoryUsageReport:
    """Comprehensive memory usage report."""
    user_id: str
    report_period: Tuple[datetime, datetime]
    total_memories: int
    total_size_bytes: int
    memories_by_tier: Dict[str, int]
    size_by_tier: Dict[str, int]
    access_patterns: Dict[str, Any]
    compression_stats: Dict[str, Any]
    top_accessed_memories: List[Dict[str, Any]]
    growth_trend: Dict[str, float]
    insights: List[MemoryInsight]
    generated_at: datetime


class MemoryAnalytics:
    """Advanced memory analytics and insights generator."""

    def __init__(self, storage, cache=None):
        """Initialize memory analytics."""
        self.storage = storage
        self.cache = cache
        self._insight_cache = {}

    def generate_usage_report(
        self,
        user_id: str,
        days: int = 30,
        include_insights: bool = True
    ) -> MemoryUsageReport:
        """Generate comprehensive usage report for a user."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        try:
            # Get all memories for the user in the time period
            memories = self._get_user_memories_in_period(user_id, start_date, end_date)

            # Calculate basic statistics
            total_memories = len(memories)
            total_size = sum(len(m.content.encode('utf-8')) for m in memories)

            # Analyze by tier
            memories_by_tier = self._analyze_by_tier(memories)
            size_by_tier = self._calculate_size_by_tier(memories)

            # Analyze access patterns
            access_patterns = self._analyze_access_patterns(memories)

            # Calculate compression statistics
            compression_stats = self._calculate_compression_stats(memories)

            # Find top accessed memories
            top_accessed = self._get_top_accessed_memories(memories, limit=10)

            # Calculate growth trend
            growth_trend = self._calculate_growth_trend(user_id, days)

            # Generate insights
            insights = []
            if include_insights:
                insights = self._generate_insights(user_id, memories, access_patterns, compression_stats)

            report = MemoryUsageReport(
                user_id=user_id,
                report_period=(start_date, end_date),
                total_memories=total_memories,
                total_size_bytes=total_size,
                memories_by_tier=memories_by_tier,
                size_by_tier=size_by_tier,
                access_patterns=access_patterns,
                compression_stats=compression_stats,
                top_accessed_memories=top_accessed,
                growth_trend=growth_trend,
                insights=insights,
                generated_at=datetime.now()
            )

            logger.info(f"Generated usage report for user {user_id}: {total_memories} memories, {total_size} bytes")
            return report

        except Exception as e:
            logger.error(f"Failed to generate usage report for user {user_id}: {e}")
            raise

    def _get_user_memories_in_period(self, user_id: str, start_date: datetime, end_date: datetime):
        """Get user memories within a specific time period."""
        # This would need to be implemented based on your storage backend
        # For now, we'll get all memories and filter
        try:
            all_memories = self.storage.search_memories(user_id=user_id, query="", limit=10000)

            filtered_memories = []
            for memory in all_memories:
                memory_date = datetime.fromtimestamp(memory.created_at) if hasattr(memory, 'created_at') else datetime.now()
                if start_date <= memory_date <= end_date:
                    filtered_memories.append(memory)

            return filtered_memories
        except Exception as e:
            logger.warning(f"Error filtering memories by date: {e}")
            return []

    def _analyze_by_tier(self, memories) -> Dict[str, int]:
        """Analyze memory distribution by tier."""
        tier_counts = {}
        for memory in memories:
            tier = getattr(memory, 'tier', 'unknown')
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        return tier_counts

    def _calculate_size_by_tier(self, memories) -> Dict[str, int]:
        """Calculate total size by tier."""
        tier_sizes = {}
        for memory in memories:
            tier = getattr(memory, 'tier', 'unknown')
            size = len(memory.content.encode('utf-8'))
            tier_sizes[tier] = tier_sizes.get(tier, 0) + size
        return tier_sizes

    def _analyze_access_patterns(self, memories) -> Dict[str, Any]:
        """Analyze memory access patterns."""
        access_counts = []
        last_accessed_times = []

        for memory in memories:
            access_count = getattr(memory, 'access_count', 0)
            last_accessed = getattr(memory, 'last_accessed', None)

            access_counts.append(access_count)
            if last_accessed:
                last_accessed_times.append(last_accessed)

        return {
            "total_accesses": sum(access_counts),
            "avg_accesses_per_memory": statistics.mean(access_counts) if access_counts else 0,
            "max_accesses": max(access_counts) if access_counts else 0,
            "memories_never_accessed": sum(1 for count in access_counts if count == 0),
            "highly_accessed_memories": sum(1 for count in access_counts if count > 10),
            "access_distribution": {
                "0_accesses": sum(1 for count in access_counts if count == 0),
                "1-5_accesses": sum(1 for count in access_counts if 1 <= count <= 5),
                "6-10_accesses": sum(1 for count in access_counts if 6 <= count <= 10),
                "11+_accesses": sum(1 for count in access_counts if count > 10),
            }
        }

    def _calculate_compression_stats(self, memories) -> Dict[str, Any]:
        """Calculate compression statistics."""
        original_sizes = []
        compressed_sizes = []
        compression_ratios = []

        for memory in memories:
            original_size = len(memory.content.encode('utf-8'))
            # For now, estimate compression based on tier
            tier = getattr(memory, 'tier', 'very_new')
            if tier == 'very_new':
                compressed_size = original_size  # No compression
                ratio = 1.0
            elif tier == 'mid_term':
                compressed_size = int(original_size * 0.6)  # 40% compression
                ratio = 0.6
            else:  # long_term
                compressed_size = int(original_size * 0.3)  # 70% compression
                ratio = 0.3

            original_sizes.append(original_size)
            compressed_sizes.append(compressed_size)
            compression_ratios.append(ratio)

        total_original = sum(original_sizes)
        total_compressed = sum(compressed_sizes)

        return {
            "total_original_size": total_original,
            "total_compressed_size": total_compressed,
            "overall_compression_ratio": total_compressed / total_original if total_original > 0 else 1.0,
            "avg_compression_ratio": statistics.mean(compression_ratios) if compression_ratios else 1.0,
            "estimated_savings_bytes": total_original - total_compressed,
            "estimated_savings_percent": ((total_original - total_compressed) / total_original * 100) if total_original > 0 else 0
        }

    def _get_top_accessed_memories(self, memories, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top accessed memories."""
        memory_data = []
        for memory in memories:
            access_count = getattr(memory, 'access_count', 0)
            memory_data.append({
                "id": memory.id,
                "content_preview": memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                "access_count": access_count,
                "tier": getattr(memory, 'tier', 'unknown'),
                "size_bytes": len(memory.content.encode('utf-8')),
                "last_accessed": getattr(memory, 'last_accessed', None)
            })

        # Sort by access count and return top N
        memory_data.sort(key=lambda x: x['access_count'], reverse=True)
        return memory_data[:limit]

    def _calculate_growth_trend(self, user_id: str, days: int) -> Dict[str, float]:
        """Calculate memory growth trend."""
        try:
            # This is a simplified implementation
            # In production, you'd want to track this data over time
            current_count = len(self.storage.search_memories(user_id=user_id, query="", limit=10000))

            # Estimate daily growth (this would be calculated from historical data)
            estimated_daily_growth = current_count / max(days, 1)

            return {
                "current_total": current_count,
                "estimated_daily_growth": estimated_daily_growth,
                "projected_30_day": current_count + (estimated_daily_growth * 30),
                "projected_90_day": current_count + (estimated_daily_growth * 90),
                "growth_rate_percent": (estimated_daily_growth / max(current_count, 1)) * 100
            }
        except Exception as e:
            logger.warning(f"Error calculating growth trend: {e}")
            return {"current_total": 0, "estimated_daily_growth": 0, "projected_30_day": 0, "projected_90_day": 0, "growth_rate_percent": 0}

    def _generate_insights(self, user_id: str, memories, access_patterns: Dict, compression_stats: Dict) -> List[MemoryInsight]:
        """Generate actionable insights based on memory analysis."""
        insights = []

        # Insight 1: Unused memories
        never_accessed = access_patterns.get("memories_never_accessed", 0)
        total_memories = len(memories)

        if never_accessed > total_memories * 0.3:  # More than 30% never accessed
            insights.append(MemoryInsight(
                type=InsightType.OPTIMIZATION,
                title="High Unused Memory Rate",
                description=f"{never_accessed} out of {total_memories} memories ({never_accessed/total_memories*100:.1f}%) have never been accessed.",
                impact=f"Consider archiving unused memories to reduce storage costs by ~{never_accessed/total_memories*30:.1f}%",
                confidence=0.9,
                data={"unused_count": never_accessed, "total_count": total_memories},
                timestamp=datetime.now()
            ))

        # Insight 2: Compression opportunities
        compression_ratio = compression_stats.get("overall_compression_ratio", 1.0)
        if compression_ratio > 0.8:  # Less than 20% compression
            insights.append(MemoryInsight(
                type=InsightType.OPTIMIZATION,
                title="Low Compression Efficiency",
                description=f"Current compression ratio is {compression_ratio:.2f}, indicating minimal space savings.",
                impact="Implementing more aggressive compression could reduce storage costs by 40-60%",
                confidence=0.8,
                data={"current_ratio": compression_ratio, "potential_savings": compression_stats.get("estimated_savings_percent", 0)},
                timestamp=datetime.now()
            ))

        # Insight 3: Access pattern analysis
        highly_accessed = access_patterns.get("highly_accessed_memories", 0)
        if highly_accessed > 0:
            insights.append(MemoryInsight(
                type=InsightType.PATTERN,
                title="High-Value Memories Identified",
                description=f"{highly_accessed} memories are frequently accessed (10+ times).",
                impact="Consider prioritizing these memories for faster retrieval and better caching",
                confidence=0.95,
                data={"high_access_count": highly_accessed, "access_threshold": 10},
                timestamp=datetime.now()
            ))

        # Insight 4: Growth rate warning
        growth_trend = self._calculate_growth_trend(user_id, 30)
        daily_growth = growth_trend.get("estimated_daily_growth", 0)
        if daily_growth > 50:  # More than 50 memories per day
            insights.append(MemoryInsight(
                type=InsightType.ANOMALY,
                title="Rapid Memory Growth Detected",
                description=f"Memory creation rate is {daily_growth:.1f} memories per day.",
                impact="High growth rate may indicate inefficient memory management or data duplication",
                confidence=0.7,
                data={"daily_growth": daily_growth, "projected_monthly": daily_growth * 30},
                timestamp=datetime.now()
            ))

        return insights

    def get_memory_health_score(self, user_id: str) -> Dict[str, Any]:
        """Calculate an overall memory health score."""
        try:
            report = self.generate_usage_report(user_id, days=30, include_insights=False)

            # Calculate various health metrics (0-100 scale)

            # Access efficiency (how many memories are actually used)
            never_accessed = report.access_patterns.get("memories_never_accessed", 0)
            access_efficiency = max(0, 100 - (never_accessed / max(report.total_memories, 1) * 100))

            # Compression efficiency
            compression_savings = report.compression_stats.get("estimated_savings_percent", 0)
            compression_efficiency = min(100, compression_savings * 2)  # Scale 0-50% savings to 0-100

            # Storage efficiency (balance between too few and too many memories)
            if report.total_memories < 10:
                storage_efficiency = 50  # Underutilized
            elif report.total_memories > 1000:
                storage_efficiency = max(50, 100 - (report.total_memories - 1000) / 100)  # Penalize excessive storage
            else:
                storage_efficiency = 100

            # Overall health score (weighted average)
            overall_score = (
                access_efficiency * 0.4 +
                compression_efficiency * 0.3 +
                storage_efficiency * 0.3
            )

            return {
                "overall_score": round(overall_score, 1),
                "access_efficiency": round(access_efficiency, 1),
                "compression_efficiency": round(compression_efficiency, 1),
                "storage_efficiency": round(storage_efficiency, 1),
                "grade": self._score_to_grade(overall_score),
                "recommendations": self._get_health_recommendations(access_efficiency, compression_efficiency, storage_efficiency)
            }

        except Exception as e:
            logger.error(f"Failed to calculate memory health score: {e}")
            return {"overall_score": 0, "grade": "F", "error": str(e)}

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _get_health_recommendations(self, access_eff: float, compression_eff: float, storage_eff: float) -> List[str]:
        """Get recommendations based on health scores."""
        recommendations = []

        if access_eff < 70:
            recommendations.append("Consider archiving or removing unused memories to improve access efficiency")

        if compression_eff < 60:
            recommendations.append("Implement more aggressive compression policies to reduce storage costs")

        if storage_eff < 70:
            recommendations.append("Review memory retention policies to optimize storage usage")

        if not recommendations:
            recommendations.append("Memory management is performing well - continue current practices")

        return recommendations