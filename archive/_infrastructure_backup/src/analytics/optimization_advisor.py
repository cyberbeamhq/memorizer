"""
Optimization Advisor Module
AI-powered recommendations for memory management optimization.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Types of optimization recommendations."""
    COMPRESSION = "compression"
    ARCHIVAL = "archival"
    CACHING = "caching"
    RETENTION = "retention"
    PERFORMANCE = "performance"
    COST = "cost"


class Priority(Enum):
    """Recommendation priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OptimizationRecommendation:
    """Represents an optimization recommendation."""
    id: str
    type: RecommendationType
    priority: Priority
    title: str
    description: str
    impact_description: str
    estimated_savings: Dict[str, Any]
    implementation_effort: str
    actionable_steps: List[str]
    affected_memories: Optional[List[str]]
    confidence_score: float
    created_at: datetime
    expires_at: Optional[datetime]
    metadata: Dict[str, Any]


class OptimizationAdvisor:
    """AI-powered optimization advisor for memory management."""

    def __init__(self, analytics, usage_tracker):
        """Initialize optimization advisor."""
        self.analytics = analytics
        self.usage_tracker = usage_tracker
        self._recommendation_cache = {}

    def generate_recommendations(self, user_id: str) -> List[OptimizationRecommendation]:
        """Generate comprehensive optimization recommendations."""
        try:
            # Get user analytics and usage data
            usage_report = self.analytics.generate_usage_report(user_id, days=30)
            usage_metrics = self.usage_tracker.get_user_metrics(user_id, hours=24 * 7)  # 7 days

            recommendations = []

            # Generate different types of recommendations
            recommendations.extend(self._generate_compression_recommendations(user_id, usage_report))
            recommendations.extend(self._generate_archival_recommendations(user_id, usage_report))
            recommendations.extend(self._generate_caching_recommendations(user_id, usage_metrics))
            recommendations.extend(self._generate_retention_recommendations(user_id, usage_report))
            recommendations.extend(self._generate_performance_recommendations(user_id, usage_metrics))
            recommendations.extend(self._generate_cost_recommendations(user_id, usage_report))

            # Sort by priority and confidence
            recommendations.sort(key=lambda x: (x.priority.value, -x.confidence_score))

            logger.info(f"Generated {len(recommendations)} optimization recommendations for user {user_id}")
            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations for user {user_id}: {e}")
            return []

    def _generate_compression_recommendations(self, user_id: str, usage_report) -> List[OptimizationRecommendation]:
        """Generate compression-related recommendations."""
        recommendations = []

        compression_stats = usage_report.compression_stats
        overall_ratio = compression_stats.get("overall_compression_ratio", 1.0)

        if overall_ratio > 0.7:  # Less than 30% compression
            estimated_savings = int(compression_stats.get("total_original_size", 0) * 0.4)  # 40% potential savings

            recommendations.append(OptimizationRecommendation(
                id=f"compression_{user_id}_{int(datetime.now().timestamp())}",
                type=RecommendationType.COMPRESSION,
                priority=Priority.HIGH,
                title="Implement Aggressive Compression",
                description=f"Current compression ratio is {overall_ratio:.2f}. Implementing more aggressive compression policies could significantly reduce storage costs.",
                impact_description=f"Potential storage reduction of {estimated_savings:,} bytes (~40% savings)",
                estimated_savings={
                    "storage_bytes": estimated_savings,
                    "cost_percentage": 40,
                    "monthly_cost_usd": estimated_savings * 0.00001  # Rough estimate
                },
                implementation_effort="Medium - requires updating compression policies",
                actionable_steps=[
                    "Enable tier-based compression for memories older than 7 days",
                    "Implement semantic deduplication for similar content",
                    "Use more aggressive compression algorithms for long-term storage",
                    "Set up automated compression scheduling"
                ],
                affected_memories=None,
                confidence_score=0.85,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=30),
                metadata={"current_ratio": overall_ratio, "potential_ratio": 0.4}
            ))

        return recommendations

    def _generate_archival_recommendations(self, user_id: str, usage_report) -> List[OptimizationRecommendation]:
        """Generate archival-related recommendations."""
        recommendations = []

        access_patterns = usage_report.access_patterns
        never_accessed = access_patterns.get("memories_never_accessed", 0)
        total_memories = usage_report.total_memories

        if never_accessed > total_memories * 0.2:  # More than 20% never accessed
            recommendations.append(OptimizationRecommendation(
                id=f"archival_{user_id}_{int(datetime.now().timestamp())}",
                type=RecommendationType.ARCHIVAL,
                priority=Priority.MEDIUM,
                title="Archive Unused Memories",
                description=f"{never_accessed} memories ({never_accessed/total_memories*100:.1f}%) have never been accessed and could be archived.",
                impact_description=f"Reduce active storage by {never_accessed} memories and improve query performance",
                estimated_savings={
                    "memories_count": never_accessed,
                    "performance_improvement": "15-25%",
                    "storage_tier_optimization": True
                },
                implementation_effort="Low - automated archival policy",
                actionable_steps=[
                    f"Set up automated archival for memories unused after 90 days",
                    "Implement tiered storage with cold storage for archived memories",
                    "Create archive retrieval process for occasional access",
                    "Monitor archival effectiveness over time"
                ],
                affected_memories=None,
                confidence_score=0.9,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=60),
                metadata={"unused_count": never_accessed, "unused_percentage": never_accessed/total_memories*100}
            ))

        return recommendations

    def _generate_caching_recommendations(self, user_id: str, usage_metrics) -> List[OptimizationRecommendation]:
        """Generate caching-related recommendations."""
        recommendations = []

        avg_response_time = usage_metrics.avg_response_time_ms

        if avg_response_time > 200:  # Slow response times
            recommendations.append(OptimizationRecommendation(
                id=f"caching_{user_id}_{int(datetime.now().timestamp())}",
                type=RecommendationType.CACHING,
                priority=Priority.HIGH,
                title="Optimize Memory Caching",
                description=f"Average response time is {avg_response_time:.1f}ms. Implementing smart caching could improve performance.",
                impact_description=f"Reduce response times by 50-70% for frequently accessed memories",
                estimated_savings={
                    "response_time_improvement": "50-70%",
                    "user_experience_score": "+25%",
                    "system_load_reduction": "30%"
                },
                implementation_effort="Medium - requires cache optimization",
                actionable_steps=[
                    "Implement LRU cache for frequently accessed memories",
                    "Pre-cache memories based on access patterns",
                    "Set up distributed caching for multi-instance deployments",
                    "Monitor cache hit rates and optimize cache size"
                ],
                affected_memories=None,
                confidence_score=0.8,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=21),
                metadata={"current_response_time": avg_response_time, "target_response_time": 100}
            ))

        return recommendations

    def _generate_retention_recommendations(self, user_id: str, usage_report) -> List[OptimizationRecommendation]:
        """Generate retention policy recommendations."""
        recommendations = []

        growth_trend = usage_report.growth_trend
        daily_growth = growth_trend.get("estimated_daily_growth", 0)
        projected_90_day = growth_trend.get("projected_90_day", 0)

        if daily_growth > 10 and projected_90_day > 1000:  # High growth trajectory
            recommendations.append(OptimizationRecommendation(
                id=f"retention_{user_id}_{int(datetime.now().timestamp())}",
                type=RecommendationType.RETENTION,
                priority=Priority.MEDIUM,
                title="Implement Smart Retention Policies",
                description=f"High growth rate ({daily_growth:.1f} memories/day) requires retention policies to prevent storage bloat.",
                impact_description=f"Prevent storage costs from escalating beyond manageable levels",
                estimated_savings={
                    "prevented_growth": projected_90_day * 0.3,  # 30% reduction
                    "cost_prevention": "significant",
                    "storage_optimization": True
                },
                implementation_effort="Low - policy configuration",
                actionable_steps=[
                    "Set maximum retention period for very_new tier (30 days)",
                    "Implement automatic deletion of low-value memories",
                    "Create user-specific retention preferences",
                    "Set up alerts for unusual growth patterns"
                ],
                affected_memories=None,
                confidence_score=0.75,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=45),
                metadata={"daily_growth": daily_growth, "projected_90_day": projected_90_day}
            ))

        return recommendations

    def _generate_performance_recommendations(self, user_id: str, usage_metrics) -> List[OptimizationRecommendation]:
        """Generate performance optimization recommendations."""
        recommendations = []

        events_by_type = usage_metrics.events_by_type
        search_count = events_by_type.get("search_performed", 0)
        access_count = events_by_type.get("memory_accessed", 0)

        if search_count > access_count * 2:  # Too many searches relative to accesses
            recommendations.append(OptimizationRecommendation(
                id=f"performance_{user_id}_{int(datetime.now().timestamp())}",
                type=RecommendationType.PERFORMANCE,
                priority=Priority.MEDIUM,
                title="Optimize Search-to-Access Ratio",
                description=f"Search operations ({search_count}) significantly exceed memory accesses ({access_count}), indicating inefficient search patterns.",
                impact_description="Improve search relevance and reduce unnecessary operations",
                estimated_savings={
                    "search_efficiency": "+40%",
                    "reduced_operations": search_count * 0.3,
                    "system_resource_savings": "20-30%"
                },
                implementation_effort="Medium - search algorithm optimization",
                actionable_steps=[
                    "Implement search result relevance scoring",
                    "Add search query suggestions and auto-completion",
                    "Optimize vector search parameters",
                    "Implement search result caching"
                ],
                affected_memories=None,
                confidence_score=0.7,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=30),
                metadata={"search_count": search_count, "access_count": access_count}
            ))

        return recommendations

    def _generate_cost_recommendations(self, user_id: str, usage_report) -> List[OptimizationRecommendation]:
        """Generate cost optimization recommendations."""
        recommendations = []

        total_size = usage_report.total_size_bytes
        memories_by_tier = usage_report.memories_by_tier

        # Check if too many memories are in expensive tiers
        very_new_count = memories_by_tier.get("very_new", 0)
        total_count = usage_report.total_memories

        if very_new_count > total_count * 0.6:  # More than 60% in most expensive tier
            estimated_monthly_cost = total_size * 0.00002  # Rough cost estimate

            recommendations.append(OptimizationRecommendation(
                id=f"cost_{user_id}_{int(datetime.now().timestamp())}",
                type=RecommendationType.COST,
                priority=Priority.HIGH,
                title="Optimize Memory Tier Distribution",
                description=f"{very_new_count}/{total_count} memories are in the expensive 'very_new' tier. Optimizing tier progression could reduce costs.",
                impact_description=f"Potential monthly cost reduction of 40-60%",
                estimated_savings={
                    "monthly_cost_usd": estimated_monthly_cost * 0.5,
                    "cost_reduction_percentage": 50,
                    "storage_optimization": True
                },
                implementation_effort="Low - tier policy adjustment",
                actionable_steps=[
                    "Reduce very_new tier retention from 30 to 14 days",
                    "Implement faster progression to mid_term tier",
                    "Set up automated tier optimization based on access patterns",
                    "Monitor cost impact of tier changes"
                ],
                affected_memories=None,
                confidence_score=0.85,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=14),
                metadata={
                    "very_new_percentage": very_new_count/total_count*100,
                    "estimated_monthly_cost": estimated_monthly_cost
                }
            ))

        return recommendations

    def get_recommendation_by_id(self, recommendation_id: str) -> Optional[OptimizationRecommendation]:
        """Get a specific recommendation by ID."""
        return self._recommendation_cache.get(recommendation_id)

    def mark_recommendation_implemented(self, recommendation_id: str, user_id: str, notes: str = ""):
        """Mark a recommendation as implemented."""
        # In a real implementation, this would update a database
        logger.info(f"Recommendation {recommendation_id} marked as implemented for user {user_id}: {notes}")

    def get_implementation_tracking(self, user_id: str) -> Dict[str, Any]:
        """Track implementation status of recommendations."""
        # This would be retrieved from a database in a real implementation
        return {
            "total_recommendations_generated": 0,
            "recommendations_implemented": 0,
            "implementation_rate": 0.0,
            "avg_time_to_implement_days": 0,
            "most_common_implementations": [],
            "estimated_total_savings": {}
        }

    def generate_optimization_summary(self, user_id: str) -> Dict[str, Any]:
        """Generate a comprehensive optimization summary."""
        try:
            recommendations = self.generate_recommendations(user_id)

            # Aggregate potential savings
            total_storage_savings = 0
            total_cost_savings = 0
            performance_improvements = []

            for rec in recommendations:
                savings = rec.estimated_savings
                if "storage_bytes" in savings:
                    total_storage_savings += savings["storage_bytes"]
                if "monthly_cost_usd" in savings:
                    total_cost_savings += savings["monthly_cost_usd"]
                if "response_time_improvement" in savings:
                    performance_improvements.append(savings["response_time_improvement"])

            priority_breakdown = {}
            for rec in recommendations:
                priority = rec.priority.value
                priority_breakdown[priority] = priority_breakdown.get(priority, 0) + 1

            return {
                "user_id": user_id,
                "generated_at": datetime.now().isoformat(),
                "total_recommendations": len(recommendations),
                "priority_breakdown": priority_breakdown,
                "potential_savings": {
                    "storage_bytes": total_storage_savings,
                    "monthly_cost_usd": total_cost_savings,
                    "performance_improvements": performance_improvements
                },
                "top_recommendations": [
                    {
                        "id": rec.id,
                        "title": rec.title,
                        "priority": rec.priority.value,
                        "type": rec.type.value,
                        "confidence": rec.confidence_score
                    }
                    for rec in recommendations[:5]
                ],
                "quick_wins": [
                    rec for rec in recommendations
                    if rec.implementation_effort.startswith("Low") and rec.priority in [Priority.HIGH, Priority.MEDIUM]
                ][:3]
            }

        except Exception as e:
            logger.error(f"Failed to generate optimization summary for user {user_id}: {e}")
            return {"error": str(e), "user_id": user_id}