"""
GraphQL API Module
Advanced GraphQL API for complex memory queries and operations.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info

from ..core.memory_manager import MemoryManager
from ..analytics.memory_analytics import MemoryAnalytics
from ..analytics.usage_tracker import UsageTracker
from ..analytics.optimization_advisor import OptimizationAdvisor
from ..tenancy.tenant_isolation import get_tenant_isolation, TenantContext

logger = logging.getLogger(__name__)


# GraphQL Types
@strawberry.type
class Memory:
    """GraphQL Memory type."""
    id: str
    content: str
    content_type: str
    tier: str
    created_at: str
    last_accessed: Optional[str]
    access_count: int
    size_bytes: int
    compression_ratio: Optional[float]
    tags: List[str]
    metadata: str  # JSON string


@strawberry.type
class MemorySearchResult:
    """GraphQL Memory search result type."""
    memory: Memory
    relevance_score: float
    similarity_score: float
    rank: int


@strawberry.type
class MemoryUsageStats:
    """GraphQL Memory usage statistics type."""
    total_memories: int
    total_size_bytes: int
    memories_by_tier: str  # JSON string
    avg_compression_ratio: float
    most_accessed_memory_id: Optional[str]
    least_accessed_memory_id: Optional[str]


@strawberry.type
class MemoryHealthScore:
    """GraphQL Memory health score type."""
    overall_score: float
    grade: str
    storage_efficiency: float
    access_patterns: float
    compression_effectiveness: float
    tier_optimization: float


@strawberry.type
class OptimizationRecommendation:
    """GraphQL Optimization recommendation type."""
    id: str
    type: str
    priority: str
    title: str
    description: str
    impact_description: str
    estimated_savings: str  # JSON string
    implementation_effort: str
    actionable_steps: List[str]
    confidence_score: float
    created_at: str


@strawberry.type
class UsageMetrics:
    """GraphQL Usage metrics type."""
    user_id: str
    time_period_hours: int
    total_events: int
    events_by_type: str  # JSON string
    total_size_processed: int
    avg_response_time_ms: float
    peak_usage_time: Optional[str]
    error_rate: float


@strawberry.type
class TenantInfo:
    """GraphQL Tenant information type."""
    id: str
    name: str
    status: str
    subscription_tier: str
    created_at: str
    user_count: int
    memory_count: int
    storage_usage_bytes: int


# Input Types
@strawberry.input
class MemoryInput:
    """GraphQL Memory input type."""
    content: str
    content_type: str = "text"
    tags: Optional[List[str]] = None
    metadata: Optional[str] = None  # JSON string


@strawberry.input
class MemoryUpdateInput:
    """GraphQL Memory update input type."""
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[str] = None  # JSON string


@strawberry.input
class SearchInput:
    """GraphQL Search input type."""
    query: str
    limit: Optional[int] = 10
    tier_filter: Optional[str] = None
    content_type_filter: Optional[str] = None
    min_relevance_score: Optional[float] = 0.0
    include_metadata: Optional[bool] = True


@strawberry.input
class AnalyticsInput:
    """GraphQL Analytics input type."""
    days: Optional[int] = 30
    include_insights: Optional[bool] = True
    include_recommendations: Optional[bool] = False


# Query and Mutation Classes
@strawberry.type
class Query:
    """GraphQL Query root."""

    @strawberry.field
    async def memory(self, info: Info, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        try:
            context = get_tenant_isolation().require_tenant_context()
            memory_manager = info.context.get("memory_manager")

            if not memory_manager:
                raise Exception("Memory manager not available")

            memory_data = memory_manager.get_memory(memory_id)
            if not memory_data:
                return None

            return Memory(
                id=memory_data["id"],
                content=memory_data["content"],
                content_type=memory_data.get("content_type", "text"),
                tier=memory_data.get("tier", "unknown"),
                created_at=memory_data.get("created_at", ""),
                last_accessed=memory_data.get("last_accessed"),
                access_count=memory_data.get("access_count", 0),
                size_bytes=memory_data.get("size_bytes", 0),
                compression_ratio=memory_data.get("compression_ratio"),
                tags=memory_data.get("tags", []),
                metadata=str(memory_data.get("metadata", {}))
            )

        except Exception as e:
            logger.error(f"GraphQL memory query failed: {e}")
            raise

    @strawberry.field
    async def memories(
        self,
        info: Info,
        limit: Optional[int] = 10,
        offset: Optional[int] = 0,
        tier_filter: Optional[str] = None
    ) -> List[Memory]:
        """Get a list of memories with optional filtering."""
        try:
            context = get_tenant_isolation().require_tenant_context()
            memory_manager = info.context.get("memory_manager")

            if not memory_manager:
                raise Exception("Memory manager not available")

            # Get memories with filtering
            memories_data = memory_manager.list_memories(
                limit=limit,
                offset=offset,
                tier_filter=tier_filter
            )

            return [
                Memory(
                    id=mem["id"],
                    content=mem["content"],
                    content_type=mem.get("content_type", "text"),
                    tier=mem.get("tier", "unknown"),
                    created_at=mem.get("created_at", ""),
                    last_accessed=mem.get("last_accessed"),
                    access_count=mem.get("access_count", 0),
                    size_bytes=mem.get("size_bytes", 0),
                    compression_ratio=mem.get("compression_ratio"),
                    tags=mem.get("tags", []),
                    metadata=str(mem.get("metadata", {}))
                )
                for mem in memories_data
            ]

        except Exception as e:
            logger.error(f"GraphQL memories query failed: {e}")
            raise

    @strawberry.field
    async def search_memories(self, info: Info, search_input: SearchInput) -> List[MemorySearchResult]:
        """Search memories with advanced parameters."""
        try:
            context = get_tenant_isolation().require_tenant_context()
            memory_manager = info.context.get("memory_manager")

            if not memory_manager:
                raise Exception("Memory manager not available")

            # Perform search
            search_results = memory_manager.search_memories(
                query=search_input.query,
                limit=search_input.limit,
                tier_filter=search_input.tier_filter,
                content_type_filter=search_input.content_type_filter,
                min_relevance_score=search_input.min_relevance_score
            )

            return [
                MemorySearchResult(
                    memory=Memory(
                        id=result["memory"]["id"],
                        content=result["memory"]["content"],
                        content_type=result["memory"].get("content_type", "text"),
                        tier=result["memory"].get("tier", "unknown"),
                        created_at=result["memory"].get("created_at", ""),
                        last_accessed=result["memory"].get("last_accessed"),
                        access_count=result["memory"].get("access_count", 0),
                        size_bytes=result["memory"].get("size_bytes", 0),
                        compression_ratio=result["memory"].get("compression_ratio"),
                        tags=result["memory"].get("tags", []),
                        metadata=str(result["memory"].get("metadata", {}))
                    ),
                    relevance_score=result.get("relevance_score", 0.0),
                    similarity_score=result.get("similarity_score", 0.0),
                    rank=result.get("rank", 0)
                )
                for result in search_results
            ]

        except Exception as e:
            logger.error(f"GraphQL search failed: {e}")
            raise

    @strawberry.field
    async def memory_analytics(self, info: Info, analytics_input: AnalyticsInput) -> MemoryUsageStats:
        """Get memory usage analytics."""
        try:
            context = get_tenant_isolation().require_tenant_context()
            analytics = info.context.get("analytics")

            if not analytics:
                raise Exception("Analytics not available")

            report = analytics.generate_usage_report(
                user_id=context.user_id,
                days=analytics_input.days,
                include_insights=analytics_input.include_insights
            )

            return MemoryUsageStats(
                total_memories=report.total_memories,
                total_size_bytes=report.total_size_bytes,
                memories_by_tier=str(report.memories_by_tier),
                avg_compression_ratio=report.compression_stats.get("overall_compression_ratio", 1.0),
                most_accessed_memory_id=report.access_patterns.get("most_accessed_memory"),
                least_accessed_memory_id=report.access_patterns.get("least_accessed_memory")
            )

        except Exception as e:
            logger.error(f"GraphQL analytics query failed: {e}")
            raise

    @strawberry.field
    async def memory_health_score(self, info: Info) -> MemoryHealthScore:
        """Get memory health score."""
        try:
            context = get_tenant_isolation().require_tenant_context()
            analytics = info.context.get("analytics")

            if not analytics:
                raise Exception("Analytics not available")

            health_score = analytics.calculate_health_score(context.user_id)

            return MemoryHealthScore(
                overall_score=health_score["overall_score"],
                grade=health_score["grade"],
                storage_efficiency=health_score["component_scores"]["storage_efficiency"],
                access_patterns=health_score["component_scores"]["access_patterns"],
                compression_effectiveness=health_score["component_scores"]["compression_effectiveness"],
                tier_optimization=health_score["component_scores"]["tier_optimization"]
            )

        except Exception as e:
            logger.error(f"GraphQL health score query failed: {e}")
            raise

    @strawberry.field
    async def optimization_recommendations(self, info: Info) -> List[OptimizationRecommendation]:
        """Get optimization recommendations."""
        try:
            context = get_tenant_isolation().require_tenant_context()
            advisor = info.context.get("optimization_advisor")

            if not advisor:
                raise Exception("Optimization advisor not available")

            recommendations = advisor.generate_recommendations(context.user_id)

            return [
                OptimizationRecommendation(
                    id=rec.id,
                    type=rec.type.value,
                    priority=rec.priority.value,
                    title=rec.title,
                    description=rec.description,
                    impact_description=rec.impact_description,
                    estimated_savings=str(rec.estimated_savings),
                    implementation_effort=rec.implementation_effort,
                    actionable_steps=rec.actionable_steps,
                    confidence_score=rec.confidence_score,
                    created_at=rec.created_at.isoformat()
                )
                for rec in recommendations
            ]

        except Exception as e:
            logger.error(f"GraphQL recommendations query failed: {e}")
            raise

    @strawberry.field
    async def usage_metrics(self, info: Info, hours: Optional[int] = 24) -> UsageMetrics:
        """Get usage metrics."""
        try:
            context = get_tenant_isolation().require_tenant_context()
            usage_tracker = info.context.get("usage_tracker")

            if not usage_tracker:
                raise Exception("Usage tracker not available")

            metrics = usage_tracker.get_user_metrics(context.user_id, hours)

            return UsageMetrics(
                user_id=metrics.user_id,
                time_period_hours=hours,
                total_events=metrics.total_events,
                events_by_type=str(metrics.events_by_type),
                total_size_processed=metrics.total_size_processed,
                avg_response_time_ms=metrics.avg_response_time_ms,
                peak_usage_time=metrics.peak_usage_time.isoformat() if metrics.peak_usage_time else None,
                error_rate=metrics.error_rate
            )

        except Exception as e:
            logger.error(f"GraphQL usage metrics query failed: {e}")
            raise

    @strawberry.field
    async def tenant_info(self, info: Info) -> TenantInfo:
        """Get current tenant information."""
        try:
            context = get_tenant_isolation().require_tenant_context()
            tenant_manager = info.context.get("tenant_manager")

            if not tenant_manager:
                raise Exception("Tenant manager not available")

            tenant = tenant_manager.get_tenant(context.tenant_id)
            if not tenant:
                raise Exception("Tenant not found")

            usage = tenant_manager.get_tenant_usage(context.tenant_id)

            return TenantInfo(
                id=tenant.id,
                name=tenant.name,
                status=tenant.status.value,
                subscription_tier=tenant.subscription_tier.value,
                created_at=tenant.created_at.isoformat(),
                user_count=usage["users"]["current"],
                memory_count=usage["memories"]["total"],
                storage_usage_bytes=usage["storage"]["current_bytes"]
            )

        except Exception as e:
            logger.error(f"GraphQL tenant info query failed: {e}")
            raise


@strawberry.type
class Mutation:
    """GraphQL Mutation root."""

    @strawberry.mutation
    async def create_memory(self, info: Info, memory_input: MemoryInput) -> Memory:
        """Create a new memory."""
        try:
            context = get_tenant_isolation().require_tenant_context()
            memory_manager = info.context.get("memory_manager")

            if not memory_manager:
                raise Exception("Memory manager not available")

            import json
            metadata = json.loads(memory_input.metadata) if memory_input.metadata else {}

            memory_data = memory_manager.create_memory(
                content=memory_input.content,
                content_type=memory_input.content_type,
                tags=memory_input.tags or [],
                metadata=metadata
            )

            return Memory(
                id=memory_data["id"],
                content=memory_data["content"],
                content_type=memory_data.get("content_type", "text"),
                tier=memory_data.get("tier", "unknown"),
                created_at=memory_data.get("created_at", ""),
                last_accessed=memory_data.get("last_accessed"),
                access_count=memory_data.get("access_count", 0),
                size_bytes=memory_data.get("size_bytes", 0),
                compression_ratio=memory_data.get("compression_ratio"),
                tags=memory_data.get("tags", []),
                metadata=str(memory_data.get("metadata", {}))
            )

        except Exception as e:
            logger.error(f"GraphQL create memory failed: {e}")
            raise

    @strawberry.mutation
    async def update_memory(self, info: Info, memory_id: str, memory_update: MemoryUpdateInput) -> Optional[Memory]:
        """Update an existing memory."""
        try:
            context = get_tenant_isolation().require_tenant_context()
            memory_manager = info.context.get("memory_manager")

            if not memory_manager:
                raise Exception("Memory manager not available")

            import json
            update_data = {}
            if memory_update.content is not None:
                update_data["content"] = memory_update.content
            if memory_update.tags is not None:
                update_data["tags"] = memory_update.tags
            if memory_update.metadata is not None:
                update_data["metadata"] = json.loads(memory_update.metadata)

            memory_data = memory_manager.update_memory(memory_id, update_data)

            if not memory_data:
                return None

            return Memory(
                id=memory_data["id"],
                content=memory_data["content"],
                content_type=memory_data.get("content_type", "text"),
                tier=memory_data.get("tier", "unknown"),
                created_at=memory_data.get("created_at", ""),
                last_accessed=memory_data.get("last_accessed"),
                access_count=memory_data.get("access_count", 0),
                size_bytes=memory_data.get("size_bytes", 0),
                compression_ratio=memory_data.get("compression_ratio"),
                tags=memory_data.get("tags", []),
                metadata=str(memory_data.get("metadata", {}))
            )

        except Exception as e:
            logger.error(f"GraphQL update memory failed: {e}")
            raise

    @strawberry.mutation
    async def delete_memory(self, info: Info, memory_id: str) -> bool:
        """Delete a memory."""
        try:
            context = get_tenant_isolation().require_tenant_context()
            memory_manager = info.context.get("memory_manager")

            if not memory_manager:
                raise Exception("Memory manager not available")

            return memory_manager.delete_memory(memory_id)

        except Exception as e:
            logger.error(f"GraphQL delete memory failed: {e}")
            raise

    @strawberry.mutation
    async def compress_memory(self, info: Info, memory_id: str) -> bool:
        """Manually trigger memory compression."""
        try:
            context = get_tenant_isolation().require_tenant_context()
            memory_manager = info.context.get("memory_manager")

            if not memory_manager:
                raise Exception("Memory manager not available")

            return memory_manager.compress_memory(memory_id)

        except Exception as e:
            logger.error(f"GraphQL compress memory failed: {e}")
            raise

    @strawberry.mutation
    async def archive_memory(self, info: Info, memory_id: str) -> bool:
        """Archive a memory."""
        try:
            context = get_tenant_isolation().require_tenant_context()
            memory_manager = info.context.get("memory_manager")

            if not memory_manager:
                raise Exception("Memory manager not available")

            return memory_manager.archive_memory(memory_id)

        except Exception as e:
            logger.error(f"GraphQL archive memory failed: {e}")
            raise


class MemorizerGraphQLAPI:
    """GraphQL API for the Memorizer framework."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        analytics: MemoryAnalytics,
        usage_tracker: UsageTracker,
        optimization_advisor: OptimizationAdvisor,
        tenant_manager
    ):
        """Initialize GraphQL API."""
        self.memory_manager = memory_manager
        self.analytics = analytics
        self.usage_tracker = usage_tracker
        self.optimization_advisor = optimization_advisor
        self.tenant_manager = tenant_manager

        self.schema = create_graphql_schema()

    def create_context(self) -> Dict[str, Any]:
        """Create GraphQL context with dependencies."""
        return {
            "memory_manager": self.memory_manager,
            "analytics": self.analytics,
            "usage_tracker": self.usage_tracker,
            "optimization_advisor": self.optimization_advisor,
            "tenant_manager": self.tenant_manager
        }

    def get_router(self) -> GraphQLRouter:
        """Get FastAPI GraphQL router."""
        return GraphQLRouter(
            self.schema,
            context_getter=self.create_context,
            path="/graphql"
        )


def create_graphql_schema():
    """Create GraphQL schema."""
    return strawberry.Schema(
        query=Query,
        mutation=Mutation
    )