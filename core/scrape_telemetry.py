# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Scrape Telemetry for MJ Data Scraper Suite

Collects, stores, and analyzes telemetry data from scraping operations.
Provides insights into performance, costs, and operational metrics.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field

# Import comprehensive telemetry models
from .telemetry.models import (
    TelemetryEvent,
    ScraperOperationEvent,
    SentinelCheckEvent,
    SafetyVerdictEvent,
    ConstraintApplicationEvent,
    ErrorEvent,
    PerformanceMetricEvent,
    TelemetrySeverity,
    create_scraper_operation_event,
    create_sentinel_check_event,
    create_safety_verdict_event,
    create_error_event,
    create_performance_metric_event
)

logger = logging.getLogger(__name__)


@dataclass
class ScrapeTelemetry:
    """Telemetry data model for scraping operations."""
    timestamp: datetime
    source: str
    hour_of_day: int
    day_of_week: str
    cost: float
    records_found: int
    blocked: bool = False
    latency_ms: int = 0
    scraper_name: str = ""  # Name of the scraper instance
    role: str = ""          # Scraper role (discovery/verification/enrichment/browser)
    blocked_reason: str = "" # Reason for blocking if applicable

    def __post_init__(self):
        """Validate telemetry data."""
        if not (0 <= self.hour_of_day <= 23):
            raise ValueError("hour_of_day must be between 0 and 23")

        if self.cost < 0:
            raise ValueError("cost must be non-negative")

        if self.records_found < 0:
            raise ValueError("records_found must be non-negative")

        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")

        valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if self.day_of_week not in valid_days:
            raise ValueError(f"day_of_week must be one of {valid_days}")

    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (records per dollar)."""
        return self.records_found / max(0.01, self.cost)

    @property
    def is_business_hours(self) -> bool:
        """Check if operation occurred during business hours."""
        return 9 <= self.hour_of_day <= 17 and self.day_of_week not in ["Saturday", "Sunday"]

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary (compatibility method)."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
            "cost": self.cost,
            "records_found": self.records_found,
            "blocked": self.blocked,
            "latency_ms": self.latency_ms,
            "scraper_name": self.scraper_name,
            "role": self.role,
            "blocked_reason": self.blocked_reason
        }


@dataclass
class TelemetryMetrics:
    """Aggregated telemetry metrics."""
    total_operations: int = 0
    total_cost: float = 0.0
    total_records: int = 0
    total_latency_ms: int = 0
    blocked_operations: int = 0
    source_counts: Dict[str, int] = field(default_factory=dict)
    hourly_distribution: Dict[int, int] = field(default_factory=dict)
    daily_distribution: Dict[str, int] = field(default_factory=dict)

    @property
    def average_cost_per_operation(self) -> float:
        """Calculate average cost per operation."""
        return self.total_cost / max(1, self.total_operations)

    @property
    def average_records_per_operation(self) -> float:
        """Calculate average records per operation."""
        return self.total_records / max(1, self.total_operations)

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        return self.total_latency_ms / max(1, self.total_operations)

    @property
    def block_rate(self) -> float:
        """Calculate rate of blocked operations."""
        return self.blocked_operations / max(1, self.total_operations)

    @property
    def efficiency_score(self) -> float:
        """Calculate overall efficiency score."""
        return self.total_records / max(0.01, self.total_cost)


class ScrapeTelemetryCollector:
    """
    Enhanced collector for comprehensive telemetry data.

    Handles collection, storage, aggregation, and analysis of telemetry data
    from multiple sources including scrapers, sentinels, safety verdicts, and more.
    Provides real-time metrics, historical analysis, and enterprise observability.
    """

    def __init__(self, max_history: int = 10000):
        # Legacy telemetry storage
        self.telemetry_data: deque[ScrapeTelemetry] = deque(maxlen=max_history)

        # Comprehensive event storage
        self.event_data: deque[TelemetryEvent] = deque(maxlen=max_history)

        # Component-specific storage for better organization
        self.scraper_events: deque[ScraperOperationEvent] = deque(maxlen=max_history // 4)
        self.sentinel_events: deque[SentinelCheckEvent] = deque(maxlen=max_history // 4)
        self.verdict_events: deque[SafetyVerdictEvent] = deque(maxlen=max_history // 4)
        self.error_events: deque[ErrorEvent] = deque(maxlen=max_history // 4)

        self.metrics = TelemetryMetrics()
        self._lock = asyncio.Lock()

        logger.info(f"ScrapeTelemetryCollector initialized with max_history={max_history}")

    async def record_telemetry(self, telemetry: ScrapeTelemetry) -> None:
        """
        Record a legacy telemetry data point.

        Args:
            telemetry: Telemetry data to record
        """
        async with self._lock:
            self.telemetry_data.append(telemetry)
            self._update_metrics(telemetry)

            # Log significant events
            if telemetry.blocked:
                logger.warning(f"Blocked scraping operation: {telemetry.scraper_name} - {telemetry.blocked_reason}")

            if telemetry.cost > 1.0:  # Log high-cost operations
                logger.info(f"High-cost operation: {telemetry.scraper_name} cost=${telemetry.cost:.2f}")

            logger.debug(f"Recorded legacy telemetry: {telemetry.scraper_name} records={telemetry.records_found} cost=${telemetry.cost:.3f}")

    async def record_event(self, event: TelemetryEvent) -> None:
        """
        Record a comprehensive telemetry event.

        Args:
            event: Telemetry event to record
        """
        async with self._lock:
            # Store in general event queue
            self.event_data.append(event)

            # Route to component-specific queues
            if isinstance(event, ScraperOperationEvent):
                self.scraper_events.append(event)
                # Also update legacy metrics for compatibility
                self._update_metrics_from_scraper_event(event)

            elif isinstance(event, SentinelCheckEvent):
                self.sentinel_events.append(event)

            elif isinstance(event, SafetyVerdictEvent):
                self.verdict_events.append(event)

            elif isinstance(event, ErrorEvent):
                self.error_events.append(event)

            # Log significant events
            await self._log_significant_event(event)

            logger.debug(f"Recorded telemetry event: {event.event_type.value} from {event.source_component}")

    def _update_metrics_from_scraper_event(self, event: ScraperOperationEvent) -> None:
        """Update legacy metrics from new scraper operation events."""
        # Ensure timestamp is set
        timestamp = event.timestamp or datetime.utcnow()

        # Create a synthetic ScrapeTelemetry for backward compatibility
        synthetic_telemetry = ScrapeTelemetry(
            timestamp=timestamp,
            source=event.source_component,
            hour_of_day=timestamp.hour,
            day_of_week=timestamp.strftime("%A"),
            cost=event.cost_estimate,
            records_found=event.records_found,
            blocked=event.operation_status == "blocked",
            latency_ms=int(event.processing_time * 1000),
            scraper_name=event.scraper_type,
            role=event.scraper_role or "",
            blocked_reason=event.blocked_reason or ""
        )

        self._update_metrics(synthetic_telemetry)

    async def _log_significant_event(self, event: TelemetryEvent) -> None:
        """Log significant telemetry events."""
        if isinstance(event, ScraperOperationEvent):
            if event.operation_status == "blocked":
                logger.warning(f"🚫 Blocked scraper operation: {event.scraper_type} - {event.blocked_reason}")
            elif event.cost_estimate > 1.0:
                logger.info(f"💰 High-cost scraper operation: {event.scraper_type} cost=${event.cost_estimate:.2f}")

        elif isinstance(event, SentinelCheckEvent):
            if event.risk_level == "critical":
                logger.error(f"🚨 Critical sentinel risk detected: {event.sentinel_name} - {event.critical_findings}")
            elif event.risk_level == "high":
                logger.warning(f"⚠️ High sentinel risk detected: {event.sentinel_name}")

        elif isinstance(event, SafetyVerdictEvent):
            if event.verdict_action == "block":
                logger.warning(f"🚫 Safety verdict blocked operation: {event.verdict_reason}")
            elif event.verdict_action == "human_required":
                logger.info(f"👤 Human approval required: {event.verdict_reason}")

        elif isinstance(event, ErrorEvent):
            if event.severity == TelemetrySeverity.CRITICAL:
                logger.critical(f"💥 Critical error: {event.error_type} - {event.error_message}")
            elif event.severity == TelemetrySeverity.ERROR:
                logger.error(f"❌ Error: {event.error_type} - {event.error_message}")
            elif event.severity == TelemetrySeverity.WARNING:
                logger.warning(f"⚠️ Warning: {event.error_type} - {event.error_message}")

    def _update_metrics(self, telemetry: ScrapeTelemetry) -> None:
        """Update aggregated metrics with new telemetry data."""
        self.metrics.total_operations += 1
        self.metrics.total_cost += telemetry.cost
        self.metrics.total_records += telemetry.records_found
        self.metrics.total_latency_ms += telemetry.latency_ms

        if telemetry.blocked:
            self.metrics.blocked_operations += 1

        # Update source counts
        self.metrics.source_counts[telemetry.source] = \
            self.metrics.source_counts.get(telemetry.source, 0) + 1

        # Update time distributions
        self.metrics.hourly_distribution[telemetry.hour_of_day] = \
            self.metrics.hourly_distribution.get(telemetry.hour_of_day, 0) + 1

        self.metrics.daily_distribution[telemetry.day_of_week] = \
            self.metrics.daily_distribution.get(telemetry.day_of_week, 0) + 1

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current aggregated metrics."""
        return {
            "total_operations": self.metrics.total_operations,
            "total_cost": self.metrics.total_cost,
            "total_records": self.metrics.total_records,
            "average_cost_per_operation": self.metrics.average_cost_per_operation,
            "average_records_per_operation": self.metrics.average_records_per_operation,
            "average_latency_ms": self.metrics.average_latency_ms,
            "block_rate": self.metrics.block_rate,
            "efficiency_score": self.metrics.efficiency_score,
            "source_counts": dict(self.metrics.source_counts),
            "hourly_distribution": dict(self.metrics.hourly_distribution),
            "daily_distribution": dict(self.metrics.daily_distribution),
            "data_points": len(self.telemetry_data)
        }

    def get_telemetry_by_source(self, source: str, limit: Optional[int] = None) -> List[ScrapeTelemetry]:
        """Get telemetry data for a specific source."""
        source_data = [t for t in self.telemetry_data if t.source == source]
        return source_data[-limit:] if limit else source_data

    def get_telemetry_in_timeframe(self, start_time: datetime, end_time: Optional[datetime] = None) -> List[ScrapeTelemetry]:
        """Get telemetry data within a time range."""
        end_time = end_time or datetime.utcnow()
        return [
            t for t in self.telemetry_data
            if start_time <= t.timestamp <= end_time
        ]

    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance trends over recent hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_data = [t for t in self.telemetry_data if t.timestamp >= cutoff_time]

        if not recent_data:
            return {"error": "No data in specified timeframe"}

        # Calculate trends
        total_cost = sum(t.cost for t in recent_data)
        total_records = sum(t.records_found for t in recent_data)
        total_latency = sum(t.latency_ms for t in recent_data)
        blocked_count = sum(1 for t in recent_data if t.blocked)

        return {
            "timeframe_hours": hours,
            "data_points": len(recent_data),
            "total_cost": total_cost,
            "total_records": total_records,
            "average_latency_ms": total_latency / len(recent_data),
            "block_rate": blocked_count / len(recent_data),
            "efficiency_trend": total_records / max(0.01, total_cost)
        }

    def get_anomaly_detection(self, threshold_stddev: float = 2.0) -> Dict[str, Any]:
        """Detect anomalies in telemetry data."""
        if len(self.telemetry_data) < 10:
            return {"error": "Insufficient data for anomaly detection"}

        # Calculate baseline metrics
        costs = [t.cost for t in self.telemetry_data]
        latencies = [t.latency_ms for t in self.telemetry_data]

        avg_cost = sum(costs) / len(costs)
        avg_latency = sum(latencies) / len(latencies)

        # Simple standard deviation calculation
        cost_variance = sum((c - avg_cost) ** 2 for c in costs) / len(costs)
        latency_variance = sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)

        cost_stddev = cost_variance ** 0.5
        latency_stddev = latency_variance ** 0.5

        # Find anomalies
        cost_anomalies = [t for t in self.telemetry_data if abs(t.cost - avg_cost) > threshold_stddev * cost_stddev]
        latency_anomalies = [t for t in self.telemetry_data if abs(t.latency_ms - avg_latency) > threshold_stddev * latency_stddev]

        return {
            "cost_anomalies": len(cost_anomalies),
            "latency_anomalies": len(latency_anomalies),
            "threshold_stddev": threshold_stddev,
            "baseline_cost": avg_cost,
            "baseline_latency": avg_latency,
            "cost_stddev": cost_stddev,
            "latency_stddev": latency_stddev
        }

    def export_telemetry_data(self, format: str = "json") -> str:
        """
        Export telemetry data in specified format.

        Args:
            format: Export format ("json" or "csv")

        Returns:
            Exported data as string
        """
        if format == "json":
            import json
            return json.dumps(
                [t.dict() for t in self.telemetry_data],
                indent=2,
                default=str
            )
        elif format == "csv":
            import csv
            import io

            output = io.StringIO()
            if self.telemetry_data:
                fieldnames = list(self.telemetry_data[0].dict().keys())
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for telemetry in self.telemetry_data:
                    writer.writerow(telemetry.dict())
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def clear_data(self) -> None:
        """Clear all telemetry data and reset metrics."""
        self.telemetry_data.clear()
        self.event_data.clear()
        self.scraper_events.clear()
        self.sentinel_events.clear()
        self.verdict_events.clear()
        self.error_events.clear()
        self.metrics = TelemetryMetrics()
        logger.info("Telemetry data cleared")

    # Comprehensive telemetry event access methods
    def get_scraper_events(self, limit: Optional[int] = None) -> List[ScraperOperationEvent]:
        """Get scraper operation events."""
        events = list(self.scraper_events)
        return events[-limit:] if limit else events

    def get_sentinel_events(self, limit: Optional[int] = None) -> List[SentinelCheckEvent]:
        """Get sentinel check events."""
        events = list(self.sentinel_events)
        return events[-limit:] if limit else events

    def get_verdict_events(self, limit: Optional[int] = None) -> List[SafetyVerdictEvent]:
        """Get safety verdict events."""
        events = list(self.verdict_events)
        return events[-limit:] if limit else events

    def get_error_events(self, limit: Optional[int] = None) -> List[ErrorEvent]:
        """Get error events."""
        events = list(self.error_events)
        return events[-limit:] if limit else events

    def get_events_by_type(self, event_type: str, limit: Optional[int] = None) -> List[TelemetryEvent]:
        """Get events of a specific type."""
        matching_events = [e for e in self.event_data if e.event_type.value == event_type]
        return matching_events[-limit:] if limit else matching_events

    def get_events_in_timeframe(self, start_time: datetime, end_time: Optional[datetime] = None) -> List[TelemetryEvent]:
        """Get telemetry events within a time range."""
        end_time = end_time or datetime.utcnow()
        return [
            e for e in self.event_data
            if start_time <= e.timestamp <= end_time
        ]

    def get_component_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all telemetry components."""
        now = datetime.utcnow()
        last_hour = now - timedelta(hours=1)

        return {
            "scraper_operations": {
                "total": len(self.scraper_events),
                "last_hour": len([e for e in self.scraper_events if e.timestamp >= last_hour]),
                "blocked_operations": len([e for e in self.scraper_events if e.operation_status == "blocked"]),
                "high_cost_operations": len([e for e in self.scraper_events if e.cost_estimate > 1.0])
            },
            "sentinel_checks": {
                "total": len(self.sentinel_events),
                "last_hour": len([e for e in self.sentinel_events if e.timestamp >= last_hour]),
                "critical_findings": len([e for e in self.sentinel_events if e.risk_level == "critical"]),
                "high_risk_findings": len([e for e in self.sentinel_events if e.risk_level == "high"])
            },
            "safety_verdicts": {
                "total": len(self.verdict_events),
                "last_hour": len([e for e in self.verdict_events if e.timestamp >= last_hour]),
                "blocks_issued": len([e for e in self.verdict_events if e.verdict_action == "block"]),
                "human_required": len([e for e in self.verdict_events if e.verdict_action == "human_required"])
            },
            "error_events": {
                "total": len(self.error_events),
                "last_hour": len([e for e in self.error_events if e.timestamp >= last_hour]),
                "critical_errors": len([e for e in self.error_events if e.severity == TelemetrySeverity.CRITICAL]),
                "by_component": self._get_error_summary_by_component()
            }
        }

    def _get_error_summary_by_component(self) -> Dict[str, int]:
        """Get error count by component."""
        component_errors = defaultdict(int)
        for error in self.error_events:
            component_errors[error.source_component] += 1
        return dict(component_errors)


# Global telemetry collector instances
_global_collector = ScrapeTelemetryCollector()
_global_telemetry_collector = _global_collector  # For backward compatibility


def get_global_collector() -> ScrapeTelemetryCollector:
    """Get the global telemetry collector instance."""
    return _global_collector


def get_global_telemetry_collector() -> ScrapeTelemetryCollector:
    """Get the global telemetry collector instance (alias for backward compatibility)."""
    return _global_telemetry_collector


async def record_scrape_telemetry(
    source: str,
    cost: float,
    records_found: int,
    blocked: bool = False,
    latency_ms: Optional[int] = None
) -> None:
    """
    Convenience function to record scrape telemetry.

    Args:
        source: Source system or scraper name
        cost: Cost incurred
        records_found: Number of records found
        blocked: Whether operation was blocked
        latency_ms: Operation latency (auto-calculated if None)
    """
    if latency_ms is None:
        # Calculate approximate latency if not provided
        latency_ms = int(time.time() * 1000) % 10000  # Placeholder

    timestamp = datetime.utcnow()
    hour_of_day = timestamp.hour
    day_of_week = timestamp.strftime("%A")

    telemetry = ScrapeTelemetry(
        timestamp=timestamp,
        source=source,
        hour_of_day=hour_of_day,
        day_of_week=day_of_week,
        cost=cost,
        records_found=records_found,
        blocked=blocked,
        latency_ms=latency_ms
    )

    await _global_collector.record_telemetry(telemetry)


def get_telemetry_summary() -> Dict[str, Any]:
    """Get a summary of current telemetry data."""
    return _global_collector.get_current_metrics()


async def record_telemetry(entry: ScrapeTelemetry) -> None:
    """
    Record telemetry entry to database/table.

    This function persists the telemetry data to a persistent storage layer.
    In a production implementation, this would write to a database, data warehouse,
    or telemetry service.

    Args:
        entry: ScrapeTelemetry entry to persist
    """
    try:
        # TODO: Implement actual database persistence
        # This is a placeholder for database/table persistence logic

        # For now, delegate to the global collector (in-memory storage)
        await _global_collector.record_telemetry(entry)

        # Example database persistence (uncomment and implement as needed):
        # await persist_to_database(entry)
        # await send_to_telemetry_service(entry)
        # await write_to_data_warehouse(entry)

        logger.info(f"Telemetry recorded: {entry.source} - {entry.records_found} records, ${entry.cost}")

    except Exception as e:
        logger.error(f"Failed to record telemetry: {e}")
        # In production, you might want to:
        # - Retry the operation
        # - Write to a dead letter queue
        # - Send alerts
        raise


def persist_to_database(entry: ScrapeTelemetry) -> None:
    """
    Placeholder for database persistence implementation.

    Args:
        entry: Telemetry entry to persist
    """
    # TODO: Implement actual database persistence
    # Examples:
    # - SQL database (PostgreSQL, MySQL)
    # - NoSQL database (MongoDB, Cassandra)
    # - Time-series database (InfluxDB, TimescaleDB)
    # - Cloud database (BigQuery, Redshift)

    logger.debug(f"Would persist to database: {entry.dict()}")


def send_to_telemetry_service(entry: ScrapeTelemetry) -> None:
    """
    Placeholder for telemetry service integration.

    Args:
        entry: Telemetry entry to send
    """
    # TODO: Implement telemetry service integration
    # Examples:
    # - Application Insights
    # - DataDog
    # - New Relic
    # - Custom telemetry service

    logger.debug(f"Would send to telemetry service: {entry.dict()}")


def write_to_data_warehouse(entry: ScrapeTelemetry) -> None:
    """
    Placeholder for data warehouse integration.

    Args:
        entry: Telemetry entry to write
    """
    # TODO: Implement data warehouse integration
    # Examples:
    # - Snowflake
    # - Redshift
    # - BigQuery
    # - Custom data lake

    logger.debug(f"Would write to data warehouse: {entry.dict()}")


# Global telemetry collector instance
_global_telemetry_collector = ScrapeTelemetryCollector()


def get_global_telemetry_collector() -> ScrapeTelemetryCollector:
    """Get the global telemetry collector instance."""
    return _global_telemetry_collector


async def emit_telemetry(
    scraper: str,
    role: str = "",
    cost_estimate: float = 0.0,
    records_found: int = 0,
    blocked_reason: str = "",
    runtime: float = 0.0,
    **kwargs
) -> None:
    """
    Enhanced telemetry emission with comprehensive event modeling.

    Creates appropriate telemetry events based on context and records them
    in the global telemetry collector for analysis and monitoring.

    Args:
        scraper: Name of the scraper instance or component
        role: Scraper role or component role
        cost_estimate: Estimated cost of the operation
        records_found: Number of records collected/processed
        blocked_reason: Reason for blocking if applicable
        runtime: Execution time in seconds
        **kwargs: Additional context-specific parameters
    """
    # Create comprehensive scraper operation event using new models
    event = create_scraper_operation_event(
        scraper_type=scraper,
        scraper_role=role,
        records_found=records_found,
        processing_time=runtime,
        cost_estimate=cost_estimate,
        operation_status="blocked" if blocked_reason else "success",
        blocked_reason=blocked_reason if blocked_reason else None,
        source_component=kwargs.get('source_component', 'scraper_engine'),
        source_workflow=kwargs.get('workflow_id'),
        correlation_id=kwargs.get('correlation_id'),
        metadata=kwargs.get('metadata', {}),
        tags=kwargs.get('tags', []),
        **{k: v for k, v in kwargs.items() if k not in [
            'source_component', 'workflow_id', 'correlation_id',
            'metadata', 'tags', 'sentinel_name', 'risk_level',
            'verdict_action', 'error_type', 'metric_name'
        ]}
    )

    # Also create legacy ScrapeTelemetry for backward compatibility
    await _emit_legacy_telemetry(scraper, role, cost_estimate, records_found, blocked_reason, runtime)

    # Record comprehensive event in global collector
    await _global_telemetry_collector.record_event(event)


async def _emit_legacy_telemetry(
    scraper: str,
    role: str,
    cost_estimate: float,
    records_found: int,
    blocked_reason: str,
    runtime: float
) -> None:
    """Emit legacy ScrapeTelemetry for backward compatibility."""
    # Determine if operation was blocked
    blocked = bool(blocked_reason)

    # Calculate latency in milliseconds
    latency_ms = int(runtime * 1000)

    # Get current timestamp and time information
    now = datetime.utcnow()
    hour_of_day = now.hour
    day_of_week = now.strftime("%A")  # Monday, Tuesday, etc.

    # Create legacy telemetry entry
    telemetry = ScrapeTelemetry(
        timestamp=now,
        source=scraper,
        hour_of_day=hour_of_day,
        day_of_week=day_of_week,
        cost=cost_estimate,
        records_found=records_found,
        blocked=blocked,
        latency_ms=latency_ms,
        scraper_name=scraper,
        role=role,
        blocked_reason=blocked_reason
    )

    # Record legacy telemetry
    await _global_telemetry_collector.record_telemetry(telemetry)


# Enhanced telemetry emission functions for different event types
async def emit_sentinel_telemetry(
    sentinel_name: str,
    risk_level: str,
    recommended_action: str,
    check_duration: float,
    **kwargs
) -> None:
    """Emit telemetry for sentinel check operations."""
    event = create_sentinel_check_event(
        sentinel_name=sentinel_name,
        risk_level=risk_level,
        recommended_action=recommended_action,
        check_duration=check_duration,
        source_component="sentinel_orchestrator",
        **kwargs
    )
    await _global_telemetry_collector.record_event(event)


async def emit_verdict_telemetry(
    verdict_action: str,
    verdict_reason: str,
    risk_level: str,
    reports_analyzed: int,
    **kwargs
) -> None:
    """Emit telemetry for safety verdict operations."""
    event = create_safety_verdict_event(
        verdict_action=verdict_action,
        verdict_reason=verdict_reason,
        risk_level=risk_level,
        reports_analyzed=reports_analyzed,
        source_component="safety_verdict",
        **kwargs
    )
    await _global_telemetry_collector.record_event(event)


async def emit_error_telemetry(
    error_type: str,
    error_message: str,
    severity: TelemetrySeverity = TelemetrySeverity.ERROR,
    **kwargs
) -> None:
    """Emit telemetry for error events."""
    event = create_error_event(
        error_type=error_type,
        error_message=error_message,
        severity=severity,
        **kwargs
    )
    await _global_telemetry_collector.record_event(event)


async def emit_performance_telemetry(
    metric_name: str,
    metric_value: Union[int, float],
    metric_unit: str,
    **kwargs
) -> None:
    """Emit telemetry for performance metrics."""
    event = create_performance_metric_event(
        metric_name=metric_name,
        metric_value=metric_value,
        metric_unit=metric_unit,
        source_component=kwargs.get('component_name', 'performance_monitor'),
        **kwargs
    )
    await _global_telemetry_collector.record_event(event)

    # Optional: Send to external telemetry services
    send_to_telemetry_service(telemetry)
    write_to_data_warehouse(telemetry)

    logger.info(
        f"Emitted telemetry: {scraper} ({role}) - "
        f"{records_found} records, ${cost_estimate:.2f}, "
        f"{runtime:.2f}s" + (f" - BLOCKED: {blocked_reason}" if blocked else "")
    )
