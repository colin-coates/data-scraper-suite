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
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field

# Using dataclasses for compatibility

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
            "latency_ms": self.latency_ms
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
    Collector for scrape telemetry data.

    Handles collection, storage, aggregation, and analysis of telemetry data.
    Provides real-time metrics and historical analysis.
    """

    def __init__(self, max_history: int = 10000):
        self.telemetry_data: deque[ScrapeTelemetry] = deque(maxlen=max_history)
        self.metrics = TelemetryMetrics()
        self._lock = asyncio.Lock()
        logger.info(f"ScrapeTelemetryCollector initialized with max_history={max_history}")

    async def record_telemetry(self, telemetry: ScrapeTelemetry) -> None:
        """
        Record a telemetry data point.

        Args:
            telemetry: Telemetry data to record
        """
        async with self._lock:
            self.telemetry_data.append(telemetry)
            self._update_metrics(telemetry)
            logger.debug(f"Recorded telemetry: {telemetry.source} - {telemetry.records_found} records")

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
        self.metrics = TelemetryMetrics()
        logger.info("Telemetry data cleared")


# Global telemetry collector instance
_global_collector = ScrapeTelemetryCollector()


def get_global_collector() -> ScrapeTelemetryCollector:
    """Get the global telemetry collector instance."""
    return _global_collector


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
