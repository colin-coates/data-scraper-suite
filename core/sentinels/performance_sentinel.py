# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Performance Sentinel for MJ Data Scraper Suite

Monitors scraping performance metrics and alerts on degradation or anomalies.
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime, timedelta

from .base import SentinelBase, SentinelConfig, SentinelSeverity
from ..scrape_telemetry import get_global_telemetry_collector

logger = logging.getLogger(__name__)


class PerformanceSentinel(SentinelBase):
    """
    Monitors scraping performance and alerts on issues.

    Tracks metrics like success rates, response times, and throughput.
    Alerts when performance drops below acceptable thresholds.
    """

    def __init__(self, config: SentinelConfig):
        super().__init__(config)

        # Performance thresholds
        self.min_success_rate = 0.8  # 80% success rate minimum
        self.max_avg_response_time = 30.0  # 30 seconds maximum average
        self.min_records_per_minute = 10.0  # Minimum throughput

        # Historical tracking
        self.performance_history = []
        self.history_window = 10  # Keep last 10 measurements

    def get_sentinel_type(self) -> str:
        """Return the sentinel type."""
        return "performance"

    async def check_condition(self) -> tuple[bool, str, Dict[str, Any]]:
        """
        Check scraping performance metrics.

        Returns:
            tuple: (is_healthy, status_message, metrics_data)
        """
        try:
            collector = get_global_telemetry_collector()
            metrics = collector.get_current_metrics()

            if not metrics or metrics.get('total_operations', 0) == 0:
                return True, "No operations to evaluate", {}

            # Calculate performance metrics
            success_rate = metrics.get('average_cost_per_operation', 0)  # This is actually success rate
            total_ops = metrics.get('total_operations', 0)
            total_records = metrics.get('total_records', 0)
            avg_response_time = metrics.get('average_latency_ms', 0) / 1000  # Convert to seconds

            # Calculate records per minute (rough estimate)
            runtime_estimate = max(total_ops * 0.5, 60)  # Assume 0.5s per operation minimum
            records_per_minute = (total_records / runtime_estimate) * 60

            # Store in history for trend analysis
            current_performance = {
                'timestamp': datetime.utcnow(),
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'records_per_minute': records_per_minute,
                'total_operations': total_ops
            }

            self.performance_history.append(current_performance)
            if len(self.performance_history) > self.history_window:
                self.performance_history.pop(0)

            # Check performance thresholds
            issues = []

            if success_rate < self.min_success_rate:
                issues.append(".2%")

            if avg_response_time > self.max_avg_response_time:
                issues.append(".1f")

            if records_per_minute < self.min_records_per_minute and total_ops > 5:
                issues.append(".1f")

            # Check for performance degradation trends
            if len(self.performance_history) >= 3:
                recent = self.performance_history[-3:]
                success_trend = [p['success_rate'] for p in recent]
                response_trend = [p['avg_response_time'] for p in recent]

                # Check if success rate is declining
                if len(success_trend) == 3 and success_trend[0] > success_trend[1] > success_trend[2]:
                    decline = (success_trend[0] - success_trend[2]) / success_trend[0]
                    if decline > 0.1:  # 10% decline
                        issues.append(".1%")

                # Check if response time is increasing
                if len(response_trend) == 3 and response_trend[0] < response_trend[1] < response_trend[2]:
                    increase = (response_trend[2] - response_trend[0]) / response_trend[0]
                    if increase > 0.5:  # 50% increase
                        issues.append(".1%")

            if issues:
                message = f"Performance issues detected: {'; '.join(issues)}"
                details = {
                    'success_rate': success_rate,
                    'avg_response_time': avg_response_time,
                    'records_per_minute': records_per_minute,
                    'total_operations': total_ops,
                    'issues': issues
                }
                return False, message, details
            else:
                message = f"Performance healthy: {total_ops} operations, {success_rate:.1%} success rate"
                details = {
                    'success_rate': success_rate,
                    'avg_response_time': avg_response_time,
                    'records_per_minute': records_per_minute,
                    'total_operations': total_ops
                }
                return True, message, details

        except Exception as e:
            return False, f"Performance check failed: {str(e)}", {'error': str(e)}

    def update_thresholds(self, min_success_rate: float = None, max_avg_response_time: float = None,
                         min_records_per_minute: float = None) -> None:
        """Update performance thresholds."""
        if min_success_rate is not None:
            self.min_success_rate = min_success_rate
        if max_avg_response_time is not None:
            self.max_avg_response_time = max_avg_response_time
        if min_records_per_minute is not None:
            self.min_records_per_minute = min_records_per_minute

        logger.info(f"Updated performance thresholds for {self.config.name}: "
                   f"success_rate={self.min_success_rate}, "
                   f"response_time={self.max_avg_response_time}, "
                   f"records_per_minute={self.min_records_per_minute}")

    def get_performance_history(self) -> list:
        """Get performance measurement history."""
        return self.performance_history.copy()


# Factory function for easy instantiation
def create_performance_sentinel(name: str = "performance_sentinel",
                              check_interval: float = 60.0,
                              min_success_rate: float = 0.8,
                              max_avg_response_time: float = 30.0,
                              min_records_per_minute: float = 10.0) -> PerformanceSentinel:
    """
    Create a performance sentinel with sensible defaults.

    Args:
        name: Sentinel name
        check_interval: How often to check (seconds)
        min_success_rate: Minimum acceptable success rate (0.0-1.0)
        max_avg_response_time: Maximum acceptable response time (seconds)
        min_records_per_minute: Minimum acceptable throughput

    Returns:
        Configured PerformanceSentinel instance
    """
    config = SentinelConfig(
        name=name,
        check_interval=check_interval,
        severity=SentinelSeverity.WARNING
    )

    sentinel = PerformanceSentinel(config)
    sentinel.update_thresholds(
        min_success_rate=min_success_rate,
        max_avg_response_time=max_avg_response_time,
        min_records_per_minute=min_records_per_minute
    )

    return sentinel
