# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Performance Sentinel for MJ Data Scraper Suite

Monitors scraping performance metrics and provides risk assessments.
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

from .base import BaseSentinel, SentinelReport
from ..scrape_telemetry import get_global_telemetry_collector

logger = logging.getLogger(__name__)


class PerformanceSentinel(BaseSentinel):
    """
    Monitors scraping performance and provides risk assessments.

    Probes targets to evaluate performance metrics and returns structured reports
    with risk levels and recommended actions.
    """

    name = "performance_sentinel"

    def __init__(self):
        # Performance thresholds
        self.min_success_rate = 0.8  # 80% success rate minimum
        self.max_avg_response_time = 30.0  # 30 seconds maximum average
        self.min_records_per_minute = 10.0  # Minimum throughput

        # Historical tracking
        self.performance_history = []
        self.history_window = 10  # Keep last 10 measurements

    async def probe(self, target: Dict[str, Any]) -> SentinelReport:
        """
        Probe scraping performance and return risk assessment.

        Args:
            target: Target information (may include time range, specific operations, etc.)

        Returns:
            SentinelReport with performance analysis and recommendations
        """
        try:
            collector = get_global_telemetry_collector()
            metrics = collector.get_current_metrics()

            if not metrics or metrics.get('total_operations', 0) == 0:
                return SentinelReport(
                    sentinel_name=self.name,
                    domain="performance",
                    timestamp=datetime.utcnow(),
                    risk_level="low",
                    findings={"status": "no_operations"},
                    recommended_action="allow"
                )

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

            # Analyze performance and determine risk level
            findings = {
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'records_per_minute': records_per_minute,
                'total_operations': total_ops,
                'metrics': metrics
            }

            issues = []
            risk_score = 0

            # Check performance thresholds
            if success_rate < self.min_success_rate:
                issues.append(f"Low success rate: {success_rate:.1%}")
                risk_score += 1

            if avg_response_time > self.max_avg_response_time:
                issues.append(f"High response time: {avg_response_time:.1f}s")
                risk_score += 1

            if records_per_minute < self.min_records_per_minute and total_ops > 5:
                issues.append(f"Low throughput: {records_per_minute:.1f} records/min")
                risk_score += 1

            # Check for performance degradation trends
            if len(self.performance_history) >= 3:
                recent = self.performance_history[-3:]
                success_trend = [p['success_rate'] for p in recent]
                response_trend = [p['avg_response_time'] for p in recent]

                # Check if success rate is declining
                if len(success_trend) == 3 and success_trend[0] > success_trend[1] > success_trend[2]:
                    decline = (success_trend[0] - success_trend[2]) / success_trend[0]
                    if decline > 0.1:  # 10% decline
                        issues.append(f"Success rate declining: -{decline:.1%}")
                        risk_score += 1

                # Check if response time is increasing
                if len(response_trend) == 3 and response_trend[0] < response_trend[1] < response_trend[2]:
                    increase = (response_trend[2] - response_trend[0]) / response_trend[0]
                    if increase > 0.5:  # 50% increase
                        issues.append(f"Response time increasing: +{increase:.1%}")
                        risk_score += 1

            findings['issues'] = issues

            # Determine risk level and recommended action
            if risk_score >= 3:
                risk_level = "critical"
                recommended_action = "block"
            elif risk_score >= 2:
                risk_level = "high"
                recommended_action = "restrict"
            elif risk_score >= 1:
                risk_level = "medium"
                recommended_action = "delay"
            else:
                risk_level = "low"
                recommended_action = "allow"

            return SentinelReport(
                sentinel_name=self.name,
                domain="performance",
                timestamp=datetime.utcnow(),
                risk_level=risk_level,
                findings=findings,
                recommended_action=recommended_action
            )

        except Exception as e:
            logger.error(f"Performance probe failed: {e}")
            return SentinelReport(
                sentinel_name=self.name,
                domain="performance",
                timestamp=datetime.utcnow(),
                risk_level="critical",
                findings={"error": str(e)},
                recommended_action="block"
            )

    def update_thresholds(self, min_success_rate: float = None, max_avg_response_time: float = None,
                         min_records_per_minute: float = None) -> None:
        """Update performance thresholds."""
        if min_success_rate is not None:
            self.min_success_rate = min_success_rate
        if max_avg_response_time is not None:
            self.max_avg_response_time = max_avg_response_time
        if min_records_per_minute is not None:
            self.min_records_per_minute = min_records_per_minute

        logger.info(f"Updated performance thresholds for {self.name}: "
                   f"success_rate={self.min_success_rate}, "
                   f"response_time={self.max_avg_response_time}, "
                   f"records_per_minute={self.min_records_per_minute}")

    def get_performance_history(self) -> list:
        """Get performance measurement history."""
        return self.performance_history.copy()


# Factory function for easy instantiation
def create_performance_sentinel(min_success_rate: float = 0.8,
                              max_avg_response_time: float = 30.0,
                              min_records_per_minute: float = 10.0) -> PerformanceSentinel:
    """
    Create a performance sentinel with sensible defaults.

    Args:
        min_success_rate: Minimum acceptable success rate (0.0-1.0)
        max_avg_response_time: Maximum acceptable response time (seconds)
        min_records_per_minute: Minimum acceptable throughput

    Returns:
        Configured PerformanceSentinel instance
    """
    sentinel = PerformanceSentinel()
    sentinel.update_thresholds(
        min_success_rate=min_success_rate,
        max_avg_response_time=max_avg_response_time,
        min_records_per_minute=min_records_per_minute
    )

    return sentinel
