# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Cost Governor for MJ Data Scraper Suite

Manages resource budgets and cost controls for scraping operations.
Tracks usage against budgets and enforces spending limits with real-time monitoring.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from .control_models import ScrapeBudget

logger = logging.getLogger(__name__)


@dataclass
class ResourceUsage:
    """Tracks actual resource usage against budgets."""
    runtime_seconds: float = 0.0
    pages_scraped: int = 0
    records_collected: int = 0
    browser_instances_used: int = 0
    memory_mb_used: float = 0.0
    start_time: datetime = field(default_factory=datetime.utcnow)

    @property
    def runtime_minutes(self) -> float:
        """Get runtime in minutes."""
        return self.runtime_seconds / 60.0

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary."""
        return {
            "runtime_minutes": self.runtime_minutes,
            "pages_scraped": self.pages_scraped,
            "records_collected": self.records_collected,
            "browser_instances_used": self.browser_instances_used,
            "memory_mb_used": self.memory_mb_used,
            "efficiency_ratio": self.records_collected / max(1, self.pages_scraped)
        }


@dataclass
class CostMetrics:
    """Tracks cost-related metrics and estimates."""
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    cost_per_record: float = 0.0
    cost_per_page: float = 0.0
    cost_efficiency_score: float = 0.0

    def update_costs(self, usage: ResourceUsage, cost_rates: Dict[str, float]) -> None:
        """Update cost calculations based on usage and rates."""
        # Calculate costs based on resource usage
        self.actual_cost = (
            usage.runtime_seconds * cost_rates.get("per_second", 0.0) +
            usage.pages_scraped * cost_rates.get("per_page", 0.0) +
            usage.records_collected * cost_rates.get("per_record", 0.0) +
            usage.browser_instances_used * cost_rates.get("per_browser_hour", 0.0) * (usage.runtime_seconds / 3600)
        )

        # Update efficiency metrics
        self.cost_per_record = self.actual_cost / max(1, usage.records_collected)
        self.cost_per_page = self.actual_cost / max(1, usage.pages_scraped)
        self.cost_efficiency_score = usage.records_collected / max(1, self.actual_cost)


class CostGovernor:
    """
    Resource budget and cost governor for scraping operations.

    Monitors resource usage in real-time, enforces budget limits,
    and provides cost tracking and optimization recommendations.
    """

    def __init__(self, budget: ScrapeBudget, operation_id: str = ""):
        self.budget = budget
        self.operation_id = operation_id or f"cost_gov_{int(time.time())}"
        self.usage = ResourceUsage()
        self.cost_metrics = CostMetrics()
        self.cost_rates = self._get_default_cost_rates()
        self.alerts: List[str] = []
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None

        logger.info(f"CostGovernor initialized for operation {self.operation_id}")

    @staticmethod
    def _get_default_cost_rates() -> Dict[str, float]:
        """Get default cost rates for resource usage."""
        return {
            "per_second": 0.01,      # $0.01 per second of runtime
            "per_page": 0.05,        # $0.05 per page scraped
            "per_record": 0.10,      # $0.10 per record collected
            "per_browser_hour": 2.0  # $2.00 per browser instance per hour
        }

    @classmethod
    async def initialize(cls, budget: ScrapeBudget, operation_id: str = "") -> 'CostGovernor':
        """
        Initialize a cost governor with monitoring.

        Args:
            budget: Resource budget constraints
            operation_id: Unique identifier for the operation

        Returns:
            Initialized and monitoring CostGovernor
        """
        governor = cls(budget, operation_id)
        await governor.start_monitoring()
        return governor

    async def start_monitoring(self) -> None:
        """Start real-time resource monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitor_resources())
        logger.info(f"Cost monitoring started for {self.operation_id}")

    async def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Cost monitoring stopped for {self.operation_id}")

    async def _monitor_resources(self) -> None:
        """Monitor resource usage in real-time."""
        while self.monitoring_active:
            try:
                # Update runtime
                self.usage.runtime_seconds = (datetime.utcnow() - self.usage.start_time).total_seconds()

                # Check budget limits
                await self._check_budget_limits()

                # Update cost metrics
                self.cost_metrics.update_costs(self.usage, self.cost_rates)

                # Log periodic status
                if int(self.usage.runtime_seconds) % 60 == 0:  # Every minute
                    self._log_status()

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in cost monitoring: {e}")
                await asyncio.sleep(10)

    async def _check_budget_limits(self) -> None:
        """Check if any budget limits have been exceeded."""
        # Runtime check
        if self.usage.runtime_minutes > self.budget.max_runtime_minutes:
            alert = f"Runtime budget exceeded: {self.usage.runtime_minutes:.1f} > {self.budget.max_runtime_minutes} minutes"
            await self._raise_budget_alert(alert)

        # Page count check
        if self.usage.pages_scraped > self.budget.max_pages:
            alert = f"Page budget exceeded: {self.usage.pages_scraped} > {self.budget.max_pages} pages"
            await self._raise_budget_alert(alert)

        # Record count check
        if self.usage.records_collected > self.budget.max_records:
            alert = f"Record budget exceeded: {self.usage.records_collected} > {self.budget.max_records} records"
            await self._raise_budget_alert(alert)

        # Browser instance check
        if self.usage.browser_instances_used > self.budget.max_browser_instances:
            alert = f"Browser budget exceeded: {self.usage.browser_instances_used} > {self.budget.max_browser_instances} instances"
            await self._raise_budget_alert(alert)

        # Memory check
        if self.usage.memory_mb_used > self.budget.max_memory_mb:
            alert = f"Memory budget exceeded: {self.usage.memory_mb_used:.1f} > {self.budget.max_memory_mb} MB"
            await self._raise_budget_alert(alert)

    async def _raise_budget_alert(self, alert: str) -> None:
        """Raise a budget alert and log it."""
        if alert not in self.alerts:
            self.alerts.append(alert)
            logger.warning(f"BUDGET ALERT: {alert}")

            # Could trigger additional actions here:
            # - Send notifications
            # - Scale down operations
            # - Emergency shutdown

    def record_page_scraped(self) -> bool:
        """
        Record that a page was scraped.

        Returns:
            True if within budget, False if budget exceeded
        """
        self.usage.pages_scraped += 1
        return self.usage.pages_scraped <= self.budget.max_pages

    def record_records_collected(self, count: int) -> bool:
        """
        Record records collected.

        Args:
            count: Number of records collected

        Returns:
            True if within budget, False if budget exceeded
        """
        self.usage.records_collected += count
        return self.usage.records_collected <= self.budget.max_records

    def record_browser_usage(self, instances: int) -> bool:
        """
        Record browser instances in use.

        Args:
            instances: Number of browser instances currently active

        Returns:
            True if within budget, False if budget exceeded
        """
        self.usage.browser_instances_used = max(self.usage.browser_instances_used, instances)
        return self.usage.browser_instances_used <= self.budget.max_browser_instances

    def record_memory_usage(self, memory_mb: float) -> bool:
        """
        Record current memory usage.

        Args:
            memory_mb: Current memory usage in MB

        Returns:
            True if within budget, False if budget exceeded
        """
        self.usage.memory_mb_used = max(self.usage.memory_mb_used, memory_mb)
        return self.usage.memory_mb_used <= self.budget.max_memory_mb

    def get_budget_status(self) -> Dict[str, Any]:
        """Get comprehensive budget status."""
        return {
            "operation_id": self.operation_id,
            "budget_limits": {
                "max_runtime_minutes": self.budget.max_runtime_minutes,
                "max_pages": self.budget.max_pages,
                "max_records": self.budget.max_records,
                "max_browser_instances": self.budget.max_browser_instances,
                "max_memory_mb": self.budget.max_memory_mb
            },
            "current_usage": self.usage.get_usage_summary(),
            "budget_utilization": {
                "runtime_percent": (self.usage.runtime_minutes / max(1, self.budget.max_runtime_minutes)) * 100,
                "pages_percent": (self.usage.pages_scraped / max(1, self.budget.max_pages)) * 100,
                "records_percent": (self.usage.records_collected / max(1, self.budget.max_records)) * 100,
                "browser_percent": (self.usage.browser_instances_used / max(1, self.budget.max_browser_instances)) * 100,
                "memory_percent": (self.usage.memory_mb_used / max(1, self.budget.max_memory_mb)) * 100
            },
            "cost_metrics": {
                "estimated_cost": self.cost_metrics.estimated_cost,
                "actual_cost": self.cost_metrics.actual_cost,
                "cost_per_record": self.cost_metrics.cost_per_record,
                "cost_efficiency_score": self.cost_metrics.cost_efficiency_score
            },
            "alerts": self.alerts.copy(),
            "within_budget": len(self.alerts) == 0
        }

    def should_shutdown(self) -> bool:
        """
        Check if operation should shutdown due to budget violations.

        Returns:
            True if shutdown is recommended
        """
        return len(self.alerts) > 0

    def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations for cost optimization."""
        recommendations = []

        status = self.get_budget_status()
        utilization = status["budget_utilization"]

        # Runtime optimization
        if utilization["runtime_percent"] > 90:
            recommendations.append("Consider reducing scraping frequency to stay within runtime budget")

        # Resource optimization
        if utilization["pages_percent"] > utilization["records_percent"] + 20:
            recommendations.append("Low record-to-page ratio detected - consider improving data extraction efficiency")

        # Cost optimization
        if self.cost_metrics.cost_per_record > 0.50:
            recommendations.append("High cost per record - consider optimizing scraping patterns")

        # Memory optimization
        if utilization["memory_percent"] > 80:
            recommendations.append("High memory usage - consider reducing concurrent operations")

        return recommendations

    def _log_status(self) -> None:
        """Log periodic status update."""
        status = self.get_budget_status()
        utilization = status["budget_utilization"]

        logger.info(
            f"Cost status for {self.operation_id}: "
            f"Runtime: {utilization['runtime_percent']:.1f}%, "
            f"Records: {utilization['records_percent']:.1f}%, "
            f"Cost: ${self.cost_metrics.actual_cost:.2f}, "
            f"Alerts: {len(self.alerts)}"
        )

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.stop_monitoring()
        logger.info(f"CostGovernor cleanup completed for {self.operation_id}")
