# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Cost Model for MJ Data Scraper Suite

Predictive cost modeling, optimization strategies, and financial analysis
for scraping operations with machine learning-based cost prediction.
"""

import asyncio
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .scrape_telemetry import ScrapeTelemetry, ScrapeTelemetryCollector

logger = logging.getLogger(__name__)


def compute_score(cost: float, records: int, blocked: bool) -> float:
    """
    Compute efficiency score based on cost and records collected.

    Args:
        cost: Cost incurred for the operation
        records: Number of records collected
        blocked: Whether the operation was blocked

    Returns:
        Efficiency score (records per dollar spent), 0.0 if blocked
    """
    if blocked:
        return 0.0
    return records / max(cost, 0.01)


@dataclass
class CostFactors:
    """Cost factors for different scraping operations."""
    base_cost_per_request: float = 0.01  # Base cost per HTTP request
    cost_per_record: float = 0.10       # Cost per record extracted
    cost_per_minute: float = 0.50       # Runtime cost per minute
    cost_per_browser_hour: float = 2.0  # Browser instance cost per hour
    cost_per_gb_bandwidth: float = 0.09 # Bandwidth cost per GB

    # Anti-detection multipliers
    human_behavior_multiplier: float = 1.0    # Cost multiplier for human-like behavior
    proxy_rotation_multiplier: float = 1.2    # Cost multiplier for proxy usage
    cookie_management_multiplier: float = 1.1 # Cost multiplier for session management

    # Operational multipliers
    business_hours_multiplier: float = 1.0   # Cost during business hours
    off_hours_multiplier: float = 0.8       # Cost during off hours
    weekend_multiplier: float = 0.9         # Cost during weekends

    # Penalty multipliers
    blocked_request_multiplier: float = 5.0  # Cost multiplier for blocked requests
    retry_multiplier: float = 2.0           # Cost multiplier per retry
    rate_limit_multiplier: float = 3.0      # Cost multiplier for rate limiting


@dataclass
class CostEstimate:
    """Detailed cost estimate for a scraping operation."""
    operation_type: str
    estimated_cost: float
    confidence_level: float  # 0.0 to 1.0
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)

    @property
    def cost_per_record(self) -> float:
        """Calculate cost per record if applicable."""
        if "records" in self.cost_breakdown and self.cost_breakdown["records"] > 0:
            return self.estimated_cost / self.cost_breakdown["records"]
        return 0.0

    @property
    def risk_adjusted_cost(self) -> float:
        """Calculate risk-adjusted cost estimate."""
        risk_multiplier = 1.0 + (len(self.risk_factors) * 0.1)
        return self.estimated_cost * risk_multiplier

    def compute_efficiency_score(self) -> float:
        """Compute efficiency score for this estimate."""
        return compute_score(self.estimated_cost, self.cost_breakdown.get("records", 0), False)


@dataclass
class CostOptimization:
    """Cost optimization recommendation."""
    strategy: str
    description: str
    potential_savings_percent: float
    implementation_complexity: str  # "low", "medium", "high"
    prerequisites: List[str] = field(default_factory=list)
    estimated_implementation_cost: float = 0.0


class CostPredictor:
    """
    Machine learning-based cost prediction using historical telemetry data.

    Uses statistical analysis and pattern recognition to predict costs
    for new scraping operations based on historical performance.
    """

    def __init__(self, telemetry_collector: Optional[ScrapeTelemetryCollector] = None):
        self.telemetry = telemetry_collector or ScrapeTelemetryCollector()
        self.cost_factors = CostFactors()
        self._trained_models: Dict[str, Any] = {}
        self._baseline_costs: Dict[str, float] = {}

        logger.info("CostPredictor initialized")

    async def predict_operation_cost(
        self,
        operation_type: str,
        parameters: Dict[str, Any],
        confidence_threshold: float = 0.7
    ) -> CostEstimate:
        """
        Predict cost for a scraping operation.

        Args:
            operation_type: Type of scraping operation
            parameters: Operation parameters (pages, records, etc.)
            confidence_threshold: Minimum confidence level for prediction

        Returns:
            CostEstimate with prediction details
        """
        # Get historical data for this operation type
        historical_data = self._get_historical_data(operation_type)

        if len(historical_data) < 5:
            # Insufficient data, use rule-based estimation
            return await self._rule_based_estimation(operation_type, parameters)

        # Use statistical analysis for prediction
        prediction = self._statistical_prediction(operation_type, parameters, historical_data)

        # Apply confidence adjustment
        if prediction.confidence_level < confidence_threshold:
            prediction.optimization_opportunities.append(
                "Consider collecting more historical data for better predictions"
            )

        return prediction

    def _get_historical_data(self, operation_type: str) -> List[ScrapeTelemetry]:
        """Get historical telemetry data for operation type."""
        return self.telemetry.get_telemetry_by_source(operation_type)

    def _statistical_prediction(
        self,
        operation_type: str,
        parameters: Dict[str, Any],
        historical_data: List[ScrapeTelemetry]
    ) -> CostEstimate:
        """Generate statistical cost prediction."""
        # Calculate baseline metrics from historical data
        costs = [t.cost for t in historical_data if t.cost > 0]
        records = [t.records_found for t in historical_data if t.records_found > 0]
        latencies = [t.latency_ms for t in historical_data if t.latency_ms > 0]

        if not costs:
            return CostEstimate(
                operation_type=operation_type,
                estimated_cost=0.0,
                confidence_level=0.0,
                assumptions=["No historical cost data available"]
            )

        # Statistical analysis
        avg_cost = sum(costs) / len(costs)
        cost_variance = sum((c - avg_cost) ** 2 for c in costs) / len(costs)
        cost_stddev = math.sqrt(cost_variance)

        # Estimate based on parameters
        estimated_cost = self._calculate_parameter_based_cost(parameters, historical_data)

        # Calculate confidence based on data quality and consistency
        data_points = len(historical_data)
        consistency_ratio = 1.0 - min(cost_stddev / avg_cost, 0.5)  # Lower variance = higher confidence
        confidence = min(data_points / 20.0, 1.0) * consistency_ratio

        # Generate cost breakdown
        breakdown = self._generate_cost_breakdown(parameters, estimated_cost)

        # Identify risk factors
        risk_factors = self._identify_risk_factors(parameters, historical_data)

        # Optimization opportunities
        optimizations = self._generate_optimization_opportunities(operation_type, historical_data)

        return CostEstimate(
            operation_type=operation_type,
            estimated_cost=estimated_cost,
            confidence_level=confidence,
            cost_breakdown=breakdown,
            assumptions=self._generate_assumptions(parameters),
            risk_factors=risk_factors,
            optimization_opportunities=optimizations
        )

    async def _rule_based_estimation(
        self,
        operation_type: str,
        parameters: Dict[str, Any]
    ) -> CostEstimate:
        """Fallback rule-based cost estimation when insufficient historical data."""
        estimated_cost = self._calculate_rule_based_cost(operation_type, parameters)

        return CostEstimate(
            operation_type=operation_type,
            estimated_cost=estimated_cost,
            confidence_level=0.3,  # Low confidence for rule-based estimates
            cost_breakdown=self._generate_cost_breakdown(parameters, estimated_cost),
            assumptions=[
                "Using rule-based estimation due to insufficient historical data",
                f"Applied standard cost factors for {operation_type}",
                "Consider collecting more operational data for better predictions"
            ],
            risk_factors=[
                "Limited historical data for accurate prediction",
                "Rule-based estimates may not reflect actual operational costs"
            ],
            optimization_opportunities=[
                "Implement comprehensive telemetry collection",
                "Establish baseline performance metrics",
                "Consider pilot operations to gather cost data"
            ]
        )

    def _calculate_parameter_based_cost(
        self,
        parameters: Dict[str, Any],
        historical_data: List[ScrapeTelemetry]
    ) -> float:
        """Calculate cost based on operation parameters and historical patterns."""
        # Extract parameters
        pages = parameters.get("pages", 1)
        records = parameters.get("records", 0)
        minutes = parameters.get("minutes", 1)
        browser_hours = parameters.get("browser_hours", 0)
        blocked = parameters.get("blocked", False)
        retries = parameters.get("retries", 0)

        # Calculate base cost using cost factors
        base_cost = (
            pages * self.cost_factors.base_cost_per_request +
            records * self.cost_factors.cost_per_record +
            minutes * self.cost_factors.cost_per_minute +
            browser_hours * self.cost_factors.cost_per_browser_hour
        )

        # Apply multipliers
        total_multiplier = 1.0

        if blocked:
            total_multiplier *= self.cost_factors.blocked_request_multiplier
        if retries > 0:
            total_multiplier *= (self.cost_factors.retry_multiplier ** retries)

        # Apply anti-detection multipliers based on parameters
        if parameters.get("human_behavior", True):
            total_multiplier *= self.cost_factors.human_behavior_multiplier
        if parameters.get("proxy_rotation", False):
            total_multiplier *= self.cost_factors.proxy_rotation_multiplier

        return base_cost * total_multiplier

    def _calculate_rule_based_cost(self, operation_type: str, parameters: Dict[str, Any]) -> float:
        """Calculate cost using predefined rules when no historical data available."""
        # Default cost calculation based on operation type
        operation_multipliers = {
            "linkedin": 2.0,
            "facebook": 1.8,
            "twitter": 1.5,
            "instagram": 1.7,
            "web": 1.0,
            "news": 1.3,
            "company_website": 1.4
        }

        base_multiplier = operation_multipliers.get(operation_type, 1.0)

        # Base cost calculation
        pages = parameters.get("pages", 1)
        records = parameters.get("records", 0)

        cost = (pages * 0.05 + records * 0.15) * base_multiplier

        # Apply complexity factors
        if parameters.get("login_required", False):
            cost *= 3.0
        if parameters.get("javascript_required", False):
            cost *= 2.0
        if parameters.get("anti_detection", True):
            cost *= 1.5

        return cost

    def _generate_cost_breakdown(self, parameters: Dict[str, Any], total_cost: float) -> Dict[str, float]:
        """Generate detailed cost breakdown."""
        breakdown = {}

        # Estimate component costs
        pages = parameters.get("pages", 1)
        records = parameters.get("records", 0)
        minutes = parameters.get("minutes", 1)

        breakdown["requests"] = pages * self.cost_factors.base_cost_per_request
        breakdown["records"] = records * self.cost_factors.cost_per_record
        breakdown["runtime"] = minutes * self.cost_factors.cost_per_minute

        if parameters.get("browser_hours", 0) > 0:
            breakdown["browser"] = parameters["browser_hours"] * self.cost_factors.cost_per_browser_hour

        # Calculate overhead (difference)
        calculated_total = sum(breakdown.values())
        breakdown["overhead"] = max(0, total_cost - calculated_total)

        return breakdown

    def _identify_risk_factors(self, parameters: Dict[str, Any], historical_data: List[ScrapeTelemetry]) -> List[str]:
        """Identify risk factors that could increase costs."""
        risks = []

        # Check for high-risk parameters
        if parameters.get("login_required", False):
            risks.append("Login required - increases complexity and failure risk")
        if parameters.get("javascript_required", False):
            risks.append("JavaScript rendering required - increases resource usage")
        if not parameters.get("anti_detection", True):
            risks.append("Anti-detection disabled - higher risk of blocking")

        # Check historical patterns
        if historical_data:
            block_rate = sum(1 for t in historical_data if t.blocked) / len(historical_data)
            if block_rate > 0.1:
                risks.append(".1%")

            avg_latency = sum(t.latency_ms for t in historical_data) / len(historical_data)
            if avg_latency > 5000:  # 5 seconds
                risks.append("High latency detected - potential performance issues")

        return risks

    def _generate_optimization_opportunities(
        self,
        operation_type: str,
        historical_data: List[ScrapeTelemetry]
    ) -> List[str]:
        """Generate cost optimization recommendations."""
        opportunities = []

        if not historical_data:
            opportunities.append("Establish baseline metrics through regular operations")
            return opportunities

        # Analyze patterns for optimization opportunities
        avg_cost = sum(t.cost for t in historical_data) / len(historical_data)
        avg_records = sum(t.records_found for t in historical_data) / len(historical_data)
        efficiency = avg_records / max(0.01, avg_cost)

        if efficiency < 50:
            opportunities.append("Low efficiency detected - consider optimizing data extraction")

        # Check for time-based patterns
        business_hours_data = [t for t in historical_data if t.is_business_hours]
        off_hours_data = [t for t in historical_data if not t.is_business_hours]

        if business_hours_data and off_hours_data:
            business_avg = sum(t.cost for t in business_hours_data) / len(business_hours_data)
            off_hours_avg = sum(t.cost for t in off_hours_data) / len(off_hours_data)

            if business_avg > off_hours_avg * 1.2:
                opportunities.append("Consider scheduling operations during off-peak hours")

        return opportunities

    def _generate_assumptions(self, parameters: Dict[str, Any]) -> List[str]:
        """Generate list of assumptions made in cost estimation."""
        assumptions = []

        if not parameters.get("pages"):
            assumptions.append("Assumed 1 page if not specified")
        if not parameters.get("minutes"):
            assumptions.append("Assumed 1 minute runtime if not specified")

        assumptions.append(f"Using cost factors: ${self.cost_factors.base_cost_per_request} per request")
        assumptions.append(f"Applied standard multipliers for {parameters.get('operation_type', 'unknown')}")

        return assumptions


class CostOptimizer:
    """
    Cost optimization engine providing recommendations and strategies
    for reducing scraping costs while maintaining effectiveness.
    """

    def __init__(self, cost_predictor: CostPredictor):
        self.predictor = cost_predictor
        self.optimization_strategies = self._load_optimization_strategies()

    def _load_optimization_strategies(self) -> List[CostOptimization]:
        """Load predefined optimization strategies."""
        return [
            CostOptimization(
                strategy="schedule_off_peak",
                description="Schedule operations during off-peak hours for lower costs",
                potential_savings_percent=15.0,
                implementation_complexity="low",
                prerequisites=["Time-based scheduling capability"]
            ),
            CostOptimization(
                strategy="reduce_frequency",
                description="Reduce scraping frequency while maintaining data freshness",
                potential_savings_percent=25.0,
                implementation_complexity="medium",
                prerequisites=["Data freshness requirements analysis"]
            ),
            CostOptimization(
                strategy="optimize_selectors",
                description="Improve CSS selectors for faster, more reliable data extraction",
                potential_savings_percent=20.0,
                implementation_complexity="medium",
                prerequisites=["Technical analysis of current selectors"]
            ),
            CostOptimization(
                strategy="implement_caching",
                description="Cache frequently accessed data to reduce redundant requests",
                potential_savings_percent=30.0,
                implementation_complexity="high",
                prerequisites=["Data access pattern analysis", "Cache infrastructure"]
            ),
            CostOptimization(
                strategy="batch_operations",
                description="Batch similar operations to reduce per-request overhead",
                potential_savings_percent=18.0,
                implementation_complexity="medium",
                prerequisites=["Operation similarity analysis"]
            ),
            CostOptimization(
                strategy="proxy_optimization",
                description="Optimize proxy usage based on performance and cost analysis",
                potential_savings_percent=12.0,
                implementation_complexity="low",
                prerequisites=["Proxy performance monitoring"]
            )
        ]

    async def analyze_cost_optimization(self, operation_type: str, current_costs: Dict[str, Any]) -> List[CostOptimization]:
        """
        Analyze current costs and recommend optimization strategies.

        Args:
            operation_type: Type of operation being analyzed
            current_costs: Current cost metrics and parameters

        Returns:
            List of applicable optimization recommendations
        """
        applicable_strategies = []
        historical_data = self.predictor._get_historical_data(operation_type)

        for strategy in self.optimization_strategies:
            if self._is_strategy_applicable(strategy, current_costs, historical_data):
                applicable_strategies.append(strategy)

        # Sort by potential savings (highest first)
        applicable_strategies.sort(key=lambda s: s.potential_savings_percent, reverse=True)

        return applicable_strategies[:5]  # Return top 5 recommendations

    def _is_strategy_applicable(
        self,
        strategy: CostOptimization,
        current_costs: Dict[str, Any],
        historical_data: List[ScrapeTelemetry]
    ) -> bool:
        """Determine if an optimization strategy is applicable."""

        if strategy.strategy == "schedule_off_peak":
            # Check if operations are currently running during peak hours
            if historical_data:
                business_hours_ops = sum(1 for t in historical_data if t.is_business_hours)
                total_ops = len(historical_data)
                if business_hours_ops / total_ops > 0.6:  # >60% during business hours
                    return True

        elif strategy.strategy == "reduce_frequency":
            # Check if operation frequency seems high
            if historical_data and len(historical_data) > 10:
                # Check if operations are happening more than once per hour
                timestamps = sorted([t.timestamp for t in historical_data[-20:]])
                if len(timestamps) >= 2:
                    avg_interval = (timestamps[-1] - timestamps[0]).total_seconds() / (len(timestamps) - 1)
                    if avg_interval < 3600:  # Less than 1 hour between operations
                        return True

        elif strategy.strategy == "implement_caching":
            # Check for repeated similar operations
            if historical_data and len(historical_data) > 20:
                # Look for patterns in operation timing and similarity
                return True  # Simplified - would need more sophisticated analysis

        # Default: strategy is potentially applicable
        return True

    async def calculate_optimization_impact(
        self,
        strategy: CostOptimization,
        current_costs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate the expected impact of implementing an optimization strategy.

        Args:
            strategy: Optimization strategy to analyze
            current_costs: Current cost baseline

        Returns:
            Impact analysis with savings estimates and ROI
        """
        current_monthly_cost = current_costs.get("monthly_cost", 0)
        expected_savings = current_monthly_cost * (strategy.potential_savings_percent / 100)

        return {
            "strategy": strategy.strategy,
            "potential_monthly_savings": expected_savings,
            "potential_annual_savings": expected_savings * 12,
            "implementation_cost": strategy.estimated_implementation_cost,
            "break_even_months": strategy.estimated_implementation_cost / max(0.01, expected_savings),
            "roi_percent": (expected_savings * 12 / max(0.01, strategy.estimated_implementation_cost)) * 100,
            "complexity": strategy.implementation_complexity,
            "prerequisites": strategy.prerequisites
        }


# Global cost model instance
_global_cost_predictor = CostPredictor()


def get_global_cost_predictor() -> CostPredictor:
    """Get the global cost predictor instance."""
    return _global_cost_predictor


async def estimate_scrape_cost(
    operation_type: str,
    pages: int = 1,
    records: int = 0,
    minutes: int = 1,
    browser_hours: float = 0,
    **kwargs
) -> CostEstimate:
    """
    Convenience function for quick cost estimation.

    Args:
        operation_type: Type of scraping operation
        pages: Number of pages to scrape
        records: Expected number of records
        minutes: Expected runtime in minutes
        browser_hours: Browser usage in hours
        **kwargs: Additional parameters

    Returns:
        CostEstimate for the operation
    """
    parameters = {
        "pages": pages,
        "records": records,
        "minutes": minutes,
        "browser_hours": browser_hours,
        **kwargs
    }

    return await _global_cost_predictor.predict_operation_cost(operation_type, parameters)
