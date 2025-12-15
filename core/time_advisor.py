# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Time Advisor for MJ Data Scraper Suite

Intelligent time-based scheduling recommendations for scraping operations.
Analyzes historical data, cost patterns, and operational constraints to
recommend optimal execution times for maximum efficiency and success.
"""

import asyncio
import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

from .scrape_telemetry import ScrapeTelemetry, ScrapeTelemetryCollector
from .control_models import ScrapeTempo

logger = logging.getLogger(__name__)


@dataclass
class TimeSlot:
    """Represents a specific time slot for scheduling."""
    start_time: datetime
    end_time: datetime
    hour_of_day: int
    day_of_week: str
    timezone: str = "UTC"

    @property
    def duration_hours(self) -> float:
        """Calculate duration of this time slot in hours."""
        return (self.end_time - self.start_time).total_seconds() / 3600

    @property
    def is_business_hours(self) -> bool:
        """Check if this slot falls within business hours."""
        return (9 <= self.hour_of_day <= 17 and
                self.day_of_week not in ["Saturday", "Sunday"])

    @property
    def is_weekend(self) -> bool:
        """Check if this slot falls on a weekend."""
        return self.day_of_week in ["Saturday", "Sunday"]


@dataclass
class TimeRecommendation:
    """Recommendation for optimal execution time."""
    recommended_slot: TimeSlot
    confidence_score: float  # 0.0 to 1.0
    expected_efficiency: float
    expected_cost_savings: float
    reasoning: List[str] = field(default_factory=list)
    alternatives: List[TimeSlot] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert recommendation to dictionary."""
        return {
            "recommended_slot": {
                "start_time": self.recommended_slot.start_time.isoformat(),
                "end_time": self.recommended_slot.end_time.isoformat(),
                "hour_of_day": self.recommended_slot.hour_of_day,
                "day_of_week": self.recommended_slot.day_of_week,
                "timezone": self.recommended_slot.timezone,
                "is_business_hours": self.recommended_slot.is_business_hours,
                "is_weekend": self.recommended_slot.is_weekend
            },
            "confidence_score": self.confidence_score,
            "expected_efficiency": self.expected_efficiency,
            "expected_cost_savings": self.expected_cost_savings,
            "reasoning": self.reasoning,
            "alternatives_count": len(self.alternatives)
        }


@dataclass
class TimePattern:
    """Analyzed time-based patterns from historical data."""
    hour_performance: Dict[int, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    day_performance: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    peak_hours: Set[int] = field(default_factory=set)
    off_peak_hours: Set[int] = field(default_factory=set)
    best_performing_hours: List[Tuple[int, float]] = field(default_factory=list)
    cost_patterns: Dict[int, float] = field(default_factory=dict)

    def get_best_hour_for_operation(self, operation_type: str) -> Optional[int]:
        """Get the best performing hour for a specific operation type."""
        if operation_type in self.hour_performance:
            # Find hour with highest efficiency for this operation
            op_performance = self.hour_performance[operation_type]
            if op_performance:
                best_hour = max(op_performance.items(), key=lambda x: x[1])
                return best_hour[0]
        return None

    def get_cost_savings_potential(self, current_hour: int, recommended_hour: int) -> float:
        """Calculate potential cost savings by switching hours."""
        current_cost = self.cost_patterns.get(current_hour, 1.0)
        recommended_cost = self.cost_patterns.get(recommended_hour, 1.0)
        if current_cost > 0:
            return ((current_cost - recommended_cost) / current_cost) * 100
        return 0.0


class TimeAdvisor:
    """
    Intelligent time-based scheduling advisor for scraping operations.

    Analyzes historical telemetry data to provide optimal scheduling recommendations
    that maximize efficiency, minimize costs, and avoid operational issues.
    """

    def __init__(self, telemetry_collector: Optional[ScrapeTelemetryCollector] = None):
        self.telemetry = telemetry_collector or ScrapeTelemetryCollector()
        self.time_patterns = TimePattern()
        self._patterns_analyzed = False

        # Cost multipliers for different times (can be customized)
        self.cost_multipliers = {
            "business_hours": 1.2,  # 20% premium during business hours
            "off_hours": 0.9,       # 10% discount during off hours
            "weekend": 0.95,        # 5% discount on weekends
            "peak_hour_penalty": 1.5  # 50% penalty for peak hours
        }

        logger.info("TimeAdvisor initialized")

    async def analyze_patterns(self) -> None:
        """Analyze historical telemetry data to build time patterns."""
        if self._patterns_analyzed:
            return

        logger.info("Analyzing time patterns from telemetry data...")

        telemetry_data = list(self.telemetry.telemetry_data)
        if not telemetry_data:
            logger.warning("No telemetry data available for pattern analysis")
            self._patterns_analyzed = True
            return

        # Analyze by hour of day
        hour_stats = defaultdict(lambda: {"total_cost": 0.0, "total_records": 0, "operations": 0, "blocks": 0})

        # Analyze by day of week
        day_stats = defaultdict(lambda: {"total_cost": 0.0, "total_records": 0, "operations": 0, "blocks": 0})

        for telemetry in telemetry_data:
            hour = telemetry.hour_of_day
            day = telemetry.day_of_week

            # Hour-based stats
            hour_stats[hour]["total_cost"] += telemetry.cost
            hour_stats[hour]["total_records"] += telemetry.records_found
            hour_stats[hour]["operations"] += 1
            if telemetry.blocked:
                hour_stats[hour]["blocks"] += 1

            # Day-based stats
            day_stats[day]["total_cost"] += telemetry.cost
            day_stats[day]["total_records"] += telemetry.records_found
            day_stats[day]["operations"] += 1
            if telemetry.blocked:
                day_stats[day]["blocks"] += 1

        # Calculate performance metrics
        for hour, stats in hour_stats.items():
            if stats["operations"] > 0:
                efficiency = stats["total_records"] / max(stats["total_cost"], 0.01)
                block_rate = stats["blocks"] / stats["operations"]
                avg_cost = stats["total_cost"] / stats["operations"]

                self.time_patterns.hour_performance[hour] = {
                    "efficiency": efficiency,
                    "block_rate": block_rate,
                    "avg_cost": avg_cost,
                    "operation_count": stats["operations"]
                }

                self.time_patterns.cost_patterns[hour] = avg_cost

        # Identify peak and off-peak hours
        if self.time_patterns.hour_performance:
            efficiencies = [(hour, stats["efficiency"])
                          for hour, stats in self.time_patterns.hour_performance.items()]

            # Sort by efficiency (highest first)
            efficiencies.sort(key=lambda x: x[1], reverse=True)

            # Top 3 hours are "best performing"
            self.time_patterns.best_performing_hours = efficiencies[:3]

            # Define peak hours (top 25% most active)
            all_hours = list(self.time_patterns.hour_performance.keys())
            all_hours.sort(key=lambda h: self.time_patterns.hour_performance[h]["operation_count"], reverse=True)
            peak_count = max(1, len(all_hours) // 4)
            self.time_patterns.peak_hours = set(all_hours[:peak_count])

            # Off-peak hours (bottom 25% least active)
            off_peak_count = max(1, len(all_hours) // 4)
            self.time_patterns.off_peak_hours = set(all_hours[-off_peak_count:])

        self._patterns_analyzed = True
        logger.info(f"Time pattern analysis complete. Analyzed {len(telemetry_data)} operations.")

    async def get_optimal_schedule(
        self,
        operation_type: str,
        duration_hours: float = 1.0,
        preferred_timezone: str = "UTC",
        tempo: ScrapeTempo = ScrapeTempo.HUMAN
    ) -> TimeRecommendation:
        """
        Get optimal scheduling recommendation for an operation.

        Args:
            operation_type: Type of scraping operation
            duration_hours: Expected duration of the operation
            preferred_timezone: Preferred timezone for scheduling
            tempo: Scraping tempo (affects time preferences)

        Returns:
            TimeRecommendation with optimal scheduling advice
        """
        await self.analyze_patterns()

        # Get current time as baseline
        now = datetime.utcnow()

        # Find best performing hour for this operation type
        best_hour = self.time_patterns.get_best_hour_for_operation(operation_type)

        if best_hour is None:
            # No historical data for this operation type, use general best hours
            if self.time_patterns.best_performing_hours:
                best_hour = self.time_patterns.best_performing_hours[0][0]
            else:
                # Default to off-peak hours
                best_hour = 2  # 2 AM as default

        # Create recommended time slot
        recommended_start = now.replace(hour=best_hour, minute=0, second=0, microsecond=0)

        # If the calculated time is in the past, move to tomorrow
        if recommended_start <= now:
            recommended_start += timedelta(days=1)

        recommended_end = recommended_start + timedelta(hours=duration_hours)

        # Determine day of week
        day_of_week = recommended_start.strftime("%A")

        recommended_slot = TimeSlot(
            start_time=recommended_start,
            end_time=recommended_end,
            hour_of_day=best_hour,
            day_of_week=day_of_week,
            timezone=preferred_timezone
        )

        # Calculate confidence and expected efficiency
        confidence_score = self._calculate_confidence(operation_type, best_hour)
        expected_efficiency = self._get_expected_efficiency(operation_type, best_hour)
        cost_savings = self._calculate_cost_savings_potential(now.hour, best_hour)

        # Generate reasoning
        reasoning = self._generate_reasoning(operation_type, recommended_slot, tempo)

        # Generate alternative slots
        alternatives = self._generate_alternatives(operation_type, duration_hours, preferred_timezone)

        recommendation = TimeRecommendation(
            recommended_slot=recommended_slot,
            confidence_score=confidence_score,
            expected_efficiency=expected_efficiency,
            expected_cost_savings=cost_savings,
            reasoning=reasoning,
            alternatives=alternatives
        )

        logger.info(f"Generated time recommendation for {operation_type}: {best_hour}:00 {day_of_week}")
        return recommendation

    def _calculate_confidence(self, operation_type: str, recommended_hour: int) -> float:
        """Calculate confidence score for the recommendation."""
        if operation_type not in self.time_patterns.hour_performance:
            return 0.3  # Low confidence for unknown operations

        hour_stats = self.time_patterns.hour_performance[recommended_hour]
        operation_count = hour_stats.get("operation_count", 0)

        # Confidence based on sample size and consistency
        sample_confidence = min(operation_count / 20.0, 1.0)  # Max confidence at 20+ samples
        consistency_factor = 1.0 - hour_stats.get("block_rate", 0.0)  # Lower blocks = higher confidence

        return min(sample_confidence * consistency_factor, 1.0)

    def _get_expected_efficiency(self, operation_type: str, hour: int) -> float:
        """Get expected efficiency for the recommended hour."""
        if operation_type in self.time_patterns.hour_performance:
            return self.time_patterns.hour_performance[operation_type].get("efficiency", 1.0)

        # Return general hour efficiency
        return self.time_patterns.hour_performance.get(hour, {}).get("efficiency", 1.0)

    def _calculate_cost_savings_potential(self, current_hour: int, recommended_hour: int) -> float:
        """Calculate potential cost savings."""
        return self.time_patterns.get_cost_savings_potential(current_hour, recommended_hour)

    def _generate_reasoning(self, operation_type: str, slot: TimeSlot, tempo: ScrapeTempo) -> List[str]:
        """Generate reasoning for the recommendation."""
        reasoning = []

        # Performance-based reasoning
        if operation_type in self.time_patterns.hour_performance:
            hour_stats = self.time_patterns.hour_performance[operation_type]
            efficiency = hour_stats.get("efficiency", 0)
            reasoning.append(".2f")
        else:
            reasoning.append(f"No historical data for {operation_type}, using general patterns")

        # Time-based reasoning
        if slot.is_business_hours:
            reasoning.append("Recommended during business hours for better data quality")
        else:
            reasoning.append("Recommended during off-hours to reduce costs and avoid peak loads")

        # Tempo-based reasoning
        if tempo == ScrapeTempo.FORENSIC:
            reasoning.append("Forensic tempo works well during off-peak hours with lower interference")
        elif tempo == ScrapeTempo.AGGRESSIVE:
            reasoning.append("Aggressive tempo scheduled during off-peak to minimize detection risk")

        # Cost-based reasoning
        if slot.hour_of_day in self.time_patterns.off_peak_hours:
            reasoning.append("Off-peak hour selected for potential cost savings")

        return reasoning

    def _generate_alternatives(self, operation_type: str, duration_hours: float, timezone: str) -> List[TimeSlot]:
        """Generate alternative time slots."""
        alternatives = []
        now = datetime.utcnow()

        # Generate alternatives from best performing hours (skip the top choice)
        for hour, _ in self.time_patterns.best_performing_hours[1:4]:  # Next 3 best hours
            alt_start = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if alt_start <= now:
                alt_start += timedelta(days=1)

            alt_end = alt_start + timedelta(hours=duration_hours)
            day_of_week = alt_start.strftime("%A")

            alternatives.append(TimeSlot(
                start_time=alt_start,
                end_time=alt_end,
                hour_of_day=hour,
                day_of_week=day_of_week,
                timezone=timezone
            ))

        return alternatives

    async def get_schedule_conflicts(
        self,
        proposed_slot: TimeSlot,
        existing_slots: List[TimeSlot]
    ) -> List[str]:
        """
        Check for scheduling conflicts with existing operations.

        Args:
            proposed_slot: Proposed time slot to check
            existing_slots: Existing scheduled operations

        Returns:
            List of conflict descriptions
        """
        conflicts = []

        for existing in existing_slots:
            # Check for time overlap
            if (proposed_slot.start_time < existing.end_time and
                proposed_slot.end_time > existing.start_time):
                conflicts.append(
                    f"Time overlap with existing operation at {existing.start_time.strftime('%H:%M')} "
                    f"on {existing.day_of_week}"
                )

        return conflicts

    def get_time_optimization_tips(self, operation_type: str) -> List[str]:
        """
        Get general time optimization tips for an operation type.

        Args:
            operation_type: Type of operation

        Returns:
            List of optimization tips
        """
        tips = []

        if self.time_patterns.best_performing_hours:
            best_hour = self.time_patterns.best_performing_hours[0][0]
            tips.append(f"Best performing hour: {best_hour}:00")

        if self.time_patterns.off_peak_hours:
            off_peak_list = sorted(list(self.time_patterns.off_peak_hours))
            tips.append(f"Consider off-peak hours: {off_peak_list}")

        tips.extend([
            "Avoid peak business hours (9 AM - 5 PM) to reduce costs",
            "Consider weekends for lower competition and costs",
            "Schedule during local off-hours of target region",
            "Monitor and adapt based on actual performance data"
        ])

        return tips

    def export_time_analysis(self) -> Dict:
        """Export comprehensive time analysis data."""
        return {
            "patterns_analyzed": self._patterns_analyzed,
            "hour_performance": dict(self.time_patterns.hour_performance),
            "day_performance": dict(self.time_patterns.day_performance),
            "peak_hours": list(self.time_patterns.peak_hours),
            "off_peak_hours": list(self.time_patterns.off_peak_hours),
            "best_performing_hours": self.time_patterns.best_performing_hours,
            "cost_patterns": dict(self.time_patterns.cost_patterns),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }


# Global time advisor instance
_global_time_advisor = TimeAdvisor()


def get_global_time_advisor() -> TimeAdvisor:
    """Get the global time advisor instance."""
    return _global_time_advisor


async def get_optimal_schedule(
    operation_type: str,
    duration_hours: float = 1.0,
    preferred_timezone: str = "UTC",
    tempo: ScrapeTempo = ScrapeTempo.HUMAN
) -> TimeRecommendation:
    """
    Convenience function for getting optimal scheduling recommendations.

    Args:
        operation_type: Type of scraping operation
        duration_hours: Expected operation duration
        preferred_timezone: Preferred timezone
        tempo: Scraping tempo preference

    Returns:
        TimeRecommendation with optimal scheduling
    """
    return await _global_time_advisor.get_optimal_schedule(
        operation_type, duration_hours, preferred_timezone, tempo
    )
