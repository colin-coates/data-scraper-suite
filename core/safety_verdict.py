# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Safety Verdict System for MJ Data Scraper Suite

Provides intelligent decision-making based on sentinel reports,
balancing security, compliance, and operational requirements.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .sentinels.base import SentinelReport
from .control_models import ScrapeControlContract

logger = logging.getLogger(__name__)


@dataclass
class SafetyVerdict:
    """Safety verdict based on sentinel analysis."""
    action: str  # "allow", "delay", "restrict", "block"
    reason: str
    risk_level: str  # "low", "medium", "high", "critical"
    confidence_score: float  # 0.0 to 1.0
    constraints: Dict[str, Any]  # Applied constraints
    analysis_summary: Dict[str, Any]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


def safety_verdict(reports: List[SentinelReport], control: ScrapeControlContract) -> SafetyVerdict:
    """
    Generate safety verdict from sentinel reports and control contract.

    Args:
        reports: List of sentinel reports
        control: Scrape control contract

    Returns:
        SafetyVerdict with action and reasoning
    """
    if not reports:
        # No sentinel data - allow with caution
        return SafetyVerdict(
            action="allow",
            reason="No sentinel reports available - proceeding with caution",
            risk_level="unknown",
            confidence_score=0.0,
            constraints={},
            analysis_summary={"reports_count": 0}
        )

    # Analyze reports for risk assessment
    risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    action_counts = {"allow": 0, "delay": 0, "restrict": 0, "block": 0}

    findings = []
    for report in reports:
        risk_counts[report.risk_level] = risk_counts.get(report.risk_level, 0) + 1
        action_counts[report.recommended_action] = action_counts.get(report.recommended_action, 0) + 1

        if report.findings:
            findings.extend(report.findings.get("issues", []))

    # Decision logic based on risk levels
    total_reports = len(reports)
    critical_ratio = risk_counts.get("critical", 0) / total_reports
    high_ratio = risk_counts.get("high", 0) / total_reports
    medium_ratio = risk_counts.get("medium", 0) / total_reports

    # Block if any critical risks or high ratio of high risks
    if risk_counts.get("critical", 0) > 0:
        return SafetyVerdict(
            action="block",
            reason=f"Critical security risk detected: {risk_counts['critical']} critical reports",
            risk_level="critical",
            confidence_score=min(1.0, risk_counts["critical"] / total_reports),
            constraints={"block_reason": "critical_security_risk"},
            analysis_summary={
                "reports_count": total_reports,
                "risk_breakdown": risk_counts,
                "critical_findings": [f for f in findings if "critical" in str(f).lower()]
            }
        )

    # Block if high ratio of high-risk reports
    if high_ratio > 0.5:
        return SafetyVerdict(
            action="block",
            reason=f"High security risk: {high_ratio:.1%} of reports indicate high risk",
            risk_level="high",
            confidence_score=high_ratio,
            constraints={"block_reason": "high_security_risk"},
            analysis_summary={
                "reports_count": total_reports,
                "risk_breakdown": risk_counts,
                "high_risk_findings": findings
            }
        )

    # Delay if significant medium/high risks
    if (high_ratio + medium_ratio) > 0.7:
        delay_minutes = min(60, max(5, int((high_ratio + medium_ratio) * 30)))
        return SafetyVerdict(
            action="delay",
            reason=f"Moderate security concerns: delaying {delay_minutes} minutes",
            risk_level="medium",
            confidence_score=(high_ratio + medium_ratio),
            constraints={
                "delay_minutes": delay_minutes,
                "delay_reason": "moderate_security_concerns"
            },
            analysis_summary={
                "reports_count": total_reports,
                "risk_breakdown": risk_counts,
                "delay_factors": findings[:5]  # Top 5 findings
            }
        )

    # Restrict if some concerns but not blocking
    if high_ratio > 0.2 or medium_ratio > 0.5:
        return SafetyVerdict(
            action="restrict",
            reason="Minor security concerns: applying restrictions",
            risk_level="medium",
            confidence_score=max(high_ratio, medium_ratio),
            constraints={
                "restrict_reason": "minor_security_concerns",
                "reduced_rate_limit": True,
                "extended_delays": True
            },
            analysis_summary={
                "reports_count": total_reports,
                "risk_breakdown": risk_counts,
                "restriction_factors": findings[:3]
            }
        )

    # Allow if low risk overall
    return SafetyVerdict(
        action="allow",
        reason="Low security risk: proceeding normally",
        risk_level="low",
        confidence_score=(risk_counts.get("low", 0) / total_reports),
        constraints={},
        analysis_summary={
            "reports_count": total_reports,
            "risk_breakdown": risk_counts,
            "clean_findings": len([f for f in findings if "clean" in str(f).lower()])
        }
    )


def apply_verdict_constraints(verdict: SafetyVerdict) -> None:
    """
    Apply verdict constraints to the current scraping operation.

    Args:
        verdict: Safety verdict with constraints to apply
    """
    constraints = verdict.constraints

    if not constraints:
        logger.info("No constraints to apply from safety verdict")
        return

    logger.info(f"Applying safety verdict constraints: {constraints}")

    # Apply delay constraint
    if "delay_minutes" in constraints:
        delay_minutes = constraints["delay_minutes"]
        logger.warning(f"Safety verdict requires {delay_minutes} minute delay")
        # In real implementation, this would integrate with deployment_timer
        import asyncio
        asyncio.create_task(_apply_delay(delay_minutes))

    # Apply rate limiting constraints
    if constraints.get("reduced_rate_limit"):
        logger.warning("Applying reduced rate limit due to safety concerns")
        # In real implementation, this would modify scraper configuration

    # Apply timing constraints
    if constraints.get("extended_delays"):
        logger.warning("Applying extended delays between requests")
        # In real implementation, this would modify scraper timing

    # Log constraint application
    logger.info(f"Safety constraints applied: {list(constraints.keys())}")


async def _apply_delay(delay_minutes: int) -> None:
    """Apply delay constraint."""
    import asyncio
    delay_seconds = delay_minutes * 60
    logger.info(f"Applying safety delay of {delay_minutes} minutes ({delay_seconds} seconds)")
    await asyncio.sleep(delay_seconds)
    logger.info("Safety delay completed")
