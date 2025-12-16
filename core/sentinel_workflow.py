# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Sentinel-Enhanced Scraping Workflow for MJ Data Scraper Suite

Provides a simplified, robust interface for sentinel-integrated scraping
with comprehensive error handling, logging, and telemetry.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from .sentinels.sentinel_orchestrator import run_sentinels
from .safety_verdict import safety_verdict, apply_verdict_constraints, SafetyVerdict
from .control_models import ScrapeControlContract
from .scrape_telemetry import emit_telemetry
from .scraper_engine import CoreScraperEngine, EngineConfig

logger = logging.getLogger(__name__)


class SentinelWorkflowError(Exception):
    """Custom exception for sentinel workflow errors."""
    def __init__(self, message: str, verdict: Optional[SafetyVerdict] = None):
        super().__init__(message)
        self.verdict = verdict


async def start_scrape_with_sentinels(
    control: ScrapeControlContract,
    enable_fallback: bool = True,
    sentinel_timeout: float = 30.0
) -> Dict[str, Any]:
    """
    Enhanced scraping workflow with comprehensive sentinel integration.

    This is an improved version of the original start_scrape function with:
    - Comprehensive error handling and logging
    - Telemetry emission for monitoring
    - Graceful fallbacks for sentinel failures
    - Timeout handling for sentinel operations
    - Audit trail generation
    - Integration with existing governance framework

    Args:
        control: Scrape control contract with governance rules
        enable_fallback: Allow scraping to proceed if sentinels fail
        sentinel_timeout: Maximum time to wait for sentinel analysis

    Returns:
        Dict with scraping results and sentinel analysis

    Raises:
        SentinelWorkflowError: If scraping is blocked by safety verdict
    """
    start_time = datetime.utcnow()
    workflow_id = f"workflow_{int(start_time.timestamp() * 1000)}"

    logger.info(f"🚀 Starting sentinel-enhanced scrape workflow {workflow_id}")
    logger.info(f"Target: {control.intent.sources}")

    try:
        # Phase 1: Prepare target for sentinel analysis
        target = _prepare_sentinel_target(control)
        logger.debug(f"Prepared sentinel target: {target}")

        # Phase 2: Run sentinel analysis with timeout
        logger.info("🔍 Running sentinel security analysis...")
        sentinel_start = datetime.utcnow()

        try:
            reports = await asyncio.wait_for(
                run_sentinels(target),
                timeout=sentinel_timeout
            )
            sentinel_duration = (datetime.utcnow() - sentinel_start).total_seconds()

            logger.info(f"✅ Sentinel analysis completed in {sentinel_duration:.2f}s: {len(reports)} reports")

            # Emit telemetry for sentinel analysis
            await emit_telemetry(
                scraper="sentinel_workflow",
                role="security",
                cost_estimate=sentinel_duration * 0.001,  # Rough cost estimate
                records_found=len(reports),
                blocked_reason=None,
                runtime=sentinel_duration
            )

        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Sentinel analysis timed out after {sentinel_timeout}s")
            if enable_fallback:
                logger.info("🔄 Proceeding with fallback (no sentinel data)")
                reports = []
            else:
                raise SentinelWorkflowError("Sentinel analysis timed out")

        except Exception as e:
            logger.error(f"❌ Sentinel analysis failed: {e}")
            if enable_fallback:
                logger.warning("🔄 Proceeding with fallback due to sentinel failure")
                reports = []
            else:
                raise SentinelWorkflowError(f"Sentinel analysis failed: {e}")

        # Phase 3: Generate safety verdict
        logger.info("⚖️ Generating safety verdict...")
        verdict = safety_verdict(reports, control)

        logger.info(f"🎯 Safety verdict: {verdict.action} ({verdict.risk_level} risk, {verdict.confidence_score:.1%} confidence)")
        logger.info(f"📋 Reason: {verdict.reason}")

        # Phase 4: Handle verdict actions
        if verdict.action == "block":
            logger.warning(f"🚫 Scraping blocked: {verdict.reason}")
            await _emit_blocked_telemetry(workflow_id, verdict, control)

            if not enable_fallback:
                raise SentinelWorkflowError(f"Scraping blocked: {verdict.reason}", verdict)

            # Even with fallback, we still block critical risks
            if verdict.risk_level == "critical":
                raise SentinelWorkflowError(f"Critical risk - scraping blocked: {verdict.reason}", verdict)

        elif verdict.action == "delay":
            delay_minutes = verdict.constraints.get("delay_minutes", 5)
            logger.warning(f"⏳ Applying safety delay: {delay_minutes} minutes")
            await _apply_safety_delay(delay_minutes, workflow_id)

        # Phase 5: Apply verdict constraints
        logger.info("🔧 Applying verdict constraints...")
        apply_verdict_constraints(verdict)

        # Phase 6: Execute scraping with monitoring
        logger.info("⚙️ Executing scraping operation...")
        scrape_result = await _execute_scraping_with_monitoring(control, verdict, workflow_id)

        # Phase 7: Finalize and emit telemetry
        total_duration = (datetime.utcnow() - start_time).total_seconds()
        final_result = {
            "workflow_id": workflow_id,
            "success": scrape_result["success"],
            "verdict": {
                "action": verdict.action,
                "risk_level": verdict.risk_level,
                "reason": verdict.reason,
                "confidence_score": verdict.confidence_score
            },
            "sentinel_analysis": {
                "reports_count": len(reports),
                "analysis_duration": sentinel_duration if 'sentinel_duration' in locals() else None,
                "risk_breakdown": verdict.analysis_summary.get("risk_breakdown", {})
            },
            "scraping_result": scrape_result,
            "total_duration": total_duration,
            "timestamp": datetime.utcnow().isoformat()
        }

        await emit_telemetry(
            scraper=f"sentinel_workflow_{control.intent.allowed_role or 'unknown'}",
            role=control.intent.allowed_role or "unknown",
            cost_estimate=scrape_result.get("cost_estimate", 0.0),
            records_found=scrape_result.get("records_found", 0),
            blocked_reason=verdict.reason if verdict.action == "block" else None,
            runtime=total_duration
        )

        logger.info(f"✅ Workflow {workflow_id} completed in {total_duration:.2f}s")
        return final_result

    except SentinelWorkflowError:
        # Re-raise sentinel workflow errors
        raise
    except Exception as e:
        logger.error(f"❌ Workflow {workflow_id} failed: {e}")
        # Emit failure telemetry
        total_duration = (datetime.utcnow() - start_time).total_seconds()
        await emit_telemetry(
            scraper="sentinel_workflow",
            role="error",
            cost_estimate=0.0,
            records_found=0,
            blocked_reason=str(e),
            runtime=total_duration
        )
        raise SentinelWorkflowError(f"Workflow execution failed: {e}")


def _prepare_sentinel_target(control: ScrapeControlContract) -> Dict[str, Any]:
    """Prepare target information for sentinel analysis."""
    # Extract primary domain from sources
    primary_domain = None
    if control.intent.sources:
        # Try to extract domain from first source
        source = control.intent.sources[0]
        if "://" in source:
            # It's a URL, extract domain
            try:
                from urllib.parse import urlparse
                parsed = urlparse(source)
                primary_domain = parsed.netloc
                if primary_domain.startswith("www."):
                    primary_domain = primary_domain[4:]
            except Exception:
                primary_domain = source
        else:
            primary_domain = source

    target = {
        "domain": primary_domain,
        "sources": control.intent.sources,
        "scraper_type": control.intent.allowed_role or "unknown",
        "event_type": control.intent.event_type,
        "geography": control.intent.geography or [],
        "has_budget": control.budget is not None,
        "max_runtime": control.budget.max_runtime_minutes if control.budget else None,
        "max_records": control.budget.max_records if control.budget else None
    }

    return target


async def _emit_blocked_telemetry(workflow_id: str, verdict: SafetyVerdict, control: ScrapeControlContract) -> None:
    """Emit telemetry for blocked scraping operations."""
    await emit_telemetry(
        scraper="sentinel_workflow",
        role="blocked",
        cost_estimate=0.0,
        records_found=0,
        blocked_reason=f"{verdict.action}: {verdict.reason}",
        runtime=0.0
    )


async def _apply_safety_delay(delay_minutes: int, workflow_id: str) -> None:
    """Apply safety delay with logging."""
    logger.info(f"⏳ Workflow {workflow_id}: Applying {delay_minutes} minute safety delay")

    # Emit delay telemetry
    await emit_telemetry(
        scraper="sentinel_workflow",
        role="delay",
        cost_estimate=delay_minutes * 0.001,  # Rough cost estimate
        records_found=0,
        blocked_reason=f"delay_{delay_minutes}min",
        runtime=delay_minutes * 60
    )

    await asyncio.sleep(delay_minutes * 60)
    logger.info(f"✅ Workflow {workflow_id}: Safety delay completed")


async def _execute_scraping_with_monitoring(
    control: ScrapeControlContract,
    verdict: SafetyVerdict,
    workflow_id: str
) -> Dict[str, Any]:
    """
    Execute scraping with additional monitoring based on verdict.

    This is a placeholder for the actual scraping execution.
    In production, this would integrate with the CoreScraperEngine.
    """
    logger.info(f"🏃 Executing scraping for workflow {workflow_id}")

    # Simulate scraping execution
    # In real implementation, this would call the scraper engine
    await asyncio.sleep(0.1)  # Simulate scraping time

    # Mock successful result
    result = {
        "success": True,
        "records_found": 42,
        "cost_estimate": 0.05,
        "duration": 0.1,
        "error_message": None,
        "verdict_applied": verdict.action
    }

    logger.info(f"✅ Scraping completed for workflow {workflow_id}: {result['records_found']} records")
    return result


# Simplified interface function for easy migration from the original
async def start_scrape(control: ScrapeControlContract) -> None:
    """
    Simplified interface matching the original function signature.

    This provides backward compatibility while using the enhanced workflow.
    """
    result = await start_scrape_with_sentinels(control)

    # Check if scraping was blocked
    if result["verdict"]["action"] in ["block", "delay"]:
        raise RuntimeError(f"Scrape aborted: {result['verdict']['reason']}")

    # In the original, this would call run_scraper(control)
    # For now, we'll just log the successful execution
    logger.info(f"Scrape completed successfully: {result['scraping_result']['records_found']} records found")


# Utility functions for workflow management
def get_workflow_status(workflow_id: str) -> Optional[Dict[str, Any]]:
    """
    Get status of a sentinel workflow.

    Note: This is a placeholder. In production, this would integrate
    with a workflow tracking system.
    """
    # Placeholder implementation
    return None


def cancel_workflow(workflow_id: str) -> bool:
    """
    Cancel a running sentinel workflow.

    Note: This is a placeholder. In production, this would integrate
    with the asyncio task management.
    """
    # Placeholder implementation
    logger.info(f"Workflow {workflow_id} cancellation requested")
    return True
