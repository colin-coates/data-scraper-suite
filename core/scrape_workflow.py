# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Scrape Workflow for MJ Data Scraper Suite

High-level workflow orchestration for scraping operations with
governance, cost control, and compliance enforcement.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from .control_models import ScrapeControlContract
from .deployment_timer import DeploymentTimer
from .cost_governor import CostGovernor
from .authorization import AuthorizationGate

logger = logging.getLogger(__name__)


async def start_scrape(control: ScrapeControlContract) -> Dict[str, Any]:
    """
    Start a scraping operation with full governance controls.

    This is the main entry point for governed scraping operations.
    Validates authorization, waits for deployment windows, initializes
    cost controls, and executes the scraping workflow.

    Args:
        control: Complete control contract with governance parameters

    Returns:
        Dict containing operation results and governance metrics

    Raises:
        ValueError: If governance checks fail
        TimeoutError: If deployment window constraints cannot be met
        RuntimeError: If scraping execution fails
    """
    operation_id = f"scrape_{int(asyncio.get_event_loop().time())}"
    logger.info(f"Starting governed scrape operation: {operation_id}")

    # Step 1: Authorization validation
    try:
        AuthorizationGate.validate(control.authorization)
        logger.info("Authorization validated successfully")
    except RuntimeError as e:
        logger.error(f"Authorization failed: {e}")
        raise ValueError(f"Authorization validation failed: {e}")

    # Step 2: Deployment window synchronization
    try:
        await DeploymentTimer.await_window(control.deployment_window)
        logger.info("Deployment window opened - proceeding with execution")
    except (ValueError, TimeoutError) as e:
        logger.error(f"Deployment window error: {e}")
        raise

    # Step 3: Cost governor initialization
    try:
        cost_governor = await CostGovernor.initialize(control.budget, operation_id)
        logger.info(f"Cost governor initialized with budget: {control.budget.max_records} records")
    except Exception as e:
        logger.error(f"Cost governor initialization failed: {e}")
        raise RuntimeError(f"Failed to initialize cost controls: {e}")

    # Step 4: Execute scraping operation
    try:
        result = await run_scrape(control, cost_governor)

        # Step 5: Final governance checks and reporting
        final_status = await finalize_scrape(operation_id, cost_governor, result)

        logger.info(f"Scraping operation completed: {operation_id}")
        return final_status

    except Exception as e:
        logger.error(f"Scraping operation failed: {e}")
        # Ensure cleanup happens even on failure
        await cost_governor.cleanup()
        raise
    finally:
        # Cleanup resources
        await cost_governor.cleanup()


async def run_scrape(control: ScrapeControlContract, cost_governor: CostGovernor) -> Dict[str, Any]:
    """
    Execute the actual scraping operation with governance monitoring.

    Args:
        control: Control contract with scraping parameters
        cost_governor: Active cost governor for resource monitoring

    Returns:
        Dict containing scraping results
    """
    logger.info(f"Executing scrape with tempo: {control.tempo.value}")

    # This is where the actual scraping logic would go
    # For now, we'll simulate a scraping operation

    result = {
        "operation_id": cost_governor.operation_id,
        "tempo": control.tempo.value,
        "intent": control.intent.get_target_criteria(),
        "pages_scraped": 0,
        "records_collected": 0,
        "errors": [],
        "start_time": asyncio.get_event_loop().time(),
        "end_time": None,
        "success": False
    }

    try:
        # Simulate scraping with tempo-based delays
        tempo_settings = control.get_tempo_settings()

        # Simulate page scraping
        for page_num in range(min(10, control.budget.max_pages)):  # Simulate up to 10 pages
            if cost_governor.should_shutdown():
                logger.warning("Cost budget exceeded - stopping scrape")
                break

            # Simulate page processing
            await asyncio.sleep(tempo_settings["base_delay"])

            # Record page scraped
            if not cost_governor.record_page_scraped():
                logger.warning("Page budget exceeded")
                break

            result["pages_scraped"] += 1

            # Simulate record extraction (random between 5-20 records per page)
            import random
            records_found = random.randint(5, 20)
            if not cost_governor.record_records_collected(records_found):
                logger.warning("Record budget exceeded")
                break

            result["records_collected"] += records_found

            # Simulate browser/memory usage
            cost_governor.record_browser_usage(1)
            cost_governor.record_memory_usage(random.uniform(50, 200))

            logger.debug(f"Processed page {page_num + 1}: {records_found} records")

        result["success"] = True
        result["end_time"] = asyncio.get_event_loop().time()

        logger.info(f"Scraping simulation completed: {result['pages_scraped']} pages, {result['records_collected']} records")

    except Exception as e:
        result["errors"].append(str(e))
        result["end_time"] = asyncio.get_event_loop().time()
        logger.error(f"Scraping execution failed: {e}")

    return result


async def finalize_scrape(operation_id: str, cost_governor: CostGovernor, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finalize scraping operation with governance reporting.

    Args:
        operation_id: Unique operation identifier
        cost_governor: Cost governor with final metrics
        result: Scraping operation results

    Returns:
        Dict with complete operation summary
    """
    # Get final budget status
    budget_status = cost_governor.get_budget_status()

    # Get optimization recommendations
    recommendations = cost_governor.get_optimization_recommendations()

    # Get authorization audit summary
    audit_summary = AuthorizationGate.get_audit_summary()

    final_status = {
        "operation_id": operation_id,
        "success": result["success"],
        "scraping_results": {
            "pages_scraped": result["pages_scraped"],
            "records_collected": result["records_collected"],
            "duration_seconds": result["end_time"] - result["start_time"] if result["end_time"] else 0,
            "errors": result["errors"]
        },
        "governance_status": {
            "budget_compliance": budget_status["within_budget"],
            "budget_utilization": budget_status["budget_utilization"],
            "cost_metrics": budget_status["cost_metrics"],
            "alerts": budget_status["alerts"],
            "recommendations": recommendations
        },
        "authorization_audit": audit_summary,
        "compliance_status": {
            "budget_compliant": budget_status["within_budget"],
            "authorization_valid": audit_summary["authorization_rate"] > 0,
            "recommendations_count": len(recommendations)
        }
    }

    logger.info(f"Operation finalized: {operation_id} - Success: {result['success']}, Budget compliant: {budget_status['within_budget']}")

    return final_status


async def validate_scrape_readiness(control: ScrapeControlContract) -> Dict[str, Any]:
    """
    Validate that a scraping operation is ready to execute.

    Performs all governance checks without actually starting the scrape.
    Useful for pre-flight validation.

    Args:
        control: Control contract to validate

    Returns:
        Dict with validation results
    """
    validation_result = {
        "ready": False,
        "checks": {},
        "issues": []
    }

    # Authorization check
    try:
        auth_result = AuthorizationGate.validate(control.authorization)
        validation_result["checks"]["authorization"] = {
            "passed": auth_result.authorized,
            "expires_in_seconds": auth_result.expires_in_seconds
        }
        if not auth_result.authorized:
            validation_result["issues"].append(f"Authorization: {auth_result.reason}")
    except ValueError as e:
        validation_result["checks"]["authorization"] = {"passed": False, "error": str(e)}
        validation_result["issues"].append(f"Authorization: {e}")

    # Deployment window check
    current_time = asyncio.get_event_loop().time()
    window_status = "unknown"

    if DeploymentTimer.is_window_expired(control.deployment_window):
        window_status = "expired"
        validation_result["issues"].append("Deployment window has expired")
    else:
        remaining_time = DeploymentTimer.get_remaining_window_time(control.deployment_window)
        if remaining_time > 0:
            window_status = "ready"
        else:
            window_status = "waiting"
            validation_result["issues"].append(f"Waiting for deployment window to open")

    validation_result["checks"]["deployment_window"] = {
        "status": window_status,
        "remaining_seconds": DeploymentTimer.get_remaining_window_time(control.deployment_window)
    }

    # Budget validation
    budget_issues = []
    budget = control.budget

    # Check for obviously invalid budgets
    if budget.max_runtime_minutes <= 0:
        budget_issues.append("Invalid runtime budget")
    if budget.max_pages <= 0:
        budget_issues.append("Invalid page budget")
    if budget.max_records <= 0:
        budget_issues.append("Invalid record budget")

    validation_result["checks"]["budget"] = {
        "valid": len(budget_issues) == 0,
        "issues": budget_issues
    }

    if budget_issues:
        validation_result["issues"].extend(budget_issues)

    # Overall readiness
    validation_result["ready"] = (
        validation_result["checks"]["authorization"]["passed"] and
        validation_result["checks"]["deployment_window"]["status"] in ["ready", "waiting"] and
        validation_result["checks"]["budget"]["valid"]
    )

    return validation_result


# Example usage and testing functions
async def example_governed_scrape():
    """
    Example of how to use the governed scraping workflow.
    """
    from datetime import datetime, timedelta
    from .control_models import (
        ScrapeIntent, ScrapeBudget, ScrapeTempo,
        DeploymentWindow, ScrapeAuthorization
    )

    # Create control contract
    intent = ScrapeIntent(
        geography={"country": "US", "regions": ["West Coast"]},
        events={"weddings": True, "corporate_events": False},
        sources=["linkedin", "company_websites"]
    )

    budget = ScrapeBudget(
        max_runtime_minutes=30,
        max_pages=50,
        max_records=500,
        max_browser_instances=2,
        max_memory_mb=1024
    )

    deployment_window = DeploymentWindow(
        earliest_start=datetime.utcnow(),
        latest_start=datetime.utcnow() + timedelta(hours=2),
        max_duration_minutes=30,
        timezone="UTC"
    )

    authorization = ScrapeAuthorization(
        approved_by="compliance@mountainjewels.com",
        purpose="Lead generation for luxury jewelry market research",
        approval_timestamp=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(days=1)
    )

    control = ScrapeControlContract(
        intent=intent,
        budget=budget,
        tempo=ScrapeTempo.HUMAN,
        deployment_window=deployment_window,
        authorization=authorization
    )

    # Validate readiness
    readiness = await validate_scrape_readiness(control)
    print(f"Scrape readiness: {readiness['ready']}")
    if readiness['issues']:
        print(f"Issues: {readiness['issues']}")

    # Execute scrape if ready
    if readiness['ready']:
        result = await start_scrape(control)
        print(f"Scrape completed: {result['success']}")
        print(f"Records collected: {result['scraping_results']['records_collected']}")
        print(f"Budget compliant: {result['governance_status']['budget_compliance']}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_governed_scrape())
