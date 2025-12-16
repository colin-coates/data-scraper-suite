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
from datetime import datetime
from typing import Dict, Any, Optional

from .control_models import ScrapeControlContract, ScrapeTempo
from .deployment_timer import DeploymentTimer
from .cost_governor import CostGovernor
from .authorization import AuthorizationGate
from .scrape_telemetry import emit_telemetry

logger = logging.getLogger(__name__)


async def run_scraper(scraper, control: ScrapeControlContract):
    """
    Execute a scraper with full governance controls and validation.

    This function orchestrates the complete scraper execution workflow:
    1. Authorization validation
    2. Intent validation
    3. Timer initialization
    4. Budget enforcement
    5. Scraper execution

    Args:
        scraper: The scraper instance to execute
        control: Complete control contract with governance parameters

    Returns:
        Execution results with governance metrics

    Raises:
        ValueError: If governance validations fail
        RuntimeError: If execution encounters errors
    """
    # Step 1: Authorization validation
    AuthorizationGate.validate(control.authorization)

    # Step 2: Intent validation
    validate_intent(control)

    # Step 3: Start timer (deployment window)
    await DeploymentTimer.await_window(control.deployment_window)

    # Step 4: Enforce budget
    cost_governor = await CostGovernor.initialize(control.budget)

    try:
        # Step 5: Run scraper with monitoring
        result = await run_scraper_with_monitoring(scraper, control, cost_governor)

        # Finalize with governance reporting
        final_status = await finalize_scraper_execution(control, cost_governor, result)
        return final_status

    except Exception as e:
        logger.error(f"Scraper execution failed: {e}")
        raise
    finally:
        # Cleanup resources
        if 'cost_governor' in locals():
            await cost_governor.cleanup()


def validate_intent(control: ScrapeControlContract) -> None:
    """
    Validate the scraping intent against business rules and constraints.

    Args:
        control: Control contract containing intent specifications

    Raises:
        ValueError: If intent validation fails
    """
    intent = control.intent

    # Validate required fields
    if not intent.geography:
        raise ValueError("Intent must specify geography targeting")

    if not intent.sources:
        raise ValueError("Intent must specify data sources")

    # Validate geography constraints (allow common geographic keys)
    valid_geo_keys = ["country", "region", "state", "city", "states", "countries", "regions"]
    for geo_key in intent.geography.keys():
        if geo_key not in valid_geo_keys:
            raise ValueError(f"Invalid geography key: {geo_key}. Valid keys: {valid_geo_keys}")

    # Validate source constraints
    allowed_sources = [
        "linkedin", "facebook", "twitter", "instagram",
        "company_websites", "news", "public_records",
        "business_directories", "social_media"
    ]

    for source in intent.sources:
        if source not in allowed_sources:
            raise ValueError(f"Unsupported data source: {source}")

    # Validate demographic constraints (if provided)
    if intent.demographics:
        valid_demo_keys = ["age_range", "income_tier", "gender", "occupation"]
        for demo_key in intent.demographics.keys():
            if demo_key not in valid_demo_keys:
                raise ValueError(f"Invalid demographic key: {demo_key}")

    # Validate event constraints (if provided)
    if intent.events:
        valid_event_keys = ["weddings", "corporate", "social", "professional"]
        for event_key in intent.events.keys():
            if event_key not in valid_event_keys:
                raise ValueError(f"Invalid event key: {event_key}")

    logger.info(f"Intent validation passed for {len(intent.sources)} sources")


async def run_scraper_with_monitoring(scraper, control: ScrapeControlContract, cost_governor: CostGovernor) -> Dict[str, Any]:
    """
    Execute scraper with real-time monitoring and governance.

    Args:
        scraper: The scraper instance to execute
        control: Control contract with governance parameters
        cost_governor: Active cost governor for resource monitoring

    Returns:
        Execution results with monitoring data
    """
    import time

    start_time = time.time()
    result = {
        "success": False,
        "records_found": 0,
        "errors": [],
        "execution_time": 0,
        "cost_incurred": 0,
        "budget_remaining": 0
    }

    try:
        logger.info(f"Starting scraper execution with governance monitoring")

        # Execute the scraper (this would be the actual scraper.run() call)
        # For now, simulate execution with monitoring
        execution_result = await simulate_scraper_execution(scraper, control, cost_governor)

        result.update(execution_result)
        result["success"] = True

        logger.info(f"Scraper execution completed: {result['records_found']} records")

    except Exception as e:
        result["errors"].append(str(e))
        logger.error(f"Scraper execution error: {e}")
        raise

    finally:
        execution_time = time.time() - start_time
        result["execution_time"] = execution_time

        # Get final cost status
        cost_status = cost_governor.get_budget_status()
        result["cost_incurred"] = cost_status.get("total_cost", 0)
        result["budget_remaining"] = (
            cost_status.get("budget_limits", {}).get("max_runtime_minutes", 0) -
            cost_status.get("current_usage", {}).get("runtime_minutes", 0)
        )

    return result


async def simulate_scraper_execution(scraper, control: ScrapeControlContract, cost_governor: CostGovernor) -> Dict[str, Any]:
    """
    Simulate scraper execution with monitoring (replace with actual scraper logic).

    Args:
        scraper: Scraper instance
        control: Control contract
        cost_governor: Cost governor for monitoring

    Returns:
        Simulated execution results
    """
    import random
    import asyncio

    # Simulate execution time and records based on control parameters
    tempo_settings = control.get_tempo_settings()
    base_delay = tempo_settings.get("base_delay", 3.0)

    # Simulate variable execution based on source type and tempo
    if "linkedin" in control.intent.sources:
        records_base = 150
        time_base = 120
    elif "facebook" in control.intent.sources:
        records_base = 80
        time_base = 90
    else:
        records_base = 100
        time_base = 60

    # Apply tempo multiplier
    if control.tempo == ScrapeTempo.FORENSIC:
        time_multiplier = 3.0
        record_multiplier = 0.7
    elif control.tempo == ScrapeTempo.AGGRESSIVE:
        time_multiplier = 0.5
        record_multiplier = 1.3
    else:  # HUMAN
        time_multiplier = 1.0
        record_multiplier = 1.0

    execution_time = time_base * time_multiplier
    expected_records = int(records_base * record_multiplier)

    # Simulate execution with monitoring
    records_found = 0
    for i in range(min(10, expected_records // 10)):  # Simulate in batches
        if cost_governor.should_shutdown():
            logger.warning("Cost budget exceeded during execution")
            break

        # Simulate processing delay
        await asyncio.sleep(base_delay)

        # Simulate records found in this batch
        batch_records = random.randint(5, 15)
        records_found += batch_records

        # Update cost monitoring
        cost_governor.record_page_scraped()
        cost_governor.record_records_collected(batch_records)
        cost_governor.record_browser_usage(1)

        if records_found >= expected_records:
            break

    # Simulate some potential errors (rare)
    if random.random() < 0.05:  # 5% chance
        raise RuntimeError("Simulated scraper execution error")

    return {
        "records_found": records_found,
        "execution_time": execution_time,
        "simulated": True
    }


async def finalize_scraper_execution(control: ScrapeControlContract, cost_governor: CostGovernor, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finalize scraper execution with comprehensive governance reporting.

    Args:
        control: Control contract
        cost_governor: Cost governor with final metrics
        result: Execution results

    Returns:
        Complete execution summary with governance metrics
    """
    # Get final governance status
    cost_status = cost_governor.get_budget_status()

    # Get authorization audit summary
    audit_summary = AuthorizationGate.get_audit_summary()

    # Compile comprehensive execution report
    final_report = {
        "execution_id": f"exec_{int(asyncio.get_event_loop().time())}",
        "timestamp": datetime.utcnow().isoformat(),
        "control_contract": {
            "tempo": control.tempo.value,
            "sources": control.intent.sources,
            "budget_limit": control.budget.max_records
        },
        "execution_results": {
            "success": result["success"],
            "records_found": result["records_found"],
            "execution_time_seconds": result["execution_time"],
            "cost_incurred": result["cost_incurred"],
            "errors": result["errors"]
        },
        "governance_status": {
            "budget_compliance": cost_status.get("within_budget", True),
            "budget_utilization": cost_status.get("budget_utilization", 0.0),
            "authorization_valid": audit_summary.get("authorization_rate", 1) > 0,
            "deployment_window_respected": True  # Would be validated earlier
        },
        "performance_metrics": {
            "records_per_second": result["records_found"] / max(result["execution_time"], 1),
            "cost_per_record": result["cost_incurred"] / max(result["records_found"], 1),
            "budget_efficiency": result["records_found"] / control.budget.max_records
        },
        "recommendations": cost_governor.get_optimization_recommendations()
    }

    # Emit telemetry data
    blocked_reason = ""
    if "Governance violation" in (result.get("errors", [""])[0] if result.get("errors") else ""):
        blocked_reason = result["errors"][0]

    await emit_telemetry(
        scraper=control.intent.sources[0] if control.intent.sources else "unknown",
        role=getattr(control.intent, 'allowed_role', 'unknown'),
        cost_estimate=result.get("cost_incurred", 0),
        records_found=result.get("records_found", 0),
        blocked_reason=blocked_reason,
        runtime=result.get("execution_time_seconds", result.get("response_time", 0))
    )

    logger.info(
        f"Scraper execution finalized: {result['records_found']} records, "
        f"${result['cost_incurred']:.2f} cost, "
        f"{'budget compliant' if cost_status['within_budget'] else 'budget exceeded'}"
    )

    return final_report


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
