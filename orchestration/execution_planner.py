from orchestration.types import ExecutionPlan
from orchestration.retry_matrix import retries_for_mode
from intelligence import IntelligenceAdvisor, ScrapeContext

def plan_execution(
    scraper_name: str,
    target: str,
    metadata: dict,
    dry_run: bool,
) -> ExecutionPlan:
    advisor = IntelligenceAdvisor()

    ctx = ScrapeContext(
        scraper_name=scraper_name,
        target=target,
        metadata=metadata,
        dry_run=dry_run,
    )

    intel = advisor.evaluate(ctx)
    retries = retries_for_mode(intel.recommended_execution_mode)

    return ExecutionPlan(
        scraper_name=scraper_name,
        execution_mode=intel.recommended_execution_mode,
        retries=retries,
        dry_run=dry_run,
        reason=intel.reason,
    )
