from intelligence.context import ScrapeContext
from intelligence.results import IntelligenceResult

class IntelligenceAdvisor:
    def evaluate(self, context: ScrapeContext) -> IntelligenceResult:
        # RULE: no IO, no network, no execution

        risk_score = 0.05 if context.dry_run else 0.25
        cost_estimate = 0.0 if context.dry_run else 1.0
        priority = 1 if context.dry_run else 5
        mode = "DRY_RUN" if context.dry_run else "NORMAL"

        return IntelligenceResult(
            risk_score=risk_score,
            cost_estimate=cost_estimate,
            priority=priority,
            recommended_execution_mode=mode,
            reason="Static intelligence evaluation"
        )
