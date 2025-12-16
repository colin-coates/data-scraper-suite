# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Sentinel Orchestrator for MJ Data Scraper Suite

Coordinates multiple sentinels, manages their execution, and provides
unified risk assessment and decision-making capabilities.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .base import SentinelReport, SentinelRunner, SentinelConfig
from .performance_sentinel import PerformanceSentinel, create_performance_sentinel
from .network_sentinel import NetworkSentinel, create_network_sentinel
from .waf_sentinel import WafSentinel, create_waf_sentinel
from .malware_sentinel import MalwareSentinel, create_malware_sentinel

logger = logging.getLogger(__name__)


class OrchestrationMode(Enum):
    """Execution modes for sentinel orchestration."""
    PARALLEL = "parallel"      # Run all sentinels concurrently
    SEQUENTIAL = "sequential"  # Run sentinels one after another
    DEPENDENT = "dependent"    # Run sentinels based on previous results


class OrchestrationStrategy(Enum):
    """Decision strategies for aggregating sentinel results."""
    CONSERVATIVE = "conservative"  # Highest risk level wins
    MAJORITY = "majority"         # Majority vote on risk levels
    WEIGHTED = "weighted"         # Risk-weighted scoring
    VETO = "veto"                 # Any critical blocks everything


@dataclass
class OrchestrationResult:
    """Aggregated result from sentinel orchestration."""
    orchestration_id: str
    timestamp: datetime
    target: Dict[str, Any]
    sentinels_executed: List[str]
    individual_reports: List[SentinelReport] = field(default_factory=list)
    aggregated_risk_level: str = "low"
    aggregated_action: str = "allow"
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    decision_factors: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "orchestration_id": self.orchestration_id,
            "timestamp": self.timestamp.isoformat(),
            "target": self.target,
            "sentinels_executed": self.sentinels_executed,
            "individual_reports": [report.dict() for report in self.individual_reports],
            "aggregated_risk_level": self.aggregated_risk_level,
            "aggregated_action": self.aggregated_action,
            "execution_time": self.execution_time,
            "success": self.success,
            "error_message": self.error_message,
            "decision_factors": self.decision_factors
        }


class SentinelOrchestrator:
    """
    Orchestrates multiple sentinels for comprehensive risk assessment.

    Manages sentinel execution, result aggregation, and coordinated decision-making
    to provide unified security and performance intelligence.
    """

    def __init__(self, mode: OrchestrationMode = OrchestrationMode.PARALLEL,
                 strategy: OrchestrationStrategy = OrchestrationStrategy.CONSERVATIVE):
        self.mode = mode
        self.strategy = strategy

        # Sentinel registry
        self.sentinels: Dict[str, SentinelRunner] = {}
        self.sentinel_configs: Dict[str, SentinelConfig] = {}

        # Execution tracking
        self.orchestrations_attempted = 0
        self.orchestrations_successful = 0
        self.orchestrations_failed = 0
        self.start_time = time.time()

        # Configuration
        self.max_concurrent_sentinels = 5
        self.timeout_per_sentinel = 30.0
        self.fail_fast = False  # Stop on first critical finding

        # Sentinel dependencies (for DEPENDENT mode)
        self.dependencies: Dict[str, Set[str]] = {}

        # Risk level hierarchy for aggregation
        self.risk_hierarchy = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4
        }

        # Action hierarchy for aggregation
        self.action_hierarchy = {
            "allow": 1,
            "delay": 2,
            "restrict": 3,
            "block": 4
        }

        logger.info(f"SentinelOrchestrator initialized with mode={mode.value}, strategy={strategy.value}")

    def register_sentinel(self, sentinel_name: str, sentinel_instance,
                         config: Optional[SentinelConfig] = None) -> None:
        """
        Register a sentinel with the orchestrator.

        Args:
            sentinel_name: Unique name for the sentinel
            sentinel_instance: The sentinel instance to register
            config: Optional configuration for the sentinel runner
        """
        if config is None:
            config = SentinelConfig(name=f"{sentinel_name}_runner")

        runner = SentinelRunner(sentinel_instance, config)
        self.sentinels[sentinel_name] = runner
        self.sentinel_configs[sentinel_name] = config

        logger.info(f"Registered sentinel: {sentinel_name} ({sentinel_instance.name})")

    def unregister_sentinel(self, sentinel_name: str) -> None:
        """Unregister a sentinel from the orchestrator."""
        if sentinel_name in self.sentinels:
            del self.sentinels[sentinel_name]
            del self.sentinel_configs[sentinel_name]
            logger.info(f"Unregistered sentinel: {sentinel_name}")

    def set_dependencies(self, dependencies: Dict[str, Set[str]]) -> None:
        """
        Set sentinel dependencies for DEPENDENT execution mode.

        Args:
            dependencies: Dict mapping sentinel names to sets of prerequisite sentinels
        """
        self.dependencies = dependencies
        logger.info(f"Set sentinel dependencies: {dependencies}")

    def add_dependency(self, sentinel: str, depends_on: str) -> None:
        """Add a dependency relationship between sentinels."""
        if sentinel not in self.dependencies:
            self.dependencies[sentinel] = set()
        self.dependencies[sentinel].add(depends_on)
        logger.info(f"Added dependency: {sentinel} depends on {depends_on}")

    async def orchestrate(self, target: Dict[str, Any],
                         sentinels_to_run: Optional[List[str]] = None) -> OrchestrationResult:
        """
        Execute sentinel orchestration for a target.

        Args:
            target: Target information to analyze
            sentinels_to_run: Optional list of sentinel names to execute

        Returns:
            OrchestrationResult with aggregated findings
        """
        orchestration_id = f"orch_{int(time.time() * 1000)}"
        start_time = time.time()

        try:
            self.orchestrations_attempted += 1

            # Determine which sentinels to run
            if sentinels_to_run is None:
                sentinels_to_run = list(self.sentinels.keys())
            else:
                # Validate requested sentinels exist
                sentinels_to_run = [s for s in sentinels_to_run if s in self.sentinels]

            if not sentinels_to_run:
                return OrchestrationResult(
                    orchestration_id=orchestration_id,
                    timestamp=datetime.utcnow(),
                    target=target,
                    sentinels_executed=[],
                    aggregated_risk_level="low",
                    aggregated_action="allow",
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message="No sentinels available to run"
                )

            # Execute sentinels based on mode
            if self.mode == OrchestrationMode.PARALLEL:
                reports = await self._execute_parallel(target, sentinels_to_run)
            elif self.mode == OrchestrationMode.SEQUENTIAL:
                reports = await self._execute_sequential(target, sentinels_to_run)
            elif self.mode == OrchestrationMode.DEPENDENT:
                reports = await self._execute_dependent(target, sentinels_to_run)
            else:
                raise ValueError(f"Unsupported orchestration mode: {self.mode}")

            # Aggregate results
            aggregated_result = self._aggregate_results(
                orchestration_id, target, sentinels_to_run, reports, start_time
            )

            self.orchestrations_successful += 1
            return aggregated_result

        except Exception as e:
            self.orchestrations_failed += 1
            execution_time = time.time() - start_time

            logger.error(f"Orchestration {orchestration_id} failed: {e}")

            return OrchestrationResult(
                orchestration_id=orchestration_id,
                timestamp=datetime.utcnow(),
                target=target,
                sentinels_executed=sentinels_to_run or [],
                aggregated_risk_level="critical",
                aggregated_action="block",
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

    async def _execute_parallel(self, target: Dict[str, Any],
                               sentinels: List[str]) -> List[SentinelReport]:
        """Execute sentinels in parallel."""
        logger.info(f"Executing {len(sentinels)} sentinels in parallel")

        # Create tasks for all sentinels
        tasks = []
        for sentinel_name in sentinels:
            if sentinel_name in self.sentinels:
                task = asyncio.create_task(
                    self._execute_sentinel_with_timeout(sentinel_name, target)
                )
                tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        reports = []
        for i, result in enumerate(results):
            sentinel_name = sentinels[i]
            if isinstance(result, Exception):
                logger.error(f"Sentinel {sentinel_name} failed: {result}")
                # Create error report
                reports.append(SentinelReport(
                    sentinel_name=sentinel_name,
                    domain="error",
                    timestamp=datetime.utcnow(),
                    risk_level="critical",
                    findings={"error": str(result)},
                    recommended_action="block"
                ))
            else:
                reports.append(result)

        return reports

    async def _execute_sequential(self, target: Dict[str, Any],
                                 sentinels: List[str]) -> List[SentinelReport]:
        """Execute sentinels sequentially."""
        logger.info(f"Executing {len(sentinels)} sentinels sequentially")

        reports = []
        for sentinel_name in sentinels:
            if sentinel_name in self.sentinels:
                report = await self._execute_sentinel_with_timeout(sentinel_name, target)
                reports.append(report)

                # Check for fail-fast condition
                if self.fail_fast and report.risk_level == "critical":
                    logger.warning(f"Fail-fast triggered by {sentinel_name} critical finding")
                    break

        return reports

    async def _execute_dependent(self, target: Dict[str, Any],
                                sentinels: List[str]) -> List[SentinelReport]:
        """Execute sentinels based on dependencies."""
        logger.info(f"Executing {len(sentinels)} sentinels with dependencies")

        executed = set()
        reports = []

        # Simple dependency resolution (topological sort would be better for complex deps)
        for sentinel_name in sentinels:
            if sentinel_name in self.sentinels:
                # Check if dependencies are satisfied
                deps = self.dependencies.get(sentinel_name, set())
                if deps.issubset(executed):
                    report = await self._execute_sentinel_with_timeout(sentinel_name, target)
                    reports.append(report)
                    executed.add(sentinel_name)

                    # Check for fail-fast condition
                    if self.fail_fast and report.risk_level == "critical":
                        logger.warning(f"Fail-fast triggered by {sentinel_name} critical finding")
                        break
                else:
                    logger.warning(f"Skipping {sentinel_name} due to unsatisfied dependencies: {deps - executed}")

        return reports

    async def _execute_sentinel_with_timeout(self, sentinel_name: str,
                                           target: Dict[str, Any]) -> SentinelReport:
        """Execute a single sentinel with timeout."""
        try:
            runner = self.sentinels[sentinel_name]
            return await asyncio.wait_for(
                runner.probe_target(target),
                timeout=self.timeout_per_sentinel
            )
        except asyncio.TimeoutError:
            logger.error(f"Sentinel {sentinel_name} timed out after {self.timeout_per_sentinel}s")
            return SentinelReport(
                sentinel_name=sentinel_name,
                domain="timeout",
                timestamp=datetime.utcnow(),
                risk_level="high",
                findings={"error": "timeout", "timeout_seconds": self.timeout_per_sentinel},
                recommended_action="delay"
            )
        except Exception as e:
            logger.error(f"Sentinel {sentinel_name} execution failed: {e}")
            return SentinelReport(
                sentinel_name=sentinel_name,
                domain="error",
                timestamp=datetime.utcnow(),
                risk_level="critical",
                findings={"error": str(e)},
                recommended_action="block"
            )

    def _aggregate_results(self, orchestration_id: str, target: Dict[str, Any],
                          sentinels_executed: List[str], reports: List[SentinelReport],
                          start_time: float) -> OrchestrationResult:
        """Aggregate results from multiple sentinels."""

        execution_time = time.time() - start_time

        # Apply aggregation strategy
        if self.strategy == OrchestrationStrategy.CONSERVATIVE:
            aggregated_risk, aggregated_action, factors = self._aggregate_conservative(reports)
        elif self.strategy == OrchestrationStrategy.MAJORITY:
            aggregated_risk, aggregated_action, factors = self._aggregate_majority(reports)
        elif self.strategy == OrchestrationStrategy.WEIGHTED:
            aggregated_risk, aggregated_action, factors = self._aggregate_weighted(reports)
        elif self.strategy == OrchestrationStrategy.VETO:
            aggregated_risk, aggregated_action, factors = self._aggregate_veto(reports)
        else:
            raise ValueError(f"Unsupported aggregation strategy: {self.strategy}")

        return OrchestrationResult(
            orchestration_id=orchestration_id,
            timestamp=datetime.utcnow(),
            target=target,
            sentinels_executed=sentinels_executed,
            individual_reports=reports,
            aggregated_risk_level=aggregated_risk,
            aggregated_action=aggregated_action,
            execution_time=execution_time,
            success=True,
            decision_factors=factors
        )

    def _aggregate_conservative(self, reports: List[SentinelReport]) -> Tuple[str, str, Dict[str, Any]]:
        """Conservative aggregation - highest risk wins."""
        max_risk_level = "low"
        max_risk_value = 0
        max_action = "allow"
        max_action_value = 0

        risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        action_counts = {"allow": 0, "delay": 0, "restrict": 0, "block": 0}

        for report in reports:
            if report.success:
                risk_value = self.risk_hierarchy.get(report.risk_level, 1)
                if risk_value > max_risk_value:
                    max_risk_level = report.risk_level
                    max_risk_value = risk_value

                action_value = self.action_hierarchy.get(report.recommended_action, 1)
                if action_value > max_action_value:
                    max_action = report.recommended_action
                    max_action_value = action_value

                risk_counts[report.risk_level] = risk_counts.get(report.risk_level, 0) + 1
                action_counts[report.recommended_action] = action_counts.get(report.recommended_action, 0) + 1

        return max_risk_level, max_action, {
            "strategy": "conservative",
            "risk_counts": risk_counts,
            "action_counts": action_counts,
            "highest_risk_reports": [
                r.sentinel_name for r in reports
                if r.success and r.risk_level == max_risk_level
            ]
        }

    def _aggregate_majority(self, reports: List[SentinelReport]) -> Tuple[str, str, Dict[str, Any]]:
        """Majority vote aggregation."""
        risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        action_counts = {"allow": 0, "delay": 0, "restrict": 0, "block": 0}

        successful_reports = [r for r in reports if r.success]

        for report in successful_reports:
            risk_counts[report.risk_level] = risk_counts.get(report.risk_level, 0) + 1
            action_counts[report.recommended_action] = action_counts.get(report.recommended_action, 0) + 1

        # Find majority risk level
        majority_risk = max(risk_counts, key=risk_counts.get)
        majority_action = max(action_counts, key=action_counts.get)

        return majority_risk, majority_action, {
            "strategy": "majority",
            "risk_counts": risk_counts,
            "action_counts": action_counts,
            "total_successful": len(successful_reports),
            "total_reports": len(reports)
        }

    def _aggregate_weighted(self, reports: List[SentinelReport]) -> Tuple[str, str, Dict[str, Any]]:
        """Weighted scoring aggregation."""
        total_weight = 0
        risk_score = 0
        action_score = 0

        successful_reports = [r for r in reports if r.success]

        for report in successful_reports:
            # Weight by response time (faster = more reliable)
            weight = max(0.1, 1.0 / (1.0 + report.response_time))

            risk_value = self.risk_hierarchy.get(report.risk_level, 1)
            action_value = self.action_hierarchy.get(report.recommended_action, 1)

            risk_score += risk_value * weight
            action_score += action_value * weight
            total_weight += weight

        if total_weight == 0:
            return "low", "allow", {"strategy": "weighted", "error": "no_successful_reports"}

        avg_risk_score = risk_score / total_weight
        avg_action_score = action_score / total_weight

        # Convert back to risk levels
        if avg_risk_score >= 3.5:
            final_risk = "critical"
        elif avg_risk_score >= 2.5:
            final_risk = "high"
        elif avg_risk_score >= 1.5:
            final_risk = "medium"
        else:
            final_risk = "low"

        if avg_action_score >= 3.5:
            final_action = "block"
        elif avg_action_score >= 2.5:
            final_action = "restrict"
        elif avg_action_score >= 1.5:
            final_action = "delay"
        else:
            final_action = "allow"

        return final_risk, final_action, {
            "strategy": "weighted",
            "average_risk_score": avg_risk_score,
            "average_action_score": avg_action_score,
            "total_weight": total_weight,
            "total_successful": len(successful_reports)
        }

    def _aggregate_veto(self, reports: List[SentinelReport]) -> Tuple[str, str, Dict[str, Any]]:
        """Veto aggregation - any critical blocks everything."""
        veto_reports = [r for r in reports if r.success and r.risk_level == "critical"]

        if veto_reports:
            return "critical", "block", {
                "strategy": "veto",
                "veto_triggered": True,
                "veto_sentinels": [r.sentinel_name for r in veto_reports]
            }

        # If no veto, use conservative aggregation
        return self._aggregate_conservative(reports)

    def get_orchestrator_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the orchestrator."""
        uptime = time.time() - self.start_time

        sentinel_info = {}
        for name, runner in self.sentinels.items():
            sentinel_info[name] = runner.get_metrics()

        return {
            "mode": self.mode.value,
            "strategy": self.strategy.value,
            "registered_sentinels": list(self.sentinels.keys()),
            "dependencies": {k: list(v) for k, v in self.dependencies.items()},
            "metrics": {
                "orchestrations_attempted": self.orchestrations_attempted,
                "orchestrations_successful": self.orchestrations_successful,
                "orchestrations_failed": self.orchestrations_failed,
                "success_rate": (self.orchestrations_successful / self.orchestrations_attempted)
                               if self.orchestrations_attempted > 0 else 0.0,
                "uptime_seconds": uptime
            },
            "configuration": {
                "max_concurrent_sentinels": self.max_concurrent_sentinels,
                "timeout_per_sentinel": self.timeout_per_sentinel,
                "fail_fast": self.fail_fast
            },
            "sentinel_details": sentinel_info
        }

    def set_mode(self, mode: OrchestrationMode) -> None:
        """Set the orchestration execution mode."""
        self.mode = mode
        logger.info(f"Orchestration mode set to: {mode.value}")

    def set_strategy(self, strategy: OrchestrationStrategy) -> None:
        """Set the result aggregation strategy."""
        self.strategy = strategy
        logger.info(f"Aggregation strategy set to: {strategy.value}")

    def enable_fail_fast(self, enabled: bool = True) -> None:
        """Enable or disable fail-fast execution."""
        self.fail_fast = enabled
        logger.info(f"Fail-fast {'enabled' if enabled else 'disabled'}")

    def reset_metrics(self) -> None:
        """Reset orchestrator metrics."""
        self.orchestrations_attempted = 0
        self.orchestrations_successful = 0
        self.orchestrations_failed = 0
        logger.info("Orchestrator metrics reset")


# Factory functions for common orchestrator configurations

def create_comprehensive_orchestrator() -> SentinelOrchestrator:
    """
    Create an orchestrator with all available sentinels.

    Returns:
        Fully configured SentinelOrchestrator with all sentinels
    """
    orchestrator = SentinelOrchestrator(
        mode=OrchestrationMode.PARALLEL,
        strategy=OrchestrationStrategy.CONSERVATIVE
    )

    # Register all sentinels
    orchestrator.register_sentinel("performance", create_performance_sentinel())
    orchestrator.register_sentinel("network", create_network_sentinel())
    orchestrator.register_sentinel("waf", create_waf_sentinel())
    orchestrator.register_sentinel("malware", create_malware_sentinel())

    # Set up dependencies (malware depends on network being OK first)
    orchestrator.add_dependency("malware", "network")

    return orchestrator


def create_security_focused_orchestrator() -> SentinelOrchestrator:
    """
    Create an orchestrator focused on security sentinels.

    Returns:
        SentinelOrchestrator configured for security analysis
    """
    orchestrator = SentinelOrchestrator(
        mode=OrchestrationMode.SEQUENTIAL,
        strategy=OrchestrationStrategy.VETO
    )

    # Register security sentinels
    orchestrator.register_sentinel("network", create_network_sentinel())
    orchestrator.register_sentinel("waf", create_waf_sentinel())
    orchestrator.register_sentinel("malware", create_malware_sentinel())

    # Enable fail-fast for security issues
    orchestrator.enable_fail_fast(True)

    return orchestrator


def create_performance_orchestrator() -> SentinelOrchestrator:
    """
    Create an orchestrator focused on performance sentinels.

    Returns:
        SentinelOrchestrator configured for performance analysis
    """
    orchestrator = SentinelOrchestrator(
        mode=OrchestrationMode.PARALLEL,
        strategy=OrchestrationStrategy.WEIGHTED
    )

    # Register performance sentinels
    orchestrator.register_sentinel("performance", create_performance_sentinel())
    orchestrator.register_sentinel("network", create_network_sentinel())

    return orchestrator
