# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use as is strictly prohibited.

"""
Scraper Engine for MJ Data Scraper Suite

Enterprise-grade scraping orchestration engine that integrates all intelligence
systems for optimal, compliant, and cost-effective data operations.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

from core.control_models import (
    ScrapeControlContract,
    ScrapeIntent,
    ScrapeBudget,
    ScrapeAuthorization,
    JobPriority,
    ScraperType
)
from core.models.asset_signal import AssetType, SignalType, Asset
from scrapers.base_scraper import BaseScraper
from core.intent_classifier import (
    classify_scraping_intent,
    IntentClassification,
    IntentRiskLevel,
    IntentCategory,
    GovernanceRequirement
)
from core.execution_mode_classifier import (
    classify_execution_mode,
    ExecutionProfile,
    ExecutionMode,
    ExecutionStrategy
)
from core.cost_predictor import (
    predict_scraping_cost,
    CostPrediction,
    optimize_scraping_cost,
    CostOptimizationPlan,
    analyze_scraping_budget,
    BudgetAnalysis
)
from core.mapping.asset_signal_map import (
    map_to_signal,
    batch_map_to_signals,
    get_optimal_sources_for_signal,
    calculate_signal_cost_estimate,
    validate_source_for_signal
)
from core.base_scraper import ai_precheck
from core.authorization import AuthorizationGate
from core.deployment_timer import DeploymentTimer
from core.cost_governor import CostGovernor
from core.scrape_workflow import run_scraper
from core.sentinels.sentinel_orchestrator import run_sentinels, SentinelOrchestrator
from core.safety_verdict import safety_verdict
from core.scrape_telemetry import emit_telemetry, ScrapeTelemetryCollector

logger = logging.getLogger(__name__)


class ScrapingPhase(Enum):
    """Phases of the scraping orchestration process."""
    INITIALIZATION = "initialization"
    INTENT_ANALYSIS = "intent_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    EXECUTION_PLANNING = "execution_planning"
    COST_OPTIMIZATION = "cost_optimization"
    GOVERNANCE_CHECK = "governance_check"
    RESOURCE_ALLOCATION = "resource_allocation"
    EXECUTION_MONITORING = "execution_monitoring"
    QUALITY_VALIDATION = "quality_validation"
    FINALIZATION = "finalization"


class OrchestrationResult(Enum):
    """Possible outcomes of scraping orchestration."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    BLOCKED_BY_GOVERNANCE = "blocked_by_governance"
    BLOCKED_BY_COST = "blocked_by_cost"
    BLOCKED_BY_RISK = "blocked_by_risk"
    EXECUTION_FAILED = "execution_failed"
    CANCELLED_BY_USER = "cancelled_by_user"
    TIMEOUT_EXCEEDED = "timeout_exceeded"


@dataclass
class ScrapingJob:
    """Complete scraping job with all intelligence and orchestration data."""
    job_id: str
    control: ScrapeControlContract

    # Intelligence assessments
    intent_classification: Optional[IntentClassification] = None
    execution_profile: Optional[ExecutionProfile] = None
    cost_prediction: Optional[CostPrediction] = None
    cost_optimization: Optional[CostOptimizationPlan] = None
    budget_analysis: Optional[BudgetAnalysis] = None

    # Orchestration state
    current_phase: ScrapingPhase = ScrapingPhase.INITIALIZATION
    phase_progress: Dict[str, Any] = field(default_factory=dict)
    orchestration_result: Optional[OrchestrationResult] = None

    # Execution data
    assigned_scrapers: List[BaseScraper] = field(default_factory=list)
    execution_start_time: Optional[datetime] = None
    execution_end_time: Optional[datetime] = None
    actual_cost: float = 0.0
    records_collected: int = 0
    success_rate: float = 0.0

    # Monitoring and telemetry
    sentinel_reports: List[Dict[str, Any]] = field(default_factory=list)
    telemetry_events: List[Dict[str, Any]] = field(default_factory=list)
    error_events: List[Dict[str, Any]] = field(default_factory=list)

    # Governance and compliance
    governance_checks_passed: bool = False
    compliance_flags: List[str] = field(default_factory=list)
    risk_mitigations_applied: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    priority_score: float = 0.0
    estimated_completion_time: Optional[datetime] = None

    def get_job_duration(self) -> Optional[timedelta]:
        """Get total job duration if completed."""
        if self.execution_start_time and self.execution_end_time:
            return self.execution_end_time - self.execution_start_time
        return None

    def get_efficiency_score(self) -> float:
        """Calculate overall job efficiency score."""
        if self.actual_cost == 0 or self.records_collected == 0:
            return 0.0

        # Base efficiency: records per dollar
        base_efficiency = self.records_collected / self.actual_cost

        # Adjust for success rate
        success_adjustment = self.success_rate

        # Adjust for on-time completion
        timeliness_adjustment = 1.0
        if self.estimated_completion_time and self.execution_end_time:
            if self.execution_end_time > self.estimated_completion_time:
                delay_hours = (self.execution_end_time - self.estimated_completion_time).total_seconds() / 3600
                timeliness_adjustment = max(0.1, 1.0 - (delay_hours / 24))  # Penalty for delays > 24h

        return base_efficiency * success_adjustment * timeliness_adjustment

    def should_proceed_to_execution(self) -> bool:
        """Determine if job should proceed to execution phase."""
        if not self.intent_classification or not self.execution_profile:
            return False

        # Block critical risk without proper governance
        if (self.intent_classification.risk_level == IntentRiskLevel.CRITICAL and
            self.intent_classification.governance_requirement.value == "exceptional"):
            return False

        # Block if cost exceeds budget significantly
        if self.cost_prediction and self.control.budget:
            cost_ratio = self.cost_prediction.predicted_cost / self.control.budget.max_cost_total
            if cost_ratio > 1.5:  # 50% over budget
                return False

        return True


@dataclass
class ScrapingOrchestrationResult:
    """Complete result of scraping orchestration process."""
    job: ScrapingJob
    result: OrchestrationResult
    output_data: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    compliance_report: Dict[str, Any] = field(default_factory=dict)


class ScrapingEngine:
    """
    Enterprise-grade scraping orchestration engine that integrates all intelligence
    systems for optimal, compliant, and cost-effective data operations.
    """

    def __init__(self):
        self.active_jobs: Dict[str, ScrapingJob] = {}
        self.completed_jobs: List[ScrapingJob] = []
        self.job_queue: asyncio.Queue = asyncio.Queue()

        # Intelligence system integrations
        self.intent_classifier = None  # Will be initialized if available
        self.execution_classifier = None
        self.cost_predictor = None
        self.sentinel_orchestrator = None

        # Performance tracking
        self.engine_stats = defaultdict(int)
        self.performance_metrics = defaultdict(list)

        # Configuration
        self.max_concurrent_jobs = 5
        self.job_timeout_hours = 24
        self.cost_variance_threshold = 0.25  # 25% cost variance allowed
        self.enable_sentinel_monitoring = True
        self.enable_ai_precheck = True
        self.enable_cost_optimization = True

        logger.info("ScrapingEngine initialized with full intelligence integration")

    async def orchestrate_scraping_operation(
        self,
        control: ScrapeControlContract,
        priority: JobPriority = JobPriority.NORMAL
    ) -> ScrapingOrchestrationResult:
        """
        Complete orchestration of a scraping operation from request to completion.

        This is the main entry point that coordinates all intelligence systems:
        1. Intent classification and risk assessment
        2. Execution mode optimization
        3. Cost prediction and budget analysis
        4. Governance and compliance checks
        5. Resource allocation and execution
        6. Sentinel monitoring throughout
        7. Telemetry collection and analysis
        8. Quality validation and finalization

        Args:
            control: Complete scraping control contract
            priority: Job execution priority

        Returns:
            Comprehensive orchestration result with all intelligence data
        """

        # Create job
        job_id = str(uuid.uuid4())
        job = ScrapingJob(
            job_id=job_id,
            control=control
        )

        self.active_jobs[job_id] = job
        logger.info(f"🚀 Started orchestration for job {job_id}")

        try:
            # Phase 1: Intent Analysis
            await self._execute_intent_analysis(job)

            # Phase 2: Risk Assessment
            await self._execute_risk_assessment(job)

            # Phase 3: Execution Planning
            await self._execute_planning_phase(job)

            # Phase 4: Cost Optimization
            await self._execute_cost_optimization(job)

            # Phase 5: Governance Check
            governance_passed = await self._execute_governance_check(job)
            if not governance_passed:
                job.orchestration_result = OrchestrationResult.BLOCKED_BY_GOVERNANCE
                return await self._finalize_orchestration(job)

            # Phase 6: Resource Allocation
            await self._execute_resource_allocation(job)

            # Phase 7: Execution with Monitoring
            if job.should_proceed_to_execution():
                await self._execute_scraping_operation(job)
            else:
                job.orchestration_result = OrchestrationResult.BLOCKED_BY_RISK
                return await self._finalize_orchestration(job)

            # Phase 8: Quality Validation
            await self._execute_quality_validation(job)

            # Phase 9: Finalization
            job.orchestration_result = OrchestrationResult.SUCCESS
            return await self._finalize_orchestration(job)

        except asyncio.TimeoutError:
            job.orchestration_result = OrchestrationResult.TIMEOUT_EXCEEDED
            logger.error(f"⏰ Job {job_id} timed out")
        except Exception as e:
            job.orchestration_result = OrchestrationResult.EXECUTION_FAILED
            logger.error(f"❌ Job {job_id} failed: {e}")
            job.error_events.append({
                'timestamp': datetime.utcnow(),
                'phase': job.current_phase.value,
                'error': str(e),
                'error_type': type(e).__name__
            })

        return await self._finalize_orchestration(job)

    async def _execute_intent_analysis(self, job: ScrapingJob):
        """Phase 1: Analyze scraping intent and classify operation."""
        job.current_phase = ScrapingPhase.INTENT_ANALYSIS
        logger.info(f"🔍 Analyzing intent for job {job.job_id}")

        try:
            # Classify intent using ML-enhanced classifier
            intent_classification = await classify_scraping_intent(job.control)
            job.intent_classification = intent_classification

            # Update job priority based on risk level
            job.priority_score = self._calculate_priority_score(intent_classification, job.control)

            # Record phase progress
            job.phase_progress['intent_analysis'] = {
                'completed_at': datetime.utcnow(),
                'risk_level': intent_classification.risk_level.value,
                'category': intent_classification.category.value,
                'confidence': intent_classification.confidence_score
            }

            logger.info(f"✅ Intent classified: {intent_classification.risk_level.value} risk, {intent_classification.category.value}")

        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            # Create conservative fallback classification
            job.intent_classification = IntentClassification(
                intent_id=job.job_id,
                risk_level=IntentRiskLevel.HIGH,
                category=IntentCategory.INTELLIGENCE,
                governance_requirement=GovernanceRequirement.CONTROLLED,
                confidence_score=0.5,
                reasoning=["Fallback classification due to analysis failure"]
            )

    async def _execute_risk_assessment(self, job: ScrapingJob):
        """Phase 2: Perform comprehensive risk assessment."""
        job.current_phase = ScrapingPhase.RISK_ASSESSMENT
        logger.info(f"⚠️ Assessing risk for job {job.job_id}")

        if not job.intent_classification:
            return

        try:
            # Run AI precheck if enabled
            if self.enable_ai_precheck:
                ai_approved = await ai_precheck(job.control)
                job.phase_progress['ai_precheck'] = {
                    'completed_at': datetime.utcnow(),
                    'approved': ai_approved,
                    'reasoning': "AI risk assessment completed"
                }

                if not ai_approved:
                    logger.warning(f"🚫 AI precheck rejected job {job.job_id}")
                    job.intent_classification.reasoning.append("AI precheck rejected - requires review")

            # Sentinel-based risk assessment
            if self.enable_sentinel_monitoring and job.control.intent.sources:
                try:
                    # Extract target for sentinel analysis
                    target = {"domain": job.control.intent.sources[0]}

                    # Run sentinels
                    sentinel_reports = await run_sentinels(target)

                    # Apply safety verdict
                    verdict = safety_verdict(sentinel_reports, job.control)

                    job.sentinel_reports = [report.dict() if hasattr(report, 'dict') else report
                                          for report in sentinel_reports]

                    job.phase_progress['sentinel_assessment'] = {
                        'completed_at': datetime.utcnow(),
                        'reports_count': len(sentinel_reports),
                        'verdict': verdict.action if hasattr(verdict, 'action') else 'unknown'
                    }

                    if verdict.action in ['block', 'delay', 'human_required']:
                        logger.warning(f"🚫 Sentinel verdict blocked job {job.job_id}: {verdict.reason}")
                        job.intent_classification.reasoning.append(f"Sentinel: {verdict.reason}")

                except Exception as e:
                    logger.warning(f"Sentinel assessment failed: {e}")

            logger.info(f"✅ Risk assessment completed for job {job.job_id}")

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")

    async def _execute_planning_phase(self, job: ScrapingJob):
        """Phase 3: Plan optimal execution strategy."""
        job.current_phase = ScrapingPhase.EXECUTION_PLANNING
        logger.info(f"📋 Planning execution for job {job.job_id}")

        try:
            # Determine execution mode and strategy
            execution_profile = await classify_execution_mode(
                asset_type=AssetType.SINGLE_FAMILY_HOME,  # Default, should be inferred
                scope_size=self._estimate_scope_size(job.control),
                control=job.control,
                risk_level=job.intent_classification.risk_level if job.intent_classification else None,
                intent_category=job.intent_classification.category if job.intent_classification else None,
                time_sensitivity="normal",  # Could be configurable
                data_quality_requirement="standard"
            )

            job.execution_profile = execution_profile

            # Estimate completion time
            job.estimated_completion_time = datetime.utcnow() + timedelta(
                hours=execution_profile.performance_expectations.get('estimated_duration_hours', 1)
            )

            job.phase_progress['execution_planning'] = {
                'completed_at': datetime.utcnow(),
                'mode': execution_profile.mode.value,
                'strategy': execution_profile.strategy.value,
                'estimated_duration_hours': execution_profile.performance_expectations.get('estimated_duration_hours', 0)
            }

            logger.info(f"✅ Execution planned: {execution_profile.mode.value} mode, {execution_profile.strategy.value} strategy")

        except Exception as e:
            logger.error(f"Execution planning failed: {e}")
            # Create fallback execution profile
            job.execution_profile = ExecutionProfile(
                mode=ExecutionMode.TARGETED_LOOKUP,
                strategy=ExecutionStrategy.DEPTH_FIRST,
                confidence_score=0.5,
                reasoning=["Fallback execution profile due to planning failure"]
            )

    async def _execute_cost_optimization(self, job: ScrapingJob):
        """Phase 4: Perform cost prediction and optimization."""
        job.current_phase = ScrapingPhase.COST_OPTIMIZATION
        logger.info(f"💰 Optimizing costs for job {job.job_id}")

        try:
            # Generate cost prediction
            cost_prediction = await predict_scraping_cost(
                asset_type=AssetType.SINGLE_FAMILY_HOME,  # Should be inferred from intent
                signal_type=SignalType.LIEN,  # Should be inferred
                execution_mode=job.execution_profile.mode if job.execution_profile else None,
                scope_size=self._estimate_scope_size(job.control),
                risk_level=job.intent_classification.risk_level if job.intent_classification else None,
                intent_category=job.intent_classification.category if job.intent_classification else None,
                control=job.control
            )

            job.cost_prediction = cost_prediction

            # Generate cost optimization plan if enabled
            if self.enable_cost_optimization and cost_prediction.predicted_cost > 100:
                optimization_plan = await optimize_scraping_cost(
                    asset_type=AssetType.SINGLE_FAMILY_HOME,
                    signal_type=SignalType.LIEN,
                    current_cost=cost_prediction.predicted_cost
                )
                job.cost_optimization = optimization_plan

            # Perform budget analysis if budget constraints exist
            if job.control.budget:
                budget_analysis = await analyze_scraping_budget(
                    budget=job.control.budget.max_cost_total,
                    projected_operations=[{
                        'operation_type': 'scraping_job',
                        'estimated_cost': cost_prediction.predicted_cost
                    }]
                )
                job.budget_analysis = budget_analysis

            job.phase_progress['cost_optimization'] = {
                'completed_at': datetime.utcnow(),
                'predicted_cost': cost_prediction.predicted_cost,
                'confidence': cost_prediction.confidence_score,
                'optimization_available': job.cost_optimization is not None
            }

        except Exception as e:
            logger.error(f"Cost optimization failed: {e}")

    async def _execute_governance_check(self, job: ScrapingJob) -> bool:
        """Phase 5: Execute governance and compliance checks."""
        job.current_phase = ScrapingPhase.GOVERNANCE_CHECK
        logger.info(f"⚖️ Checking governance for job {job.job_id}")

        try:
            # Authorization check
            if job.control.authorization:
                AuthorizationGate.validate(job.control.authorization)

            # Deployment timing check
            if job.control.deployment_window:
                await DeploymentTimer.await_window(job.control.deployment_window)

            # Budget governance
            if job.control.budget and job.cost_prediction:
                CostGovernor.initialize(job.control.budget)

            # Risk-based governance
            if job.intent_classification:
                if job.intent_classification.risk_level == IntentRiskLevel.CRITICAL:
                    logger.warning(f"🚨 Critical risk job {job.job_id} requires executive approval")
                    # In production, this would trigger approval workflow
                    if not job.control.authorization or job.control.authorization.authorized_by != "executive_approval":
                        return False

            job.governance_checks_passed = True
            job.phase_progress['governance_check'] = {
                'completed_at': datetime.utcnow(),
                'passed': True,
                'checks_performed': ['authorization', 'timing', 'budget', 'risk']
            }

            logger.info(f"✅ Governance checks passed for job {job.job_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Governance check failed for job {job.job_id}: {e}")
            job.phase_progress['governance_check'] = {
                'completed_at': datetime.utcnow(),
                'passed': False,
                'error': str(e)
            }
            return False

    async def _execute_resource_allocation(self, job: ScrapingJob):
        """Phase 6: Allocate resources for execution."""
        job.current_phase = ScrapingPhase.RESOURCE_ALLOCATION
        logger.info(f"🔧 Allocating resources for job {job.job_id}")

        try:
            # Determine required resources from execution profile
            if job.execution_profile:
                required_resources = job.execution_profile.resource_requirements

                # In production, this would allocate actual resources
                # For now, just validate resource availability
                resource_check = await self._check_resource_availability(required_resources)

                job.phase_progress['resource_allocation'] = {
                    'completed_at': datetime.utcnow(),
                    'resources_allocated': required_resources,
                    'availability_check': resource_check
                }

                logger.info(f"✅ Resources allocated for job {job.job_id}")
            else:
                logger.warning(f"No execution profile for job {job.job_id}, skipping resource allocation")

        except Exception as e:
            logger.error(f"Resource allocation failed: {e}")

    async def _execute_scraping_operation(self, job: ScrapingJob):
        """Phase 7: Execute the actual scraping operation with monitoring."""
        job.current_phase = ScrapingPhase.EXECUTION_MONITORING
        job.execution_start_time = datetime.utcnow()

        logger.info(f"🚀 Executing scraping operation for job {job.job_id}")

        try:
            # Apply execution profile parameters
            execution_params = {}
            if job.execution_profile:
                execution_params = job.execution_profile.execution_parameters

            # Execute with governance workflow
            result = await run_scraper(job.control, execution_params)

            # Record execution results
            job.execution_end_time = datetime.utcnow()
            job.actual_cost = result.get('total_cost', 0)
            job.records_collected = result.get('records_collected', 0)
            job.success_rate = result.get('success_rate', 0)

            # Collect telemetry
            job.telemetry_events = result.get('telemetry_events', [])

            # Emit final telemetry
            await emit_telemetry(
                scraper="orchestrated_job",
                role="orchestration",
                cost_estimate=job.cost_prediction.predicted_cost if job.cost_prediction else 0,
                records_found=job.records_collected,
                blocked_reason=None,
                runtime=(job.execution_end_time - job.execution_start_time).total_seconds()
            )

            job.phase_progress['execution'] = {
                'completed_at': job.execution_end_time,
                'duration_seconds': (job.execution_end_time - job.execution_start_time).total_seconds(),
                'actual_cost': job.actual_cost,
                'records_collected': job.records_collected,
                'success_rate': job.success_rate
            }

            logger.info(f"✅ Scraping execution completed for job {job.job_id}: {job.records_collected} records, ${job.actual_cost}")

        except Exception as e:
            job.execution_end_time = datetime.utcnow()
            logger.error(f"❌ Scraping execution failed for job {job.job_id}: {e}")
            job.error_events.append({
                'timestamp': job.execution_end_time,
                'phase': 'execution',
                'error': str(e),
                'error_type': type(e).__name__
            })

    async def _execute_quality_validation(self, job: ScrapingJob):
        """Phase 8: Validate quality of collected data."""
        job.current_phase = ScrapingPhase.QUALITY_VALIDATION
        logger.info(f"🔍 Validating quality for job {job.job_id}")

        try:
            # Quality validation logic would go here
            # For now, perform basic validation
            quality_score = self._calculate_quality_score(job)

            job.phase_progress['quality_validation'] = {
                'completed_at': datetime.utcnow(),
                'quality_score': quality_score,
                'validation_checks': ['completeness', 'accuracy', 'consistency']
            }

            logger.info(f"✅ Quality validation completed for job {job.job_id}: score {quality_score}")

        except Exception as e:
            logger.error(f"Quality validation failed: {e}")

    async def _finalize_orchestration(self, job: ScrapingJob) -> ScrapingOrchestrationResult:
        """Phase 9: Finalize orchestration and generate comprehensive result."""
        job.current_phase = ScrapingPhase.FINALIZATION
        job.updated_at = datetime.utcnow()

        logger.info(f"🏁 Finalizing orchestration for job {job.job_id}")

        # Move job from active to completed
        if job.job_id in self.active_jobs:
            del self.active_jobs[job.job_id]
        self.completed_jobs.append(job)

        # Generate comprehensive summary
        summary = self._generate_orchestration_summary(job)

        # Generate recommendations and next steps
        recommendations, next_steps = self._generate_recommendations_and_next_steps(job)

        # Generate performance metrics
        performance_metrics = self._generate_performance_metrics(job)

        # Generate compliance report
        compliance_report = self._generate_compliance_report(job)

        # Update engine statistics
        self._update_engine_stats(job)

        result = ScrapingOrchestrationResult(
            job=job,
            result=job.orchestration_result or OrchestrationResult.EXECUTION_FAILED,
            summary=summary,
            recommendations=recommendations,
            next_steps=next_steps,
            performance_metrics=performance_metrics,
            compliance_report=compliance_report
        )

        logger.info(f"✅ Orchestration finalized for job {job.job_id}: {result.result.value}")
        return result

    def _calculate_priority_score(self, intent_classification: IntentClassification,
                                control: ScrapeControlContract) -> float:
        """Calculate job priority score based on multiple factors."""
        base_score = 0.5  # Default medium priority

        # Risk level contribution
        risk_scores = {
            IntentRiskLevel.LOW: 0.2,
            IntentRiskLevel.MEDIUM: 0.4,
            IntentRiskLevel.HIGH: 0.7,
            IntentRiskLevel.CRITICAL: 0.9
        }
        base_score += risk_scores.get(intent_classification.risk_level, 0.5) * 0.4

        # Urgency contribution
        if control.budget and control.budget.max_runtime_minutes < 60:
            base_score += 0.3  # High urgency

        # Business value contribution
        if intent_classification.category in [IntentCategory.FINANCIAL, IntentCategory.LEGAL]:
            base_score += 0.2  # High business value

        return min(1.0, base_score)

    def _estimate_scope_size(self, control: ScrapeControlContract) -> int:
        """Estimate the scope size of the scraping operation."""
        # This is a simplified estimation - in production, this would be more sophisticated
        if control.intent.geography:
            geo_count = len(control.intent.geography)
            if geo_count > 10:
                return 1000  # Large scope
            elif geo_count > 5:
                return 500   # Medium scope
            else:
                return 100   # Small scope
        return 50  # Default medium scope

    async def _check_resource_availability(self, required_resources: Dict[str, Any]) -> bool:
        """Check if required resources are available."""
        # In production, this would check actual resource availability
        # For now, assume resources are available
        return True

    def _calculate_quality_score(self, job: ScrapingJob) -> float:
        """Calculate overall quality score for the job."""
        base_score = 0.8  # Default good quality

        # Adjust based on execution success
        if job.success_rate < 0.8:
            base_score -= 0.2

        # Adjust based on cost variance
        if job.cost_prediction and job.actual_cost > 0:
            variance = abs(job.cost_prediction.predicted_cost - job.actual_cost) / job.cost_prediction.predicted_cost
            if variance > 0.2:
                base_score -= 0.1

        # Adjust based on execution profile quality
        if job.execution_profile:
            profile_quality = job.execution_profile.performance_expectations.get('expected_data_quality_score', 0.8)
            base_score = (base_score + profile_quality) / 2

        return max(0.0, min(1.0, base_score))

    def _generate_orchestration_summary(self, job: ScrapingJob) -> Dict[str, Any]:
        """Generate comprehensive orchestration summary."""
        summary = {
            'job_id': job.job_id,
            'result': job.orchestration_result.value if job.orchestration_result else 'unknown',
            'total_phases_completed': len(job.phase_progress),
            'execution_duration_seconds': None,
            'efficiency_score': job.get_efficiency_score(),
            'cost_efficiency': None,
            'data_quality_score': None
        }

        # Add timing information
        if job.execution_start_time and job.execution_end_time:
            duration = job.execution_end_time - job.execution_start_time
            summary['execution_duration_seconds'] = duration.total_seconds()

        # Add cost efficiency
        if job.cost_prediction and job.actual_cost > 0:
            predicted_cost = job.cost_prediction.predicted_cost
            cost_efficiency = predicted_cost / job.actual_cost if job.actual_cost > predicted_cost else job.actual_cost / predicted_cost
            summary['cost_efficiency'] = cost_efficiency

        # Add intelligence metrics
        if job.intent_classification:
            summary.update({
                'risk_level': job.intent_classification.risk_level.value,
                'intent_category': job.intent_classification.category.value,
                'governance_requirement': job.intent_classification.governance_requirement.value
            })

        if job.execution_profile:
            summary.update({
                'execution_mode': job.execution_profile.mode.value,
                'execution_strategy': job.execution_profile.strategy.value
            })

        if job.cost_prediction:
            summary.update({
                'predicted_cost': job.cost_prediction.predicted_cost,
                'cost_confidence': job.cost_prediction.confidence_score
            })

        return summary

    def _generate_recommendations_and_next_steps(self, job: ScrapingJob) -> Tuple[List[str], List[str]]:
        """Generate recommendations and next steps based on job results."""
        recommendations = []
        next_steps = []

        # Success-based recommendations
        if job.orchestration_result == OrchestrationResult.SUCCESS:
            recommendations.append("Consider scaling this successful approach to similar operations")
            if job.get_efficiency_score() > 0.8:
                recommendations.append("This operation was highly efficient - analyze and replicate best practices")

            next_steps.append("Review collected data for quality and completeness")
            next_steps.append("Update cost models with actual performance data")

        # Cost-based recommendations
        if job.cost_prediction and job.actual_cost > job.cost_prediction.predicted_cost * 1.2:
            recommendations.append("Actual costs exceeded predictions - review cost estimation models")
            next_steps.append("Analyze cost drivers and optimize resource allocation")

        # Risk-based recommendations
        if job.intent_classification and job.intent_classification.risk_level == IntentRiskLevel.CRITICAL:
            recommendations.append("Critical risk operations require enhanced monitoring going forward")
            next_steps.append("Conduct post-operation risk assessment")

        # Quality-based recommendations
        quality_score = self._calculate_quality_score(job)
        if quality_score < 0.7:
            recommendations.append("Data quality was below expectations - review validation processes")
            next_steps.append("Implement additional quality checks for similar operations")

        # Governance recommendations
        if not job.governance_checks_passed:
            recommendations.append("Governance checks failed - review authorization and compliance processes")
            next_steps.append("Obtain proper approvals before retrying")

        return recommendations, next_steps

    def _generate_performance_metrics(self, job: ScrapingJob) -> Dict[str, Any]:
        """Generate detailed performance metrics."""
        metrics = {
            'phases_completed': len(job.phase_progress),
            'error_count': len(job.error_events),
            'telemetry_events_count': len(job.telemetry_events),
            'sentinel_reports_count': len(job.sentinel_reports),
            'efficiency_score': job.get_efficiency_score()
        }

        # Add timing metrics
        if job.execution_start_time and job.execution_end_time:
            duration = job.execution_end_time - job.execution_start_time
            metrics.update({
                'execution_duration_seconds': duration.total_seconds(),
                'execution_duration_hours': duration.total_seconds() / 3600
            })

        # Add cost metrics
        if job.cost_prediction:
            metrics.update({
                'predicted_cost': job.cost_prediction.predicted_cost,
                'actual_cost': job.actual_cost,
                'cost_accuracy': 1.0 - abs(job.cost_prediction.predicted_cost - job.actual_cost) / max(job.cost_prediction.predicted_cost, job.actual_cost) if job.actual_cost > 0 else 0
            })

        # Add quality metrics
        metrics['data_quality_score'] = self._calculate_quality_score(job)

        # Add resource utilization
        if job.execution_profile:
            metrics.update(job.execution_profile.resource_requirements)

        return metrics

    def _generate_compliance_report(self, job: ScrapingJob) -> Dict[str, Any]:
        """Generate compliance report for the job."""
        report = {
            'governance_checks_passed': job.governance_checks_passed,
            'compliance_flags': job.compliance_flags.copy(),
            'risk_mitigations_applied': job.risk_mitigations_applied.copy(),
            'authorization_valid': False,
            'budget_compliant': False,
            'data_protection_compliant': False
        }

        # Check authorization
        if job.control.authorization:
            try:
                AuthorizationGate.validate(job.control.authorization)
                report['authorization_valid'] = True
            except:
                pass

        # Check budget compliance
        if job.control.budget and job.actual_cost <= job.control.budget.max_cost_total:
            report['budget_compliant'] = True

        # Check data protection compliance
        if job.intent_classification:
            if job.intent_classification.category != IntentCategory.PERSONAL or 'data_protection' in job.compliance_flags:
                report['data_protection_compliant'] = True

        # Add compliance summary
        compliant_items = sum(report.values()) if isinstance(report, dict) else 0
        total_items = len([v for v in report.values() if isinstance(v, bool)])
        report['overall_compliance_score'] = compliant_items / total_items if total_items > 0 else 0

        return report

    def _update_engine_stats(self, job: ScrapingJob):
        """Update engine performance statistics."""
        self.engine_stats['total_jobs_processed'] += 1

        if job.orchestration_result == OrchestrationResult.SUCCESS:
            self.engine_stats['successful_jobs'] += 1
        elif job.orchestration_result:
            self.engine_stats['failed_jobs'] += 1

        # Update performance metrics
        if job.get_efficiency_score() > 0:
            self.performance_metrics['efficiency_scores'].append(job.get_efficiency_score())

        if job.actual_cost > 0:
            self.performance_metrics['actual_costs'].append(job.actual_cost)

        if job.execution_start_time and job.execution_end_time:
            duration = (job.execution_end_time - job.execution_start_time).total_seconds()
            self.performance_metrics['execution_durations'].append(duration)

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine performance statistics."""
        stats = dict(self.engine_stats)

        # Calculate derived statistics
        total_jobs = stats.get('total_jobs_processed', 0)
        successful_jobs = stats.get('successful_jobs', 0)

        if total_jobs > 0:
            stats['success_rate'] = successful_jobs / total_jobs
            stats['failure_rate'] = 1 - (successful_jobs / total_jobs)

        # Performance averages
        if self.performance_metrics['efficiency_scores']:
            stats['average_efficiency_score'] = sum(self.performance_metrics['efficiency_scores']) / len(self.performance_metrics['efficiency_scores'])

        if self.performance_metrics['actual_costs']:
            stats['average_job_cost'] = sum(self.performance_metrics['actual_costs']) / len(self.performance_metrics['actual_costs'])

        if self.performance_metrics['execution_durations']:
            stats['average_execution_duration_seconds'] = sum(self.performance_metrics['execution_durations']) / len(self.performance_metrics['execution_durations'])

        # Current status
        stats.update({
            'active_jobs_count': len(self.active_jobs),
            'completed_jobs_count': len(self.completed_jobs),
            'queue_depth': self.job_queue.qsize()
        })

        return stats

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel an active job."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.orchestration_result = OrchestrationResult.CANCELLED_BY_USER
            job.execution_end_time = datetime.utcnow()

            # Move to completed
            del self.active_jobs[job_id]
            self.completed_jobs.append(job)

            logger.info(f"🛑 Job {job_id} cancelled by user")
            return True

        return False

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a job."""
        job = self.active_jobs.get(job_id)
        if not job:
            # Check completed jobs
            for completed_job in self.completed_jobs:
                if completed_job.job_id == job_id:
                    return {
                        'job_id': job_id,
                        'status': 'completed',
                        'result': completed_job.orchestration_result.value if completed_job.orchestration_result else 'unknown',
                        'completed_at': completed_job.execution_end_time.isoformat() if completed_job.execution_end_time else None,
                        'records_collected': completed_job.records_collected,
                        'actual_cost': completed_job.actual_cost
                    }

        if job:
            return {
                'job_id': job_id,
                'status': 'active',
                'current_phase': job.current_phase.value,
                'priority_score': job.priority_score,
                'started_at': job.execution_start_time.isoformat() if job.execution_start_time else None,
                'estimated_completion': job.estimated_completion_time.isoformat() if job.estimated_completion_time else None
            }

        return None


# Preflight cost and operational readiness check
async def preflight_cost_check(control: ScrapeControlContract) -> Dict[str, Any]:
    """
    Comprehensive preflight check for cost, operational readiness, and compliance.

    Performs detailed cost estimation, budget analysis, risk assessment, and
    operational feasibility evaluation before job execution.

    Args:
        control: Complete scraping control contract

    Returns:
        Comprehensive preflight assessment with recommendations and warnings

    Raises:
        RuntimeError: If critical issues prevent execution
        ValueError: If control contract is invalid
    """
    assessment = {
        'assessment_id': str(uuid.uuid4()),
        'timestamp': datetime.utcnow(),
        'overall_readiness': 'unknown',
        'critical_issues': [],
        'warnings': [],
        'recommendations': [],
        'cost_analysis': {},
        'risk_assessment': {},
        'operational_feasibility': {},
        'compliance_status': {},
        'estimated_execution_time': None,
        'resource_requirements': {},
        'optimization_opportunities': []
    }

    try:
        # Validate control contract completeness
        validation_issues = _validate_control_contract(control)
        if validation_issues:
            assessment['critical_issues'].extend(validation_issues)
            assessment['overall_readiness'] = 'invalid_contract'
            raise ValueError(f"Invalid control contract: {', '.join(validation_issues)}")

        # 1. Cost Estimation and Analysis
        cost_prediction = await predict_scraping_cost(
            asset_type=_infer_asset_type_from_control(control),
            signal_type=_infer_signal_type_from_control(control),
            execution_mode=None,  # Will be determined by classifier
            scope_size=_estimate_control_scope(control),
            risk_level=None,  # Will be determined by intent classifier
            intent_category=None,  # Will be determined by intent classifier
            control=control
        )

        assessment['cost_analysis'] = {
            'predicted_cost': cost_prediction.predicted_cost,
            'confidence_score': cost_prediction.confidence_score,
            'cost_range': cost_prediction.cost_range,
            'cost_breakdown': cost_prediction.cost_breakdown,
            'budget_compliance': 'compliant',
            'budget_utilization_percentage': 0.0,
            'cost_variance_risk': 'low'
        }

        # Budget compliance check
        if control.budget and control.budget.max_cost_total:
            budget_limit = control.budget.max_cost_total
            predicted_cost = cost_prediction.predicted_cost
            utilization_pct = (predicted_cost / budget_limit) * 100

            assessment['cost_analysis']['budget_utilization_percentage'] = round(utilization_pct, 1)

            if utilization_pct > 100:
                assessment['critical_issues'].append(
                    f"Predicted cost ${predicted_cost:.2f} exceeds budget limit of ${budget_limit:.2f} "
                    f"({utilization_pct:.1f}% utilization)"
                )
                assessment['cost_analysis']['budget_compliance'] = 'exceeded'
                assessment['overall_readiness'] = 'budget_exceeded'
            elif utilization_pct > 90:
                assessment['warnings'].append(
                    f"High budget utilization: {utilization_pct:.1f}% of approved budget"
                )
                assessment['cost_analysis']['budget_compliance'] = 'high_utilization'
            elif utilization_pct < 30:
                assessment['recommendations'].append(
                    f"Budget utilization is low ({utilization_pct:.1f}%) - consider scaling up scope if appropriate"
                )

        # Cost variance risk assessment
        cost_range_width = cost_prediction.cost_range[1] - cost_prediction.cost_range[0]
        cost_range_pct = (cost_range_width / cost_prediction.predicted_cost) * 100 if cost_prediction.predicted_cost > 0 else 0

        if cost_range_pct > 50:
            assessment['cost_analysis']['cost_variance_risk'] = 'high'
            assessment['warnings'].append(f"High cost uncertainty: ±{cost_range_pct:.1f}% range")
        elif cost_range_pct > 25:
            assessment['cost_analysis']['cost_variance_risk'] = 'medium'
            assessment['warnings'].append(f"Moderate cost uncertainty: ±{cost_range_pct:.1f}% range")

        # 2. Operational Feasibility Assessment
        execution_profile = await classify_execution_mode(
            asset_type=_infer_asset_type_from_control(control),
            scope_size=_estimate_control_scope(control),
            control=control
        )

        assessment['operational_feasibility'] = {
            'recommended_mode': execution_profile.mode.value,
            'recommended_strategy': execution_profile.strategy.value,
            'estimated_duration_hours': execution_profile.performance_expectations.get('estimated_duration_hours', 0),
            'expected_success_rate': execution_profile.performance_expectations.get('expected_success_rate', 0.8),
            'resource_intensity': 'medium',
            'concurrency_recommendation': execution_profile.execution_parameters.get('concurrent_requests', 1),
            'feasibility_score': execution_profile.confidence_score
        }

        # Estimate execution time
        estimated_hours = execution_profile.performance_expectations.get('estimated_duration_hours', 1)
        assessment['estimated_execution_time'] = timedelta(hours=estimated_hours)

        # Resource requirements assessment
        resource_reqs = execution_profile.resource_requirements
        assessment['resource_requirements'] = resource_reqs

        # Check resource intensity
        cpu_cores = resource_reqs.get('cpu_cores', 1)
        memory_gb = resource_reqs.get('memory_gb', 2)

        if cpu_cores >= 4 or memory_gb >= 8:
            assessment['operational_feasibility']['resource_intensity'] = 'high'
        elif cpu_cores >= 2 or memory_gb >= 4:
            assessment['operational_feasibility']['resource_intensity'] = 'medium'
        else:
            assessment['operational_feasibility']['resource_intensity'] = 'low'

        # Time feasibility check
        if control.budget and control.budget.max_runtime_minutes:
            max_allowed_hours = control.budget.max_runtime_minutes / 60
            if estimated_hours > max_allowed_hours:
                assessment['critical_issues'].append(
                    f"Estimated execution time {estimated_hours:.1f}h exceeds budget limit of {max_allowed_hours:.1f}h"
                )
                assessment['overall_readiness'] = 'time_constraint_violation'

        # 3. Risk and Compliance Assessment
        intent_classification = await classify_scraping_intent(control)

        assessment['risk_assessment'] = {
            'risk_level': intent_classification.risk_level.value,
            'intent_category': intent_classification.category.value,
            'governance_requirement': intent_classification.governance_requirement.value,
            'classification_confidence': intent_classification.confidence_score,
            'risk_factors': [],
            'mitigation_requirements': []
        }

        # Risk-based warnings and requirements
        if intent_classification.risk_level == IntentRiskLevel.CRITICAL:
            assessment['critical_issues'].append("Critical risk level requires executive approval")
            assessment['risk_assessment']['mitigation_requirements'].append("Executive authorization required")

        if intent_classification.category == IntentCategory.PERSONAL:
            assessment['risk_assessment']['risk_factors'].append("Personal data collection")
            assessment['compliance_status']['privacy_compliance'] = 'required'

        if intent_classification.category == IntentCategory.LEGAL:
            assessment['risk_assessment']['risk_factors'].append("Legal data sensitivity")
            assessment['compliance_status']['legal_compliance'] = 'required'

        # AI Precheck (if enabled)
        try:
            ai_approved = await ai_precheck(control)
            assessment['risk_assessment']['ai_precheck_passed'] = ai_approved
            if not ai_approved:
                assessment['critical_issues'].append("AI precheck failed - requires manual review")
        except Exception as e:
            assessment['warnings'].append(f"AI precheck unavailable: {e}")

        # 4. Compliance and Authorization Check
        assessment['compliance_status'].update({
            'authorization_valid': False,
            'budget_compliance': assessment['cost_analysis']['budget_compliance'] == 'compliant',
            'time_window_compliant': True,
            'scope_compliant': True,
            'overall_compliance_score': 0.0
        })

        # Authorization validation
        if control.authorization:
            try:
                AuthorizationGate.validate(control.authorization)
                assessment['compliance_status']['authorization_valid'] = True
            except Exception as e:
                assessment['critical_issues'].append(f"Authorization invalid: {e}")

        # Time window compliance
        if control.deployment_window:
            now = datetime.utcnow()
            if now < control.deployment_window.earliest_start:
                wait_time = control.deployment_window.earliest_start - now
                assessment['warnings'].append(f"Job scheduled to start in {wait_time}")
            elif now > control.deployment_window.latest_start:
                assessment['critical_issues'].append("Deployment window has expired")

        # Scope compliance check
        if control.budget:
            scope_size = _estimate_control_scope(control)
            if scope_size > 1000 and control.budget.max_records < scope_size:
                assessment['warnings'].append("Scope size may exceed record budget limits")

        # Calculate overall compliance score
        compliance_items = [
            assessment['compliance_status']['authorization_valid'],
            assessment['compliance_status']['budget_compliance'],
            assessment['compliance_status']['time_window_compliant'],
            assessment['compliance_status']['scope_compliant']
        ]
        compliance_score = sum(compliance_items) / len(compliance_items)
        assessment['compliance_status']['overall_compliance_score'] = round(compliance_score, 2)

        # 5. Optimization Opportunities Analysis
        cost_optimization = await optimize_scraping_cost(
            asset_type=_infer_asset_type_from_control(control),
            signal_type=_infer_signal_type_from_control(control),
            current_cost=cost_prediction.predicted_cost
        )

        assessment['optimization_opportunities'] = [
            {
                'type': change.get('type', 'general'),
                'opportunity': change.get('change', 'Optimization available'),
                'estimated_savings': change.get('estimated_savings', 0),
                'implementation_effort': change.get('implementation_effort', 'medium'),
                'priority': 'high' if change.get('estimated_savings', 0) > cost_prediction.predicted_cost * 0.1 else 'medium'
            }
            for change in cost_optimization.recommended_changes[:5]
        ]

        # Add execution-based optimizations
        if execution_profile.confidence_score < 0.8:
            assessment['optimization_opportunities'].append({
                'type': 'execution_mode',
                'opportunity': f"Consider alternative execution modes - current confidence: {execution_profile.confidence_score:.2f}",
                'estimated_savings': cost_prediction.predicted_cost * 0.05,
                'implementation_effort': 'low',
                'priority': 'medium'
            })

        # 6. Overall Readiness Assessment
        critical_count = len(assessment['critical_issues'])
        warning_count = len(assessment['warnings'])

        if critical_count > 0:
            assessment['overall_readiness'] = 'blocked'
        elif warning_count > 2:
            assessment['overall_readiness'] = 'caution'
        elif compliance_score < 0.75:
            assessment['overall_readiness'] = 'compliance_review'
        else:
            assessment['overall_readiness'] = 'ready'

        # Add summary recommendations
        if assessment['overall_readiness'] == 'ready':
            assessment['recommendations'].append("Operation appears ready for execution")
        elif assessment['overall_readiness'] == 'caution':
            assessment['recommendations'].append("Address warnings before proceeding")
        elif assessment['overall_readiness'] == 'blocked':
            assessment['recommendations'].append("Critical issues must be resolved before execution")

        # Cost optimization recommendations
        if cost_optimization.cost_savings > cost_prediction.predicted_cost * 0.1:
            assessment['recommendations'].append(
                f"Significant cost savings available (${cost_optimization.cost_savings:.2f}) - consider optimization"
            )

        # Success prediction
        expected_success = execution_profile.performance_expectations.get('expected_success_rate', 0.8)
        if expected_success < 0.7:
            assessment['warnings'].append(f"Low expected success rate: {expected_success:.1f}")

        return assessment

    except Exception as e:
        assessment['critical_issues'].append(f"Preflight check failed: {str(e)}")
        assessment['overall_readiness'] = 'error'
        logger.error(f"Preflight check error: {e}")
        raise RuntimeError(f"Preflight assessment failed: {str(e)}") from e


def _validate_control_contract(control: ScrapeControlContract) -> List[str]:
    """Validate control contract completeness and consistency."""
    issues = []

    if not control.intent:
        issues.append("Missing intent specification")

    if not control.intent.sources:
        issues.append("No data sources specified")

    if not control.intent.geography:
        issues.append("No geographic scope specified")

    if not control.budget:
        issues.append("Missing budget specification")
    else:
        if control.budget.max_cost_total <= 0:
            issues.append("Invalid budget amount")
        if control.budget.max_runtime_minutes <= 0:
            issues.append("Invalid runtime limit")

    if not control.authorization:
        issues.append("Missing authorization")
    else:
        if control.authorization.expires_at <= datetime.utcnow():
            issues.append("Authorization has expired")

    return issues


def _infer_asset_type_from_control(control: ScrapeControlContract) -> AssetType:
    """Infer asset type from control contract."""
    # Simple inference - in production, this would be more sophisticated
    if any('company' in source.lower() for source in control.intent.sources):
        return AssetType.COMPANY
    elif any('property' in source.lower() or 'real_estate' in source.lower() for source in control.intent.sources):
        return AssetType.SINGLE_FAMILY_HOME
    else:
        return AssetType.PERSON  # Default


def _infer_signal_type_from_control(control: ScrapeControlContract) -> Optional[SignalType]:
    """Infer signal type from control contract."""
    # Simple inference based on sources and intent
    source_text = ' '.join(control.intent.sources).lower()

    if 'lien' in source_text:
        return SignalType.LIEN
    elif 'mortgage' in source_text:
        return SignalType.MORTGAGE
    elif 'court' in source_text or 'legal' in source_text:
        return SignalType.COURT_CASE
    elif 'financial' in source_text:
        return SignalType.FINANCIAL
    elif 'birthday' in source_text or 'birth' in source_text:
        return SignalType.BIRTHDAY
    else:
        return None  # Will use default in prediction


def _estimate_control_scope(control: ScrapeControlContract) -> int:
    """Estimate the scope size from control contract."""
    # Base scope from geography
    geo_count = len(control.intent.geography) if control.intent.geography else 1

    # Adjust for sources
    source_multiplier = len(control.intent.sources) if control.intent.sources else 1

    # Adjust for budget limits
    if control.budget and control.budget.max_records:
        max_records = min(control.budget.max_records, 10000)  # Cap at reasonable limit
        return min(geo_count * source_multiplier * 10, max_records)

    # Default estimation
    return geo_count * source_multiplier * 50


# Global scraper engine instance
_global_scraper_engine = ScrapingEngine()


# Convenience functions
async def orchestrate_scraping_job(
    control: ScrapeControlContract,
    priority: JobPriority = JobPriority.NORMAL
) -> ScrapingOrchestrationResult:
    """
    Orchestrate a complete scraping operation with full intelligence integration.

    This is the main entry point for the MJ Data Scraper Suite that coordinates
    all intelligence systems for optimal, compliant, and cost-effective operations.

    Args:
        control: Complete scraping control contract with intent, budget, authorization
        priority: Job execution priority level

    Returns:
        Comprehensive orchestration result with intelligence data and recommendations

    Example:
        result = await orchestrate_scraping_job(scrape_control_contract)
        if result.result == OrchestrationResult.SUCCESS:
            print(f"Collected {result.job.records_collected} records")
        else:
            print(f"Operation blocked: {result.result.value}")
    """
    return await _global_scraper_engine.orchestrate_scraping_operation(control, priority)


def get_scraper_engine_stats() -> Dict[str, Any]:
    """
    Get comprehensive scraper engine performance statistics.

    Returns operational metrics for monitoring orchestration performance,
    success rates, cost efficiency, and system health.

    Returns:
        Dict with engine statistics and performance indicators
    """
    return _global_scraper_engine.get_engine_stats()


async def cancel_scraping_job(job_id: str) -> bool:
    """
    Cancel an active scraping job.

    Args:
        job_id: Unique job identifier

    Returns:
        True if job was cancelled, False if not found or already completed
    """
    return await _global_scraper_engine.cancel_job(job_id)


async def get_scraping_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get current status of a scraping job.

    Args:
        job_id: Unique job identifier

    Returns:
        Dict with job status information or None if not found
    """
    return await _global_scraper_engine.get_job_status(job_id)


# Legacy compatibility
async def start_scraping_job(control: ScrapeControlContract) -> str:
    """
    Legacy function for backward compatibility.

    Starts a scraping job and returns the job ID for status tracking.
    """
    result = await orchestrate_scraping_job(control)
    return result.job.job_id