# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use as is strictly prohibited.

"""
What-If Analysis Engine for MJ Data Scraper Suite

Advanced scenario planning and comparative analysis system that enables
exploration of different operational strategies, cost implications, and
risk assessments before execution commitment.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy

from ..models.asset_signal import AssetType, SignalType
from .intent_classifier import IntentRiskLevel, IntentCategory
from .execution_mode_classifier import ExecutionMode, ExecutionStrategy
from .cost_predictor import (
    predict_scraping_cost,
    CostPrediction,
    optimize_scraping_cost,
    CostOptimizationPlan,
    analyze_scraping_budget,
    BudgetAnalysis
)
from .mapping.asset_signal_map import (
    get_optimal_sources_for_signal,
    calculate_signal_cost_estimate,
    validate_source_for_signal
)
from .control_models import ScrapeControlContract, ScrapeIntent, ScrapeBudget, ScrapeAuthorization
from .ai_precheck import ai_precheck
from ..sentinels.sentinel_orchestrator import run_sentinels
from .safety_verdict import safety_verdict
from .scraper_engine import preflight_cost_check

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of what-if scenarios."""
    BASELINE = "baseline"                    # Original configuration
    COST_OPTIMIZATION = "cost_optimization"  # Cost reduction focus
    QUALITY_ENHANCEMENT = "quality_enhancement"  # Quality improvement focus
    SPEED_OPTIMIZATION = "speed_optimization"    # Time reduction focus
    RISK_MINIMIZATION = "risk_minimization"  # Risk reduction focus
    SCALE_EXPANSION = "scale_expansion"      # Scope increase
    SCOPE_REDUCTION = "scope_reduction"      # Scope decrease
    SOURCE_ALTERNATIVE = "source_alternative"  # Different data sources
    GEOGRAPHY_MODIFICATION = "geography_modification"  # Geographic changes
    BUDGET_CONSTRAINT = "budget_constraint"  # Budget limitation scenario
    COMPLIANCE_ENHANCED = "compliance_enhanced"  # Enhanced compliance
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"  # Parameter sensitivity
    CUSTOM = "custom"                        # User-defined modifications


class SensitivityParameter(Enum):
    """Parameters for sensitivity analysis."""
    COST_INCREASE = "cost_increase"
    TIME_INCREASE = "time_increase"
    SUCCESS_RATE_DECREASE = "success_rate_decrease"
    RESOURCE_COST_INCREASE = "resource_cost_increase"
    COMPLIANCE_COST_INCREASE = "compliance_cost_increase"
    SCOPE_EXPANSION = "scope_expansion"
    RISK_LEVEL_INCREASE = "risk_level_increase"


@dataclass
class ScenarioConfiguration:
    """Configuration for a what-if scenario."""
    scenario_id: str
    scenario_name: str
    scenario_type: ScenarioType
    description: str

    # Base configuration
    base_control: ScrapeControlContract

    # Modifications
    modified_sources: Optional[List[str]] = None
    modified_geography: Optional[List[str]] = None
    modified_budget: Optional[ScrapeBudget] = None
    modified_tempo: Optional[str] = None
    modified_time_window: Optional[Tuple[datetime, datetime]] = None
    modified_signals: Optional[List[SignalType]] = None

    # Scenario parameters
    cost_sensitivity_factor: float = 1.0  # Cost multiplier
    time_sensitivity_factor: float = 1.0  # Time multiplier
    risk_sensitivity_level: Optional[IntentRiskLevel] = None
    quality_requirement: str = "standard"  # basic/standard/premium

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"

    def get_modified_control(self) -> ScrapeControlContract:
        """Generate modified control contract for this scenario."""
        # Deep copy base control
        modified_control = deepcopy(self.base_control)

        # Apply source modifications
        if self.modified_sources is not None:
            modified_control.intent.sources = self.modified_sources

        # Apply geography modifications
        if self.modified_geography is not None:
            modified_control.intent.geography = self.modified_geography

        # Apply budget modifications
        if self.modified_budget is not None:
            modified_control.budget = self.modified_budget

        # Apply signal modifications
        if self.modified_signals is not None:
            # This would modify the intent to focus on specific signals
            # Implementation depends on how signals are specified in control
            pass

        return modified_control


@dataclass
class ScenarioAnalysis:
    """Complete analysis of a what-if scenario."""
    scenario_id: str
    scenario_name: str
    scenario_type: ScenarioType

    # Intelligence assessments
    preflight_assessment: Dict[str, Any]
    cost_prediction: Optional[CostPrediction] = None
    cost_optimization: Optional[CostOptimizationPlan] = None
    budget_analysis: Optional[BudgetAnalysis] = None

    # Comparative metrics (vs baseline)
    cost_difference: float = 0.0
    cost_difference_percentage: float = 0.0
    time_difference_hours: float = 0.0
    risk_level_change: str = "unchanged"  # improved/worsened/unchanged
    compliance_change: str = "unchanged"  # improved/worsened/unchanged

    # Scenario evaluation
    feasibility_score: float = 0.0  # 0-1 feasibility rating
    recommendation_score: float = 0.0  # 0-1 recommendation rating
    trade_off_analysis: Dict[str, Any] = field(default_factory=dict)

    # Performance projections
    projected_success_rate: float = 0.0
    projected_data_quality: str = "unknown"
    projected_resource_utilization: Dict[str, float] = field(default_factory=dict)

    # Risk assessment
    risk_factors_introduced: List[str] = field(default_factory=list)
    risk_factors_mitigated: List[str] = field(default_factory=list)
    compliance_implications: List[str] = field(default_factory=list)

    # Analysis metadata
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    analysis_duration_seconds: float = 0.0
    intelligence_components_used: List[str] = field(default_factory=list)

    def get_cost_efficiency(self) -> float:
        """Calculate cost efficiency score for this scenario."""
        if not self.cost_prediction or self.cost_prediction.predicted_cost <= 0:
            return 0.0

        # Efficiency based on cost, success rate, and compliance
        base_efficiency = 1.0 / self.cost_prediction.predicted_cost

        success_bonus = self.projected_success_rate * 0.2
        quality_bonus = {'basic': 0.0, 'standard': 0.1, 'premium': 0.2}.get(self.projected_data_quality, 0.0)

        return base_efficiency * (1.0 + success_bonus + quality_bonus)

    def get_overall_score(self) -> float:
        """Calculate overall scenario score (0-1)."""
        # Weighted combination of multiple factors
        weights = {
            'feasibility': 0.25,
            'cost_efficiency': 0.25,
            'success_probability': 0.20,
            'compliance': 0.15,
            'risk': 0.15
        }

        feasibility_score = self.feasibility_score
        cost_score = min(1.0, self.get_cost_efficiency() * 1000)  # Normalize
        success_score = self.projected_success_rate
        compliance_score = 1.0 if self.compliance_change != 'worsened' else 0.5
        risk_score = 1.0 if self.risk_level_change != 'worsened' else 0.5

        return (
            feasibility_score * weights['feasibility'] +
            cost_score * weights['cost_efficiency'] +
            success_score * weights['success_probability'] +
            compliance_score * weights['compliance'] +
            risk_score * weights['risk']
        )


@dataclass
class WhatIfAnalysis:
    """Complete what-if analysis with multiple scenarios."""
    analysis_id: str
    baseline_scenario: ScenarioAnalysis
    alternative_scenarios: List[ScenarioAnalysis] = field(default_factory=list)

    # Comparative analysis
    scenario_comparison: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sensitivity_analysis: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    optimization_recommendations: List[Dict[str, Any]] = field(default_factory=list)

    # Analysis summary
    best_overall_scenario: Optional[str] = None
    best_cost_scenario: Optional[str] = None
    best_quality_scenario: Optional[str] = None
    best_speed_scenario: Optional[str] = None
    risk_adjusted_recommendation: Optional[str] = None

    # Trade-off analysis
    cost_vs_quality_tradeoffs: List[Dict[str, Any]] = field(default_factory=list)
    speed_vs_cost_tradeoffs: List[Dict[str, Any]] = field(default_factory=list)
    risk_vs_benefit_tradeoffs: List[Dict[str, Any]] = field(default_factory=list)

    # Analysis metadata
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    total_analysis_time_seconds: float = 0.0
    scenarios_analyzed: int = 0

    def get_scenario_ranking(self, criteria: str = "overall") -> List[Tuple[str, float]]:
        """Get scenarios ranked by specified criteria."""
        rankings = []

        all_scenarios = [self.baseline_scenario] + self.alternative_scenarios

        for scenario in all_scenarios:
            if criteria == "overall":
                score = scenario.get_overall_score()
            elif criteria == "cost":
                score = -scenario.cost_prediction.predicted_cost if scenario.cost_prediction else 0  # Lower cost = higher score
            elif criteria == "quality":
                quality_scores = {'basic': 0.3, 'standard': 0.6, 'premium': 1.0}
                score = quality_scores.get(scenario.projected_data_quality, 0.5)
            elif criteria == "speed":
                # Lower time = higher score (assuming baseline time as reference)
                baseline_time = self.baseline_scenario.preflight_assessment.get('operational_feasibility', {}).get('estimated_duration_hours', 1)
                scenario_time = scenario.preflight_assessment.get('operational_feasibility', {}).get('estimated_duration_hours', baseline_time)
                score = max(0, baseline_time / scenario_time) if scenario_time > 0 else 0
            elif criteria == "risk":
                risk_scores = {'low': 1.0, 'medium': 0.7, 'high': 0.4, 'critical': 0.1}
                risk_level = scenario.preflight_assessment.get('risk_assessment', {}).get('risk_level', 'medium')
                score = risk_scores.get(risk_level, 0.5)
            else:
                score = scenario.get_overall_score()

            rankings.append((scenario.scenario_name, score))

        return sorted(rankings, key=lambda x: x[1], reverse=True)


class WhatIfAnalyzer:
    """
    Advanced what-if analysis engine for scenario planning and comparative evaluation.

    Enables exploration of different operational strategies, cost implications, risk assessments,
    and optimization opportunities through comprehensive scenario analysis.
    """

    def __init__(self):
        self.analyses: Dict[str, WhatIfAnalysis] = {}
        self.scenario_templates: Dict[str, Dict[str, Any]] = self._load_scenario_templates()

        # Performance tracking
        self.analysis_stats = defaultdict(int)

        logger.info("WhatIfAnalyzer initialized with comprehensive scenario analysis capabilities")

    def _load_scenario_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined scenario templates."""
        return {
            'cost_optimization': {
                'description': 'Optimize for minimum cost while maintaining quality',
                'modifications': {
                    'source_selection': 'lowest_cost',
                    'execution_mode': 'batch_efficient',
                    'quality_requirement': 'standard'
                }
            },
            'quality_enhancement': {
                'description': 'Maximize data quality and completeness',
                'modifications': {
                    'source_selection': 'highest_quality',
                    'execution_mode': 'comprehensive_audit',
                    'quality_requirement': 'premium',
                    'validation_level': 'strict'
                }
            },
            'speed_optimization': {
                'description': 'Minimize execution time with parallel processing',
                'modifications': {
                    'execution_mode': 'parallel_burst',
                    'concurrency': 'maximum',
                    'source_selection': 'fastest_response'
                }
            },
            'risk_minimization': {
                'description': 'Minimize operational and compliance risks',
                'modifications': {
                    'source_selection': 'most_compliant',
                    'execution_mode': 'conservative',
                    'monitoring_level': 'maximum'
                }
            },
            'scale_expansion': {
                'description': 'Expand operational scale for broader coverage',
                'modifications': {
                    'geography_expansion': True,
                    'source_diversification': True,
                    'execution_mode': 'distributed'
                }
            }
        }

    async def create_scenario(
        self,
        base_control: ScrapeControlContract,
        scenario_type: ScenarioType,
        scenario_name: str,
        modifications: Optional[Dict[str, Any]] = None
    ) -> ScenarioConfiguration:
        """
        Create a what-if scenario configuration.

        Args:
            base_control: Base scraping control contract
            scenario_type: Type of scenario to create
            scenario_name: Human-readable scenario name
            modifications: Specific modifications to apply

        Returns:
            Configured scenario ready for analysis
        """
        scenario_id = str(uuid.uuid4())

        # Get template modifications
        template_mods = self.scenario_templates.get(scenario_type.value, {})

        # Merge with user modifications
        all_modifications = {**template_mods, **(modifications or {})}

        # Create scenario configuration
        scenario = ScenarioConfiguration(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            scenario_type=scenario_type,
            description=all_modifications.get('description', f'{scenario_type.value} scenario'),
            base_control=base_control
        )

        # Apply modifications based on scenario type
        await self._apply_scenario_modifications(scenario, all_modifications)

        logger.info(f"Created scenario {scenario_name} ({scenario_type.value})")
        return scenario

    async def _apply_scenario_modifications(
        self,
        scenario: ScenarioConfiguration,
        modifications: Dict[str, Any]
    ):
        """Apply modifications to scenario configuration."""
        # Source modifications
        if 'source_selection' in modifications:
            selection_type = modifications['source_selection']
            if selection_type == 'lowest_cost':
                scenario.modified_sources = await self._select_lowest_cost_sources(scenario.base_control)
            elif selection_type == 'highest_quality':
                scenario.modified_sources = await self._select_highest_quality_sources(scenario.base_control)
            elif selection_type == 'most_compliant':
                scenario.modified_sources = await self._select_most_compliant_sources(scenario.base_control)

        # Geography modifications
        if modifications.get('geography_expansion'):
            scenario.modified_geography = await self._expand_geography(scenario.base_control.intent.geography)

        # Budget modifications
        if 'budget_adjustment' in modifications:
            adjustment_factor = modifications['budget_adjustment']
            scenario.modified_budget = self._adjust_budget(scenario.base_control.budget, adjustment_factor)

        # Quality modifications
        if 'quality_requirement' in modifications:
            scenario.quality_requirement = modifications['quality_requirement']

        # Cost sensitivity
        if 'cost_sensitivity' in modifications:
            scenario.cost_sensitivity_factor = modifications['cost_sensitivity']

        # Time sensitivity
        if 'time_sensitivity' in modifications:
            scenario.time_sensitivity_factor = modifications['time_sensitivity']

    async def _select_lowest_cost_sources(self, control: ScrapeControlContract) -> List[str]:
        """Select lowest cost data sources."""
        # This would integrate with cost predictor to find optimal sources
        # For now, return a simulated selection
        return ['public_records', 'free_databases', 'government_sources']

    async def _select_highest_quality_sources(self, control: ScrapeControlContract) -> List[str]:
        """Select highest quality data sources."""
        return ['premium_databases', 'official_registries', 'verified_sources']

    async def _select_most_compliant_sources(self, control: ScrapeControlContract) -> List[str]:
        """Select most compliant data sources."""
        return ['government_apis', 'licensed_databases', 'official_sources']

    async def _expand_geography(self, base_geography: List[str]) -> List[str]:
        """Expand geography scope."""
        if not base_geography:
            return ['national', 'international']
        # Expand existing geography
        return base_geography + ['adjacent_regions', 'expanded_scope']

    def _adjust_budget(self, base_budget: ScrapeBudget, factor: float) -> ScrapeBudget:
        """Adjust budget by factor."""
        return ScrapeBudget(
            max_runtime_minutes=int(base_budget.max_runtime_minutes * factor),
            max_pages=int(base_budget.max_pages * factor),
            max_records=int(base_budget.max_records * factor),
            max_cost_total=base_budget.max_cost_total * factor
        )

    async def analyze_scenario(self, scenario: ScenarioConfiguration) -> ScenarioAnalysis:
        """
        Analyze a single what-if scenario comprehensively.

        Args:
            scenario: Scenario configuration to analyze

        Returns:
            Complete scenario analysis with intelligence assessments
        """
        start_time = asyncio.get_event_loop().time()

        # Get modified control
        modified_control = scenario.get_modified_control()

        # Run preflight assessment
        preflight_assessment = await preflight_cost_check(modified_control)

        # Generate cost prediction
        cost_prediction = await predict_scraping_cost(
            asset_type=self._infer_asset_type(modified_control),
            signal_type=self._infer_signal_type(modified_control),
            execution_mode=None,
            scope_size=self._estimate_scenario_scope(modified_control),
            risk_level=None,
            intent_category=None,
            control=modified_control
        )

        # Generate cost optimization
        cost_optimization = await optimize_scraping_cost(
            asset_type=self._infer_asset_type(modified_control),
            signal_type=self._infer_signal_type(modified_control),
            current_cost=cost_prediction.predicted_cost
        )

        # Generate budget analysis if budget exists
        budget_analysis = None
        if modified_control.budget:
            budget_analysis = await analyze_scraping_budget(
                budget=modified_control.budget.max_cost_total,
                projected_operations=[{
                    'operation_type': 'scenario_analysis',
                    'estimated_cost': cost_prediction.predicted_cost
                }]
            )

        # Create scenario analysis
        analysis = ScenarioAnalysis(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.scenario_name,
            scenario_type=scenario.scenario_type,
            preflight_assessment=preflight_assessment,
            cost_prediction=cost_prediction,
            cost_optimization=cost_optimization,
            budget_analysis=budget_analysis
        )

        # Populate analysis with derived metrics
        await self._populate_scenario_metrics(analysis, scenario)

        # Record analysis time
        end_time = asyncio.get_event_loop().time()
        analysis.analysis_duration_seconds = end_time - start_time

        return analysis

    def _infer_asset_type(self, control: ScrapeControlContract) -> AssetType:
        """Infer asset type from control."""
        source_text = ' '.join(control.intent.sources or []).lower()
        if 'company' in source_text:
            return AssetType.COMPANY
        elif 'property' in source_text:
            return AssetType.SINGLE_FAMILY_HOME
        return AssetType.PERSON

    def _infer_signal_type(self, control: ScrapeControlContract) -> Optional[SignalType]:
        """Infer signal type from control."""
        source_text = ' '.join(control.intent.sources or []).lower()
        if 'lien' in source_text:
            return SignalType.LIEN
        elif 'court' in source_text:
            return SignalType.COURT_CASE
        return None

    def _estimate_scenario_scope(self, control: ScrapeControlContract) -> int:
        """Estimate scenario scope size."""
        geo_count = len(control.intent.geography or [])
        source_count = len(control.intent.sources or [])
        return max(1, geo_count * source_count * 25)

    async def _populate_scenario_metrics(self, analysis: ScenarioAnalysis, scenario: ScenarioConfiguration):
        """Populate derived metrics for scenario analysis."""
        # Extract metrics from preflight assessment
        feasibility = analysis.preflight_assessment.get('operational_feasibility', {})
        risk = analysis.preflight_assessment.get('risk_assessment', {})
        compliance = analysis.preflight_assessment.get('compliance_status', {})

        analysis.projected_success_rate = feasibility.get('expected_success_rate', 0.8)
        analysis.projected_data_quality = scenario.quality_requirement

        # Calculate feasibility score
        readiness = analysis.preflight_assessment.get('overall_readiness', 'unknown')
        readiness_scores = {
            'ready': 0.9,
            'caution': 0.7,
            'blocked': 0.3,
            'invalid_contract': 0.1
        }
        analysis.feasibility_score = readiness_scores.get(readiness, 0.5)

        # Resource utilization
        analysis.projected_resource_utilization = {
            'cpu_intensity': feasibility.get('resource_intensity', 'medium'),
            'estimated_duration_hours': feasibility.get('estimated_duration_hours', 1.0),
            'concurrency_level': feasibility.get('concurrency_recommendation', 1)
        }

        # Risk factors
        analysis.risk_factors_introduced = []
        analysis.risk_factors_mitigated = []

        if scenario.scenario_type == ScenarioType.COST_OPTIMIZATION:
            analysis.risk_factors_introduced.append("potentially lower data quality")
            analysis.risk_factors_mitigated.append("reduced cost exposure")
        elif scenario.scenario_type == ScenarioType.QUALITY_ENHANCEMENT:
            analysis.risk_factors_mitigated.append("improved data accuracy")
            analysis.risk_factors_introduced.append("higher operational complexity")

        # Compliance implications
        analysis.compliance_implications = []
        if scenario.scenario_type == ScenarioType.COMPLIANCE_ENHANCED:
            analysis.compliance_implications.append("enhanced regulatory compliance")
        elif scenario.scenario_type == ScenarioType.SPEED_OPTIMIZATION:
            analysis.compliance_implications.append("potential monitoring gaps")

        # Intelligence components used
        analysis.intelligence_components_used = [
            'preflight_check', 'cost_predictor', 'execution_classifier',
            'intent_classifier', 'budget_analyzer'
        ]

    async def perform_what_if_analysis(
        self,
        base_control: ScrapeControlContract,
        scenarios: List[ScenarioConfiguration],
        include_sensitivity_analysis: bool = True
    ) -> WhatIfAnalysis:
        """
        Perform comprehensive what-if analysis comparing multiple scenarios.

        Args:
            base_control: Base scraping control contract
            scenarios: List of alternative scenarios to analyze
            include_sensitivity_analysis: Whether to include sensitivity analysis

        Returns:
            Complete what-if analysis with comparative insights
        """
        analysis_id = str(uuid.uuid4())
        start_time = asyncio.get_event_loop().time()

        # Analyze baseline scenario
        baseline_scenario = await self.analyze_scenario(
            ScenarioConfiguration(
                scenario_id='baseline',
                scenario_name='Baseline Configuration',
                scenario_type=ScenarioType.BASELINE,
                description='Original configuration for comparison',
                base_control=base_control
            )
        )

        # Analyze alternative scenarios
        alternative_scenarios = []
        for scenario in scenarios:
            try:
                analysis = await self.analyze_scenario(scenario)
                alternative_scenarios.append(analysis)
            except Exception as e:
                logger.error(f"Failed to analyze scenario {scenario.scenario_name}: {e}")

        # Create what-if analysis
        analysis = WhatIfAnalysis(
            analysis_id=analysis_id,
            baseline_scenario=baseline_scenario,
            alternative_scenarios=alternative_scenarios
        )

        # Generate comparative analysis
        await self._generate_scenario_comparison(analysis)

        # Perform sensitivity analysis if requested
        if include_sensitivity_analysis:
            analysis.sensitivity_analysis = await self._perform_sensitivity_analysis(
                base_control, scenarios[:3]  # Limit to first 3 for performance
            )

        # Generate optimization recommendations
        analysis.optimization_recommendations = self._generate_optimization_recommendations(analysis)

        # Identify best scenarios
        self._identify_best_scenarios(analysis)

        # Generate trade-off analysis
        self._generate_tradeoff_analysis(analysis)

        # Record total analysis time
        end_time = asyncio.get_event_loop().time()
        analysis.total_analysis_time_seconds = end_time - start_time
        analysis.scenarios_analyzed = len(alternative_scenarios) + 1

        self.analyses[analysis_id] = analysis
        self.analysis_stats['total_analyses'] += 1
        self.analysis_stats['scenarios_analyzed'] += analysis.scenarios_analyzed

        logger.info(f"Completed what-if analysis {analysis_id} with {analysis.scenarios_analyzed} scenarios")
        return analysis

    async def _generate_scenario_comparison(self, analysis: WhatIfAnalysis):
        """Generate comparative analysis between scenarios."""
        baseline_cost = analysis.baseline_scenario.cost_prediction.predicted_cost if analysis.baseline_scenario.cost_prediction else 0
        baseline_time = analysis.baseline_scenario.preflight_assessment.get('operational_feasibility', {}).get('estimated_duration_hours', 1)

        for scenario in analysis.alternative_scenarios:
            scenario_cost = scenario.cost_prediction.predicted_cost if scenario.cost_prediction else 0
            scenario_time = scenario.preflight_assessment.get('operational_feasibility', {}).get('estimated_duration_hours', baseline_time)

            # Cost comparison
            scenario.cost_difference = scenario_cost - baseline_cost
            scenario.cost_difference_percentage = ((scenario_cost - baseline_cost) / baseline_cost * 100) if baseline_cost > 0 else 0

            # Time comparison
            scenario.time_difference_hours = scenario_time - baseline_time

            # Risk comparison
            baseline_risk = analysis.baseline_scenario.preflight_assessment.get('risk_assessment', {}).get('risk_level', 'medium')
            scenario_risk = scenario.preflight_assessment.get('risk_assessment', {}).get('risk_level', 'medium')

            risk_levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            baseline_risk_num = risk_levels.get(baseline_risk, 2)
            scenario_risk_num = risk_levels.get(scenario_risk, 2)

            if scenario_risk_num < baseline_risk_num:
                scenario.risk_level_change = 'improved'
            elif scenario_risk_num > baseline_risk_num:
                scenario.risk_level_change = 'worsened'
            else:
                scenario.risk_level_change = 'unchanged'

            # Store in comparison dict
            analysis.scenario_comparison[scenario.scenario_name] = {
                'cost_difference': scenario.cost_difference,
                'cost_difference_percentage': scenario.cost_difference_percentage,
                'time_difference_hours': scenario.time_difference_hours,
                'risk_level_change': scenario.risk_level_change,
                'feasibility_score': scenario.feasibility_score,
                'overall_score': scenario.get_overall_score()
            }

    async def _perform_sensitivity_analysis(
        self,
        base_control: ScrapeControlContract,
        scenarios: List[ScenarioConfiguration]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Perform sensitivity analysis on key parameters."""
        sensitivity_results = defaultdict(list)

        # Define sensitivity parameters to test
        sensitivity_tests = [
            (SensitivityParameter.COST_INCREASE, [1.1, 1.25, 1.5]),
            (SensitivityParameter.TIME_INCREASE, [1.2, 1.5, 2.0]),
            (SensitivityParameter.SUCCESS_RATE_DECREASE, [0.9, 0.8, 0.7]),
            (SensitivityParameter.SCOPE_EXPANSION, [1.5, 2.0, 3.0])
        ]

        for param, values in sensitivity_tests:
            for value in values:
                # Create sensitivity scenario
                sensitivity_scenario = ScenarioConfiguration(
                    scenario_id=f"sensitivity_{param.value}_{value}",
                    scenario_name=f"{param.value} {value}",
                    scenario_type=ScenarioType.SENSITIVITY_ANALYSIS,
                    description=f"Sensitivity analysis: {param.value} = {value}",
                    base_control=base_control
                )

                # Apply sensitivity modification
                if param == SensitivityParameter.COST_INCREASE:
                    sensitivity_scenario.cost_sensitivity_factor = value
                elif param == SensitivityParameter.TIME_INCREASE:
                    sensitivity_scenario.time_sensitivity_factor = value
                elif param == SensitivityParameter.SUCCESS_RATE_DECREASE:
                    # This would modify expected success rates
                    pass
                elif param == SensitivityParameter.SCOPE_EXPANSION:
                    # This would modify scope parameters
                    pass

                # Analyze sensitivity scenario
                try:
                    sensitivity_analysis = await self.analyze_scenario(sensitivity_scenario)

                    sensitivity_results[param.value].append({
                        'parameter_value': value,
                        'cost_impact': sensitivity_analysis.cost_prediction.predicted_cost if sensitivity_analysis.cost_prediction else 0,
                        'feasibility_impact': sensitivity_analysis.feasibility_score,
                        'overall_impact': sensitivity_analysis.get_overall_score()
                    })

                except Exception as e:
                    logger.warning(f"Sensitivity analysis failed for {param.value}={value}: {e}")

        return dict(sensitivity_results)

    def _generate_optimization_recommendations(self, analysis: WhatIfAnalysis) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on scenario analysis."""
        recommendations = []

        # Analyze cost vs quality trade-offs
        cost_quality_scenarios = sorted(
            [analysis.baseline_scenario] + analysis.alternative_scenarios,
            key=lambda s: s.cost_prediction.predicted_cost if s.cost_prediction else float('inf')
        )

        if len(cost_quality_scenarios) >= 2:
            best_cost = cost_quality_scenarios[0]
            best_quality = max(
                [analysis.baseline_scenario] + analysis.alternative_scenarios,
                key=lambda s: {'basic': 1, 'standard': 2, 'premium': 3}.get(s.projected_data_quality, 0)
            )

            if best_cost.scenario_name != best_quality.scenario_name:
                recommendations.append({
                    'type': 'trade_off_optimization',
                    'title': 'Cost-Quality Balance',
                    'description': f"Consider {best_cost.scenario_name} for cost efficiency or {best_quality.scenario_name} for quality",
                    'cost_savings': abs(best_cost.cost_difference) if hasattr(best_cost, 'cost_difference') else 0,
                    'quality_improvement': best_quality.projected_data_quality,
                    'recommendation': 'evaluate_based_on_requirements'
                })

        # Risk mitigation recommendations
        high_risk_scenarios = [
            s for s in [analysis.baseline_scenario] + analysis.alternative_scenarios
            if s.preflight_assessment.get('risk_assessment', {}).get('risk_level') in ['high', 'critical']
        ]

        if high_risk_scenarios:
            recommendations.append({
                'type': 'risk_mitigation',
                'title': 'Risk Mitigation Required',
                'description': f"{len(high_risk_scenarios)} scenarios have high risk - consider compliance enhancements",
                'affected_scenarios': [s.scenario_name for s in high_risk_scenarios],
                'recommendation': 'implement_additional_safeguards'
            })

        # Budget optimization recommendations
        budget_constrained_scenarios = [
            s for s in [analysis.baseline_scenario] + analysis.alternative_scenarios
            if s.preflight_assessment.get('overall_readiness') == 'budget_exceeded'
        ]

        if budget_constrained_scenarios:
            recommendations.append({
                'type': 'budget_optimization',
                'title': 'Budget Constraints Identified',
                'description': f"{len(budget_constrained_scenarios)} scenarios exceed budget limits",
                'affected_scenarios': [s.scenario_name for s in budget_constrained_scenarios],
                'recommendation': 'reduce_scope_or_increase_budget'
            })

        return recommendations

    def _identify_best_scenarios(self, analysis: WhatIfAnalysis):
        """Identify best scenarios for different criteria."""
        all_scenarios = [analysis.baseline_scenario] + analysis.alternative_scenarios

        # Best overall
        analysis.best_overall_scenario = max(
            all_scenarios,
            key=lambda s: s.get_overall_score()
        ).scenario_name

        # Best cost
        analysis.best_cost_scenario = min(
            (s for s in all_scenarios if s.cost_prediction),
            key=lambda s: s.cost_prediction.predicted_cost
        ).scenario_name

        # Best quality
        analysis.best_quality_scenario = max(
            all_scenarios,
            key=lambda s: {'basic': 1, 'standard': 2, 'premium': 3}.get(s.projected_data_quality, 0)
        ).scenario_name

        # Best speed
        baseline_time = analysis.baseline_scenario.preflight_assessment.get('operational_feasibility', {}).get('estimated_duration_hours', 1)
        analysis.best_speed_scenario = min(
            all_scenarios,
            key=lambda s: s.preflight_assessment.get('operational_feasibility', {}).get('estimated_duration_hours', baseline_time)
        ).scenario_name

    def _generate_tradeoff_analysis(self, analysis: WhatIfAnalysis):
        """Generate trade-off analysis between different dimensions."""
        all_scenarios = [analysis.baseline_scenario] + analysis.alternative_scenarios

        # Cost vs Quality trade-offs
        for scenario in all_scenarios:
            if scenario.cost_prediction:
                quality_score = {'basic': 1, 'standard': 2, 'premium': 3}.get(scenario.projected_data_quality, 2)
                analysis.cost_vs_quality_tradeoffs.append({
                    'scenario': scenario.scenario_name,
                    'cost': scenario.cost_prediction.predicted_cost,
                    'quality_score': quality_score,
                    'efficiency': quality_score / scenario.cost_prediction.predicted_cost
                })

        # Speed vs Cost trade-offs
        baseline_time = analysis.baseline_scenario.preflight_assessment.get('operational_feasibility', {}).get('estimated_duration_hours', 1)
        for scenario in all_scenarios:
            scenario_time = scenario.preflight_assessment.get('operational_feasibility', {}).get('estimated_duration_hours', baseline_time)
            if scenario.cost_prediction:
                analysis.speed_vs_cost_tradeoffs.append({
                    'scenario': scenario.scenario_name,
                    'time_hours': scenario_time,
                    'cost': scenario.cost_prediction.predicted_cost,
                    'cost_per_hour': scenario.cost_prediction.predicted_cost / scenario_time if scenario_time > 0 else 0
                })

        # Risk vs Benefit trade-offs
        for scenario in all_scenarios:
            risk_level = scenario.preflight_assessment.get('risk_assessment', {}).get('risk_level', 'medium')
            risk_score = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}.get(risk_level, 2)
            benefit_score = scenario.get_overall_score()

            analysis.risk_vs_benefit_tradeoffs.append({
                'scenario': scenario.scenario_name,
                'risk_score': risk_score,
                'benefit_score': benefit_score,
                'risk_adjusted_benefit': benefit_score / risk_score
            })

    async def generate_scenario_recommendations(
        self,
        analysis: WhatIfAnalysis,
        criteria_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive scenario recommendations based on analysis.

        Args:
            analysis: Complete what-if analysis
            criteria_weights: Optional custom weights for different criteria

        Returns:
            Comprehensive recommendation package
        """
        # Default weights if not provided
        weights = criteria_weights or {
            'cost': 0.25,
            'quality': 0.25,
            'speed': 0.20,
            'risk': 0.15,
            'compliance': 0.15
        }

        # Calculate weighted scores for each scenario
        scenario_scores = {}
        all_scenarios = [analysis.baseline_scenario] + analysis.alternative_scenarios

        for scenario in all_scenarios:
            scores = analysis.get_scenario_ranking('all_criteria')  # This would need to be implemented
            scenario_scores[scenario.scenario_name] = {
                'cost_score': analysis.get_scenario_ranking('cost')[scenario.scenario_name][1],
                'quality_score': analysis.get_scenario_ranking('quality')[scenario.scenario_name][1],
                'speed_score': analysis.get_scenario_ranking('speed')[scenario.scenario_name][1],
                'risk_score': analysis.get_scenario_ranking('risk')[scenario.scenario_name][1],
                'compliance_score': 1.0 if scenario.compliance_change != 'worsened' else 0.5,
                'overall_score': scenario.get_overall_score()
            }

        # Generate recommendation package
        recommendations = {
            'primary_recommendation': analysis.best_overall_scenario,
            'alternative_recommendations': [
                s[0] for s in analysis.get_scenario_ranking()[:3]
            ],
            'scenario_scores': scenario_scores,
            'trade_off_analysis': {
                'cost_vs_quality': sorted(analysis.cost_vs_quality_tradeoffs, key=lambda x: x['efficiency'], reverse=True),
                'speed_vs_cost': sorted(analysis.speed_vs_cost_tradeoffs, key=lambda x: x['cost_per_hour']),
                'risk_vs_benefit': sorted(analysis.risk_vs_benefit_tradeoffs, key=lambda x: x['risk_adjusted_benefit'], reverse=True)
            },
            'decision_factors': {
                'budget_considerations': self._analyze_budget_impact(analysis),
                'risk_considerations': self._analyze_risk_impact(analysis),
                'operational_considerations': self._analyze_operational_impact(analysis),
                'compliance_considerations': self._analyze_compliance_impact(analysis)
            },
            'implementation_guidance': self._generate_implementation_guidance(analysis)
        }

        return recommendations

    def _analyze_budget_impact(self, analysis: WhatIfAnalysis) -> Dict[str, Any]:
        """Analyze budget impact across scenarios."""
        return {
            'cost_range': self._calculate_cost_range(analysis),
            'budget_risk_scenarios': [
                s.scenario_name for s in analysis.alternative_scenarios
                if s.preflight_assessment.get('overall_readiness') == 'budget_exceeded'
            ],
            'savings_opportunities': [
                s.scenario_name for s in analysis.alternative_scenarios
                if hasattr(s, 'cost_difference') and s.cost_difference < -100  # Savings > $100
            ]
        }

    def _analyze_risk_impact(self, analysis: WhatIfAnalysis) -> Dict[str, Any]:
        """Analyze risk impact across scenarios."""
        return {
            'risk_reduction_scenarios': [
                s.scenario_name for s in analysis.alternative_scenarios
                if s.risk_level_change == 'improved'
            ],
            'risk_increase_scenarios': [
                s.scenario_name for s in analysis.alternative_scenarios
                if s.risk_level_change == 'worsened'
            ],
            'high_risk_scenarios': [
                s.scenario_name for s in [analysis.baseline_scenario] + analysis.alternative_scenarios
                if s.preflight_assessment.get('risk_assessment', {}).get('risk_level') in ['high', 'critical']
            ]
        }

    def _analyze_operational_impact(self, analysis: WhatIfAnalysis) -> Dict[str, Any]:
        """Analyze operational impact across scenarios."""
        return {
            'fastest_scenarios': [
                s[0] for s in analysis.get_scenario_ranking('speed')[:2]
            ],
            'most_feasible_scenarios': [
                s.scenario_name for s in [analysis.baseline_scenario] + analysis.alternative_scenarios
                if s.feasibility_score > 0.8
            ],
            'resource_intensive_scenarios': [
                s.scenario_name for s in [analysis.baseline_scenario] + analysis.alternative_scenarios
                if s.preflight_assessment.get('operational_feasibility', {}).get('resource_intensity') == 'high'
            ]
        }

    def _analyze_compliance_impact(self, analysis: WhatIfAnalysis) -> Dict[str, Any]:
        """Analyze compliance impact across scenarios."""
        return {
            'compliance_improved_scenarios': [
                s.scenario_name for s in analysis.alternative_scenarios
                if s.compliance_change == 'improved'
            ],
            'compliance_concerns': [
                s.scenario_name for s in analysis.alternative_scenarios
                if s.compliance_change == 'worsened' or len(s.compliance_implications) > 0
            ]
        }

    def _generate_implementation_guidance(self, analysis: WhatIfAnalysis) -> Dict[str, Any]:
        """Generate implementation guidance for recommended scenarios."""
        guidance = {
            'phased_implementation': [],
            'resource_requirements': {},
            'risk_mitigations': [],
            'monitoring_requirements': [],
            'rollback_procedures': []
        }

        # Implementation phases
        guidance['phased_implementation'] = [
            "Phase 1: Pilot selected scenario with limited scope",
            "Phase 2: Gradual rollout with monitoring and metrics collection",
            "Phase 3: Full implementation with contingency plans",
            "Phase 4: Optimization and refinement based on performance data"
        ]

        # Resource requirements
        best_scenario = analysis.best_overall_scenario
        if best_scenario:
            scenario_obj = next(
                (s for s in [analysis.baseline_scenario] + analysis.alternative_scenarios
                 if s.scenario_name == best_scenario),
                None
            )
            if scenario_obj:
                guidance['resource_requirements'] = scenario_obj.projected_resource_utilization

        # Risk mitigations
        guidance['risk_mitigations'] = [
            "Implement comprehensive monitoring and alerting",
            "Establish performance baselines and thresholds",
            "Prepare contingency plans for alternative scenarios",
            "Conduct regular risk assessments and updates"
        ]

        # Monitoring requirements
        guidance['monitoring_requirements'] = [
            "Cost tracking and budget monitoring",
            "Performance metrics and success rate monitoring",
            "Quality validation and data accuracy checks",
            "Compliance monitoring and audit trail maintenance"
        ]

        # Rollback procedures
        guidance['rollback_procedures'] = [
            "Maintain baseline configuration as rollback option",
            "Implement gradual rollback to minimize disruption",
            "Preserve data integrity during rollback operations",
            "Document lessons learned for future scenario planning"
        ]

        return guidance

    def _calculate_cost_range(self, analysis: WhatIfAnalysis) -> Dict[str, float]:
        """Calculate cost range across all scenarios."""
        all_costs = []
        for scenario in [analysis.baseline_scenario] + analysis.alternative_scenarios:
            if scenario.cost_prediction:
                all_costs.append(scenario.cost_prediction.predicted_cost)

        if not all_costs:
            return {'min': 0, 'max': 0, 'range': 0, 'average': 0}

        return {
            'min': min(all_costs),
            'max': max(all_costs),
            'range': max(all_costs) - min(all_costs),
            'average': sum(all_costs) / len(all_costs)
        }

    def get_analyzer_stats(self) -> Dict[str, Any]:
        """Get comprehensive analyzer performance statistics."""
        stats = dict(self.analysis_stats)

        # Calculate derived statistics
        total_analyses = stats.get('total_analyses', 0)
        total_scenarios = stats.get('scenarios_analyzed', 0)

        if total_analyses > 0:
            stats['average_scenarios_per_analysis'] = total_scenarios / total_analyses

        # Analysis success rates (would need tracking)
        stats['analysis_success_rate'] = 0.95  # Placeholder

        # Performance metrics
        stats['active_analyses'] = len(self.analyses)

        return stats


# Enhanced what-if cost analysis function
async def what_if_cost_analysis(
    signals: Optional[List[SignalType]] = None,
    geography_levels: Optional[List[str]] = None,
    browser_pages: int = 0,
    base_budget: Optional[float] = None,
    risk_tolerance: str = "medium",
    time_sensitivity: str = "normal",
    quality_requirement: str = "standard",
    include_optimizations: bool = True,
    include_sensitivity: bool = True,
    include_comparative: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive what-if cost analysis for different operational scenarios.

    Performs detailed cost estimation, optimization analysis, budget impact assessment,
    sensitivity testing, and comparative evaluation across multiple geography levels
    and operational parameters.

    Args:
        signals: List of signal types to analyze (optional)
        geography_levels: List of geography levels to test (optional)
        browser_pages: Number of browser pages required (default: 0)
        base_budget: Base budget constraint for analysis (optional)
        risk_tolerance: Risk tolerance level ("low", "medium", "high")
        time_sensitivity: Time sensitivity ("low", "normal", "high", "critical")
        quality_requirement: Quality requirement level ("basic", "standard", "verified", "premium")
        include_optimizations: Whether to include cost optimization analysis
        include_sensitivity: Whether to include sensitivity analysis
        include_comparative: Whether to include comparative analysis

    Returns:
        Comprehensive what-if cost analysis with scenarios, recommendations, and insights

    Example:
        analysis = await what_if_cost_analysis(
            signals=[SignalType.LIEN, SignalType.MORTGAGE],
            geography_levels=["city", "county", "state"],
            browser_pages=25,
            base_budget=5000.0
        )
        print(f"Best geography: {analysis['recommendations']['optimal_geography']}")
        print(f"Cost range: ${analysis['cost_summary']['min_cost']}-${analysis['cost_summary']['max_cost']}")
    """
    analysis = {
        'analysis_id': str(uuid.uuid4()),
        'timestamp': datetime.utcnow(),
        'parameters': {
            'signals': [s.value for s in signals] if signals else None,
            'geography_levels': geography_levels,
            'browser_pages': browser_pages,
            'base_budget': base_budget,
            'risk_tolerance': risk_tolerance,
            'time_sensitivity': time_sensitivity,
            'quality_requirement': quality_requirement
        },
        'geography_analysis': [],
        'cost_summary': {},
        'optimization_opportunities': [],
        'sensitivity_analysis': {},
        'comparative_insights': {},
        'budget_impact': {},
        'recommendations': {},
        'risk_assessment': {},
        'performance_projections': {}
    }

    try:
        # Default values if not provided
        if not geography_levels:
            geography_levels = ["city", "county", "state", "national"]
        if not signals:
            signals = [SignalType.LIEN, SignalType.MORTGAGE, SignalType.COURT_CASE]

        # Analyze each geography level
        geography_results = []
        for geo_level in geography_levels:
            try:
                # Create control contract for this geography scenario
                control = _create_geography_scenario_control(
                    signals, geo_level, browser_pages, base_budget,
                    risk_tolerance, time_sensitivity, quality_requirement
                )

                # Perform comprehensive cost prediction
                cost_prediction = await predict_scraping_cost(
                    asset_type=_infer_asset_type_from_signals(signals),
                    signal_type=signals[0] if len(signals) == 1 else None,  # Primary signal
                    execution_mode=None,  # Let classifier determine optimal
                    scope_size=_estimate_geography_scope(geo_level, signals),
                    risk_level=None,  # Let classifier determine
                    intent_category=None,  # Let classifier determine
                    control=control
                )

                # Get optimization recommendations
                optimization_plan = None
                if include_optimizations and cost_prediction.predicted_cost > 100:
                    optimization_plan = await optimize_scraping_cost(
                        asset_type=_infer_asset_type_from_signals(signals),
                        signal_type=signals[0] if len(signals) == 1 else None,
                        current_cost=cost_prediction.predicted_cost
                    )

                # Perform budget analysis if budget provided
                budget_analysis = None
                if base_budget:
                    budget_analysis = await analyze_scraping_budget(
                        budget=base_budget,
                        projected_operations=[{
                            'operation_type': f'{geo_level}_scraping',
                            'estimated_cost': cost_prediction.predicted_cost
                        }]
                    )

                # Calculate geography-specific metrics
                geography_result = {
                    'geography_level': geo_level,
                    'estimated_cost': cost_prediction.predicted_cost,
                    'cost_confidence': cost_prediction.confidence_score,
                    'cost_range': cost_prediction.cost_range,
                    'cost_breakdown': cost_prediction.cost_breakdown,
                    'optimization_available': optimization_plan is not None,
                    'optimization_savings': optimization_plan.cost_savings if optimization_plan else 0,
                    'budget_compliance': 'compliant',
                    'budget_utilization': 0.0,
                    'cost_efficiency_score': _calculate_cost_efficiency(cost_prediction, geo_level, signals),
                    'scalability_score': _calculate_scalability_score(geo_level, signals),
                    'data_density_score': _calculate_data_density_score(geo_level, signals),
                    'recommendation_score': 0.0  # Will be calculated comparatively
                }

                # Budget compliance check
                if base_budget:
                    utilization = (cost_prediction.predicted_cost / base_budget) * 100
                    geography_result['budget_utilization'] = round(utilization, 1)

                    if utilization > 100:
                        geography_result['budget_compliance'] = 'exceeded'
                    elif utilization > 90:
                        geography_result['budget_compliance'] = 'high_utilization'
                    else:
                        geography_result['budget_compliance'] = 'compliant'

                # Add optimization details
                if optimization_plan:
                    geography_result['optimization_details'] = {
                        'savings_percentage': optimization_plan.savings_percentage,
                        'recommended_changes': [
                            change.get('change', 'Optimization available')
                            for change in optimization_plan.recommended_changes[:3]
                        ],
                        'implementation_effort': optimization_plan.implementation_priority
                    }

                geography_results.append(geography_result)

            except Exception as e:
                logger.warning(f"Failed to analyze geography level {geo_level}: {e}")
                # Add error result
                geography_results.append({
                    'geography_level': geo_level,
                    'error': str(e),
                    'estimated_cost': 0,
                    'cost_confidence': 0,
                    'recommendation_score': 0
                })

        analysis['geography_analysis'] = geography_results

        # Generate cost summary
        analysis['cost_summary'] = _generate_cost_summary(geography_results)

        # Generate optimization opportunities
        if include_optimizations:
            analysis['optimization_opportunities'] = _generate_cost_optimization_opportunities(
                geography_results, base_budget
            )

        # Perform sensitivity analysis
        if include_sensitivity:
            analysis['sensitivity_analysis'] = await _perform_cost_sensitivity_analysis(
                signals, geography_levels, browser_pages, base_budget
            )

        # Generate comparative insights
        if include_comparative:
            analysis['comparative_insights'] = _generate_comparative_insights(geography_results)

        # Analyze budget impact
        if base_budget:
            analysis['budget_impact'] = _analyze_budget_impact(geography_results, base_budget)

        # Generate recommendations
        analysis['recommendations'] = _generate_cost_recommendations(
            geography_results, base_budget, risk_tolerance, time_sensitivity, quality_requirement
        )

        # Add risk assessment
        analysis['risk_assessment'] = _assess_cost_risks(
            geography_results, risk_tolerance, time_sensitivity, quality_requirement
        )

        # Add performance projections
        analysis['performance_projections'] = _generate_performance_projections(
            geography_results, signals, browser_pages
        )

        # Calculate final recommendation scores
        analysis = _calculate_recommendation_scores(analysis)

        return analysis

    except Exception as e:
        logger.error(f"What-if cost analysis failed: {e}")
        analysis['error'] = str(e)
        analysis['overall_status'] = 'failed'
        return analysis


def _create_geography_scenario_control(
    signals: List[SignalType],
    geography_level: str,
    browser_pages: int,
    base_budget: Optional[float],
    risk_tolerance: str,
    time_sensitivity: str,
    quality_requirement: str
) -> ScrapeControlContract:
    """Create a control contract for geography-specific scenario analysis."""
    from core.control_models import ScrapeIntent, ScrapeBudget

    # Create intent based on signals and geography
    intent = ScrapeIntent(
        purpose=f"What-if cost analysis for {geography_level} geography with {len(signals)} signal types",
        sources=_select_sources_for_signals(signals, geography_level),
        geography=[geography_level],  # Single geography for this scenario
        event_type=None  # Not event-specific
    )

    # Create budget if provided
    budget = None
    if base_budget:
        # Estimate runtime and pages based on geography and signals
        estimated_runtime = _estimate_runtime_minutes(geography_level, len(signals), browser_pages)
        estimated_pages = _estimate_page_count(geography_level, len(signals), browser_pages)

        budget = ScrapeBudget(
            max_runtime_minutes=estimated_runtime,
            max_pages=estimated_pages,
            max_records=_estimate_record_count(geography_level, len(signals)),
            max_cost_total=base_budget
        )

    # Create authorization (placeholder for analysis)
    authorization = ScrapeAuthorization(
        approved_by="what_if_analyzer",
        purpose="Cost analysis scenario testing",
        expires_at=datetime.utcnow() + timedelta(days=1)
    )

    return ScrapeControlContract(
        intent=intent,
        budget=budget,
        authorization=authorization
    )


def _select_sources_for_signals(signals: List[SignalType], geography_level: str) -> List[str]:
    """Select appropriate data sources based on signals and geography."""
    # Geography-based source selection
    geo_sources = {
        "city": ["city_clerk", "municipal_records", "local_databases"],
        "county": ["county_clerk", "county_recorder", "county_databases"],
        "state": ["state_registry", "state_court", "state_databases"],
        "national": ["federal_databases", "national_registries", "commercial_databases"]
    }

    base_sources = geo_sources.get(geography_level, ["general_databases"])

    # Signal-specific source additions
    signal_sources = []
    for signal in signals:
        if signal == SignalType.LIEN:
            signal_sources.extend(["lien_records", "property_liens"])
        elif signal == SignalType.MORTGAGE:
            signal_sources.extend(["mortgage_records", "lending_databases"])
        elif signal == SignalType.COURT_CASE:
            signal_sources.extend(["court_records", "legal_databases"])
        elif signal == SignalType.FORECLOSURE:
            signal_sources.extend(["foreclosure_records", "distressed_property"])
        elif signal == SignalType.DEED:
            signal_sources.extend(["deed_records", "property_transfers"])

    return list(set(base_sources + signal_sources))[:5]  # Limit to 5 sources


def _estimate_geography_scope(geography_level: str, signals: List[SignalType]) -> int:
    """Estimate the scope size for a geography level and signal set."""
    # Base scope multipliers by geography
    geo_multipliers = {
        "city": 1,
        "county": 5,
        "state": 25,
        "national": 100
    }

    base_multiplier = geo_multipliers.get(geography_level, 10)

    # Signal complexity multiplier
    signal_multiplier = len(signals) * 0.8

    # Estimate scope (rough approximation)
    return int(base_multiplier * signal_multiplier * 100)


def _infer_asset_type_from_signals(signals: List[SignalType]) -> AssetType:
    """Infer asset type from signal types."""
    if any(s in [SignalType.LIEN, SignalType.MORTGAGE, SignalType.DEED, SignalType.FORECLOSURE] for s in signals):
        return AssetType.SINGLE_FAMILY_HOME
    elif any(s in [SignalType.COURT_CASE, SignalType.JUDGMENT] for s in signals):
        return AssetType.COMPANY
    else:
        return AssetType.PERSON


def _estimate_runtime_minutes(geography_level: str, signal_count: int, browser_pages: int) -> int:
    """Estimate runtime in minutes based on geography, signals, and browser usage."""
    # Base time by geography
    geo_times = {
        "city": 30,
        "county": 120,
        "state": 480,
        "national": 1440
    }

    base_time = geo_times.get(geography_level, 240)

    # Signal complexity factor
    signal_factor = 1 + (signal_count - 1) * 0.3

    # Browser time factor
    browser_factor = 1 + (browser_pages / 10) * 0.5

    return int(base_time * signal_factor * browser_factor)


def _estimate_page_count(geography_level: str, signal_count: int, browser_pages: int) -> int:
    """Estimate page count based on geography, signals, and browser usage."""
    # Base pages by geography
    geo_pages = {
        "city": 100,
        "county": 500,
        "state": 2000,
        "national": 10000
    }

    base_pages = geo_pages.get(geography_level, 1000)

    # Signal page factor
    signal_factor = signal_count * 0.8

    # Browser pages are additional
    return int(base_pages * signal_factor) + browser_pages


def _estimate_record_count(geography_level: str, signal_count: int) -> int:
    """Estimate record count based on geography and signals."""
    # Base records by geography
    geo_records = {
        "city": 1000,
        "county": 5000,
        "state": 25000,
        "national": 100000
    }

    base_records = geo_records.get(geography_level, 10000)

    # Signal record factor
    signal_factor = signal_count * 0.9

    return int(base_records * signal_factor)


def _calculate_cost_efficiency(cost_prediction: CostPrediction, geography_level: str, signals: List[SignalType]) -> float:
    """Calculate cost efficiency score for geography/signal combination."""
    if not cost_prediction or cost_prediction.predicted_cost <= 0:
        return 0.0

    # Efficiency based on cost per signal per geography scope
    geography_scope_factor = {
        "city": 1.0,    # Smallest scope = highest efficiency
        "county": 0.8,
        "state": 0.6,
        "national": 0.4
    }.get(geography_level, 0.7)

    signal_efficiency_factor = 1.0 / len(signals)  # More signals = lower per-signal efficiency

    base_efficiency = geography_scope_factor * signal_efficiency_factor

    # Adjust for cost confidence
    confidence_adjustment = cost_prediction.confidence_score * 0.2

    return base_efficiency + confidence_adjustment


def _calculate_scalability_score(geography_level: str, signals: List[SignalType]) -> float:
    """Calculate scalability score for geography/signal combination."""
    # Smaller geographies are more scalable
    geography_scalability = {
        "city": 0.9,
        "county": 0.7,
        "state": 0.5,
        "national": 0.3
    }.get(geography_level, 0.6)

    # Fewer signals are more scalable
    signal_scalability = 1.0 - (len(signals) - 1) * 0.1
    signal_scalability = max(0.3, signal_scalability)

    return geography_scalability * signal_scalability


def _calculate_data_density_score(geography_level: str, signals: List[SignalType]) -> float:
    """Calculate data density score (higher = more data per unit cost)."""
    # Larger geographies generally have higher data density
    geography_density = {
        "city": 0.6,
        "county": 0.8,
        "state": 0.9,
        "national": 1.0
    }.get(geography_level, 0.7)

    # Certain signals have higher data density
    high_density_signals = [SignalType.LIEN, SignalType.MORTGAGE, SignalType.DEED]
    density_signals = sum(1 for s in signals if s in high_density_signals)

    signal_density_bonus = density_signals / len(signals) * 0.2

    return geography_density + signal_density_bonus


def _generate_cost_summary(geography_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive cost summary across all geography levels."""
    valid_results = [r for r in geography_results if 'error' not in r]

    if not valid_results:
        return {'error': 'No valid geography results'}

    costs = [r['estimated_cost'] for r in valid_results]

    return {
        'total_geography_levels': len(geography_results),
        'valid_analyses': len(valid_results),
        'min_cost': min(costs) if costs else 0,
        'max_cost': max(costs) if costs else 0,
        'average_cost': sum(costs) / len(costs) if costs else 0,
        'cost_range': (max(costs) - min(costs)) if len(costs) > 1 else 0,
        'cost_variance_coefficient': _calculate_cost_variance_coefficient(costs),
        'most_cost_effective': min(valid_results, key=lambda x: x['estimated_cost'])['geography_level'],
        'least_cost_effective': max(valid_results, key=lambda x: x['estimated_cost'])['geography_level'],
        'budget_compliant_geographies': [
            r['geography_level'] for r in valid_results
            if r.get('budget_compliance') == 'compliant'
        ],
        'high_utilization_geographies': [
            r['geography_level'] for r in valid_results
            if r.get('budget_compliance') == 'high_utilization'
        ],
        'budget_exceeded_geographies': [
            r['geography_level'] for r in valid_results
            if r.get('budget_compliance') == 'exceeded'
        ]
    }


def _calculate_cost_variance_coefficient(costs: List[float]) -> float:
    """Calculate coefficient of variation for costs."""
    if len(costs) < 2:
        return 0.0

    mean_cost = sum(costs) / len(costs)
    if mean_cost == 0:
        return 0.0

    variance = sum((cost - mean_cost) ** 2 for cost in costs) / len(costs)
    std_dev = variance ** 0.5

    return std_dev / mean_cost


def _generate_cost_optimization_opportunities(
    geography_results: List[Dict[str, Any]],
    base_budget: Optional[float]
) -> List[Dict[str, Any]]:
    """Generate cost optimization opportunities across geographies."""
    opportunities = []

    valid_results = [r for r in geography_results if 'error' not in r]

    if len(valid_results) < 2:
        return opportunities

    # Geography optimization
    most_efficient = min(valid_results, key=lambda x: x['estimated_cost'])
    least_efficient = max(valid_results, key=lambda x: x['estimated_cost'])
    savings_potential = least_efficient['estimated_cost'] - most_efficient['estimated_cost']

    if savings_potential > 100:  # Significant savings
        opportunities.append({
            'type': 'geography_optimization',
            'title': 'Geography Selection Optimization',
            'description': f"Switch from {least_efficient['geography_level']} to {most_efficient['geography_level']} for ${savings_potential:.2f} savings",
            'savings_amount': savings_potential,
            'savings_percentage': (savings_potential / least_efficient['estimated_cost']) * 100,
            'implementation_effort': 'medium',
            'impact_level': 'high'
        })

    # Budget optimization
    if base_budget:
        high_utilization = [
            r for r in valid_results
            if r.get('budget_utilization', 0) > 90
        ]

        if high_utilization:
            opportunities.append({
                'type': 'budget_optimization',
                'title': 'Budget Constraint Optimization',
                'description': f"{len(high_utilization)} geographies exceed 90% budget utilization",
                'affected_geographies': [r['geography_level'] for r in high_utilization],
                'recommendation': 'Consider phased execution or budget increase',
                'implementation_effort': 'high',
                'impact_level': 'critical'
            })

    # Efficiency optimization
    low_efficiency = [
        r for r in valid_results
        if r.get('cost_efficiency_score', 0) < 0.5
    ]

    if low_efficiency:
        opportunities.append({
            'type': 'efficiency_optimization',
            'title': 'Efficiency Improvement Opportunities',
            'description': f"{len(low_efficiency)} geographies have low cost efficiency",
            'affected_geographies': [r['geography_level'] for r in low_efficiency],
            'recommendation': 'Consider signal optimization or source selection',
            'implementation_effort': 'medium',
            'impact_level': 'medium'
        })

    return opportunities


async def _perform_cost_sensitivity_analysis(
    signals: List[SignalType],
    geography_levels: List[str],
    browser_pages: int,
    base_budget: Optional[float]
) -> Dict[str, Any]:
    """Perform sensitivity analysis on cost parameters."""
    sensitivity_results = {
        'cost_increase_sensitivity': [],
        'time_increase_sensitivity': [],
        'scope_reduction_sensitivity': [],
        'quality_degradation_sensitivity': []
    }

    # Test cost increases
    for increase_factor in [1.1, 1.25, 1.5]:  # 10%, 25%, 50% increases
        cost_impacts = []
        for geo_level in geography_levels[:2]:  # Test first 2 geographies for speed
            try:
                control = _create_geography_scenario_control(
                    signals, geo_level, browser_pages, base_budget, "medium", "normal", "standard"
                )

                # Apply cost sensitivity
                cost_prediction = await predict_scraping_cost(
                    asset_type=_infer_asset_type_from_signals(signals),
                    signal_type=signals[0] if len(signals) == 1 else None,
                    execution_mode=None,
                    scope_size=_estimate_geography_scope(geo_level, signals),
                    risk_level=None,
                    intent_category=None,
                    control=control
                )

                # Apply sensitivity factor
                adjusted_cost = cost_prediction.predicted_cost * increase_factor

                cost_impacts.append({
                    'geography': geo_level,
                    'base_cost': cost_prediction.predicted_cost,
                    'adjusted_cost': adjusted_cost,
                    'increase_amount': adjusted_cost - cost_prediction.predicted_cost,
                    'increase_percentage': (increase_factor - 1) * 100
                })

            except Exception as e:
                logger.debug(f"Sensitivity analysis failed for {geo_level}: {e}")

        if cost_impacts:
            avg_impact = sum(impact['increase_amount'] for impact in cost_impacts) / len(cost_impacts)
            sensitivity_results['cost_increase_sensitivity'].append({
                'increase_percentage': (increase_factor - 1) * 100,
                'average_cost_impact': avg_impact,
                'geography_impacts': cost_impacts
            })

    # Test time increases (simplified - would affect execution time estimates)
    for time_factor in [1.2, 1.5, 2.0]:  # 20%, 50%, 100% time increases
        sensitivity_results['time_increase_sensitivity'].append({
            'time_increase_percentage': (time_factor - 1) * 100,
            'estimated_cost_impact_percentage': (time_factor - 1) * 20,  # Rough estimate
            'recommendation': 'Monitor execution time closely'
        })

    # Test scope reductions
    for reduction_factor in [0.8, 0.6, 0.4]:  # 20%, 40%, 60% reductions
        sensitivity_results['scope_reduction_sensitivity'].append({
            'scope_reduction_percentage': (1 - reduction_factor) * 100,
            'estimated_cost_impact_percentage': (1 - reduction_factor) * 70,  # Significant impact
            'recommendation': 'Scope reduction significantly impacts data coverage'
        })

    return sensitivity_results


def _generate_comparative_insights(geography_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comparative insights across geography levels."""
    valid_results = [r for r in geography_results if 'error' not in r]

    insights = {
        'cost_distribution': {},
        'efficiency_comparison': {},
        'scalability_comparison': {},
        'data_density_comparison': {},
        'trade_off_analysis': {}
    }

    if len(valid_results) < 2:
        return insights

    # Cost distribution analysis
    costs = [r['estimated_cost'] for r in valid_results]
    insights['cost_distribution'] = {
        'cost_spread': max(costs) - min(costs),
        'cost_spread_percentage': ((max(costs) - min(costs)) / min(costs)) * 100 if min(costs) > 0 else 0,
        'most_cost_effective': min(valid_results, key=lambda x: x['estimated_cost']),
        'least_cost_effective': max(valid_results, key=lambda x: x['estimated_cost'])
    }

    # Efficiency comparison
    efficiency_scores = [(r['geography_level'], r.get('cost_efficiency_score', 0)) for r in valid_results]
    insights['efficiency_comparison'] = {
        'most_efficient': max(efficiency_scores, key=lambda x: x[1]),
        'least_efficient': min(efficiency_scores, key=lambda x: x[1]),
        'efficiency_variance': max(s for _, s in efficiency_scores) - min(s for _, s in efficiency_scores)
    }

    # Scalability comparison
    scalability_scores = [(r['geography_level'], r.get('scalability_score', 0)) for r in valid_results]
    insights['scalability_comparison'] = {
        'most_scalable': max(scalability_scores, key=lambda x: x[1]),
        'least_scalable': min(scalability_scores, key=lambda x: x[1])
    }

    # Data density comparison
    density_scores = [(r['geography_level'], r.get('data_density_score', 0)) for r in valid_results]
    insights['data_density_comparison'] = {
        'highest_density': max(density_scores, key=lambda x: x[1]),
        'lowest_density': min(density_scores, key=lambda x: x[1])
    }

    # Trade-off analysis
    insights['trade_off_analysis'] = {
        'cost_vs_efficiency': [
            {
                'geography': r['geography_level'],
                'cost': r['estimated_cost'],
                'efficiency': r.get('cost_efficiency_score', 0),
                'trade_off_score': r['estimated_cost'] / max(r.get('cost_efficiency_score', 0.1), 0.1)
            }
            for r in valid_results
        ],
        'efficiency_vs_scalability': [
            {
                'geography': r['geography_level'],
                'efficiency': r.get('cost_efficiency_score', 0),
                'scalability': r.get('scalability_score', 0),
                'balanced_score': (r.get('cost_efficiency_score', 0) + r.get('scalability_score', 0)) / 2
            }
            for r in valid_results
        ]
    }

    return insights


def _analyze_budget_impact(geography_results: List[Dict[str, Any]], base_budget: float) -> Dict[str, Any]:
    """Analyze budget impact across geography scenarios."""
    valid_results = [r for r in geography_results if 'error' not in r]

    budget_analysis = {
        'budget_constraint': base_budget,
        'feasible_geographies': [],
        'infeasible_geographies': [],
        'high_risk_geographies': [],
        'budget_efficiency': {},
        'budget_optimization_opportunities': []
    }

    for result in valid_results:
        cost = result['estimated_cost']
        utilization = result.get('budget_utilization', 0)

        if cost <= base_budget:
            budget_analysis['feasible_geographies'].append({
                'geography': result['geography_level'],
                'cost': cost,
                'utilization': utilization,
                'remaining_budget': base_budget - cost
            })

            if utilization > 80:
                budget_analysis['high_risk_geographies'].append(result['geography_level'])
        else:
            budget_analysis['infeasible_geographies'].append({
                'geography': result['geography_level'],
                'cost': cost,
                'overspend_amount': cost - base_budget,
                'overspend_percentage': ((cost - base_budget) / base_budget) * 100
            })

    # Budget efficiency analysis
    if budget_analysis['feasible_geographies']:
        efficiencies = []
        for geo in budget_analysis['feasible_geographies']:
            # Efficiency = data potential per dollar spent
            efficiency = geo['remaining_budget'] / base_budget  # Higher remaining budget = lower efficiency
            efficiencies.append((geo['geography'], efficiency))

        budget_analysis['budget_efficiency'] = {
            'most_efficient': max(efficiencies, key=lambda x: x[1]),
            'least_efficient': min(efficiencies, key=lambda x: x[1])
        }

    # Budget optimization opportunities
    if budget_analysis['infeasible_geographies']:
        budget_analysis['budget_optimization_opportunities'].append({
            'type': 'budget_increase',
            'description': f"Consider budget increase of ${max(r['overspend_amount'] for r in budget_analysis['infeasible_geographies']):.2f} to enable {len(budget_analysis['infeasible_geographies'])} geographies",
            'estimated_cost': max(r['overspend_amount'] for r in budget_analysis['infeasible_geographies'])
        })

    if len(budget_analysis['high_risk_geographies']) > 0:
        budget_analysis['budget_optimization_opportunities'].append({
            'type': 'phased_execution',
            'description': f"Implement phased execution for {len(budget_analysis['high_risk_geographies'])} high-utilization geographies",
            'estimated_savings': 'Variable - depends on phasing strategy'
        })

    return budget_analysis


def _generate_cost_recommendations(
    geography_results: List[Dict[str, Any]],
    base_budget: Optional[float],
    risk_tolerance: str,
    time_sensitivity: str,
    quality_requirement: str
) -> Dict[str, Any]:
    """Generate comprehensive cost-based recommendations."""
    valid_results = [r for r in geography_results if 'error' not in r]

    recommendations = {
        'primary_recommendation': {},
        'alternative_recommendations': [],
        'cost_optimization_recommendations': [],
        'risk_based_recommendations': [],
        'implementation_priorities': [],
        'contingency_planning': []
    }

    if not valid_results:
        recommendations['primary_recommendation'] = {
            'geography': None,
            'reasoning': 'No valid geography analyses available',
            'confidence': 0
        }
        return recommendations

    # Calculate recommendation scores based on multiple factors
    scored_results = []
    for result in valid_results:
        score = _calculate_geography_recommendation_score(
            result, base_budget, risk_tolerance, time_sensitivity, quality_requirement
        )
        scored_results.append((result, score))

    # Sort by recommendation score
    scored_results.sort(key=lambda x: x[1], reverse=True)

    # Primary recommendation
    primary_result, primary_score = scored_results[0]
    recommendations['primary_recommendation'] = {
        'geography': primary_result['geography_level'],
        'estimated_cost': primary_result['estimated_cost'],
        'confidence_score': primary_score,
        'reasoning': _generate_recommendation_reasoning(primary_result, base_budget, risk_tolerance),
        'expected_benefits': _generate_expected_benefits(primary_result),
        'potential_risks': _generate_potential_risks(primary_result, base_budget)
    }

    # Alternative recommendations (top 3)
    recommendations['alternative_recommendations'] = [
        {
            'geography': result['geography_level'],
            'estimated_cost': result['estimated_cost'],
            'confidence_score': score,
            'trade_off_vs_primary': _calculate_trade_off_vs_primary(result, primary_result)
        }
        for result, score in scored_results[1:4]  # Top 3 alternatives
    ]

    # Cost optimization recommendations
    cost_recommendations = _generate_cost_optimization_recommendations(valid_results, base_budget)
    recommendations['cost_optimization_recommendations'] = cost_recommendations

    # Risk-based recommendations
    risk_recommendations = _generate_risk_based_recommendations(
        valid_results, risk_tolerance, base_budget
    )
    recommendations['risk_based_recommendations'] = risk_recommendations

    # Implementation priorities
    recommendations['implementation_priorities'] = _generate_implementation_priorities(scored_results)

    # Contingency planning
    recommendations['contingency_planning'] = _generate_contingency_planning(
        valid_results, base_budget, risk_tolerance
    )

    return recommendations


def _calculate_geography_recommendation_score(
    result: Dict[str, Any],
    base_budget: Optional[float],
    risk_tolerance: str,
    time_sensitivity: str,
    quality_requirement: str
) -> float:
    """Calculate recommendation score for a geography result."""
    score = 0.0

    # Cost factor (lower cost = higher score, but within budget)
    cost = result['estimated_cost']
    if base_budget and cost <= base_budget:
        cost_score = (base_budget - cost) / base_budget  # Higher remaining budget = higher score
    elif base_budget:
        cost_score = - (cost - base_budget) / base_budget  # Penalty for overspend
    else:
        cost_score = 1.0 / max(cost, 1)  # Basic cost efficiency
    score += cost_score * 0.3

    # Efficiency factor
    efficiency_score = result.get('cost_efficiency_score', 0.5)
    score += efficiency_score * 0.25

    # Scalability factor
    scalability_score = result.get('scalability_score', 0.5)
    score += scalability_score * 0.15

    # Data density factor
    density_score = result.get('data_density_score', 0.5)
    score += density_score * 0.15

    # Risk tolerance adjustment
    if risk_tolerance == "low":
        # Prefer smaller, more controlled geographies
        geography_risk_adjustment = {
            "city": 0.2,
            "county": 0.1,
            "state": -0.1,
            "national": -0.2
        }.get(result['geography_level'], 0)
    elif risk_tolerance == "high":
        # Can handle larger, more complex geographies
        geography_risk_adjustment = {
            "city": -0.1,
            "county": 0,
            "state": 0.1,
            "national": 0.2
        }.get(result['geography_level'], 0)
    else:
        geography_risk_adjustment = 0

    score += geography_risk_adjustment * 0.15

    return max(0.0, min(1.0, score))


def _generate_recommendation_reasoning(
    result: Dict[str, Any],
    base_budget: Optional[float],
    risk_tolerance: str
) -> str:
    """Generate reasoning for geography recommendation."""
    reasons = []

    # Cost reasoning
    if base_budget:
        utilization = result.get('budget_utilization', 0)
        if utilization <= 80:
            reasons.append(f"efficient budget utilization ({utilization:.1f}%)")
        elif utilization <= 100:
            reasons.append(f"acceptable budget utilization ({utilization:.1f}%)")
        else:
            reasons.append(f"requires budget adjustment for {utilization:.1f}% utilization")

    # Efficiency reasoning
    efficiency = result.get('cost_efficiency_score', 0)
    if efficiency > 0.7:
        reasons.append("high cost efficiency")
    elif efficiency > 0.5:
        reasons.append("good cost efficiency")
    else:
        reasons.append("acceptable cost efficiency")

    # Scale reasoning
    scalability = result.get('scalability_score', 0)
    if scalability > 0.7:
        reasons.append("highly scalable")
    elif scalability > 0.5:
        reasons.append("moderately scalable")

    # Risk reasoning
    if risk_tolerance == "low" and result['geography_level'] in ["city", "county"]:
        reasons.append("appropriate for low risk tolerance")
    elif risk_tolerance == "high" and result['geography_level'] in ["state", "national"]:
        reasons.append("leverages high risk tolerance for broader coverage")

    return "; ".join(reasons)


def _generate_expected_benefits(result: Dict[str, Any]) -> List[str]:
    """Generate expected benefits for geography selection."""
    benefits = []

    # Cost benefits
    if result.get('optimization_savings', 0) > 0:
        benefits.append(f"${result['optimization_savings']:.2f} in potential cost savings")

    # Efficiency benefits
    efficiency = result.get('cost_efficiency_score', 0)
    if efficiency > 0.7:
        benefits.append("High operational efficiency")
    elif efficiency > 0.5:
        benefits.append("Good operational efficiency")

    # Data benefits
    density = result.get('data_density_score', 0)
    if density > 0.8:
        benefits.append("High data density and coverage")
    elif density > 0.6:
        benefits.append("Good data density and coverage")

    # Scalability benefits
    scalability = result.get('scalability_score', 0)
    if scalability > 0.7:
        benefits.append("Excellent scalability for future expansion")
    elif scalability > 0.5:
        benefits.append("Good scalability characteristics")

    return benefits


def _generate_potential_risks(result: Dict[str, Any], base_budget: Optional[float]) -> List[str]:
    """Generate potential risks for geography selection."""
    risks = []

    # Budget risks
    if base_budget:
        utilization = result.get('budget_utilization', 0)
        if utilization > 100:
            risks.append(f"Budget overspend of ${(result['estimated_cost'] - base_budget):.2f}")
        elif utilization > 90:
            risks.append("High budget utilization increases financial risk")

    # Cost uncertainty risks
    confidence = result.get('cost_confidence', 1.0)
    if confidence < 0.7:
        risks.append(f"Cost uncertainty (±{((1-confidence)*50):.1f}%) may affect budgeting")

    # Scalability risks
    scalability = result.get('scalability_score', 1.0)
    if scalability < 0.5:
        risks.append("Low scalability may limit future expansion")

    # Geography-specific risks
    if result['geography_level'] == "national":
        risks.append("National scope increases complexity and compliance requirements")
    elif result['geography_level'] == "city":
        risks.append("City scope may have limited data availability")

    return risks


def _calculate_trade_off_vs_primary(result: Dict[str, Any], primary_result: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate trade-offs compared to primary recommendation."""
    trade_off = {
        'cost_difference': result['estimated_cost'] - primary_result['estimated_cost'],
        'cost_difference_percentage': ((result['estimated_cost'] - primary_result['estimated_cost']) / primary_result['estimated_cost']) * 100 if primary_result['estimated_cost'] > 0 else 0,
        'efficiency_vs_primary': result.get('cost_efficiency_score', 0) - primary_result.get('cost_efficiency_score', 0),
        'scalability_vs_primary': result.get('scalability_score', 0) - primary_result.get('scalability_score', 0),
        'density_vs_primary': result.get('data_density_score', 0) - primary_result.get('data_density_score', 0)
    }

    # Determine if this is a meaningful alternative
    significant_differences = sum(1 for diff in [
        trade_off['cost_difference_percentage'],
        trade_off['efficiency_vs_primary'],
        trade_off['scalability_vs_primary']
    ] if abs(diff) > 0.1)  # 10% or 0.1 difference threshold

    trade_off['significant_alternative'] = significant_differences >= 1

    return trade_off


def _generate_cost_optimization_recommendations(
    valid_results: List[Dict[str, Any]],
    base_budget: Optional[float]
) -> List[Dict[str, Any]]:
    """Generate cost optimization recommendations."""
    recommendations = []

    if len(valid_results) < 2:
        return recommendations

    # Compare geographies for optimization opportunities
    costs = [(r['geography_level'], r['estimated_cost']) for r in valid_results]
    min_cost_geo, min_cost = min(costs, key=lambda x: x[1])
    max_cost_geo, max_cost = max(costs, key=lambda x: x[1])

    savings_opportunity = max_cost - min_cost
    if savings_opportunity > 500:  # Significant savings
        recommendations.append({
            'type': 'geography_switch',
            'title': 'Geography Optimization',
            'description': f"Switch from {max_cost_geo} to {min_cost_geo} for ${savings_opportunity:.2f} savings",
            'savings_amount': savings_opportunity,
            'savings_percentage': (savings_opportunity / max_cost) * 100,
            'implementation_effort': 'medium'
        })

    # Budget optimization
    if base_budget:
        over_budget = [r for r in valid_results if r['estimated_cost'] > base_budget]
        if over_budget:
            total_overspend = sum(r['estimated_cost'] - base_budget for r in over_budget)
            recommendations.append({
                'type': 'budget_adjustment',
                'title': 'Budget Adjustment Required',
                'description': f"{len(over_budget)} geographies exceed budget by total ${total_overspend:.2f}",
                'required_budget_increase': total_overspend,
                'implementation_effort': 'high'
            })

    # Efficiency optimization
    low_efficiency = [r for r in valid_results if r.get('cost_efficiency_score', 1.0) < 0.6]
    if low_efficiency:
        recommendations.append({
            'type': 'efficiency_improvement',
            'title': 'Efficiency Optimization',
            'description': f"Improve efficiency for {len(low_efficiency)} geographies through signal optimization",
            'affected_geographies': [r['geography_level'] for r in low_efficiency],
            'implementation_effort': 'medium'
        })

    return recommendations


def _generate_risk_based_recommendations(
    valid_results: List[Dict[str, Any]],
    risk_tolerance: str,
    base_budget: Optional[float]
) -> List[Dict[str, Any]]:
    """Generate risk-based recommendations."""
    recommendations = []

    # Risk tolerance-based recommendations
    if risk_tolerance == "low":
        # Recommend smaller, more controlled geographies
        safe_geographies = [r for r in valid_results if r['geography_level'] in ["city", "county"]]
        if safe_geographies:
            recommendations.append({
                'type': 'risk_mitigation',
                'title': 'Conservative Geography Selection',
                'description': f"Prioritize {len(safe_geographies)} smaller geographies for risk control",
                'recommended_geographies': [r['geography_level'] for r in safe_geographies],
                'risk_benefit': 'Lower operational risk and complexity'
            })

    elif risk_tolerance == "high":
        # Can recommend larger, more comprehensive geographies
        comprehensive_geographies = [r for r in valid_results if r['geography_level'] in ["state", "national"]]
        if comprehensive_geographies:
            recommendations.append({
                'type': 'risk_leverage',
                'title': 'Comprehensive Coverage Opportunity',
                'description': f"Leverage {len(comprehensive_geographies)} larger geographies for broader data coverage",
                'recommended_geographies': [r['geography_level'] for r in comprehensive_geographies],
                'risk_benefit': 'Higher data coverage with managed risk'
            })

    # Budget risk recommendations
    if base_budget:
        high_risk_budget = [r for r in valid_results if r.get('budget_utilization', 0) > 95]
        if high_risk_budget:
            recommendations.append({
                'type': 'budget_risk_mitigation',
                'title': 'Budget Risk Mitigation',
                'description': f"Implement monitoring for {len(high_risk_budget)} high-budget-utilization geographies",
                'affected_geographies': [r['geography_level'] for r in high_risk_budget],
                'risk_benefit': 'Reduced financial risk exposure'
            })

    return recommendations


def _generate_implementation_priorities(scored_results: List[Tuple[Dict[str, Any], float]]) -> List[Dict[str, Any]]:
    """Generate implementation priority recommendations."""
    priorities = []

    # Sort by implementation feasibility and impact
    for result, score in scored_results[:5]:  # Top 5
        priority_level = "high" if score > 0.8 else "medium" if score > 0.6 else "low"

        priorities.append({
            'geography': result['geography_level'],
            'priority_level': priority_level,
            'implementation_order': len(priorities) + 1,
            'estimated_cost': result['estimated_cost'],
            'estimated_effort': _estimate_implementation_effort(result),
            'expected_roi': _estimate_expected_roi(result, score)
        })

    return priorities


def _generate_contingency_planning(
    valid_results: List[Dict[str, Any]],
    base_budget: Optional[float],
    risk_tolerance: str
) -> List[Dict[str, Any]]:
    """Generate contingency planning recommendations."""
    contingencies = []

    # Cost contingency
    if base_budget:
        contingency_buffer = base_budget * 0.1  # 10% buffer
        contingencies.append({
            'type': 'cost_contingency',
            'title': 'Cost Overrun Contingency',
            'description': f"Maintain ${contingency_buffer:.2f} contingency budget for unexpected costs",
            'trigger_conditions': ['Cost exceeds estimate by 15%', 'New requirements discovered'],
            'response_actions': ['Pause execution', 'Renegotiate budget', 'Switch to lower-cost geography']
        })

    # Performance contingency
    contingencies.append({
        'type': 'performance_contingency',
        'title': 'Performance Degradation Contingency',
        'description': 'Monitor execution performance and implement fallback strategies',
        'trigger_conditions': ['Success rate below 70%', 'Execution time exceeds 150% of estimate'],
        'response_actions': ['Reduce scope', 'Switch sources', 'Implement parallel processing']
    })

    # Data quality contingency
    contingencies.append({
        'type': 'quality_contingency',
        'title': 'Data Quality Contingency',
        'description': 'Ensure data quality meets requirements with backup strategies',
        'trigger_conditions': ['Data completeness below 80%', 'Data accuracy issues discovered'],
        'response_actions': ['Switch to higher quality sources', 'Implement additional validation', 'Extend execution time']
    })

    # Risk-specific contingencies
    if risk_tolerance == "low":
        contingencies.append({
            'type': 'compliance_contingency',
            'title': 'Compliance Violation Contingency',
            'description': 'Immediate response to compliance issues',
            'trigger_conditions': ['Compliance violation detected', 'Regulatory change impact'],
            'response_actions': ['Stop execution', 'Legal review', 'Data deletion']
        })

    return contingencies


def _estimate_implementation_effort(result: Dict[str, Any]) -> str:
    """Estimate implementation effort for a geography."""
    effort_score = 0

    # Geography complexity
    geography_complexity = {
        "city": 1,
        "county": 2,
        "state": 3,
        "national": 4
    }.get(result['geography_level'], 2)

    effort_score += geography_complexity

    # Cost complexity
    if result['estimated_cost'] > 5000:
        effort_score += 2
    elif result['estimated_cost'] > 2000:
        effort_score += 1

    # Confidence adjustment
    confidence = result.get('cost_confidence', 0.8)
    if confidence < 0.7:
        effort_score += 1

    if effort_score <= 2:
        return "low"
    elif effort_score <= 4:
        return "medium"
    else:
        return "high"


def _estimate_expected_roi(result: Dict[str, Any], score: float) -> str:
    """Estimate expected ROI for geography implementation."""
    # Simplified ROI estimation based on score and cost
    cost = result['estimated_cost']

    if score > 0.8 and cost < 2000:
        return "high"
    elif score > 0.7 and cost < 5000:
        return "medium"
    elif score > 0.6:
        return "low"
    else:
        return "negative"


def _assess_cost_risks(
    geography_results: List[Dict[str, Any]],
    risk_tolerance: str,
    time_sensitivity: str,
    quality_requirement: str
) -> Dict[str, Any]:
    """Assess overall cost risks across geography scenarios."""
    risk_assessment = {
        'overall_risk_level': 'low',
        'cost_uncertainty_risk': 'low',
        'budget_overrun_risk': 'low',
        'execution_risk': 'low',
        'compliance_risk': 'low',
        'risk_factors': [],
        'mitigation_strategies': [],
        'contingency_budget_required': 0
    }

    valid_results = [r for r in geography_results if 'error' not in r]

    if not valid_results:
        risk_assessment['overall_risk_level'] = 'unknown'
        return risk_assessment

    # Cost uncertainty risk
    low_confidence_count = sum(1 for r in valid_results if r.get('cost_confidence', 1.0) < 0.7)
    if low_confidence_count > len(valid_results) * 0.5:
        risk_assessment['cost_uncertainty_risk'] = 'high'
        risk_assessment['risk_factors'].append('High cost uncertainty across geographies')
        risk_assessment['mitigation_strategies'].append('Implement detailed cost monitoring')

    # Budget overrun risk
    budget_overruns = [r for r in valid_results if r.get('budget_compliance') == 'exceeded']
    if budget_overruns:
        risk_assessment['budget_overrun_risk'] = 'high' if len(budget_overruns) > len(valid_results) * 0.3 else 'medium'
        risk_assessment['risk_factors'].append(f"{len(budget_overruns)} geographies exceed budget")
        risk_assessment['contingency_budget_required'] = max(r['estimated_cost'] for r in budget_overruns)

    # Execution risk based on time sensitivity
    if time_sensitivity == "critical":
        risk_assessment['execution_risk'] = 'high'
        risk_assessment['risk_factors'].append('Critical time sensitivity increases execution risk')
        risk_assessment['mitigation_strategies'].append('Implement parallel processing and monitoring')

    # Compliance risk based on risk tolerance
    if risk_tolerance == "low":
        risk_assessment['compliance_risk'] = 'low'
        risk_assessment['mitigation_strategies'].append('Conservative compliance approach minimizes risk')
    elif risk_tolerance == "high":
        risk_assessment['compliance_risk'] = 'medium'
        risk_assessment['risk_factors'].append('High risk tolerance requires enhanced compliance monitoring')

    # Overall risk level determination
    risk_levels = ['low', 'medium', 'high']
    risk_scores = [
        risk_levels.index(risk_assessment['cost_uncertainty_risk']),
        risk_levels.index(risk_assessment['budget_overrun_risk']),
        risk_levels.index(risk_assessment['execution_risk']),
        risk_levels.index(risk_assessment['compliance_risk'])
    ]

    overall_risk_index = sum(risk_scores) / len(risk_scores)
    risk_assessment['overall_risk_level'] = risk_levels[min(int(overall_risk_index), 2)]

    return risk_assessment


def _generate_performance_projections(
    geography_results: List[Dict[str, Any]],
    signals: List[SignalType],
    browser_pages: int
) -> Dict[str, Any]:
    """Generate performance projections for geography scenarios."""
    projections = {
        'execution_time_estimates': {},
        'success_rate_projections': {},
        'resource_utilization_forecast': {},
        'data_quality_expectations': {},
        'scalability_projections': {}
    }

    for result in geography_results:
        if 'error' in result:
            continue

        geo_level = result['geography_level']

        # Execution time estimates
        base_time_hours = _estimate_execution_time_hours(geo_level, len(signals), browser_pages)
        projections['execution_time_estimates'][geo_level] = {
            'estimated_hours': base_time_hours,
            'confidence_range': (base_time_hours * 0.8, base_time_hours * 1.2),
            'optimistic_scenario': base_time_hours * 0.7,
            'pessimistic_scenario': base_time_hours * 1.5
        }

        # Success rate projections
        base_success_rate = _estimate_success_rate(geo_level, result.get('cost_confidence', 0.8))
        projections['success_rate_projections'][geo_level] = {
            'projected_rate': base_success_rate,
            'confidence_interval': (max(0, base_success_rate - 0.1), min(1.0, base_success_rate + 0.1)),
            'risk_factors': _identify_success_risk_factors(result)
        }

        # Resource utilization forecast
        projections['resource_utilization_forecast'][geo_level] = {
            'cpu_hours': base_time_hours * 2,  # Rough estimate
            'memory_peak_gb': _estimate_memory_usage(geo_level, len(signals)),
            'network_usage_gb': _estimate_network_usage(geo_level, len(signals), browser_pages),
            'cost_per_hour': result['estimated_cost'] / max(base_time_hours, 1)
        }

        # Data quality expectations
        projections['data_quality_expectations'][geo_level] = {
            'expected_completeness': _estimate_data_completeness(geo_level, signals),
            'expected_accuracy': _estimate_data_accuracy(geo_level, signals),
            'quality_confidence': result.get('cost_confidence', 0.8),
            'validation_recommendations': _generate_validation_recommendations(result)
        }

        # Scalability projections
        projections['scalability_projections'][geo_level] = {
            'expansion_potential': result.get('scalability_score', 0.5) * 100,
            'parallelization_efficiency': _estimate_parallelization_efficiency(geo_level),
            'future_growth_capacity': _estimate_growth_capacity(result)
        }

    return projections


def _estimate_execution_time_hours(geography_level: str, signal_count: int, browser_pages: int) -> float:
    """Estimate execution time in hours."""
    base_times = {
        "city": 1.5,
        "county": 4.0,
        "state": 12.0,
        "national": 48.0
    }

    base_time = base_times.get(geography_level, 6.0)
    signal_multiplier = 1 + (signal_count - 1) * 0.2
    browser_multiplier = 1 + (browser_pages / 20) * 0.3

    return base_time * signal_multiplier * browser_multiplier


def _estimate_success_rate(geography_level: str, confidence_score: float) -> float:
    """Estimate success rate based on geography and confidence."""
    base_rates = {
        "city": 0.88,
        "county": 0.85,
        "state": 0.80,
        "national": 0.75
    }

    base_rate = base_rates.get(geography_level, 0.82)
    confidence_adjustment = (confidence_score - 0.5) * 0.1  # ±10% adjustment

    return max(0.5, min(0.95, base_rate + confidence_adjustment))


def _identify_success_risk_factors(result: Dict[str, Any]) -> List[str]:
    """Identify risk factors that could affect success rate."""
    risks = []

    if result.get('cost_confidence', 1.0) < 0.7:
        risks.append('Low cost confidence may indicate execution challenges')

    if result['geography_level'] == 'national':
        risks.append('National scope increases complexity and potential failure points')

    if result.get('scalability_score', 1.0) < 0.6:
        risks.append('Low scalability may limit execution flexibility')

    if result.get('estimated_cost', 0) > 10000:
        risks.append('High cost operations may face resource constraints')

    return risks


def _estimate_memory_usage(geography_level: str, signal_count: int) -> float:
    """Estimate peak memory usage in GB."""
    base_memory = {
        "city": 2,
        "county": 4,
        "state": 8,
        "national": 16
    }

    base_gb = base_memory.get(geography_level, 4)
    signal_multiplier = 1 + (signal_count - 1) * 0.15

    return base_gb * signal_multiplier


def _estimate_network_usage(geography_level: str, signal_count: int, browser_pages: int) -> float:
    """Estimate network usage in GB."""
    base_network = {
        "city": 1,
        "county": 5,
        "state": 20,
        "national": 100
    }

    base_gb = base_network.get(geography_level, 10)
    signal_multiplier = signal_count * 0.8
    browser_multiplier = 1 + (browser_pages / 10) * 0.5

    return base_gb * signal_multiplier * browser_multiplier


def _estimate_data_completeness(geography_level: str, signals: List[SignalType]) -> float:
    """Estimate expected data completeness percentage."""
    base_completeness = {
        "city": 0.92,
        "county": 0.88,
        "state": 0.82,
        "national": 0.75
    }

    base_rate = base_completeness.get(geography_level, 0.85)

    # Adjust based on signal types
    high_completeness_signals = [SignalType.LIEN, SignalType.MORTGAGE]
    completeness_signals = sum(1 for s in signals if s in high_completeness_signals)
    signal_bonus = (completeness_signals / len(signals)) * 0.05

    return min(0.95, base_rate + signal_bonus)


def _estimate_data_accuracy(geography_level: str, signals: List[SignalType]) -> float:
    """Estimate expected data accuracy percentage."""
    base_accuracy = {
        "city": 0.94,
        "county": 0.91,
        "state": 0.87,
        "national": 0.82
    }

    base_rate = base_accuracy.get(geography_level, 0.89)

    # Adjust based on signal types
    high_accuracy_signals = [SignalType.COURT_CASE, SignalType.DEED]
    accuracy_signals = sum(1 for s in signals if s in high_accuracy_signals)
    signal_bonus = (accuracy_signals / len(signals)) * 0.03

    return min(0.97, base_rate + signal_bonus)


def _generate_validation_recommendations(result: Dict[str, Any]) -> List[str]:
    """Generate data validation recommendations."""
    recommendations = []

    if result.get('cost_confidence', 1.0) < 0.8:
        recommendations.append('Implement additional data validation due to cost uncertainty')

    if result['geography_level'] in ['state', 'national']:
        recommendations.append('Consider sample validation for large geography scopes')

    if result.get('data_density_score', 0) < 0.7:
        recommendations.append('Enhanced validation recommended for lower data density')

    recommendations.append('Standard validation protocols recommended')

    return recommendations


def _estimate_parallelization_efficiency(geography_level: str) -> float:
    """Estimate parallelization efficiency (0-1)."""
    # Smaller geographies are easier to parallelize
    efficiencies = {
        "city": 0.85,
        "county": 0.75,
        "state": 0.65,
        "national": 0.50
    }

    return efficiencies.get(geography_level, 0.70)


def _estimate_growth_capacity(result: Dict[str, Any]) -> str:
    """Estimate future growth capacity."""
    scalability = result.get('scalability_score', 0.5)

    if scalability > 0.8:
        return "high"
    elif scalability > 0.6:
        return "medium"
    else:
        return "limited"


def _calculate_recommendation_scores(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate final recommendation scores for all geographies."""
    geography_results = analysis['geography_analysis']
    valid_results = [r for r in geography_results if 'error' not in r]

    # Calculate recommendation scores based on multiple factors
    for result in valid_results:
        # Base score from individual metrics
        base_score = (
            result.get('cost_efficiency_score', 0) * 0.3 +
            result.get('scalability_score', 0) * 0.25 +
            result.get('data_density_score', 0) * 0.25 +
            result.get('cost_confidence', 0.8) * 0.2
        )

        # Adjust for budget compliance
        budget_multiplier = 1.0
        if result.get('budget_compliance') == 'compliant':
            budget_multiplier = 1.0
        elif result.get('budget_compliance') == 'high_utilization':
            budget_multiplier = 0.9
        elif result.get('budget_compliance') == 'exceeded':
            budget_multiplier = 0.7

        # Adjust for optimization opportunities
        optimization_multiplier = 1.0
        if result.get('optimization_available'):
            optimization_multiplier = 1.1

        result['recommendation_score'] = min(1.0, base_score * budget_multiplier * optimization_multiplier)

    return analysis


# Enhanced what-if cost analysis function (replacement for the original)
async def what_if_cost(
    signals: Optional[List[SignalType]] = None,
    geography_levels: Optional[List[str]] = None,
    browser_pages: int = 0,
    base_budget: Optional[float] = None,
    risk_tolerance: str = "medium",
    time_sensitivity: str = "normal",
    quality_requirement: str = "standard",
    include_optimizations: bool = True,
    include_sensitivity: bool = True,
    include_comparative: bool = True
) -> Dict[str, Any]:
    """
    Enhanced what-if cost analysis for geography and operational scenarios.

    This is a comprehensive replacement for the original what_if_cost function,
    providing detailed cost analysis, optimization recommendations, budget impact
    assessment, sensitivity analysis, and strategic recommendations.

    Args:
        signals: List of signal types to analyze
        geography_levels: List of geography levels to test
        browser_pages: Number of browser pages required
        base_budget: Base budget constraint for analysis
        risk_tolerance: Risk tolerance level ("low", "medium", "high")
        time_sensitivity: Time sensitivity ("low", "normal", "high", "critical")
        quality_requirement: Quality requirement level ("basic", "standard", "verified", "premium")
        include_optimizations: Whether to include cost optimization analysis
        include_sensitivity: Whether to include sensitivity analysis
        include_comparative: Whether to include comparative analysis

    Returns:
        Comprehensive what-if cost analysis with detailed insights and recommendations

    Example:
        analysis = await what_if_cost(
            signals=[SignalType.LIEN, SignalType.MORTGAGE],
            geography_levels=["city", "county", "state"],
            browser_pages=25,
            base_budget=5000.0
        )
        print(f"Recommended geography: {analysis['recommendations']['primary_recommendation']['geography']}")
        print(f"Estimated cost: ${analysis['recommendations']['primary_recommendation']['estimated_cost']:.2f}")
    """
    return await what_if_cost_analysis(
        signals=signals,
        geography_levels=geography_levels,
        browser_pages=browser_pages,
        base_budget=base_budget,
        risk_tolerance=risk_tolerance,
        time_sensitivity=time_sensitivity,
        quality_requirement=quality_requirement,
        include_optimizations=include_optimizations,
        include_sensitivity=include_sensitivity,
        include_comparative=include_comparative
    )


# Global what-if analyzer instance
_global_what_if_analyzer = WhatIfAnalyzer()


# Convenience functions
async def create_what_if_scenario(
    base_control: ScrapeControlContract,
    scenario_type: ScenarioType,
    scenario_name: str,
    modifications: Optional[Dict[str, Any]] = None
) -> ScenarioConfiguration:
    """
    Create a what-if scenario configuration.

    This is the main entry point for creating scenario configurations
    that can be analyzed and compared.

    Args:
        base_control: Base scraping control contract
        scenario_type: Type of scenario (cost optimization, quality enhancement, etc.)
        scenario_name: Human-readable scenario name
        modifications: Optional specific modifications

    Returns:
        Configured scenario ready for analysis

    Example:
        scenario = await create_what_if_scenario(
            control_contract,
            ScenarioType.COST_OPTIMIZATION,
            "Budget-Friendly Approach"
        )
    """
    return await _global_what_if_analyzer.create_scenario(
        base_control, scenario_type, scenario_name, modifications
    )


async def analyze_what_if_scenarios(
    base_control: ScrapeControlContract,
    scenarios: List[ScenarioConfiguration],
    include_sensitivity_analysis: bool = True
) -> WhatIfAnalysis:
    """
    Perform comprehensive what-if analysis comparing multiple scenarios.

    This is the main entry point for comparative scenario analysis,
    providing detailed insights into costs, risks, feasibility, and trade-offs.

    Args:
        base_control: Base scraping control contract
        scenarios: List of alternative scenarios to analyze
        include_sensitivity_analysis: Whether to include parameter sensitivity analysis

    Returns:
        Complete what-if analysis with comparative insights and recommendations

    Example:
        analysis = await analyze_what_if_scenarios(
            base_control, [scenario1, scenario2, scenario3]
        )
        print(f"Best overall scenario: {analysis.best_overall_scenario}")
    """
    return await _global_what_if_analyzer.perform_what_if_analysis(
        base_control, scenarios, include_sensitivity_analysis
    )


async def generate_scenario_recommendations(
    analysis: WhatIfAnalysis,
    criteria_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive scenario recommendations from what-if analysis.

    Provides decision support with trade-off analysis, implementation guidance,
    and risk-adjusted recommendations.

    Args:
        analysis: Complete what-if analysis
        criteria_weights: Optional custom weights for decision criteria

    Returns:
        Comprehensive recommendation package with implementation guidance

    Example:
        recommendations = await generate_scenario_recommendations(analysis)
        print(f"Recommended scenario: {recommendations['primary_recommendation']}")
    """
    return await _global_what_if_analyzer.generate_scenario_recommendations(
        analysis, criteria_weights
    )


def get_what_if_analyzer_stats() -> Dict[str, Any]:
    """
    Get comprehensive what-if analyzer performance statistics.

    Returns operational metrics for monitoring analysis performance,
    scenario coverage, and system health.

    Returns:
        Dict with analyzer statistics and performance indicators
    """
    return _global_what_if_analyzer.get_analyzer_stats()
