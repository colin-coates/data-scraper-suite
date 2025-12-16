# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use as is strictly prohibited.

"""
Enhanced Execution Mode Classifier for MJ Data Scraper Suite

Sophisticated execution mode classification that determines optimal scraping
strategies based on asset types, scope, risk levels, budget constraints,
and intelligence requirements.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field

from core.models.asset_signal import AssetType, SignalType
from core.intent_classifier import IntentRiskLevel, IntentCategory
from core.mapping.asset_signal_map import (
    get_optimal_sources_for_signal,
    calculate_signal_cost_estimate,
    get_data_freshness_requirement,
    SIGNAL_COST_WEIGHT,
    SOURCE_RELIABILITY
)
from core.control_models import ScrapeControlContract, ScrapeBudget, ScrapeIntent

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution mode classifications for scraping operations."""

    # Precision Operations (High accuracy, low volume)
    PRECISION_SEARCH = "precision_search"              # Single target, maximum accuracy
    TARGETED_LOOKUP = "targeted_lookup"                # Few targets, high precision
    VERIFICATION_SCAN = "verification_scan"           # Confirm existing data

    # Discovery Operations (Balanced exploration)
    FOCUSED_DISCOVERY = "focused_discovery"           # Targeted exploration
    CONTROLLED_EXPLORATION = "controlled_exploration" # Guided discovery
    INTELLIGENT_SWEEP = "intelligent_sweep"          # Smart breadth-first search

    # Broad Operations (High volume, lower precision)
    DISCOVERY_SCRAPE = "discovery_scrape"            # Broad data collection
    COMPREHENSIVE_SURVEY = "comprehensive_survey"    # Extensive coverage
    EXHAUSTIVE_ANALYSIS = "exhaustive_analysis"      # Maximum data gathering

    # Specialized Operations
    MONITORING_MODE = "monitoring_mode"              # Continuous monitoring
    VALIDATION_MODE = "validation_mode"             # Data quality validation
    COMPLIANCE_AUDIT = "compliance_audit"           # Regulatory compliance check

    # Emergency Operations
    RAPID_RESPONSE = "rapid_response"               # Time-critical operations
    CRISIS_MODE = "crisis_mode"                     # Emergency data gathering


class ExecutionStrategy(Enum):
    """Execution strategy classifications."""
    DEPTH_FIRST = "depth_first"         # Deep, thorough investigation
    BREADTH_FIRST = "breadth_first"     # Broad, comprehensive coverage
    PRIORITY_BASED = "priority_based"   # Focus on high-value targets
    RISK_WEIGHTED = "risk_weighted"     # Adjust based on risk levels
    COST_OPTIMIZED = "cost_optimized"   # Minimize resource usage
    TIME_OPTIMIZED = "time_optimized"   # Maximize speed
    QUALITY_OPTIMIZED = "quality_optimized" # Maximize data quality


@dataclass
class ExecutionProfile:
    """Complete execution mode profile with strategy and parameters."""
    mode: ExecutionMode
    strategy: ExecutionStrategy
    confidence_score: float
    reasoning: List[str] = field(default_factory=list)
    execution_parameters: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    risk_mitigations: List[str] = field(default_factory=list)
    performance_expectations: Dict[str, Any] = field(default_factory=dict)
    compliance_requirements: List[str] = field(default_factory=list)
    cost_projections: Dict[str, Any] = field(default_factory=dict)

    def get_concurrent_limit(self) -> int:
        """Get recommended concurrent request limit."""
        mode_limits = {
            ExecutionMode.PRECISION_SEARCH: 1,
            ExecutionMode.TARGETED_LOOKUP: 3,
            ExecutionMode.VERIFICATION_SCAN: 5,
            ExecutionMode.FOCUSED_DISCOVERY: 8,
            ExecutionMode.CONTROLLED_EXPLORATION: 10,
            ExecutionMode.INTELLIGENT_SWEEP: 15,
            ExecutionMode.DISCOVERY_SCRAPE: 20,
            ExecutionMode.COMPREHENSIVE_SURVEY: 25,
            ExecutionMode.EXHAUSTIVE_ANALYSIS: 30,
            ExecutionMode.MONITORING_MODE: 5,
            ExecutionMode.VALIDATION_MODE: 8,
            ExecutionMode.COMPLIANCE_AUDIT: 3,
            ExecutionMode.RAPID_RESPONSE: 50,
            ExecutionMode.CRISIS_MODE: 100
        }
        return mode_limits.get(self.mode, 10)

    def get_rate_limit_multiplier(self) -> float:
        """Get rate limit multiplier for this execution mode."""
        mode_multipliers = {
            ExecutionMode.PRECISION_SEARCH: 0.1,      # Very slow, careful
            ExecutionMode.TARGETED_LOOKUP: 0.3,       # Slow, deliberate
            ExecutionMode.VERIFICATION_SCAN: 0.5,     # Moderate pace
            ExecutionMode.FOCUSED_DISCOVERY: 0.7,     # Steady exploration
            ExecutionMode.CONTROLLED_EXPLORATION: 1.0, # Normal pace
            ExecutionMode.INTELLIGENT_SWEEP: 1.2,     # Slightly faster
            ExecutionMode.DISCOVERY_SCRAPE: 1.5,      # Fast discovery
            ExecutionMode.COMPREHENSIVE_SURVEY: 1.8,  # Very fast
            ExecutionMode.EXHAUSTIVE_ANALYSIS: 2.0,   # Maximum speed
            ExecutionMode.MONITORING_MODE: 0.3,       # Slow, continuous
            ExecutionMode.VALIDATION_MODE: 0.5,       # Moderate validation
            ExecutionMode.COMPLIANCE_AUDIT: 0.2,      # Very careful
            ExecutionMode.RAPID_RESPONSE: 3.0,        # Emergency speed
            ExecutionMode.CRISIS_MODE: 5.0            # Maximum emergency speed
        }
        return mode_multipliers.get(self.mode, 1.0)

    def get_monitoring_intensity(self) -> str:
        """Get monitoring intensity level."""
        mode_monitoring = {
            ExecutionMode.PRECISION_SEARCH: "maximum",
            ExecutionMode.TARGETED_LOOKUP: "high",
            ExecutionMode.VERIFICATION_SCAN: "high",
            ExecutionMode.FOCUSED_DISCOVERY: "medium",
            ExecutionMode.CONTROLLED_EXPLORATION: "medium",
            ExecutionMode.INTELLIGENT_SWEEP: "medium",
            ExecutionMode.DISCOVERY_SCRAPE: "standard",
            ExecutionMode.COMPREHENSIVE_SURVEY: "standard",
            ExecutionMode.EXHAUSTIVE_ANALYSIS: "standard",
            ExecutionMode.MONITORING_MODE: "continuous",
            ExecutionMode.VALIDATION_MODE: "detailed",
            ExecutionMode.COMPLIANCE_AUDIT: "maximum",
            ExecutionMode.RAPID_RESPONSE: "intensive",
            ExecutionMode.CRISIS_MODE: "comprehensive"
        }
        return mode_monitoring.get(self.mode, "standard")


class ExecutionModeClassifier:
    """
    Enhanced execution mode classifier with intelligence-driven decision making.

    Analyzes asset types, scope, risk levels, budget constraints, and operational
    requirements to determine optimal execution strategies for scraping operations.
    """

    def __init__(self):
        self.execution_history: Dict[str, List[ExecutionProfile]] = {}
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        self.mode_success_rates: Dict[ExecutionMode, float] = {}
        self.strategy_effectiveness: Dict[ExecutionStrategy, float] = {}

        # Initialize success rates with defaults
        for mode in ExecutionMode:
            self.mode_success_rates[mode] = 0.8  # Default 80% success

        for strategy in ExecutionStrategy:
            self.strategy_effectiveness[strategy] = 0.75  # Default 75% effectiveness

        logger.info("ExecutionModeClassifier initialized with intelligent mode selection")

    async def classify_execution_mode(
        self,
        asset_type: AssetType,
        scope_size: int,
        control: Optional[ScrapeControlContract] = None,
        risk_level: Optional[IntentRiskLevel] = None,
        intent_category: Optional[IntentCategory] = None,
        time_sensitivity: str = "normal",
        data_quality_requirement: str = "standard"
    ) -> ExecutionProfile:
        """
        Classify the optimal execution mode based on comprehensive intelligence.

        Args:
            asset_type: Type of asset being targeted
            scope_size: Number of targets/assets in scope
            control: Optional scraping control contract
            risk_level: Risk classification level
            intent_category: Intent category classification
            time_sensitivity: "low", "normal", "high", "critical"
            data_quality_requirement: "basic", "standard", "verified", "premium"

        Returns:
            Complete ExecutionProfile with mode, strategy, and parameters
        """

        # Base classification using original logic
        base_mode = self._get_base_execution_mode(asset_type, scope_size)

        # Enhance with intelligence factors
        enhanced_mode = await self._enhance_with_intelligence(
            base_mode, asset_type, scope_size, control, risk_level,
            intent_category, time_sensitivity, data_quality_requirement
        )

        # Apply historical performance adjustments
        optimized_mode = self._apply_historical_optimization(enhanced_mode, asset_type, scope_size)

        # Generate comprehensive execution profile
        profile = await self._generate_execution_profile(
            optimized_mode, asset_type, scope_size, control, risk_level,
            intent_category, time_sensitivity, data_quality_requirement
        )

        # Store for learning
        execution_key = f"{asset_type.value}_{scope_size}_{time_sensitivity}_{data_quality_requirement}"
        if execution_key not in self.execution_history:
            self.execution_history[execution_key] = []
        self.execution_history[execution_key].append(profile)

        logger.info(f"✅ Classified execution mode: {profile.mode.value} for {asset_type.value} (scope: {scope_size})")
        return profile

    def _get_base_execution_mode(self, asset_type: AssetType, scope_size: int) -> ExecutionMode:
        """Get base execution mode using enhanced version of original logic."""
        # Original logic with enhancements
        if asset_type == AssetType.PERSON and scope_size == 1:
            return ExecutionMode.PRECISION_SEARCH
        elif scope_size <= 5:
            return ExecutionMode.TARGETED_LOOKUP
        elif scope_size <= 20:
            return ExecutionMode.FOCUSED_DISCOVERY
        elif scope_size <= 100:
            return ExecutionMode.CONTROLLED_EXPLORATION
        elif scope_size <= 1000:
            return ExecutionMode.DISCOVERY_SCRAPE
        elif scope_size <= 10000:
            return ExecutionMode.COMPREHENSIVE_SURVEY
        else:
            return ExecutionMode.EXHAUSTIVE_ANALYSIS

    async def _enhance_with_intelligence(
        self,
        base_mode: ExecutionMode,
        asset_type: AssetType,
        scope_size: int,
        control: Optional[ScrapeControlContract],
        risk_level: Optional[IntentRiskLevel],
        intent_category: Optional[IntentCategory],
        time_sensitivity: str,
        data_quality_requirement: str
    ) -> ExecutionMode:
        """Enhance base mode with intelligence factors."""

        enhanced_mode = base_mode

        # Time sensitivity adjustments
        if time_sensitivity == "critical":
            enhanced_mode = ExecutionMode.CRISIS_MODE
        elif time_sensitivity == "high":
            if base_mode in [ExecutionMode.PRECISION_SEARCH, ExecutionMode.TARGETED_LOOKUP]:
                enhanced_mode = ExecutionMode.RAPID_RESPONSE
            else:
                enhanced_mode = ExecutionMode.INTELLIGENT_SWEEP

        # Risk level adjustments
        if risk_level:
            if risk_level == IntentRiskLevel.CRITICAL:
                if enhanced_mode != ExecutionMode.CRISIS_MODE:
                    enhanced_mode = ExecutionMode.COMPLIANCE_AUDIT
            elif risk_level == IntentRiskLevel.HIGH:
                if enhanced_mode in [ExecutionMode.DISCOVERY_SCRAPE, ExecutionMode.COMPREHENSIVE_SURVEY]:
                    enhanced_mode = ExecutionMode.CONTROLLED_EXPLORATION

        # Data quality requirements
        if data_quality_requirement in ["verified", "premium"]:
            if enhanced_mode in [ExecutionMode.DISCOVERY_SCRAPE, ExecutionMode.COMPREHENSIVE_SURVEY]:
                enhanced_mode = ExecutionMode.VERIFICATION_SCAN
        elif data_quality_requirement == "basic":
            if enhanced_mode in [ExecutionMode.PRECISION_SEARCH, ExecutionMode.TARGETED_LOOKUP]:
                enhanced_mode = ExecutionMode.INTELLIGENT_SWEEP

        # Intent category specific adjustments
        if intent_category:
            if intent_category == IntentCategory.COMPLIANCE:
                enhanced_mode = ExecutionMode.COMPLIANCE_AUDIT
            elif intent_category == IntentCategory.EVENT:
                if scope_size > 50:
                    enhanced_mode = ExecutionMode.MONITORING_MODE
            elif intent_category == IntentCategory.LEGAL:
                if enhanced_mode not in [ExecutionMode.PRECISION_SEARCH, ExecutionMode.COMPLIANCE_AUDIT]:
                    enhanced_mode = ExecutionMode.VERIFICATION_SCAN

        # Asset type specific adjustments
        if asset_type == AssetType.PERSON:
            if intent_category == IntentCategory.PERSONAL:
                enhanced_mode = ExecutionMode.PRECISION_SEARCH
        elif asset_type in [AssetType.COMMERCIAL_PROPERTY, AssetType.APARTMENT_BUILDING]:
            if enhanced_mode == ExecutionMode.DISCOVERY_SCRAPE:
                enhanced_mode = ExecutionMode.CONTROLLED_EXPLORATION

        # Control contract specific intelligence
        if control:
            enhanced_mode = await self._apply_control_intelligence(enhanced_mode, control)

        return enhanced_mode

    async def _apply_control_intelligence(self, current_mode: ExecutionMode, control: ScrapeControlContract) -> ExecutionMode:
        """Apply intelligence from control contract."""

        # Budget-based adjustments
        if control.budget:
            budget_intensity = self._calculate_budget_intensity(control.budget)

            if budget_intensity > 0.8:  # High budget
                if current_mode in [ExecutionMode.PRECISION_SEARCH, ExecutionMode.TARGETED_LOOKUP]:
                    current_mode = ExecutionMode.CONTROLLED_EXPLORATION
            elif budget_intensity < 0.3:  # Low budget
                if current_mode in [ExecutionMode.COMPREHENSIVE_SURVEY, ExecutionMode.EXHAUSTIVE_ANALYSIS]:
                    current_mode = ExecutionMode.FOCUSED_DISCOVERY

        # Geography scope adjustments
        if control.intent.geography:
            geo_scope = len(control.intent.geography)
            if geo_scope > 10:  # Very broad geography
                if current_mode in [ExecutionMode.PRECISION_SEARCH, ExecutionMode.TARGETED_LOOKUP]:
                    current_mode = ExecutionMode.INTELLIGENT_SWEEP

        # Source type adjustments
        if control.intent.sources:
            high_reliability_sources = sum(1 for source in control.intent.sources
                                         if SOURCE_RELIABILITY.get(source, 0) > 0.8)
            if high_reliability_sources / len(control.intent.sources) > 0.7:
                # Mostly high-reliability sources
                if current_mode in [ExecutionMode.DISCOVERY_SCRAPE, ExecutionMode.COMPREHENSIVE_SURVEY]:
                    current_mode = ExecutionMode.CONTROLLED_EXPLORATION

        return current_mode

    def _calculate_budget_intensity(self, budget: ScrapeBudget) -> float:
        """Calculate budget intensity score (0-1)."""
        if not budget:
            return 0.5

        # Normalize different budget dimensions
        time_intensity = min(budget.max_runtime_minutes / 480, 1.0)  # Max 8 hours = 1.0
        page_intensity = min(budget.max_pages / 2000, 1.0)          # Max 2000 pages = 1.0
        record_intensity = min(budget.max_records / 50000, 1.0)     # Max 50k records = 1.0

        # Weighted average
        return (time_intensity * 0.4 + page_intensity * 0.3 + record_intensity * 0.3)

    def _apply_historical_optimization(self, mode: ExecutionMode, asset_type: AssetType, scope_size: int) -> ExecutionMode:
        """Apply historical performance optimization."""
        execution_key = f"{asset_type.value}_{scope_size}"

        if execution_key in self.performance_metrics:
            metrics = self.performance_metrics[execution_key]

            # Check if current mode has poor performance
            current_success = self.mode_success_rates.get(mode, 0.8)
            if current_success < 0.7:
                # Try to find a better performing alternative
                alternative_modes = self._get_alternative_modes(mode, asset_type, scope_size)
                for alt_mode in alternative_modes:
                    alt_success = self.mode_success_rates.get(alt_mode, 0.8)
                    if alt_success > current_success + 0.1:  # At least 10% better
                        logger.info(f"Optimizing execution mode from {mode.value} to {alt_mode.value} based on historical performance")
                        return alt_mode

        return mode

    def _get_alternative_modes(self, current_mode: ExecutionMode, asset_type: AssetType, scope_size: int) -> List[ExecutionMode]:
        """Get alternative execution modes for optimization."""
        # Define mode alternatives based on scope and type
        alternatives = {
            ExecutionMode.PRECISION_SEARCH: [ExecutionMode.TARGETED_LOOKUP, ExecutionMode.VERIFICATION_SCAN],
            ExecutionMode.TARGETED_LOOKUP: [ExecutionMode.PRECISION_SEARCH, ExecutionMode.FOCUSED_DISCOVERY],
            ExecutionMode.FOCUSED_DISCOVERY: [ExecutionMode.TARGETED_LOOKUP, ExecutionMode.CONTROLLED_EXPLORATION],
            ExecutionMode.CONTROLLED_EXPLORATION: [ExecutionMode.FOCUSED_DISCOVERY, ExecutionMode.INTELLIGENT_SWEEP],
            ExecutionMode.INTELLIGENT_SWEEP: [ExecutionMode.CONTROLLED_EXPLORATION, ExecutionMode.DISCOVERY_SCRAPE],
            ExecutionMode.DISCOVERY_SCRAPE: [ExecutionMode.INTELLIGENT_SWEEP, ExecutionMode.COMPREHENSIVE_SURVEY],
            ExecutionMode.COMPREHENSIVE_SURVEY: [ExecutionMode.DISCOVERY_SCRAPE, ExecutionMode.EXHAUSTIVE_ANALYSIS],
        }

        return alternatives.get(current_mode, [current_mode])

    async def _generate_execution_profile(
        self,
        mode: ExecutionMode,
        asset_type: AssetType,
        scope_size: int,
        control: Optional[ScrapeControlContract],
        risk_level: Optional[IntentRiskLevel],
        intent_category: Optional[IntentCategory],
        time_sensitivity: str,
        data_quality_requirement: str
    ) -> ExecutionProfile:
        """Generate comprehensive execution profile."""

        # Determine optimal strategy
        strategy = self._determine_execution_strategy(mode, asset_type, scope_size, risk_level, intent_category)

        # Generate reasoning
        reasoning = self._generate_execution_reasoning(
            mode, strategy, asset_type, scope_size, risk_level, intent_category,
            time_sensitivity, data_quality_requirement
        )

        # Calculate confidence
        confidence = self._calculate_execution_confidence(
            mode, strategy, asset_type, scope_size, control, risk_level
        )

        # Generate execution parameters
        execution_params = self._generate_execution_parameters(
            mode, strategy, asset_type, scope_size, time_sensitivity, data_quality_requirement
        )

        # Generate resource requirements
        resource_reqs = self._generate_resource_requirements(
            mode, strategy, scope_size, time_sensitivity
        )

        # Generate risk mitigations
        risk_mitigations = self._generate_risk_mitigations(
            mode, risk_level, intent_category, asset_type
        )

        # Generate performance expectations
        performance_exp = self._generate_performance_expectations(
            mode, strategy, scope_size, data_quality_requirement
        )

        # Generate compliance requirements
        compliance_reqs = self._generate_compliance_requirements(
            mode, intent_category, risk_level, asset_type
        )

        # Generate cost projections
        cost_projections = await self._generate_cost_projections(
            mode, asset_type, scope_size, control
        )

        return ExecutionProfile(
            mode=mode,
            strategy=strategy,
            confidence_score=confidence,
            reasoning=reasoning,
            execution_parameters=execution_params,
            resource_requirements=resource_reqs,
            risk_mitigations=risk_mitigations,
            performance_expectations=performance_exp,
            compliance_requirements=compliance_reqs,
            cost_projections=cost_projections
        )

    def _determine_execution_strategy(
        self,
        mode: ExecutionMode,
        asset_type: AssetType,
        scope_size: int,
        risk_level: Optional[IntentRiskLevel],
        intent_category: Optional[IntentCategory]
    ) -> ExecutionStrategy:
        """Determine optimal execution strategy."""

        # Risk-based strategy selection
        if risk_level == IntentRiskLevel.CRITICAL:
            return ExecutionStrategy.QUALITY_OPTIMIZED
        elif risk_level == IntentRiskLevel.HIGH:
            return ExecutionStrategy.RISK_WEIGHTED

        # Mode-based strategy selection
        mode_strategies = {
            ExecutionMode.PRECISION_SEARCH: ExecutionStrategy.DEPTH_FIRST,
            ExecutionMode.TARGETED_LOOKUP: ExecutionStrategy.DEPTH_FIRST,
            ExecutionMode.VERIFICATION_SCAN: ExecutionStrategy.QUALITY_OPTIMIZED,
            ExecutionMode.FOCUSED_DISCOVERY: ExecutionStrategy.PRIORITY_BASED,
            ExecutionMode.CONTROLLED_EXPLORATION: ExecutionStrategy.BREADTH_FIRST,
            ExecutionMode.INTELLIGENT_SWEEP: ExecutionStrategy.PRIORITY_BASED,
            ExecutionMode.DISCOVERY_SCRAPE: ExecutionStrategy.BREADTH_FIRST,
            ExecutionMode.COMPREHENSIVE_SURVEY: ExecutionStrategy.BREADTH_FIRST,
            ExecutionMode.EXHAUSTIVE_ANALYSIS: ExecutionStrategy.BREADTH_FIRST,
            ExecutionMode.MONITORING_MODE: ExecutionStrategy.TIME_OPTIMIZED,
            ExecutionMode.VALIDATION_MODE: ExecutionStrategy.QUALITY_OPTIMIZED,
            ExecutionMode.COMPLIANCE_AUDIT: ExecutionStrategy.QUALITY_OPTIMIZED,
            ExecutionMode.RAPID_RESPONSE: ExecutionStrategy.TIME_OPTIMIZED,
            ExecutionMode.CRISIS_MODE: ExecutionStrategy.TIME_OPTIMIZED
        }

        base_strategy = mode_strategies.get(mode, ExecutionStrategy.PRIORITY_BASED)

        # Asset type adjustments
        if asset_type == AssetType.PERSON and intent_category == IntentCategory.PERSONAL:
            base_strategy = ExecutionStrategy.DEPTH_FIRST
        elif asset_type in [AssetType.COMMERCIAL_PROPERTY, AssetType.COMPANY]:
            base_strategy = ExecutionStrategy.COST_OPTIMIZED

        # Scope size adjustments
        if scope_size > 1000:
            base_strategy = ExecutionStrategy.BREADTH_FIRST
        elif scope_size <= 3:
            base_strategy = ExecutionStrategy.DEPTH_FIRST

        return base_strategy

    def _generate_execution_reasoning(
        self,
        mode: ExecutionMode,
        strategy: ExecutionStrategy,
        asset_type: AssetType,
        scope_size: int,
        risk_level: Optional[IntentRiskLevel],
        intent_category: Optional[IntentCategory],
        time_sensitivity: str,
        data_quality_requirement: str
    ) -> List[str]:
        """Generate human-readable reasoning for execution mode selection."""

        reasoning = []

        # Base reasoning
        reasoning.append(f"Selected {mode.value} mode for {asset_type.value} asset type with scope size {scope_size}")

        # Strategy reasoning
        strategy_descriptions = {
            ExecutionStrategy.DEPTH_FIRST: "Deep, thorough investigation approach",
            ExecutionStrategy.BREADTH_FIRST: "Broad coverage exploration approach",
            ExecutionStrategy.PRIORITY_BASED: "Focus on high-value targets first",
            ExecutionStrategy.RISK_WEIGHTED: "Risk-adjusted execution approach",
            ExecutionStrategy.COST_OPTIMIZED: "Resource-efficient execution approach",
            ExecutionStrategy.TIME_OPTIMIZED: "Speed-optimized execution approach",
            ExecutionStrategy.QUALITY_OPTIMIZED: "Quality-focused execution approach"
        }

        if strategy in strategy_descriptions:
            reasoning.append(f"Using {strategy_descriptions[strategy]}")

        # Risk level reasoning
        if risk_level:
            risk_reasoning = {
                IntentRiskLevel.LOW: "Low-risk operation allows standard execution",
                IntentRiskLevel.MEDIUM: "Medium-risk operation requires balanced approach",
                IntentRiskLevel.HIGH: "High-risk operation demands controlled execution",
                IntentRiskLevel.CRITICAL: "Critical-risk operation requires maximum caution"
            }
            if risk_level in risk_reasoning:
                reasoning.append(risk_reasoning[risk_level])

        # Time sensitivity reasoning
        if time_sensitivity != "normal":
            time_reasoning = {
                "low": "Low time sensitivity allows thorough execution",
                "high": "High time sensitivity requires accelerated execution",
                "critical": "Critical time sensitivity demands emergency execution"
            }
            if time_sensitivity in time_reasoning:
                reasoning.append(time_reasoning[time_sensitivity])

        # Data quality reasoning
        if data_quality_requirement != "standard":
            quality_reasoning = {
                "basic": "Basic quality requirements allow efficient execution",
                "verified": "Verification requirements demand careful execution",
                "premium": "Premium quality requirements necessitate precise execution"
            }
            if data_quality_requirement in quality_reasoning:
                reasoning.append(quality_reasoning[data_quality_requirement])

        # Scope-based reasoning
        if scope_size == 1:
            reasoning.append("Single target allows precision-focused approach")
        elif scope_size <= 5:
            reasoning.append("Small scope enables targeted lookup strategy")
        elif scope_size > 1000:
            reasoning.append("Large scope requires broad discovery approach")

        return reasoning

    def _calculate_execution_confidence(
        self,
        mode: ExecutionMode,
        strategy: ExecutionStrategy,
        asset_type: AssetType,
        scope_size: int,
        control: Optional[ScrapeControlContract],
        risk_level: Optional[IntentRiskLevel]
    ) -> float:
        """Calculate confidence score for execution mode selection."""

        base_confidence = 0.8  # Default confidence

        # Mode success rate contribution
        mode_success = self.mode_success_rates.get(mode, 0.8)
        base_confidence = base_confidence * 0.3 + mode_success * 0.4

        # Strategy effectiveness contribution
        strategy_effective = self.strategy_effectiveness.get(strategy, 0.75)
        base_confidence = base_confidence * 0.8 + strategy_effective * 0.2

        # Risk level alignment contribution
        risk_alignment = 1.0
        if risk_level and mode:
            # Critical risk should have conservative modes
            if risk_level == IntentRiskLevel.CRITICAL:
                conservative_modes = [ExecutionMode.PRECISION_SEARCH, ExecutionMode.COMPLIANCE_AUDIT,
                                    ExecutionMode.VERIFICATION_SCAN]
                if mode in conservative_modes:
                    risk_alignment = 1.0
                else:
                    risk_alignment = 0.7
            # Low risk can handle broader modes
            elif risk_level == IntentRiskLevel.LOW:
                if mode in [ExecutionMode.DISCOVERY_SCRAPE, ExecutionMode.COMPREHENSIVE_SURVEY]:
                    risk_alignment = 1.0
                else:
                    risk_alignment = 0.9

        base_confidence *= risk_alignment

        # Historical performance contribution
        execution_key = f"{asset_type.value}_{scope_size}"
        if execution_key in self.performance_metrics:
            historical_success = self.performance_metrics[execution_key].get('success_rate', 0.8)
            base_confidence = base_confidence * 0.9 + historical_success * 0.1

        return min(1.0, max(0.0, base_confidence))

    def _generate_execution_parameters(
        self,
        mode: ExecutionMode,
        strategy: ExecutionStrategy,
        asset_type: AssetType,
        scope_size: int,
        time_sensitivity: str,
        data_quality_requirement: str
    ) -> Dict[str, Any]:
        """Generate execution parameters for the selected mode."""

        params = {
            'concurrent_requests': self._get_concurrent_limit_static(mode),
            'rate_limit_multiplier': self._get_rate_limit_multiplier_static(mode),
            'monitoring_intensity': self._get_monitoring_intensity_static(mode),
            'strategy': strategy.value,
            'time_sensitivity': time_sensitivity,
            'data_quality_requirement': data_quality_requirement,
            'batch_size': self._calculate_batch_size(mode, scope_size),
            'retry_policy': self._get_retry_policy(mode, risk_level=None),
            'timeout_settings': self._get_timeout_settings(mode, time_sensitivity),
            'data_validation_level': self._get_validation_level(data_quality_requirement)
        }

        # Strategy-specific parameters
        if strategy == ExecutionStrategy.DEPTH_FIRST:
            params.update({
                'exploration_depth': 'maximum',
                'breadth_control': 'minimal',
                'quality_threshold': 0.9
            })
        elif strategy == ExecutionStrategy.BREADTH_FIRST:
            params.update({
                'exploration_depth': 'minimal',
                'breadth_control': 'maximum',
                'quality_threshold': 0.7
            })
        elif strategy == ExecutionStrategy.PRIORITY_BASED:
            params.update({
                'priority_queue': True,
                'scoring_algorithm': 'intelligence_weighted',
                'fallback_threshold': 0.6
            })

        return params

    def _get_concurrent_limit_static(self, mode: ExecutionMode) -> int:
        """Static version of concurrent limit calculation."""
        limits = {
            ExecutionMode.PRECISION_SEARCH: 1,
            ExecutionMode.TARGETED_LOOKUP: 3,
            ExecutionMode.VERIFICATION_SCAN: 5,
            ExecutionMode.FOCUSED_DISCOVERY: 8,
            ExecutionMode.CONTROLLED_EXPLORATION: 10,
            ExecutionMode.INTELLIGENT_SWEEP: 15,
            ExecutionMode.DISCOVERY_SCRAPE: 20,
            ExecutionMode.COMPREHENSIVE_SURVEY: 25,
            ExecutionMode.EXHAUSTIVE_ANALYSIS: 30,
            ExecutionMode.MONITORING_MODE: 5,
            ExecutionMode.VALIDATION_MODE: 8,
            ExecutionMode.COMPLIANCE_AUDIT: 3,
            ExecutionMode.RAPID_RESPONSE: 50,
            ExecutionMode.CRISIS_MODE: 100
        }
        return limits.get(mode, 10)

    def _get_rate_limit_multiplier_static(self, mode: ExecutionMode) -> float:
        """Static version of rate limit multiplier."""
        multipliers = {
            ExecutionMode.PRECISION_SEARCH: 0.1,
            ExecutionMode.TARGETED_LOOKUP: 0.3,
            ExecutionMode.VERIFICATION_SCAN: 0.5,
            ExecutionMode.FOCUSED_DISCOVERY: 0.7,
            ExecutionMode.CONTROLLED_EXPLORATION: 1.0,
            ExecutionMode.INTELLIGENT_SWEEP: 1.2,
            ExecutionMode.DISCOVERY_SCRAPE: 1.5,
            ExecutionMode.COMPREHENSIVE_SURVEY: 1.8,
            ExecutionMode.EXHAUSTIVE_ANALYSIS: 2.0,
            ExecutionMode.MONITORING_MODE: 0.3,
            ExecutionMode.VALIDATION_MODE: 0.5,
            ExecutionMode.COMPLIANCE_AUDIT: 0.2,
            ExecutionMode.RAPID_RESPONSE: 3.0,
            ExecutionMode.CRISIS_MODE: 5.0
        }
        return multipliers.get(mode, 1.0)

    def _get_monitoring_intensity_static(self, mode: ExecutionMode) -> str:
        """Static version of monitoring intensity."""
        intensities = {
            ExecutionMode.PRECISION_SEARCH: "maximum",
            ExecutionMode.TARGETED_LOOKUP: "high",
            ExecutionMode.VERIFICATION_SCAN: "high",
            ExecutionMode.FOCUSED_DISCOVERY: "medium",
            ExecutionMode.CONTROLLED_EXPLORATION: "medium",
            ExecutionMode.INTELLIGENT_SWEEP: "medium",
            ExecutionMode.DISCOVERY_SCRAPE: "standard",
            ExecutionMode.COMPREHENSIVE_SURVEY: "standard",
            ExecutionMode.EXHAUSTIVE_ANALYSIS: "standard",
            ExecutionMode.MONITORING_MODE: "continuous",
            ExecutionMode.VALIDATION_MODE: "detailed",
            ExecutionMode.COMPLIANCE_AUDIT: "maximum",
            ExecutionMode.RAPID_RESPONSE: "intensive",
            ExecutionMode.CRISIS_MODE: "comprehensive"
        }
        return intensities.get(mode, "standard")

    def _calculate_batch_size(self, mode: ExecutionMode, scope_size: int) -> int:
        """Calculate optimal batch size for execution."""
        base_batch_sizes = {
            ExecutionMode.PRECISION_SEARCH: 1,
            ExecutionMode.TARGETED_LOOKUP: 3,
            ExecutionMode.VERIFICATION_SCAN: 5,
            ExecutionMode.FOCUSED_DISCOVERY: 10,
            ExecutionMode.CONTROLLED_EXPLORATION: 20,
            ExecutionMode.INTELLIGENT_SWEEP: 25,
            ExecutionMode.DISCOVERY_SCRAPE: 50,
            ExecutionMode.COMPREHENSIVE_SURVEY: 100,
            ExecutionMode.EXHAUSTIVE_ANALYSIS: 200,
            ExecutionMode.MONITORING_MODE: 5,
            ExecutionMode.VALIDATION_MODE: 10,
            ExecutionMode.COMPLIANCE_AUDIT: 2,
            ExecutionMode.RAPID_RESPONSE: 100,
            ExecutionMode.CRISIS_MODE: 500
        }

        base_size = base_batch_sizes.get(mode, 10)
        # Adjust based on scope size to avoid oversized batches
        max_batch = min(base_size, scope_size // 4 + 1)
        return max(1, max_batch)

    def _get_retry_policy(self, mode: ExecutionMode, risk_level: Optional[IntentRiskLevel]) -> Dict[str, Any]:
        """Get retry policy for execution mode."""
        base_policy = {
            'max_retries': 3,
            'backoff_factor': 2.0,
            'jitter': True
        }

        # Adjust based on mode
        mode_policies = {
            ExecutionMode.PRECISION_SEARCH: {'max_retries': 5, 'backoff_factor': 1.5},
            ExecutionMode.COMPLIANCE_AUDIT: {'max_retries': 5, 'backoff_factor': 1.0},
            ExecutionMode.RAPID_RESPONSE: {'max_retries': 2, 'backoff_factor': 1.2},
            ExecutionMode.CRISIS_MODE: {'max_retries': 1, 'backoff_factor': 1.0},
            ExecutionMode.MONITORING_MODE: {'max_retries': 10, 'backoff_factor': 3.0}
        }

        if mode in mode_policies:
            base_policy.update(mode_policies[mode])

        # Adjust for risk level
        if risk_level == IntentRiskLevel.CRITICAL:
            base_policy['max_retries'] = min(base_policy['max_retries'] + 2, 7)
        elif risk_level == IntentRiskLevel.LOW:
            base_policy['max_retries'] = max(base_policy['max_retries'] - 1, 1)

        return base_policy

    def _get_timeout_settings(self, mode: ExecutionMode, time_sensitivity: str) -> Dict[str, Any]:
        """Get timeout settings for execution."""
        base_timeout = 30  # seconds

        mode_timeouts = {
            ExecutionMode.PRECISION_SEARCH: 60,
            ExecutionMode.COMPLIANCE_AUDIT: 120,
            ExecutionMode.RAPID_RESPONSE: 15,
            ExecutionMode.CRISIS_MODE: 10,
            ExecutionMode.EXHAUSTIVE_ANALYSIS: 300,
            ExecutionMode.MONITORING_MODE: 45
        }

        timeout = mode_timeouts.get(mode, base_timeout)

        # Adjust for time sensitivity
        if time_sensitivity == "high":
            timeout = max(timeout * 0.7, 10)
        elif time_sensitivity == "critical":
            timeout = max(timeout * 0.5, 5)
        elif time_sensitivity == "low":
            timeout = timeout * 1.5

        return {
            'request_timeout': timeout,
            'total_operation_timeout': timeout * 10,
            'batch_timeout': timeout * 5
        }

    def _get_validation_level(self, data_quality_requirement: str) -> str:
        """Get validation level based on quality requirements."""
        validation_levels = {
            "basic": "minimal",
            "standard": "standard",
            "verified": "comprehensive",
            "premium": "exhaustive"
        }
        return validation_levels.get(data_quality_requirement, "standard")

    def _generate_resource_requirements(
        self,
        mode: ExecutionMode,
        strategy: ExecutionStrategy,
        scope_size: int,
        time_sensitivity: str
    ) -> Dict[str, Any]:
        """Generate resource requirements for execution."""

        # Base resource calculations
        concurrent_requests = self._get_concurrent_limit_static(mode)
        estimated_duration_hours = self._estimate_execution_duration(mode, scope_size)

        # CPU and memory estimates
        cpu_cores = max(1, concurrent_requests // 5)
        memory_gb = max(1, concurrent_requests // 10)

        # Network bandwidth estimate
        bandwidth_mbps = concurrent_requests * 2  # Rough estimate

        # Adjust for time sensitivity
        if time_sensitivity == "critical":
            cpu_cores = int(cpu_cores * 1.5)
            memory_gb = int(memory_gb * 1.3)
            bandwidth_mbps = int(bandwidth_mbps * 2)

        # Strategy-specific adjustments
        if strategy == ExecutionStrategy.DEPTH_FIRST:
            memory_gb = int(memory_gb * 1.2)  # More memory for deep analysis
        elif strategy == ExecutionStrategy.BREADTH_FIRST:
            bandwidth_mbps = int(bandwidth_mbps * 1.3)  # More network for breadth

        return {
            'cpu_cores': cpu_cores,
            'memory_gb': memory_gb,
            'network_bandwidth_mbps': bandwidth_mbps,
            'estimated_duration_hours': estimated_duration_hours,
            'storage_gb': max(1, scope_size // 1000),  # Rough storage estimate
            'concurrent_requests': concurrent_requests
        }

    def _estimate_execution_duration(self, mode: ExecutionMode, scope_size: int) -> float:
        """Estimate execution duration in hours."""
        # Base rates (items per hour) by mode
        rates_per_hour = {
            ExecutionMode.PRECISION_SEARCH: 10,
            ExecutionMode.TARGETED_LOOKUP: 50,
            ExecutionMode.VERIFICATION_SCAN: 100,
            ExecutionMode.FOCUSED_DISCOVERY: 200,
            ExecutionMode.CONTROLLED_EXPLORATION: 500,
            ExecutionMode.INTELLIGENT_SWEEP: 800,
            ExecutionMode.DISCOVERY_SCRAPE: 1500,
            ExecutionMode.COMPREHENSIVE_SURVEY: 3000,
            ExecutionMode.EXHAUSTIVE_ANALYSIS: 5000,
            ExecutionMode.MONITORING_MODE: 100,
            ExecutionMode.VALIDATION_MODE: 200,
            ExecutionMode.COMPLIANCE_AUDIT: 25,
            ExecutionMode.RAPID_RESPONSE: 5000,
            ExecutionMode.CRISIS_MODE: 10000
        }

        rate = rates_per_hour.get(mode, 500)
        return max(0.1, scope_size / rate)

    def _generate_risk_mitigations(
        self,
        mode: ExecutionMode,
        risk_level: Optional[IntentRiskLevel],
        intent_category: Optional[IntentCategory],
        asset_type: AssetType
    ) -> List[str]:
        """Generate risk mitigation strategies."""

        mitigations = []

        # Base mitigations for all operations
        mitigations.extend([
            "Implement comprehensive logging and audit trails",
            "Use rate limiting to avoid detection",
            "Maintain session diversity and rotation"
        ])

        # Risk level specific mitigations
        if risk_level == IntentRiskLevel.CRITICAL:
            mitigations.extend([
                "Require dual authorization for execution",
                "Implement real-time monitoring and kill switches",
                "Use maximum session isolation and cleanup",
                "Enforce strict data retention policies"
            ])
        elif risk_level == IntentRiskLevel.HIGH:
            mitigations.extend([
                "Implement enhanced monitoring and alerting",
                "Use conservative rate limiting",
                "Require manual approval for high-volume operations"
            ])

        # Mode-specific mitigations
        if mode in [ExecutionMode.COMPREHENSIVE_SURVEY, ExecutionMode.EXHAUSTIVE_ANALYSIS]:
            mitigations.extend([
                "Implement progressive rate limiting",
                "Use distributed execution to avoid concentration",
                "Monitor for blocking patterns and adapt"
            ])

        if mode in [ExecutionMode.PRECISION_SEARCH, ExecutionMode.COMPLIANCE_AUDIT]:
            mitigations.extend([
                "Use maximum data quality validation",
                "Implement detailed error handling and recovery",
                "Maintain comprehensive execution logs"
            ])

        # Asset type specific mitigations
        if asset_type == AssetType.PERSON:
            mitigations.extend([
                "Ensure privacy compliance and data minimization",
                "Implement strict data anonymization",
                "Follow data protection regulations"
            ])

        if asset_type in [AssetType.COMMERCIAL_PROPERTY, AssetType.COMPANY]:
            mitigations.extend([
                "Verify legal authority for data collection",
                "Implement business record access controls",
                "Follow industry-specific compliance requirements"
            ])

        return list(set(mitigations))  # Remove duplicates

    def _generate_performance_expectations(
        self,
        mode: ExecutionMode,
        strategy: ExecutionStrategy,
        scope_size: int,
        data_quality_requirement: str
    ) -> Dict[str, Any]:
        """Generate performance expectations for execution."""

        # Base success rates by mode
        base_success_rates = {
            ExecutionMode.PRECISION_SEARCH: 0.95,
            ExecutionMode.TARGETED_LOOKUP: 0.90,
            ExecutionMode.VERIFICATION_SCAN: 0.85,
            ExecutionMode.FOCUSED_DISCOVERY: 0.80,
            ExecutionMode.CONTROLLED_EXPLORATION: 0.75,
            ExecutionMode.INTELLIGENT_SWEEP: 0.75,
            ExecutionMode.DISCOVERY_SCRAPE: 0.70,
            ExecutionMode.COMPREHENSIVE_SURVEY: 0.65,
            ExecutionMode.EXHAUSTIVE_ANALYSIS: 0.60,
            ExecutionMode.MONITORING_MODE: 0.85,
            ExecutionMode.VALIDATION_MODE: 0.90,
            ExecutionMode.COMPLIANCE_AUDIT: 0.95,
            ExecutionMode.RAPID_RESPONSE: 0.70,
            ExecutionMode.CRISIS_MODE: 0.60
        }

        success_rate = base_success_rates.get(mode, 0.75)

        # Adjust for data quality requirements
        if data_quality_requirement == "premium":
            success_rate *= 0.9  # Higher quality reduces success rate
        elif data_quality_requirement == "basic":
            success_rate *= 1.1  # Lower quality increases success rate

        # Adjust for strategy
        if strategy == ExecutionStrategy.QUALITY_OPTIMIZED:
            success_rate *= 0.95
        elif strategy == ExecutionStrategy.TIME_OPTIMIZED:
            success_rate *= 0.9

        # Calculate expected completion metrics
        estimated_duration = self._estimate_execution_duration(mode, scope_size)
        expected_completion_rate = min(95.0, success_rate * 100)

        return {
            'expected_success_rate': success_rate,
            'expected_completion_rate': expected_completion_rate,
            'estimated_duration_hours': estimated_duration,
            'expected_data_quality_score': self._get_expected_quality_score(data_quality_requirement),
            'performance_confidence': min(1.0, success_rate * 1.2),
            'scalability_score': self._calculate_scalability_score(mode, scope_size)
        }

    def _get_expected_quality_score(self, data_quality_requirement: str) -> float:
        """Get expected data quality score."""
        quality_scores = {
            "basic": 0.7,
            "standard": 0.85,
            "verified": 0.95,
            "premium": 0.98
        }
        return quality_scores.get(data_quality_requirement, 0.85)

    def _calculate_scalability_score(self, mode: ExecutionMode, scope_size: int) -> float:
        """Calculate scalability score for the execution mode."""
        # Higher scores mean better scalability
        base_scalability = {
            ExecutionMode.PRECISION_SEARCH: 0.3,      # Poor scalability
            ExecutionMode.TARGETED_LOOKUP: 0.5,       # Limited scalability
            ExecutionMode.VERIFICATION_SCAN: 0.6,     # Moderate scalability
            ExecutionMode.FOCUSED_DISCOVERY: 0.7,     # Good scalability
            ExecutionMode.CONTROLLED_EXPLORATION: 0.8, # Very good scalability
            ExecutionMode.INTELLIGENT_SWEEP: 0.9,     # Excellent scalability
            ExecutionMode.DISCOVERY_SCRAPE: 0.95,     # Near linear scalability
            ExecutionMode.COMPREHENSIVE_SURVEY: 0.9,  # Very good scalability
            ExecutionMode.EXHAUSTIVE_ANALYSIS: 0.85,  # Good scalability
            ExecutionMode.MONITORING_MODE: 0.6,       # Moderate (continuous)
            ExecutionMode.VALIDATION_MODE: 0.7,       # Good scalability
            ExecutionMode.COMPLIANCE_AUDIT: 0.4,      # Limited scalability
            ExecutionMode.RAPID_RESPONSE: 0.9,        # Excellent scalability
            ExecutionMode.CRISIS_MODE: 0.95           # Maximum scalability
        }

        base_score = base_scalability.get(mode, 0.7)

        # Adjust based on scope size (very large scopes may have diminishing returns)
        if scope_size > 10000:
            scale_penalty = min(0.3, (scope_size - 10000) / 100000)
            base_score *= (1 - scale_penalty)

        return base_score

    def _generate_compliance_requirements(
        self,
        mode: ExecutionMode,
        intent_category: Optional[IntentCategory],
        risk_level: Optional[IntentRiskLevel],
        asset_type: AssetType
    ) -> List[str]:
        """Generate compliance requirements for execution."""

        requirements = []

        # Base requirements for all operations
        requirements.extend([
            "data_collection_transparency",
            "audit_trail_maintenance",
            "data_retention_policy_compliance"
        ])

        # Risk level specific requirements
        if risk_level == IntentRiskLevel.CRITICAL:
            requirements.extend([
                "executive_approval_required",
                "legal_review_mandatory",
                "enhanced_audit_logging",
                "data_minimization_enforced",
                "privacy_impact_assessment_required"
            ])
        elif risk_level == IntentRiskLevel.HIGH:
            requirements.extend([
                "senior_approval_required",
                "compliance_review_mandatory",
                "detailed_audit_logging",
                "data_usage_restrictions"
            ])

        # Intent category specific requirements
        if intent_category == IntentCategory.PERSONAL:
            requirements.extend([
                "gdpr_compliance_required",
                "ccpa_compliance_required",
                "data_subject_rights_respected",
                "privacy_by_design_implemented"
            ])
        elif intent_category == IntentCategory.FINANCIAL:
            requirements.extend([
                "financial_data_protection",
                "pci_compliance_if_applicable",
                "financial_regulation_compliance",
                "data_encryption_required"
            ])
        elif intent_category == IntentCategory.LEGAL:
            requirements.extend([
                "legal_authority_verification",
                "court_order_compliance",
                "attorney_client_privilege_respected",
                "evidence_chain_maintenance"
            ])

        # Asset type specific requirements
        if asset_type == AssetType.PERSON:
            requirements.extend([
                "individual_privacy_protection",
                "consent_mechanism_verification",
                "data_portability_rights_support"
            ])

        # Mode-specific requirements
        if mode == ExecutionMode.COMPLIANCE_AUDIT:
            requirements.extend([
                "independent_audit_trail",
                "regulatory_reporting_compliance",
                "third_party_verification",
                "statutory_compliance_validation"
            ])

        if mode in [ExecutionMode.RAPID_RESPONSE, ExecutionMode.CRISIS_MODE]:
            requirements.extend([
                "emergency_procedures_activation",
                "crisis_communication_protocol",
                "accelerated_compliance_review",
                "temporary_regulatory_relief_verification"
            ])

        return list(set(requirements))  # Remove duplicates

    async def _generate_cost_projections(
        self,
        mode: ExecutionMode,
        asset_type: AssetType,
        scope_size: int,
        control: Optional[ScrapeControlContract]
    ) -> Dict[str, Any]:
        """Generate cost projections for execution."""

        # Get primary signal type for cost estimation
        primary_signal = self._infer_primary_signal_type_for_cost(asset_type)

        # Estimate base costs
        if primary_signal:
            base_cost_estimate = calculate_signal_cost_estimate(
                asset_type, primary_signal, data_quality="standard"
            )
        else:
            base_cost_estimate = {"total_estimated_cost": 2.5}

        base_cost_per_item = base_cost_estimate["total_estimated_cost"]

        # Adjust for mode and scope
        mode_multipliers = {
            ExecutionMode.PRECISION_SEARCH: 1.5,      # More expensive due to care
            ExecutionMode.TARGETED_LOOKUP: 1.2,       # Moderately expensive
            ExecutionMode.VERIFICATION_SCAN: 1.3,     # Validation adds cost
            ExecutionMode.FOCUSED_DISCOVERY: 1.0,     # Baseline
            ExecutionMode.CONTROLLED_EXPLORATION: 1.1, # Slight premium for control
            ExecutionMode.INTELLIGENT_SWEEP: 1.0,     # Efficient
            ExecutionMode.DISCOVERY_SCRAPE: 0.9,      # Bulk discount
            ExecutionMode.COMPREHENSIVE_SURVEY: 0.8,  # Volume discount
            ExecutionMode.EXHAUSTIVE_ANALYSIS: 1.2,   # Resource intensive
            ExecutionMode.MONITORING_MODE: 1.8,       # Continuous monitoring
            ExecutionMode.VALIDATION_MODE: 1.4,       # Quality validation
            ExecutionMode.COMPLIANCE_AUDIT: 2.0,      # Compliance overhead
            ExecutionMode.RAPID_RESPONSE: 1.3,        # Premium for speed
            ExecutionMode.CRISIS_MODE: 1.5            # Emergency premium
        }

        mode_multiplier = mode_multipliers.get(mode, 1.0)
        total_estimated_cost = base_cost_per_item * scope_size * mode_multiplier

        # Breakdown by cost categories
        breakdown = {
            'base_signal_cost': base_cost_per_item * scope_size,
            'execution_mode_premium': base_cost_per_item * scope_size * (mode_multiplier - 1),
            'infrastructure_cost': total_estimated_cost * 0.2,
            'compliance_cost': total_estimated_cost * 0.15,
            'total_estimated_cost': total_estimated_cost
        }

        # Cost confidence and ranges
        confidence_range = self._calculate_cost_confidence(total_estimated_cost)
        breakdown.update(confidence_range)

        # Optimization suggestions
        optimizations = self._generate_cost_optimizations(mode, asset_type, scope_size, total_estimated_cost)
        breakdown['optimization_suggestions'] = optimizations

        return breakdown

    def _infer_primary_signal_type_for_cost(self, asset_type: AssetType) -> Optional[SignalType]:
        """Infer primary signal type for cost estimation."""
        type_mapping = {
            AssetType.PERSON: SignalType.IDENTITY,
            AssetType.SINGLE_FAMILY_HOME: SignalType.LIEN,
            AssetType.MULTI_FAMILY_SMALL: SignalType.MORTGAGE,
            AssetType.APARTMENT_BUILDING: SignalType.COURT_CASE,
            AssetType.COMMERCIAL_PROPERTY: SignalType.DEED,
            AssetType.COMPANY: SignalType.FINANCIAL
        }
        return type_mapping.get(asset_type)

    def _calculate_cost_confidence(self, total_cost: float) -> Dict[str, Any]:
        """Calculate cost estimate confidence ranges."""
        # Cost estimates have inherent uncertainty
        # Higher costs generally have higher uncertainty
        base_uncertainty = 0.2  # 20% base uncertainty

        # Scale uncertainty with cost magnitude
        if total_cost > 100:
            uncertainty_multiplier = 1.5
        elif total_cost > 50:
            uncertainty_multiplier = 1.2
        else:
            uncertainty_multiplier = 1.0

        uncertainty = base_uncertainty * uncertainty_multiplier

        return {
            'cost_confidence_level': max(0.5, 1.0 - uncertainty),
            'estimated_range_low': total_cost * (1 - uncertainty),
            'estimated_range_high': total_cost * (1 + uncertainty),
            'uncertainty_percentage': uncertainty * 100
        }

    def _generate_cost_optimizations(
        self,
        mode: ExecutionMode,
        asset_type: AssetType,
        scope_size: int,
        total_cost: float
    ) -> List[str]:
        """Generate cost optimization suggestions."""
        optimizations = []

        # Mode-specific optimizations
        if mode in [ExecutionMode.COMPREHENSIVE_SURVEY, ExecutionMode.EXHAUSTIVE_ANALYSIS]:
            optimizations.append("Consider switching to FOCUSED_DISCOVERY mode for 15-20% cost savings")
            optimizations.append("Implement priority-based execution to reduce scope by 30%")

        if mode == ExecutionMode.PRECISION_SEARCH and scope_size > 3:
            optimizations.append("Consider TARGETED_LOOKUP mode for batch processing efficiency")

        # Scope-based optimizations
        if scope_size > 1000:
            optimizations.append("Implement phased execution to reduce peak resource costs")
            optimizations.append("Consider sampling approach for initial cost estimation")

        # Asset type optimizations
        if asset_type == AssetType.PERSON:
            optimizations.append("Leverage public records for cost-effective data sources")
        elif asset_type in [AssetType.COMMERCIAL_PROPERTY, AssetType.COMPANY]:
            optimizations.append("Utilize commercial data providers for bulk discounts")

        # Cost threshold optimizations
        if total_cost > 500:
            optimizations.append("Consider distributed execution across multiple instances")
            optimizations.append("Implement caching strategies for repeated data access")

        if total_cost > 100:
            optimizations.append("Batch operations to leverage volume discounts")
            optimizations.append("Schedule execution during off-peak hours for cost savings")

        return optimizations[:5]  # Limit to top 5 suggestions

    def update_performance_metrics(self, execution_key: str, success: bool, cost: float, duration: float):
        """Update performance metrics for learning and optimization."""
        if execution_key not in self.performance_metrics:
            self.performance_metrics[execution_key] = {
                'executions': 0,
                'successes': 0,
                'total_cost': 0.0,
                'total_duration': 0.0,
                'avg_cost': 0.0,
                'avg_duration': 0.0,
                'success_rate': 0.0
            }

        metrics = self.performance_metrics[execution_key]
        metrics['executions'] += 1
        if success:
            metrics['successes'] += 1
        metrics['total_cost'] += cost
        metrics['total_duration'] += duration

        # Recalculate averages
        metrics['avg_cost'] = metrics['total_cost'] / metrics['executions']
        metrics['avg_duration'] = metrics['total_duration'] / metrics['executions']
        metrics['success_rate'] = metrics['successes'] / metrics['executions']

    def get_classifier_stats(self) -> Dict[str, Any]:
        """Get comprehensive classifier statistics."""
        total_profiles = sum(len(profiles) for profiles in self.execution_history.values())

        if total_profiles == 0:
            return {"total_execution_profiles": 0}

        # Mode distribution
        mode_counts = {}
        for profiles in self.execution_history.values():
            for profile in profiles:
                mode_counts[profile.mode.value] = mode_counts.get(profile.mode.value, 0) + 1

        # Strategy distribution
        strategy_counts = {}
        for profiles in self.execution_history.values():
            for profile in profiles:
                strategy_counts[profile.strategy.value] = strategy_counts.get(profile.strategy.value, 0) + 1

        # Performance metrics summary
        performance_summary = {}
        for key, metrics in self.performance_metrics.items():
            performance_summary[key] = {
                'success_rate': metrics['success_rate'],
                'avg_cost': metrics['avg_cost'],
                'avg_duration': metrics['avg_duration'],
                'total_executions': metrics['executions']
            }

        return {
            "total_execution_profiles": total_profiles,
            "unique_execution_keys": len(self.execution_history),
            "mode_distribution": mode_counts,
            "strategy_distribution": strategy_counts,
            "performance_tracked_keys": len(self.performance_metrics),
            "performance_summary": performance_summary,
            "mode_success_rates": {mode.value: rate for mode, rate in self.mode_success_rates.items()},
            "strategy_effectiveness": {strategy.value: eff for strategy, eff in self.strategy_effectiveness.items()}
        }


# Global classifier instance
_global_classifier = ExecutionModeClassifier()


# Enhanced convenience functions
async def classify_execution_mode(
    asset_type: AssetType,
    scope_size: int,
    control: Optional[ScrapeControlContract] = None,
    risk_level: Optional[IntentRiskLevel] = None,
    intent_category: Optional[IntentCategory] = None,
    time_sensitivity: str = "normal",
    data_quality_requirement: str = "standard"
) -> ExecutionProfile:
    """
    Enhanced execution mode classification with comprehensive intelligence.

    This is the main entry point for execution mode classification in the MJ Data Scraper Suite.
    Provides intelligent execution mode selection based on asset types, scope, risk levels,
    operational requirements, and performance optimization.

    Args:
        asset_type: Type of asset being targeted
        scope_size: Number of targets/assets in scope
        control: Optional scraping control contract for additional context
        risk_level: Risk classification level
        intent_category: Intent category classification
        time_sensitivity: "low", "normal", "high", "critical"
        data_quality_requirement: "basic", "standard", "verified", "premium"

    Returns:
        Complete ExecutionProfile with mode, strategy, parameters, and intelligence
    """
    return await _global_classifier.classify_execution_mode(
        asset_type, scope_size, control, risk_level, intent_category,
        time_sensitivity, data_quality_requirement
    )


def get_execution_mode_statistics() -> Dict[str, Any]:
    """
    Get comprehensive execution mode classification statistics.

    Returns operational metrics for monitoring execution mode selection
    and performance optimization across the scraping ecosystem.

    Returns:
        Dict with execution mode statistics and performance metrics
    """
    return _global_classifier.get_classifier_stats()


def update_execution_performance(execution_key: str, success: bool, cost: float, duration: float):
    """
    Update execution performance metrics for learning and optimization.

    Args:
        execution_key: Unique key identifying the execution scenario
        success: Whether the execution was successful
        cost: Actual cost incurred
        duration: Execution duration in hours
    """
    _global_classifier.update_performance_metrics(execution_key, success, cost, duration)


# Legacy compatibility function (enhanced)
def classify_execution_mode_simple(asset_type: AssetType, scope_size: int) -> str:
    """
    Simplified execution mode classification (legacy compatibility).

    Enhanced version of the original function with intelligent improvements
    while maintaining backward compatibility.

    Args:
        asset_type: Type of asset
        scope_size: Number of targets

    Returns:
        Execution mode string (legacy format)
    """
    # Use the enhanced classifier with defaults
    profile = asyncio.run(classify_execution_mode(asset_type, scope_size))
    return profile.mode.value


# Mode and strategy enumeration exports
__all__ = [
    'ExecutionMode',
    'ExecutionStrategy',
    'ExecutionProfile',
    'ExecutionModeClassifier',
    'classify_execution_mode',
    'get_execution_mode_statistics',
    'update_execution_performance',
    'classify_execution_mode_simple'
]
