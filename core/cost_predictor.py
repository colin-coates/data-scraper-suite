# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use as is strictly prohibited.

"""
Cost Prediction Engine for MJ Data Scraper Suite

Advanced machine learning and analytical cost prediction system that forecasts
scraping costs, optimizes execution strategies, and provides budget intelligence
for enterprise-grade data operations.
"""

import asyncio
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Fallback implementations
    class RandomForestRegressor:
        def fit(self, X, y): pass
        def predict(self, X): return np.mean(y) if hasattr(np, 'mean') else sum(y)/len(y) if y else 0
        feature_importances_: list = []

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    def np_array(data): return data
    def np_mean(data): return sum(data) / len(data) if data else 0
    def np_std(data): return 0

from core.models.asset_signal import AssetType, SignalType
from core.execution_mode_classifier import ExecutionMode, ExecutionStrategy
from core.intent_classifier import IntentRiskLevel, IntentCategory
from core.control_models import ScrapeControlContract, ScrapeBudget
from core.mapping.asset_signal_map import (
    SIGNAL_COST_WEIGHT,
    SOURCE_RELIABILITY,
    get_signal_cost_weight,
    get_source_reliability_score,
    calculate_signal_cost_estimate
)

logger = logging.getLogger(__name__)


class CostPredictionModel(Enum):
    """Cost prediction model types."""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


class CostOptimizationStrategy(Enum):
    """Cost optimization strategy types."""
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_VALUE = "maximize_value"
    BALANCE_COST_VALUE = "balance_cost_value"
    TIME_CONSTRAINED = "time_constrained"
    QUALITY_CONSTRAINED = "quality_constrained"
    RISK_CONSTRAINED = "risk_constrained"


@dataclass
class CostPrediction:
    """Comprehensive cost prediction result."""
    prediction_id: str
    predicted_cost: float
    confidence_score: float
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    cost_range: Tuple[float, float] = (0.0, 0.0)
    risk_adjustments: Dict[str, float] = field(default_factory=dict)
    optimization_recommendations: List[str] = field(default_factory=list)
    alternative_scenarios: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    prediction_metadata: Dict[str, Any] = field(default_factory=dict)
    prediction_timestamp: datetime = field(default_factory=datetime.utcnow)

    def get_cost_efficiency_ratio(self) -> float:
        """Calculate cost efficiency ratio (lower is better)."""
        if self.predicted_cost <= 0:
            return 0.0
        # Efficiency based on confidence and cost magnitude
        efficiency_penalty = (1.0 - self.confidence_score) * 0.2
        return self.predicted_cost * (1.0 + efficiency_penalty)

    def get_cost_volatility(self) -> float:
        """Calculate cost prediction volatility."""
        if self.cost_range[1] <= self.cost_range[0]:
            return 0.0
        mean_cost = (self.cost_range[0] + self.cost_range[1]) / 2
        if mean_cost <= 0:
            return 0.0
        return (self.cost_range[1] - self.cost_range[0]) / mean_cost


@dataclass
class CostOptimizationPlan:
    """Cost optimization plan with recommendations."""
    plan_id: str
    original_cost: float
    optimized_cost: float
    cost_savings: float
    savings_percentage: float
    optimization_strategy: CostOptimizationStrategy
    recommended_changes: List[Dict[str, Any]] = field(default_factory=list)
    implementation_priority: str = "medium"
    expected_roi: float = 0.0
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    timeline_estimate: str = ""
    created_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BudgetAnalysis:
    """Comprehensive budget analysis and forecasting."""
    analysis_id: str
    total_budget: float
    projected_cost: float
    budget_utilization: float
    cost_variance: float
    risk_of_overspend: str
    cost_drivers: List[Dict[str, str]] = field(default_factory=list)
    budget_optimization_opportunities: List[str] = field(default_factory=list)
    forecasting_accuracy: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)


class CostPredictor:
    """
    Advanced cost prediction and optimization engine for MJ Data Scraper Suite.

    Uses machine learning, historical data, and operational intelligence to predict
    scraping costs, optimize execution strategies, and provide budget intelligence.
    """

    def __init__(self):
        self.cost_history: List[Dict[str, Any]] = []
        self.cost_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        ) if ML_AVAILABLE else None
        self.feature_scaler = StandardScaler() if ML_AVAILABLE else None
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        ) if ML_AVAILABLE else None

        # Performance tracking
        self.prediction_accuracy = defaultdict(list)
        self.model_performance = {}
        self.cost_patterns = defaultdict(list)

        # Initialize with synthetic training data
        try:
            asyncio.create_task(self._initialize_models())
        except RuntimeError:
            # No event loop running
            pass

        logger.info("CostPredictor initialized with ML-enhanced cost prediction")

    async def _initialize_models(self):
        """Initialize and train cost prediction models."""
        if not ML_AVAILABLE or len(self.cost_history) < 10:
            logger.info("Insufficient data for ML model training")
            return

        try:
            # Prepare training data
            X, y = self._prepare_training_data()

            if len(X) > 0:
                # Train cost prediction model
                self.cost_model.fit(X, y)

                # Train anomaly detector
                self.anomaly_detector.fit(X)

                logger.info(f"Cost prediction models trained with {len(X)} samples")

        except Exception as e:
            logger.warning(f"Model training failed: {e}")

    def _prepare_training_data(self) -> Tuple[List, List]:
        """Prepare training data from cost history."""
        if not self.cost_history:
            return [], []

        features = []
        targets = []

        for record in self.cost_history[-1000:]:  # Use last 1000 records
            feature_vector = self._extract_cost_features(record)
            if feature_vector and 'actual_cost' in record:
                features.append(feature_vector)
                targets.append(record['actual_cost'])

        return features, targets

    async def predict_cost(
        self,
        asset_type: AssetType,
        signal_type: Optional[SignalType] = None,
        execution_mode: Optional[ExecutionMode] = None,
        scope_size: int = 1,
        risk_level: Optional[IntentRiskLevel] = None,
        intent_category: Optional[IntentCategory] = None,
        time_sensitivity: str = "normal",
        data_quality: str = "standard",
        control: Optional[ScrapeControlContract] = None
    ) -> CostPrediction:
        """
        Predict comprehensive scraping costs using ML and analytical models.

        Args:
            asset_type: Type of asset being targeted
            signal_type: Type of signal (optional)
            execution_mode: Execution mode (optional)
            scope_size: Number of targets
            risk_level: Risk classification
            intent_category: Intent category
            time_sensitivity: Time requirements
            data_quality: Quality requirements
            control: Optional control contract

        Returns:
            Comprehensive cost prediction with breakdowns and recommendations
        """

        prediction_id = self._generate_prediction_id(
            asset_type, signal_type, execution_mode, scope_size,
            risk_level, intent_category, time_sensitivity, data_quality
        )

        # Extract features for prediction
        features = self._extract_prediction_features(
            asset_type, signal_type, execution_mode, scope_size,
            risk_level, intent_category, time_sensitivity, data_quality, control
        )

        # Generate base cost estimate
        base_cost = await self._calculate_base_cost_estimate(
            asset_type, signal_type, execution_mode, scope_size, control
        )

        # Apply ML prediction if available
        ml_adjustment = await self._apply_ml_prediction(features, base_cost)
        predicted_cost = base_cost * ml_adjustment

        # Calculate confidence and ranges
        confidence_score = self._calculate_prediction_confidence(features, base_cost)
        cost_range = self._calculate_cost_range(predicted_cost, confidence_score)

        # Generate cost breakdown
        cost_breakdown = await self._generate_cost_breakdown(
            asset_type, signal_type, execution_mode, scope_size,
            risk_level, intent_category, control, predicted_cost
        )

        # Generate risk adjustments
        risk_adjustments = self._calculate_risk_adjustments(
            risk_level, intent_category, asset_type, predicted_cost
        )

        # Generate optimization recommendations
        optimization_recs = await self._generate_optimization_recommendations(
            asset_type, signal_type, execution_mode, scope_size,
            risk_level, intent_category, predicted_cost, confidence_score
        )

        # Generate alternative scenarios
        alternative_scenarios = await self._generate_alternative_scenarios(
            asset_type, signal_type, scope_size, risk_level, intent_category,
            time_sensitivity, data_quality, predicted_cost
        )

        # Generate prediction metadata
        prediction_metadata = {
            'asset_type': asset_type.value,
            'signal_type': signal_type.value if signal_type else None,
            'execution_mode': execution_mode.value if execution_mode else None,
            'scope_size': scope_size,
            'risk_level': risk_level.value if risk_level else None,
            'intent_category': intent_category.value if intent_category else None,
            'time_sensitivity': time_sensitivity,
            'data_quality': data_quality,
            'ml_model_used': ML_AVAILABLE and self.cost_model is not None,
            'historical_data_points': len(self.cost_history),
            'feature_count': len(features)
        }

        return CostPrediction(
            prediction_id=prediction_id,
            predicted_cost=round(predicted_cost, 2),
            confidence_score=round(confidence_score, 3),
            cost_breakdown=cost_breakdown,
            cost_range=(round(cost_range[0], 2), round(cost_range[1], 2)),
            risk_adjustments=risk_adjustments,
            optimization_recommendations=optimization_recs,
            alternative_scenarios=alternative_scenarios,
            prediction_metadata=prediction_metadata
        )

    def _generate_prediction_id(self, *args) -> str:
        """Generate unique prediction identifier."""
        content = "_".join(str(arg) for arg in args if arg is not None)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _extract_prediction_features(
        self,
        asset_type: AssetType,
        signal_type: Optional[SignalType],
        execution_mode: Optional[ExecutionMode],
        scope_size: int,
        risk_level: Optional[IntentRiskLevel],
        intent_category: Optional[IntentCategory],
        time_sensitivity: str,
        data_quality: str,
        control: Optional[ScrapeControlContract]
    ) -> List[float]:
        """Extract numerical features for cost prediction."""

        features = []

        # Asset type encoding (one-hot style)
        asset_features = [1 if asset_type == at else 0 for at in AssetType]
        features.extend(asset_features)

        # Signal type encoding
        if signal_type:
            signal_features = [1 if signal_type == st else 0 for st in SignalType]
        else:
            signal_features = [0] * len(SignalType)
        features.extend(signal_features[:10])  # Limit to first 10 for dimensionality

        # Execution mode encoding
        if execution_mode:
            mode_features = [1 if execution_mode == em else 0 for em in ExecutionMode]
        else:
            mode_features = [0] * len(ExecutionMode)
        features.extend(mode_features[:5])  # Limit for dimensionality

        # Scope size (log-transformed to handle large ranges)
        features.append(np.log1p(scope_size) if NUMPY_AVAILABLE else scope_size ** 0.5)

        # Risk level encoding
        if risk_level:
            risk_mapping = {
                IntentRiskLevel.LOW: 0.2,
                IntentRiskLevel.MEDIUM: 0.4,
                IntentRiskLevel.HIGH: 0.7,
                IntentRiskLevel.CRITICAL: 0.9
            }
            features.append(risk_mapping.get(risk_level, 0.5))
        else:
            features.append(0.5)

        # Time sensitivity encoding
        time_mapping = {
            "low": 0.3,
            "normal": 0.5,
            "high": 0.8,
            "critical": 1.0
        }
        features.append(time_mapping.get(time_sensitivity, 0.5))

        # Data quality encoding
        quality_mapping = {
            "basic": 0.3,
            "standard": 0.5,
            "verified": 0.7,
            "premium": 1.0
        }
        features.append(quality_mapping.get(data_quality, 0.5))

        # Control contract features
        if control and control.budget:
            budget_intensity = self._calculate_budget_intensity(control.budget)
            features.append(budget_intensity)
            features.append(control.budget.max_runtime_minutes / 480.0)  # Normalized
            features.append(control.budget.max_records / 50000.0)  # Normalized
        else:
            features.extend([0.5, 0.5, 0.5])  # Default values

        return features

    async def _calculate_base_cost_estimate(
        self,
        asset_type: AssetType,
        signal_type: Optional[SignalType],
        execution_mode: Optional[ExecutionMode],
        scope_size: int,
        control: Optional[ScrapeControlContract]
    ) -> float:
        """Calculate base cost estimate using existing intelligence."""

        # Use signal cost estimation as foundation
        if signal_type:
            cost_estimate = calculate_signal_cost_estimate(
                asset_type, signal_type,
                data_quality="standard"
            )
            base_signal_cost = cost_estimate.get("total_estimated_cost", 2.5)
        else:
            # Fallback to average signal cost
            base_signal_cost = 2.5

        # Apply execution mode multipliers
        mode_multiplier = 1.0
        if execution_mode:
            mode_multipliers = {
                ExecutionMode.PRECISION_SEARCH: 1.5,
                ExecutionMode.TARGETED_LOOKUP: 1.2,
                ExecutionMode.DISCOVERY_SCRAPE: 0.9,
                ExecutionMode.COMPREHENSIVE_SURVEY: 0.8,
                ExecutionMode.EXHAUSTIVE_ANALYSIS: 1.2,
                ExecutionMode.COMPLIANCE_AUDIT: 2.0,
                ExecutionMode.CRISIS_MODE: 1.5,
                ExecutionMode.RAPID_RESPONSE: 1.3
            }
            mode_multiplier = mode_multipliers.get(execution_mode, 1.0)

        # Apply scope scaling
        scope_multiplier = self._calculate_scope_multiplier(scope_size)

        # Apply control contract adjustments
        control_multiplier = 1.0
        if control:
            control_multiplier = await self._calculate_control_multiplier(control)

        base_cost = base_signal_cost * mode_multiplier * scope_multiplier * control_multiplier

        return max(0.1, base_cost)  # Minimum cost threshold

    def _calculate_scope_multiplier(self, scope_size: int) -> float:
        """Calculate cost multiplier based on scope size."""
        if scope_size <= 1:
            return 1.0
        elif scope_size <= 5:
            return 1.1
        elif scope_size <= 20:
            return 1.3
        elif scope_size <= 100:
            return 1.6
        elif scope_size <= 1000:
            return 2.2
        elif scope_size <= 10000:
            return 3.5
        else:
            return 5.0

    async def _calculate_control_multiplier(self, control: ScrapeControlContract) -> float:
        """Calculate cost multiplier based on control contract."""
        multiplier = 1.0

        # Budget intensity adjustment
        if control.budget:
            budget_intensity = self._calculate_budget_intensity(control.budget)
            if budget_intensity > 0.8:
                multiplier *= 1.3  # High budget operations cost more
            elif budget_intensity < 0.3:
                multiplier *= 0.8  # Low budget operations cost less

        # Geography scope adjustment
        if control.intent.geography and len(control.intent.geography) > 5:
            multiplier *= 1.4  # Broad geography increases cost

        # Source count adjustment
        if control.intent.sources and len(control.intent.sources) > 3:
            multiplier *= 1.2  # Multiple sources increase cost

        return multiplier

    def _calculate_budget_intensity(self, budget: ScrapeBudget) -> float:
        """Calculate budget intensity score (0-1)."""
        if not budget:
            return 0.5

        time_intensity = min(budget.max_runtime_minutes / 480, 1.0)
        page_intensity = min(budget.max_pages / 2000, 1.0)
        record_intensity = min(budget.max_records / 50000, 1.0)

        return (time_intensity * 0.4 + page_intensity * 0.3 + record_intensity * 0.3)

    async def _apply_ml_prediction(self, features: List[float], base_cost: float) -> float:
        """Apply ML model prediction adjustment."""
        if not ML_AVAILABLE or not self.cost_model:
            return 1.0  # No adjustment

        try:
            # Scale features
            if self.feature_scaler and hasattr(self.feature_scaler, 'transform'):
                scaled_features = self.feature_scaler.transform([features])
            else:
                scaled_features = [features]

            # Get prediction
            predicted_cost = self.cost_model.predict(scaled_features)[0]

            # Calculate adjustment factor
            if base_cost > 0:
                adjustment = predicted_cost / base_cost
                # Limit extreme adjustments
                adjustment = max(0.3, min(3.0, adjustment))
                return adjustment
            else:
                return 1.0

        except Exception as e:
            logger.debug(f"ML prediction failed: {e}")
            return 1.0

    def _calculate_prediction_confidence(self, features: List[float], base_cost: float) -> float:
        """Calculate confidence score for cost prediction."""
        base_confidence = 0.7  # Default confidence

        # Feature completeness bonus
        feature_completeness = sum(1 for f in features if f != 0) / len(features)
        base_confidence += feature_completeness * 0.2

        # Historical data bonus
        if len(self.cost_history) > 50:
            base_confidence += 0.1

        # ML model bonus
        if ML_AVAILABLE and self.cost_model:
            base_confidence += 0.1

        # Cost magnitude penalty (higher costs are harder to predict accurately)
        if base_cost > 100:
            magnitude_penalty = min(0.2, (base_cost - 100) / 1000)
            base_confidence -= magnitude_penalty

        return max(0.1, min(1.0, base_confidence))

    def _calculate_cost_range(self, predicted_cost: float, confidence: float) -> Tuple[float, float]:
        """Calculate cost prediction range based on confidence."""
        uncertainty_factor = (1.0 - confidence) * 0.5  # 0-50% uncertainty range
        uncertainty_amount = predicted_cost * uncertainty_factor

        low_range = max(0.1, predicted_cost - uncertainty_amount)
        high_range = predicted_cost + uncertainty_amount

        return (low_range, high_range)

    async def _generate_cost_breakdown(
        self,
        asset_type: AssetType,
        signal_type: Optional[SignalType],
        execution_mode: Optional[ExecutionMode],
        scope_size: int,
        risk_level: Optional[IntentRiskLevel],
        intent_category: Optional[IntentCategory],
        control: Optional[ScrapeControlContract],
        total_cost: float
    ) -> Dict[str, float]:
        """Generate detailed cost breakdown."""

        breakdown = {}

        # Signal acquisition costs
        if signal_type:
            signal_weight = get_signal_cost_weight(signal_type)
            breakdown['signal_acquisition'] = signal_weight * scope_size * 0.7
        else:
            breakdown['signal_acquisition'] = total_cost * 0.4

        # Execution infrastructure costs
        breakdown['infrastructure'] = total_cost * 0.25

        # Compliance and legal costs
        compliance_multiplier = 1.0
        if risk_level == IntentRiskLevel.CRITICAL:
            compliance_multiplier = 2.0
        elif risk_level == IntentRiskLevel.HIGH:
            compliance_multiplier = 1.5
        elif intent_category == IntentCategory.LEGAL:
            compliance_multiplier = 1.3

        breakdown['compliance_legal'] = total_cost * 0.15 * compliance_multiplier

        # Data quality and validation costs
        quality_multiplier = 1.0
        if intent_category == IntentCategory.COMPLIANCE:
            quality_multiplier = 1.8
        elif risk_level == IntentRiskLevel.CRITICAL:
            quality_multiplier = 1.4

        breakdown['quality_validation'] = total_cost * 0.1 * quality_multiplier

        # Operational overhead
        breakdown['operational_overhead'] = total_cost * 0.1

        # Risk contingency
        risk_contingency = 0.0
        if risk_level:
            risk_multipliers = {
                IntentRiskLevel.LOW: 0.05,
                IntentRiskLevel.MEDIUM: 0.1,
                IntentRiskLevel.HIGH: 0.2,
                IntentRiskLevel.CRITICAL: 0.3
            }
            risk_contingency = total_cost * risk_multipliers.get(risk_level, 0.1)

        breakdown['risk_contingency'] = risk_contingency

        # Ensure breakdown sums to total
        current_sum = sum(breakdown.values())
        if current_sum > 0:
            scale_factor = total_cost / current_sum
            breakdown = {k: v * scale_factor for k, v in breakdown.items()}

        return {k: round(v, 2) for k, v in breakdown.items()}

    def _calculate_risk_adjustments(
        self,
        risk_level: Optional[IntentRiskLevel],
        intent_category: Optional[IntentCategory],
        asset_type: AssetType,
        total_cost: float
    ) -> Dict[str, float]:
        """Calculate risk-based cost adjustments."""

        adjustments = {}

        if risk_level:
            # Risk level adjustments
            risk_multipliers = {
                IntentRiskLevel.LOW: 0.95,      # Slight discount for low risk
                IntentRiskLevel.MEDIUM: 1.0,    # Baseline
                IntentRiskLevel.HIGH: 1.15,    # Premium for high risk
                IntentRiskLevel.CRITICAL: 1.3   # Significant premium for critical
            }
            adjustments['risk_level_adjustment'] = risk_multipliers.get(risk_level, 1.0)

        if intent_category:
            # Category-specific adjustments
            category_multipliers = {
                IntentCategory.PERSONAL: 1.1,    # Privacy compliance costs
                IntentCategory.FINANCIAL: 1.2,   # Financial regulation costs
                IntentCategory.LEGAL: 1.25,     # Legal compliance costs
                IntentCategory.COMPLIANCE: 1.4, # Maximum compliance costs
                IntentCategory.PROPERTY: 1.05,  # Moderate compliance
                IntentCategory.EVENT: 0.95,     # Lower compliance requirements
                IntentCategory.INTELLIGENCE: 1.0 # Baseline
            }
            adjustments['category_adjustment'] = category_multipliers.get(intent_category, 1.0)

        if asset_type == AssetType.PERSON:
            adjustments['privacy_compliance'] = 1.15  # Additional privacy costs
        elif asset_type in [AssetType.COMPANY, AssetType.COMMERCIAL_PROPERTY]:
            adjustments['enterprise_compliance'] = 1.2  # Enterprise compliance costs

        return adjustments

    async def _generate_optimization_recommendations(
        self,
        asset_type: AssetType,
        signal_type: Optional[SignalType],
        execution_mode: Optional[ExecutionMode],
        scope_size: int,
        risk_level: Optional[IntentRiskLevel],
        intent_category: Optional[IntentCategory],
        predicted_cost: float,
        confidence_score: float
    ) -> List[str]:
        """Generate cost optimization recommendations."""

        recommendations = []

        # Scope-based optimizations
        if scope_size > 1000:
            recommendations.append("Consider phased execution to reduce peak resource costs by 20-30%")
            recommendations.append("Implement sampling approach for initial cost estimation")

        if scope_size <= 10:
            recommendations.append("Bundle with other small operations for efficiency gains")

        # Mode-based optimizations
        if execution_mode == ExecutionMode.PRECISION_SEARCH and scope_size > 5:
            recommendations.append("Consider TARGETED_LOOKUP mode for 15-25% cost savings on medium scopes")

        if execution_mode == ExecutionMode.EXHAUSTIVE_ANALYSIS:
            recommendations.append("Evaluate if COMPREHENSIVE_SURVEY mode meets requirements with 30% cost savings")

        # Risk-based optimizations
        if risk_level == IntentRiskLevel.CRITICAL:
            recommendations.append("High-risk operations may justify premium sources for risk reduction")
        elif risk_level == IntentRiskLevel.LOW:
            recommendations.append("Consider cost-optimized sources for low-risk operations")

        # Asset type optimizations
        if asset_type == AssetType.PERSON:
            recommendations.append("Leverage public records for cost-effective personal data acquisition")
        elif asset_type == AssetType.COMPANY:
            recommendations.append("Utilize SEC filings and public company data for cost efficiency")

        # Confidence-based recommendations
        if confidence_score < 0.7:
            recommendations.append("Low confidence estimate - consider pilot execution for cost validation")
            recommendations.append("Implement monitoring and cost controls for budget protection")

        if predicted_cost > 100:
            recommendations.append("High-cost operation - consider distributed execution across time periods")
            recommendations.append("Evaluate alternative data sources for cost comparison")

        return recommendations[:6]  # Limit to top recommendations

    async def _generate_alternative_scenarios(
        self,
        asset_type: AssetType,
        signal_type: Optional[SignalType],
        scope_size: int,
        risk_level: Optional[IntentRiskLevel],
        intent_category: Optional[IntentCategory],
        time_sensitivity: str,
        data_quality: str,
        current_cost: float
    ) -> Dict[str, Dict[str, Any]]:
        """Generate alternative execution scenarios with cost comparisons."""

        scenarios = {}

        # Time sensitivity alternatives
        if time_sensitivity == "normal":
            # Slower execution scenario
            slow_cost = await self.predict_cost(
                asset_type, signal_type, None, scope_size,
                risk_level, intent_category, "low", data_quality
            )
            scenarios['cost_optimized'] = {
                'description': 'Lower time sensitivity for cost optimization',
                'cost': slow_cost.predicted_cost,
                'savings': current_cost - slow_cost.predicted_cost,
                'trade_offs': ['Longer execution time', 'Lower urgency']
            }

        # Quality alternatives
        if data_quality == "standard":
            # Premium quality scenario
            premium_cost = await self.predict_cost(
                asset_type, signal_type, None, scope_size,
                risk_level, intent_category, time_sensitivity, "premium"
            )
            scenarios['quality_optimized'] = {
                'description': 'Higher data quality for better results',
                'cost': premium_cost.predicted_cost,
                'premium': premium_cost.predicted_cost - current_cost,
                'trade_offs': ['Higher cost', 'Better data quality']
            }

            # Basic quality scenario
            basic_cost = await self.predict_cost(
                asset_type, signal_type, None, scope_size,
                risk_level, intent_category, time_sensitivity, "basic"
            )
            scenarios['budget_optimized'] = {
                'description': 'Lower quality requirements for cost savings',
                'cost': basic_cost.predicted_cost,
                'savings': current_cost - basic_cost.predicted_cost,
                'trade_offs': ['Lower data quality', 'Higher risk of issues']
            }

        # Scope alternatives
        if scope_size > 100:
            # Reduced scope scenario
            reduced_scope = min(scope_size // 2, 50)
            reduced_cost = await self.predict_cost(
                asset_type, signal_type, None, reduced_scope,
                risk_level, intent_category, time_sensitivity, data_quality
            )
            scenarios['scope_optimized'] = {
                'description': f'Reduced scope to {reduced_scope} targets',
                'cost': reduced_cost.predicted_cost,
                'savings': current_cost - reduced_cost.predicted_cost,
                'trade_offs': ['Reduced coverage', 'Lower cost']
            }

        return scenarios

    async def optimize_cost(
        self,
        asset_type: AssetType,
        signal_type: Optional[SignalType] = None,
        current_cost: float = 0.0,
        constraints: Optional[Dict[str, Any]] = None,
        optimization_strategy: CostOptimizationStrategy = CostOptimizationStrategy.BALANCE_COST_VALUE
    ) -> CostOptimizationPlan:
        """
        Generate comprehensive cost optimization plan.

        Args:
            asset_type: Type of asset
            signal_type: Type of signal (optional)
            current_cost: Current estimated cost
            constraints: Optimization constraints
            optimization_strategy: Optimization approach

        Returns:
            Detailed optimization plan with recommendations
        """

        plan_id = self._generate_optimization_plan_id(asset_type, signal_type, current_cost)

        # Analyze current situation
        current_analysis = await self._analyze_current_cost_structure(
            asset_type, signal_type, current_cost, constraints
        )

        # Generate optimization recommendations
        recommendations = await self._generate_cost_optimization_recommendations(
            asset_type, signal_type, current_cost, constraints, optimization_strategy
        )

        # Calculate optimized cost
        optimized_cost = await self._calculate_optimized_cost(
            asset_type, signal_type, current_cost, recommendations
        )

        # Calculate savings and ROI
        cost_savings = current_cost - optimized_cost
        savings_percentage = (cost_savings / current_cost * 100) if current_cost > 0 else 0

        # Assess implementation priority
        implementation_priority = self._assess_implementation_priority(
            cost_savings, optimization_strategy, constraints
        )

        # Calculate expected ROI
        expected_roi = self._calculate_expected_roi(
            cost_savings, recommendations, asset_type, signal_type
        )

        # Generate risk assessment
        risk_assessment = await self._assess_optimization_risks(
            recommendations, asset_type, signal_type, optimization_strategy
        )

        # Estimate timeline
        timeline_estimate = self._estimate_implementation_timeline(recommendations)

        return CostOptimizationPlan(
            plan_id=plan_id,
            original_cost=current_cost,
            optimized_cost=round(optimized_cost, 2),
            cost_savings=round(cost_savings, 2),
            savings_percentage=round(savings_percentage, 1),
            optimization_strategy=optimization_strategy,
            recommended_changes=recommendations,
            implementation_priority=implementation_priority,
            expected_roi=round(expected_roi, 1),
            risk_assessment=risk_assessment,
            timeline_estimate=timeline_estimate
        )

    def _generate_optimization_plan_id(self, *args) -> str:
        """Generate unique optimization plan identifier."""
        content = "_".join(str(arg) for arg in args if arg is not None)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    async def _analyze_current_cost_structure(
        self,
        asset_type: AssetType,
        signal_type: Optional[SignalType],
        current_cost: float,
        constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze current cost structure for optimization opportunities."""
        analysis = {
            'cost_distribution': {},
            'efficiency_score': 0.0,
            'optimization_potential': 0.0,
            'constraint_impacts': {}
        }

        # Analyze cost distribution
        if signal_type:
            cost_estimate = calculate_signal_cost_estimate(asset_type, signal_type)
            analysis['cost_distribution'] = cost_estimate

        # Calculate efficiency score (simplified)
        base_efficiency = 0.7  # Default efficiency
        if constraints:
            if 'time_limit' in constraints:
                base_efficiency *= 0.9  # Time constraints reduce efficiency
            if 'quality_minimum' in constraints:
                base_efficiency *= 1.1  # Quality requirements can improve efficiency

        analysis['efficiency_score'] = base_efficiency

        # Estimate optimization potential
        optimization_potential = min(0.4, (1.0 - base_efficiency) + 0.1)
        analysis['optimization_potential'] = optimization_potential

        return analysis

    async def _generate_cost_optimization_recommendations(
        self,
        asset_type: AssetType,
        signal_type: Optional[SignalType],
        current_cost: float,
        constraints: Optional[Dict[str, Any]],
        optimization_strategy: CostOptimizationStrategy
    ) -> List[Dict[str, Any]]:
        """Generate specific cost optimization recommendations."""

        recommendations = []

        # Strategy-specific recommendations
        if optimization_strategy == CostOptimizationStrategy.MINIMIZE_COST:
            recommendations.extend([
                {
                    'type': 'execution_mode',
                    'change': 'Switch to cost-optimized execution mode',
                    'estimated_savings': current_cost * 0.15,
                    'implementation_effort': 'medium',
                    'risk_level': 'low'
                },
                {
                    'type': 'source_selection',
                    'change': 'Use lower-cost data sources',
                    'estimated_savings': current_cost * 0.12,
                    'implementation_effort': 'low',
                    'risk_level': 'medium'
                }
            ])

        elif optimization_strategy == CostOptimizationStrategy.MAXIMIZE_VALUE:
            recommendations.extend([
                {
                    'type': 'quality_improvement',
                    'change': 'Invest in higher quality data sources',
                    'estimated_cost_increase': current_cost * 0.2,
                    'value_benefit': 'Improved data accuracy and reliability',
                    'implementation_effort': 'high',
                    'risk_level': 'low'
                }
            ])

        elif optimization_strategy == CostOptimizationStrategy.BALANCE_COST_VALUE:
            recommendations.extend([
                {
                    'type': 'batch_optimization',
                    'change': 'Optimize batch sizes for efficiency',
                    'estimated_savings': current_cost * 0.08,
                    'implementation_effort': 'low',
                    'risk_level': 'low'
                },
                {
                    'type': 'source_diversification',
                    'change': 'Use multiple cost-effective sources',
                    'estimated_savings': current_cost * 0.1,
                    'implementation_effort': 'medium',
                    'risk_level': 'low'
                }
            ])

        # Asset type specific recommendations
        if asset_type == AssetType.PERSON:
            recommendations.append({
                'type': 'source_optimization',
                'change': 'Prioritize free public records over paid sources',
                'estimated_savings': current_cost * 0.18,
                'implementation_effort': 'low',
                'risk_level': 'low'
            })

        elif asset_type == AssetType.COMPANY:
            recommendations.append({
                'type': 'regulatory_data',
                'change': 'Leverage SEC and regulatory filings',
                'estimated_savings': current_cost * 0.25,
                'implementation_effort': 'medium',
                'risk_level': 'low'
            })

        # Signal type specific recommendations
        if signal_type == SignalType.LIEN:
            recommendations.append({
                'type': 'county_records',
                'change': 'Focus on county clerk records for cost efficiency',
                'estimated_savings': current_cost * 0.15,
                'implementation_effort': 'low',
                'risk_level': 'low'
            })

        return recommendations[:8]  # Limit to top recommendations

    async def _calculate_optimized_cost(
        self,
        asset_type: AssetType,
        signal_type: Optional[SignalType],
        current_cost: float,
        recommendations: List[Dict[str, Any]]
    ) -> float:
        """Calculate optimized cost based on recommendations."""

        optimized_cost = current_cost

        for rec in recommendations:
            if 'estimated_savings' in rec:
                optimized_cost -= rec['estimated_savings']
            elif 'estimated_cost_increase' in rec:
                # For value-maximizing recommendations, we might accept cost increases
                optimized_cost += rec['estimated_cost_increase']

        return max(current_cost * 0.5, optimized_cost)  # Minimum 50% reduction

    def _assess_implementation_priority(
        self,
        cost_savings: float,
        optimization_strategy: CostOptimizationStrategy,
        constraints: Optional[Dict[str, Any]]
    ) -> str:
        """Assess implementation priority for optimization plan."""

        if cost_savings > 1000:
            return "critical"
        elif cost_savings > 500:
            return "high"
        elif cost_savings > 100:
            return "medium"
        elif cost_savings > 25:
            return "low"
        else:
            return "optional"

    def _calculate_expected_roi(
        self,
        cost_savings: float,
        recommendations: List[Dict[str, Any]],
        asset_type: AssetType,
        signal_type: Optional[SignalType]
    ) -> float:
        """Calculate expected ROI for optimization plan."""

        # Estimate implementation effort
        effort_score = sum(
            {'low': 1, 'medium': 2, 'high': 3}.get(rec.get('implementation_effort', 'medium'), 2)
            for rec in recommendations
        ) / len(recommendations) if recommendations else 2

        # Estimate value benefit
        value_multiplier = 1.0
        if asset_type == AssetType.COMPANY:
            value_multiplier = 1.5  # Higher value for company data
        elif signal_type == SignalType.FORECLOSURE:
            value_multiplier = 2.0  # High value for distressed assets

        # Calculate ROI (simplified)
        implementation_cost = effort_score * 50  # Estimated implementation cost
        annual_savings = cost_savings * 12  # Assume monthly operations

        if implementation_cost > 0:
            roi = ((annual_savings - implementation_cost) / implementation_cost) * 100 * value_multiplier
            return max(-50, min(500, roi))  # Reasonable bounds

        return 0.0

    async def _assess_optimization_risks(
        self,
        recommendations: List[Dict[str, Any]],
        asset_type: AssetType,
        signal_type: Optional[SignalType],
        optimization_strategy: CostOptimizationStrategy
    ) -> Dict[str, Any]:
        """Assess risks associated with optimization recommendations."""

        risk_assessment = {
            'overall_risk_level': 'low',
            'risk_factors': [],
            'mitigation_strategies': [],
            'monitoring_recommendations': []
        }

        high_risk_count = 0

        for rec in recommendations:
            risk_level = rec.get('risk_level', 'low')

            if risk_level == 'high':
                high_risk_count += 1
                risk_assessment['risk_factors'].append(f"High-risk change: {rec.get('change', 'Unknown')}")

            elif risk_level == 'medium':
                risk_assessment['risk_factors'].append(f"Medium-risk change: {rec.get('change', 'Unknown')}")

        # Determine overall risk level
        if high_risk_count > 2:
            risk_assessment['overall_risk_level'] = 'high'
        elif high_risk_count > 0 or len(recommendations) > 5:
            risk_assessment['overall_risk_level'] = 'medium'

        # Add mitigation strategies
        if risk_assessment['overall_risk_level'] in ['medium', 'high']:
            risk_assessment['mitigation_strategies'].extend([
                "Implement phased rollout with monitoring",
                "Establish rollback procedures",
                "Conduct pilot testing before full implementation"
            ])

        # Add monitoring recommendations
        risk_assessment['monitoring_recommendations'].extend([
            "Track cost savings vs. projected amounts",
            "Monitor data quality and completeness",
            "Monitor execution time and resource usage",
            "Track any increase in error rates or failures"
        ])

        return risk_assessment

    def _estimate_implementation_timeline(self, recommendations: List[Dict[str, Any]]) -> str:
        """Estimate implementation timeline for optimization plan."""

        effort_levels = [rec.get('implementation_effort', 'medium') for rec in recommendations]
        effort_scores = [{'low': 1, 'medium': 2, 'high': 3}.get(level, 2) for level in effort_levels]

        avg_effort = sum(effort_scores) / len(effort_scores) if effort_scores else 2
        max_effort = max(effort_scores) if effort_scores else 2

        # Timeline estimation
        if max_effort == 1 and avg_effort <= 1.5:
            return "1-2 weeks"
        elif max_effort <= 2 and avg_effort <= 2.5:
            return "2-4 weeks"
        elif max_effort == 3 or avg_effort > 2.5:
            return "1-2 months"
        else:
            return "2-3 months"

    async def analyze_budget(
        self,
        budget: float,
        projected_operations: List[Dict[str, Any]],
        risk_tolerance: str = "medium"
    ) -> BudgetAnalysis:
        """
        Perform comprehensive budget analysis and forecasting.

        Args:
            budget: Total available budget
            projected_operations: List of planned operations with cost estimates
            risk_tolerance: Risk tolerance level ("low", "medium", "high")

        Returns:
            Comprehensive budget analysis with recommendations
        """

        analysis_id = self._generate_budget_analysis_id(budget, projected_operations)

        # Calculate total projected cost
        total_projected_cost = sum(op.get('estimated_cost', 0) for op in projected_operations)
        budget_utilization = (total_projected_cost / budget * 100) if budget > 0 else 0

        # Calculate variance from historical averages
        cost_variance = await self._calculate_budget_variance(total_projected_cost, projected_operations)

        # Assess risk of overspend
        risk_of_overspend = self._assess_overspend_risk(budget_utilization, cost_variance, risk_tolerance)

        # Identify cost drivers
        cost_drivers = await self._identify_cost_drivers(projected_operations)

        # Generate optimization opportunities
        budget_optimizations = await self._generate_budget_optimizations(
            budget, total_projected_cost, projected_operations, risk_tolerance
        )

        # Calculate forecasting accuracy
        forecasting_accuracy = self._calculate_forecasting_accuracy(projected_operations)

        # Generate recommendations
        recommendations = await self._generate_budget_recommendations(
            budget, total_projected_cost, budget_utilization, risk_of_overspend, projected_operations
        )

        return BudgetAnalysis(
            analysis_id=analysis_id,
            total_budget=budget,
            projected_cost=round(total_projected_cost, 2),
            budget_utilization=round(budget_utilization, 1),
            cost_variance=round(cost_variance, 1),
            risk_of_overspend=risk_of_overspend,
            cost_drivers=cost_drivers,
            budget_optimization_opportunities=budget_optimizations,
            forecasting_accuracy=round(forecasting_accuracy, 1),
            recommendations=recommendations
        )

    def _generate_budget_analysis_id(self, budget: float, operations: List[Dict[str, Any]]) -> str:
        """Generate unique budget analysis identifier."""
        content = f"{budget}_{len(operations)}_{sum(op.get('estimated_cost', 0) for op in operations)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    async def _calculate_budget_variance(
        self,
        total_projected_cost: float,
        projected_operations: List[Dict[str, Any]]
    ) -> float:
        """Calculate variance from historical budget performance."""

        if not self.cost_history:
            return 0.0

        # Calculate average historical cost per operation
        historical_costs = [record.get('actual_cost', 0) for record in self.cost_history[-100:]]
        avg_historical_cost = np_mean(historical_costs) if historical_costs else 0

        # Calculate expected cost based on operation count
        operation_count = len(projected_operations)
        expected_cost = avg_historical_cost * operation_count

        # Calculate variance percentage
        if expected_cost > 0:
            variance = ((total_projected_cost - expected_cost) / expected_cost) * 100
            return variance

        return 0.0

    def _assess_overspend_risk(
        self,
        budget_utilization: float,
        cost_variance: float,
        risk_tolerance: str
    ) -> str:
        """Assess risk of budget overspend."""

        risk_score = 0

        # Budget utilization risk
        if budget_utilization > 100:
            risk_score += 3
        elif budget_utilization > 90:
            risk_score += 2
        elif budget_utilization > 75:
            risk_score += 1

        # Cost variance risk
        if abs(cost_variance) > 50:
            risk_score += 2
        elif abs(cost_variance) > 25:
            risk_score += 1

        # Adjust for risk tolerance
        if risk_tolerance == "low":
            risk_score += 1
        elif risk_tolerance == "high":
            risk_score -= 1

        # Determine risk level
        if risk_score >= 4:
            return "critical"
        elif risk_score >= 2:
            return "high"
        elif risk_score >= 1:
            return "medium"
        else:
            return "low"

    async def _identify_cost_drivers(self, projected_operations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Identify primary cost drivers in the budget."""

        cost_drivers = []

        # Analyze by operation type
        operation_types = {}
        for op in projected_operations:
            op_type = op.get('operation_type', 'unknown')
            cost = op.get('estimated_cost', 0)
            if op_type not in operation_types:
                operation_types[op_type] = {'count': 0, 'total_cost': 0}
            operation_types[op_type]['count'] += 1
            operation_types[op_type]['total_cost'] += cost

        # Find top cost drivers
        sorted_drivers = sorted(operation_types.items(),
                               key=lambda x: x[1]['total_cost'], reverse=True)

        for driver_type, data in sorted_drivers[:5]:
            percentage = (data['total_cost'] / sum(op.get('estimated_cost', 0)
                                                   for op in projected_operations) * 100)
            cost_drivers.append({
                'driver': driver_type,
                'cost': f"${data['total_cost']:.2f}",
                'percentage': ".1f",
                'operation_count': data['count']
            })

        return cost_drivers

    async def _generate_budget_optimizations(
        self,
        budget: float,
        total_projected_cost: float,
        projected_operations: List[Dict[str, Any]],
        risk_tolerance: str
    ) -> List[str]:
        """Generate budget optimization opportunities."""

        optimizations = []
        overspend_amount = total_projected_cost - budget

        if overspend_amount > 0:
            optimizations.extend([
                f"Reduce budget overspend by ${overspend_amount:.2f} ({overspend_amount/budget*100:.1f}%)",
                "Prioritize high-value operations and defer lower-priority ones",
                "Optimize execution modes for cost efficiency",
                "Consider phased implementation to spread costs over time"
            ])

        # General optimizations
        optimizations.extend([
            "Implement cost monitoring and alerts for budget control",
            "Use historical performance data for more accurate forecasting",
            "Consider bulk operations for volume discounts",
            "Evaluate alternative data sources for cost comparison"
        ])

        if risk_tolerance != "low":
            optimizations.append("Consider reducing data quality requirements for cost savings")

        return optimizations[:6]

    def _calculate_forecasting_accuracy(self, projected_operations: List[Dict[str, Any]]) -> float:
        """Calculate forecasting accuracy based on historical data."""

        if len(self.cost_history) < 10:
            return 50.0  # Default accuracy when limited data

        # Calculate accuracy based on historical prediction vs actual
        prediction_errors = []
        for record in self.cost_history[-50:]:  # Last 50 records
            predicted = record.get('predicted_cost', 0)
            actual = record.get('actual_cost', 0)
            if predicted > 0 and actual > 0:
                error = abs(predicted - actual) / predicted
                prediction_errors.append(error)

        if prediction_errors:
            avg_error = np_mean(prediction_errors)
            accuracy = (1.0 - avg_error) * 100
            return max(0, min(100, accuracy))

        return 50.0

    async def _generate_budget_recommendations(
        self,
        budget: float,
        total_projected_cost: float,
        budget_utilization: float,
        risk_of_overspend: str,
        projected_operations: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate budget management recommendations."""

        recommendations = []

        if budget_utilization > 100:
            recommendations.extend([
                "CRITICAL: Budget exceeded - immediate action required",
                "Cancel or postpone non-essential operations",
                "Negotiate extended budget or seek additional funding",
                "Implement emergency cost controls"
            ])

        elif budget_utilization > 90:
            recommendations.extend([
                "HIGH PRIORITY: Approaching budget limit",
                "Review and optimize remaining operations",
                "Consider reducing scope of planned operations",
                "Implement strict cost monitoring"
            ])

        elif budget_utilization < 50:
            recommendations.extend([
                "Budget utilization is low - consider expanding operations",
                "Opportunity to take on additional high-value projects",
                "Review if budget allocation is appropriate"
            ])

        # Risk-based recommendations
        if risk_of_overspend in ["high", "critical"]:
            recommendations.extend([
                "Implement contingency budget planning",
                "Establish cost variance monitoring and alerts",
                "Develop budget risk mitigation strategies"
            ])

        # Operational recommendations
        operation_count = len(projected_operations)
        if operation_count > 20:
            recommendations.append("High operation volume - consider automation and batch processing")

        # Forecasting recommendations
        forecasting_accuracy = self._calculate_forecasting_accuracy(projected_operations)
        if forecasting_accuracy < 70:
            recommendations.append("Improve forecasting accuracy by collecting more historical data")

        return recommendations[:8]

    def record_cost_outcome(
        self,
        prediction: CostPrediction,
        actual_cost: float,
        success: bool,
        notes: Optional[str] = None
    ):
        """Record actual cost outcome for learning and model improvement."""

        outcome_record = {
            'prediction_id': prediction.prediction_id,
            'predicted_cost': prediction.predicted_cost,
            'actual_cost': actual_cost,
            'cost_variance': actual_cost - prediction.predicted_cost,
            'variance_percentage': ((actual_cost - prediction.predicted_cost) / prediction.predicted_cost * 100) if prediction.predicted_cost > 0 else 0,
            'success': success,
            'confidence_score': prediction.confidence_score,
            'timestamp': datetime.utcnow(),
            'notes': notes,
            **prediction.prediction_metadata
        }

        self.cost_history.append(outcome_record)

        # Update performance metrics
        self._update_performance_metrics(outcome_record)

        # Retrain model if enough data
        if len(self.cost_history) % 50 == 0:  # Retrain every 50 records
            asyncio.create_task(self._retrain_model())

    def _extract_cost_features(self, record: Dict[str, Any]) -> List[float]:
        """Extract features from cost record for ML training."""
        return [
            record.get('scope_size', 1),
            record.get('risk_level_numeric', 0.5),
            record.get('time_sensitivity_numeric', 0.5),
            record.get('data_quality_numeric', 0.5),
            record.get('asset_type_encoded', 0),
            record.get('signal_type_encoded', 0),
            record.get('execution_mode_encoded', 0)
        ]

    def _update_performance_metrics(self, outcome_record: Dict[str, Any]):
        """Update performance tracking metrics."""
        # This would update various performance indicators
        # For now, just track basic metrics
        pass

    async def _retrain_model(self):
        """Retrain the ML model with new data."""
        await self._initialize_models()

    def detect_cost_anomalies(self, recent_costs: List[float], threshold: float = 2.0) -> List[int]:
        """
        Detect cost anomalies using statistical methods.

        Args:
            recent_costs: List of recent cost observations
            threshold: Anomaly detection threshold (standard deviations)

        Returns:
            List of indices of anomalous costs
        """
        if not NUMPY_AVAILABLE or len(recent_costs) < 10:
            return []

        try:
            if self.anomaly_detector and ML_AVAILABLE:
                # Use ML-based anomaly detection
                predictions = self.anomaly_detector.fit_predict(np_array(recent_costs).reshape(-1, 1))
                anomalies = [i for i, pred in enumerate(predictions) if pred == -1]
            else:
                # Use statistical method
                mean_cost = np_mean(recent_costs)
                std_cost = np.std(recent_costs) if len(recent_costs) > 1 else 0

                if std_cost > 0:
                    anomalies = [i for i, cost in enumerate(recent_costs)
                               if abs(cost - mean_cost) > threshold * std_cost]
                else:
                    anomalies = []

            return anomalies

        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return []

    def get_predictor_stats(self) -> Dict[str, Any]:
        """Get comprehensive predictor statistics."""
        total_predictions = len([r for r in self.cost_history if 'prediction_id' in r])

        if total_predictions == 0:
            return {"total_predictions": 0}

        # Calculate accuracy metrics
        prediction_errors = []
        successful_predictions = 0

        for record in self.cost_history[-200:]:  # Last 200 records
            if 'predicted_cost' in record and 'actual_cost' in record:
                predicted = record['predicted_cost']
                actual = record['actual_cost']

                if predicted > 0:
                    error = abs(predicted - actual) / predicted
                    prediction_errors.append(error)

                    # Consider prediction successful if within 25% error
                    if error <= 0.25:
                        successful_predictions += 1

        avg_error = np_mean(prediction_errors) if prediction_errors else 0
        accuracy_rate = (successful_predictions / len(prediction_errors) * 100) if prediction_errors else 0

        return {
            "total_predictions": total_predictions,
            "historical_records": len(self.cost_history),
            "average_prediction_error": round(avg_error, 3),
            "prediction_accuracy_rate": round(accuracy_rate, 1),
            "ml_model_available": ML_AVAILABLE and self.cost_model is not None,
            "anomaly_detector_available": ML_AVAILABLE and self.anomaly_detector is not None,
            "last_model_training": self.last_training_date.isoformat() if self.last_training_date else None,
            "cost_variance_trends": self._analyze_cost_variance_trends(),
            "prediction_confidence_distribution": self._analyze_confidence_distribution()
        }

    def _analyze_cost_variance_trends(self) -> Dict[str, Any]:
        """Analyze cost variance trends over time."""
        if len(self.cost_history) < 10:
            return {"insufficient_data": True}

        recent_records = self.cost_history[-50:]
        variances = [abs(r.get('variance_percentage', 0)) for r in recent_records if 'variance_percentage' in r]

        if variances:
            avg_variance = np_mean(variances)
            variance_trend = "improving" if avg_variance < 15 else "stable" if avg_variance < 25 else "concerning"

            return {
                "average_variance": round(avg_variance, 1),
                "variance_trend": variance_trend,
                "high_variance_count": sum(1 for v in variances if v > 30)
            }

        return {"no_variance_data": True}

    def _analyze_confidence_distribution(self) -> Dict[str, Any]:
        """Analyze prediction confidence distribution."""
        confidences = [r.get('confidence_score', 0) for r in self.cost_history[-100:] if 'confidence_score' in r]

        if confidences:
            avg_confidence = np_mean(confidences)
            high_confidence_count = sum(1 for c in confidences if c >= 0.8)
            low_confidence_count = sum(1 for c in confidences if c < 0.6)

            return {
                "average_confidence": round(avg_confidence, 3),
                "high_confidence_predictions": high_confidence_count,
                "low_confidence_predictions": low_confidence_count,
                "confidence_distribution": "healthy" if avg_confidence >= 0.7 else "needs_improvement"
            }

        return {"no_confidence_data": True}


# Global cost predictor instance
_global_cost_predictor = CostPredictor()


# Convenience functions
async def predict_scraping_cost(
    asset_type: AssetType,
    signal_type: Optional[SignalType] = None,
    execution_mode: Optional[ExecutionMode] = None,
    scope_size: int = 1,
    risk_level: Optional[IntentRiskLevel] = None,
    intent_category: Optional[IntentCategory] = None,
    time_sensitivity: str = "normal",
    data_quality: str = "standard",
    control: Optional[ScrapeControlContract] = None
) -> CostPrediction:
    """
    Predict comprehensive scraping costs with ML-enhanced intelligence.

    This is the main entry point for cost prediction in the MJ Data Scraper Suite.
    Provides detailed cost estimates, confidence intervals, optimization recommendations,
    and alternative scenarios for informed decision making.

    Args:
        asset_type: Type of asset being targeted
        signal_type: Type of signal (optional)
        execution_mode: Execution mode (optional)
        scope_size: Number of targets
        risk_level: Risk classification level
        intent_category: Intent category classification
        time_sensitivity: Time requirements ("low", "normal", "high", "critical")
        data_quality: Quality requirements ("basic", "standard", "verified", "premium")
        control: Optional scraping control contract

    Returns:
        Comprehensive CostPrediction with breakdowns, ranges, and recommendations
    """
    return await _global_cost_predictor.predict_cost(
        asset_type, signal_type, execution_mode, scope_size,
        risk_level, intent_category, time_sensitivity, data_quality, control
    )


async def optimize_scraping_cost(
    asset_type: AssetType,
    signal_type: Optional[SignalType] = None,
    current_cost: float = 0.0,
    constraints: Optional[Dict[str, Any]] = None,
    optimization_strategy: CostOptimizationStrategy = CostOptimizationStrategy.BALANCE_COST_VALUE
) -> CostOptimizationPlan:
    """
    Generate comprehensive cost optimization plan.

    Provides actionable recommendations for cost reduction while maintaining
    operational effectiveness and compliance requirements.

    Args:
        asset_type: Type of asset
        signal_type: Type of signal (optional)
        current_cost: Current estimated cost
        constraints: Optimization constraints (time limits, quality requirements, etc.)
        optimization_strategy: Optimization approach

    Returns:
        Detailed CostOptimizationPlan with savings projections and implementation guidance
    """
    return await _global_cost_predictor.optimize_cost(
        asset_type, signal_type, current_cost, constraints, optimization_strategy
    )


async def analyze_scraping_budget(
    budget: float,
    projected_operations: List[Dict[str, Any]],
    risk_tolerance: str = "medium"
) -> BudgetAnalysis:
    """
    Perform comprehensive budget analysis and forecasting.

    Analyzes budget utilization, identifies cost drivers, and provides
    optimization opportunities for budget management.

    Args:
        budget: Total available budget
        projected_operations: List of planned operations with cost estimates
        risk_tolerance: Risk tolerance level ("low", "medium", "high")

    Returns:
        Comprehensive BudgetAnalysis with utilization metrics and recommendations
    """
    return await _global_cost_predictor.analyze_budget(budget, projected_operations, risk_tolerance)


def get_cost_predictor_stats() -> Dict[str, Any]:
    """
    Get comprehensive cost predictor statistics and performance metrics.

    Returns operational metrics for monitoring prediction accuracy, model performance,
    and cost analysis effectiveness across the scraping ecosystem.

    Returns:
        Dict with predictor statistics, accuracy metrics, and performance indicators
    """
    return _global_cost_predictor.get_predictor_stats()


def record_cost_performance(
    prediction: CostPrediction,
    actual_cost: float,
    success: bool,
    notes: Optional[str] = None
):
    """
    Record actual cost performance for model learning and improvement.

    Args:
        prediction: Original cost prediction
        actual_cost: Actual cost incurred
        success: Whether the operation was successful
        notes: Optional notes about the outcome
    """
    _global_cost_predictor.record_cost_outcome(prediction, actual_cost, success, notes)


def detect_cost_anomalies(recent_costs: List[float], threshold: float = 2.0) -> List[int]:
    """
    Detect anomalous costs in recent operations.

    Args:
        recent_costs: List of recent cost observations
        threshold: Anomaly detection threshold in standard deviations

    Returns:
        List of indices of anomalous costs
    """
    return _global_cost_predictor.detect_cost_anomalies(recent_costs, threshold)
