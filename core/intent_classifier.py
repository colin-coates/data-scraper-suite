# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Intent Classification Engine for MJ Data Scraper Suite

Advanced machine learning and rule-based intent classification system that
analyzes scraping requests to determine appropriate governance, risk levels,
and operational parameters for optimal execution.
"""

import asyncio
import logging
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Fallback implementations
    class RandomForestClassifier:
        def fit(self, X, y): pass
        def predict(self, X): return np.zeros(len(X)) if 'numpy' in globals() else [0] * len(X)
        def predict_proba(self, X): return np.zeros((len(X), 2)) if 'numpy' in globals() else [[0.5, 0.5]] * len(X)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Basic array operations
    def np_array(data): return data
    def np_zeros(shape): return [0] * (shape[0] if hasattr(shape, '__len__') else shape)
    def np_mean(data): return sum(data) / len(data) if data else 0

from core.control_models import (
    ScrapeControlContract,
    ScrapeIntent,
    ScrapeBudget,
    ScrapeTempo,
    JobPriority,
    ScraperType
)
from core.mapping.asset_signal_map import (
    get_optimal_sources_for_signal,
    calculate_signal_cost_estimate,
    get_data_freshness_requirement,
    get_signal_cost_weight,
    get_source_reliability_score,
    ASSET_SIGNAL_SOURCES,
    SIGNAL_COST_WEIGHT
)
from core.models.asset_signal import AssetType, SignalType

logger = logging.getLogger(__name__)


class IntentRiskLevel(Enum):
    """Risk classification levels for scraping intents."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IntentCategory(Enum):
    """Categories for classifying scraping intent."""
    PERSONAL = "personal"               # Individual person data
    PROFESSIONAL = "professional"       # Career/business information
    FINANCIAL = "financial"            # Financial records and transactions
    LEGAL = "legal"                   # Court cases, judgments, liens
    PROPERTY = "property"             # Real estate and property records
    EVENT = "event"                  # Life events and announcements
    COMPLIANCE = "compliance"        # Regulatory and compliance data
    INTELLIGENCE = "intelligence"    # General intelligence gathering


class GovernanceRequirement(Enum):
    """Required governance levels for different intent types."""
    BASIC = "basic"                   # Standard approval process
    ENHANCED = "enhanced"            # Additional oversight
    RESTRICTED = "restricted"        # Limited execution windows
    CONTROLLED = "controlled"        # Strict controls and monitoring
    EXCEPTIONAL = "exceptional"      # Executive approval required


@dataclass
class IntentClassification:
    """Complete intent classification result."""
    intent_id: str
    risk_level: IntentRiskLevel
    category: IntentCategory
    governance_requirement: GovernanceRequirement
    confidence_score: float
    reasoning: List[str] = field(default_factory=list)
    recommended_sources: List[str] = field(default_factory=list)
    cost_estimate: Dict[str, Any] = field(default_factory=dict)
    execution_parameters: Dict[str, Any] = field(default_factory=dict)
    compliance_flags: List[str] = field(default_factory=list)
    processing_timestamp: datetime = field(default_factory=datetime.utcnow)

    def get_risk_score(self) -> float:
        """Convert risk level to numerical score."""
        risk_scores = {
            IntentRiskLevel.LOW: 0.2,
            IntentRiskLevel.MEDIUM: 0.4,
            IntentRiskLevel.HIGH: 0.7,
            IntentRiskLevel.CRITICAL: 0.9
        }
        return risk_scores.get(self.risk_level, 0.5)

    def requires_human_approval(self) -> bool:
        """Check if human approval is required."""
        return self.governance_requirement in [GovernanceRequirement.CONTROLLED, GovernanceRequirement.EXCEPTIONAL]

    def get_execution_priority(self) -> JobPriority:
        """Get recommended execution priority."""
        priority_map = {
            IntentRiskLevel.LOW: JobPriority.NORMAL,
            IntentRiskLevel.MEDIUM: JobPriority.NORMAL,
            IntentRiskLevel.HIGH: JobPriority.HIGH,
            IntentRiskLevel.CRITICAL: JobPriority.URGENT
        }
        return priority_map.get(self.risk_level, JobPriority.NORMAL)

    def get_tempo_recommendation(self) -> ScrapeTempo:
        """Get recommended scraping tempo."""
        tempo_map = {
            IntentRiskLevel.LOW: ScrapeTempo.human,
            IntentRiskLevel.MEDIUM: ScrapeTempo.human,
            IntentRiskLevel.HIGH: ScrapeTempo.aggressive,
            IntentRiskLevel.CRITICAL: ScrapeTempo.forensic
        }
        return tempo_map.get(self.risk_level, ScrapeTempo.human)


@dataclass
class IntentPattern:
    """Pattern for intent classification."""
    pattern_id: str
    category: IntentCategory
    risk_indicators: List[str] = field(default_factory=list)
    source_indicators: List[str] = field(default_factory=list)
    geography_indicators: List[str] = field(default_factory=list)
    event_indicators: List[str] = field(default_factory=list)
    compliance_triggers: List[str] = field(default_factory=list)
    weight: float = 1.0
    description: str = ""


class IntentClassifier:
    """
    Advanced intent classification engine for MJ Data Scraper Suite.

    Uses machine learning and rule-based classification to analyze scraping
    requests and determine appropriate governance, risk levels, and execution
    parameters for optimal and compliant operations.
    """

    def __init__(self):
        self.patterns: List[IntentPattern] = []
        self.classification_history: List[IntentClassification] = []
        self.feature_scaler = StandardScaler() if ML_AVAILABLE else None
        self.risk_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        ) if ML_AVAILABLE else RandomForestClassifier()

        # Performance tracking
        self.classification_stats = defaultdict(int)
        self.model_accuracy = 0.0
        self.last_training_date = None

        # Initialize classification patterns
        self._initialize_patterns()

        # Load or train ML model
        try:
            asyncio.create_task(self._initialize_ml_model())
        except RuntimeError:
            # No event loop running, defer ML initialization
            pass

        logger.info("IntentClassifier initialized with ML-enhanced classification")

    def _initialize_patterns(self):
        """Initialize rule-based classification patterns."""

        # High-risk legal patterns
        self.add_pattern(IntentPattern(
            pattern_id="court_judgment_search",
            category=IntentCategory.LEGAL,
            risk_indicators=["court", "judgment", "lawsuit", "litigation"],
            source_indicators=["court_records", "federal_court", "state_court"],
            compliance_triggers=["legal_research", "due_diligence"],
            weight=2.0,
            description="Court case and judgment research"
        ))

        # Property intelligence patterns
        self.add_pattern(IntentPattern(
            pattern_id="property_due_diligence",
            category=IntentCategory.PROPERTY,
            risk_indicators=["property", "real_estate", "title_search"],
            source_indicators=["county_clerk", "county_recorder", "title_company"],
            geography_indicators=["specific_address", "property_bounds"],
            compliance_triggers=["title_search", "property_verification"],
            weight=1.5,
            description="Real estate due diligence and title research"
        ))

        # Personal data patterns
        self.add_pattern(IntentPattern(
            pattern_id="personal_background_check",
            category=IntentCategory.PERSONAL,
            risk_indicators=["background", "personal", "individual"],
            source_indicators=["social_media", "public_records", "court_records"],
            compliance_triggers=["privacy_compliance", "data_protection"],
            weight=1.8,
            description="Personal background investigation"
        ))

        # Financial intelligence patterns
        self.add_pattern(IntentPattern(
            pattern_id="financial_investigation",
            category=IntentCategory.FINANCIAL,
            risk_indicators=["financial", "bank", "credit", "transaction"],
            source_indicators=["financial_records", "credit_reports", "bank_records"],
            compliance_triggers=["financial_privacy", "data_security"],
            weight=2.2,
            description="Financial record investigation"
        ))

        # Event intelligence patterns
        self.add_pattern(IntentPattern(
            pattern_id="event_monitoring",
            category=IntentCategory.EVENT,
            risk_indicators=["event", "announcement", "celebration"],
            source_indicators=["newspapers", "social_media", "event_sites"],
            event_indicators=["wedding", "engagement", "birthday", "graduation"],
            weight=0.8,
            description="Life event monitoring and announcements"
        ))

        # Compliance monitoring patterns
        self.add_pattern(IntentPattern(
            pattern_id="regulatory_compliance",
            category=IntentCategory.COMPLIANCE,
            risk_indicators=["compliance", "regulatory", "audit"],
            source_indicators=["government_records", "regulatory_filings"],
            compliance_triggers=["regulatory_compliance", "audit_trail"],
            weight=1.3,
            description="Regulatory compliance and audit activities"
        ))

        # Critical foreclosure patterns
        self.add_pattern(IntentPattern(
            pattern_id="distressed_property",
            category=IntentCategory.PROPERTY,
            risk_indicators=["foreclosure", "distressed", "bankruptcy"],
            source_indicators=["court_records", "state_registry", "federal_records"],
            compliance_triggers=["distressed_property_review", "financial_distress"],
            weight=2.5,
            description="Distressed property and foreclosure monitoring"
        ))

    def add_pattern(self, pattern: IntentPattern):
        """Add a classification pattern."""
        self.patterns.append(pattern)
        logger.debug(f"Added classification pattern: {pattern.pattern_id}")

    async def _initialize_ml_model(self):
        """Initialize and train the ML model with historical data."""
        # This would load historical classification data and train the model
        # For now, we'll use synthetic training data
        if ML_AVAILABLE:
            try:
                # Generate synthetic training data
                X_train, y_train = self._generate_training_data()
                if len(X_train) > 0:
                    self.risk_classifier.fit(X_train, y_train)
                    self.last_training_date = datetime.utcnow()
                    logger.info("ML model trained successfully")
            except Exception as e:
                logger.warning(f"ML model training failed: {e}")

    def _generate_training_data(self) -> Tuple[List, List]:
        """Generate synthetic training data for ML model."""
        if not ML_AVAILABLE or not NUMPY_AVAILABLE:
            return [], []

        # Synthetic examples based on our patterns
        examples = [
            # Low risk examples
            ([0.2, 0.3, 0.1, 0.1, 0.1], IntentRiskLevel.LOW),  # Personal event
            ([0.3, 0.2, 0.2, 0.1, 0.2], IntentRiskLevel.LOW),  # Public records

            # Medium risk examples
            ([0.4, 0.4, 0.3, 0.2, 0.3], IntentRiskLevel.MEDIUM),  # Property search
            ([0.5, 0.3, 0.4, 0.3, 0.4], IntentRiskLevel.MEDIUM),  # Professional background

            # High risk examples
            ([0.7, 0.6, 0.5, 0.4, 0.6], IntentRiskLevel.HIGH),  # Financial investigation
            ([0.8, 0.7, 0.6, 0.5, 0.7], IntentRiskLevel.HIGH),  # Legal research

            # Critical risk examples
            ([0.9, 0.8, 0.8, 0.7, 0.9], IntentRiskLevel.CRITICAL),  # Court case investigation
            ([0.95, 0.9, 0.9, 0.8, 0.95], IntentRiskLevel.CRITICAL),  # Foreclosure monitoring
        ]

        X = [features for features, _ in examples]
        y = [risk.value for _, risk in examples]

        return X, y

    async def classify_intent(self, control: ScrapeControlContract) -> IntentClassification:
        """
        Classify the intent of a scraping request using ML and rule-based analysis.

        Args:
            control: ScrapeControlContract containing intent, budget, etc.

        Returns:
            Complete IntentClassification with risk assessment and recommendations
        """
        try:
            # Generate intent ID
            intent_id = self._generate_intent_id(control)

            # Extract features for classification
            features = self._extract_features(control)

            # Rule-based classification
            rule_based_result = self._rule_based_classification(control)

            # ML-based classification (if available)
            ml_result = await self._ml_based_classification(features)

            # Combine results
            final_classification = self._combine_classifications(
                intent_id, rule_based_result, ml_result, control
            )

            # Enhance with intelligence
            await self._enhance_classification(final_classification, control)

            # Store in history
            self.classification_history.append(final_classification)
            self.classification_stats['classifications_performed'] += 1

            logger.info(f"✅ Classified intent {intent_id}: {final_classification.risk_level.value} risk")
            return final_classification

        except Exception as e:
            logger.error(f"❌ Intent classification failed: {e}")
            self.classification_stats['classification_errors'] += 1
            # Return conservative fallback
            return self._create_conservative_classification(control)

    def _generate_intent_id(self, control: ScrapeControlContract) -> str:
        """Generate unique intent identifier."""
        content = f"{control.intent.purpose}_{control.intent.sources}_{control.intent.geography}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _extract_features(self, control: ScrapeControlContract) -> List[float]:
        """Extract numerical features for ML classification."""
        features = []

        # Geography scope (0-1 scale)
        geo_scope = len(control.intent.geography or []) / 10.0  # Normalize
        features.append(min(geo_scope, 1.0))

        # Source diversity (0-1 scale)
        source_diversity = len(set(control.intent.sources or [])) / 5.0
        features.append(min(source_diversity, 1.0))

        # Budget intensity (0-1 scale)
        budget_intensity = 0.0
        if control.budget:
            total_budget = (control.budget.max_records * 0.1 +
                          control.budget.max_runtime_minutes * 0.2 +
                          control.budget.max_pages * 0.05)
            budget_intensity = min(total_budget / 1000.0, 1.0)
        features.append(budget_intensity)

        # Event sensitivity (0-1 scale)
        event_sensitivity = 1.0 if control.intent.event_type else 0.0
        features.append(event_sensitivity)

        # Legal/financial indicators (0-1 scale)
        legal_indicators = sum(1 for source in (control.intent.sources or [])
                             if any(term in source.lower() for term in
                                  ['court', 'legal', 'financial', 'bank', 'lien']))
        legal_score = min(legal_indicators / 3.0, 1.0)
        features.append(legal_score)

        return features

    def _rule_based_classification(self, control: ScrapeControlContract) -> Dict[str, Any]:
        """Perform rule-based intent classification."""
        scores = defaultdict(float)
        reasoning = []
        compliance_flags = []

        # Analyze each pattern
        for pattern in self.patterns:
            match_score = self._calculate_pattern_match(control, pattern)
            if match_score > 0:
                scores[pattern.category] += match_score * pattern.weight
                if match_score > 0.7:  # Strong match
                    reasoning.extend([f"Strong match for {pattern.description}"])
                    compliance_flags.extend(pattern.compliance_triggers)

        # Determine category
        if scores:
            primary_category = max(scores.items(), key=lambda x: x[1])[0]
        else:
            primary_category = IntentCategory.INTELLIGENCE

        # Determine risk level based on category and factors
        risk_level = self._determine_risk_level(primary_category, control, scores)

        # Determine governance requirement
        governance = self._determine_governance_requirement(risk_level, control)

        return {
            'category': primary_category,
            'risk_level': risk_level,
            'governance_requirement': governance,
            'reasoning': reasoning,
            'compliance_flags': list(set(compliance_flags)),
            'confidence_score': 0.8  # Rule-based confidence
        }

    def _calculate_pattern_match(self, control: ScrapeControlContract, pattern: IntentPattern) -> float:
        """Calculate how well a control matches a pattern."""
        score = 0.0
        total_indicators = 0

        # Check risk indicators in purpose and sources
        text_to_check = f"{control.intent.purpose} {' '.join(control.intent.sources or [])}"
        for indicator in pattern.risk_indicators:
            if re.search(indicator, text_to_check, re.IGNORECASE):
                score += 1.0
                total_indicators += 1

        # Check source indicators
        for indicator in pattern.source_indicators:
            if any(indicator in source for source in (control.intent.sources or [])):
                score += 1.0
                total_indicators += 1

        # Check geography indicators
        if control.intent.geography:
            geo_text = ' '.join(control.intent.geography)
            for indicator in pattern.geography_indicators:
                if re.search(indicator, geo_text, re.IGNORECASE):
                    score += 0.5
                    total_indicators += 1

        # Check event indicators
        if control.intent.event_type:
            for indicator in pattern.event_indicators:
                if indicator in control.intent.event_type.lower():
                    score += 1.0
                    total_indicators += 1

        return score / max(total_indicators, 1)

    def _determine_risk_level(self, category: IntentCategory, control: ScrapeControlContract,
                            pattern_scores: Dict) -> IntentRiskLevel:
        """Determine risk level based on category and control factors."""
        base_risk = {
            IntentCategory.PERSONAL: IntentRiskLevel.MEDIUM,
            IntentCategory.PROFESSIONAL: IntentRiskLevel.MEDIUM,
            IntentCategory.FINANCIAL: IntentRiskLevel.HIGH,
            IntentCategory.LEGAL: IntentRiskLevel.CRITICAL,
            IntentCategory.PROPERTY: IntentRiskLevel.HIGH,
            IntentCategory.EVENT: IntentRiskLevel.LOW,
            IntentCategory.COMPLIANCE: IntentRiskLevel.MEDIUM,
            IntentCategory.INTELLIGENCE: IntentRiskLevel.MEDIUM
        }.get(category, IntentRiskLevel.MEDIUM)

        # Adjust based on factors
        risk_score = 0.0

        # Geography scope
        if control.intent.geography and len(control.intent.geography) > 5:
            risk_score += 0.2  # Broad geography increases risk

        # Budget scale
        if control.budget:
            if control.budget.max_records > 10000:
                risk_score += 0.3
            if control.budget.max_runtime_minutes > 120:
                risk_score += 0.2

        # Source types
        high_risk_sources = ['court_records', 'financial_records', 'government_records']
        if any(source in (control.intent.sources or []) for source in high_risk_sources):
            risk_score += 0.3

        # Event sensitivity
        if control.intent.event_type in ['corporate', 'political', 'legal']:
            risk_score += 0.2

        # Adjust risk level based on score
        if risk_score > 0.5:
            if base_risk == IntentRiskLevel.LOW:
                return IntentRiskLevel.MEDIUM
            elif base_risk == IntentRiskLevel.MEDIUM:
                return IntentRiskLevel.HIGH
            else:
                return IntentRiskLevel.CRITICAL
        elif risk_score > 0.2:
            if base_risk == IntentRiskLevel.LOW:
                return IntentRiskLevel.MEDIUM

        return base_risk

    def _determine_governance_requirement(self, risk_level: IntentRiskLevel,
                                       control: ScrapeControlContract) -> GovernanceRequirement:
        """Determine governance requirement based on risk and control factors."""
        if risk_level == IntentRiskLevel.CRITICAL:
            return GovernanceRequirement.EXCEPTIONAL
        elif risk_level == IntentRiskLevel.HIGH:
            # Check for additional high-risk factors
            if (control.budget and
                (control.budget.max_records > 50000 or control.budget.max_runtime_minutes > 240)):
                return GovernanceRequirement.EXCEPTIONAL
            return GovernanceRequirement.CONTROLLED
        elif risk_level == IntentRiskLevel.MEDIUM:
            return GovernanceRequirement.ENHANCED
        else:
            return GovernanceRequirement.BASIC

    async def _ml_based_classification(self, features: List[float]) -> Dict[str, Any]:
        """Perform ML-based classification."""
        if not ML_AVAILABLE or not self.risk_classifier:
            return {'confidence': 0.0, 'prediction': IntentRiskLevel.MEDIUM}

        try:
            # Scale features
            if self.feature_scaler and hasattr(self.feature_scaler, 'transform'):
                scaled_features = self.feature_scaler.transform([features])
            else:
                scaled_features = [features]

            # Get prediction and probabilities
            prediction = self.risk_classifier.predict(scaled_features)[0]
            probabilities = self.risk_classifier.predict_proba(scaled_features)[0]

            # Convert prediction to enum
            risk_mapping = {
                'low': IntentRiskLevel.LOW,
                'medium': IntentRiskLevel.MEDIUM,
                'high': IntentRiskLevel.HIGH,
                'critical': IntentRiskLevel.CRITICAL
            }

            predicted_risk = risk_mapping.get(prediction, IntentRiskLevel.MEDIUM)
            confidence = max(probabilities)

            return {
                'risk_level': predicted_risk,
                'confidence': float(confidence),
                'probabilities': {k: float(v) for k, v in zip(risk_mapping.keys(), probabilities)}
            }

        except Exception as e:
            logger.warning(f"ML classification failed: {e}")
            return {'confidence': 0.0, 'prediction': IntentRiskLevel.MEDIUM}

    def _combine_classifications(self, intent_id: str, rule_result: Dict, ml_result: Dict,
                               control: ScrapeControlContract) -> IntentClassification:
        """Combine rule-based and ML results into final classification."""
        # Weight the results (rule-based gets higher weight initially)
        rule_weight = 0.7
        ml_weight = 0.3

        # Risk level combination
        if ml_result.get('confidence', 0) > 0.8:
            # High ML confidence, use ML result
            final_risk = ml_result['risk_level']
            confidence = ml_result['confidence']
        else:
            # Use rule-based result
            final_risk = rule_result['risk_level']
            confidence = rule_result['confidence_score']

        return IntentClassification(
            intent_id=intent_id,
            risk_level=final_risk,
            category=rule_result['category'],
            governance_requirement=rule_result['governance_requirement'],
            confidence_score=confidence,
            reasoning=rule_result.get('reasoning', []),
            compliance_flags=rule_result.get('compliance_flags', [])
        )

    async def _enhance_classification(self, classification: IntentClassification,
                                    control: ScrapeControlContract):
        """Enhance classification with additional intelligence."""
        # Add recommended sources
        await self._add_recommended_sources(classification, control)

        # Add cost estimate
        await self._add_cost_estimate(classification, control)

        # Add execution parameters
        await self._add_execution_parameters(classification, control)

    async def _add_recommended_sources(self, classification: IntentClassification,
                                     control: ScrapeControlContract):
        """Add recommended optimal sources."""
        try:
            # Determine primary signal type from intent
            primary_signal = self._infer_primary_signal_type(classification.category, control)

            if primary_signal:
                # Get optimal sources for the asset type and signal
                asset_type = self._infer_asset_type(classification.category)
                optimal_sources = get_optimal_sources_for_signal(asset_type, primary_signal)

                # Filter to top 5 and ensure diversity
                classification.recommended_sources = optimal_sources[:5]

                classification.reasoning.append(
                    f"Recommended sources for {primary_signal.value}: {', '.join(optimal_sources[:3])}"
                )
        except Exception as e:
            logger.debug(f"Source recommendation failed: {e}")

    async def _add_cost_estimate(self, classification: IntentClassification,
                               control: ScrapeControlContract):
        """Add cost estimate to classification."""
        try:
            asset_type = self._infer_asset_type(classification.category)
            primary_signal = self._infer_primary_signal_type(classification.category, control)

            if primary_signal:
                cost_estimate = calculate_signal_cost_estimate(
                    asset_type,
                    primary_signal,
                    sources=classification.recommended_sources[:2] if classification.recommended_sources else None,
                    data_quality="premium" if classification.risk_level in [IntentRiskLevel.HIGH, IntentRiskLevel.CRITICAL] else "standard"
                )

                classification.cost_estimate = cost_estimate
                classification.reasoning.append(
                    f"Estimated cost: ${cost_estimate.get('total_estimated_cost', 'unknown')}"
                )
        except Exception as e:
            logger.debug(f"Cost estimation failed: {e}")

    async def _add_execution_parameters(self, classification: IntentClassification,
                                      control: ScrapeControlContract):
        """Add execution parameters to classification."""
        params = {
            'priority': classification.get_execution_priority().value,
            'tempo': classification.get_tempo_recommendation().value,
            'requires_human_approval': classification.requires_human_approval(),
            'max_concurrent_requests': self._get_concurrent_limit(classification.risk_level),
            'rate_limit_multiplier': self._get_rate_limit_multiplier(classification.risk_level),
            'monitoring_level': self._get_monitoring_level(classification.governance_requirement)
        }

        # Add freshness requirements
        primary_signal = self._infer_primary_signal_type(classification.category, control)
        if primary_signal:
            freshness_days = get_data_freshness_requirement(primary_signal)
            params['data_freshness_requirement_days'] = freshness_days

        classification.execution_parameters = params

    def _infer_primary_signal_type(self, category: IntentCategory, control: ScrapeControlContract) -> Optional[SignalType]:
        """Infer the primary signal type from category and control."""
        category_signals = {
            IntentCategory.LEGAL: SignalType.COURT_CASE,
            IntentCategory.PROPERTY: SignalType.LIEN,
            IntentCategory.FINANCIAL: SignalType.FINANCIAL,
            IntentCategory.PERSONAL: SignalType.IDENTITY,
            IntentCategory.EVENT: SignalType.WEDDING,
            IntentCategory.COMPLIANCE: SignalType.LEGAL,
            IntentCategory.PROFESSIONAL: SignalType.PROFESSIONAL
        }

        signal = category_signals.get(category, SignalType.IDENTITY)

        # Check if control specifies particular signals
        if control.intent.sources:
            for source in control.intent.sources:
                if 'court' in source.lower():
                    return SignalType.COURT_CASE
                elif 'lien' in source.lower():
                    return SignalType.LIEN
                elif 'wedding' in source.lower():
                    return SignalType.WEDDING

        return signal

    def _infer_asset_type(self, category: IntentCategory) -> AssetType:
        """Infer asset type from category."""
        category_assets = {
            IntentCategory.PERSONAL: AssetType.PERSON,
            IntentCategory.PROPERTY: AssetType.SINGLE_FAMILY_HOME,
            IntentCategory.LEGAL: AssetType.PERSON,  # Legal can apply to persons or companies
            IntentCategory.FINANCIAL: AssetType.COMPANY,
            IntentCategory.EVENT: AssetType.PERSON,
            IntentCategory.COMPLIANCE: AssetType.COMPANY,
            IntentCategory.PROFESSIONAL: AssetType.PERSON
        }
        return category_assets.get(category, AssetType.ASSET)

    def _get_concurrent_limit(self, risk_level: IntentRiskLevel) -> int:
        """Get concurrent request limit based on risk level."""
        limits = {
            IntentRiskLevel.LOW: 10,
            IntentRiskLevel.MEDIUM: 5,
            IntentRiskLevel.HIGH: 3,
            IntentRiskLevel.CRITICAL: 1
        }
        return limits.get(risk_level, 5)

    def _get_rate_limit_multiplier(self, risk_level: IntentRiskLevel) -> float:
        """Get rate limit multiplier based on risk level."""
        multipliers = {
            IntentRiskLevel.LOW: 1.0,
            IntentRiskLevel.MEDIUM: 0.7,
            IntentRiskLevel.HIGH: 0.4,
            IntentRiskLevel.CRITICAL: 0.2
        }
        return multipliers.get(risk_level, 0.7)

    def _get_monitoring_level(self, governance: GovernanceRequirement) -> str:
        """Get monitoring level based on governance requirement."""
        levels = {
            GovernanceRequirement.BASIC: "standard",
            GovernanceRequirement.ENHANCED: "enhanced",
            GovernanceRequirement.RESTRICTED: "restricted",
            GovernanceRequirement.CONTROLLED: "intensive",
            GovernanceRequirement.EXCEPTIONAL: "maximum"
        }
        return levels.get(governance, "standard")

    def _create_conservative_classification(self, control: ScrapeControlContract) -> IntentClassification:
        """Create conservative fallback classification."""
        return IntentClassification(
            intent_id=self._generate_intent_id(control),
            risk_level=IntentRiskLevel.HIGH,
            category=IntentCategory.INTELLIGENCE,
            governance_requirement=GovernanceRequirement.CONTROLLED,
            confidence_score=0.5,
            reasoning=["Conservative fallback due to classification error"]
        )

    def get_classification_stats(self) -> Dict[str, Any]:
        """Get comprehensive classification statistics."""
        total_classifications = len(self.classification_history)

        if total_classifications == 0:
            return {"total_classifications": 0}

        # Risk level distribution
        risk_counts = Counter(cls.risk_level.value for cls in self.classification_history)

        # Category distribution
        category_counts = Counter(cls.category.value for cls in self.classification_history)

        # Governance distribution
        governance_counts = Counter(cls.governance_requirement.value for cls in self.classification_history)

        # Average confidence
        avg_confidence = sum(cls.confidence_score for cls in self.classification_history) / total_classifications

        # Recent performance (last 100 classifications)
        recent = self.classification_history[-100:] if len(self.classification_history) >= 100 else self.classification_history
        recent_avg_confidence = sum(cls.confidence_score for cls in recent) / len(recent)

        return {
            "total_classifications": total_classifications,
            "risk_distribution": dict(risk_counts),
            "category_distribution": dict(category_counts),
            "governance_distribution": dict(governance_counts),
            "average_confidence": round(avg_confidence, 3),
            "recent_average_confidence": round(recent_avg_confidence, 3),
            "ml_model_available": ML_AVAILABLE,
            "last_training_date": self.last_training_date.isoformat() if self.last_training_date else None,
            "model_accuracy": round(self.model_accuracy, 3) if self.model_accuracy > 0 else None,
            "classification_errors": self.classification_stats.get('classification_errors', 0),
            "classifications_performed": self.classification_stats.get('classifications_performed', 0)
        }

    def retrain_model(self):
        """Retrain the ML model with historical classification data."""
        if not ML_AVAILABLE or len(self.classification_history) < 10:
            logger.info("Insufficient data for model retraining")
            return

        try:
            # Extract features and labels from history
            features = []
            labels = []

            for classification in self.classification_history[-500:]:  # Use last 500
                # Reconstruct features from stored data (simplified)
                # In practice, you'd store the original features
                feature_vector = [classification.get_risk_score(), 0.5, 0.5, 0.5, 0.5]  # Placeholder
                features.append(feature_vector)
                labels.append(classification.risk_level.value)

            # Retrain model
            self.risk_classifier.fit(features, labels)
            self.last_training_date = datetime.utcnow()

            # Evaluate accuracy (simplified)
            predictions = self.risk_classifier.predict(features)
            accuracy = sum(1 for pred, actual in zip(predictions, labels) if pred == actual) / len(labels)
            self.model_accuracy = accuracy

            logger.info(f"Model retrained with {len(features)} samples, accuracy: {accuracy:.3f}")

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")


# Global classifier instance
_global_classifier = IntentClassifier()


# Convenience functions
async def classify_scraping_intent(control: ScrapeControlContract) -> IntentClassification:
    """
    Classify the intent of a scraping request.

    This is the main entry point for intent classification in the MJ Data Scraper Suite.
    Analyzes the scraping control contract to determine risk level, governance requirements,
    and optimal execution parameters.

    Args:
        control: ScrapeControlContract to classify

    Returns:
        Complete IntentClassification with risk assessment and recommendations
    """
    return await _global_classifier.classify_intent(control)


def get_intent_classification_stats() -> Dict[str, Any]:
    """
    Get comprehensive intent classification statistics.

    Returns operational metrics for monitoring classification performance and
    intent patterns in the scraping ecosystem.

    Returns:
        Dict with classification statistics and performance metrics
    """
    return _global_classifier.get_classification_stats()


def retrain_intent_classifier():
    """
    Retrain the intent classification ML model.

    Uses historical classification data to improve model accuracy and
    adapt to new scraping patterns and risk profiles.
    """
    _global_classifier.retrain_model()


# Classification helper functions
def get_risk_level_description(risk_level: IntentRiskLevel) -> str:
    """
    Get human-readable description of risk level.

    Args:
        risk_level: Risk level enum

    Returns:
        Description string
    """
    descriptions = {
        IntentRiskLevel.LOW: "Low risk - Standard monitoring and approval processes",
        IntentRiskLevel.MEDIUM: "Medium risk - Enhanced oversight recommended",
        IntentRiskLevel.HIGH: "High risk - Strict controls and human oversight required",
        IntentRiskLevel.CRITICAL: "Critical risk - Executive approval and maximum monitoring required"
    }
    return descriptions.get(risk_level, "Unknown risk level")


def get_governance_description(governance: GovernanceRequirement) -> str:
    """
    Get human-readable description of governance requirement.

    Args:
        governance: Governance requirement enum

    Returns:
        Description string
    """
    descriptions = {
        GovernanceRequirement.BASIC: "Basic approval process with standard monitoring",
        GovernanceRequirement.ENHANCED: "Enhanced oversight with additional review steps",
        GovernanceRequirement.RESTRICTED: "Restricted execution with limited time windows",
        GovernanceRequirement.CONTROLLED: "Controlled environment with strict monitoring",
        GovernanceRequirement.EXCEPTIONAL: "Exceptional approval required with executive oversight"
    }
    return descriptions.get(governance, "Unknown governance requirement")
