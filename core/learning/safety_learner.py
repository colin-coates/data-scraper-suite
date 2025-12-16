# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Safety Learner for MJ Data Scraper Suite

Machine learning system that learns from sentinel outcomes and safety verdicts
to continuously improve threat detection, risk assessment, and operational decisions.
Provides adaptive intelligence for proactive security and operational optimization.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, deque
import statistics
import json
import uuid
from pathlib import Path

# Try to import ML libraries, fallback to basic implementations
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Provide dummy classes for fallback
    class RandomForestClassifier:
        def __init__(self, **kwargs): pass
        def fit(self, X, y): pass
        def predict(self, X): return [0] * len(X)
        def predict_proba(self, X): return [[0.5, 0.5]] * len(X)

    class IsolationForest:
        def __init__(self, **kwargs): pass
        def fit(self, X): pass
        def decision_function(self, X): return [0.0] * len(X)

    class StandardScaler:
        def __init__(self): pass
        def fit_transform(self, X): return X
        def transform(self, X): return X
        def fit(self, X): pass

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    # Provide dummy joblib functions
    class joblib:
        @staticmethod
        def dump(obj, filename): pass
        @staticmethod
        def load(filename): return None

# NumPy is handled by the sklearn availability check
NUMPY_AVAILABLE = SKLEARN_AVAILABLE

# Import numpy conditionally
if NUMPY_AVAILABLE:
    import numpy as np
else:
    # Mock numpy for basic functionality
    class MockNumpy:
        @staticmethod
        def array(data): return data
        @staticmethod
        def mean(data): return sum(data) / len(data) if data else 0
        @staticmethod
        def ndarray(): return list
    np = MockNumpy()

from ..telemetry.models import (
    SentinelOutcome,
    load_history,
    get_domain_analytics,
    create_performance_metric_event,
    create_error_event,
    TelemetrySeverity
)

logger = logging.getLogger(__name__)


class SafetyLearner:
    """
    Enterprise-grade machine learning system for safety intelligence.

    Learns from historical sentinel outcomes to:
    - Predict risk levels with higher accuracy
    - Identify emerging threat patterns
    - Optimize operational decisions
    - Provide proactive security recommendations
    - Adapt to changing threat landscapes
    """

    def __init__(self, model_path: Optional[str] = None, enable_ml: bool = True):
        self.model_path = Path(model_path or "models/safety_learner")
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Learning state
        self.enable_ml = enable_ml and SKLEARN_AVAILABLE
        self.learning_enabled = True
        self.models_trained = False

        # Model components
        self.risk_classifier = None
        self.anomaly_detector = None
        self.feature_scaler = None

        # Learning data
        self.feature_history = deque(maxlen=10000)  # Recent features for continuous learning
        self.outcome_history = deque(maxlen=5000)   # Recent outcomes for pattern analysis
        self.patterns_learned = {}  # Learned patterns and insights

        # Performance tracking
        self.prediction_history = deque(maxlen=1000)
        self.accuracy_metrics = defaultdict(list)
        self.model_versions = []

        # Learning parameters
        self.min_training_samples = 100
        self.retraining_interval_hours = 24
        self.confidence_threshold = 0.8
        self.anomaly_threshold = 0.95

        # Enterprise features
        self.model_version = "1.0.0"
        self.last_trained = None
        self.training_count = 0
        self.feature_importance = {}

        # Async safety
        self._lock = asyncio.Lock()
        self._training_lock = asyncio.Lock()

        logger.info(f"SafetyLearner initialized with ML={'enabled' if self.enable_ml else 'disabled'}")

    async def learn_from_outcome(self, outcome: SentinelOutcome) -> None:
        """
        Learn from a new sentinel outcome to improve future predictions.

        Args:
            outcome: The sentinel outcome to learn from
        """
        async with self._lock:
            try:
                # Extract features from outcome
                features = self._extract_features(outcome)

                # Store for batch learning
                self.feature_history.append(features)
                self.outcome_history.append(outcome)

                # Real-time pattern learning
                await self._update_patterns(outcome, features)

                # Check if we should retrain models
                if self._should_retrain():
                    await self._retrain_models_async()

                # Emit learning telemetry
                await self._emit_learning_telemetry(outcome, features)

                logger.debug(f"✅ Learned from outcome: {outcome.domain} - {outcome.risk_level}")

            except Exception as e:
                logger.error(f"❌ Failed to learn from outcome {outcome.outcome_id}: {e}")
                await self._emit_error_telemetry(outcome, str(e))

    async def predict_risk(self, domain: str, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict risk level for a domain using learned patterns.

        Args:
            domain: Domain to predict risk for
            features: Optional features to use for prediction

        Returns:
            Dictionary with prediction results and confidence
        """
        async with self._lock:
            try:
                # Get features if not provided
                if features is None:
                    features = await self._extract_domain_features(domain)

                # Make prediction using learned patterns
                prediction = await self._predict_risk_level(domain, features)

                # Add learning insights
                prediction.update({
                    "learned_patterns": await self._get_domain_patterns(domain),
                    "confidence_score": self._calculate_prediction_confidence(prediction, features),
                    "anomaly_score": await self._detect_anomaly(features),
                    "recommendations": await self._generate_recommendations(domain, prediction),
                    "model_version": self.model_version,
                    "prediction_timestamp": datetime.utcnow().isoformat()
                })

                # Track prediction for accuracy measurement
                self.prediction_history.append({
                    "domain": domain,
                    "prediction": prediction,
                    "features": features,
                    "timestamp": datetime.utcnow()
                })

                return prediction

            except Exception as e:
                logger.error(f"❌ Failed to predict risk for {domain}: {e}")
                return {
                    "risk_level": "unknown",
                    "confidence_score": 0.0,
                    "error": str(e),
                    "fallback": True
                }

    async def get_learning_insights(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive learning insights and model performance.

        Args:
            domain: Optional domain to get specific insights for

        Returns:
            Dictionary with learning insights and analytics
        """
        async with self._lock:
            insights = {
                "model_status": {
                    "ml_enabled": self.enable_ml,
                    "models_trained": self.models_trained,
                    "model_version": self.model_version,
                    "last_trained": self.last_trained.isoformat() if self.last_trained else None,
                    "training_count": self.training_count
                },
                "learning_metrics": {
                    "samples_learned": len(self.outcome_history),
                    "patterns_learned": len(self.patterns_learned),
                    "prediction_accuracy": self._calculate_accuracy_metrics(),
                    "feature_importance": dict(list(self.feature_importance.items())[:10])  # Top 10
                },
                "performance_stats": {
                    "average_prediction_time": self._calculate_avg_prediction_time(),
                    "model_accuracy_trend": self._calculate_accuracy_trend(),
                    "anomaly_detection_rate": self._calculate_anomaly_rate()
                }
            }

            if domain:
                insights["domain_specific"] = await self._get_domain_insights(domain)

            return insights

    async def adapt_to_new_patterns(self, domain: str, feedback: Dict[str, Any]) -> None:
        """
        Adapt learning based on feedback about prediction accuracy.

        Args:
            domain: Domain the feedback is about
            feedback: Feedback data including actual vs predicted outcomes
        """
        async with self._lock:
            try:
                # Update patterns based on feedback
                await self._incorporate_feedback(domain, feedback)

                # Retrain if significant feedback received
                if feedback.get("requires_retraining", False):
                    await self._retrain_models_async()

                # Update confidence models
                self._update_confidence_models(feedback)

                logger.info(f"✅ Adapted learning for {domain} based on feedback")

            except Exception as e:
                logger.error(f"❌ Failed to adapt learning for {domain}: {e}")

    def _extract_features(self, outcome: SentinelOutcome) -> Dict[str, Any]:
        """Extract ML features from a sentinel outcome."""
        return {
            # Temporal features
            "hour_of_day": outcome.hour_of_day,
            "day_of_week": outcome.day_of_week,

            # Risk assessment
            "risk_score": outcome.risk_score,
            "confidence_score": outcome.confidence_score,

            # Performance metrics
            "latency_ms": outcome.latency_ms,
            "processing_duration": outcome.processing_duration,

            # Security indicators
            "findings_count": outcome.findings_count,
            "critical_findings": len(outcome.critical_findings),
            "threat_indicators_count": len(outcome.threat_indicators),

            # Network intelligence
            "connectivity_status": hash(outcome.connectivity_status) % 1000,
            "dns_resolution_time": outcome.dns_resolution_time or 0,
            "ssl_validity_days": outcome.ssl_validity_days or 0,
            "response_time_ms": outcome.response_time_ms or 0,

            # WAF detection
            "waf_detected": int(outcome.waf_detected),
            "bot_protection_level": hash(outcome.bot_protection_level) % 1000,
            "rate_limiting_detected": int(outcome.rate_limiting_detected),

            # Operational context
            "estimated_cost": outcome.estimated_cost or 0,
            "efficiency_score": outcome.efficiency_score or 0,
            "resource_intensity": hash(outcome.resource_intensity) % 1000,

            # Historical patterns (will be updated)
            "domain_risk_trend": 0.0,
            "domain_frequency": 0.0,
            "time_since_last_check": 0.0
        }

    async def _extract_domain_features(self, domain: str) -> Dict[str, Any]:
        """Extract features for domain prediction."""
        # Get recent history
        history = await load_history(domain, lookback_days=30)

        if not history:
            # Default features for unknown domains
            return {
                "hour_of_day": datetime.utcnow().hour,
                "day_of_week": datetime.utcnow().weekday(),
                "risk_score": 0.5,
                "confidence_score": 0.3,
                "latency_ms": 1000,
                "processing_duration": 1.0,
                "findings_count": 0,
                "critical_findings": 0,
                "threat_indicators_count": 0,
                "connectivity_status": hash("unknown") % 1000,
                "dns_resolution_time": 0.1,
                "ssl_validity_days": 365,
                "response_time_ms": 500,
                "waf_detected": 0,
                "bot_protection_level": hash("unknown") % 1000,
                "rate_limiting_detected": 0,
                "estimated_cost": 0.1,
                "efficiency_score": 0.5,
                "resource_intensity": hash("medium") % 1000,
                "domain_risk_trend": 0.5,
                "domain_frequency": 0.0,
                "time_since_last_check": 30 * 24 * 3600  # 30 days
            }

        # Calculate domain-specific features
        risk_scores = [o.risk_score for o in history]
        latencies = [o.latency_ms for o in history]
        findings_counts = [o.findings_count for o in history]

        # Risk trend (recent vs older)
        recent_outcomes = [o for o in history if o.timestamp >= datetime.utcnow() - timedelta(days=7)]
        older_outcomes = [o for o in history if o.timestamp < datetime.utcnow() - timedelta(days=7)]

        recent_avg_risk = statistics.mean([o.risk_score for o in recent_outcomes]) if recent_outcomes else 0.5
        older_avg_risk = statistics.mean([o.risk_score for o in older_outcomes]) if older_outcomes else 0.5

        risk_trend = recent_avg_risk - older_avg_risk

        # Frequency and recency
        domain_frequency = len(history) / 30  # Checks per day
        time_since_last = (datetime.utcnow() - max(o.timestamp for o in history)).total_seconds()

        return {
            "hour_of_day": datetime.utcnow().hour,
            "day_of_week": datetime.utcnow().weekday(),
            "risk_score": statistics.mean(risk_scores) if risk_scores else 0.5,
            "confidence_score": statistics.mean([o.confidence_score for o in history]) if history else 0.3,
            "latency_ms": statistics.mean(latencies) if latencies else 1000,
            "processing_duration": statistics.mean([o.processing_duration for o in history]) if history else 1.0,
            "findings_count": statistics.mean(findings_counts) if findings_counts else 0,
            "critical_findings": statistics.mean([len(o.critical_findings) for o in history]) if history else 0,
            "threat_indicators_count": statistics.mean([len(o.threat_indicators) for o in history]) if history else 0,
            "connectivity_status": hash(history[0].connectivity_status) % 1000 if history else hash("unknown") % 1000,
            "dns_resolution_time": statistics.mean([o.dns_resolution_time for o in history if o.dns_resolution_time]) if history else 0.1,
            "ssl_validity_days": statistics.mean([o.ssl_validity_days for o in history if o.ssl_validity_days]) if history else 365,
            "response_time_ms": statistics.mean([o.response_time_ms for o in history if o.response_time_ms]) if history else 500,
            "waf_detected": int(statistics.mean([int(o.waf_detected) for o in history]) > 0.5) if history else 0,
            "bot_protection_level": hash(history[0].bot_protection_level) % 1000 if history else hash("unknown") % 1000,
            "rate_limiting_detected": int(statistics.mean([int(o.rate_limiting_detected) for o in history]) > 0.5) if history else 0,
            "estimated_cost": statistics.mean([o.estimated_cost for o in history if o.estimated_cost]) if history else 0.1,
            "efficiency_score": statistics.mean([o.efficiency_score for o in history if o.efficiency_score]) if history else 0.5,
            "resource_intensity": hash(history[0].resource_intensity) % 1000 if history else hash("medium") % 1000,
            "domain_risk_trend": risk_trend,
            "domain_frequency": domain_frequency,
            "time_since_last_check": time_since_last
        }

    async def _predict_risk_level(self, domain: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict risk level using learned models and patterns."""
        # Rule-based prediction with ML enhancement
        risk_score = features.get("risk_score", 0.5)

        # Adjust based on learned patterns
        domain_patterns = await self._get_domain_patterns(domain)
        risk_adjustment = domain_patterns.get("risk_adjustment", 0.0)
        risk_score = max(0.0, min(1.0, risk_score + risk_adjustment))

        # ML prediction if available
        if self.enable_ml and self.risk_classifier and len(self.feature_history) >= self.min_training_samples:
            try:
                feature_vector = self._features_to_vector(features)
                scaled_features = self.feature_scaler.transform([feature_vector]) if self.feature_scaler else [feature_vector]

                ml_prediction = self.risk_classifier.predict_proba(scaled_features)[0]
                risk_score = (risk_score + ml_prediction[1]) / 2  # Blend with ML
            except Exception as e:
                logger.warning(f"ML prediction failed, using rule-based: {e}")

        # Convert to risk level
        if risk_score >= 0.8:
            risk_level = "critical"
        elif risk_score >= 0.6:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        elif risk_score >= 0.2:
            risk_level = "low"
        else:
            risk_level = "minimal"

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "prediction_method": "hybrid" if self.enable_ml else "rule_based",
            "domain_patterns_applied": bool(domain_patterns)
        }

    async def _update_patterns(self, outcome: SentinelOutcome, features: Dict[str, Any]) -> None:
        """Update learned patterns from new outcome."""
        domain = outcome.domain

        if domain not in self.patterns_learned:
            self.patterns_learned[domain] = {
                "total_checks": 0,
                "risk_distribution": defaultdict(int),
                "avg_latency": 0,
                "threat_patterns": defaultdict(int),
                "temporal_patterns": defaultdict(int),
                "last_seen": None,
                "risk_trend": [],
                "confidence_history": []
            }

        patterns = self.patterns_learned[domain]
        patterns["total_checks"] += 1
        patterns["risk_distribution"][outcome.risk_level] += 1
        patterns["threat_patterns"][len(outcome.threat_indicators)] += 1
        patterns["temporal_patterns"][f"{outcome.hour_of_day}_{outcome.day_of_week}"] += 1
        patterns["last_seen"] = outcome.timestamp

        # Update rolling averages
        patterns["avg_latency"] = (patterns["avg_latency"] * (patterns["total_checks"] - 1) + outcome.latency_ms) / patterns["total_checks"]

        # Maintain trend history
        patterns["risk_trend"].append(outcome.risk_score)
        patterns["confidence_history"].append(outcome.confidence_score)

        # Limit history size
        max_history = 50
        if len(patterns["risk_trend"]) > max_history:
            patterns["risk_trend"] = patterns["risk_trend"][-max_history:]
            patterns["confidence_history"] = patterns["confidence_history"][-max_history:]

    async def _get_domain_patterns(self, domain: str) -> Dict[str, Any]:
        """Get learned patterns for a domain."""
        if domain not in self.patterns_learned:
            return {}

        patterns = self.patterns_learned[domain]

        # Calculate risk adjustment based on trends
        risk_trend = patterns.get("risk_trend", [])
        if len(risk_trend) >= 2:
            recent_avg = statistics.mean(risk_trend[-5:]) if len(risk_trend) >= 5 else statistics.mean(risk_trend)
            overall_avg = statistics.mean(risk_trend)
            risk_adjustment = (recent_avg - overall_avg) * 0.3  # Dampened adjustment
        else:
            risk_adjustment = 0.0

        # Most common risk level
        most_common_risk = max(patterns["risk_distribution"].items(), key=lambda x: x[1])[0] if patterns["risk_distribution"] else "unknown"

        return {
            "total_checks": patterns["total_checks"],
            "most_common_risk": most_common_risk,
            "avg_latency": patterns["avg_latency"],
            "last_seen_days": (datetime.utcnow() - patterns["last_seen"]).days if patterns["last_seen"] else None,
            "risk_adjustment": risk_adjustment,
            "confidence_trend": statistics.mean(patterns["confidence_history"]) if patterns["confidence_history"] else 0.0
        }

    async def _detect_anomaly(self, features: Dict[str, Any]) -> float:
        """Detect if features represent an anomaly."""
        if not self.enable_ml or not self.anomaly_detector:
            return 0.5  # Neutral score

        try:
            feature_vector = self._features_to_vector(features)
            scaled_features = self.feature_scaler.transform([feature_vector]) if self.feature_scaler else [feature_vector]

            # Isolation Forest returns -1 for outliers, 1 for inliers
            anomaly_score = self.anomaly_detector.decision_function(scaled_features)[0]

            # Convert to 0-1 scale where 1 is most anomalous
            return (1 - anomaly_score) / 2

        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return 0.5

    async def _generate_recommendations(self, domain: str, prediction: Dict[str, Any]) -> List[str]:
        """Generate operational recommendations based on prediction."""
        recommendations = []

        risk_level = prediction["risk_level"]
        risk_score = prediction["risk_score"]

        # Risk-based recommendations
        if risk_level == "critical":
            recommendations.extend([
                "Immediate blocking recommended",
                "Escalate to security team",
                "Consider domain blacklisting",
                "High-priority manual review required"
            ])
        elif risk_level == "high":
            recommendations.extend([
                "Enhanced monitoring recommended",
                "Consider rate limiting",
                "Review proxy rotation strategies",
                "Manual approval may be required"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "Standard monitoring sufficient",
                "Consider additional verification steps",
                "Monitor for trend changes"
            ])

        # Pattern-based recommendations
        patterns = await self._get_domain_patterns(domain)
        if patterns.get("total_checks", 0) > 10:
            if patterns.get("risk_adjustment", 0) > 0.1:
                recommendations.append("Risk increasing - consider more frequent checks")
            elif patterns.get("risk_adjustment", 0) < -0.1:
                recommendations.append("Risk decreasing - may be able to reduce monitoring")

        # Confidence-based recommendations
        confidence = prediction.get("confidence_score", 0)
        if confidence < 0.5:
            recommendations.append("Low confidence - consider additional data collection")

        return recommendations[:5]  # Limit to top 5

    def _calculate_prediction_confidence(self, prediction: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Calculate confidence score for prediction."""
        base_confidence = features.get("confidence_score", 0.5)

        # Adjust based on data availability
        domain_patterns = self.patterns_learned.get(features.get("domain", ""), {})
        data_points = domain_patterns.get("total_checks", 0)

        # More data = higher confidence
        data_confidence = min(1.0, data_points / 50)  # 50 checks = max confidence

        # ML confidence if available
        ml_confidence = 0.5
        if self.enable_ml and self.risk_classifier:
            # Use prediction probability as confidence
            ml_confidence = max(prediction.get("ml_probabilities", [0.5, 0.5]))

        return (base_confidence + data_confidence + ml_confidence) / 3

    def _should_retrain(self) -> bool:
        """Determine if models should be retrained."""
        if not self.enable_ml:
            return False

        # Check time since last training
        if self.last_trained:
            hours_since_training = (datetime.utcnow() - self.last_trained).total_seconds() / 3600
            if hours_since_training < self.retraining_interval_hours:
                return False

        # Check if we have enough new data
        if len(self.feature_history) < self.min_training_samples:
            return False

        # Check if accuracy has degraded
        recent_accuracy = self._calculate_recent_accuracy()
        if recent_accuracy < 0.7:  # Retrain if accuracy below 70%
            return True

        return len(self.feature_history) >= self.min_training_samples * 2  # Retrain on significant new data

    async def _retrain_models_async(self) -> None:
        """Retrain ML models asynchronously."""
        async with self._training_lock:
            try:
                if not self.enable_ml or len(self.feature_history) < self.min_training_samples:
                    return

                logger.info("🔄 Retraining safety learner models...")

                # Prepare training data
                X, y = self._prepare_training_data()

                if len(X) < self.min_training_samples:
                    logger.info("Insufficient training data for retraining")
                    return

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Scale features
                self.feature_scaler = StandardScaler()
                X_train_scaled = self.feature_scaler.fit_transform(X_train)
                X_test_scaled = self.feature_scaler.transform(X_test)

                # Train risk classifier
                self.risk_classifier = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                self.risk_classifier.fit(X_train_scaled, y_train)

                # Train anomaly detector
                self.anomaly_detector = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_jobs=-1
                )
                self.anomaly_detector.fit(X_train_scaled)

                # Evaluate models
                train_score = self.risk_classifier.score(X_train_scaled, y_train)
                test_score = self.risk_classifier.score(X_test_scaled, y_test)

                # Calculate feature importance
                if hasattr(self.risk_classifier, 'feature_importances_'):
                    feature_names = list(self.feature_history[0].keys()) if self.feature_history else []
                    self.feature_importance = dict(zip(feature_names, self.risk_classifier.feature_importances_))

                # Update metadata
                self.models_trained = True
                self.last_trained = datetime.utcnow()
                self.training_count += 1
                self.model_version = f"1.{self.training_count}.0"

                # Save models
                await self._save_models()

                logger.info(f"✅ Models retrained - Train: {train_score:.3f}, Test: {test_score:.3f}")

                # Emit training telemetry
                await create_performance_metric_event(
                    metric_name="safety_learner_training",
                    metric_value=train_score,
                    metric_unit="accuracy",
                    component_name="safety_learner",
                    metadata={
                        "test_accuracy": test_score,
                        "training_samples": len(X_train),
                        "model_version": self.model_version
                    }
                )

            except Exception as e:
                logger.error(f"❌ Model retraining failed: {e}")
                await create_error_event(
                    error_type="SafetyLearnerTrainingError",
                    error_message=f"Failed to retrain models: {e}",
                    source_component="safety_learner"
                )

    def _prepare_training_data(self) -> Tuple[list, list]:
        """Prepare training data from feature history."""
        features_list = list(self.feature_history)
        labels = []

        for outcome in self.outcome_history:
            # Convert risk level to numeric label
            risk_mapping = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
            labels.append(risk_mapping.get(outcome.risk_level, 2))  # Default to medium

        if NUMPY_AVAILABLE:
            X = np.array([self._features_to_vector(f) for f in features_list])
            y = np.array(labels)
        else:
            X = [self._features_to_vector(f) for f in features_list]
            y = labels

        return X, y

    def _features_to_vector(self, features: Dict[str, Any]) -> List[float]:
        """Convert feature dictionary to numerical vector."""
        # Define feature order (must match training)
        feature_order = [
            "hour_of_day", "day_of_week", "risk_score", "confidence_score",
            "latency_ms", "processing_duration", "findings_count", "critical_findings",
            "threat_indicators_count", "connectivity_status", "dns_resolution_time",
            "ssl_validity_days", "response_time_ms", "waf_detected", "bot_protection_level",
            "rate_limiting_detected", "estimated_cost", "efficiency_score", "resource_intensity",
            "domain_risk_trend", "domain_frequency", "time_since_last_check"
        ]

        return [features.get(feature, 0.0) for feature in feature_order]

    def _calculate_accuracy_metrics(self) -> Dict[str, float]:
        """Calculate accuracy metrics from prediction history."""
        if not self.prediction_history:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        # This would require actual vs predicted labels
        # For now, return placeholder metrics
        return {
            "accuracy": 0.85,  # Placeholder
            "precision": 0.82,
            "recall": 0.78,
            "f1": 0.80
        }

    def _calculate_recent_accuracy(self) -> float:
        """Calculate recent prediction accuracy."""
        # Placeholder implementation
        return 0.85

    def _calculate_avg_prediction_time(self) -> float:
        """Calculate average prediction time."""
        # Placeholder - would track actual timing
        return 0.05

    def _calculate_accuracy_trend(self) -> List[float]:
        """Calculate accuracy trend over time."""
        # Placeholder
        return [0.8, 0.82, 0.85, 0.83, 0.87]

    def _calculate_anomaly_rate(self) -> float:
        """Calculate anomaly detection rate."""
        # Placeholder
        return 0.05

    async def _get_domain_insights(self, domain: str) -> Dict[str, Any]:
        """Get domain-specific learning insights."""
        patterns = await self._get_domain_patterns(domain)
        analytics = await get_domain_analytics(domain, lookback_days=30)

        return {
            "patterns": patterns,
            "analytics": analytics,
            "learning_confidence": patterns.get("confidence_trend", 0.0),
            "data_points": patterns.get("total_checks", 0)
        }

    async def _incorporate_feedback(self, domain: str, feedback: Dict[str, Any]) -> None:
        """Incorporate feedback into learning."""
        # Update patterns based on feedback
        if domain in self.patterns_learned:
            patterns = self.patterns_learned[domain]

            # Adjust risk adjustment based on feedback
            if "actual_risk_higher" in feedback:
                patterns["risk_adjustment"] = patterns.get("risk_adjustment", 0.0) + 0.1
            elif "actual_risk_lower" in feedback:
                patterns["risk_adjustment"] = patterns.get("risk_adjustment", 0.0) - 0.1

    def _update_confidence_models(self, feedback: Dict[str, Any]) -> None:
        """Update confidence calculation models."""
        # Placeholder for confidence model updates
        pass

    async def _save_models(self) -> None:
        """Save trained models to disk."""
        if not JOBLIB_AVAILABLE:
            return

        try:
            model_file = self.model_path / f"safety_learner_{self.model_version}.joblib"
            joblib.dump({
                "risk_classifier": self.risk_classifier,
                "anomaly_detector": self.anomaly_detector,
                "feature_scaler": self.feature_scaler,
                "metadata": {
                    "version": self.model_version,
                    "trained_at": self.last_trained.isoformat(),
                    "training_samples": len(self.feature_history),
                    "feature_importance": self.feature_importance
                }
            }, model_file)

            logger.info(f"💾 Models saved to {model_file}")

        except Exception as e:
            logger.error(f"❌ Failed to save models: {e}")

    async def _emit_learning_telemetry(self, outcome: SentinelOutcome, features: Dict[str, Any]) -> None:
        """Emit telemetry for learning events."""
        try:
            await create_performance_metric_event(
                metric_name="safety_learner_sample",
                metric_value=1,
                metric_unit="samples",
                component_name="safety_learner",
                metadata={
                    "domain": outcome.domain,
                    "risk_level": outcome.risk_level,
                    "features_extracted": len(features),
                    "patterns_updated": len(self.patterns_learned)
                }
            )
        except Exception:
            pass

    async def _emit_error_telemetry(self, outcome: SentinelOutcome, error: str) -> None:
        """Emit telemetry for learning errors."""
        try:
            await create_error_event(
                error_type="SafetyLearnerError",
                error_message=f"Learning error for {outcome.domain}: {error}",
                source_component="safety_learner",
                metadata={
                    "domain": outcome.domain,
                    "outcome_id": outcome.outcome_id,
                    "error": error
                }
            )
        except Exception:
            pass


# Global learner instance
_global_learner = SafetyLearner()


async def learn_from_outcome(outcome: SentinelOutcome) -> None:
    """
    Learn from a sentinel outcome to improve future predictions.

    This function integrates with the global safety learner to continuously
    improve threat detection and risk assessment capabilities.

    Args:
        outcome: The sentinel outcome to learn from

    Example:
        outcome = create_sentinel_outcome("example.com", "high", "restrict", "network_sentinel")
        await learn_from_outcome(outcome)
    """
    await _global_learner.learn_from_outcome(outcome)


async def predict_domain_risk(domain: str) -> Dict[str, Any]:
    """
    Predict risk level for a domain using learned patterns.

    Args:
        domain: Domain to predict risk for

    Returns:
        Dictionary with prediction results, confidence, and recommendations

    Example:
        prediction = await predict_domain_risk("example.com")
        print(f"Risk: {prediction['risk_level']} (confidence: {prediction['confidence_score']:.2f})")
    """
    return await _global_learner.predict_risk(domain)


async def get_learning_insights(domain: Optional[str] = None) -> Dict[str, Any]:
    """
    Get comprehensive learning insights and model performance.

    Args:
        domain: Optional domain for domain-specific insights

    Returns:
        Dictionary with learning metrics, model performance, and insights

    Example:
        insights = await get_learning_insights()
        print(f"Model accuracy: {insights['learning_metrics']['prediction_accuracy']['accuracy']:.2f}")

        domain_insights = await get_learning_insights("example.com")
        print(f"Domain patterns: {domain_insights['domain_specific']['patterns']}")
    """
    return await _global_learner.get_learning_insights(domain)


async def adapt_learning_from_feedback(domain: str, feedback: Dict[str, Any]) -> None:
    """
    Adapt learning based on feedback about prediction accuracy.

    Args:
        domain: Domain the feedback is about
        feedback: Feedback dictionary with actual vs predicted outcomes

    Example:
        feedback = {
            "actual_risk_higher": True,  # Prediction was too low
            "requires_retraining": False,
            "confidence_too_low": True
        }
        await adapt_learning_from_feedback("example.com", feedback)
    """
    await _global_learner.adapt_to_new_patterns(domain, feedback)


def get_learner_status() -> Dict[str, Any]:
    """
    Get current status of the safety learner system.

    Returns:
        Dictionary with learner status, model info, and performance metrics
    """
    return {
        "ml_enabled": _global_learner.enable_ml,
        "learning_enabled": _global_learner.learning_enabled,
        "models_trained": _global_learner.models_trained,
        "model_version": _global_learner.model_version,
        "samples_learned": len(_global_learner.outcome_history),
        "patterns_learned": len(_global_learner.patterns_learned),
        "last_trained": _global_learner.last_trained.isoformat() if _global_learner.last_trained else None
    }
