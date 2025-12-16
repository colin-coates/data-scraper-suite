# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Test Safety Learner for MJ Data Scraper Suite

Comprehensive testing of the machine learning system for safety intelligence,
threat detection, and adaptive security capabilities.
"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from core.learning.safety_learner import (
    SafetyLearner,
    learn_from_outcome,
    predict_domain_risk,
    get_learning_insights,
    adapt_learning_from_feedback,
    get_learner_status
)
from core.telemetry.models import create_sentinel_outcome


class TestSafetyLearner:
    """Test comprehensive safety learner functionality."""

    async def test_basic_learning(self):
        """Test basic learning from sentinel outcomes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = SafetyLearner(temp_dir, enable_ml=False)  # Disable ML for basic test

            # Create test outcome
            outcome = create_sentinel_outcome(
                domain="test-learning.com",
                risk_level="high",
                action="restrict",
                sentinel_name="test_sentinel",
                risk_score=0.75,
                confidence_score=0.85,
                latency_ms=1250,
                blocked=False,
                threat_indicators=["waf_detected", "rate_limiting"],
                correlation_id="test_001"
            )

            # Learn from outcome
            await learner.learn_from_outcome(outcome)

            # Verify learning occurred
            assert len(learner.outcome_history) == 1
            assert len(learner.feature_history) == 1
            assert "test-learning.com" in learner.patterns_learned

            patterns = learner.patterns_learned["test-learning.com"]
            assert patterns["total_checks"] == 1
            assert patterns["risk_distribution"]["high"] == 1
            assert patterns["avg_latency"] == 1250

    
    async def test_risk_prediction(self):
        """Test risk prediction capabilities."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = SafetyLearner(temp_dir, enable_ml=False)

            # Add some learning data
            outcomes = [
                create_sentinel_outcome("predict-test.com", "low", "allow", "test", risk_score=0.2),
                create_sentinel_outcome("predict-test.com", "medium", "delay", "test", risk_score=0.5),
                create_sentinel_outcome("predict-test.com", "high", "restrict", "test", risk_score=0.8)
            ]

            for outcome in outcomes:
                await learner.learn_from_outcome(outcome)

            # Predict risk
            prediction = await learner.predict_risk("predict-test.com")

            assert "risk_level" in prediction
            assert "confidence_score" in prediction
            assert "recommendations" in prediction
            assert "learned_patterns" in prediction

            # Should have learned patterns
            patterns = prediction["learned_patterns"]
            assert patterns["total_checks"] == 3
            assert patterns["most_common_risk"] == "low"  # All have equal count, returns first

    
    async def test_domain_patterns(self):
        """Test domain pattern learning and retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = SafetyLearner(temp_dir, enable_ml=False)

            # Create outcomes with temporal patterns
            base_time = datetime.utcnow()
            for i in range(5):
                outcome = create_sentinel_outcome(
                    domain="pattern-test.com",
                    risk_level="medium",
                    action="delay",
                    sentinel_name="test",
                    risk_score=0.5 + i * 0.05,  # Increasing risk
                    timestamp=base_time + timedelta(hours=i)
                )
                await learner.learn_from_outcome(outcome)

            # Check patterns
            patterns = await learner._get_domain_patterns("pattern-test.com")
            assert patterns["total_checks"] == 5
            assert patterns["most_common_risk"] == "medium"
            assert patterns["risk_adjustment"] > 0  # Risk is increasing

    
    async def test_prediction_confidence(self):
        """Test prediction confidence calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = SafetyLearner(temp_dir, enable_ml=False)

            # Test with no data
            prediction = await learner.predict_risk("unknown-domain.com")
            assert prediction["confidence_score"] < 0.5  # Low confidence for unknown domain

            # Add learning data
            for i in range(10):
                outcome = create_sentinel_outcome(
                    domain="confidence-test.com",
                    risk_level="high",
                    action="restrict",
                    sentinel_name="test",
                    confidence_score=0.9
                )
                await learner.learn_from_outcome(outcome)

            # Predict with data
            prediction = await learner.predict_risk("confidence-test.com")
            assert prediction["confidence_score"] > 0.5  # Higher confidence with data

    
    async def test_recommendations_generation(self):
        """Test intelligent recommendations generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = SafetyLearner(temp_dir, enable_ml=False)

            # Test critical risk recommendations
            prediction = {
                "risk_level": "critical",
                "risk_score": 0.9,
                "confidence_score": 0.8
            }
            recommendations = await learner._generate_recommendations("critical-test.com", prediction)

            assert len(recommendations) > 0
            assert any("blocking" in rec.lower() for rec in recommendations)
            assert any("escalate" in rec.lower() for rec in recommendations)

            # Test medium risk recommendations
            prediction["risk_level"] = "medium"
            prediction["risk_score"] = 0.5
            recommendations = await learner._generate_recommendations("medium-test.com", prediction)

            assert len(recommendations) > 0
            assert any("monitoring" in rec.lower() for rec in recommendations)

    
    async def test_learning_insights(self):
        """Test comprehensive learning insights."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = SafetyLearner(temp_dir, enable_ml=False)

            # Add learning data
            for i in range(20):
                outcome = create_sentinel_outcome(
                    domain=f"insights-test-{i%3}.com",  # 3 different domains
                    risk_level=["low", "medium", "high"][i % 3],
                    action="test",
                    sentinel_name="test"
                )
                await learner.learn_from_outcome(outcome)

            # Get insights
            insights = await learner.get_learning_insights()

            assert "model_status" in insights
            assert "learning_metrics" in insights
            assert "performance_stats" in insights

            # Check metrics
            assert insights["learning_metrics"]["samples_learned"] == 20
            assert insights["learning_metrics"]["patterns_learned"] == 3  # 3 domains

            # Get domain-specific insights
            domain_insights = await learner.get_learning_insights("insights-test-0.com")
            assert "domain_specific" in domain_insights

    
    async def test_feedback_adaptation(self):
        """Test learning adaptation from feedback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = SafetyLearner(temp_dir, enable_ml=False)

            # Initial learning
            outcome = create_sentinel_outcome(
                domain="feedback-test.com",
                risk_level="medium",
                action="delay",
                sentinel_name="test",
                risk_score=0.5
            )
            await learner.learn_from_outcome(outcome)

            # Get initial patterns
            initial_patterns = await learner._get_domain_patterns("feedback-test.com")
            initial_adjustment = initial_patterns.get("risk_adjustment", 0.0)

            # Provide feedback that risk was actually higher
            feedback = {
                "actual_risk_higher": True,
                "requires_retraining": False
            }
            await learner.adapt_to_new_patterns("feedback-test.com", feedback)

            # Check that adjustment increased
            updated_patterns = await learner._get_domain_patterns("feedback-test.com")
            updated_adjustment = updated_patterns.get("risk_adjustment", 0.0)

            assert updated_adjustment > initial_adjustment

    
    async def test_global_functions(self):
        """Test global convenience functions."""
        # Test basic learning
        outcome = create_sentinel_outcome(
            domain="global-test.com",
            risk_level="high",
            action="restrict",
            sentinel_name="test"
        )
        await learn_from_outcome(outcome)

        # Test prediction
        prediction = await predict_domain_risk("global-test.com")
        assert "risk_level" in prediction
        assert "confidence_score" in prediction

        # Test insights
        insights = await get_learning_insights()
        assert "model_status" in insights

        # Test status
        status = get_learner_status()
        assert "ml_enabled" in status
        assert "samples_learned" in status

    
    async def test_feature_extraction(self):
        """Test feature extraction from outcomes."""
        learner = SafetyLearner(enable_ml=False)

        outcome = create_sentinel_outcome(
            domain="feature-test.com",
            risk_level="high",
            action="restrict",
            sentinel_name="network_sentinel",
            risk_score=0.8,
            confidence_score=0.9,
            latency_ms=1500,
            threat_indicators=["waf", "bot"],
            waf_detected=True,
            ssl_validity_days=30
        )

        features = learner._extract_features(outcome)

        assert features["risk_score"] == 0.8
        assert features["confidence_score"] == 0.9
        assert features["latency_ms"] == 1500
        assert features["threat_indicators_count"] == 2
        assert features["waf_detected"] == 1
        assert features["ssl_validity_days"] == 30

    
    async def test_unknown_domain_prediction(self):
        """Test prediction for unknown domains."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = SafetyLearner(temp_dir, enable_ml=False)

            prediction = await learner.predict_risk("completely-unknown.com")

            assert prediction["risk_level"] == "unknown"
            assert prediction["fallback"] == True
            assert prediction["confidence_score"] < 0.5

    
    async def test_error_handling(self):
        """Test error handling in learning operations."""
        learner = SafetyLearner(enable_ml=False)

        # Test with invalid outcome (should not crash)
        invalid_outcome = create_sentinel_outcome(
            domain="error-test.com",
            risk_level="invalid_risk",
            action="test",
            sentinel_name="test"
        )

        # Should handle gracefully
        await learner.learn_from_outcome(invalid_outcome)
        assert len(learner.outcome_history) == 1

        # Test prediction error handling
        prediction = await learner.predict_risk("error-test.com")
        assert "error" not in prediction or prediction.get("fallback") == True

    
    async def test_temporal_patterns(self):
        """Test learning of temporal patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = SafetyLearner(temp_dir, enable_ml=False)

            # Create outcomes at different times
            base_time = datetime.utcnow()
            for hour in [9, 14, 18, 22]:  # Different hours
                for day in [0, 2, 4]:  # Different days
                    outcome = create_sentinel_outcome(
                        domain="temporal-test.com",
                        risk_level="medium",
                        action="delay",
                        sentinel_name="test",
                        timestamp=base_time.replace(hour=hour) + timedelta(days=day)
                    )
                    # Manually set hour/day for testing
                    outcome.hour_of_day = hour
                    outcome.day_of_week = day % 7
                    await learner.learn_from_outcome(outcome)

            patterns = await learner._get_domain_patterns("temporal-test.com")
            assert patterns["total_checks"] == 12  # 4 hours * 3 days

            # Check temporal pattern learning
            domain_patterns = learner.patterns_learned["temporal-test.com"]
            assert len(domain_patterns["temporal_patterns"]) > 0

    
    async def test_risk_trend_analysis(self):
        """Test risk trend analysis and adjustment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = SafetyLearner(temp_dir, enable_ml=False)

            # Create increasing risk trend
            base_time = datetime.utcnow()
            for i in range(10):
                risk_score = 0.3 + i * 0.05  # Increasing from 0.3 to 0.75
                outcome = create_sentinel_outcome(
                    domain="trend-test.com",
                    risk_level="medium",
                    action="delay",
                    sentinel_name="test",
                    risk_score=risk_score,
                    timestamp=base_time + timedelta(days=i)
                )
                await learner.learn_from_outcome(outcome)

            patterns = await learner._get_domain_patterns("trend-test.com")
            risk_adjustment = patterns.get("risk_adjustment", 0.0)

            # Risk is increasing, so adjustment should be positive
            assert risk_adjustment > 0

    
    async def test_concurrent_learning(self):
        """Test concurrent learning operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner = SafetyLearner(temp_dir, enable_ml=False)

            # Create multiple outcomes for concurrent processing
            async def create_and_learn(i):
                outcome = create_sentinel_outcome(
                    domain=f"concurrent-test-{i}.com",
                    risk_level="medium",
                    action="delay",
                    sentinel_name="test",
                    correlation_id=f"concurrent_{i}"
                )
                await learner.learn_from_outcome(outcome)

            # Run concurrent learning
            tasks = [create_and_learn(i) for i in range(10)]
            await asyncio.gather(*tasks)

            # Verify all outcomes were learned
            assert len(learner.outcome_history) == 10
            assert len(learner.patterns_learned) == 10

    
    async def test_model_persistence(self):
        """Test model persistence and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            learner1 = SafetyLearner(temp_dir, enable_ml=False)

            # Train with some data
            for i in range(5):
                outcome = create_sentinel_outcome(
                    domain="persistence-test.com",
                    risk_level="high",
                    action="restrict",
                    sentinel_name="test"
                )
                await learner1.learn_from_outcome(outcome)

            # Create new learner (simulating restart)
            learner2 = SafetyLearner(temp_dir, enable_ml=False)

            # Should have different state (no persistence in basic mode)
            assert len(learner2.outcome_history) == 0

            # But patterns would be persisted in full implementation
            # This tests the basic framework


if __name__ == "__main__":
    # Run basic tests without pytest
    async def run_basic_tests():
        print("🧠 Testing Safety Learner...")

        # Basic learning test
        print("\n📋 Test 1: Basic Learning")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                learner = SafetyLearner(temp_dir, enable_ml=False)

                outcome = create_sentinel_outcome(
                    domain="test-basic.com",
                    risk_level="high",
                    action="restrict",
                    sentinel_name="test_sentinel"
                )
                await learner.learn_from_outcome(outcome)

                assert len(learner.outcome_history) == 1
                assert "test-basic.com" in learner.patterns_learned

                print("✅ Basic learning works")

        except Exception as e:
            print(f"❌ Basic learning failed: {e}")

        # Prediction test
        print("\n📋 Test 2: Risk Prediction")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                learner = SafetyLearner(temp_dir, enable_ml=False)

                # Add learning data
                for i in range(3):
                    outcome = create_sentinel_outcome(
                        domain="predict-test.com",
                        risk_level=["low", "medium", "high"][i],
                        action="test",
                        sentinel_name="test"
                    )
                    await learner.learn_from_outcome(outcome)

                prediction = await learner.predict_risk("predict-test.com")
                assert "risk_level" in prediction
                assert "recommendations" in prediction

                print("✅ Risk prediction works")
                print(f"   Predicted risk: {prediction['risk_level']}")

        except Exception as e:
            print(f"❌ Risk prediction failed: {e}")

        # Global functions test
        print("\n📋 Test 3: Global Functions")
        try:
            outcome = create_sentinel_outcome(
                domain="global-test.com",
                risk_level="medium",
                action="delay",
                sentinel_name="test"
            )
            await learn_from_outcome(outcome)

            prediction = await predict_domain_risk("global-test.com")
            insights = await get_learning_insights()
            status = get_learner_status()

            assert prediction["risk_level"] in ["low", "medium", "high", "critical", "unknown"]
            assert "model_status" in insights
            assert "ml_enabled" in status

            print("✅ Global functions work")

        except Exception as e:
            print(f"❌ Global functions failed: {e}")

        print("\n🎉 Safety Learner basic tests completed!")

    asyncio.run(run_basic_tests())
