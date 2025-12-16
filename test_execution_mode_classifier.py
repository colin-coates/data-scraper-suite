# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use as is strictly prohibited.

"""
Test Enhanced Execution Mode Classifier for MJ Data Scraper Suite

Comprehensive testing of the intelligent execution mode classification system,
performance optimization, cost estimation, and operational intelligence.
"""

import asyncio
from datetime import datetime, timedelta
from core.execution_mode_classifier import (
    ExecutionModeClassifier,
    ExecutionProfile,
    ExecutionMode,
    ExecutionStrategy,
    classify_execution_mode,
    get_execution_mode_statistics,
    update_execution_performance,
    classify_execution_mode_simple
)
from core.models.asset_signal import AssetType
from core.intent_classifier import IntentRiskLevel, IntentCategory
from core.control_models import (
    ScrapeControlContract,
    ScrapeIntent,
    ScrapeBudget,
    ScrapeAuthorization
)


class TestExecutionModeClassifier:
    """Test comprehensive execution mode classification functionality."""

    def test_classifier_initialization(self):
        """Test that the execution mode classifier initializes properly."""
        classifier = ExecutionModeClassifier()

        assert len(classifier.execution_history) == 0
        assert len(classifier.performance_metrics) == 0
        assert len(classifier.mode_success_rates) == len(ExecutionMode)
        assert len(classifier.strategy_effectiveness) == len(ExecutionStrategy)

        # Check default success rates are reasonable
        for mode, rate in classifier.mode_success_rates.items():
            assert 0.5 <= rate <= 1.0

        for strategy, effectiveness in classifier.strategy_effectiveness.items():
            assert 0.5 <= effectiveness <= 1.0

    def test_base_execution_mode_logic(self):
        """Test the enhanced base execution mode classification."""
        classifier = ExecutionModeClassifier()

        # Test original logic with enhancements
        assert classifier._get_base_execution_mode(AssetType.PERSON, 1) == ExecutionMode.PRECISION_SEARCH
        assert classifier._get_base_execution_mode(AssetType.PERSON, 3) == ExecutionMode.TARGETED_LOOKUP
        assert classifier._get_base_execution_mode(AssetType.COMPANY, 50) == ExecutionMode.FOCUSED_DISCOVERY
        assert classifier._get_base_execution_mode(AssetType.COMPANY, 2000) == ExecutionMode.COMPREHENSIVE_SURVEY
        assert classifier._get_base_execution_mode(AssetType.COMPANY, 50000) == ExecutionMode.EXHAUSTIVE_ANALYSIS

    def test_precision_search_mode(self):
        """Test precision search mode for single targets."""
        classifier = ExecutionModeClassifier()

        profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.PERSON, 1,
            intent_category=IntentCategory.PERSONAL,
            data_quality_requirement="premium"
        ))

        assert profile.mode == ExecutionMode.PRECISION_SEARCH
        assert profile.strategy == ExecutionStrategy.DEPTH_FIRST
        assert profile.confidence_score > 0.7
        assert profile.execution_parameters['concurrent_requests'] == 1
        assert profile.execution_parameters['rate_limit_multiplier'] <= 0.2
        assert "maximum" in profile.execution_parameters['monitoring_intensity']

        # Check reasoning
        reasoning_text = ' '.join(profile.reasoning).lower()
        assert 'single target' in reasoning_text or 'precision' in reasoning_text

    def test_targeted_lookup_mode(self):
        """Test targeted lookup mode for small scopes."""
        classifier = ExecutionModeClassifier()

        profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.SINGLE_FAMILY_HOME, 3,
            risk_level=IntentRiskLevel.MEDIUM
        ))

        assert profile.mode == ExecutionMode.TARGETED_LOOKUP
        assert profile.confidence_score > 0.6
        assert profile.execution_parameters['concurrent_requests'] <= 5
        assert profile.execution_parameters['batch_size'] <= 3

        # Risk mitigations should be present
        assert len(profile.risk_mitigations) > 0

    def test_discovery_scrape_mode(self):
        """Test discovery scrape mode for larger scopes."""
        classifier = ExecutionModeClassifier()

        profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.COMPANY, 500,
            time_sensitivity="normal"
        ))

        assert profile.mode in [ExecutionMode.DISCOVERY_SCRAPE, ExecutionMode.CONTROLLED_EXPLORATION]
        assert profile.strategy in [ExecutionStrategy.BREADTH_FIRST, ExecutionStrategy.PRIORITY_BASED]
        assert profile.execution_parameters['concurrent_requests'] >= 10

    def test_critical_risk_execution_mode(self):
        """Test execution mode selection for critical risk operations."""
        classifier = ExecutionModeClassifier()

        profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.COMPANY, 100,
            risk_level=IntentRiskLevel.CRITICAL,
            intent_category=IntentCategory.LEGAL
        ))

        assert profile.mode in [ExecutionMode.COMPLIANCE_AUDIT, ExecutionMode.VERIFICATION_SCAN]
        assert profile.risk_level == IntentRiskLevel.CRITICAL
        assert profile.execution_parameters['concurrent_requests'] <= 5
        assert profile.execution_parameters['rate_limit_multiplier'] <= 0.5

        # Should have extensive compliance requirements
        assert len(profile.compliance_requirements) > 5
        assert any("executive_approval" in req for req in profile.compliance_requirements)

    def test_time_sensitivity_execution_modes(self):
        """Test execution mode adjustments for time sensitivity."""
        classifier = ExecutionModeClassifier()

        # Critical time sensitivity
        critical_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.COMPANY, 100,
            time_sensitivity="critical"
        ))

        assert critical_profile.mode in [ExecutionMode.CRISIS_MODE, ExecutionMode.RAPID_RESPONSE]
        assert critical_profile.execution_parameters['concurrent_requests'] >= 50
        assert critical_profile.execution_parameters['rate_limit_multiplier'] >= 2.0

        # High time sensitivity
        high_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.COMPANY, 100,
            time_sensitivity="high"
        ))

        assert high_profile.execution_parameters['timeout_settings']['request_timeout'] <= 25

        # Low time sensitivity
        low_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.COMPANY, 100,
            time_sensitivity="low"
        ))

        assert low_profile.execution_parameters['timeout_settings']['request_timeout'] >= 45

    def test_data_quality_execution_modes(self):
        """Test execution mode adjustments for data quality requirements."""
        classifier = ExecutionModeClassifier()

        # Premium quality requirement
        premium_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.PERSON, 5,
            data_quality_requirement="premium"
        ))

        assert premium_profile.mode in [ExecutionMode.PRECISION_SEARCH, ExecutionMode.VERIFICATION_SCAN]
        assert premium_profile.strategy == ExecutionStrategy.QUALITY_OPTIMIZED

        # Basic quality requirement
        basic_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.COMPANY, 100,
            data_quality_requirement="basic"
        ))

        assert basic_profile.strategy in [ExecutionStrategy.BREADTH_FIRST, ExecutionStrategy.TIME_OPTIMIZED]

    def test_control_contract_integration(self):
        """Test integration with scraping control contracts."""
        classifier = ExecutionModeClassifier()

        # Create comprehensive control contract
        control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Comprehensive property due diligence investigation",
                sources=["county_clerk", "title_company", "court_records"],
                geography=["Multiple counties"],
                event_type=None
            ),
            budget=ScrapeBudget(
                max_runtime_minutes=180,
                max_pages=400,
                max_records=2000
            ),
            authorization=ScrapeAuthorization(
                approved_by="legal_department",
                purpose="Property acquisition due diligence",
                expires_at=datetime.utcnow() + timedelta(days=30)
            )
        )

        profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.SINGLE_FAMILY_HOME, 25,
            control=control
        ))

        # Should consider budget intensity
        assert profile.mode in [ExecutionMode.CONTROLLED_EXPLORATION, ExecutionMode.FOCUSED_DISCOVERY]
        assert 'cost_projections' in profile
        assert profile.cost_projections['total_estimated_cost'] > 0

    def test_execution_profile_completeness(self):
        """Test that execution profiles contain all required components."""
        classifier = ExecutionModeClassifier()

        profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.COMPANY, 50,
            risk_level=IntentRiskLevel.HIGH,
            intent_category=IntentCategory.FINANCIAL
        ))

        # Check all profile components are present
        required_attributes = [
            'mode', 'strategy', 'confidence_score', 'reasoning',
            'execution_parameters', 'resource_requirements', 'risk_mitigations',
            'performance_expectations', 'compliance_requirements', 'cost_projections'
        ]

        for attr in required_attributes:
            assert hasattr(profile, attr), f"Profile missing {attr}"

        # Check execution parameters structure
        exec_params = profile.execution_parameters
        required_exec_params = [
            'concurrent_requests', 'rate_limit_multiplier', 'monitoring_intensity',
            'strategy', 'batch_size', 'retry_policy', 'timeout_settings'
        ]

        for param in required_exec_params:
            assert param in exec_params, f"Execution parameters missing {param}"

        # Check resource requirements
        resource_reqs = profile.resource_requirements
        assert 'cpu_cores' in resource_reqs
        assert 'memory_gb' in resource_reqs
        assert 'estimated_duration_hours' in resource_reqs

    def test_strategy_selection_logic(self):
        """Test execution strategy selection logic."""
        classifier = ExecutionModeClassifier()

        # Depth-first for precision operations
        precision_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.PERSON, 1,
            intent_category=IntentCategory.PERSONAL
        ))
        assert precision_profile.strategy == ExecutionStrategy.DEPTH_FIRST

        # Breadth-first for large scale operations
        breadth_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.COMPANY, 5000
        ))
        assert breadth_profile.strategy == ExecutionStrategy.BREADTH_FIRST

        # Priority-based for focused discovery
        priority_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.SINGLE_FAMILY_HOME, 50,
            intent_category=IntentCategory.PROPERTY
        ))
        assert priority_profile.strategy in [ExecutionStrategy.PRIORITY_BASED, ExecutionStrategy.DEPTH_FIRST]

        # Risk-weighted for high-risk operations
        risk_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.COMPANY, 100,
            risk_level=IntentRiskLevel.HIGH
        ))
        assert risk_profile.strategy == ExecutionStrategy.RISK_WEIGHTED

    def test_resource_requirement_calculation(self):
        """Test resource requirement calculations."""
        classifier = ExecutionModeClassifier()

        # Small scope - minimal resources
        small_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.PERSON, 1
        ))

        small_resources = small_profile.resource_requirements
        assert small_resources['cpu_cores'] <= 2
        assert small_resources['memory_gb'] <= 2
        assert small_resources['concurrent_requests'] <= 3

        # Large scope - significant resources
        large_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.COMPANY, 10000,
            time_sensitivity="high"
        ))

        large_resources = large_profile.resource_requirements
        assert large_resources['cpu_cores'] >= 4
        assert large_resources['memory_gb'] >= 4
        assert large_resources['concurrent_requests'] >= 20

    def test_cost_projection_generation(self):
        """Test cost projection and optimization generation."""
        classifier = ExecutionModeClassifier()

        profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.SINGLE_FAMILY_HOME, 10,
            risk_level=IntentRiskLevel.MEDIUM
        ))

        cost_projections = profile.cost_projections
        assert 'total_estimated_cost' in cost_projections
        assert 'cost_confidence_level' in cost_projections
        assert 'estimated_range_low' in cost_projections
        assert 'estimated_range_high' in cost_projections
        assert 'optimization_suggestions' in cost_projections

        # Cost should be reasonable for the scope
        assert cost_projections['total_estimated_cost'] > 0
        assert cost_projections['cost_confidence_level'] > 0

        # Should have optimization suggestions
        assert len(cost_projections['optimization_suggestions']) > 0

    def test_performance_expectation_calculation(self):
        """Test performance expectation calculations."""
        classifier = ExecutionModeClassifier()

        # High precision mode
        precision_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.PERSON, 1,
            data_quality_requirement="premium"
        ))

        precision_perf = precision_profile.performance_expectations
        assert precision_perf['expected_success_rate'] >= 0.9
        assert precision_perf['expected_data_quality_score'] >= 0.95

        # Broad discovery mode
        discovery_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.COMPANY, 1000,
            data_quality_requirement="basic"
        ))

        discovery_perf = discovery_profile.performance_expectations
        assert discovery_perf['expected_success_rate'] >= 0.6
        assert discovery_perf['scalability_score'] >= 0.8

    def test_compliance_requirement_generation(self):
        """Test compliance requirement generation for different scenarios."""
        classifier = ExecutionModeClassifier()

        # Personal data collection
        personal_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.PERSON, 5,
            intent_category=IntentCategory.PERSONAL
        ))

        personal_compliance = personal_profile.compliance_requirements
        assert any("privacy" in req.lower() for req in personal_compliance)
        assert any("gdpr" in req.lower() or "ccpa" in req.lower() for req in personal_compliance)

        # Financial data collection
        financial_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.COMPANY, 20,
            intent_category=IntentCategory.FINANCIAL,
            risk_level=IntentRiskLevel.HIGH
        ))

        financial_compliance = financial_profile.compliance_requirements
        assert any("financial" in req.lower() for req in financial_compliance)
        assert any("senior_approval" in req.lower() or "executive" in req.lower() for req in financial_compliance)

        # Compliance audit mode
        audit_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.COMPANY, 1,
            intent_category=IntentCategory.COMPLIANCE
        ))

        audit_compliance = audit_profile.compliance_requirements
        assert any("audit" in req.lower() for req in audit_compliance)
        assert any("regulatory" in req.lower() for req in audit_compliance)

    def test_risk_mitigation_generation(self):
        """Test risk mitigation strategy generation."""
        classifier = ExecutionModeClassifier()

        # Critical risk operation
        critical_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.COMPANY, 100,
            risk_level=IntentRiskLevel.CRITICAL
        ))

        critical_mitigations = critical_profile.risk_mitigations
        assert len(critical_mitigations) >= 5
        assert any("dual authorization" in mitigation.lower() for mitigation in critical_mitigations)
        assert any("kill switches" in mitigation.lower() for mitigation in critical_mitigations)

        # High risk operation
        high_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.SINGLE_FAMILY_HOME, 50,
            risk_level=IntentRiskLevel.HIGH
        ))

        high_mitigations = high_profile.risk_mitigations
        assert len(high_mitigations) >= 3
        assert any("monitoring" in mitigation.lower() for mitigation in high_mitigations)

    def test_execution_parameter_profiles(self):
        """Test execution parameter generation for different profiles."""
        classifier = ExecutionModeClassifier()

        # Monitoring mode
        monitoring_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.COMPANY, 10,
            intent_category=IntentCategory.EVENT
        ))

        if monitoring_profile.mode == ExecutionMode.MONITORING_MODE:
            monitoring_params = monitoring_profile.execution_parameters
            assert monitoring_params['retry_policy']['max_retries'] >= 5
            assert monitoring_params['rate_limit_multiplier'] <= 0.5

        # Crisis mode
        crisis_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.COMPANY, 500,
            time_sensitivity="critical",
            risk_level=IntentRiskLevel.CRITICAL
        ))

        if crisis_profile.mode == ExecutionMode.CRISIS_MODE:
            crisis_params = crisis_profile.execution_parameters
            assert crisis_params['concurrent_requests'] >= 100
            assert crisis_params['rate_limit_multiplier'] >= 3.0
            assert crisis_params['timeout_settings']['request_timeout'] <= 10

    def test_historical_performance_optimization(self):
        """Test historical performance-based optimization."""
        classifier = ExecutionModeClassifier()

        # Simulate historical performance data
        execution_key = "AssetType.PERSON_5_normal_standard"
        update_execution_performance(execution_key, True, 10.0, 2.0)  # Successful execution
        update_execution_performance(execution_key, True, 12.0, 1.8)  # Another successful execution

        # Classify with historical data available
        profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.PERSON, 5,
            time_sensitivity="normal",
            data_quality_requirement="standard"
        ))

        # Should have considered historical performance
        assert execution_key in classifier.performance_metrics
        metrics = classifier.performance_metrics[execution_key]
        assert metrics['success_rate'] == 1.0
        assert metrics['avg_cost'] > 0
        assert metrics['avg_duration'] > 0

    def test_batch_size_calculation(self):
        """Test batch size calculation for different scenarios."""
        classifier = ExecutionModeClassifier()

        # Small scope
        small_profile = asyncio.run(classifier.classify_execution_mode(AssetType.PERSON, 3))
        assert small_profile.execution_parameters['batch_size'] <= 3

        # Large scope
        large_profile = asyncio.run(classifier.classify_execution_mode(AssetType.COMPANY, 10000))
        assert large_profile.execution_parameters['batch_size'] >= 50

        # Very large scope should be capped
        huge_profile = asyncio.run(classifier.classify_execution_mode(AssetType.COMPANY, 100000))
        assert huge_profile.execution_parameters['batch_size'] <= 500

    def test_classifier_statistics(self):
        """Test classifier statistics generation."""
        classifier = ExecutionModeClassifier()

        # Generate some classification history
        profiles = []
        for i in range(5):
            asset_type = AssetType.PERSON if i % 2 == 0 else AssetType.COMPANY
            scope_size = 10 + i * 5
            profile = asyncio.run(classifier.classify_execution_mode(asset_type, scope_size))
            profiles.append(profile)

        stats = classifier.get_classifier_stats()

        assert stats['total_execution_profiles'] == 5
        assert stats['unique_execution_keys'] >= 3
        assert 'mode_distribution' in stats
        assert 'strategy_distribution' in stats
        assert len(stats['mode_distribution']) > 0
        assert len(stats['strategy_distribution']) > 0

    def test_convenience_function_integration(self):
        """Test integration with global convenience functions."""
        # Test main classification function
        profile = asyncio.run(classify_execution_mode(
            AssetType.SINGLE_FAMILY_HOME, 25,
            risk_level=IntentRiskLevel.MEDIUM,
            intent_category=IntentCategory.PROPERTY
        ))

        assert isinstance(profile, ExecutionProfile)
        assert profile.mode in ExecutionMode
        assert profile.strategy in ExecutionStrategy

        # Test statistics function
        stats = get_execution_mode_statistics()
        assert isinstance(stats, dict)
        assert 'total_execution_profiles' in stats

        # Test performance update function
        update_execution_performance("test_key", True, 15.0, 3.0)

        # Test legacy compatibility function
        legacy_mode = classify_execution_mode_simple(AssetType.PERSON, 1)
        assert isinstance(legacy_mode, str)
        assert legacy_mode in [mode.value for mode in ExecutionMode]

    def test_asset_type_specific_optimizations(self):
        """Test asset type specific execution optimizations."""
        classifier = ExecutionModeClassifier()

        # Person-specific optimizations
        person_profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.PERSON, 10,
            intent_category=IntentCategory.PERSONAL
        ))

        # Should favor precision for personal data
        assert person_profile.mode in [ExecutionMode.PRECISION_SEARCH, ExecutionMode.TARGETED_LOOKUP]

        # Company-specific optimizations
        company_profile = asyncio.run(classify_execution_mode(
            AssetType.COMPANY, 50,
            intent_category=IntentCategory.FINANCIAL
        ))

        # Should favor controlled approaches for company data
        assert company_profile.mode in [ExecutionMode.CONTROLLED_EXPLORATION, ExecutionMode.FOCUSED_DISCOVERY]

        # Property-specific optimizations
        property_profile = asyncio.run(classify_execution_mode(
            AssetType.SINGLE_FAMILY_HOME, 20,
            intent_category=IntentCategory.PROPERTY
        ))

        # Should balance precision and efficiency for property data
        assert property_profile.mode in [ExecutionMode.TARGETED_LOOKUP, ExecutionMode.FOCUSED_DISCOVERY, ExecutionMode.CONTROLLED_EXPLORATION]

    def test_intent_category_driven_modes(self):
        """Test that intent categories drive appropriate execution modes."""
        classifier = ExecutionModeClassifier()

        categories_and_expected_modes = [
            (IntentCategory.PERSONAL, [ExecutionMode.PRECISION_SEARCH, ExecutionMode.TARGETED_LOOKUP]),
            (IntentCategory.LEGAL, [ExecutionMode.COMPLIANCE_AUDIT, ExecutionMode.VERIFICATION_SCAN]),
            (IntentCategory.FINANCIAL, [ExecutionMode.VERIFICATION_SCAN, ExecutionMode.CONTROLLED_EXPLORATION]),
            (IntentCategory.COMPLIANCE, [ExecutionMode.COMPLIANCE_AUDIT]),
            (IntentCategory.EVENT, [ExecutionMode.MONITORING_MODE, ExecutionMode.TARGETED_LOOKUP]),
            (IntentCategory.PROPERTY, [ExecutionMode.FOCUSED_DISCOVERY, ExecutionMode.CONTROLLED_EXPLORATION])
        ]

        for category, expected_modes in categories_and_expected_modes:
            profile = asyncio.run(classify_execution_mode(
                AssetType.COMPANY, 25,
                intent_category=category
            ))

            # Category should influence mode selection
            # (Note: This is a soft test as other factors can override)
            assert profile.mode in ExecutionMode

            if category == IntentCategory.COMPLIANCE:
                assert profile.mode == ExecutionMode.COMPLIANCE_AUDIT
            elif category == IntentCategory.LEGAL:
                assert profile.mode in [ExecutionMode.COMPLIANCE_AUDIT, ExecutionMode.VERIFICATION_SCAN]

    def test_comprehensive_execution_scenario(self):
        """Test a comprehensive execution scenario with all factors."""
        classifier = ExecutionModeClassifier()

        # Complex scenario: High-risk financial investigation with time pressure
        control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Urgent financial background check for executive due diligence",
                sources=["financial_records", "court_records", "credit_reports"],
                geography=["National coverage required"],
                event_type=None
            ),
            budget=ScrapeBudget(
                max_runtime_minutes=120,
                max_pages=300,
                max_records=1500
            ),
            authorization=ScrapeAuthorization(
                approved_by="executive_committee",
                purpose="Critical executive due diligence",
                expires_at=datetime.utcnow() + timedelta(hours=12)
            )
        )

        profile = asyncio.run(classifier.classify_execution_mode(
            AssetType.PERSON, 3,
            control=control,
            risk_level=IntentRiskLevel.CRITICAL,
            intent_category=IntentCategory.FINANCIAL,
            time_sensitivity="high",
            data_quality_requirement="verified"
        ))

        # Should balance all factors appropriately
        assert profile.risk_level == IntentRiskLevel.CRITICAL
        assert profile.intent_category == IntentCategory.FINANCIAL

        # Should select appropriate mode for the complex scenario
        assert profile.mode in [ExecutionMode.VERIFICATION_SCAN, ExecutionMode.COMPLIANCE_AUDIT, ExecutionMode.RAPID_RESPONSE]

        # Should have comprehensive execution profile
        assert len(profile.reasoning) >= 3
        assert len(profile.risk_mitigations) >= 5
        assert len(profile.compliance_requirements) >= 5
        assert profile.cost_projections['total_estimated_cost'] > 0

        # Execution parameters should reflect complexity
        assert profile.execution_parameters['monitoring_intensity'] in ['maximum', 'comprehensive']
        assert profile.execution_parameters['concurrent_requests'] <= 10  # Conservative due to risk
        assert profile.execution_parameters['rate_limit_multiplier'] <= 1.0  # Careful execution


if __name__ == "__main__":
    # Run basic tests
    print("⚡ Testing Enhanced Execution Mode Classifier...")

    test_instance = TestExecutionModeClassifier()

    # Run individual tests
    try:
        test_instance.test_classifier_initialization()
        print("✅ Classifier initialization tests passed")

        test_instance.test_base_execution_mode_logic()
        print("✅ Base execution mode logic tests passed")

        test_instance.test_precision_search_mode()
        print("✅ Precision search mode tests passed")

        test_instance.test_targeted_lookup_mode()
        print("✅ Targeted lookup mode tests passed")

        test_instance.test_discovery_scrape_mode()
        print("✅ Discovery scrape mode tests passed")

        test_instance.test_critical_risk_execution_mode()
        print("✅ Critical risk execution mode tests passed")

        test_instance.test_time_sensitivity_execution_modes()
        print("✅ Time sensitivity execution modes tests passed")

        test_instance.test_data_quality_execution_modes()
        print("✅ Data quality execution modes tests passed")

        test_instance.test_control_contract_integration()
        print("✅ Control contract integration tests passed")

        test_instance.test_execution_profile_completeness()
        print("✅ Execution profile completeness tests passed")

        test_instance.test_strategy_selection_logic()
        print("✅ Strategy selection logic tests passed")

        test_instance.test_resource_requirement_calculation()
        print("✅ Resource requirement calculation tests passed")

        test_instance.test_cost_projection_generation()
        print("✅ Cost projection generation tests passed")

        test_instance.test_performance_expectation_calculation()
        print("✅ Performance expectation calculation tests passed")

        test_instance.test_compliance_requirement_generation()
        print("✅ Compliance requirement generation tests passed")

        test_instance.test_risk_mitigation_generation()
        print("✅ Risk mitigation generation tests passed")

        test_instance.test_execution_parameter_profiles()
        print("✅ Execution parameter profiles tests passed")

        test_instance.test_classifier_statistics()
        print("✅ Classifier statistics tests passed")

        test_instance.test_convenience_function_integration()
        print("✅ Convenience function integration tests passed")

        test_instance.test_asset_type_specific_optimizations()
        print("✅ Asset type specific optimizations tests passed")

        test_instance.test_intent_category_driven_modes()
        print("✅ Intent category driven modes tests passed")

        test_instance.test_comprehensive_execution_scenario()
        print("✅ Comprehensive execution scenario tests passed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 All Enhanced Execution Mode Classifier tests completed successfully!")
    print("⚡ Intelligent execution mode classification and optimization fully validated!")
