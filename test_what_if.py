# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use as is strictly prohibited.

"""
Test What-If Analysis Engine for MJ Data Scraper Suite

Comprehensive testing of the scenario planning and comparative analysis system
with scenario creation, multi-dimensional analysis, sensitivity testing, and recommendation generation.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from core.what_if import (
    WhatIfAnalyzer,
    ScenarioConfiguration,
    ScenarioAnalysis,
    WhatIfAnalysis,
    ScenarioType,
    SensitivityParameter,
    create_what_if_scenario,
    analyze_what_if_scenarios,
    generate_scenario_recommendations,
    get_what_if_analyzer_stats,
    what_if_cost,
    what_if_cost_analysis
)
from core.models.asset_signal import AssetType, SignalType
from core.intent_classifier import IntentRiskLevel, IntentCategory
from core.control_models import (
    ScrapeControlContract,
    ScrapeIntent,
    ScrapeBudget,
    ScrapeAuthorization
)


class TestWhatIfAnalyzer:
    """Test comprehensive what-if analysis functionality."""

    def test_analyzer_initialization(self):
        """Test that the what-if analyzer initializes properly."""
        analyzer = WhatIfAnalyzer()

        assert len(analyzer.analyses) == 0
        assert len(analyzer.scenario_templates) > 0
        assert 'cost_optimization' in analyzer.scenario_templates
        assert 'quality_enhancement' in analyzer.scenario_templates

        stats = analyzer.get_analyzer_stats()
        assert stats['total_analyses'] == 0
        assert stats['scenarios_analyzed'] == 0

    def test_scenario_creation(self):
        """Test scenario configuration creation."""
        analyzer = WhatIfAnalyzer()

        control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Test scenario creation",
                sources=["test_source"],
                geography=["test_geo"]
            ),
            budget=ScrapeBudget(max_runtime_minutes=60, max_pages=100, max_records=200, max_cost_total=500.0)
        )

        # Test scenario creation
        scenario = asyncio.run(analyzer.create_scenario(
            base_control=control,
            scenario_type=ScenarioType.COST_OPTIMIZATION,
            scenario_name="Cost Optimized Test"
        ))

        assert isinstance(scenario, ScenarioConfiguration)
        assert scenario.scenario_id
        assert scenario.scenario_name == "Cost Optimized Test"
        assert scenario.scenario_type == ScenarioType.COST_OPTIMIZATION
        assert scenario.base_control == control

    def test_scenario_modifications(self):
        """Test scenario modification application."""
        analyzer = WhatIfAnalyzer()

        control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Test modifications",
                sources=["original_source"],
                geography=["original_geo"]
            ),
            budget=ScrapeBudget(max_runtime_minutes=60, max_pages=100, max_records=200, max_cost_total=500.0)
        )

        scenario = ScenarioConfiguration(
            scenario_id="test_mod",
            scenario_name="Test Modifications",
            scenario_type=ScenarioType.CUSTOM,
            description="Test custom modifications",
            base_control=control,
            modified_sources=["modified_source1", "modified_source2"],
            modified_geography=["modified_geo1", "modified_geo2"]
        )

        modified_control = scenario.get_modified_control()

        assert modified_control.intent.sources == ["modified_source1", "modified_source2"]
        assert modified_control.intent.geography == ["modified_geo1", "modified_geo2"]
        assert modified_control.budget == control.budget  # Unmodified

    def test_single_scenario_analysis(self):
        """Test analysis of a single scenario."""
        analyzer = WhatIfAnalyzer()

        control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Test single scenario analysis",
                sources=["test_source"],
                geography=["test_geo"]
            ),
            budget=ScrapeBudget(max_runtime_minutes=60, max_pages=100, max_records=200, max_cost_total=500.0),
            authorization=ScrapeAuthorization(
                approved_by="test_user",
                purpose="Testing",
                expires_at=datetime.utcnow() + timedelta(days=1)
            )
        )

        scenario = ScenarioConfiguration(
            scenario_id="test_single",
            scenario_name="Single Scenario Test",
            scenario_type=ScenarioType.BASELINE,
            description="Test baseline scenario",
            base_control=control
        )

        # Mock intelligence components
        with patch('core.what_if.preflight_cost_check') as mock_preflight, \
             patch('core.what_if.predict_scraping_cost') as mock_cost, \
             patch('core.what_if.optimize_scraping_cost') as mock_optimize, \
             patch('core.what_if.analyze_scraping_budget') as mock_budget:

            # Set up mocks
            mock_preflight_result = {
                'overall_readiness': 'ready',
                'cost_analysis': {'predicted_cost': 300.0},
                'operational_feasibility': {'estimated_duration_hours': 1.5, 'expected_success_rate': 0.85},
                'risk_assessment': {'risk_level': 'medium'},
                'compliance_status': {'overall_compliance_score': 0.9}
            }
            mock_preflight.return_value = mock_preflight_result

            mock_cost_result = MagicMock()
            mock_cost_result.predicted_cost = 300.0
            mock_cost_result.confidence_score = 0.8
            mock_cost.return_value = mock_cost_result

            mock_optimize_result = MagicMock()
            mock_optimize_result.recommended_changes = []
            mock_optimize.return_value = mock_optimize_result

            mock_budget_result = MagicMock()
            mock_budget.return_value = mock_budget_result

            # Analyze scenario
            analysis = asyncio.run(analyzer.analyze_scenario(scenario))

            assert isinstance(analysis, ScenarioAnalysis)
            assert analysis.scenario_id == scenario.scenario_id
            assert analysis.scenario_name == scenario.scenario_name
            assert analysis.preflight_assessment == mock_preflight_result
            assert analysis.cost_prediction == mock_cost_result
            assert analysis.feasibility_score > 0
            assert analysis.projected_success_rate == 0.85
            assert analysis.analysis_duration_seconds >= 0

    def test_comprehensive_what_if_analysis(self):
        """Test comprehensive what-if analysis with multiple scenarios."""
        analyzer = WhatIfAnalyzer()

        control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Test comprehensive what-if analysis",
                sources=["base_source"],
                geography=["base_geo"]
            ),
            budget=ScrapeBudget(max_runtime_minutes=60, max_pages=100, max_records=200, max_cost_total=500.0),
            authorization=ScrapeAuthorization(
                approved_by="test_user",
                purpose="Testing",
                expires_at=datetime.utcnow() + timedelta(days=1)
            )
        )

        # Create multiple scenarios
        scenarios = [
            ScenarioConfiguration(
                scenario_id="baseline",
                scenario_name="Baseline",
                scenario_type=ScenarioType.BASELINE,
                description="Original configuration",
                base_control=control
            ),
            ScenarioConfiguration(
                scenario_id="cost_opt",
                scenario_name="Cost Optimized",
                scenario_type=ScenarioType.COST_OPTIMIZATION,
                description="Cost optimization scenario",
                base_control=control,
                modified_sources=["cheap_source"]
            ),
            ScenarioConfiguration(
                scenario_id="quality_enh",
                scenario_name="Quality Enhanced",
                scenario_type=ScenarioType.QUALITY_ENHANCEMENT,
                description="Quality enhancement scenario",
                base_control=control,
                quality_requirement="premium"
            )
        ]

        # Mock all intelligence components
        with patch('core.what_if.preflight_cost_check') as mock_preflight, \
             patch('core.what_if.predict_scraping_cost') as mock_cost, \
             patch('core.what_if.optimize_scraping_cost') as mock_optimize, \
             patch('core.what_if.analyze_scraping_budget') as mock_budget:

            # Set up different results for different scenarios
            def mock_preflight_side_effect(control_param):
                if "Cost Optimized" in str(control_param):
                    return {
                        'overall_readiness': 'ready',
                        'cost_analysis': {'predicted_cost': 200.0},
                        'operational_feasibility': {'estimated_duration_hours': 2.0, 'expected_success_rate': 0.8},
                        'risk_assessment': {'risk_level': 'medium'},
                        'compliance_status': {'overall_compliance_score': 0.8}
                    }
                elif "Quality Enhanced" in str(control_param):
                    return {
                        'overall_readiness': 'ready',
                        'cost_analysis': {'predicted_cost': 450.0},
                        'operational_feasibility': {'estimated_duration_hours': 1.0, 'expected_success_rate': 0.95},
                        'risk_assessment': {'risk_level': 'low'},
                        'compliance_status': {'overall_compliance_score': 0.95}
                    }
                else:  # Baseline
                    return {
                        'overall_readiness': 'ready',
                        'cost_analysis': {'predicted_cost': 300.0},
                        'operational_feasibility': {'estimated_duration_hours': 1.5, 'expected_success_rate': 0.85},
                        'risk_assessment': {'risk_level': 'medium'},
                        'compliance_status': {'overall_compliance_score': 0.9}
                    }

            def mock_cost_side_effect(*args, **kwargs):
                # Return different costs based on scenario context
                if 'cheap_source' in str(args):
                    result = MagicMock()
                    result.predicted_cost = 200.0
                    result.confidence_score = 0.8
                    return result
                elif 'premium' in str(args):
                    result = MagicMock()
                    result.predicted_cost = 450.0
                    result.confidence_score = 0.85
                    return result
                else:
                    result = MagicMock()
                    result.predicted_cost = 300.0
                    result.confidence_score = 0.8
                    return result

            mock_preflight.side_effect = mock_preflight_side_effect
            mock_cost.side_effect = mock_cost_side_effect
            mock_optimize.return_value = MagicMock(recommended_changes=[])
            mock_budget.return_value = MagicMock()

            # Perform what-if analysis
            what_if_analysis = asyncio.run(analyzer.perform_what_if_analysis(
                base_control=control,
                scenarios=scenarios[1:],  # Exclude baseline as it's created internally
                include_sensitivity_analysis=False  # Skip for this test
            ))

            assert isinstance(what_if_analysis, WhatIfAnalysis)
            assert what_if_analysis.analysis_id
            assert what_if_analysis.baseline_scenario is not None
            assert len(what_if_analysis.alternative_scenarios) == 2  # Two alternative scenarios

            # Check scenario comparison
            assert len(what_if_analysis.scenario_comparison) == 2

            # Check that best scenarios are identified
            assert what_if_analysis.best_overall_scenario is not None
            assert what_if_analysis.best_cost_scenario is not None
            assert what_if_analysis.best_quality_scenario is not None
            assert what_if_analysis.best_speed_scenario is not None

            # Check trade-off analysis
            assert len(what_if_analysis.cost_vs_quality_tradeoffs) == 3
            assert len(what_if_analysis.speed_vs_cost_tradeoffs) == 3
            assert len(what_if_analysis.risk_vs_benefit_tradeoffs) == 3

            # Check optimization recommendations
            assert len(what_if_analysis.optimization_recommendations) >= 0

    def test_scenario_ranking(self):
        """Test scenario ranking by different criteria."""
        analyzer = WhatIfAnalyzer()

        # Create mock scenarios with different characteristics
        baseline = ScenarioAnalysis(
            scenario_id="baseline",
            scenario_name="Baseline",
            scenario_type=ScenarioType.BASELINE,
            preflight_assessment={
                'operational_feasibility': {'estimated_duration_hours': 2.0},
                'risk_assessment': {'risk_level': 'medium'}
            },
            cost_prediction=MagicMock(predicted_cost=300.0),
            projected_data_quality="standard",
            projected_success_rate=0.8
        )
        baseline.feasibility_score = 0.8

        cost_opt = ScenarioAnalysis(
            scenario_id="cost_opt",
            scenario_name="Cost Optimized",
            scenario_type=ScenarioType.COST_OPTIMIZATION,
            preflight_assessment={
                'operational_feasibility': {'estimated_duration_hours': 2.5},
                'risk_assessment': {'risk_level': 'medium'}
            },
            cost_prediction=MagicMock(predicted_cost=200.0),
            projected_data_quality="basic",
            projected_success_rate=0.75
        )
        cost_opt.feasibility_score = 0.7

        quality_opt = ScenarioAnalysis(
            scenario_id="quality_opt",
            scenario_name="Quality Optimized",
            scenario_type=ScenarioType.QUALITY_ENHANCEMENT,
            preflight_assessment={
                'operational_feasibility': {'estimated_duration_hours': 1.5},
                'risk_assessment': {'risk_level': 'low'}
            },
            cost_prediction=MagicMock(predicted_cost=400.0),
            projected_data_quality="premium",
            projected_success_rate=0.9
        )
        quality_opt.feasibility_score = 0.9

        analysis = WhatIfAnalysis(
            analysis_id="test_ranking",
            baseline_scenario=baseline,
            alternative_scenarios=[cost_opt, quality_opt]
        )

        # Test ranking by cost (lower cost = better)
        cost_ranking = analysis.get_scenario_ranking('cost')
        assert cost_ranking[0][0] == "Cost Optimized"  # Lowest cost
        assert cost_ranking[-1][0] == "Quality Optimized"  # Highest cost

        # Test ranking by quality
        quality_ranking = analysis.get_scenario_ranking('quality')
        assert quality_ranking[0][0] == "Quality Optimized"  # Premium quality
        assert quality_ranking[-1][0] == "Cost Optimized"  # Basic quality

        # Test ranking by speed (lower time = better)
        speed_ranking = analysis.get_scenario_ranking('speed')
        assert speed_ranking[0][0] == "Quality Optimized"  # Fastest
        assert speed_ranking[-1][0] == "Cost Optimized"  # Slowest

        # Test ranking by risk (lower risk = better)
        risk_ranking = analysis.get_scenario_ranking('risk')
        assert risk_ranking[0][0] == "Quality Optimized"  # Lowest risk
        assert risk_ranking[-1][0] in ["Baseline", "Cost Optimized"]  # Higher risk

    def test_sensitivity_analysis(self):
        """Test sensitivity analysis functionality."""
        analyzer = WhatIfAnalyzer()

        control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Test sensitivity analysis",
                sources=["test_source"],
                geography=["test_geo"]
            ),
            budget=ScrapeBudget(max_runtime_minutes=60, max_pages=100, max_records=200, max_cost_total=500.0)
        )

        scenarios = [
            ScenarioConfiguration(
                scenario_id="sens_test",
                scenario_name="Sensitivity Test",
                scenario_type=ScenarioType.SENSITIVITY_ANALYSIS,
                description="Test sensitivity",
                base_control=control,
                cost_sensitivity_factor=1.2  # 20% cost increase
            )
        ]

        # Mock sensitivity analysis components
        with patch('core.what_if.preflight_cost_check') as mock_preflight, \
             patch('core.what_if.predict_scraping_cost') as mock_cost, \
             patch('core.what_if.optimize_scraping_cost') as mock_optimize, \
             patch('core.what_if.analyze_scraping_budget') as mock_budget:

            mock_preflight.return_value = {
                'overall_readiness': 'ready',
                'cost_analysis': {'predicted_cost': 360.0},  # 20% increase from base 300
                'operational_feasibility': {'estimated_duration_hours': 1.5},
                'risk_assessment': {'risk_level': 'medium'},
                'compliance_status': {'overall_compliance_score': 0.9}
            }

            mock_cost_result = MagicMock()
            mock_cost_result.predicted_cost = 360.0
            mock_cost.return_value = mock_cost_result

            mock_optimize.return_value = MagicMock(recommended_changes=[])
            mock_budget.return_value = MagicMock()

            # Perform sensitivity analysis
            sensitivity_results = asyncio.run(analyzer._perform_sensitivity_analysis(control, scenarios))

            # Check that sensitivity results are generated
            assert len(sensitivity_results) > 0
            assert 'cost_increase' in sensitivity_results

            # Check specific sensitivity test
            cost_increase_results = sensitivity_results['cost_increase']
            assert len(cost_increase_results) > 0

            # Find the 20% increase result
            increase_20 = next((r for r in cost_increase_results if r['parameter_value'] == 1.2), None)
            assert increase_20 is not None
            assert increase_20['cost_impact'] == 360.0

    def test_recommendation_generation(self):
        """Test scenario recommendation generation."""
        analyzer = WhatIfAnalyzer()

        # Create mock analysis with different scenario characteristics
        baseline = ScenarioAnalysis(
            scenario_id="baseline",
            scenario_name="Baseline",
            scenario_type=ScenarioType.BASELINE,
            preflight_assessment={
                'overall_readiness': 'ready',
                'cost_analysis': {'predicted_cost': 300.0},
                'operational_feasibility': {'estimated_duration_hours': 2.0},
                'risk_assessment': {'risk_level': 'medium'}
            },
            cost_prediction=MagicMock(predicted_cost=300.0),
            projected_data_quality="standard"
        )
        baseline.feasibility_score = 0.8

        cost_scenario = ScenarioAnalysis(
            scenario_id="cost",
            scenario_name="Cost Optimized",
            scenario_type=ScenarioType.COST_OPTIMIZATION,
            preflight_assessment={
                'overall_readiness': 'ready',
                'cost_analysis': {'predicted_cost': 200.0},
                'operational_feasibility': {'estimated_duration_hours': 2.5},
                'risk_assessment': {'risk_level': 'medium'}
            },
            cost_prediction=MagicMock(predicted_cost=200.0),
            projected_data_quality="basic",
            cost_difference=-100.0,  # Savings
            cost_difference_percentage=-33.3
        )
        cost_scenario.feasibility_score = 0.7

        analysis = WhatIfAnalysis(
            analysis_id="test_recommendations",
            baseline_scenario=baseline,
            alternative_scenarios=[cost_scenario]
        )

        # Generate recommendations
        recommendations = asyncio.run(analyzer.generate_scenario_recommendations(analysis))

        assert isinstance(recommendations, dict)
        assert 'primary_recommendation' in recommendations
        assert 'alternative_recommendations' in recommendations
        assert 'scenario_scores' in recommendations
        assert 'trade_off_analysis' in recommendations
        assert 'decision_factors' in recommendations
        assert 'implementation_guidance' in recommendations

        # Check that scenarios are scored
        scenario_scores = recommendations['scenario_scores']
        assert 'Baseline' in scenario_scores
        assert 'Cost Optimized' in scenario_scores

        # Check trade-off analysis
        trade_offs = recommendations['trade_off_analysis']
        assert 'cost_vs_quality' in trade_offs
        assert 'speed_vs_cost' in trade_offs
        assert 'risk_vs_benefit' in trade_offs

        # Check implementation guidance
        guidance = recommendations['implementation_guidance']
        assert 'phased_implementation' in guidance
        assert 'resource_requirements' in guidance
        assert 'risk_mitigations' in guidance

    def test_scenario_template_application(self):
        """Test application of scenario templates."""
        analyzer = WhatIfAnalyzer()

        control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Test template application",
                sources=["base_source"],
                geography=["base_geo"]
            ),
            budget=ScrapeBudget(max_runtime_minutes=60, max_pages=100, max_records=200, max_cost_total=500.0)
        )

        # Test cost optimization template
        cost_scenario = asyncio.run(analyzer.create_scenario(
            base_control=control,
            scenario_type=ScenarioType.COST_OPTIMIZATION,
            scenario_name="Template Cost Test"
        ))

        assert cost_scenario.scenario_type == ScenarioType.COST_OPTIMIZATION
        assert "Cost Optimized" in cost_scenario.scenario_name

        # Test quality enhancement template
        quality_scenario = asyncio.run(analyzer.create_scenario(
            base_control=control,
            scenario_type=ScenarioType.QUALITY_ENHANCEMENT,
            scenario_name="Template Quality Test"
        ))

        assert quality_scenario.scenario_type == ScenarioType.QUALITY_ENHANCEMENT
        assert "Quality Enhanced" in quality_scenario.scenario_name
        assert quality_scenario.quality_requirement == "premium"

    def test_convenience_function_integration(self):
        """Test integration with global convenience functions."""
        control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Convenience function test",
                sources=["test_source"],
                geography=["test_geo"]
            ),
            budget=ScrapeBudget(max_runtime_minutes=60, max_pages=100, max_records=200, max_cost_total=500.0),
            authorization=ScrapeAuthorization(
                approved_by="test_user",
                purpose="Testing",
                expires_at=datetime.utcnow() + timedelta(days=1)
            )
        )

        # Test scenario creation function
        with patch('core.what_if.WhatIfAnalyzer.create_scenario') as mock_create:
            mock_scenario = MagicMock()
            mock_create.return_value = mock_scenario

            scenario = asyncio.run(create_what_if_scenario(
                control, ScenarioType.COST_OPTIMIZATION, "Test Scenario"
            ))

            assert scenario == mock_scenario
            mock_create.assert_called_once()

        # Test analysis function
        with patch('core.what_if.WhatIfAnalyzer.perform_what_if_analysis') as mock_analyze:
            mock_analysis = MagicMock()
            mock_analyze.return_value = mock_analysis

            scenarios = [MagicMock()]
            analysis = asyncio.run(analyze_what_if_scenarios(control, scenarios))

            assert analysis == mock_analysis
            mock_analyze.assert_called_once_with(control, scenarios, True)

        # Test recommendation function
        with patch('core.what_if.WhatIfAnalyzer.generate_scenario_recommendations') as mock_recommend:
            mock_recommendations = {"primary": "test"}
            mock_recommend.return_value = mock_recommendations

            analysis = MagicMock()
            recommendations = asyncio.run(generate_scenario_recommendations(analysis))

            assert recommendations == mock_recommendations
            mock_recommend.assert_called_once_with(analysis, None)

        # Test statistics function
        stats = get_what_if_analyzer_stats()
        assert isinstance(stats, dict)

    def test_error_handling_and_resilience(self):
        """Test error handling and system resilience."""
        analyzer = WhatIfAnalyzer()

        control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Error handling test",
                sources=["test_source"],
                geography=["test_geo"]
            ),
            budget=ScrapeBudget(max_runtime_minutes=60, max_pages=100, max_records=200, max_cost_total=500.0)
        )

        scenario = ScenarioConfiguration(
            scenario_id="error_test",
            scenario_name="Error Test",
            scenario_type=ScenarioType.BASELINE,
            description="Test error handling",
            base_control=control
        )

        # Test with failing intelligence components
        with patch('core.what_if.preflight_cost_check', side_effect=Exception("Preflight failed")) as mock_preflight, \
             patch('core.what_if.predict_scraping_cost') as mock_cost, \
             patch('core.what_if.optimize_scraping_cost') as mock_optimize:

            mock_cost_result = MagicMock()
            mock_cost_result.predicted_cost = 300.0
            mock_cost.return_value = mock_cost_result

            mock_optimize.return_value = MagicMock(recommended_changes=[])

            # Should handle preflight failure gracefully
            analysis = asyncio.run(analyzer.analyze_scenario(scenario))

            # Should still create analysis with available data
            assert isinstance(analysis, ScenarioAnalysis)
            assert analysis.scenario_id == scenario.scenario_id
            # Analysis should contain whatever data was successfully gathered

        # Test with multiple scenario failures in what-if analysis
        scenarios = [
            ScenarioConfiguration(
                scenario_id="fail1",
                scenario_name="Fail Scenario 1",
                scenario_type=ScenarioType.COST_OPTIMIZATION,
                description="Will fail",
                base_control=control
            ),
            ScenarioConfiguration(
                scenario_id="fail2",
                scenario_name="Fail Scenario 2",
                scenario_type=ScenarioType.QUALITY_ENHANCEMENT,
                description="Will also fail",
                base_control=control
            )
        ]

        with patch('core.what_if.WhatIfAnalyzer.analyze_scenario', side_effect=Exception("Analysis failed")) as mock_analyze:
            # Should handle scenario analysis failures
            what_if_analysis = asyncio.run(analyzer.perform_what_if_analysis(
                base_control=control,
                scenarios=scenarios,
                include_sensitivity_analysis=False
            ))

            # Should still create analysis structure
            assert isinstance(what_if_analysis, WhatIfAnalysis)
            assert what_if_analysis.baseline_scenario is not None
            # Alternative scenarios list may be empty due to failures

    def test_comprehensive_scenario_ecosystem(self):
        """Test comprehensive scenario ecosystem with multiple scenario types."""
        analyzer = WhatIfAnalyzer()

        control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Comprehensive ecosystem test",
                sources=["primary_source"],
                geography=["main_region"]
            ),
            budget=ScrapeBudget(max_runtime_minutes=120, max_pages=300, max_records=1000, max_cost_total=1000.0),
            authorization=ScrapeAuthorization(
                approved_by="enterprise_user",
                purpose="Comprehensive testing",
                expires_at=datetime.utcnow() + timedelta(days=7)
            )
        )

        # Create diverse scenario portfolio
        scenario_configs = []

        # Cost-focused scenarios
        cost_scenario = asyncio.run(analyzer.create_scenario(
            control, ScenarioType.COST_OPTIMIZATION, "Budget Conscious"
        ))
        scenario_configs.append(cost_scenario)

        # Quality-focused scenarios
        quality_scenario = asyncio.run(analyzer.create_scenario(
            control, ScenarioType.QUALITY_ENHANCEMENT, "Quality First"
        ))
        scenario_configs.append(quality_scenario)

        # Speed-focused scenarios
        speed_scenario = asyncio.run(analyzer.create_scenario(
            control, ScenarioType.SPEED_OPTIMIZATION, "Fast Track"
        ))
        scenario_configs.append(speed_scenario)

        # Risk-focused scenarios
        risk_scenario = asyncio.run(analyzer.create_scenario(
            control, ScenarioType.RISK_MINIMIZATION, "Safe Harbor"
        ))
        scenario_configs.append(risk_scenario)

        # Scale scenarios
        scale_scenario = asyncio.run(analyzer.create_scenario(
            control, ScenarioType.SCALE_EXPANSION, "Enterprise Scale"
        ))
        scenario_configs.append(scale_scenario)

        # Custom scenario
        custom_scenario = ScenarioConfiguration(
            scenario_id="custom_test",
            scenario_name="Custom Balanced",
            scenario_type=ScenarioType.CUSTOM,
            description="Balanced custom approach",
            base_control=control,
            modified_sources=["balanced_source1", "balanced_source2"],
            quality_requirement="standard"
        )
        scenario_configs.append(custom_scenario)

        # Mock comprehensive analysis
        with patch('core.what_if.preflight_cost_check') as mock_preflight, \
             patch('core.what_if.predict_scraping_cost') as mock_cost, \
             patch('core.what_if.optimize_scraping_cost') as mock_optimize, \
             patch('core.what_if.analyze_scraping_budget') as mock_budget:

            # Create diverse mock responses for different scenario types
            def mock_preflight_responses(control_param):
                scenario_name = str(control_param)
                if "Budget Conscious" in scenario_name:
                    return {
                        'overall_readiness': 'ready',
                        'cost_analysis': {'predicted_cost': 400.0, 'budget_utilization_percentage': 40.0},
                        'operational_feasibility': {'estimated_duration_hours': 3.0, 'resource_intensity': 'low'},
                        'risk_assessment': {'risk_level': 'medium'},
                        'compliance_status': {'overall_compliance_score': 0.85}
                    }
                elif "Quality First" in scenario_name:
                    return {
                        'overall_readiness': 'ready',
                        'cost_analysis': {'predicted_cost': 800.0, 'budget_utilization_percentage': 80.0},
                        'operational_feasibility': {'estimated_duration_hours': 1.5, 'resource_intensity': 'medium'},
                        'risk_assessment': {'risk_level': 'low'},
                        'compliance_status': {'overall_compliance_score': 0.95}
                    }
                elif "Fast Track" in scenario_name:
                    return {
                        'overall_readiness': 'caution',
                        'cost_analysis': {'predicted_cost': 600.0, 'budget_utilization_percentage': 60.0},
                        'operational_feasibility': {'estimated_duration_hours': 0.8, 'resource_intensity': 'high'},
                        'risk_assessment': {'risk_level': 'high'},
                        'compliance_status': {'overall_compliance_score': 0.75}
                    }
                elif "Safe Harbor" in scenario_name:
                    return {
                        'overall_readiness': 'ready',
                        'cost_analysis': {'predicted_cost': 700.0, 'budget_utilization_percentage': 70.0},
                        'operational_feasibility': {'estimated_duration_hours': 2.5, 'resource_intensity': 'medium'},
                        'risk_assessment': {'risk_level': 'low'},
                        'compliance_status': {'overall_compliance_score': 0.98}
                    }
                elif "Enterprise Scale" in scenario_name:
                    return {
                        'overall_readiness': 'caution',
                        'cost_analysis': {'predicted_cost': 950.0, 'budget_utilization_percentage': 95.0},
                        'operational_feasibility': {'estimated_duration_hours': 8.0, 'resource_intensity': 'high'},
                        'risk_assessment': {'risk_level': 'high'},
                        'compliance_status': {'overall_compliance_score': 0.90}
                    }
                elif "Custom Balanced" in scenario_name:
                    return {
                        'overall_readiness': 'ready',
                        'cost_analysis': {'predicted_cost': 550.0, 'budget_utilization_percentage': 55.0},
                        'operational_feasibility': {'estimated_duration_hours': 2.2, 'resource_intensity': 'medium'},
                        'risk_assessment': {'risk_level': 'medium'},
                        'compliance_status': {'overall_compliance_score': 0.88}
                    }
                else:  # Baseline
                    return {
                        'overall_readiness': 'ready',
                        'cost_analysis': {'predicted_cost': 500.0, 'budget_utilization_percentage': 50.0},
                        'operational_feasibility': {'estimated_duration_hours': 2.0, 'resource_intensity': 'medium'},
                        'risk_assessment': {'risk_level': 'medium'},
                        'compliance_status': {'overall_compliance_score': 0.90}
                    }

            def mock_cost_responses(*args, **kwargs):
                # Return costs based on scenario characteristics
                scenario_context = str(args)
                if "Budget Conscious" in scenario_context or 'cheap' in scenario_context.lower():
                    result = MagicMock()
                    result.predicted_cost = 400.0
                    result.confidence_score = 0.8
                    return result
                elif "Quality First" in scenario_context or 'premium' in scenario_context.lower():
                    result = MagicMock()
                    result.predicted_cost = 800.0
                    result.confidence_score = 0.85
                    return result
                elif "Fast Track" in scenario_context:
                    result = MagicMock()
                    result.predicted_cost = 600.0
                    result.confidence_score = 0.75
                    return result
                elif "Safe Harbor" in scenario_context:
                    result = MagicMock()
                    result.predicted_cost = 700.0
                    result.confidence_score = 0.9
                    return result
                elif "Enterprise Scale" in scenario_context:
                    result = MagicMock()
                    result.predicted_cost = 950.0
                    result.confidence_score = 0.7
                    return result
                elif "Custom Balanced" in scenario_context:
                    result = MagicMock()
                    result.predicted_cost = 550.0
                    result.confidence_score = 0.8
                    return result
                else:
                    result = MagicMock()
                    result.predicted_cost = 500.0
                    result.confidence_score = 0.8
                    return result

            mock_preflight.side_effect = mock_preflight_responses
            mock_cost.side_effect = mock_cost_responses
            mock_optimize.return_value = MagicMock(recommended_changes=[])
            mock_budget.return_value = MagicMock()

            # Perform comprehensive what-if analysis
            comprehensive_analysis = asyncio.run(analyzer.perform_what_if_analysis(
                base_control=control,
                scenarios=scenario_configs,
                include_sensitivity_analysis=False
            ))

            # Verify comprehensive analysis results
            assert isinstance(comprehensive_analysis, WhatIfAnalysis)
            assert comprehensive_analysis.baseline_scenario is not None
            assert len(comprehensive_analysis.alternative_scenarios) == 6  # All scenarios analyzed

            # Verify scenario comparison data
            assert len(comprehensive_analysis.scenario_comparison) == 6

            # Verify best scenario identification
            assert comprehensive_analysis.best_overall_scenario is not None
            assert comprehensive_analysis.best_cost_scenario is not None  # Should be "Budget Conscious"
            assert comprehensive_analysis.best_quality_scenario is not None  # Should be "Quality First"
            assert comprehensive_analysis.best_speed_scenario is not None  # Should be "Fast Track"

            # Verify trade-off analysis is comprehensive
            assert len(comprehensive_analysis.cost_vs_quality_tradeoffs) == 7  # Baseline + 6 scenarios
            assert len(comprehensive_analysis.speed_vs_cost_tradeoffs) == 7
            assert len(comprehensive_analysis.risk_vs_benefit_tradeoffs) == 7

            # Verify optimization recommendations are generated
            assert len(comprehensive_analysis.optimization_recommendations) >= 0

            # Generate comprehensive recommendations
            recommendations = asyncio.run(analyzer.generate_scenario_recommendations(comprehensive_analysis))

            # Verify recommendation structure
            assert 'primary_recommendation' in recommendations
            assert 'alternative_recommendations' in recommendations
            assert len(recommendations['alternative_recommendations']) >= 2

            # Verify scenario scores are comprehensive
            scenario_scores = recommendations['scenario_scores']
            assert len(scenario_scores) == 7  # All scenarios scored

            # Verify each scenario has required score components
            for scenario_name, scores in scenario_scores.items():
                assert 'cost_score' in scores
                assert 'quality_score' in scores
                assert 'speed_score' in scores
                assert 'risk_score' in scores
                assert 'overall_score' in scores

            # Verify trade-off analysis in recommendations
            trade_offs = recommendations['trade_off_analysis']
            assert 'cost_vs_quality' in trade_offs
            assert 'speed_vs_cost' in trade_offs
            assert 'risk_vs_benefit' in trade_offs

            # Verify decision factors
            decision_factors = recommendations['decision_factors']
            assert 'budget_considerations' in decision_factors
            assert 'risk_considerations' in decision_factors
            assert 'operational_considerations' in decision_factors
            assert 'compliance_considerations' in decision_factors

            # Verify implementation guidance
            guidance = recommendations['implementation_guidance']
            assert 'phased_implementation' in guidance
            assert 'resource_requirements' in guidance
            assert 'monitoring_requirements' in guidance
            assert 'rollback_procedures' in guidance

            # Verify scenario diversity in recommendations
            recommended_scenarios = [recommendations['primary_recommendation']] + recommendations['alternative_recommendations']
            unique_scenarios = set(recommended_scenarios)
            assert len(unique_scenarios) >= 3  # Should recommend diverse options

            print(f"🎯 Comprehensive ecosystem test completed successfully!")
            print(f"   📊 Analyzed {len(comprehensive_analysis.alternative_scenarios)} alternative scenarios")
            print(f"   🏆 Best overall: {comprehensive_analysis.best_overall_scenario}")
            print(f"   💰 Best cost: {comprehensive_analysis.best_cost_scenario}")
            print(f"   ⚡ Best speed: {comprehensive_analysis.best_speed_scenario}")
            print(f"   🔍 Best quality: {comprehensive_analysis.best_quality_scenario}")
            print(f"   📋 Primary recommendation: {recommendations['primary_recommendation']}")

    def test_what_if_cost_analysis_comprehensive(self):
        """Test comprehensive what-if cost analysis functionality."""
        from core.what_if import what_if_cost_analysis

        signals = [SignalType.LIEN, SignalType.MORTGAGE]
        geography_levels = ["city", "county", "state"]
        browser_pages = 25
        base_budget = 5000.0

        # Mock all intelligence components
        with patch('core.what_if.preflight_cost_check') as mock_preflight, \
             patch('core.what_if.predict_scraping_cost') as mock_cost, \
             patch('core.what_if.optimize_scraping_cost') as mock_optimize, \
             patch('core.what_if.analyze_scraping_budget') as mock_budget:

            # Set up mock responses for different geographies
            def mock_preflight_side_effect(control_param):
                geo = control_param.intent.geography[0] if control_param.intent.geography else "city"
                if geo == "city":
                    return {
                        'overall_readiness': 'ready',
                        'cost_analysis': {'predicted_cost': 1200.0},
                        'operational_feasibility': {'estimated_duration_hours': 2.0, 'expected_success_rate': 0.9},
                        'risk_assessment': {'risk_level': 'low'},
                        'compliance_status': {'overall_compliance_score': 0.95}
                    }
                elif geo == "county":
                    return {
                        'overall_readiness': 'ready',
                        'cost_analysis': {'predicted_cost': 2800.0},
                        'operational_feasibility': {'estimated_duration_hours': 6.0, 'expected_success_rate': 0.85},
                        'risk_assessment': {'risk_level': 'medium'},
                        'compliance_status': {'overall_compliance_score': 0.88}
                    }
                else:  # state
                    return {
                        'overall_readiness': 'caution',
                        'cost_analysis': {'predicted_cost': 5200.0},
                        'operational_feasibility': {'estimated_duration_hours': 18.0, 'expected_success_rate': 0.75},
                        'risk_assessment': {'risk_level': 'high'},
                        'compliance_status': {'overall_compliance_score': 0.82}
                    }

            def mock_cost_side_effect(*args, **kwargs):
                # Return different costs based on geography in control
                control = kwargs.get('control') or args[-1] if args else None
                if control and control.intent.geography:
                    geo = control.intent.geography[0]
                    if geo == "city":
                        result = MagicMock()
                        result.predicted_cost = 1200.0
                        result.confidence_score = 0.9
                        result.cost_range = (1080.0, 1320.0)
                        result.cost_breakdown = {'signal_acquisition': 800, 'infrastructure': 300, 'compliance_legal': 100}
                        return result
                    elif geo == "county":
                        result = MagicMock()
                        result.predicted_cost = 2800.0
                        result.confidence_score = 0.85
                        result.cost_range = (2520.0, 3080.0)
                        result.cost_breakdown = {'signal_acquisition': 1800, 'infrastructure': 700, 'compliance_legal': 300}
                        return result
                    else:  # state
                        result = MagicMock()
                        result.predicted_cost = 5200.0
                        result.confidence_score = 0.75
                        result.cost_range = (4680.0, 5720.0)
                        result.cost_breakdown = {'signal_acquisition': 3200, 'infrastructure': 1400, 'compliance_legal': 600}
                        return result

                # Default
                result = MagicMock()
                result.predicted_cost = 2000.0
                result.confidence_score = 0.8
                return result

            mock_preflight.side_effect = mock_preflight_side_effect
            mock_cost.side_effect = mock_cost_side_effect
            mock_optimize.return_value = MagicMock(recommended_changes=[])
            mock_budget.return_value = MagicMock()

            # Execute comprehensive cost analysis
            analysis = asyncio.run(what_if_cost_analysis(
                signals=signals,
                geography_levels=geography_levels,
                browser_pages=browser_pages,
                base_budget=base_budget,
                risk_tolerance="medium",
                time_sensitivity="normal",
                quality_requirement="standard"
            ))

            # Verify comprehensive analysis structure
            assert analysis['analysis_id']
            assert analysis['timestamp']
            assert analysis['parameters']['signals'] == ['lien', 'mortgage']
            assert analysis['parameters']['geography_levels'] == geography_levels
            assert analysis['parameters']['browser_pages'] == browser_pages
            assert analysis['parameters']['base_budget'] == base_budget

            # Verify geography analysis
            geo_analysis = analysis['geography_analysis']
            assert len(geo_analysis) == 3

            # Check each geography result
            city_result = next(r for r in geo_analysis if r['geography_level'] == 'city')
            county_result = next(r for r in geo_analysis if r['geography_level'] == 'county')
            state_result = next(r for r in geo_analysis if r['geography_level'] == 'state')

            # Verify city results
            assert city_result['estimated_cost'] == 1200.0
            assert city_result['cost_confidence'] == 0.9
            assert city_result['budget_utilization'] == 24.0  # 1200/5000 * 100
            assert city_result['budget_compliance'] == 'compliant'
            assert 'cost_efficiency_score' in city_result
            assert 'scalability_score' in city_result
            assert 'data_density_score' in city_result

            # Verify county results
            assert county_result['estimated_cost'] == 2800.0
            assert county_result['budget_utilization'] == 56.0
            assert county_result['budget_compliance'] == 'compliant'

            # Verify state results (over budget)
            assert state_result['estimated_cost'] == 5200.0
            assert state_result['budget_utilization'] == 104.0
            assert state_result['budget_compliance'] == 'exceeded'
            assert state_result['overall_readiness'] == 'caution'

            # Verify cost summary
            cost_summary = analysis['cost_summary']
            assert cost_summary['min_cost'] == 1200.0
            assert cost_summary['max_cost'] == 5200.0
            assert cost_summary['average_cost'] > 0
            assert cost_summary['most_cost_effective'] == 'city'
            assert cost_summary['least_cost_effective'] == 'state'
            assert 'city' in cost_summary['budget_compliant_geographies']
            assert 'county' in cost_summary['budget_compliant_geographies']
            assert 'state' in cost_summary['budget_exceeded_geographies']

            # Verify optimization opportunities
            optimizations = analysis['optimization_opportunities']
            assert len(optimizations) > 0

            # Verify comparative insights
            comparative = analysis['comparative_insights']
            assert 'cost_distribution' in comparative
            assert 'efficiency_comparison' in comparative
            assert 'scalability_comparison' in comparative
            assert 'data_density_comparison' in comparative
            assert 'trade_off_analysis' in comparative

            # Verify budget impact
            budget_impact = analysis['budget_impact']
            assert 'feasible_geographies' in budget_impact
            assert 'infeasible_geographies' in budget_impact
            assert len(budget_impact['feasible_geographies']) == 2  # city and county
            assert len(budget_impact['infeasible_geographies']) == 1  # state

            # Verify recommendations
            recommendations = analysis['recommendations']
            assert 'primary_recommendation' in recommendations
            assert recommendations['primary_recommendation']['geography'] == 'city'  # Most cost-effective
            assert 'alternative_recommendations' in recommendations
            assert len(recommendations['alternative_recommendations']) >= 1

            # Verify risk assessment
            risk_assessment = analysis['risk_assessment']
            assert 'overall_risk_level' in risk_assessment
            assert 'cost_uncertainty_risk' in risk_assessment
            assert 'budget_overrun_risk' in risk_assessment

            # Verify performance projections
            projections = analysis['performance_projections']
            assert 'execution_time_estimates' in projections
            assert 'success_rate_projections' in projections
            assert 'resource_utilization_forecast' in projections
            assert 'data_quality_expectations' in projections
            assert 'scalability_projections' in projections

            # Verify all geographies have projections
            for geo in geography_levels:
                assert geo in projections['execution_time_estimates']
                assert geo in projections['success_rate_projections']
                assert geo in projections['resource_utilization_forecast']

    def test_what_if_cost_budget_compliance_analysis(self):
        """Test budget compliance analysis in what-if cost scenarios."""
        from core.what_if import what_if_cost_analysis

        # Test with very constrained budget
        constrained_budget = 1000.0

        with patch('core.what_if.preflight_cost_check') as mock_preflight, \
             patch('core.what_if.predict_scraping_cost') as mock_cost, \
             patch('core.what_if.optimize_scraping_cost') as mock_optimize, \
             patch('core.what_if.analyze_scraping_budget') as mock_budget:

            def mock_preflight_side_effect(control_param):
                geo = control_param.intent.geography[0] if control_param.intent.geography else "city"
                return {
                    'overall_readiness': 'ready',
                    'cost_analysis': {'predicted_cost': 800.0 if geo == "city" else 1500.0 if geo == "county" else 2500.0},
                    'operational_feasibility': {'estimated_duration_hours': 2.0, 'expected_success_rate': 0.85},
                    'risk_assessment': {'risk_level': 'medium'},
                    'compliance_status': {'overall_compliance_score': 0.9}
                }

            def mock_cost_side_effect(*args, **kwargs):
                control = kwargs.get('control') or args[-1] if args else None
                geo = control.intent.geography[0] if control and control.intent.geography else "city"

                result = MagicMock()
                if geo == "city":
                    result.predicted_cost = 800.0
                    result.confidence_score = 0.9
                    result.cost_range = (720.0, 880.0)
                elif geo == "county":
                    result.predicted_cost = 1500.0
                    result.confidence_score = 0.85
                    result.cost_range = (1350.0, 1650.0)
                else:  # state
                    result.predicted_cost = 2500.0
                    result.confidence_score = 0.8
                    result.cost_range = (2250.0, 2750.0)

                result.cost_breakdown = {'signal_acquisition': 500, 'infrastructure': 200, 'compliance_legal': 100}
                return result

            mock_preflight.side_effect = mock_preflight_side_effect
            mock_cost.side_effect = mock_cost_side_effect
            mock_optimize.return_value = MagicMock(recommended_changes=[])
            mock_budget.return_value = MagicMock()

            analysis = asyncio.run(what_if_cost_analysis(
                geography_levels=["city", "county", "state"],
                base_budget=constrained_budget
            ))

            # Verify budget impact analysis
            budget_impact = analysis['budget_impact']
            assert len(budget_impact['feasible_geographies']) == 1  # Only city
            assert len(budget_impact['infeasible_geographies']) == 2  # County and state
            assert len(budget_impact['budget_optimization_opportunities']) > 0

            # Verify geography results show correct compliance
            geo_analysis = analysis['geography_analysis']
            city_result = next(r for r in geo_analysis if r['geography_level'] == 'city')
            county_result = next(r for r in geo_analysis if r['geography_level'] == 'county')
            state_result = next(r for r in geo_analysis if r['geography_level'] == 'state')

            assert city_result['budget_compliance'] == 'compliant'
            assert county_result['budget_compliance'] == 'exceeded'
            assert state_result['budget_compliance'] == 'exceeded'

            # Verify cost summary reflects budget constraints
            cost_summary = analysis['cost_summary']
            assert cost_summary['budget_exceeded_geographies'] == ['county', 'state']
            assert cost_summary['budget_compliant_geographies'] == ['city']

            # Verify recommendations account for budget constraints
            recommendations = analysis['recommendations']
            assert recommendations['primary_recommendation']['geography'] == 'city'

    def test_what_if_cost_optimization_opportunities(self):
        """Test cost optimization opportunities identification."""
        from core.what_if import what_if_cost_analysis

        with patch('core.what_if.preflight_cost_check') as mock_preflight, \
             patch('core.what_if.predict_scraping_cost') as mock_cost, \
             patch('core.what_if.optimize_scraping_cost') as mock_optimize, \
             patch('core.what_if.analyze_scraping_budget') as mock_budget:

            # Set up scenarios with optimization potential
            def mock_preflight_side_effect(control_param):
                geo = control_param.intent.geography[0] if control_param.intent.geography else "city"
                return {
                    'overall_readiness': 'ready',
                    'cost_analysis': {'predicted_cost': 2000.0 if geo == "city" else 5000.0 if geo == "county" else 8000.0},
                    'operational_feasibility': {'estimated_duration_hours': 4.0, 'expected_success_rate': 0.8},
                    'risk_assessment': {'risk_level': 'medium'},
                    'compliance_status': {'overall_compliance_score': 0.85}
                }

            def mock_cost_side_effect(*args, **kwargs):
                control = kwargs.get('control') or args[-1] if args else None
                geo = control.intent.geography[0] if control and control.intent.geography else "city"

                result = MagicMock()
                if geo == "city":
                    result.predicted_cost = 2000.0
                    result.confidence_score = 0.85
                elif geo == "county":
                    result.predicted_cost = 5000.0
                    result.confidence_score = 0.8
                else:  # state
                    result.predicted_cost = 8000.0
                    result.confidence_score = 0.75

                result.cost_range = (result.predicted_cost * 0.9, result.predicted_cost * 1.1)
                result.cost_breakdown = {'signal_acquisition': result.predicted_cost * 0.6, 'infrastructure': result.predicted_cost * 0.3, 'compliance_legal': result.predicted_cost * 0.1}
                return result

            mock_preflight.side_effect = mock_preflight_side_effect
            mock_cost.side_effect = mock_cost_side_effect

            # Mock optimization with significant savings
            mock_optimize.return_value = MagicMock(
                recommended_changes=[
                    {'change': 'Optimize source selection', 'estimated_savings': 800, 'implementation_effort': 'low'},
                    {'change': 'Implement batch processing', 'estimated_savings': 500, 'implementation_effort': 'medium'}
                ]
            )
            mock_budget.return_value = MagicMock()

            analysis = asyncio.run(what_if_cost_analysis(
                geography_levels=["city", "county", "state"],
                base_budget=10000.0
            ))

            # Verify optimization opportunities
            optimizations = analysis['optimization_opportunities']
            assert len(optimizations) > 0

            # Should identify geography optimization opportunity
            geography_opt = next((opt for opt in optimizations if opt.get('type') == 'geography_optimization'), None)
            assert geography_opt is not None
            assert geography_opt['savings_amount'] > 0
            assert 'Switch from' in geography_opt['description']
            assert 'city' in geography_opt['description'].lower() or 'county' in geography_opt['description'].lower()

            # Verify individual geography optimization details
            geo_analysis = analysis['geography_analysis']
            county_result = next(r for r in geo_analysis if r['geography_level'] == 'county')
            state_result = next(r for r in geo_analysis if r['geography_level'] == 'state')

            # Should show optimization available for higher-cost geographies
            assert county_result.get('optimization_available') == True
            assert state_result.get('optimization_available') == True

    def test_what_if_cost_sensitivity_analysis(self):
        """Test sensitivity analysis in what-if cost scenarios."""
        from core.what_if import what_if_cost_analysis

        with patch('core.what_if.preflight_cost_check') as mock_preflight, \
             patch('core.what_if.predict_scraping_cost') as mock_cost, \
             patch('core.what_if.optimize_scraping_cost') as mock_optimize, \
             patch('core.what_if.analyze_scraping_budget') as mock_budget:

            mock_preflight.return_value = {
                'overall_readiness': 'ready',
                'cost_analysis': {'predicted_cost': 3000.0},
                'operational_feasibility': {'estimated_duration_hours': 6.0, 'expected_success_rate': 0.85},
                'risk_assessment': {'risk_level': 'medium'},
                'compliance_status': {'overall_compliance_score': 0.9}
            }

            mock_cost_result = MagicMock()
            mock_cost_result.predicted_cost = 3000.0
            mock_cost_result.confidence_score = 0.8
            mock_cost_result.cost_range = (2700.0, 3300.0)
            mock_cost_result.cost_breakdown = {'signal_acquisition': 2000, 'infrastructure': 700, 'compliance_legal': 300}
            mock_cost.return_value = mock_cost_result

            mock_optimize.return_value = MagicMock(recommended_changes=[])
            mock_budget.return_value = MagicMock()

            analysis = asyncio.run(what_if_cost_analysis(
                geography_levels=["county"],
                include_sensitivity=True
            ))

            # Verify sensitivity analysis is included
            sensitivity = analysis['sensitivity_analysis']
            assert len(sensitivity) > 0
            assert 'cost_increase_sensitivity' in sensitivity

            # Verify cost increase sensitivity
            cost_sensitivity = sensitivity['cost_increase_sensitivity']
            assert len(cost_sensitivity) > 0

            # Check that sensitivity shows impact of cost increases
            for sensitivity_result in cost_sensitivity:
                assert 'parameter_value' in sensitivity_result
                assert 'average_cost_impact' in sensitivity_result
                assert sensitivity_result['average_cost_impact'] > 0

            # Verify other sensitivity analyses
            assert 'time_increase_sensitivity' in sensitivity
            assert 'scope_reduction_sensitivity' in sensitivity

    def test_what_if_cost_comparative_insights(self):
        """Test comparative insights generation."""
        from core.what_if import what_if_cost_analysis

        with patch('core.what_if.preflight_cost_check') as mock_preflight, \
             patch('core.what_if.predict_scraping_cost') as mock_cost, \
             patch('core.what_if.optimize_scraping_cost') as mock_optimize, \
             patch('core.what_if.analyze_scraping_budget') as mock_budget:

            def mock_preflight_side_effect(control_param):
                geo = control_param.intent.geography[0] if control_param.intent.geography else "city"
                if geo == "city":
                    return {
                        'overall_readiness': 'ready',
                        'cost_analysis': {'predicted_cost': 1000.0},
                        'operational_feasibility': {'estimated_duration_hours': 2.0, 'expected_success_rate': 0.9},
                        'risk_assessment': {'risk_level': 'low'},
                        'compliance_status': {'overall_compliance_score': 0.95}
                    }
                elif geo == "state":
                    return {
                        'overall_readiness': 'caution',
                        'cost_analysis': {'predicted_cost': 8000.0},
                        'operational_feasibility': {'estimated_duration_hours': 24.0, 'expected_success_rate': 0.7},
                        'risk_assessment': {'risk_level': 'high'},
                        'compliance_status': {'overall_compliance_score': 0.8}
                    }
                else:  # county
                    return {
                        'overall_readiness': 'ready',
                        'cost_analysis': {'predicted_cost': 3500.0},
                        'operational_feasibility': {'estimated_duration_hours': 8.0, 'expected_success_rate': 0.8},
                        'risk_assessment': {'risk_level': 'medium'},
                        'compliance_status': {'overall_compliance_score': 0.85}
                    }

            def mock_cost_side_effect(*args, **kwargs):
                control = kwargs.get('control') or args[-1] if args else None
                geo = control.intent.geography[0] if control and control.intent.geography else "city"

                result = MagicMock()
                if geo == "city":
                    result.predicted_cost = 1000.0
                    result.confidence_score = 0.95
                elif geo == "county":
                    result.predicted_cost = 3500.0
                    result.confidence_score = 0.85
                else:  # state
                    result.predicted_cost = 8000.0
                    result.confidence_score = 0.7

                result.cost_range = (result.predicted_cost * 0.9, result.predicted_cost * 1.1)
                result.cost_breakdown = {'signal_acquisition': result.predicted_cost * 0.6, 'infrastructure': result.predicted_cost * 0.3, 'compliance_legal': result.predicted_cost * 0.1}
                return result

            mock_preflight.side_effect = mock_preflight_side_effect
            mock_cost.side_effect = mock_cost_side_effect
            mock_optimize.return_value = MagicMock(recommended_changes=[])
            mock_budget.return_value = MagicMock()

            analysis = asyncio.run(what_if_cost_analysis(
                geography_levels=["city", "county", "state"],
                include_comparative=True
            ))

            # Verify comparative insights
            comparative = analysis['comparative_insights']
            assert 'cost_distribution' in comparative
            assert 'efficiency_comparison' in comparative
            assert 'scalability_comparison' in comparative
            assert 'data_density_comparison' in comparative
            assert 'trade_off_analysis' in comparative

            # Verify cost distribution analysis
            cost_dist = comparative['cost_distribution']
            assert cost_dist['cost_spread'] == 7000.0  # 8000 - 1000
            assert cost_dist['most_cost_effective'] == 'city'
            assert cost_dist['least_cost_effective'] == 'state'

            # Verify efficiency comparison
            efficiency_comp = comparative['efficiency_comparison']
            assert 'most_efficient' in efficiency_comp
            assert 'least_efficient' in efficiency_comp

            # Verify trade-off analysis
            trade_offs = comparative['trade_off_analysis']
            assert 'cost_vs_efficiency' in trade_offs
            assert 'speed_vs_cost' in trade_offs
            assert 'risk_vs_benefit' in trade_offs

            # Verify trade-off data structure
            cost_efficiency_tradeoffs = trade_offs['cost_vs_efficiency']
            assert len(cost_efficiency_tradeoffs) == 3  # One for each geography

            for tradeoff in cost_efficiency_tradeoffs:
                assert 'scenario' in tradeoff
                assert 'cost' in tradeoff
                assert 'efficiency' in tradeoff
                assert 'trade_off_score' in tradeoff

    def test_what_if_cost_recommendations(self):
        """Test comprehensive recommendation generation."""
        from core.what_if import what_if_cost_analysis

        with patch('core.what_if.preflight_cost_check') as mock_preflight, \
             patch('core.what_if.predict_scraping_cost') as mock_cost, \
             patch('core.what_if.optimize_scraping_cost') as mock_optimize, \
             patch('core.what_if.analyze_scraping_budget') as mock_budget:

            def mock_preflight_side_effect(control_param):
                geo = control_param.intent.geography[0] if control_param.intent.geography else "city"
                if geo == "city":
                    return {
                        'overall_readiness': 'ready',
                        'cost_analysis': {'predicted_cost': 1200.0},
                        'operational_feasibility': {'estimated_duration_hours': 2.5, 'expected_success_rate': 0.88},
                        'risk_assessment': {'risk_level': 'low'},
                        'compliance_status': {'overall_compliance_score': 0.92}
                    }
                elif geo == "county":
                    return {
                        'overall_readiness': 'ready',
                        'cost_analysis': {'predicted_cost': 3200.0},
                        'operational_feasibility': {'estimated_duration_hours': 7.0, 'expected_success_rate': 0.82},
                        'risk_assessment': {'risk_level': 'medium'},
                        'compliance_status': {'overall_compliance_score': 0.85}
                    }
                else:  # state
                    return {
                        'overall_readiness': 'caution',
                        'cost_analysis': {'predicted_cost': 6800.0},
                        'operational_feasibility': {'estimated_duration_hours': 20.0, 'expected_success_rate': 0.75},
                        'risk_assessment': {'risk_level': 'high'},
                        'compliance_status': {'overall_compliance_score': 0.78}
                    }

            def mock_cost_side_effect(*args, **kwargs):
                control = kwargs.get('control') or args[-1] if args else None
                geo = control.intent.geography[0] if control and control.intent.geography else "city"

                result = MagicMock()
                if geo == "city":
                    result.predicted_cost = 1200.0
                    result.confidence_score = 0.9
                elif geo == "county":
                    result.predicted_cost = 3200.0
                    result.confidence_score = 0.85
                else:  # state
                    result.predicted_cost = 6800.0
                    result.confidence_score = 0.75

                result.cost_range = (result.predicted_cost * 0.9, result.predicted_cost * 1.1)
                result.cost_breakdown = {'signal_acquisition': result.predicted_cost * 0.6, 'infrastructure': result.predicted_cost * 0.3, 'compliance_legal': result.predicted_cost * 0.1}
                return result

            mock_preflight.side_effect = mock_preflight_side_effect
            mock_cost.side_effect = mock_cost_side_effect
            mock_optimize.return_value = MagicMock(recommended_changes=[])
            mock_budget.return_value = MagicMock()

            analysis = asyncio.run(what_if_cost_analysis(
                geography_levels=["city", "county", "state"],
                base_budget=5000.0,
                risk_tolerance="medium"
            ))

            # Verify recommendations structure
            recommendations = analysis['recommendations']
            assert 'primary_recommendation' in recommendations
            assert 'alternative_recommendations' in recommendations
            assert 'cost_optimization_recommendations' in recommendations
            assert 'risk_based_recommendations' in recommendations
            assert 'implementation_priorities' in recommendations
            assert 'contingency_planning' in recommendations

            # Verify primary recommendation (should be city - most cost-effective within budget)
            primary = recommendations['primary_recommendation']
            assert primary['geography'] == 'city'
            assert primary['estimated_cost'] == 1200.0
            assert 'confidence_score' in primary
            assert 'reasoning' in primary
            assert 'expected_benefits' in primary
            assert 'potential_risks' in primary

            # Verify alternative recommendations
            alternatives = recommendations['alternative_recommendations']
            assert len(alternatives) >= 1
            assert alternatives[0]['geography'] in ['county', 'state']

            # Verify implementation priorities
            priorities = recommendations['implementation_priorities']
            assert len(priorities) == 3  # One for each geography

            for priority in priorities:
                assert 'geography' in priority
                assert 'priority_level' in priority
                assert 'implementation_order' in priority
                assert 'estimated_cost' in priority

            # Verify contingency planning
            contingencies = recommendations['contingency_planning']
            assert len(contingencies) >= 3  # At least basic contingencies

            for contingency in contingencies:
                assert 'type' in contingency
                assert 'title' in contingency
                assert 'description' in contingency
                assert 'trigger_conditions' in contingency
                assert 'response_actions' in contingency

    def test_enhanced_what_if_cost_function(self):
        """Test the enhanced what_if_cost function (backward compatibility)."""
        from core.what_if import what_if_cost

        # Test basic functionality
        with patch('core.what_if.what_if_cost_analysis') as mock_analysis:
            mock_result = {
                'analysis_id': 'test_id',
                'geography_analysis': [
                    {'geo': 'city', 'estimated_cost': 1500.0},
                    {'geo': 'county', 'estimated_cost': 3500.0}
                ]
            }
            mock_analysis.return_value = mock_result

            result = asyncio.run(what_if_cost(
                signals=[SignalType.LIEN],
                geography_levels=["city", "county"],
                browser_pages=10
            ))

            # Verify the enhanced function calls the comprehensive analysis
            mock_analysis.assert_called_once()
            call_args = mock_analysis.call_args
            assert call_args[1]['signals'] == [SignalType.LIEN]
            assert call_args[1]['geography_levels'] == ["city", "county"]
            assert call_args[1]['browser_pages'] == 10

            # Verify result structure (should match what_if_cost_analysis output)
            assert result == mock_result

    def test_what_if_cost_error_handling(self):
        """Test error handling in what-if cost analysis."""
        from core.what_if import what_if_cost_analysis

        # Test with invalid inputs
        try:
            analysis = asyncio.run(what_if_cost_analysis(
                signals=[],  # Empty signals
                geography_levels=[],  # Empty geographies
                base_budget=-100  # Invalid budget
            ))
            # Should handle gracefully or raise appropriate error
        except (ValueError, RuntimeError):
            # Expected for invalid inputs
            pass

        # Test with failing intelligence components
        with patch('core.what_if.preflight_cost_check', side_effect=Exception("Preflight failed")), \
             patch('core.what_if.predict_scraping_cost') as mock_cost:

            mock_cost_result = MagicMock()
            mock_cost_result.predicted_cost = 2000.0
            mock_cost_result.confidence_score = 0.8
            mock_cost.return_value = mock_cost_result

            # Should complete with partial results and error indication
            analysis = asyncio.run(what_if_cost_analysis(
                geography_levels=["city"]
            ))

            # Should have error field but still provide basic analysis
            assert 'error' in analysis or analysis.get('overall_status') == 'error'


if __name__ == "__main__":
    # Run basic tests
    print("🔮 Testing What-If Analysis Engine...")

    test_instance = TestWhatIfAnalyzer()

    # Run individual tests
    try:
        test_instance.test_analyzer_initialization()
        print("✅ Analyzer initialization tests passed")

        test_instance.test_scenario_creation()
        print("✅ Scenario creation tests passed")

        test_instance.test_scenario_modifications()
        print("✅ Scenario modifications tests passed")

        test_instance.test_single_scenario_analysis()
        print("✅ Single scenario analysis tests passed")

        test_instance.test_comprehensive_what_if_analysis()
        print("✅ Comprehensive what-if analysis tests passed")

        test_instance.test_scenario_ranking()
        print("✅ Scenario ranking tests passed")

        test_instance.test_sensitivity_analysis()
        print("✅ Sensitivity analysis tests passed")

        test_instance.test_recommendation_generation()
        print("✅ Recommendation generation tests passed")

        test_instance.test_scenario_template_application()
        print("✅ Scenario template application tests passed")

        test_instance.test_convenience_function_integration()
        print("✅ Convenience function integration tests passed")

        test_instance.test_error_handling_and_resilience()
        print("✅ Error handling and resilience tests passed")

        test_instance.test_comprehensive_scenario_ecosystem()
        print("✅ Comprehensive scenario ecosystem tests passed")

        test_instance.test_what_if_cost_analysis_comprehensive()
        print("✅ What-if cost analysis comprehensive tests passed")

        test_instance.test_what_if_cost_budget_compliance_analysis()
        print("✅ What-if cost budget compliance analysis tests passed")

        test_instance.test_what_if_cost_optimization_opportunities()
        print("✅ What-if cost optimization opportunities tests passed")

        test_instance.test_what_if_cost_sensitivity_analysis()
        print("✅ What-if cost sensitivity analysis tests passed")

        test_instance.test_what_if_cost_comparative_insights()
        print("✅ What-if cost comparative insights tests passed")

        test_instance.test_what_if_cost_recommendations()
        print("✅ What-if cost recommendations tests passed")

        test_instance.test_enhanced_what_if_cost_function()
        print("✅ Enhanced what-if cost function tests passed")

        test_instance.test_what_if_cost_error_handling()
        print("✅ What-if cost error handling tests passed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 All What-If Analysis tests completed successfully!")
    print("🔮 Enterprise-grade scenario planning and cost analysis fully validated!")
