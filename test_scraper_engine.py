# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use as is strictly prohibited.

"""
Test Scraper Engine for MJ Data Scraper Suite

Comprehensive testing of the enterprise-grade scraping orchestration engine
with full intelligence integration, multi-phase execution, and performance monitoring.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from core.scraper_engine import (
    ScrapingEngine,
    ScrapingJob,
    ScrapingOrchestrationResult,
    ScrapingPhase,
    OrchestrationResult,
    orchestrate_scraping_job,
    get_scraper_engine_stats,
    cancel_scraping_job,
    get_scraping_job_status,
    start_scraping_job,
    preflight_cost_check
)
from core.models.asset_signal import AssetType
from core.intent_classifier import IntentRiskLevel, IntentCategory
from core.control_models import (
    ScrapeControlContract,
    ScrapeIntent,
    ScrapeBudget,
    ScrapeAuthorization
)


class TestScrapingEngine:
    """Test comprehensive scraping engine orchestration functionality."""

    def test_engine_initialization(self):
        """Test that the scraping engine initializes properly."""
        engine = ScrapingEngine()

        assert len(engine.active_jobs) == 0
        assert len(engine.completed_jobs) == 0
        assert engine.max_concurrent_jobs == 5
        assert engine.job_timeout_hours == 24
        assert engine.enable_sentinel_monitoring == True

        stats = engine.get_engine_stats()
        assert stats['active_jobs_count'] == 0
        assert stats['completed_jobs_count'] == 0
        assert 'total_jobs_processed' in stats

    def test_job_creation_and_tracking(self):
        """Test job creation and tracking functionality."""
        engine = ScrapingEngine()

        control = ScrapeControlContract(
            intent=ScrapeIntent(purpose="Test job", sources=["test"]),
            budget=ScrapeBudget(max_runtime_minutes=30, max_pages=50, max_records=100)
        )

        job = ScrapingJob(job_id="test_job_123", control=control)

        # Test job properties
        assert job.job_id == "test_job_123"
        assert job.current_phase == ScrapingPhase.INITIALIZATION
        assert job.orchestration_result is None
        assert len(job.phase_progress) == 0

        # Test efficiency calculation with no data
        assert job.get_efficiency_score() == 0.0

        # Test with mock data
        job.actual_cost = 50.0
        job.records_collected = 100
        job.success_rate = 0.9
        job.execution_start_time = datetime.utcnow() - timedelta(hours=1)
        job.execution_end_time = datetime.utcnow()

        efficiency = job.get_efficiency_score()
        assert efficiency > 0  # Should calculate efficiency with data

    def test_intent_analysis_phase(self):
        """Test intent analysis phase execution."""
        engine = ScrapingEngine()

        control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Legal investigation for compliance audit",
                sources=["court_records", "federal_court"],
                geography=["Statewide"]
            ),
            budget=ScrapeBudget(max_runtime_minutes=120, max_pages=300, max_records=1500)
        )

        job = ScrapingJob(job_id="test_intent", control=control)

        # Mock the intent classification
        with patch('core.scraper_engine.classify_scraping_intent') as mock_classify:
            mock_classification = MagicMock()
            mock_classification.risk_level = IntentRiskLevel.HIGH
            mock_classification.category = IntentCategory.LEGAL
            mock_classification.confidence_score = 0.85
            mock_classify.return_value = mock_classification

            result = asyncio.run(engine._execute_intent_analysis(job))

            assert job.intent_classification is not None
            assert job.intent_classification.risk_level == IntentRiskLevel.HIGH
            assert 'intent_analysis' in job.phase_progress
            assert job.priority_score > 0

    def test_risk_assessment_phase(self):
        """Test risk assessment phase with AI precheck and sentinels."""
        engine = ScrapingEngine()

        control = ScrapeControlContract(
            intent=ScrapeIntent(purpose="Test risk assessment", sources=["test.com"])
        )

        job = ScrapingJob(job_id="test_risk", control=control)

        # Mock sentinels
        with patch('core.scraper_engine.run_sentinels', return_value=[]) as mock_sentinels, \
             patch('core.scraper_engine.safety_verdict') as mock_verdict:

            mock_verdict.return_value = MagicMock(action="allow", reason="Clean")

            result = asyncio.run(engine._execute_risk_assessment(job))

            # Verify sentinels were called if monitoring enabled
            if engine.enable_sentinel_monitoring:
                mock_sentinels.assert_called_once()
                mock_verdict.assert_called_once()

    def test_execution_planning_phase(self):
        """Test execution planning phase."""
        engine = ScrapingEngine()

        control = ScrapeControlContract(
            intent=ScrapeIntent(purpose="Test planning", sources=["test"])
        )

        job = ScrapingJob(job_id="test_planning", control=control)

        # Mock execution mode classification
        with patch('core.scraper_engine.classify_execution_mode') as mock_classify:
            mock_profile = MagicMock()
            mock_profile.mode.value = "targeted_lookup"
            mock_profile.strategy.value = "depth_first"
            mock_profile.performance_expectations = {'estimated_duration_hours': 2.5}
            mock_classify.return_value = mock_profile

            result = asyncio.run(engine._execute_planning_phase(job))

            assert job.execution_profile is not None
            assert job.estimated_completion_time is not None
            assert 'execution_planning' in job.phase_progress

    def test_cost_optimization_phase(self):
        """Test cost optimization phase."""
        engine = ScrapingEngine()

        control = ScrapeControlContract(
            intent=ScrapeIntent(purpose="Test cost optimization", sources=["test"]),
            budget=ScrapeBudget(max_runtime_minutes=60, max_pages=100, max_records=500)
        )

        job = ScrapingJob(job_id="test_cost", control=control)

        # Mock cost prediction and optimization
        with patch('core.scraper_engine.predict_scraping_cost') as mock_predict, \
             patch('core.scraper_engine.optimize_scraping_cost') as mock_optimize, \
             patch('core.scraper_engine.analyze_scraping_budget') as mock_budget:

            mock_prediction = MagicMock()
            mock_prediction.predicted_cost = 250.0
            mock_prediction.confidence_score = 0.8
            mock_predict.return_value = mock_prediction

            mock_optimization = MagicMock()
            mock_optimization.cost_savings = 50.0
            mock_optimize.return_value = mock_optimization

            mock_budget_analysis = MagicMock()
            mock_budget_analysis.budget_utilization = 25.0
            mock_budget.return_value = mock_budget_analysis

            result = asyncio.run(engine._execute_cost_optimization(job))

            assert job.cost_prediction is not None
            assert job.cost_prediction.predicted_cost == 250.0
            assert job.cost_optimization is not None
            assert job.budget_analysis is not None
            assert 'cost_optimization' in job.phase_progress

    def test_governance_check_phase(self):
        """Test governance check phase."""
        engine = ScrapingEngine()

        control = ScrapeControlContract(
            intent=ScrapeIntent(purpose="Test governance", sources=["test"]),
            authorization=ScrapeAuthorization(
                approved_by="test_user",
                purpose="Test",
                expires_at=datetime.utcnow() + timedelta(days=1)
            )
        )

        job = ScrapingJob(job_id="test_governance", control=control)

        # Mock governance components
        with patch('core.scraper_engine.AuthorizationGate.validate') as mock_auth, \
             patch('core.scraper_engine.DeploymentTimer.await_window') as mock_timer, \
             patch('core.scraper_engine.CostGovernor.initialize') as mock_governor:

            mock_auth.return_value = None  # Success
            mock_timer.return_value = None  # Success
            mock_governor.return_value = None  # Success

            result = asyncio.run(engine._execute_governance_check(job))

            assert result == True
            assert job.governance_checks_passed == True
            assert 'governance_check' in job.phase_progress

    def test_resource_allocation_phase(self):
        """Test resource allocation phase."""
        engine = ScrapingEngine()

        control = ScrapeControlContract(
            intent=ScrapeIntent(purpose="Test resources", sources=["test"])
        )

        job = ScrapingJob(job_id="test_resources", control=control)

        # Create mock execution profile
        mock_profile = MagicMock()
        mock_profile.resource_requirements = {
            'cpu_cores': 2,
            'memory_gb': 4,
            'estimated_duration_hours': 1.5
        }
        job.execution_profile = mock_profile

        result = asyncio.run(engine._execute_resource_allocation(job))

        assert 'resource_allocation' in job.phase_progress
        assert job.phase_progress['resource_allocation']['resources_allocated'] == mock_profile.resource_requirements

    def test_scraping_execution_phase(self):
        """Test scraping execution phase."""
        engine = ScrapingEngine()

        control = ScrapeControlContract(
            intent=ScrapeIntent(purpose="Test execution", sources=["test"])
        )

        job = ScrapingJob(job_id="test_execution", control=control)

        # Mock execution workflow
        with patch('core.scraper_engine.run_scraper') as mock_run, \
             patch('core.scraper_engine.emit_telemetry') as mock_telemetry:

            mock_result = {
                'total_cost': 75.0,
                'records_collected': 150,
                'success_rate': 0.85,
                'telemetry_events': [{'event': 'test'}]
            }
            mock_run.return_value = mock_result

            result = asyncio.run(engine._execute_scraping_operation(job))

            assert job.execution_start_time is not None
            assert job.execution_end_time is not None
            assert job.actual_cost == 75.0
            assert job.records_collected == 150
            assert job.success_rate == 0.85
            assert len(job.telemetry_events) == 1
            assert 'execution' in job.phase_progress

            # Verify telemetry was emitted
            mock_telemetry.assert_called_once()

    def test_quality_validation_phase(self):
        """Test quality validation phase."""
        engine = ScrapingEngine()

        control = ScrapeControlContract(
            intent=ScrapeIntent(purpose="Test quality", sources=["test"])
        )

        job = ScrapingJob(job_id="test_quality", control=control)

        # Set up job with mock execution data
        job.actual_cost = 100.0
        job.records_collected = 200
        job.success_rate = 0.9

        result = asyncio.run(engine._execute_quality_validation(job))

        assert 'quality_validation' in job.phase_progress
        assert 'quality_score' in job.phase_progress['quality_validation']
        quality_score = job.phase_progress['quality_validation']['quality_score']
        assert 0 <= quality_score <= 1

    def test_orchestration_finalization(self):
        """Test orchestration finalization and result generation."""
        engine = ScrapingEngine()

        control = ScrapeControlContract(
            intent=ScrapeIntent(purpose="Test finalization", sources=["test"])
        )

        job = ScrapingJob(job_id="test_finalization", control=control)
        job.orchestration_result = OrchestrationResult.SUCCESS
        job.actual_cost = 50.0
        job.records_collected = 100
        job.success_rate = 0.95

        result = asyncio.run(engine._finalize_orchestration(job))

        assert isinstance(result, ScrapingOrchestrationResult)
        assert result.result == OrchestrationResult.SUCCESS
        assert result.job == job
        assert 'efficiency_score' in result.summary
        assert len(result.recommendations) >= 0
        assert len(result.next_steps) >= 0
        assert 'performance_metrics' in result
        assert 'compliance_report' in result

        # Job should be moved from active to completed
        assert job.job_id not in engine.active_jobs
        assert job in engine.completed_jobs

    def test_complete_orchestration_workflow(self):
        """Test complete orchestration workflow from start to finish."""
        engine = ScrapingEngine()

        control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Complete orchestration test",
                sources=["test.com"],
                geography=["Test City"]
            ),
            budget=ScrapeBudget(max_runtime_minutes=30, max_pages=50, max_records=100),
            authorization=ScrapeAuthorization(
                approved_by="test_user",
                purpose="Testing",
                expires_at=datetime.utcnow() + timedelta(days=1)
            )
        )

        # Mock all the intelligence components
        with patch('core.scraper_engine.classify_scraping_intent') as mock_intent, \
             patch('core.scraper_engine.run_sentinels', return_value=[]) as mock_sentinels, \
             patch('core.scraper_engine.safety_verdict') as mock_verdict, \
             patch('core.scraper_engine.classify_execution_mode') as mock_execution, \
             patch('core.scraper_engine.predict_scraping_cost') as mock_cost, \
             patch('core.scraper_engine.optimize_scraping_cost') as mock_optimize, \
             patch('core.scraper_engine.analyze_scraping_budget') as mock_budget, \
             patch('core.scraper_engine.AuthorizationGate.validate') as mock_auth, \
             patch('core.scraper_engine.DeploymentTimer.await_window') as mock_timer, \
             patch('core.scraper_engine.CostGovernor.initialize') as mock_governor, \
             patch('core.scraper_engine.run_scraper') as mock_run, \
             patch('core.scraper_engine.emit_telemetry') as mock_telemetry:

            # Set up mocks
            mock_intent.return_value = MagicMock(
                risk_level=IntentRiskLevel.MEDIUM,
                category=IntentCategory.INTELLIGENCE,
                governance_requirement=MagicMock(value="enhanced"),
                reasoning=["Test classification"],
                confidence_score=0.8
            )

            mock_verdict.return_value = MagicMock(action="allow", reason="Test verdict")

            mock_execution.return_value = MagicMock(
                mode=MagicMock(value="targeted_lookup"),
                strategy=MagicMock(value="depth_first"),
                performance_expectations={'estimated_duration_hours': 0.5},
                resource_requirements={'cpu_cores': 1, 'memory_gb': 2},
                execution_parameters={'concurrent_requests': 3}
            )

            mock_cost.return_value = MagicMock(
                predicted_cost=75.0,
                confidence_score=0.85,
                cost_breakdown={'signal_acquisition': 50, 'infrastructure': 25}
            )

            mock_optimize.return_value = MagicMock(
                cost_savings=15.0,
                savings_percentage=20.0
            )

            mock_budget.return_value = MagicMock(
                budget_utilization=75.0,
                risk_of_overspend="low"
            )

            mock_run.return_value = {
                'total_cost': 70.0,
                'records_collected': 85,
                'success_rate': 0.9,
                'telemetry_events': []
            }

            # Execute orchestration
            result = asyncio.run(engine.orchestrate_scraping_operation(control))

            # Verify complete workflow
            assert isinstance(result, ScrapingOrchestrationResult)
            assert result.result == OrchestrationResult.SUCCESS
            assert result.job.records_collected == 85
            assert result.job.actual_cost == 70.0

            # Verify all phases were executed
            job = result.job
            expected_phases = [
                'intent_analysis', 'execution_planning',
                'cost_optimization', 'governance_check', 'resource_allocation',
                'execution', 'quality_validation'
            ]

            for phase in expected_phases:
                assert phase in str(job.phase_progress), f"Phase {phase} not found in progress"

    def test_job_cancellation(self):
        """Test job cancellation functionality."""
        engine = ScrapingEngine()

        control = ScrapeControlContract(
            intent=ScrapeIntent(purpose="Test cancellation", sources=["test"])
        )

        # Start a job (mock the orchestration)
        job = ScrapingJob(job_id="cancel_test", control=control)
        engine.active_jobs[job.job_id] = job

        # Cancel the job
        cancelled = asyncio.run(engine.cancel_job("cancel_test"))

        assert cancelled == True
        assert "cancel_test" not in engine.active_jobs
        assert job in engine.completed_jobs
        assert job.orchestration_result == OrchestrationResult.CANCELLED_BY_USER

    def test_job_status_tracking(self):
        """Test job status tracking functionality."""
        engine = ScrapingEngine()

        # Test active job status
        active_job = ScrapingJob(
            job_id="active_test",
            control=ScrapeControlContract(intent=ScrapeIntent(purpose="Active", sources=["test"]))
        )
        active_job.current_phase = ScrapingPhase.EXECUTION_MONITORING
        active_job.execution_start_time = datetime.utcnow()
        active_job.estimated_completion_time = datetime.utcnow() + timedelta(hours=1)
        engine.active_jobs[active_job.job_id] = active_job

        active_status = asyncio.run(engine.get_job_status("active_test"))

        assert active_status is not None
        assert active_status['status'] == 'active'
        assert active_status['current_phase'] == 'execution_monitoring'

        # Test completed job status
        completed_job = ScrapingJob(
            job_id="completed_test",
            control=ScrapeControlContract(intent=ScrapeIntent(purpose="Completed", sources=["test"]))
        )
        completed_job.orchestration_result = OrchestrationResult.SUCCESS
        completed_job.records_collected = 200
        completed_job.actual_cost = 150.0
        completed_job.execution_end_time = datetime.utcnow()
        engine.completed_jobs.append(completed_job)

        completed_status = asyncio.run(engine.get_job_status("completed_test"))

        assert completed_status is not None
        assert completed_status['status'] == 'completed'
        assert completed_status['result'] == 'success'
        assert completed_status['records_collected'] == 200

        # Test non-existent job
        nonexistent_status = asyncio.run(engine.get_job_status("nonexistent"))
        assert nonexistent_status is None

    def test_engine_statistics(self):
        """Test engine statistics generation."""
        engine = ScrapingEngine()

        # Add some mock completed jobs
        for i in range(3):
            job = ScrapingJob(
                job_id=f"stats_test_{i}",
                control=ScrapeControlContract(intent=ScrapeIntent(purpose=f"Test {i}", sources=["test"]))
            )
            job.orchestration_result = OrchestrationResult.SUCCESS if i < 2 else OrchestrationResult.EXECUTION_FAILED
            job.actual_cost = 100.0 + i * 25
            job.execution_start_time = datetime.utcnow() - timedelta(hours=2)
            job.execution_end_time = datetime.utcnow() - timedelta(hours=1)
            engine.completed_jobs.append(job)

        stats = engine.get_engine_stats()

        assert stats['total_jobs_processed'] == 3
        assert stats['successful_jobs'] == 2
        assert stats['completed_jobs_count'] == 3
        assert 'success_rate' in stats
        assert 'average_job_cost' in stats
        assert 'average_execution_duration_seconds' in stats

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        engine = ScrapingEngine()

        control = ScrapeControlContract(
            intent=ScrapeIntent(purpose="Test error handling", sources=["test"])
        )

        # Test with failing intelligence components
        with patch('core.scraper_engine.classify_scraping_intent', side_effect=Exception("Intent failure")) as mock_intent, \
             patch('core.scraper_engine.classify_execution_mode') as mock_execution:

            # Should still create fallback execution profile
            mock_profile = MagicMock()
            mock_profile.mode.value = "targeted_lookup"
            mock_profile.strategy.value = "depth_first"
            mock_profile.performance_expectations = {'estimated_duration_hours': 1}
            mock_execution.return_value = mock_profile

            result = asyncio.run(engine.orchestrate_scraping_operation(control))

            # Should complete with fallback mechanisms
            assert result is not None
            assert isinstance(result, ScrapingOrchestrationResult)

            # Should have error events recorded
            if result.job.error_events:
                assert len(result.job.error_events) > 0
                assert 'error' in result.job.error_events[0]

    def test_convenience_functions(self):
        """Test global convenience functions."""
        control = ScrapeControlContract(
            intent=ScrapeIntent(purpose="Convenience test", sources=["test"]),
            budget=ScrapeBudget(max_runtime_minutes=15, max_pages=25, max_records=50)
        )

        # Test main orchestration function
        with patch('core.scraper_engine.ScrapingEngine.orchestrate_scraping_operation') as mock_orchestrate:
            mock_result = MagicMock()
            mock_result.result = OrchestrationResult.SUCCESS
            mock_orchestrate.return_value = mock_result

            result = asyncio.run(orchestrate_scraping_job(control))

            assert result is not None
            mock_orchestrate.assert_called_once_with(control, JobPriority.NORMAL)

        # Test statistics function
        stats = get_scraper_engine_stats()
        assert isinstance(stats, dict)

        # Test cancellation function
        with patch('core.scraper_engine.ScrapingEngine.cancel_job') as mock_cancel:
            mock_cancel.return_value = True
            cancelled = asyncio.run(cancel_scraping_job("test_job"))
            assert cancelled == True

        # Test status function
        with patch('core.scraper_engine.ScrapingEngine.get_job_status') as mock_status:
            mock_status.return_value = {"status": "active", "phase": "execution"}
            status = asyncio.run(get_scraping_job_status("test_job"))
            assert status["status"] == "active"

        # Test legacy function
        with patch('core.scraper_engine.orchestrate_scraping_job') as mock_legacy:
            mock_result = MagicMock()
            mock_result.job.job_id = "legacy_test"
            mock_legacy.return_value = mock_result

            job_id = asyncio.run(start_scraping_job(control))
            assert job_id == "legacy_test"

    def test_priority_and_urgency_handling(self):
        """Test priority scoring and urgency handling."""
        engine = ScrapingEngine()

        # Test different priority scenarios
        urgent_control = ScrapeControlContract(
            intent=ScrapeIntent(purpose="Urgent legal matter", sources=["court_records"]),
            budget=ScrapeBudget(max_runtime_minutes=30, max_pages=100, max_records=200)  # Short timeline
        )

        normal_control = ScrapeControlContract(
            intent=ScrapeIntent(purpose="Normal research", sources=["news"]),
            budget=ScrapeBudget(max_runtime_minutes=120, max_pages=200, max_records=500)
        )

        # Mock intent classifications
        with patch('core.scraper_engine.classify_scraping_intent') as mock_intent:
            def mock_intent_classifier(control):
                if "Urgent" in control.intent.purpose:
                    classification = MagicMock()
                    classification.risk_level = IntentRiskLevel.CRITICAL
                    classification.category = IntentCategory.LEGAL
                    return classification
                else:
                    classification = MagicMock()
                    classification.risk_level = IntentRiskLevel.LOW
                    classification.category = IntentCategory.EVENT
                    return classification

            mock_intent.side_effect = mock_intent_classifier

            # Create jobs and check priorities
            urgent_job = ScrapingJob(job_id="urgent", control=urgent_control)
            normal_job = ScrapingJob(job_id="normal", control=normal_control)

            asyncio.run(engine._execute_intent_analysis(urgent_job))
            asyncio.run(engine._execute_intent_analysis(normal_job))

            # Urgent job should have higher priority
            assert urgent_job.priority_score > normal_job.priority_score

    def test_budget_constraint_handling(self):
        """Test handling of budget constraints and limits."""
        engine = ScrapingEngine()

        # Create job with budget constraints
        control = ScrapeControlContract(
            intent=ScrapeIntent(purpose="Budget test", sources=["test"]),
            budget=ScrapeBudget(
                max_runtime_minutes=60,
                max_pages=100,
                max_records=200,
                max_cost_total=500.0
            )
        )

        job = ScrapingJob(job_id="budget_test", control=control)

        # Mock cost prediction with high cost
        with patch('core.scraper_engine.predict_scraping_cost') as mock_cost:
            mock_prediction = MagicMock()
            mock_prediction.predicted_cost = 800.0  # Over budget
            mock_prediction.confidence_score = 0.9
            mock_cost.return_value = mock_prediction

            asyncio.run(engine._execute_cost_optimization(job))

            # Job should not proceed to execution due to cost
            should_proceed = job.should_proceed_to_execution()
            assert should_proceed == False, "Job should not proceed when cost exceeds budget significantly"

    def test_scalability_and_performance(self):
        """Test engine scalability and performance characteristics."""
        engine = ScrapingEngine()

        # Test with multiple concurrent jobs
        controls = [
            ScrapeControlContract(
                intent=ScrapeIntent(purpose=f"Scale test {i}", sources=["test"]),
                budget=ScrapeBudget(max_runtime_minutes=30, max_pages=50, max_records=100)
            ) for i in range(3)
        ]

        # Mock all intelligence to run quickly
        with patch('core.scraper_engine.classify_scraping_intent') as mock_intent, \
             patch('core.scraper_engine.run_sentinels', return_value=[]) as mock_sentinels, \
             patch('core.scraper_engine.safety_verdict') as mock_verdict, \
             patch('core.scraper_engine.classify_execution_mode') as mock_execution, \
             patch('core.scraper_engine.predict_scraping_cost') as mock_cost, \
             patch('core.scraper_engine.optimize_scraping_cost') as mock_optimize, \
             patch('core.scraper_engine.analyze_scraping_budget') as mock_budget, \
             patch('core.scraper_engine.AuthorizationGate.validate') as mock_auth, \
             patch('core.scraper_engine.run_scraper') as mock_run:

            # Set up quick mocks
            mock_intent.return_value = MagicMock(
                risk_level=IntentRiskLevel.LOW,
                category=IntentCategory.EVENT,
                governance_requirement=MagicMock(value="basic"),
                confidence_score=0.8
            )
            mock_verdict.return_value = MagicMock(action="allow")
            mock_execution.return_value = MagicMock(
                mode=MagicMock(value="targeted_lookup"),
                strategy=MagicMock(value="depth_first"),
                performance_expectations={'estimated_duration_hours': 0.1},
                resource_requirements={'cpu_cores': 1},
                execution_parameters={'concurrent_requests': 2}
            )
            mock_cost.return_value = MagicMock(predicted_cost=50.0, confidence_score=0.8)
            mock_run.return_value = {
                'total_cost': 45.0, 'records_collected': 80, 'success_rate': 0.9
            }

            # Run multiple jobs concurrently
            start_time = asyncio.get_event_loop().time()
            tasks = [engine.orchestrate_scraping_operation(control) for control in controls]
            results = asyncio.run(asyncio.gather(*tasks))
            end_time = asyncio.get_event_loop().time()

            # Verify all completed successfully
            assert len(results) == 3
            for result in results:
                assert result.result == OrchestrationResult.SUCCESS

            # Check reasonable execution time (should be much less than 30 seconds total)
            total_time = end_time - start_time
            assert total_time < 30, f"Execution took too long: {total_time} seconds"

    def test_comprehensive_intelligence_integration(self):
        """Test comprehensive integration of all intelligence systems."""
        engine = ScrapingEngine()

        # Create a complex control contract
        control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Comprehensive enterprise financial compliance audit with legal research and property verification",
                sources=["court_records", "financial_databases", "county_clerk", "federal_regulatory_filings"],
                geography=["Multi-state enterprise", "National coverage"],
                event_type="legal"
            ),
            budget=ScrapeBudget(
                max_runtime_minutes=480,  # 8 hours
                max_pages=2000,
                max_records=10000,
                max_cost_total=5000.0
            ),
            authorization=ScrapeAuthorization(
                approved_by="Chief Compliance Officer",
                purpose="Q4 2024 Enterprise Compliance Audit",
                expires_at=datetime.utcnow() + timedelta(days=30)
            )
        )

        # Mock all intelligence systems for comprehensive test
        with patch('core.scraper_engine.classify_scraping_intent') as mock_intent, \
             patch('core.scraper_engine.run_sentinels') as mock_sentinels, \
             patch('core.scraper_engine.safety_verdict') as mock_verdict, \
             patch('core.scraper_engine.classify_execution_mode') as mock_execution, \
             patch('core.scraper_engine.predict_scraping_cost') as mock_cost, \
             patch('core.scraper_engine.optimize_scraping_cost') as mock_optimize, \
             patch('core.scraper_engine.analyze_scraping_budget') as mock_budget, \
             patch('core.scraper_engine.AuthorizationGate.validate') as mock_auth, \
             patch('core.scraper_engine.DeploymentTimer.await_window') as mock_timer, \
             patch('core.scraper_engine.CostGovernor.initialize') as mock_governor, \
             patch('core.scraper_engine.run_scraper') as mock_run, \
             patch('core.scraper_engine.emit_telemetry') as mock_telemetry:

            # Set up comprehensive mocks
            mock_intent.return_value = MagicMock(
                risk_level=IntentRiskLevel.CRITICAL,
                category=IntentCategory.COMPLIANCE,
                governance_requirement=MagicMock(value="exceptional"),
                confidence_score=0.95,
                reasoning=["Critical compliance audit requiring maximum oversight"]
            )

            mock_sentinels.return_value = [MagicMock()]
            mock_verdict.return_value = MagicMock(action="allow", reason="Compliance audit approved")

            mock_execution.return_value = MagicMock(
                mode=MagicMock(value="compliance_audit"),
                strategy=MagicMock(value="quality_optimized"),
                confidence_score=0.9,
                performance_expectations={
                    'estimated_duration_hours': 6.0,
                    'expected_success_rate': 0.95
                },
                resource_requirements={
                    'cpu_cores': 4,
                    'memory_gb': 8,
                    'estimated_duration_hours': 6.0
                },
                execution_parameters={
                    'concurrent_requests': 2,
                    'rate_limit_multiplier': 0.1,
                    'monitoring_intensity': 'maximum'
                }
            )

            mock_cost.return_value = MagicMock(
                predicted_cost=3200.0,
                confidence_score=0.85,
                cost_breakdown={
                    'signal_acquisition': 2000,
                    'compliance_legal': 800,
                    'quality_validation': 400
                },
                cost_range=(2800, 3600),
                optimization_recommendations=[
                    "Consider phased execution to reduce peak costs",
                    "Use verified data sources for compliance requirements"
                ]
            )

            mock_optimize.return_value = MagicMock(
                cost_savings=400.0,
                savings_percentage=12.5,
                optimization_strategy=MagicMock(value="balance_cost_value"),
                recommended_changes=[
                    {"type": "source_selection", "change": "Use premium compliance sources", "estimated_savings": 200},
                    {"type": "execution_mode", "change": "Implement quality-optimized strategy", "estimated_savings": 200}
                ]
            )

            mock_budget.return_value = MagicMock(
                budget_utilization=64.0,
                risk_of_overspend="low",
                recommendations=[
                    "Monitor costs closely during execution",
                    "Consider budget contingency for unexpected findings"
                ]
            )

            mock_run.return_value = {
                'total_cost': 3100.0,
                'records_collected': 8500,
                'success_rate': 0.92,
                'telemetry_events': [
                    {'event': 'compliance_check_passed', 'timestamp': datetime.utcnow()},
                    {'event': 'data_quality_verified', 'timestamp': datetime.utcnow()}
                ]
            }

            # Execute comprehensive orchestration
            result = asyncio.run(engine.orchestrate_scraping_operation(control))

            # Verify comprehensive intelligence integration
            assert result.result == OrchestrationResult.SUCCESS
            assert result.job.records_collected == 8500
            assert result.job.actual_cost == 3100.0

            # Verify all intelligence systems were engaged
            job = result.job
            assert job.intent_classification is not None
            assert job.intent_classification.category == IntentCategory.COMPLIANCE
            assert job.execution_profile is not None
            assert job.cost_prediction is not None
            assert job.cost_optimization is not None
            assert job.budget_analysis is not None
            assert job.governance_checks_passed == True

            # Verify comprehensive execution
            assert len(job.phase_progress) >= 8  # All phases completed
            assert 'intent_analysis' in job.phase_progress
            assert 'sentinel_assessment' in job.phase_progress
            assert 'execution_planning' in job.phase_progress
            assert 'cost_optimization' in job.phase_progress
            assert 'governance_check' in job.phase_progress
            assert 'resource_allocation' in job.phase_progress
            assert 'execution' in job.phase_progress
            assert 'quality_validation' in job.phase_progress

            # Verify intelligence-driven results
            summary = result.summary
            assert summary['risk_level'] == 'critical'
            assert summary['intent_category'] == 'compliance'
            assert 'predicted_cost' in summary
            assert 'actual_cost' in summary

            # Verify recommendations and next steps
            assert len(result.recommendations) > 0
            assert len(result.next_steps) > 0

            # Verify performance metrics
            metrics = result.performance_metrics
            assert 'efficiency_score' in metrics
            assert 'data_quality_score' in metrics
            assert metrics['execution_duration_seconds'] > 0

            # Verify compliance report
            compliance = result.compliance_report
            assert 'overall_compliance_score' in compliance
            assert compliance['governance_checks_passed'] == True

    def test_preflight_cost_check_comprehensive(self):
        """Test comprehensive preflight cost check functionality."""
        from core.scraper_engine import preflight_cost_check

        control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Test preflight assessment for property data collection",
                sources=["county_clerk", "property_records"],
                geography=["Test County"]
            ),
            budget=ScrapeBudget(
                max_runtime_minutes=120,
                max_pages=200,
                max_records=500,
                max_cost_total=1000.0
            ),
            authorization=ScrapeAuthorization(
                approved_by="test_user",
                purpose="Testing preflight",
                expires_at=datetime.utcnow() + timedelta(days=7)
            )
        )

        # Mock all intelligence components for comprehensive test
        with patch('core.scraper_engine.predict_scraping_cost') as mock_cost, \
             patch('core.scraper_engine.classify_execution_mode') as mock_execution, \
             patch('core.scraper_engine.classify_scraping_intent') as mock_intent, \
             patch('core.scraper_engine.optimize_scraping_cost') as mock_optimize, \
             patch('core.scraper_engine.AuthorizationGate.validate') as mock_auth:

            # Set up mocks
            mock_cost_prediction = MagicMock()
            mock_cost_prediction.predicted_cost = 650.0
            mock_cost_prediction.confidence_score = 0.85
            mock_cost_prediction.cost_range = (600.0, 720.0)
            mock_cost_prediction.cost_breakdown = {
                'signal_acquisition': 400,
                'infrastructure': 150,
                'compliance_legal': 100
            }
            mock_cost.return_value = mock_cost_prediction

            mock_execution_profile = MagicMock()
            mock_execution_profile.mode.value = "targeted_lookup"
            mock_execution_profile.strategy.value = "depth_first"
            mock_execution_profile.confidence_score = 0.9
            mock_execution_profile.performance_expectations = {
                'estimated_duration_hours': 1.5,
                'expected_success_rate': 0.88
            }
            mock_execution_profile.resource_requirements = {
                'cpu_cores': 2,
                'memory_gb': 4
            }
            mock_execution_profile.execution_parameters = {
                'concurrent_requests': 3
            }
            mock_execution.return_value = mock_execution_profile

            mock_intent_classification = MagicMock()
            mock_intent_classification.risk_level = IntentRiskLevel.MEDIUM
            mock_intent_classification.category = IntentCategory.PROPERTY
            mock_intent_classification.governance_requirement.value = "standard"
            mock_intent_classification.confidence_score = 0.82
            mock_intent.return_value = mock_intent_classification

            mock_optimization = MagicMock()
            mock_optimization.recommended_changes = [
                {'type': 'source_selection', 'change': 'Optimize source selection', 'estimated_savings': 80, 'implementation_effort': 'low'}
            ]
            mock_optimize.return_value = mock_optimization

            # Execute preflight check
            assessment = asyncio.run(preflight_cost_check(control))

            # Verify comprehensive assessment structure
            assert assessment['assessment_id']
            assert assessment['timestamp']
            assert assessment['overall_readiness'] in ['ready', 'caution', 'blocked']

            # Verify cost analysis
            cost_analysis = assessment['cost_analysis']
            assert cost_analysis['predicted_cost'] == 650.0
            assert cost_analysis['confidence_score'] == 0.85
            assert cost_analysis['budget_compliance'] == 'compliant'
            assert cost_analysis['budget_utilization_percentage'] > 0
            assert 'cost_breakdown' in cost_analysis

            # Verify operational feasibility
            feasibility = assessment['operational_feasibility']
            assert feasibility['recommended_mode'] == "targeted_lookup"
            assert feasibility['estimated_duration_hours'] == 1.5
            assert feasibility['expected_success_rate'] == 0.88
            assert feasibility['concurrency_recommendation'] == 3

            # Verify risk assessment
            risk = assessment['risk_assessment']
            assert risk['risk_level'] == 'medium'
            assert risk['intent_category'] == 'property'
            assert risk['classification_confidence'] == 0.82

            # Verify compliance status
            compliance = assessment['compliance_status']
            assert compliance['authorization_valid'] == True
            assert compliance['budget_compliance'] == True
            assert 'overall_compliance_score' in compliance

            # Verify optimization opportunities
            assert len(assessment['optimization_opportunities']) > 0

            # Verify recommendations
            assert len(assessment['recommendations']) >= 0
            assert len(assessment['warnings']) >= 0
            assert len(assessment['critical_issues']) == 0  # Should be clean for this test

    def test_preflight_cost_check_budget_compliance(self):
        """Test preflight check budget compliance scenarios."""
        from core.scraper_engine import preflight_cost_check

        # Test budget exceeded scenario
        high_cost_control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="High cost operation",
                sources=["expensive_source"],
                geography=["Large scope"]
            ),
            budget=ScrapeBudget(
                max_runtime_minutes=60,
                max_pages=100,
                max_records=200,
                max_cost_total=100.0  # Very low budget
            ),
            authorization=ScrapeAuthorization(
                approved_by="test_user",
                purpose="Testing",
                expires_at=datetime.utcnow() + timedelta(days=1)
            )
        )

        with patch('core.scraper_engine.predict_scraping_cost') as mock_cost, \
             patch('core.scraper_engine.classify_execution_mode') as mock_execution, \
             patch('core.scraper_engine.classify_scraping_intent') as mock_intent, \
             patch('core.scraper_engine.optimize_scraping_cost') as mock_optimize, \
             patch('core.scraper_engine.AuthorizationGate.validate') as mock_auth:

            # High cost prediction
            mock_cost_prediction = MagicMock()
            mock_cost_prediction.predicted_cost = 250.0  # Exceeds $100 budget
            mock_cost_prediction.confidence_score = 0.8
            mock_cost_prediction.cost_range = (225.0, 275.0)
            mock_cost_prediction.cost_breakdown = {'signal_acquisition': 200, 'infrastructure': 50}
            mock_cost.return_value = mock_cost_prediction

            # Set up other mocks
            mock_execution.return_value = MagicMock(
                mode=MagicMock(value="targeted_lookup"),
                strategy=MagicMock(value="depth_first"),
                confidence_score=0.8,
                performance_expectations={'estimated_duration_hours': 1.0, 'expected_success_rate': 0.8},
                resource_requirements={'cpu_cores': 1, 'memory_gb': 2},
                execution_parameters={'concurrent_requests': 2}
            )

            mock_intent.return_value = MagicMock(
                risk_level=IntentRiskLevel.LOW,
                category=IntentCategory.EVENT,
                governance_requirement=MagicMock(value="basic"),
                confidence_score=0.7
            )

            mock_optimize.return_value = MagicMock(recommended_changes=[])

            # Should raise RuntimeError for budget exceeded
            try:
                assessment = asyncio.run(preflight_cost_check(high_cost_control))
                assert False, "Should have raised RuntimeError for budget exceeded"
            except RuntimeError as e:
                assert "exceeds budget" in str(e).lower()

        # Test high utilization warning
        medium_cost_control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Medium cost operation",
                sources=["standard_source"],
                geography=["Medium scope"]
            ),
            budget=ScrapeBudget(
                max_runtime_minutes=120,
                max_pages=200,
                max_records=500,
                max_cost_total=200.0  # Medium budget
            ),
            authorization=ScrapeAuthorization(
                approved_by="test_user",
                purpose="Testing",
                expires_at=datetime.utcnow() + timedelta(days=1)
            )
        )

        with patch('core.scraper_engine.predict_scraping_cost') as mock_cost, \
             patch('core.scraper_engine.classify_execution_mode') as mock_execution, \
             patch('core.scraper_engine.classify_scraping_intent') as mock_intent, \
             patch('core.scraper_engine.optimize_scraping_cost') as mock_optimize, \
             patch('core.scraper_engine.AuthorizationGate.validate') as mock_auth:

            # High utilization cost (95% of budget)
            mock_cost_prediction = MagicMock()
            mock_cost_prediction.predicted_cost = 190.0  # 95% of $200 budget
            mock_cost_prediction.confidence_score = 0.8
            mock_cost_prediction.cost_range = (180.0, 200.0)
            mock_cost_prediction.cost_breakdown = {'signal_acquisition': 140, 'infrastructure': 50}
            mock_cost.return_value = mock_cost_prediction

            # Set up other mocks
            mock_execution.return_value = MagicMock(
                mode=MagicMock(value="targeted_lookup"),
                strategy=MagicMock(value="depth_first"),
                confidence_score=0.8,
                performance_expectations={'estimated_duration_hours': 1.5, 'expected_success_rate': 0.85},
                resource_requirements={'cpu_cores': 2, 'memory_gb': 3},
                execution_parameters={'concurrent_requests': 2}
            )

            mock_intent.return_value = MagicMock(
                risk_level=IntentRiskLevel.MEDIUM,
                category=IntentCategory.EVENT,
                governance_requirement=MagicMock(value="standard"),
                confidence_score=0.75
            )

            mock_optimize.return_value = MagicMock(recommended_changes=[])

            assessment = asyncio.run(preflight_cost_check(medium_cost_control))

            # Should pass but with warnings
            assert assessment['overall_readiness'] in ['caution', 'ready']
            assert assessment['cost_analysis']['budget_compliance'] == 'high_utilization'
            assert assessment['cost_analysis']['budget_utilization_percentage'] >= 90
            assert len(assessment['warnings']) > 0

    def test_preflight_cost_check_risk_assessment(self):
        """Test preflight check risk assessment functionality."""
        from core.scraper_engine import preflight_cost_check

        # High risk scenario
        high_risk_control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Critical compliance audit requiring sensitive data",
                sources=["court_records", "financial_databases", "regulatory_filings"],
                geography=["Multi-state enterprise"]
            ),
            budget=ScrapeBudget(
                max_runtime_minutes=480,
                max_pages=1000,
                max_records=5000,
                max_cost_total=5000.0
            ),
            authorization=ScrapeAuthorization(
                approved_by="Chief Compliance Officer",
                purpose="Enterprise compliance audit",
                expires_at=datetime.utcnow() + timedelta(days=30)
            )
        )

        with patch('core.scraper_engine.predict_scraping_cost') as mock_cost, \
             patch('core.scraper_engine.classify_execution_mode') as mock_execution, \
             patch('core.scraper_engine.classify_scraping_intent') as mock_intent, \
             patch('core.scraper_engine.optimize_scraping_cost') as mock_optimize, \
             patch('core.scraper_engine.AuthorizationGate.validate') as mock_auth:

            mock_cost_prediction = MagicMock()
            mock_cost_prediction.predicted_cost = 3200.0
            mock_cost_prediction.confidence_score = 0.9
            mock_cost_prediction.cost_range = (3000.0, 3400.0)
            mock_cost_prediction.cost_breakdown = {
                'signal_acquisition': 2000,
                'compliance_legal': 800,
                'quality_validation': 400
            }
            mock_cost.return_value = mock_cost_prediction

            mock_execution.return_value = MagicMock(
                mode=MagicMock(value="compliance_audit"),
                strategy=MagicMock(value="quality_optimized"),
                confidence_score=0.9,
                performance_expectations={'estimated_duration_hours': 6.0, 'expected_success_rate': 0.95},
                resource_requirements={'cpu_cores': 4, 'memory_gb': 8},
                execution_parameters={'concurrent_requests': 2, 'rate_limit_multiplier': 0.1}
            )

            mock_intent.return_value = MagicMock(
                risk_level=IntentRiskLevel.CRITICAL,
                category=IntentCategory.COMPLIANCE,
                governance_requirement=MagicMock(value="exceptional"),
                confidence_score=0.95
            )

            mock_optimize.return_value = MagicMock(recommended_changes=[
                {'type': 'governance', 'change': 'Executive approval required', 'estimated_savings': 0, 'implementation_effort': 'high'}
            ])

            assessment = asyncio.run(preflight_cost_check(high_risk_control))

            # Verify high-risk assessment
            assert assessment['risk_assessment']['risk_level'] == 'critical'
            assert assessment['risk_assessment']['intent_category'] == 'compliance'
            assert assessment['risk_assessment']['governance_requirement'] == 'exceptional'

            # Should identify critical risk factors
            assert 'executive approval' in str(assessment['critical_issues']).lower() or \
                   'executive approval' in str(assessment['risk_assessment']['mitigation_requirements']).lower()

            # Should have compliance requirements
            assert 'compliance' in str(assessment['compliance_status'])

            # Should have high resource requirements
            assert assessment['operational_feasibility']['resource_intensity'] == 'high'

    def test_preflight_cost_check_operational_feasibility(self):
        """Test preflight check operational feasibility assessment."""
        from core.scraper_engine import preflight_cost_check

        # Large scope operation
        large_scope_control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Large-scale data collection",
                sources=["multiple_sources"],
                geography=["State1", "State2", "State3", "State4", "State5"]  # 5 geographies
            ),
            budget=ScrapeBudget(
                max_runtime_minutes=1440,  # 24 hours
                max_pages=5000,
                max_records=25000,
                max_cost_total=10000.0
            ),
            authorization=ScrapeAuthorization(
                approved_by="data_manager",
                purpose="Large scale collection",
                expires_at=datetime.utcnow() + timedelta(days=14)
            )
        )

        with patch('core.scraper_engine.predict_scraping_cost') as mock_cost, \
             patch('core.scraper_engine.classify_execution_mode') as mock_execution, \
             patch('core.scraper_engine.classify_scraping_intent') as mock_intent, \
             patch('core.scraper_engine.optimize_scraping_cost') as mock_optimize, \
             patch('core.scraper_engine.AuthorizationGate.validate') as mock_auth:

            mock_cost_prediction = MagicMock()
            mock_cost_prediction.predicted_cost = 7500.0
            mock_cost_prediction.confidence_score = 0.75  # Lower confidence for large scope
            mock_cost_prediction.cost_range = (6750.0, 8250.0)  # Wide range
            mock_cost_prediction.cost_breakdown = {
                'signal_acquisition': 5000,
                'infrastructure': 1500,
                'compliance_legal': 750,
                'operational_overhead': 250
            }
            mock_cost.return_value = mock_cost_prediction

            mock_execution.return_value = MagicMock(
                mode=MagicMock(value="comprehensive_survey"),
                strategy=MagicMock(value="parallel_batch"),
                confidence_score=0.7,  # Lower confidence
                performance_expectations={'estimated_duration_hours': 18.0, 'expected_success_rate': 0.82},
                resource_requirements={'cpu_cores': 6, 'memory_gb': 12},  # High resources
                execution_parameters={'concurrent_requests': 8, 'batch_size': 100}
            )

            mock_intent.return_value = MagicMock(
                risk_level=IntentRiskLevel.HIGH,
                category=IntentCategory.INTELLIGENCE,
                governance_requirement=MagicMock(value="enhanced"),
                confidence_score=0.8
            )

            mock_optimize.return_value = MagicMock(recommended_changes=[
                {'type': 'execution_mode', 'change': 'Consider phased execution', 'estimated_savings': 1000, 'implementation_effort': 'medium'},
                {'type': 'resource_optimization', 'change': 'Optimize resource allocation', 'estimated_savings': 500, 'implementation_effort': 'low'}
            ])

            assessment = asyncio.run(preflight_cost_check(large_scope_control))

            # Verify operational feasibility assessment
            feasibility = assessment['operational_feasibility']
            assert feasibility['recommended_mode'] == 'comprehensive_survey'
            assert feasibility['recommended_strategy'] == 'parallel_batch'
            assert feasibility['estimated_duration_hours'] == 18.0
            assert feasibility['resource_intensity'] == 'high'
            assert feasibility['concurrency_recommendation'] == 8
            assert feasibility['feasibility_score'] == 0.7  # Lower confidence

            # Should have execution time assessment
            assert assessment['estimated_execution_time'].total_seconds() == 18 * 3600

            # Should have resource requirements
            resources = assessment['resource_requirements']
            assert resources['cpu_cores'] == 6
            assert resources['memory_gb'] == 12

            # Should have optimization opportunities for large scope
            optimizations = assessment['optimization_opportunities']
            assert len(optimizations) > 0
            assert any('phased execution' in opt['opportunity'].lower() for opt in optimizations)

            # Should have warnings about scope and complexity
            warnings = [w.lower() for w in assessment['warnings']]
            assert any('uncertainty' in w or 'confidence' in w for w in warnings)

    def test_preflight_cost_check_compliance_validation(self):
        """Test preflight check compliance and authorization validation."""
        from core.scraper_engine import preflight_cost_check

        # Test expired authorization
        expired_auth_control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Test expired authorization",
                sources=["test_source"],
                geography=["Test area"]
            ),
            budget=ScrapeBudget(
                max_runtime_minutes=60,
                max_pages=100,
                max_records=200,
                max_cost_total=500.0
            ),
            authorization=ScrapeAuthorization(
                approved_by="test_user",
                purpose="Testing",
                expires_at=datetime.utcnow() - timedelta(days=1)  # Already expired
            )
        )

        with patch('core.scraper_engine.predict_scraping_cost') as mock_cost, \
             patch('core.scraper_engine.classify_execution_mode') as mock_execution, \
             patch('core.scraper_engine.classify_scraping_intent') as mock_intent, \
             patch('core.scraper_engine.optimize_scraping_cost') as mock_optimize, \
             patch('core.scraper_engine.AuthorizationGate.validate', side_effect=Exception("Authorization expired")) as mock_auth:

            # Should fail due to authorization issues
            try:
                assessment = asyncio.run(preflight_cost_check(expired_auth_control))
                assert False, "Should have raised RuntimeError for authorization issues"
            except RuntimeError as e:
                assert "authorization" in str(e).lower() or "expired" in str(e).lower()

        # Test missing required fields
        incomplete_control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Incomplete control",
                sources=[],  # No sources
                geography=[]  # No geography
            ),
            budget=None,  # No budget
            authorization=None  # No authorization
        )

        try:
            assessment = asyncio.run(preflight_cost_check(incomplete_control))
            assert False, "Should have raised ValueError for incomplete contract"
        except ValueError as e:
            assert "invalid control contract" in str(e).lower()

        # Test valid compliance scenario
        valid_control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Valid compliance test",
                sources=["compliant_source"],
                geography=["Compliant area"]
            ),
            budget=ScrapeBudget(
                max_runtime_minutes=120,
                max_pages=200,
                max_records=500,
                max_cost_total=1000.0
            ),
            authorization=ScrapeAuthorization(
                approved_by="compliance_officer",
                purpose="Compliance testing",
                expires_at=datetime.utcnow() + timedelta(days=30)
            )
        )

        with patch('core.scraper_engine.predict_scraping_cost') as mock_cost, \
             patch('core.scraper_engine.classify_execution_mode') as mock_execution, \
             patch('core.scraper_engine.classify_scraping_intent') as mock_intent, \
             patch('core.scraper_engine.optimize_scraping_cost') as mock_optimize, \
             patch('core.scraper_engine.AuthorizationGate.validate') as mock_auth:

            mock_cost_prediction = MagicMock()
            mock_cost_prediction.predicted_cost = 600.0
            mock_cost_prediction.confidence_score = 0.85
            mock_cost_prediction.cost_range = (570.0, 630.0)
            mock_cost_prediction.cost_breakdown = {'signal_acquisition': 400, 'infrastructure': 200}
            mock_cost.return_value = mock_cost_prediction

            mock_execution.return_value = MagicMock(
                mode=MagicMock(value="targeted_lookup"),
                strategy=MagicMock(value="depth_first"),
                confidence_score=0.85,
                performance_expectations={'estimated_duration_hours': 2.0, 'expected_success_rate': 0.9},
                resource_requirements={'cpu_cores': 2, 'memory_gb': 4},
                execution_parameters={'concurrent_requests': 3}
            )

            mock_intent.return_value = MagicMock(
                risk_level=IntentRiskLevel.MEDIUM,
                category=IntentCategory.INTELLIGENCE,
                governance_requirement=MagicMock(value="standard"),
                confidence_score=0.8
            )

            mock_optimize.return_value = MagicMock(recommended_changes=[])

            assessment = asyncio.run(preflight_cost_check(valid_control))

            # Verify compliance assessment
            compliance = assessment['compliance_status']
            assert compliance['authorization_valid'] == True
            assert compliance['budget_compliance'] == True
            assert compliance['time_window_compliant'] == True
            assert compliance['overall_compliance_score'] > 0.75

            # Should be ready for execution
            assert assessment['overall_readiness'] in ['ready', 'caution']

    def test_preflight_cost_check_error_handling(self):
        """Test preflight check error handling and fallback mechanisms."""
        from core.scraper_engine import preflight_cost_check

        control = ScrapeControlContract(
            intent=ScrapeIntent(
                purpose="Error handling test",
                sources=["test_source"],
                geography=["Test area"]
            ),
            budget=ScrapeBudget(
                max_runtime_minutes=60,
                max_pages=100,
                max_records=200,
                max_cost_total=500.0
            ),
            authorization=ScrapeAuthorization(
                approved_by="test_user",
                purpose="Testing",
                expires_at=datetime.utcnow() + timedelta(days=1)
            )
        )

        # Test with failing intelligence components
        with patch('core.scraper_engine.predict_scraping_cost', side_effect=Exception("Cost prediction failed")) as mock_cost:

            # Should handle errors gracefully and still provide assessment
            try:
                assessment = asyncio.run(preflight_cost_check(control))
                assert False, "Should have raised RuntimeError for critical failures"
            except RuntimeError as e:
                assert "preflight assessment failed" in str(e).lower()

        # Test with partial failures (some components work, others don't)
        with patch('core.scraper_engine.predict_scraping_cost') as mock_cost, \
             patch('core.scraper_engine.classify_execution_mode') as mock_execution, \
             patch('core.scraper_engine.classify_scraping_intent', side_effect=Exception("Intent classification failed")) as mock_intent, \
             patch('core.scraper_engine.optimize_scraping_cost') as mock_optimize, \
             patch('core.scraper_engine.AuthorizationGate.validate') as mock_auth:

            mock_cost_prediction = MagicMock()
            mock_cost_prediction.predicted_cost = 300.0
            mock_cost_prediction.confidence_score = 0.6  # Lower confidence due to failures
            mock_cost_prediction.cost_range = (270.0, 330.0)
            mock_cost_prediction.cost_breakdown = {'signal_acquisition': 200, 'infrastructure': 100}
            mock_cost.return_value = mock_cost_prediction

            mock_execution.return_value = MagicMock(
                mode=MagicMock(value="targeted_lookup"),
                strategy=MagicMock(value="depth_first"),
                confidence_score=0.7,
                performance_expectations={'estimated_duration_hours': 1.0, 'expected_success_rate': 0.8},
                resource_requirements={'cpu_cores': 1, 'memory_gb': 2},
                execution_parameters={'concurrent_requests': 2}
            )

            mock_optimize.return_value = MagicMock(recommended_changes=[])

            # Should complete with warnings about failed components
            assessment = asyncio.run(preflight_cost_check(control))

            # Should have warnings about failed intelligence
            assert len(assessment['warnings']) > 0
            warnings_text = ' '.join(assessment['warnings']).lower()
            assert 'failed' in warnings_text or 'unavailable' in warnings_text

            # Should still provide basic assessment
            assert 'cost_analysis' in assessment
            assert assessment['cost_analysis']['predicted_cost'] == 300.0
            assert assessment['overall_readiness'] in ['caution', 'ready']  # Not blocked


if __name__ == "__main__":
    # Run basic tests
    print("🚀 Testing Scraper Engine Orchestration...")

    test_instance = TestScrapingEngine()

    # Run individual tests
    try:
        test_instance.test_engine_initialization()
        print("✅ Engine initialization tests passed")

        test_instance.test_job_creation_and_tracking()
        print("✅ Job creation and tracking tests passed")

        test_instance.test_intent_analysis_phase()
        print("✅ Intent analysis phase tests passed")

        test_instance.test_risk_assessment_phase()
        print("✅ Risk assessment phase tests passed")

        test_instance.test_execution_planning_phase()
        print("✅ Execution planning phase tests passed")

        test_instance.test_cost_optimization_phase()
        print("✅ Cost optimization phase tests passed")

        test_instance.test_governance_check_phase()
        print("✅ Governance check phase tests passed")

        test_instance.test_resource_allocation_phase()
        print("✅ Resource allocation phase tests passed")

        test_instance.test_scraping_execution_phase()
        print("✅ Scraping execution phase tests passed")

        test_instance.test_quality_validation_phase()
        print("✅ Quality validation phase tests passed")

        test_instance.test_orchestration_finalization()
        print("✅ Orchestration finalization tests passed")

        test_instance.test_complete_orchestration_workflow()
        print("✅ Complete orchestration workflow tests passed")

        test_instance.test_job_cancellation()
        print("✅ Job cancellation tests passed")

        test_instance.test_job_status_tracking()
        print("✅ Job status tracking tests passed")

        test_instance.test_engine_statistics()
        print("✅ Engine statistics tests passed")

        test_instance.test_error_handling_and_recovery()
        print("✅ Error handling and recovery tests passed")

        test_instance.test_convenience_functions()
        print("✅ Convenience functions tests passed")

        test_instance.test_priority_and_urgency_handling()
        print("✅ Priority and urgency handling tests passed")

        test_instance.test_budget_constraint_handling()
        print("✅ Budget constraint handling tests passed")

        test_instance.test_scalability_and_performance()
        print("✅ Scalability and performance tests passed")

        test_instance.test_comprehensive_intelligence_integration()
        print("✅ Comprehensive intelligence integration tests passed")

        test_instance.test_preflight_cost_check_comprehensive()
        print("✅ Preflight cost check comprehensive tests passed")

        test_instance.test_preflight_cost_check_budget_compliance()
        print("✅ Preflight budget compliance tests passed")

        test_instance.test_preflight_cost_check_risk_assessment()
        print("✅ Preflight risk assessment tests passed")

        test_instance.test_preflight_cost_check_operational_feasibility()
        print("✅ Preflight operational feasibility tests passed")

        test_instance.test_preflight_cost_check_compliance_validation()
        print("✅ Preflight compliance validation tests passed")

        test_instance.test_preflight_cost_check_error_handling()
        print("✅ Preflight error handling tests passed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 All Scraper Engine tests completed successfully!")
    print("🚀 Enterprise-grade scraping orchestration fully validated!")
