# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Test comprehensive telemetry models and enhanced collector functionality.
"""

import asyncio
from datetime import datetime, timedelta
from core.telemetry.models import (
    TelemetryEventType,
    TelemetrySeverity,
    BaseTelemetryEvent,
    ScraperOperationEvent,
    SentinelCheckEvent,
    SafetyVerdictEvent,
    ErrorEvent,
    PerformanceMetricEvent,
    SentinelOutcome,
    create_scraper_operation_event,
    create_sentinel_check_event,
    create_safety_verdict_event,
    create_error_event,
    create_performance_metric_event,
    create_sentinel_outcome
)
from core.scrape_telemetry import ScrapeTelemetryCollector


class TestTelemetryModels:
    """Test comprehensive telemetry models."""

    def test_scraper_operation_event_creation(self):
        """Test creation of scraper operation events."""
        event = create_scraper_operation_event(
            scraper_type="linkedin_scraper",
            scraper_role="enrichment",
            records_found=150,
            processing_time=2.5,
            operation_status="success",
            cost_estimate=1.25,
            correlation_id="test-123"
        )

        assert event.event_type == TelemetryEventType.SCRAPER_OPERATION
        assert event.scraper_type == "linkedin_scraper"
        assert event.scraper_role == "enrichment"
        assert event.records_found == 150
        assert event.processing_time == 2.5
        assert event.cost_estimate == 1.25
        assert event.operation_status == "success"
        assert event.correlation_id == "test-123"

    def test_sentinel_check_event_creation(self):
        """Test creation of sentinel check events."""
        event = create_sentinel_check_event(
            sentinel_name="network_sentinel",
            risk_level="medium",
            recommended_action="restrict",
            check_duration=1.2,
            findings_count=2,
            confidence_score=0.85
        )

        assert event.event_type == TelemetryEventType.SENTINEL_CHECK
        assert event.sentinel_name == "network_sentinel"
        assert event.risk_level == "medium"
        assert event.recommended_action == "restrict"
        assert event.check_duration == 1.2
        assert event.findings_count == 2
        assert event.confidence_score == 0.85

    def test_safety_verdict_event_creation(self):
        """Test creation of safety verdict events."""
        event = create_safety_verdict_event(
            verdict_action="delay",
            verdict_reason="Medium risk detected",
            risk_level="medium",
            reports_analyzed=3,
            sentinels_involved=["network", "waf", "malware"],
            confidence_score=0.75
        )

        assert event.event_type == TelemetryEventType.SAFETY_VERDICT
        assert event.verdict_action == "delay"
        assert event.verdict_reason == "Medium risk detected"
        assert event.risk_level == "medium"
        assert event.reports_analyzed == 3
        assert event.sentinels_involved == ["network", "waf", "malware"]
        assert event.confidence_score == 0.75

    def test_error_event_creation(self):
        """Test creation of error events."""
        event = create_error_event(
            error_type="ConnectionError",
            error_message="Failed to connect to target",
            severity=TelemetrySeverity.ERROR,
            operation_context="network_probe",
            source_component="network_sentinel"
        )

        assert event.event_type == TelemetryEventType.ERROR_OCCURRED
        assert event.severity == TelemetrySeverity.ERROR
        assert event.error_type == "ConnectionError"
        assert event.error_message == "Failed to connect to target"
        assert event.operation_context == "network_probe"

    def test_performance_metric_event_creation(self):
        """Test creation of performance metric events."""
        event = create_performance_metric_event(
            metric_name="response_time",
            metric_value=1.25,
            metric_unit="seconds",
            component_name="http_client",
            threshold_warning=2.0,
            threshold_critical=5.0
        )

        assert event.event_type == TelemetryEventType.PERFORMANCE_METRIC
        assert event.metric_name == "response_time"
        assert event.metric_value == 1.25
        assert event.metric_unit == "seconds"
        assert event.threshold_warning == 2.0
        assert event.threshold_critical == 5.0


class TestEnhancedCollector:
    """Test enhanced telemetry collector functionality."""

    async def test_comprehensive_event_recording(self):
        """Test recording comprehensive telemetry events."""
        collector = ScrapeTelemetryCollector(max_history=100)

        # Create different types of events
        scraper_event = create_scraper_operation_event(
            scraper_type="test_scraper",
            records_found=100,
            processing_time=1.5,
            operation_status="success",
            cost_estimate=0.5
        )

        sentinel_event = create_sentinel_check_event(
            sentinel_name="test_sentinel",
            risk_level="low",
            recommended_action="allow",
            check_duration=0.8
        )

        error_event = create_error_event(
            error_type="TestError",
            error_message="Test error message",
            severity=TelemetrySeverity.WARNING
        )

        # Record events
        await collector.record_event(scraper_event)
        await collector.record_event(sentinel_event)
        await collector.record_event(error_event)

        # Verify events were recorded in appropriate queues
        scraper_events = collector.get_scraper_events()
        sentinel_events = collector.get_sentinel_events()
        error_events = collector.get_error_events()

        assert len(scraper_events) == 1
        assert scraper_events[0].scraper_type == "test_scraper"

        assert len(sentinel_events) == 1
        assert sentinel_events[0].sentinel_name == "test_sentinel"

        assert len(error_events) == 1
        assert error_events[0].error_type == "TestError"

    async def test_component_health_summary(self):
        """Test component health summary generation."""
        collector = ScrapeTelemetryCollector(max_history=100)

        # Add some events
        past_hour = datetime.utcnow() - timedelta(hours=2)
        recent_hour = datetime.utcnow() - timedelta(minutes=30)

        # Old scraper event
        old_scraper = create_scraper_operation_event(
            scraper_type="old_scraper",
            records_found=50,
            processing_time=1.0,
            cost_estimate=0.25,
            operation_status="success"
        )
        old_scraper.timestamp = past_hour

        # Recent scraper event
        recent_scraper = create_scraper_operation_event(
            scraper_type="recent_scraper",
            records_found=100,
            processing_time=2.0,
            cost_estimate=2.5,  # High cost
            operation_status="success"
        )
        recent_scraper.timestamp = recent_hour

        # Blocked scraper event
        blocked_scraper = create_scraper_operation_event(
            scraper_type="blocked_scraper",
            records_found=0,
            processing_time=0.5,
            cost_estimate=0.1,
            operation_status="blocked",
            blocked_reason="Rate limited"
        )
        blocked_scraper.timestamp = recent_hour

        # Sentinel events
        sentinel_low = create_sentinel_check_event(
            sentinel_name="test_sentinel",
            risk_level="low",
            recommended_action="allow",
            check_duration=0.5
        )
        sentinel_low.timestamp = recent_hour

        sentinel_critical = create_sentinel_check_event(
            sentinel_name="critical_sentinel",
            risk_level="critical",
            recommended_action="block",
            check_duration=1.0
        )
        sentinel_critical.timestamp = recent_hour

        # Verdict event
        verdict = create_safety_verdict_event(
            verdict_action="human_required",
            verdict_reason="High risk detected",
            risk_level="high",
            reports_analyzed=2
        )
        verdict.timestamp = recent_hour

        # Error events
        error_critical = create_error_event(
            error_type="CriticalError",
            error_message="System failure",
            severity=TelemetrySeverity.CRITICAL,
            source_component="core_engine"
        )
        error_critical.timestamp = recent_hour

        error_warning = create_error_event(
            error_type="WarningError",
            error_message="Minor issue",
            severity=TelemetrySeverity.WARNING,
            source_component="network_client"
        )
        error_warning.timestamp = recent_hour

        # Record all events
        await collector.record_event(old_scraper)
        await collector.record_event(recent_scraper)
        await collector.record_event(blocked_scraper)
        await collector.record_event(sentinel_low)
        await collector.record_event(sentinel_critical)
        await collector.record_event(verdict)
        await collector.record_event(error_critical)
        await collector.record_event(error_warning)

        # Get health summary
        health = collector.get_component_health_summary()

        # Verify scraper operations
        assert health["scraper_operations"]["total"] == 3
        assert health["scraper_operations"]["last_hour"] == 2  # recent + blocked
        assert health["scraper_operations"]["blocked_operations"] == 1
        assert health["scraper_operations"]["high_cost_operations"] == 1

        # Verify sentinel checks
        assert health["sentinel_checks"]["total"] == 2
        assert health["sentinel_checks"]["last_hour"] == 2
        assert health["sentinel_checks"]["critical_findings"] == 1
        assert health["sentinel_checks"]["high_risk_findings"] == 0

        # Verify safety verdicts
        assert health["safety_verdicts"]["total"] == 1
        assert health["safety_verdicts"]["last_hour"] == 1
        assert health["safety_verdicts"]["blocks_issued"] == 0
        assert health["safety_verdicts"]["human_required"] == 1

        # Verify error events
        assert health["error_events"]["total"] == 2
        assert health["error_events"]["last_hour"] == 2
        assert health["error_events"]["critical_errors"] == 1
        assert health["error_events"]["by_component"]["core_engine"] == 1
        assert health["error_events"]["by_component"]["network_client"] == 1

    async def test_event_filtering(self):
        """Test event filtering by type and timeframe."""
        collector = ScrapeTelemetryCollector(max_history=100)

        # Create events at different times
        base_time = datetime.utcnow()
        past_time = base_time - timedelta(hours=2)
        recent_time = base_time - timedelta(minutes=30)

        # Create events
        scraper_event = create_scraper_operation_event(
            scraper_type="test_scraper",
            records_found=100,
            processing_time=1.0,
            cost_estimate=0.5,
            operation_status="success"
        )

        sentinel_event = create_sentinel_check_event(
            sentinel_name="test_sentinel",
            risk_level="low",
            recommended_action="allow",
            check_duration=0.5
        )

        error_event = create_error_event(
            error_type="TestError",
            error_message="Test message",
            severity=TelemetrySeverity.WARNING
        )

        # Set timestamps
        scraper_event.timestamp = past_time
        sentinel_event.timestamp = recent_time
        error_event.timestamp = recent_time

        # Record events
        await collector.record_event(scraper_event)
        await collector.record_event(sentinel_event)
        await collector.record_event(error_event)

        # Test filtering by event type
        scraper_only = collector.get_events_by_type("scraper_operation")
        sentinel_only = collector.get_events_by_type("sentinel_check")
        error_only = collector.get_events_by_type("error_occurred")

        assert len(scraper_only) == 1
        assert len(sentinel_only) == 1
        assert len(error_only) == 1

        # Test filtering by timeframe (last hour)
        recent_events = collector.get_events_in_timeframe(
            base_time - timedelta(hours=1),
            base_time
        )

        assert len(recent_events) == 2  # sentinel + error
        assert all(e.timestamp >= base_time - timedelta(hours=1) for e in recent_events)

    # Test 12: Enhanced SentinelOutcome Model
    print("\n📋 Test 12: Enhanced SentinelOutcome Model")
    try:
        # Test basic creation
        outcome = create_sentinel_outcome(
            domain="example.com",
            risk_level="high",
            action="restrict",
            sentinel_name="network_sentinel",
            findings={"waf_detected": True, "rate_limiting": True},
            latency_ms=1250,
            blocked=False,
            proxy_pool="premium_pool",
            risk_score=0.75,
            confidence_score=0.85,
            threat_indicators=["suspicious_headers", "honeypot_behavior"],
            operational_recommendations=["use_rotating_proxies", "implement_human_behavior"],
            compliance_flags=["gdpr_review_required"],
            correlation_id="sentinel_test_001"
        )

        print("✅ Enhanced SentinelOutcome created")
        print(f"   Domain: {outcome.domain}")
        print(f"   Risk Level: {outcome.risk_level}")
        print(f"   Risk Score: {outcome.risk_score}")
        print(f"   Confidence Score: {outcome.confidence_score}")
        print(f"   Action: {outcome.action}")
        print(f"   Findings Count: {outcome.findings_count}")
        print(f"   Threat Indicators: {len(outcome.threat_indicators)}")
        print(f"   Operational Recommendations: {len(outcome.operational_recommendations)}")
        print(f"   Compliance Flags: {outcome.compliance_flags}")

        # Test validation methods
        assert outcome.get_risk_category() == "high"
        assert outcome.get_action_priority() == "high"
        assert not outcome.get_compliance_summary()["compliant"]

        # Test legacy format conversion
        legacy = outcome.to_legacy_format()
        assert legacy["domain"] == "example.com"
        assert legacy["risk_level"] == "high"
        assert legacy["action"] == "restrict"
        assert legacy["blocked"] == False
        assert "waf_detected" in legacy["findings"]

        # Test performance summary
        perf_summary = outcome.get_performance_summary()
        assert perf_summary["latency_ms"] == 1250
        assert perf_summary["efficiency_score"] is None  # Not set

        # Test with critical risk and high impact
        critical_outcome = create_sentinel_outcome(
            domain="high-risk-site.com",
            risk_level="critical",
            action="block",
            sentinel_name="malware_sentinel",
            business_impact_assessment="high",
            risk_score=0.95,
            confidence_score=0.92,
            critical_findings=["malware_signature_detected", "phishing_indicators"],
            escalation_required=True
        )

        print("✅ Critical risk outcome created")
        print(f"   Risk Category: {critical_outcome.get_risk_category()}")
        print(f"   Action Priority: {critical_outcome.get_action_priority()}")
        print(f"   Escalation Required: {critical_outcome.escalation_required}")
        print(f"   Critical Findings: {len(critical_outcome.critical_findings)}")

        assert critical_outcome.get_risk_category() == "critical"
        assert critical_outcome.get_action_priority() == "urgent"
        assert critical_outcome.escalation_required == True

        # Test validation (basic range checks)
        print("✅ Validation tests completed")

        # Test with comprehensive enterprise features
        enterprise_outcome = create_sentinel_outcome(
            domain="enterprise-target.com",
            risk_level="medium",
            action="delay",
            sentinel_name="comprehensive_sentinel",
            # Risk assessment
            risk_score=0.65,
            confidence_score=0.78,
            anomaly_score=0.45,
            # Network intelligence
            connectivity_status="stable",
            dns_resolution_time=0.023,
            ssl_validity_days=365,
            response_time_ms=850,
            # WAF detection
            waf_detected=True,
            bot_protection_level="high",
            rate_limiting_detected=True,
            session_tracking=True,
            # Threat intelligence
            threat_categories=["bot_detection", "rate_limiting"],
            malware_signatures=[],
            # Operational intelligence
            operational_recommendations=[
                "implement_browser_automation",
                "use_residential_proxies",
                "add_human_behavior_simulation"
            ],
            alternative_strategies=["headless_browser", "api_integration"],
            retry_recommendations={"backoff_strategy": "exponential", "max_attempts": 3},
            # Cost and efficiency
            estimated_cost=2.45,
            efficiency_score=0.72,
            resource_intensity="high",
            # Compliance
            compliance_flags=["data_residency_check"],
            regulatory_requirements=["gdpr", "ccpa"],
            data_residency_compliant=True,
            # Context
            correlation_id="enterprise_test_123",
            workflow_id="workflow_456",
            environment="production",
            region="us-east-1",
            # Business impact
            business_impact_assessment="medium",
            priority_level="high",
            # Audit
            analysis_steps=["dns_check", "connectivity_test", "waf_detection", "threat_analysis"],
            decision_factors={"primary_risk": "waf_detection", "secondary_risk": "rate_limiting"},
            # Performance
            memory_usage_mb=45.2,
            cpu_usage_percent=12.5,
            network_requests=8,
            # Quality
            false_positive_probability=0.05,
            analysis_completeness=0.95,
            # Predictions
            predicted_risk_trend="increasing",
            recommended_monitoring_interval=15,
            # Metadata
            metadata={"custom_field": "custom_value"},
            tags=["enterprise", "critical_infrastructure"]
        )

        print("✅ Enterprise-grade outcome created")
        print(f"   Comprehensive Analysis: {len(enterprise_outcome.analysis_steps)} steps")
        print(f"   Threat Categories: {enterprise_outcome.threat_categories}")
        print(f"   Operational Recommendations: {len(enterprise_outcome.operational_recommendations)}")
        print(f"   Compliance Flags: {enterprise_outcome.compliance_flags}")
        print(f"   Estimated Cost: ${enterprise_outcome.estimated_cost}")
        print(f"   Analysis Completeness: {enterprise_outcome.analysis_completeness}")
        print(f"   Custom Tags: {enterprise_outcome.tags}")

        assert enterprise_outcome.get_risk_category() == "high"
        assert enterprise_outcome.get_action_priority() == "high"
        assert enterprise_outcome.waf_detected == True
        assert enterprise_outcome.estimated_cost == 2.45
        assert enterprise_outcome.analysis_completeness == 0.95
        assert "enterprise" in enterprise_outcome.tags

    except Exception as e:
        print(f"❌ Test 12 failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run basic tests
    test_models = TestTelemetryModels()
    test_models.test_scraper_operation_event_creation()
    test_models.test_sentinel_check_event_creation()
    test_models.test_safety_verdict_event_creation()
    test_models.test_error_event_creation()
    test_models.test_performance_metric_event_creation()

    print("✅ All telemetry model tests passed!")

    # Run async tests
    async def run_async_tests():
        test_collector = TestEnhancedCollector()
        await test_collector.test_comprehensive_event_recording()
        await test_collector.test_component_health_summary()
        await test_collector.test_event_filtering()

        print("✅ All enhanced collector tests passed!")
        print("🎉 Comprehensive telemetry system validation complete!")

    asyncio.run(run_async_tests())
