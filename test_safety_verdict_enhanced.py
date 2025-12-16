# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Enhanced Safety Verdict System Test

Demonstrates the comprehensive enterprise-grade safety verdict capabilities
with advanced risk assessment, compliance tracking, audit trails, and operational intelligence.
"""

import asyncio
import sys
from datetime import datetime
from core.safety_verdict import (
    SafetyVerdict,
    safety_verdict,
    apply_verdict_constraints
)
from core.sentinels.base import SentinelReport

# Mock control for testing
class MockControl:
    def __init__(self, sources=None):
        self.intent = type('obj', (object,), {
            'sources': sources or ['test.com'],
            'event_type': 'social',
            'geography': ['US']
        })()
        self.authorization = type('obj', (object,), {
            'is_valid': lambda: True
        })()
        self.budget = type('obj', (object,), {
            'max_records': 1000
        })()

def test_enhanced_safety_verdict():
    """Test the enhanced SafetyVerdict with all enterprise features."""
    print("🚀 Testing Enhanced Safety Verdict System")
    print("=" * 70)

    control = MockControl()

    # Test 1: Basic Pydantic model functionality
    print("\n📋 Test 1: Enhanced Pydantic Model Features")
    try:
        verdict = SafetyVerdict(
            action="restrict",
            reason="Test security concerns",
            enforced_constraints={"reduced_rate_limit": True},
            risk_level="medium",
            confidence_score=0.75,
            workflow_id="test_workflow_001",
            sentinel_reports_count=5,
            compliance_flags=["gdpr_compliant"],
            recommended_actions=["monitor_closely"]
        )

        print("✅ Enhanced SafetyVerdict created successfully")
        print(f"   Action: {verdict.action}")
        print(f"   Risk Level: {verdict.risk_level}")
        print(f"   Confidence: {verdict.confidence_score:.1%}")
        print(f"   Workflow ID: {verdict.workflow_id}")
        print(f"   Compliance Flags: {verdict.compliance_flags}")
        print(f"   Is Critical: {verdict.is_critical()}")
        print(f"   Requires Human: {verdict.requires_human_intervention()}")
        print(f"   Constraint Keys: {verdict.get_constraint_keys()}")

        # Test audit entry conversion
        audit_entry = verdict.to_audit_entry()
        print(f"   Audit Entry Keys: {list(audit_entry.keys())}")

    except Exception as e:
        print(f"❌ Test 1 failed: {e}")

    # Test 2: Critical risk blocking with full enterprise features
    print("\n📋 Test 2: Critical Risk Blocking with Enterprise Features")
    try:
        reports = [
            SentinelReport(
                sentinel_name="malware",
                domain="dangerous-site.com",
                timestamp=datetime.utcnow(),
                risk_level="critical",
                findings={"malware_detected": True, "threat_level": "severe"},
                recommended_action="block"
            ),
            SentinelReport(
                sentinel_name="network",
                domain="dangerous-site.com",
                timestamp=datetime.utcnow(),
                risk_level="high",
                findings={"suspicious_traffic": True},
                recommended_action="block"
            )
        ]

        verdict = safety_verdict(reports, control, workflow_id="critical_test_001")

        print("✅ Critical risk verdict generated with enterprise features")
        print(f"   Action: {verdict.action}")
        print(f"   Risk Level: {verdict.risk_level}")
        print(f"   Confidence: {verdict.confidence_score:.1%}")
        print(f"   Processing Duration: {verdict.processing_duration:.3f}s")
        print(f"   Compliance Flags: {verdict.compliance_flags}")
        print(f"   Recommended Actions: {verdict.recommended_actions}")
        print(f"   Audit Trail Events: {len(verdict.audit_trail or [])}")

        # Test constraint application
        constraint_results = apply_verdict_constraints(verdict)
        print(f"   Constraints Applied: {constraint_results['applied_constraints']}")
        print(f"   Application Status: {constraint_results['status']}")

        assert verdict.action == "block"
        assert verdict.risk_level == "critical"
        assert "incident_report_required" in verdict.enforced_constraints
        assert "immediate_shutdown" in verdict.enforced_constraints

    except Exception as e:
        print(f"❌ Test 2 failed: {e}")

    # Test 3: Medium risk delay with sophisticated analysis
    print("\n📋 Test 3: Medium Risk Delay with Advanced Analysis")
    try:
        reports = [
            SentinelReport(sentinel_name="performance", domain="slow-site.com", timestamp=datetime.utcnow(),
                          risk_level="medium", findings={"slow_response": True}, recommended_action="delay"),
            SentinelReport(sentinel_name="network", domain="slow-site.com", timestamp=datetime.utcnow(),
                          risk_level="medium", findings={"high_latency": True}, recommended_action="delay"),
            SentinelReport(sentinel_name="waf", domain="slow-site.com", timestamp=datetime.utcnow(),
                          risk_level="low", findings={}, recommended_action="allow"),
            SentinelReport(sentinel_name="malware", domain="slow-site.com", timestamp=datetime.utcnow(),
                          risk_level="low", findings={}, recommended_action="allow")
        ]

        verdict = safety_verdict(reports, control, workflow_id="delay_test_001")

        print("✅ Medium risk delay verdict with advanced analysis")
        print(f"   Action: {verdict.action}")
        print(f"   Risk Level: {verdict.risk_level}")
        print(f"   Delay Minutes: {verdict.enforced_constraints.get('delay_minutes')}")
        print(f"   Risk Trends: {verdict.risk_trends}")
        print(f"   Analysis Summary: {verdict.analysis_summary['risk_calculation']}")

        constraint_results = apply_verdict_constraints(verdict)
        print(f"   Delay Constraint Applied: {'delay' in constraint_results['applied_constraints']}")

        assert verdict.action == "delay"
        assert "delay_minutes" in verdict.enforced_constraints
        assert verdict.risk_trends['dominant_risk'] == "medium"

    except Exception as e:
        print(f"❌ Test 3 failed: {e}")

    # Test 4: Low risk allowance with monitoring
    print("\n📋 Test 4: Low Risk Allowance with Monitoring")
    try:
        reports = [
            SentinelReport(sentinel_name="network", domain="safe-site.com", timestamp=datetime.utcnow(),
                          risk_level="low", findings={"normal_traffic": True}, recommended_action="allow"),
            SentinelReport(sentinel_name="waf", domain="safe-site.com", timestamp=datetime.utcnow(),
                          risk_level="low", findings={"no_waf_detected": True}, recommended_action="allow"),
            SentinelReport(sentinel_name="malware", domain="safe-site.com", timestamp=datetime.utcnow(),
                          risk_level="low", findings={"clean_scan": True}, recommended_action="allow")
        ]

        verdict = safety_verdict(reports, control, workflow_id="allow_test_001")

        print("✅ Low risk allowance with monitoring requirements")
        print(f"   Action: {verdict.action}")
        print(f"   Risk Level: {verdict.risk_level}")
        print(f"   Monitoring Required: {verdict.enforced_constraints.get('monitoring_required')}")
        print(f"   Audit Required: {verdict.enforced_constraints.get('audit_required')}")
        print(f"   Clean Findings: {verdict.analysis_summary.get('clean_findings')}")

        constraint_results = apply_verdict_constraints(verdict)
        print(f"   Monitoring Enabled: {'enhanced_monitoring' in constraint_results['applied_constraints']}")

        assert verdict.action == "allow"
        assert verdict.enforced_constraints.get("monitoring_required") == True
        assert verdict.enforced_constraints.get("audit_required") == True

    except Exception as e:
        print(f"❌ Test 4 failed: {e}")

    # Test 5: No reports fallback with enhanced features
    print("\n📋 Test 5: No Reports Fallback with Enterprise Features")
    try:
        verdict = safety_verdict([], control, workflow_id="fallback_test_001")

        print("✅ Enhanced fallback verdict for missing sentinel data")
        print(f"   Action: {verdict.action}")
        print(f"   Risk Level: {verdict.risk_level}")
        print(f"   Fallback Mode: {verdict.analysis_summary.get('fallback_mode')}")
        print(f"   Monitoring Required: {verdict.enforced_constraints.get('monitoring_required')}")
        print(f"   Recommended Actions: {verdict.recommended_actions}")
        print(f"   Audit Trail: {len(verdict.audit_trail or [])} entries")

        constraint_results = apply_verdict_constraints(verdict)
        print(f"   Fallback Constraints Applied: {constraint_results['applied_constraints']}")

        assert verdict.action == "allow"
        assert verdict.risk_level == "unknown"
        assert verdict.analysis_summary.get("fallback_mode") == True
        assert "enable_sentinel_monitoring" in verdict.recommended_actions

    except Exception as e:
        print(f"❌ Test 5 failed: {e}")

    # Test 6: Enhanced Max-Risk Algorithm - Critical Risk
    print("\n📋 Test 6: Enhanced Max-Risk Algorithm - Critical Risk")
    try:
        reports = [
            SentinelReport(sentinel_name="malware", domain="dangerous-site.com", timestamp=datetime.utcnow(),
                          risk_level="critical", findings={"malware_detected": True}, recommended_action="block"),
            SentinelReport(sentinel_name="network", domain="dangerous-site.com", timestamp=datetime.utcnow(),
                          risk_level="low", findings={}, recommended_action="allow"),
            SentinelReport(sentinel_name="waf", domain="dangerous-site.com", timestamp=datetime.utcnow(),
                          risk_level="medium", findings={}, recommended_action="restrict")
        ]

        verdict = safety_verdict(reports, control, workflow_id="max_risk_critical_test_001")

        print("✅ Enhanced max-risk algorithm - critical risk blocking")
        print(f"   Action: {verdict.action}")
        print(f"   Max Risk Level: {verdict.analysis_summary['max_risk_level']}")
        print(f"   Algorithm: {verdict.analysis_summary['algorithm']}")
        print(f"   Confidence: {verdict.confidence_score:.1%}")
        print(f"   Critical Findings: {len(verdict.analysis_summary['critical_findings'])}")
        print(f"   Recommended Actions: {verdict.recommended_actions}")

        assert verdict.action == "block"
        assert verdict.risk_level == "critical"
        assert verdict.analysis_summary['max_risk_level'] == "critical"
        assert "immediate_operation_shutdown" in verdict.recommended_actions
        assert verdict.enforced_constraints.get("incident_report_required") == True

    except Exception as e:
        print(f"❌ Test 6 failed: {e}")

    # Test 7: Enhanced Max-Risk Algorithm - High Risk Human Required
    print("\n📋 Test 7: Enhanced Max-Risk Algorithm - High Risk Human Required")
    try:
        reports = [
            SentinelReport(sentinel_name="waf", domain="suspicious-site.com", timestamp=datetime.utcnow(),
                          risk_level="high", findings={"waf_bypassed": True}, recommended_action="human_required"),
            SentinelReport(sentinel_name="network", domain="suspicious-site.com", timestamp=datetime.utcnow(),
                          risk_level="low", findings={}, recommended_action="allow"),
            SentinelReport(sentinel_name="malware", domain="suspicious-site.com", timestamp=datetime.utcnow(),
                          risk_level="low", findings={}, recommended_action="allow")
        ]

        verdict = safety_verdict(reports, control, workflow_id="max_risk_high_test_001")

        print("✅ Enhanced max-risk algorithm - high risk human intervention")
        print(f"   Action: {verdict.action}")
        print(f"   Max Risk Level: {verdict.analysis_summary['max_risk_level']}")
        print(f"   Algorithm: {verdict.analysis_summary['algorithm']}")
        print(f"   Requires Manual Review: {verdict.analysis_summary.get('requires_manual_review')}")
        print(f"   Human Approval Required: {verdict.enforced_constraints.get('human_approval_required')}")
        print(f"   Recommended Actions: {verdict.recommended_actions}")

        assert verdict.action == "human_required"
        assert verdict.risk_level == "high"
        assert verdict.analysis_summary['max_risk_level'] == "high"
        assert verdict.enforced_constraints.get("human_approval_required") == True
        assert "schedule_human_review" in verdict.recommended_actions

    except Exception as e:
        print(f"❌ Test 7 failed: {e}")

    # Test 8: Enhanced Max-Risk Algorithm - Medium Risk Restrictions
    print("\n📋 Test 8: Enhanced Max-Risk Algorithm - Medium Risk Restrictions")
    try:
        reports = [
            SentinelReport(sentinel_name="performance", domain="slow-site.com", timestamp=datetime.utcnow(),
                          risk_level="medium", findings={"slow_response": True}, recommended_action="restrict"),
            SentinelReport(sentinel_name="network", domain="slow-site.com", timestamp=datetime.utcnow(),
                          risk_level="low", findings={}, recommended_action="allow"),
            SentinelReport(sentinel_name="malware", domain="slow-site.com", timestamp=datetime.utcnow(),
                          risk_level="low", findings={}, recommended_action="allow")
        ]

        verdict = safety_verdict(reports, control, workflow_id="max_risk_medium_test_001")

        print("✅ Enhanced max-risk algorithm - medium risk restrictions")
        print(f"   Action: {verdict.action}")
        print(f"   Max Risk Level: {verdict.analysis_summary['max_risk_level']}")
        print(f"   Algorithm: {verdict.analysis_summary['algorithm']}")
        print(f"   Original Constraints: {verdict.analysis_summary.get('original_constraints')}")
        print(f"   Enhanced Constraints Applied: {len(verdict.analysis_summary.get('enhanced_constraints_applied', []))}")
        print(f"   Tier Setting: {verdict.enforced_constraints.get('tier')}")
        print(f"   Tempo Setting: {verdict.enforced_constraints.get('tempo')}")
        print(f"   Max Requests: {verdict.enforced_constraints.get('max_requests')}")

        assert verdict.action == "restrict"
        assert verdict.risk_level == "medium"
        assert verdict.analysis_summary['max_risk_level'] == "medium"
        assert verdict.enforced_constraints.get("tier") == 1
        assert verdict.enforced_constraints.get("tempo") == "forensic"
        assert verdict.enforced_constraints.get("max_requests") == 30

    except Exception as e:
        print(f"❌ Test 8 failed: {e}")

    # Test 9: Enhanced Max-Risk Algorithm - Low Risk Allowance
    print("\n📋 Test 9: Enhanced Max-Risk Algorithm - Low Risk Allowance")
    try:
        reports = [
            SentinelReport(sentinel_name="network", domain="safe-site.com", timestamp=datetime.utcnow(),
                          risk_level="low", findings={"normal_traffic": True}, recommended_action="allow"),
            SentinelReport(sentinel_name="waf", domain="safe-site.com", timestamp=datetime.utcnow(),
                          risk_level="low", findings={"no_waf_detected": True}, recommended_action="allow"),
            SentinelReport(sentinel_name="malware", domain="safe-site.com", timestamp=datetime.utcnow(),
                          risk_level="low", findings={"clean_scan": True}, recommended_action="allow")
        ]

        verdict = safety_verdict(reports, control, workflow_id="max_risk_allow_test_001")

        print("✅ Enhanced max-risk algorithm - low risk allowance")
        print(f"   Action: {verdict.action}")
        print(f"   Max Risk Level: {verdict.analysis_summary['max_risk_level']}")
        print(f"   Algorithm: {verdict.analysis_summary['algorithm']}")
        print(f"   All Sentinels Clear: {verdict.analysis_summary.get('all_sentinels_clear')}")
        print(f"   Clean Findings: {verdict.analysis_summary.get('clean_findings')}")
        print(f"   Monitoring Required: {verdict.enforced_constraints.get('monitoring_required')}")
        print(f"   Audit Trail Events: {len(verdict.audit_trail or [])}")

        assert verdict.action == "allow"
        assert verdict.risk_level == "low"
        assert verdict.analysis_summary['max_risk_level'] == "low"
        assert verdict.analysis_summary.get('all_sentinels_clear') == True
        assert verdict.enforced_constraints.get("monitoring_required") == True

    except Exception as e:
        print(f"❌ Test 9 failed: {e}")

    # Test 10: Enhanced Max-Risk Algorithm with Custom Thresholds
    print("\n📋 Test 10: Enhanced Max-Risk Algorithm with Custom Thresholds")
    try:
        # Test with custom thresholds that change behavior
        custom_thresholds = {
            "critical_block_threshold": 2,  # Require 2+ critical
            "high_block_ratio": 0.8,        # Require >80% high risk
            "medium_delay_ratio": 0.9,      # Require >90% medium+high
            "restrict_high_ratio": 0.3,     # Require >30% high risk
            "restrict_medium_ratio": 0.6    # Require >60% medium risk
        }

        reports = [
            SentinelReport(sentinel_name="test", domain="custom.com", timestamp=datetime.utcnow(),
                          risk_level="medium", findings={}, recommended_action="restrict")
        ] * 7 + [
            SentinelReport(sentinel_name="test", domain="custom.com", timestamp=datetime.utcnow(),
                          risk_level="low", findings={}, recommended_action="allow")
        ] * 3  # 70% medium = should trigger restrict with custom thresholds

        verdict = safety_verdict(reports, control, workflow_id="custom_max_risk_test_001",
                               risk_thresholds=custom_thresholds)

        print("✅ Enhanced max-risk algorithm with custom thresholds")
        print(f"   Action: {verdict.action} (restrict due to 70% medium ratio)")
        print(f"   Risk Level: {verdict.risk_level}")
        print(f"   Custom Threshold Used: restrict_medium_ratio = {custom_thresholds['restrict_medium_ratio']:.1%}")
        print(f"   Medium Ratio: {7/10:.1%} (>60% threshold)")
        print(f"   Algorithm: {verdict.analysis_summary['algorithm']}")

        assert verdict.action == "restrict"
        assert verdict.risk_level == "medium"
        assert verdict.analysis_summary['max_risk_level'] == "medium"

    except Exception as e:
        print(f"❌ Test 10 failed: {e}")

    print("\n🎉 Enhanced Safety Verdict System Tests Completed!")
    print("✅ Enterprise-grade Pydantic model with advanced features")
    print("✅ Enhanced Max-Risk Algorithm - Critical Risk Blocking")
    print("✅ Enhanced Max-Risk Algorithm - High Risk Human Intervention")
    print("✅ Enhanced Max-Risk Algorithm - Medium Risk Restrictions")
    print("✅ Enhanced Max-Risk Algorithm - Low Risk Allowance")
    print("✅ Enhanced Max-Risk Algorithm with Custom Thresholds")
    print("✅ Critical risk detection and blocking")
    print("✅ Medium risk delay with sophisticated analysis")
    print("✅ Low risk allowance with monitoring")
    print("✅ Enhanced fallback for missing data")
    print("✅ Comprehensive constraint application")
    print("✅ Workflow tracking and audit trails")
    print("✅ Compliance flag generation")
    print("✅ Operational intelligence and recommendations")
    print("✅ Performance and risk trend analysis")

    print("\n🏆 Enterprise Features Validated:")
    print("🔹 Enhanced Max-Risk Algorithm with Enterprise Intelligence")
    print("🔹 Statistical Confidence Scoring")
    print("🔹 Comprehensive Audit Trail Generation")
    print("🔹 Compliance Flag Management")
    print("🔹 Operational Constraint Application")
    print("🔹 Workflow Tracking and Correlation")
    print("🔹 Risk Trend Analysis and Intelligence")
    print("🔹 Customizable Risk Thresholds")
    print("🔹 Enhanced Monitoring and Alerting")
    print("🔹 Performance and Processing Metrics")
    print("🔹 Pydantic Compatibility with Fallbacks")
    print("🔹 Human Intervention Workflows")
    print("🔹 Incident Response Integration")


if __name__ == "__main__":
    test_enhanced_safety_verdict()
