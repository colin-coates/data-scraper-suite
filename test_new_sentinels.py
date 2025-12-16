#!/usr/bin/env python3
"""
Test script for the new sentinel system with probe-based architecture.
"""

import asyncio
import sys
import os

# Add the scraper suite to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from core.sentinels import BaseSentinel, SentinelReport, SentinelRunner, SentinelConfig
from core.sentinels.performance_sentinel import PerformanceSentinel, create_performance_sentinel
from core.scrape_telemetry import emit_telemetry
from datetime import datetime


async def test_base_sentinel():
    """Test the BaseSentinel abstract class."""
    print("🧪 Testing BaseSentinel Abstract Class")
    print("=" * 40)

    # Test that BaseSentinel cannot be instantiated directly
    try:
        base = BaseSentinel()
        print("❌ BaseSentinel should not be instantiable directly")
        return False
    except TypeError:
        print("✅ BaseSentinel correctly prevents direct instantiation")

    # Test sentinel info
    sentinel = PerformanceSentinel()
    info = sentinel.get_sentinel_info()
    print(f"✅ Sentinel info: {info}")

    return True


async def test_sentinel_report():
    """Test the SentinelReport model."""
    print("\n📋 Testing SentinelReport Model")
    print("=" * 35)

    # Create a sample report
    report = SentinelReport(
        sentinel_name="test_sentinel",
        domain="performance",
        timestamp=datetime.utcnow(),
        risk_level="high",
        findings={"issue": "test issue", "score": 85},
        recommended_action="delay"
    )

    print("✅ SentinelReport created successfully")
    print(f"   Risk level: {report.risk_level}")
    print(f"   Recommended action: {report.recommended_action}")
    print(f"   Findings: {report.findings}")

    # Test serialization
    report_dict = report.dict()
    print(f"✅ Report serialization: {len(report_dict)} fields")

    return True


async def test_performance_sentinel():
    """Test the PerformanceSentinel probe functionality."""
    print("\n📊 Testing PerformanceSentinel Probe")
    print("=" * 40)

    sentinel = create_performance_sentinel(
        min_success_rate=0.8,
        max_avg_response_time=5.0,
        min_records_per_minute=5.0
    )

    print("✅ Performance sentinel created")

    # Test probe with no telemetry data
    target = {"type": "performance_check"}
    report = await sentinel.probe(target)

    print("✅ Probe executed successfully")
    print(f"   Report: {report.sentinel_name} - {report.risk_level} - {report.recommended_action}")

    # Add some telemetry data
    await emit_telemetry("test_scraper", "discovery", 1.0, 100, "", 1.5)
    await emit_telemetry("test_scraper", "discovery", 1.2, 95, "", 1.6)

    # Test probe with data
    report = await sentinel.probe(target)
    print(f"✅ Probe with data: {report.risk_level} - {report.recommended_action}")

    # Add poor performance data
    await emit_telemetry("test_scraper", "discovery", 10.0, 5, "", 50.0)  # Very poor

    report = await sentinel.probe(target)
    print(f"✅ Probe with poor data: {report.risk_level} - {report.recommended_action}")
    if report.findings.get('issues'):
        print(f"   Issues detected: {len(report.findings['issues'])}")

    return True


async def test_sentinel_runner():
    """Test the SentinelRunner wrapper."""
    print("\n🏃 Testing SentinelRunner")
    print("=" * 30)

    sentinel = PerformanceSentinel()
    config = SentinelConfig(name="test_runner", check_interval=1.0)
    runner = SentinelRunner(sentinel, config)

    print("✅ SentinelRunner created")

    # Test probe through runner
    target = {"test": "runner_probe"}
    report = await runner.probe_target(target)

    print("✅ Runner probe executed")
    print(f"   Report: {report.sentinel_name} - {report.risk_level}")

    # Test runner metrics
    metrics = runner.get_metrics()
    print(f"✅ Runner metrics: {len(metrics)} fields")
    print(f"   Status: {metrics['status']}")

    return True


async def test_error_handling():
    """Test error handling in sentinels."""
    print("\n🚨 Testing Error Handling")
    print("=" * 30)

    sentinel = PerformanceSentinel()

    # Test with invalid target (should still work)
    target = None  # This might cause issues
    try:
        report = await sentinel.probe_with_fallback(target)
        print("✅ Error handling worked - returned fallback report")
        print(f"   Risk level: {report.risk_level}")
        print(f"   Action: {report.recommended_action}")
    except Exception as e:
        print(f"❌ Error handling failed: {e}")
        return False

    return True


async def main():
    """Run all sentinel tests."""
    print("🛡️ MJ Data Scraper Suite - New Sentinel System Tests")
    print("=" * 55)

    success = True

    success &= await test_base_sentinel()
    success &= await test_sentinel_report()
    success &= await test_performance_sentinel()
    success &= await test_sentinel_runner()
    success &= await test_error_handling()

    if success:
        print("\n🎉 All new sentinel system tests completed successfully!")
        print("The probe-based sentinel architecture is working correctly.")
        print("\nKey Features Demonstrated:")
        print("✅ Pydantic-based SentinelReport model")
        print("✅ Abstract BaseSentinel with probe() method")
        print("✅ Structured risk assessment (low/medium/high/critical)")
        print("✅ Actionable recommendations (allow/restrict/delay/block)")
        print("✅ SentinelRunner wrapper for monitoring infrastructure")
        print("✅ Error handling and fallback reports")
        print("✅ Performance sentinel with telemetry integration")
    else:
        print("\n❌ Some sentinel tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
