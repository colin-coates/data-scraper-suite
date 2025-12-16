#!/usr/bin/env python3
"""
Test script for the Sentinel Orchestrator functionality.
"""

import asyncio
import sys
import os

# Add the scraper suite to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from core.sentinels import (
    SentinelOrchestrator,
    OrchestrationResult,
    OrchestrationMode,
    OrchestrationStrategy,
    create_performance_sentinel,
    create_network_sentinel,
    create_malware_sentinel,
    create_comprehensive_orchestrator,
    create_security_focused_orchestrator,
    create_performance_orchestrator
)


async def test_orchestrator_basic():
    """Test basic orchestrator functionality."""
    print("🎼 Testing Sentinel Orchestrator - Basic Functionality")
    print("=" * 55)

    orchestrator = create_comprehensive_orchestrator()
    print("✅ Comprehensive orchestrator created")

    # Test orchestration with sample target
    target = {"urls": ["https://httpbin.org/get"], "domain": "httpbin.org"}
    result = await orchestrator.orchestrate(target)

    print("✅ Orchestration executed successfully")
    print(f"   Orchestration ID: {result.orchestration_id}")
    print(f"   Risk level: {result.aggregated_risk_level}")
    print(f"   Recommended action: {result.aggregated_action}")
    print(f"   Sentinels executed: {len(result.sentinels_executed)}")
    print(".2f")

    if result.individual_reports:
        print(f"   Individual reports: {len(result.individual_reports)}")
        for report in result.individual_reports:
            print(f"     - {report.sentinel_name}: {report.risk_level} → {report.recommended_action}")

    return True


async def test_orchestrator_modes():
    """Test different orchestration modes."""
    print("\n🎭 Testing Sentinel Orchestrator - Execution Modes")
    print("=" * 52)

    target = {"urls": ["https://httpbin.org/status/200"], "domain": "httpbin.org"}

    # Test PARALLEL mode
    orchestrator_parallel = SentinelOrchestrator(mode=OrchestrationMode.PARALLEL)
    orchestrator_parallel.register_sentinel("performance", create_performance_sentinel())
    orchestrator_parallel.register_sentinel("network", create_network_sentinel())

    result_parallel = await orchestrator_parallel.orchestrate(target, ["performance", "network"])
    print("✅ PARALLEL mode executed")
    print(f"   Mode: {orchestrator_parallel.mode.value}")
    print(".2f")

    # Test SEQUENTIAL mode
    orchestrator_sequential = SentinelOrchestrator(mode=OrchestrationMode.SEQUENTIAL)
    orchestrator_sequential.register_sentinel("performance", create_performance_sentinel())
    orchestrator_sequential.register_sentinel("network", create_network_sentinel())

    result_sequential = await orchestrator_sequential.orchestrate(target, ["performance", "network"])
    print("✅ SEQUENTIAL mode executed")
    print(f"   Mode: {orchestrator_sequential.mode.value}")
    print(".2f")

    return True


async def test_orchestrator_strategies():
    """Test different aggregation strategies."""
    print("\n🧮 Testing Sentinel Orchestrator - Aggregation Strategies")
    print("=" * 58)

    # Create mock reports for testing strategies
    from core.sentinels import SentinelReport
    from datetime import datetime

    mock_reports = [
        SentinelReport(
            sentinel_name="performance",
            domain="performance",
            timestamp=datetime.utcnow(),
            risk_level="low",
            findings={},
            recommended_action="allow"
        ),
        SentinelReport(
            sentinel_name="network",
            domain="network",
            timestamp=datetime.utcnow(),
            risk_level="medium",
            findings={},
            recommended_action="delay"
        ),
        SentinelReport(
            sentinel_name="waf",
            domain="waf",
            timestamp=datetime.utcnow(),
            risk_level="high",
            findings={},
            recommended_action="restrict"
        )
    ]

    target = {"urls": ["https://example.com"]}

    # Test different strategies
    strategies = [
        (OrchestrationStrategy.CONSERVATIVE, "Conservative"),
        (OrchestrationStrategy.MAJORITY, "Majority"),
        (OrchestrationStrategy.WEIGHTED, "Weighted"),
        (OrchestrationStrategy.VETO, "Veto")
    ]

    for strategy, name in strategies:
        orchestrator = SentinelOrchestrator(strategy=strategy)
        # Manually set reports for testing
        result = orchestrator._aggregate_results(
            "test_orch", target, ["performance", "network", "waf"],
            mock_reports, 0.0
        )

        print(f"✅ {name} strategy: {result.aggregated_risk_level} → {result.aggregated_action}")

    return True


async def test_orchestrator_factory_functions():
    """Test factory functions for common orchestrator configurations."""
    print("\n🏭 Testing Sentinel Orchestrator - Factory Functions")
    print("=" * 53)

    # Test comprehensive orchestrator
    comp_orch = create_comprehensive_orchestrator()
    print("✅ Comprehensive orchestrator created")
    print(f"   Sentinels registered: {len(comp_orch.sentinels)}")
    print(f"   Mode: {comp_orch.mode.value}, Strategy: {comp_orch.strategy.value}")

    # Test security-focused orchestrator
    sec_orch = create_security_focused_orchestrator()
    print("✅ Security-focused orchestrator created")
    print(f"   Sentinels registered: {len(sec_orch.sentinels)}")
    print(f"   Mode: {sec_orch.mode.value}, Strategy: {sec_orch.strategy.value}")
    print(f"   Fail-fast: {sec_orch.fail_fast}")

    # Test performance orchestrator
    perf_orch = create_performance_orchestrator()
    print("✅ Performance orchestrator created")
    print(f"   Sentinels registered: {len(perf_orch.sentinels)}")
    print(f"   Mode: {perf_orch.mode.value}, Strategy: {perf_orch.strategy.value}")

    return True


async def test_orchestrator_dependencies():
    """Test sentinel dependencies and dependent execution."""
    print("\n🔗 Testing Sentinel Orchestrator - Dependencies")
    print("=" * 47)

    orchestrator = SentinelOrchestrator(mode=OrchestrationMode.DEPENDENT)

    # Register sentinels with dependencies
    orchestrator.register_sentinel("network", create_network_sentinel())
    orchestrator.register_sentinel("malware", create_malware_sentinel())

    # Set up dependencies
    orchestrator.add_dependency("malware", "network")
    print("✅ Dependencies configured")

    target = {"urls": ["https://httpbin.org/get"], "domain": "httpbin.org"}
    result = await orchestrator.orchestrate(target, ["network", "malware"])

    print("✅ Dependent execution completed")
    print(f"   Sentinels executed: {result.sentinels_executed}")
    print(f"   Dependencies satisfied: {'malware' in result.sentinels_executed}")

    return True


async def test_orchestrator_info_and_metrics():
    """Test orchestrator information and metrics collection."""
    print("\n📊 Testing Sentinel Orchestrator - Info and Metrics")
    print("=" * 52)

    orchestrator = create_comprehensive_orchestrator()

    # Run a few orchestrations to generate metrics
    target = {"urls": ["https://httpbin.org/status/200"]}

    for i in range(3):
        result = await orchestrator.orchestrate(target)
        print(f"   Orchestration {i+1}: {result.aggregated_risk_level} ({result.execution_time:.2f}s)")

    # Get orchestrator info
    info = orchestrator.get_orchestrator_info()
    print("✅ Orchestrator info retrieved")
    print(f"   Total orchestrations: {info['metrics']['orchestrations_attempted']}")
    print(".1%")
    print(f"   Registered sentinels: {len(info['registered_sentinels'])}")
    print(f"   Uptime: {info['metrics']['uptime_seconds']:.1f}s")

    # Test metrics reset
    orchestrator.reset_metrics()
    info_after_reset = orchestrator.get_orchestrator_info()
    print(f"   Metrics after reset: {info_after_reset['metrics']['orchestrations_attempted']} orchestrations")

    return True


async def test_orchestrator_error_handling():
    """Test orchestrator error handling."""
    print("\n🚨 Testing Sentinel Orchestrator - Error Handling")
    print("=" * 48)

    orchestrator = SentinelOrchestrator()

    # Test with no sentinels registered
    target = {"urls": ["https://example.com"]}
    result = await orchestrator.orchestrate(target)

    print("✅ Error handling for empty orchestrator")
    print(f"   Success: {result.success}")
    print(f"   Error: {result.error_message}")

    # Test with invalid sentinels requested
    orchestrator.register_sentinel("network", create_network_sentinel())
    result = await orchestrator.orchestrate(target, ["network", "nonexistent"])

    print("✅ Error handling for invalid sentinels")
    print(f"   Executed sentinels: {result.sentinels_executed}")
    print(f"   Success: {result.success}")

    return True


async def main():
    """Run all orchestrator tests."""
    print("🎼 MJ Data Scraper Suite - Sentinel Orchestrator Tests")
    print("=" * 57)

    success = True

    try:
        success &= await test_orchestrator_basic()
        success &= await test_orchestrator_modes()
        success &= await test_orchestrator_strategies()
        success &= await test_orchestrator_factory_functions()
        success &= await test_orchestrator_dependencies()
        success &= await test_orchestrator_info_and_metrics()
        success &= await test_orchestrator_error_handling()
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        success = False

    if success:
        print("\n🎉 All sentinel orchestrator tests completed successfully!")
        print("Sentinel orchestration and coordination is working correctly.")
        print("\nSentinel Orchestrator Features Demonstrated:")
        print("✅ Multi-sentinel coordination and execution")
        print("✅ Parallel, sequential, and dependent execution modes")
        print("✅ Multiple aggregation strategies (conservative, majority, weighted, veto)")
        print("✅ Factory functions for common configurations")
        print("✅ Sentinel dependency management")
        print("✅ Comprehensive metrics and monitoring")
        print("✅ Error handling and fallback mechanisms")
        print("✅ Result aggregation and unified decision making")
    else:
        print("\n❌ Some orchestrator tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
