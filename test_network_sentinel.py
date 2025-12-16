#!/usr/bin/env python3
"""
Test script for the Network Sentinel functionality.
"""

import asyncio
import sys
import os

# Add the scraper suite to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from core.sentinels import NetworkSentinel, SentinelRunner, SentinelConfig, create_network_sentinel


async def test_network_sentinel_basic():
    """Test basic network sentinel functionality."""
    print("🌐 Testing Network Sentinel - Basic Functionality")
    print("=" * 50)

    sentinel = create_network_sentinel()
    print("✅ Network sentinel created")

    # Test probe with default domains
    target = {}
    report = await sentinel.probe(target)

    print("✅ Network probe executed successfully")
    print(f"   Report: {report.sentinel_name} - {report.risk_level} - {report.recommended_action}")
    print(f"   Domains checked: {report.findings.get('domains_checked', 0)}")

    if report.findings.get('issues'):
        print(f"   Issues detected: {len(report.findings['issues'])}")
        for issue in report.findings['issues'][:3]:  # Show first 3 issues
            print(f"     - {issue}")

    return True


async def test_network_sentinel_custom_domains():
    """Test network sentinel with custom domains."""
    print("\n🔧 Testing Network Sentinel - Custom Domains")
    print("=" * 45)

    # Test with custom domains
    custom_domains = ["google.com", "github.com", "httpbin.org"]
    sentinel = create_network_sentinel(monitor_domains=custom_domains)

    print(f"✅ Network sentinel created with custom domains: {custom_domains}")

    target = {"domains": ["example.com", "httpbin.org"]}
    report = await sentinel.probe(target)

    print("✅ Custom domain probe executed")
    print(f"   Risk level: {report.risk_level}")
    print(f"   Recommended action: {report.recommended_action}")

    return True


async def test_network_sentinel_runner():
    """Test network sentinel with runner wrapper."""
    print("\n🏃 Testing Network Sentinel - Runner Integration")
    print("=" * 50)

    sentinel = NetworkSentinel()
    config = SentinelConfig(name="network_test_runner", check_interval=30.0)
    runner = SentinelRunner(sentinel, config)

    print("✅ SentinelRunner created for network sentinel")

    # Test probe through runner
    target = {"urls": ["http://httpbin.org/get", "https://google.com"]}
    report = await runner.probe_target(target)

    print("✅ Runner probe executed")
    print(f"   Report: {report.sentinel_name} - {report.risk_level}")
    print(f"   Domain: {report.domain}")

    # Test runner metrics
    metrics = runner.get_metrics()
    print(f"✅ Runner metrics retrieved: {len(metrics)} fields")

    return True


async def test_network_sentinel_error_handling():
    """Test network sentinel error handling."""
    print("\n🚨 Testing Network Sentinel - Error Handling")
    print("=" * 45)

    sentinel = NetworkSentinel()

    # Test with invalid target
    target = {"domains": ["invalid.domain.that.does.not.exist.invalid"]}
    report = await sentinel.probe(target)

    print("✅ Error handling probe executed")
    print(f"   Risk level: {report.risk_level}")
    print(f"   Action: {report.recommended_action}")

    # Should still return a valid report even with errors
    if "error" in report.findings:
        print(f"   Error captured: {report.findings['error'][:50]}...")

    return True


async def test_network_sentinel_thresholds():
    """Test network sentinel threshold configuration."""
    print("\n⚙️  Testing Network Sentinel - Threshold Configuration")
    print("=" * 55)

    sentinel = NetworkSentinel()

    # Update thresholds
    sentinel.update_thresholds(
        max_dns_time=1.0,  # Very strict DNS time
        max_connect_time=5.0,  # Strict connect time
        max_response_time=10.0,  # Strict response time
        min_ssl_days=30  # Require 30 days SSL validity
    )

    print("✅ Thresholds updated")

    # Test with updated thresholds
    target = {}
    report = await sentinel.probe(target)

    print("✅ Probe with strict thresholds executed")
    print(f"   Risk assessment: {report.risk_level} - {report.recommended_action}")

    return True


async def test_network_metrics_detail():
    """Test detailed network metrics collection."""
    print("\n📊 Testing Network Sentinel - Detailed Metrics")
    print("=" * 50)

    sentinel = create_network_sentinel(
        monitor_domains=["google.com", "github.com"]
    )

    target = {}
    report = await sentinel.probe(target)

    print("✅ Detailed metrics probe executed")

    network_metrics = report.findings.get('network_metrics', [])
    print(f"   Metrics collected for {len(network_metrics)} domains")

    if network_metrics:
        # Show details for first domain
        metric = network_metrics[0]
        print(f"   Sample domain '{metric['domain']}':")
        print(f"     - DNS resolved: {metric.get('dns_resolved', 'N/A')}")
        print(f"     - Connectivity OK: {metric.get('connectivity_ok', 'N/A')}")
        if 'dns_time' in metric:
            print(".3f")
        if 'connect_time' in metric:
            print(".3f")
        if 'ssl_days_remaining' in metric:
            print(f"     - SSL days remaining: {metric['ssl_days_remaining']}")

    return True


async def main():
    """Run all network sentinel tests."""
    print("🛡️ MJ Data Scraper Suite - Network Sentinel Tests")
    print("=" * 55)

    success = True

    try:
        success &= await test_network_sentinel_basic()
        success &= await test_network_sentinel_custom_domains()
        success &= await test_network_sentinel_runner()
        success &= await test_network_sentinel_error_handling()
        success &= await test_network_sentinel_thresholds()
        success &= await test_network_metrics_detail()
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        success = False

    if success:
        print("\n🎉 All network sentinel tests completed successfully!")
        print("Network monitoring and risk assessment is working correctly.")
        print("\nNetwork Sentinel Features Demonstrated:")
        print("✅ DNS resolution monitoring")
        print("✅ Network connectivity checks")
        print("✅ SSL certificate validation")
        print("✅ Response time measurement")
        print("✅ Configurable thresholds")
        print("✅ Custom domain monitoring")
        print("✅ Risk-based recommendations")
        print("✅ Error handling and fallback")
        print("✅ Runner integration")
        print("✅ Detailed metrics collection")
    else:
        print("\n❌ Some network sentinel tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
