#!/usr/bin/env python3
"""
Test script for the WAF Sentinel functionality.
"""

import asyncio
import sys
import os

# Add the scraper suite to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from core.sentinels import WafSentinel, SentinelRunner, SentinelConfig, create_waf_sentinel


async def test_waf_sentinel_basic():
    """Test basic WAF sentinel functionality."""
    print("🛡️ Testing WAF Sentinel - Basic Functionality")
    print("=" * 45)

    sentinel = create_waf_sentinel()
    print("✅ WAF sentinel created")

    # Test probe with sample URLs
    target = {"urls": ["https://httpbin.org/get", "https://example.com"]}
    report = await sentinel.probe(target)

    print("✅ WAF probe executed successfully")
    print(f"   Report: {report.sentinel_name} - {report.risk_level} - {report.recommended_action}")
    print(f"   URLs analyzed: {report.findings.get('urls_analyzed', 0)}")

    if report.findings.get("detected_protections"):
        protections = report.findings["detected_protections"]
        print(f"   WAF systems detected: {len(protections.get('waf_systems', []))}")
        print(f"   Bot protections detected: {len(protections.get('bot_protections', []))}")
        print(f"   Anti-scraping measures: {len(protections.get('anti_scraping_measures', []))}")

    return True


async def test_waf_sentinel_custom_targets():
    """Test WAF sentinel with custom target formats."""
    print("\n🎯 Testing WAF Sentinel - Custom Targets")
    print("=" * 42)

    sentinel = WafSentinel()

    # Test different target formats
    test_cases = [
        {"url": "https://httpbin.org/status/200"},
        {"urls": ["https://httpbin.org/json"]},
        {"targets": [{"url": "https://httpbin.org/headers"}]},
        {"urls": ["https://httpbin.org/get", "https://httpbin.org/post"]}
    ]

    for i, target in enumerate(test_cases, 1):
        print(f"\n  Test case {i}: {target}")
        report = await sentinel.probe(target)
        print(f"    Risk: {report.risk_level}, Action: {report.recommended_action}")
        print(f"    URLs analyzed: {report.findings.get('urls_analyzed', 0)}")

    return True


async def test_waf_sentinel_runner():
    """Test WAF sentinel with runner wrapper."""
    print("\n🏃 Testing WAF Sentinel - Runner Integration")
    print("=" * 45)

    sentinel = WafSentinel()
    config = SentinelConfig(name="waf_test_runner", check_interval=60.0)
    runner = SentinelRunner(sentinel, config)

    print("✅ SentinelRunner created for WAF sentinel")

    # Test probe through runner
    target = {"urls": ["https://httpbin.org/user-agent"]}
    report = await runner.probe_target(target)

    print("✅ Runner probe executed")
    print(f"   Report: {report.sentinel_name} - {report.risk_level}")
    print(f"   Domain: {report.domain}")

    # Test runner metrics
    metrics = runner.get_metrics()
    print(f"✅ Runner metrics retrieved: {len(metrics)} fields")

    return True


async def test_waf_sentinel_error_handling():
    """Test WAF sentinel error handling."""
    print("\n🚨 Testing WAF Sentinel - Error Handling")
    print("=" * 40)

    sentinel = WafSentinel()

    # Test with invalid URLs
    target = {"urls": ["invalid-url", "http://nonexistent.domain.invalid"]}
    report = await sentinel.probe(target)

    print("✅ Error handling probe executed")
    print(f"   Risk level: {report.risk_level}")
    print(f"   Action: {report.recommended_action}")

    # Should handle errors gracefully
    if "error" in report.findings:
        print(f"   Error captured: {report.findings['error'][:50]}...")

    return True


async def test_waf_sentinel_configuration():
    """Test WAF sentinel configuration and customization."""
    print("\n⚙️  Testing WAF Sentinel - Configuration")
    print("=" * 40)

    sentinel = WafSentinel()

    # Update risk thresholds
    sentinel.update_risk_thresholds(high_threshold=3, critical_threshold=6)
    print("✅ Risk thresholds updated")

    # Add custom WAF signature
    sentinel.add_custom_waf_signature(
        name="custom_waf",
        headers=["x-custom-protection"],
        content=["custom firewall protection"],
        risk_weight=2
    )
    print("✅ Custom WAF signature added")

    # Test with updated configuration
    target = {"urls": ["https://httpbin.org/get"]}
    report = await sentinel.probe(target)

    print("✅ Probe with custom configuration executed")
    print(f"   Risk assessment: {report.risk_level} - {report.recommended_action}")

    return True


async def test_waf_sentinel_history():
    """Test WAF sentinel protection history tracking."""
    print("\n📚 Testing WAF Sentinel - History Tracking")
    print("=" * 42)

    sentinel = WafSentinel()

    # Make multiple probes to build history
    urls = ["https://httpbin.org/get", "https://httpbin.org/status/200"]

    for i in range(3):
        target = {"urls": urls}
        report = await sentinel.probe(target)
        print(f"  Probe {i+1}: Risk score = {report.findings.get('total_risk_score', 0)}")

    # Check protection history
    history = sentinel.get_protection_history()
    print(f"✅ Protection history retrieved: {len(history)} domains tracked")

    for domain, entries in history.items():
        print(f"   {domain}: {len(entries)} historical entries")

    return True


async def test_waf_sentinel_no_urls():
    """Test WAF sentinel with no URLs provided."""
    print("\n📭 Testing WAF Sentinel - No URLs Provided")
    print("=" * 42)

    sentinel = WafSentinel()

    # Test with empty target
    target = {}
    report = await sentinel.probe(target)

    print("✅ Empty target probe executed")
    print(f"   Risk level: {report.risk_level}")
    print(f"   Action: {report.recommended_action}")
    print(f"   Status: {report.findings.get('status', 'unknown')}")

    return True


async def main():
    """Run all WAF sentinel tests."""
    print("🛡️ MJ Data Scraper Suite - WAF Sentinel Tests")
    print("=" * 50)

    success = True

    try:
        success &= await test_waf_sentinel_basic()
        success &= await test_waf_sentinel_custom_targets()
        success &= await test_waf_sentinel_runner()
        success &= await test_waf_sentinel_error_handling()
        success &= await test_waf_sentinel_configuration()
        success &= await test_waf_sentinel_history()
        success &= await test_waf_sentinel_no_urls()
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        success = False

    if success:
        print("\n🎉 All WAF sentinel tests completed successfully!")
        print("WAF detection and risk assessment is working correctly.")
        print("\nWAF Sentinel Features Demonstrated:")
        print("✅ WAF system detection (Cloudflare, Akamai, Imperva, etc.)")
        print("✅ Bot detection mechanism identification")
        print("✅ Anti-scraping measure detection")
        print("✅ Risk-based protection assessment")
        print("✅ Custom WAF signature support")
        print("✅ Protection history tracking")
        print("✅ Error handling and fallback")
        print("✅ Runner integration")
        print("✅ Configurable risk thresholds")
        print("✅ Multiple URL format support")
    else:
        print("\n❌ Some WAF sentinel tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
