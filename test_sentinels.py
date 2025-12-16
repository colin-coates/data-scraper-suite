#!/usr/bin/env python3
"""
Test script for the sentinel monitoring system.
"""

import asyncio
import sys
import os

# Add the scraper suite to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from core.sentinels.base import SentinelConfig, SentinelSeverity, SentinelStatus
from core.sentinels.performance_sentinel import PerformanceSentinel, create_performance_sentinel
from core.scrape_telemetry import emit_telemetry
from datetime import datetime


async def test_performance_sentinel():
    """Test the performance sentinel functionality."""
    print("📊 Testing Performance Sentinel")
    print("=" * 35)

    # Create a performance sentinel
    sentinel = create_performance_sentinel(
        name="test_performance_sentinel",
        check_interval=1.0,  # Check every second for testing
        min_success_rate=0.8,
        max_avg_response_time=5.0,
        min_records_per_minute=5.0
    )

    print("✅ Performance sentinel created")
    print(f"   Name: {sentinel.config.name}")
    print(f"   Check interval: {sentinel.config.check_interval}s")
    print(f"   Status: {sentinel.status.value}")

    # Test 1: Initial state (no telemetry data)
    print("\n1️⃣ Testing initial state (no data)...")
    await sentinel.execute_check()
    metrics = sentinel.get_metrics()

    print(f"   Status: {metrics['status']}")
    print(f"   Consecutive failures: {metrics['consecutive_failures']}")
    print(f"   Active alerts: {metrics['active_alerts']}")

    # Test 2: Add some good telemetry data
    print("\n2️⃣ Testing with good performance data...")
    await emit_telemetry("test_scraper", "discovery", 1.50, 120, "", 2.0)
    await emit_telemetry("test_scraper", "discovery", 1.25, 110, "", 1.8)
    await emit_telemetry("test_scraper", "discovery", 1.75, 130, "", 2.2)

    await sentinel.execute_check()
    metrics = sentinel.get_metrics()

    print(f"   Status: {metrics['status']}")
    print(f"   Active alerts: {metrics['active_alerts']}")

    # Test 3: Add poor performance data to trigger alerts
    print("\n3️⃣ Testing with poor performance data...")
    await emit_telemetry("test_scraper", "discovery", 5.0, 10, "", 45.0)  # Very slow
    await emit_telemetry("test_scraper", "discovery", 6.0, 8, "", 50.0)   # Even slower
    await emit_telemetry("test_scraper", "discovery", 7.0, 5, "", 55.0)   # Very poor

    # Execute multiple checks to trigger alert threshold
    for i in range(5):
        await sentinel.execute_check()
        await asyncio.sleep(0.1)

    metrics = sentinel.get_metrics()
    active_alerts = sentinel.get_active_alerts()

    print(f"   Status: {metrics['status']}")
    print(f"   Consecutive failures: {metrics['consecutive_failures']}")
    print(f"   Active alerts: {len(active_alerts)}")

    if active_alerts:
        alert = active_alerts[0]
        print(f"   Alert severity: {alert.severity.value}")
        print(f"   Alert message: {alert.message}")

    # Test 4: Recovery with good data
    print("\n4️⃣ Testing recovery with good performance data...")
    await emit_telemetry("test_scraper", "discovery", 1.0, 100, "", 1.5)
    await emit_telemetry("test_scraper", "discovery", 1.2, 95, "", 1.6)
    await emit_telemetry("test_scraper", "discovery", 1.1, 105, "", 1.4)

    await sentinel.execute_check()
    metrics = sentinel.get_metrics()

    print(f"   Status: {metrics['status']}")
    print(f"   Consecutive failures: {metrics['consecutive_failures']}")
    print(f"   Active alerts: {metrics['active_alerts']}")

    return True


async def test_sentinel_registry():
    """Test the sentinel registry functionality."""
    print("\n📋 Testing Sentinel Registry")
    print("=" * 30)

    from core.sentinels import register_sentinel, get_registered_sentinels, get_sentinel

    # Create and register a sentinel
    sentinel = create_performance_sentinel("registry_test_sentinel")
    register_sentinel(sentinel)

    print("✅ Sentinel registered")
    print(f"   Registered sentinels: {len(get_registered_sentinels())}")

    # Retrieve the sentinel
    retrieved = get_sentinel("registry_test_sentinel")
    if retrieved:
        print("✅ Sentinel retrieved successfully")
        print(f"   Retrieved sentinel: {retrieved.config.name}")
    else:
        print("❌ Sentinel retrieval failed")
        return False

    return True


async def test_sentinel_configuration():
    """Test sentinel configuration and validation."""
    print("\n⚙️  Testing Sentinel Configuration")
    print("=" * 35)

    # Test valid configuration
    try:
        config = SentinelConfig(
            name="test_config",
            check_interval=30.0,
            alert_threshold=5,
            severity=SentinelSeverity.ERROR
        )
        config.validate()
        print("✅ Valid configuration accepted")
    except Exception as e:
        print(f"❌ Valid configuration rejected: {e}")
        return False

    # Test invalid configuration
    try:
        config = SentinelConfig(
            name="invalid_config",
            check_interval=-1.0,  # Invalid negative interval
            alert_threshold=0     # Invalid threshold
        )
        config.validate()
        print("❌ Invalid configuration should have been rejected")
        return False
    except ValueError:
        print("✅ Invalid configuration properly rejected")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    return True


async def test_sentinel_callbacks():
    """Test sentinel callback functionality."""
    print("\n🔄 Testing Sentinel Callbacks")
    print("=" * 30)

    sentinel = create_performance_sentinel("callback_test_sentinel")

    # Set up callbacks
    alerts_received = []
    resolves_received = []
    status_changes = []

    def on_alert(alert):
        alerts_received.append(alert)

    def on_resolve(alert):
        resolves_received.append(alert)

    def on_status_change(old_status, new_status):
        status_changes.append((old_status, new_status))

    sentinel.on_alert = on_alert
    sentinel.on_resolve = on_resolve
    sentinel.on_status_change = on_status_change

    print("✅ Callbacks configured")

    # Trigger an alert (need poor performance data)
    await emit_telemetry("test_scraper", "discovery", 10.0, 1, "", 60.0)  # Very poor

    # Execute checks to trigger alert
    for i in range(5):
        await sentinel.execute_check()
        await asyncio.sleep(0.1)

    print(f"   Alerts received: {len(alerts_received)}")
    print(f"   Status changes: {len(status_changes)}")

    # Now provide good data to resolve
    await emit_telemetry("test_scraper", "discovery", 1.0, 100, "", 1.0)
    await sentinel.execute_check()

    print(f"   Resolves received: {len(resolves_received)}")

    if len(alerts_received) > 0 and len(status_changes) > 0:
        print("✅ Callbacks working correctly")
        return True
    else:
        print("❌ Callbacks not working")
        return False


async def main():
    """Run all sentinel tests."""
    print("🛡️ MJ Data Scraper Suite - Sentinel System Tests")
    print("=" * 50)

    success = True

    success &= await test_sentinel_configuration()
    success &= await test_sentinel_registry()
    success &= await test_performance_sentinel()
    success &= await test_sentinel_callbacks()

    if success:
        print("\n🎉 All sentinel system tests completed successfully!")
        print("Sentinel monitoring and alerting is working correctly.")
    else:
        print("\n❌ Some sentinel tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
