#!/usr/bin/env python3
"""
Test Retry Utilities

Tests the retry logic with exponential backoff for resilient operations.
"""

import asyncio
import sys
import os
import time
from unittest.mock import Mock, AsyncMock, patch

# Add the scraper suite to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.retry_utils import (
    retry_async, retry_sync, RetryConfig, RetryResult, RetryError,
    retry_on_network_errors, retry_on_rate_limits, retry_call_async,
    _calculate_delay, _should_retry_exception
)


async def test_retry_config():
    """Test retry configuration."""
    print("🔄 Testing Retry Configuration")
    print("=" * 30)

    # Test 1: Default Configuration
    print("\n1️⃣ Testing Default Configuration:")
    config = RetryConfig()

    assert config.max_attempts == 3
    assert config.base_delay == 1.0
    assert config.backoff_factor == 2.0
    assert config.jitter == True
    assert ConnectionError in config.retry_on_exceptions

    print("✅ Default configuration works")

    # Test 2: Custom Configuration
    print("\n2️⃣ Testing Custom Configuration:")
    custom_config = RetryConfig(
        max_attempts=5,
        base_delay=2.0,
        max_delay=30.0,
        backoff_factor=1.5,
        jitter=False
    )

    assert custom_config.max_attempts == 5
    assert custom_config.base_delay == 2.0
    assert custom_config.max_delay == 30.0
    assert custom_config.jitter == False

    print("✅ Custom configuration works")


async def test_delay_calculation():
    """Test delay calculation with exponential backoff."""
    print("\n3️⃣ Testing Delay Calculation:")
    config = RetryConfig(base_delay=1.0, backoff_factor=2.0, max_delay=10.0, jitter=False)

    # Test exponential backoff
    delay1 = _calculate_delay(1, config)  # 1.0 * (2.0^0) = 1.0
    delay2 = _calculate_delay(2, config)  # 1.0 * (2.0^1) = 2.0
    delay3 = _calculate_delay(3, config)  # 1.0 * (2.0^2) = 4.0
    delay4 = _calculate_delay(4, config)  # 1.0 * (2.0^3) = 8.0
    delay5 = _calculate_delay(5, config)  # 1.0 * (2.0^4) = 16.0 -> capped at 10.0

    assert abs(delay1 - 1.0) < 0.1
    assert abs(delay2 - 2.0) < 0.1
    assert abs(delay3 - 4.0) < 0.1
    assert abs(delay4 - 8.0) < 0.1
    assert abs(delay5 - 10.0) < 0.1  # Capped at max_delay

    print("✅ Delay calculation works")


async def test_exception_filtering():
    """Test exception filtering for retries."""
    print("\n4️⃣ Testing Exception Filtering:")
    config = RetryConfig()

    # Test retryable exceptions
    assert _should_retry_exception(ConnectionError("Connection failed"), config) == True
    assert _should_retry_exception(TimeoutError("Timeout"), config) == True
    assert _should_retry_exception(OSError("Network error"), config) == True

    # Test non-retryable exceptions
    assert _should_retry_exception(ValueError("Invalid input"), config) == False
    assert _should_retry_exception(KeyError("Missing key"), config) == False

    # Test status code detection in exception messages
    class MockHTTPError(Exception):
        def __init__(self, message):
            super().__init__(message)

    assert _should_retry_exception(MockHTTPError("HTTP 429 Too Many Requests"), config) == True
    assert _should_retry_exception(MockHTTPError("HTTP 500 Internal Server Error"), config) == True
    assert _should_retry_exception(MockHTTPError("HTTP 200 OK"), config) == False

    print("✅ Exception filtering works")


async def test_async_retry_decorator():
    """Test async retry decorator."""
    print("\n5️⃣ Testing Async Retry Decorator:")

    call_count = 0

    @retry_async(RetryConfig(max_attempts=3, base_delay=0.1, jitter=False))
    async def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Temporary failure")
        return "success"

    # Test successful retry
    start_time = time.time()
    result = await failing_function()
    end_time = time.time()

    assert result == "success"
    assert call_count == 3  # Should have failed twice, succeeded on third try
    assert end_time - start_time >= 0.3  # Should have delays: 0.1 + 0.2 = 0.3

    print("✅ Async retry decorator works")


async def test_async_retry_failure():
    """Test async retry decorator failure."""
    print("\n6️⃣ Testing Async Retry Failure:")

    call_count = 0

    @retry_async(RetryConfig(max_attempts=2, base_delay=0.1, jitter=False))
    async def always_failing_function():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("Always fails")

    # Test complete failure
    try:
        await always_failing_function()
        assert False, "Should have raised RetryError"
    except RetryError as e:
        assert e.result.attempts == 2
        assert e.result.exception is not None
        assert len(e.result.attempt_details) == 2

    assert call_count == 2

    print("✅ Async retry failure handling works")


async def test_sync_retry_decorator():
    """Test sync retry decorator."""
    print("\n7️⃣ Testing Sync Retry Decorator:")

    call_count = 0

    @retry_sync(RetryConfig(max_attempts=3, base_delay=0.05, jitter=False))
    def failing_sync_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Temporary failure")
        return "success"

    # Test successful retry
    start_time = time.time()
    result = failing_sync_function()
    end_time = time.time()

    assert result == "success"
    assert call_count == 3
    assert end_time - start_time >= 0.15  # Should have delays: 0.05 + 0.1 = 0.15

    print("✅ Sync retry decorator works")


async def test_manual_retry_execution():
    """Test manual retry execution."""
    print("\n8️⃣ Testing Manual Retry Execution:")

    call_count = 0

    async def failing_async_func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("Temporary failure")
        return "success"

    config = RetryConfig(max_attempts=3, base_delay=0.1, jitter=False)

    # Test successful manual retry
    result = await retry_call_async(failing_async_func, config)

    assert result == "success"
    assert call_count == 2

    print("✅ Manual retry execution works")


async def test_predefined_decorators():
    """Test predefined retry decorators."""
    print("\n9️⃣ Testing Predefined Decorators:")

    call_count = 0

    @retry_on_network_errors(max_attempts=2)
    async def network_failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("Network failure")
        return "success"

    # Test network error retry
    result = await network_failing_function()
    assert result == "success"
    assert call_count == 2

    print("✅ Predefined decorators work")


async def test_jitter():
    """Test jitter functionality."""
    print("\n🔟 Testing Jitter Functionality:")

    config_with_jitter = RetryConfig(base_delay=1.0, jitter=True, jitter_factor=0.5, backoff_factor=1.0)
    config_without_jitter = RetryConfig(base_delay=1.0, jitter=False, backoff_factor=1.0)

    # Generate multiple delays to check jitter variation
    delays_with_jitter = [_calculate_delay(1, config_with_jitter) for _ in range(10)]
    delays_without_jitter = [_calculate_delay(1, config_without_jitter) for _ in range(10)]

    # Without jitter, all delays should be the same
    assert all(abs(d - 1.0) < 0.01 for d in delays_without_jitter)

    # With jitter, delays should vary
    has_variation = any(abs(d - 1.0) > 0.1 for d in delays_with_jitter)
    assert has_variation, "Jitter should introduce variation in delays"

    print("✅ Jitter functionality works")


async def test_retry_hooks():
    """Test retry hooks."""
    print("\n1️⃣1️⃣ Testing Retry Hooks:")

    success_called = False
    failure_called = False
    failure_exception = None
    failure_attempts = 0

    def success_hook(attempt):
        nonlocal success_called
        success_called = True

    def failure_hook(exception, attempts):
        nonlocal failure_called, failure_exception, failure_attempts
        failure_called = True
        failure_exception = exception
        failure_attempts = attempts

    call_count = 0

    @retry_async(RetryConfig(
        max_attempts=2,
        base_delay=0.1,
        jitter=False,
        success_hook=success_hook,
        failure_hook=failure_hook
    ))
    async def hook_test_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("Test failure")
        return "success"

    # Test success hook
    result = await hook_test_function()
    assert result == "success"
    assert success_called == True

    # Reset for failure test
    call_count = 0
    success_called = False

    @retry_async(RetryConfig(
        max_attempts=2,
        base_delay=0.1,
        jitter=False,
        success_hook=success_hook,
        failure_hook=failure_hook
    ))
    async def always_failing_function():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("Always fails")

    # Test failure hook
    try:
        await always_failing_function()
        assert False, "Should have failed"
    except RetryError:
        assert failure_called == True
        assert isinstance(failure_exception, ConnectionError)
        assert failure_attempts == 2

    print("✅ Retry hooks work")


async def test_retry_result():
    """Test RetryResult structure."""
    print("\n1️⃣2️⃣ Testing RetryResult Structure:")

    result = RetryResult(
        success=True,
        result="test_data",
        attempts=2,
        total_delay=0.3
    )

    assert result.success == True
    assert result.result == "test_data"
    assert result.attempts == 2
    assert result.total_delay == 0.3
    assert isinstance(result.attempt_details, list)

    print("✅ RetryResult structure works")


if __name__ == "__main__":
    async def main():
        try:
            # Run all tests
            await test_retry_config()
            await test_delay_calculation()
            await test_exception_filtering()
            await test_async_retry_decorator()
            await test_async_retry_failure()
            await test_sync_retry_decorator()
            await test_manual_retry_execution()
            await test_predefined_decorators()
            await test_jitter()
            await test_retry_hooks()
            await test_retry_result()

            print("\n🎉 All retry utility tests passed!")
            sys.exit(0)

        except Exception as e:
            print(f"\n💥 Retry utility test suite failed with exception: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    asyncio.run(main())
