# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Retry Utilities for MJ Data Scraper Suite

Implements retry logic with exponential backoff for resilient scraping operations.
Provides decorators and utilities for handling transient failures gracefully.
"""

import asyncio
import functools
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay between retries
    backoff_factor: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to prevent thundering herd
    jitter_factor: float = 0.1  # Jitter as fraction of delay
    retry_on_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
        asyncio.TimeoutError
    )
    retry_on_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504)
    success_hook: Optional[Callable[[int], None]] = None  # Called on successful retry
    failure_hook: Optional[Callable[[Exception, int], None]] = None  # Called on final failure


@dataclass
class RetryResult:
    """Result of a retry operation."""
    success: bool
    result: Any = None
    exception: Optional[Exception] = None
    attempts: int = 0
    total_delay: float = 0.0
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    attempt_details: List[Dict[str, Any]] = field(default_factory=list)


class RetryError(Exception):
    """Exception raised when all retry attempts are exhausted."""
    def __init__(self, message: str, result: RetryResult):
        super().__init__(message)
        self.result = result


def retry_async(config: Optional[RetryConfig] = None):
    """
    Decorator for async functions with retry logic and exponential backoff.

    Args:
        config: Retry configuration. If None, uses default config.

    Returns:
        Decorated async function with retry logic.
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await _execute_with_retry(func, config, *args, **kwargs)
        return wrapper
    return decorator


def retry_sync(config: Optional[RetryConfig] = None):
    """
    Decorator for sync functions with retry logic and exponential backoff.

    Args:
        config: Retry configuration. If None, uses default config.

    Returns:
        Decorated sync function with retry logic.
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return _execute_with_retry_sync(func, config, *args, **kwargs)
        return wrapper
    return decorator


async def _execute_with_retry(func: Callable[..., T], config: RetryConfig,
                             *args, **kwargs) -> T:
    """Execute async function with retry logic."""
    result = RetryResult(success=False)
    last_exception = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            attempt_start = time.time()

            # Execute the function
            result.result = await func(*args, **kwargs)

            # Success!
            result.success = True
            result.attempts = attempt
            result.end_time = datetime.utcnow()

            # Calculate total delay
            result.total_delay = sum(
                detail.get('delay', 0) for detail in result.attempt_details
            )

            # Call success hook if provided
            if config.success_hook:
                config.success_hook(attempt)

            logger.debug(f"Function {func.__name__} succeeded on attempt {attempt}")
            return result.result

        except Exception as e:
            last_exception = e
            result.exception = e
            attempt_duration = time.time() - attempt_start

            # Record attempt details
            attempt_detail = {
                'attempt': attempt,
                'exception': str(e),
                'exception_type': type(e).__name__,
                'duration': attempt_duration,
                'timestamp': datetime.utcnow().isoformat()
            }

            # Check if we should retry this exception
            should_retry = _should_retry_exception(e, config)

            # If this is not the last attempt and we should retry
            if attempt < config.max_attempts and should_retry:
                # Calculate delay
                delay = _calculate_delay(attempt, config)

                attempt_detail['delay'] = delay
                attempt_detail['will_retry'] = True

                result.attempt_details.append(attempt_detail)
                result.attempts = attempt

                logger.warning(f"Function {func.__name__} failed on attempt {attempt}: {e}. "
                             f"Retrying in {delay:.2f}s")

                await asyncio.sleep(delay)
            else:
                # Final failure
                attempt_detail['will_retry'] = False
                result.attempt_details.append(attempt_detail)
                result.attempts = attempt
                result.end_time = datetime.utcnow()

                # Calculate total delay
                result.total_delay = sum(
                    detail.get('delay', 0) for detail in result.attempt_details
                )

                # Call failure hook if provided
                if config.failure_hook:
                    config.failure_hook(e, attempt)

                logger.error(f"Function {func.__name__} failed after {attempt} attempts: {e}")

                # Raise RetryError with result details
                raise RetryError(
                    f"Function {func.__name__} failed after {attempt} attempts",
                    result
                ) from e

    # This should never be reached, but just in case
    raise RetryError(f"Unexpected error in retry logic for {func.__name__}", result)


def _execute_with_retry_sync(func: Callable[..., T], config: RetryConfig,
                            *args, **kwargs) -> T:
    """Execute sync function with retry logic."""
    result = RetryResult(success=False)
    last_exception = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            attempt_start = time.time()

            # Execute the function
            result.result = func(*args, **kwargs)

            # Success!
            result.success = True
            result.attempts = attempt
            result.end_time = datetime.utcnow()

            # Calculate total delay
            result.total_delay = sum(
                detail.get('delay', 0) for detail in result.attempt_details
            )

            # Call success hook if provided
            if config.success_hook:
                config.success_hook(attempt)

            logger.debug(f"Function {func.__name__} succeeded on attempt {attempt}")
            return result.result

        except Exception as e:
            last_exception = e
            result.exception = e
            attempt_duration = time.time() - attempt_start

            # Record attempt details
            attempt_detail = {
                'attempt': attempt,
                'exception': str(e),
                'exception_type': type(e).__name__,
                'duration': attempt_duration,
                'timestamp': datetime.utcnow().isoformat()
            }

            # Check if we should retry this exception
            should_retry = _should_retry_exception(e, config)

            # If this is not the last attempt and we should retry
            if attempt < config.max_attempts and should_retry:
                # Calculate delay
                delay = _calculate_delay(attempt, config)

                attempt_detail['delay'] = delay
                attempt_detail['will_retry'] = True

                result.attempt_details.append(attempt_detail)
                result.attempts = attempt

                logger.warning(f"Function {func.__name__} failed on attempt {attempt}: {e}. "
                             f"Retrying in {delay:.2f}s")

                time.sleep(delay)
            else:
                # Final failure
                attempt_detail['will_retry'] = False
                result.attempt_details.append(attempt_detail)
                result.attempts = attempt
                result.end_time = datetime.utcnow()

                # Calculate total delay
                result.total_delay = sum(
                    detail.get('delay', 0) for detail in result.attempt_details
                )

                # Call failure hook if provided
                if config.failure_hook:
                    config.failure_hook(e, attempt)

                logger.error(f"Function {func.__name__} failed after {attempt} attempts: {e}")

                # Raise RetryError with result details
                raise RetryError(
                    f"Function {func.__name__} failed after {attempt} attempts",
                    result
                ) from e

    # This should never be reached, but just in case
    raise RetryError(f"Unexpected error in retry logic for {func.__name__}", result)


def _calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for exponential backoff with jitter."""
    # Exponential backoff: base_delay * (backoff_factor ^ (attempt - 1))
    delay = config.base_delay * (config.backoff_factor ** (attempt - 1))

    # Cap at max_delay
    delay = min(delay, config.max_delay)

    # Add jitter if enabled
    if config.jitter:
        jitter_amount = delay * config.jitter_factor
        jitter = random.uniform(-jitter_amount, jitter_amount)
        delay = max(0.1, delay + jitter)  # Minimum 100ms delay

    return delay


def _should_retry_exception(exception: Exception, config: RetryConfig) -> bool:
    """Determine if an exception should trigger a retry."""
    # Check exception types
    if isinstance(exception, config.retry_on_exceptions):
        return True

    # Check for HTTP status codes in exception message/attributes
    exception_str = str(exception).lower()
    for status_code in config.retry_on_status_codes:
        if str(status_code) in exception_str:
            return True

    # Check for rate limiting indicators
    rate_limit_indicators = ['rate limit', 'too many requests', '429', 'retry-after']
    if any(indicator in exception_str for indicator in rate_limit_indicators):
        return True

    # Check for network-related errors
    network_errors = ['connection', 'timeout', 'network', 'dns', 'unreachable']
    if any(error in exception_str for error in network_errors):
        return True

    # Check for server errors
    server_errors = ['server error', 'internal server', 'bad gateway', 'service unavailable']
    if any(error in exception_str for error in server_errors):
        return True

    return False


# Convenience decorators with common configurations
def retry_on_network_errors(max_attempts: int = 3):
    """Decorator for retrying on network-related errors."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=1.0,
        retry_on_exceptions=(ConnectionError, TimeoutError, OSError, asyncio.TimeoutError)
    )
    return retry_async(config)


def retry_on_rate_limits(max_attempts: int = 5):
    """Decorator for retrying on rate limit errors."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=2.0,
        max_delay=300.0,  # 5 minutes max for rate limits
        backoff_factor=2.0,
        retry_on_exceptions=(),
        retry_on_status_codes=(429,)
    )
    return retry_async(config)


def retry_on_server_errors(max_attempts: int = 3):
    """Decorator for retrying on server errors."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=1.0,
        retry_on_status_codes=(500, 502, 503, 504)
    )
    return retry_async(config)


# Utility functions for manual retry execution
async def retry_call_async(func: Callable[..., T], config: Optional[RetryConfig] = None,
                          *args, **kwargs) -> T:
    """
    Manually execute an async function with retry logic.

    Args:
        func: Async function to execute
        config: Retry configuration
        *args, **kwargs: Arguments to pass to func

    Returns:
        Function result

    Raises:
        RetryError: If all retry attempts fail
    """
    if config is None:
        config = RetryConfig()

    return await _execute_with_retry(func, config, *args, **kwargs)


def retry_call_sync(func: Callable[..., T], config: Optional[RetryConfig] = None,
                    *args, **kwargs) -> T:
    """
    Manually execute a sync function with retry logic.

    Args:
        func: Sync function to execute
        config: Retry configuration
        *args, **kwargs: Arguments to pass to func

    Returns:
        Function result

    Raises:
        RetryError: If all retry attempts fail
    """
    if config is None:
        config = RetryConfig()

    return _execute_with_retry_sync(func, config, *args, **kwargs)


# Context manager for retry configuration
class RetryContext:
    """Context manager for retry configuration."""

    def __init__(self, config: RetryConfig):
        self.config = config
        self.results: List[RetryResult] = []

    async def execute_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute async function within retry context."""
        try:
            result = await retry_call_async(func, self.config, *args, **kwargs)
            return result
        except RetryError as e:
            self.results.append(e.result)
            raise

    def execute_sync(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute sync function within retry context."""
        try:
            result = retry_call_sync(func, self.config, *args, **kwargs)
            return result
        except RetryError as e:
            self.results.append(e.result)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about retry operations."""
        if not self.results:
            return {"total_operations": 0}

        total_attempts = sum(result.attempts for result in self.results)
        total_delay = sum(result.total_delay for result in self.results)
        success_rate = sum(1 for result in self.results if result.success) / len(self.results)

        return {
            "total_operations": len(self.results),
            "total_attempts": total_attempts,
            "total_delay": total_delay,
            "success_rate": success_rate,
            "average_attempts": total_attempts / len(self.results),
            "average_delay": total_delay / len(self.results)
        }
