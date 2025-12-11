# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Base Scraper Class for MJ Data Scraper Suite

Provides common functionality for all scrapers including error handling,
logging hooks, proxy rotation, and rate-limit callbacks.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .retry_utils import retry_async, RetryConfig, retry_on_network_errors, retry_on_rate_limits

logger = logging.getLogger(__name__)


@dataclass
class ScraperResult:
    """Standardized result structure for all scrapers."""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    scraper_name: str = ""
    target_url: str = ""
    response_time: float = 0.0
    retry_count: int = 0


@dataclass
class ScraperConfig:
    """Configuration for scraper instances."""
    name: str
    user_agent_rotation: bool = True
    proxy_rotation: bool = True
    rate_limit_delay: float = 1.0
    max_retries: int = 3
    timeout: int = 30
    headers: Dict[str, str] = field(default_factory=dict)
    cookies: Dict[str, str] = field(default_factory=dict)
    proxies: List[str] = field(default_factory=list)
    retry_config: Optional[RetryConfig] = None  # Custom retry configuration
    enable_retry: bool = True  # Enable retry logic


class BaseScraper(ABC):
    """Base class for all scrapers with common functionality."""

    def __init__(self, config: ScraperConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self._session_cookies = {}
        self._rate_limit_until = 0.0
        self._error_count = 0
        self._success_count = 0

        # Callbacks
        self.on_error: Optional[Callable[[Exception, Dict[str, Any]], None]] = None
        self.on_rate_limit: Optional[Callable[[float], None]] = None
        self.on_success: Optional[Callable[[ScraperResult], None]] = None
        self.on_retry: Optional[Callable[[int, Exception], None]] = None

        # Anti-detection integration
        self.anti_detection = None

    async def scrape(self, target: Dict[str, Any]) -> ScraperResult:
        """
        Main scraping method with error handling and retries.

        Args:
            target: Target information (URL, ID, search terms, etc.)

        Returns:
            ScraperResult with scraping outcome
        """
        start_time = time.time()
        result = ScraperResult(
            success=False,
            scraper_name=self.config.name,
            target_url=target.get('url', target.get('target', ''))
        )

        try:
            # Check rate limiting
            await self._check_rate_limit()

            # Pre-scrape hooks
            await self._pre_scrape_hook(target)

            # Execute scrape with retries
            result = await self._scrape_with_retries(target, result)

            # Post-scrape hooks
            await self._post_scrape_hook(result)

            # Update metrics
            result.response_time = time.time() - start_time

            if result.success:
                self._success_count += 1
                if self.on_success:
                    self.on_success(result)
            else:
                self._error_count += 1

        except Exception as e:
            result.error_message = str(e)
            result.response_time = time.time() - start_time
            self._error_count += 1

            self.logger.error(f"Scraping failed for target {target}: {e}")

            if self.on_error:
                self.on_error(e, target)

        return result

    async def _scrape_with_retries(self, target: Dict[str, Any], result: ScraperResult) -> ScraperResult:
        """Execute scraping with retry logic using the retry utility."""
        if not self.config.enable_retry:
            # Fallback to single attempt if retry is disabled
            try:
                result.retry_count = 0
                scrape_result = await self._execute_scrape(target)

                if await self._validate_result(scrape_result):
                    result.success = True
                    result.data = scrape_result
                    return result
                else:
                    result.error_message = "Result validation failed"
                    return result

            except Exception as e:
                result.error_message = str(e)
                return result

        # Use retry utility for enhanced retry logic
        retry_config = self.config.retry_config or RetryConfig(
            max_attempts=self.config.max_retries + 1,
            base_delay=self.config.rate_limit_delay,
            max_delay=60.0,  # Max 1 minute between retries
            backoff_factor=2.0,
            success_hook=lambda attempt: self.logger.info(f"Scrape succeeded on attempt {attempt}"),
            failure_hook=lambda e, attempt: self.logger.error(f"Scrape failed after {attempt} attempts: {e}")
        )

        # Create retry-enabled scrape function
        @retry_async(retry_config)
        async def scrape_with_retry():
            # Execute the actual scraping logic
            scrape_result = await self._execute_scrape(target)

            # Validate result
            if await self._validate_result(scrape_result):
                return scrape_result
            else:
                raise ValueError("Result validation failed")

        try:
            # Execute with retry logic
            scrape_result = await scrape_with_retry()
            result.success = True
            result.data = scrape_result
            result.retry_count = retry_config.max_attempts - 1  # This would need to be tracked better

        except Exception as e:
            result.error_message = str(e)
            result.retry_count = retry_config.max_attempts

            # Call retry callback if available
            if self.on_retry:
                self.on_retry(result.retry_count, e)

        return result

    @abstractmethod
    async def _execute_scrape(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the actual scraping logic.

        This method must be implemented by subclasses.

        Args:
            target: Target information specific to the scraper type

        Returns:
            Dict containing scraped data
        """
        pass

    async def _validate_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate scraping result.

        Override in subclasses for scraper-specific validation.

        Args:
            result: Scraped data to validate

        Returns:
            True if result is valid
        """
        return bool(result and isinstance(result, dict))

    async def _pre_scrape_hook(self, target: Dict[str, Any]) -> None:
        """Hook called before scraping begins."""
        self.logger.info(f"Starting scrape for target: {target.get('url', target.get('target', 'unknown'))}")

        # Update anti-detection layer if available
        if self.anti_detection:
            await self.anti_detection.prepare_for_request(target)

    async def _post_scrape_hook(self, result: ScraperResult) -> None:
        """Hook called after scraping completes."""
        status = "SUCCESS" if result.success else "FAILED"
        self.logger.info(f"Scrape {status}: {result.response_time:.2f}s, "
                        f"{len(result.data) if result.data else 0} items collected")

        # Update anti-detection layer
        if self.anti_detection:
            await self.anti_detection.post_request_update(result)

    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        current_time = time.time()

        if current_time < self._rate_limit_until:
            wait_time = self._rate_limit_until - current_time
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")

            if self.on_rate_limit:
                self.on_rate_limit(wait_time)

            await asyncio.sleep(wait_time)

        # Update rate limit
        self._rate_limit_until = current_time + self.config.rate_limit_delay

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update scraper configuration."""
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated config {key}: {value}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get scraper performance metrics."""
        return {
            'scraper_name': self.config.name,
            'success_count': self._success_count,
            'error_count': self._error_count,
            'total_requests': self._success_count + self._error_count,
            'success_rate': self._success_count / max(1, self._success_count + self._error_count),
            'rate_limit_delay': self.config.rate_limit_delay,
            'max_retries': self.config.max_retries,
            'timeout': self.config.timeout
        }

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._success_count = 0
        self._error_count = 0
        self.logger.info("Metrics reset")

    def set_anti_detection(self, anti_detection_layer) -> None:
        """Set anti-detection layer for this scraper."""
        self.anti_detection = anti_detection_layer
        self.logger.info("Anti-detection layer configured")

    async def cleanup(self) -> None:
        """Cleanup scraper resources."""
        self.logger.info(f"Cleaning up scraper {self.config.name}")
        self._session_cookies.clear()
        self._rate_limit_until = 0.0

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name}, success_rate={self.get_metrics()['success_rate']:.2f})"
