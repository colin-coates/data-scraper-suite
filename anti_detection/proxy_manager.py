# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Proxy Rotation Manager for MJ Data Scraper Suite

Supports multiple proxy providers:
- Bright Data (Luminati)
- Oxylabs
- SmartProxy
- Custom proxy lists

Features:
- Automatic rotation strategies
- Geo-targeting
- Session management
- Health checking
- Cost tracking
"""

import asyncio
import aiohttp
import logging
import random
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class ProxyProvider(Enum):
    """Supported proxy providers."""
    BRIGHT_DATA = "bright_data"
    OXYLABS = "oxylabs"
    SMARTPROXY = "smartproxy"
    CUSTOM = "custom"


class ProxyType(Enum):
    """Types of proxies."""
    DATACENTER = "datacenter"
    RESIDENTIAL = "residential"
    MOBILE = "mobile"
    ISP = "isp"


class RotationStrategy(Enum):
    """Proxy rotation strategies."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_USED = "least_used"
    STICKY_SESSION = "sticky_session"
    GEO_TARGETED = "geo_targeted"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class ProxyConfig:
    """Configuration for a proxy provider."""
    provider: ProxyProvider
    username: str
    password: str
    host: str = ""
    port: int = 0
    proxy_type: ProxyType = ProxyType.RESIDENTIAL
    country: Optional[str] = None
    city: Optional[str] = None
    session_id: Optional[str] = None
    session_duration: int = 600  # seconds


@dataclass
class ProxyStats:
    """Statistics for a proxy."""
    requests: int = 0
    successes: int = 0
    failures: int = 0
    total_time: float = 0.0
    last_used: float = 0.0
    last_success: float = 0.0
    blocked_domains: Set[str] = field(default_factory=set)


@dataclass
class ProxyResult:
    """Result of using a proxy."""
    proxy_url: str
    success: bool
    response_time: float = 0.0
    status_code: Optional[int] = None
    error: Optional[str] = None
    ip_address: Optional[str] = None
    country: Optional[str] = None


class ProxyManager:
    """
    Intelligent proxy rotation manager with multi-provider support.
    
    Features:
    - Multiple provider integration
    - Smart rotation strategies
    - Geo-targeting
    - Session persistence
    - Health monitoring
    - Cost optimization
    """

    # Provider-specific configurations
    PROVIDER_CONFIGS = {
        ProxyProvider.BRIGHT_DATA: {
            "host": "brd.superproxy.io",
            "port": 22225,
            "format": "http://{username}-country-{country}{session}:{password}@{host}:{port}"
        },
        ProxyProvider.OXYLABS: {
            "host": "pr.oxylabs.io",
            "port": 7777,
            "format": "http://{username}-country-{country}{session}:{password}@{host}:{port}"
        },
        ProxyProvider.SMARTPROXY: {
            "host": "gate.smartproxy.com",
            "port": 7000,
            "format": "http://{username}:{password}@{host}:{port}"
        }
    }

    # Approximate costs per GB (USD)
    COSTS_PER_GB = {
        ProxyType.DATACENTER: 0.50,
        ProxyType.RESIDENTIAL: 12.00,
        ProxyType.MOBILE: 25.00,
        ProxyType.ISP: 15.00,
    }

    def __init__(
        self,
        configs: List[ProxyConfig],
        rotation_strategy: RotationStrategy = RotationStrategy.ROUND_ROBIN,
        health_check_interval: int = 300,
        max_failures_before_disable: int = 5,
        default_country: str = "us"
    ):
        """
        Initialize proxy manager.

        Args:
            configs: List of proxy provider configurations
            rotation_strategy: Strategy for rotating proxies
            health_check_interval: Seconds between health checks
            max_failures_before_disable: Failures before disabling proxy
            default_country: Default country for geo-targeting
        """
        self.configs = {c.provider: c for c in configs}
        self.rotation_strategy = rotation_strategy
        self.health_check_interval = health_check_interval
        self.max_failures = max_failures_before_disable
        self.default_country = default_country

        # Proxy pools
        self.custom_proxies: List[str] = []
        self.proxy_stats: Dict[str, ProxyStats] = defaultdict(ProxyStats)
        self.disabled_proxies: Set[str] = set()
        self.domain_sessions: Dict[str, str] = {}  # domain -> session_id

        # Rotation state
        self._round_robin_index = 0
        self._session_counter = 0

        # Metrics
        self.total_requests = 0
        self.total_bandwidth_bytes = 0
        self.total_cost = 0.0

        self._session: Optional[aiohttp.ClientSession] = None
        self._health_check_task: Optional[asyncio.Task] = None

    def add_custom_proxies(self, proxies: List[str]) -> None:
        """
        Add custom proxy URLs to the pool.

        Args:
            proxies: List of proxy URLs (http://user:pass@host:port)
        """
        self.custom_proxies.extend(proxies)
        logger.info(f"Added {len(proxies)} custom proxies")

    def _build_proxy_url(
        self,
        provider: ProxyProvider,
        country: Optional[str] = None,
        city: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Build proxy URL for a provider."""
        if provider not in self.configs:
            raise ValueError(f"Provider {provider} not configured")

        config = self.configs[provider]
        provider_config = self.PROVIDER_CONFIGS.get(provider, {})

        country = country or config.country or self.default_country
        host = config.host or provider_config.get("host", "")
        port = config.port or provider_config.get("port", 0)

        # Build session string
        session_str = ""
        if session_id:
            if provider == ProxyProvider.BRIGHT_DATA:
                session_str = f"-session-{session_id}"
            elif provider == ProxyProvider.OXYLABS:
                session_str = f"-sessid-{session_id}"

        # Build city string
        city_str = ""
        if city:
            if provider == ProxyProvider.BRIGHT_DATA:
                city_str = f"-city-{city}"

        # Format URL
        url_format = provider_config.get("format", "http://{username}:{password}@{host}:{port}")
        
        proxy_url = url_format.format(
            username=config.username,
            password=config.password,
            host=host,
            port=port,
            country=country,
            city=city_str,
            session=session_str
        )

        return proxy_url

    async def get_proxy(
        self,
        target_domain: Optional[str] = None,
        country: Optional[str] = None,
        city: Optional[str] = None,
        proxy_type: Optional[ProxyType] = None,
        sticky_session: bool = False
    ) -> str:
        """
        Get a proxy URL based on the rotation strategy.

        Args:
            target_domain: Domain being scraped (for session stickiness)
            country: Target country for geo-targeting
            city: Target city for geo-targeting
            proxy_type: Preferred proxy type
            sticky_session: Whether to use sticky sessions

        Returns:
            Proxy URL string
        """
        self.total_requests += 1

        # Check for existing session
        if sticky_session and target_domain and target_domain in self.domain_sessions:
            session_id = self.domain_sessions[target_domain]
            # Return same session proxy
            for provider in self.configs:
                return self._build_proxy_url(provider, country, city, session_id)

        # Select proxy based on strategy
        if self.rotation_strategy == RotationStrategy.ROUND_ROBIN:
            proxy = await self._get_round_robin(country, city)
        elif self.rotation_strategy == RotationStrategy.RANDOM:
            proxy = await self._get_random(country, city)
        elif self.rotation_strategy == RotationStrategy.LEAST_USED:
            proxy = await self._get_least_used(country, city)
        elif self.rotation_strategy == RotationStrategy.PERFORMANCE_BASED:
            proxy = await self._get_performance_based(country, city)
        elif self.rotation_strategy == RotationStrategy.STICKY_SESSION:
            proxy = await self._get_sticky_session(target_domain, country, city)
        else:
            proxy = await self._get_random(country, city)

        # Track session for domain
        if sticky_session and target_domain:
            self._session_counter += 1
            session_id = f"mj{self._session_counter}"
            self.domain_sessions[target_domain] = session_id

        return proxy

    async def _get_round_robin(self, country: Optional[str], city: Optional[str]) -> str:
        """Get proxy using round-robin rotation."""
        all_proxies = self._get_all_available_proxies(country, city)
        if not all_proxies:
            raise RuntimeError("No proxies available")

        proxy = all_proxies[self._round_robin_index % len(all_proxies)]
        self._round_robin_index += 1
        return proxy

    async def _get_random(self, country: Optional[str], city: Optional[str]) -> str:
        """Get random proxy."""
        all_proxies = self._get_all_available_proxies(country, city)
        if not all_proxies:
            raise RuntimeError("No proxies available")
        return random.choice(all_proxies)

    async def _get_least_used(self, country: Optional[str], city: Optional[str]) -> str:
        """Get least used proxy."""
        all_proxies = self._get_all_available_proxies(country, city)
        if not all_proxies:
            raise RuntimeError("No proxies available")

        # Sort by usage count
        sorted_proxies = sorted(
            all_proxies,
            key=lambda p: self.proxy_stats[p].requests
        )
        return sorted_proxies[0]

    async def _get_performance_based(self, country: Optional[str], city: Optional[str]) -> str:
        """Get best performing proxy."""
        all_proxies = self._get_all_available_proxies(country, city)
        if not all_proxies:
            raise RuntimeError("No proxies available")

        # Score based on success rate and response time
        def score(proxy: str) -> float:
            stats = self.proxy_stats[proxy]
            if stats.requests == 0:
                return 0.5  # Neutral score for unused
            success_rate = stats.successes / max(1, stats.requests)
            avg_time = stats.total_time / max(1, stats.successes)
            # Higher success rate and lower time = better score
            return success_rate - (avg_time / 10)

        sorted_proxies = sorted(all_proxies, key=score, reverse=True)
        
        # Add some randomness to avoid always picking the same one
        top_proxies = sorted_proxies[:max(3, len(sorted_proxies) // 3)]
        return random.choice(top_proxies)

    async def _get_sticky_session(
        self,
        target_domain: Optional[str],
        country: Optional[str],
        city: Optional[str]
    ) -> str:
        """Get proxy with sticky session for domain."""
        session_id = None
        if target_domain:
            if target_domain in self.domain_sessions:
                session_id = self.domain_sessions[target_domain]
            else:
                self._session_counter += 1
                session_id = f"mj{self._session_counter}"
                self.domain_sessions[target_domain] = session_id

        # Use first available provider with session support
        for provider in [ProxyProvider.BRIGHT_DATA, ProxyProvider.OXYLABS]:
            if provider in self.configs:
                return self._build_proxy_url(provider, country, city, session_id)

        # Fallback to random
        return await self._get_random(country, city)

    def _get_all_available_proxies(
        self,
        country: Optional[str],
        city: Optional[str]
    ) -> List[str]:
        """Get all available proxy URLs."""
        proxies = []

        # Add provider proxies
        for provider in self.configs:
            try:
                proxy = self._build_proxy_url(provider, country, city)
                if proxy not in self.disabled_proxies:
                    proxies.append(proxy)
            except Exception as e:
                logger.warning(f"Failed to build proxy for {provider}: {e}")

        # Add custom proxies
        for proxy in self.custom_proxies:
            if proxy not in self.disabled_proxies:
                proxies.append(proxy)

        return proxies

    def report_result(self, proxy_url: str, result: ProxyResult) -> None:
        """
        Report the result of using a proxy.

        Args:
            proxy_url: The proxy URL that was used
            result: Result of the request
        """
        stats = self.proxy_stats[proxy_url]
        stats.requests += 1
        stats.last_used = time.time()

        if result.success:
            stats.successes += 1
            stats.last_success = time.time()
            stats.total_time += result.response_time
        else:
            stats.failures += 1

            # Check if should disable
            recent_failures = stats.failures - (stats.successes * 0.1)
            if recent_failures >= self.max_failures:
                self.disabled_proxies.add(proxy_url)
                logger.warning(f"Disabled proxy due to failures: {proxy_url[:50]}...")

    async def check_proxy_health(self, proxy_url: str) -> ProxyResult:
        """
        Check if a proxy is working.

        Args:
            proxy_url: Proxy URL to check

        Returns:
            ProxyResult with health check results
        """
        if self._session is None:
            self._session = aiohttp.ClientSession()

        start_time = time.time()
        try:
            async with self._session.get(
                "https://httpbin.org/ip",
                proxy=proxy_url,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                data = await resp.json()
                return ProxyResult(
                    proxy_url=proxy_url,
                    success=resp.status == 200,
                    response_time=time.time() - start_time,
                    status_code=resp.status,
                    ip_address=data.get("origin")
                )
        except Exception as e:
            return ProxyResult(
                proxy_url=proxy_url,
                success=False,
                response_time=time.time() - start_time,
                error=str(e)
            )

    async def start_health_checks(self) -> None:
        """Start background health check task."""
        if self._health_check_task is not None:
            return

        async def health_check_loop():
            while True:
                await asyncio.sleep(self.health_check_interval)
                await self._run_health_checks()

        self._health_check_task = asyncio.create_task(health_check_loop())
        logger.info("Started proxy health check background task")

    async def _run_health_checks(self) -> None:
        """Run health checks on all proxies."""
        all_proxies = self._get_all_available_proxies(None, None)
        
        for proxy in all_proxies:
            result = await self.check_proxy_health(proxy)
            self.report_result(proxy, result)

            if result.success and proxy in self.disabled_proxies:
                # Re-enable if health check passes
                self.disabled_proxies.remove(proxy)
                logger.info(f"Re-enabled proxy after health check: {proxy[:50]}...")

    def get_proxy_for_country(self, country_code: str) -> str:
        """
        Get a proxy for a specific country.

        Args:
            country_code: ISO 2-letter country code

        Returns:
            Proxy URL for that country
        """
        for provider in [ProxyProvider.BRIGHT_DATA, ProxyProvider.OXYLABS, ProxyProvider.SMARTPROXY]:
            if provider in self.configs:
                return self._build_proxy_url(provider, country=country_code.lower())

        raise RuntimeError(f"No geo-targeting proxy available for {country_code}")

    def clear_session(self, domain: str) -> None:
        """Clear sticky session for a domain."""
        if domain in self.domain_sessions:
            del self.domain_sessions[domain]
            logger.debug(f"Cleared session for domain: {domain}")

    def clear_all_sessions(self) -> None:
        """Clear all sticky sessions."""
        self.domain_sessions.clear()
        logger.info("Cleared all proxy sessions")

    def get_metrics(self) -> Dict[str, Any]:
        """Get proxy manager metrics."""
        total_successes = sum(s.successes for s in self.proxy_stats.values())
        total_failures = sum(s.failures for s in self.proxy_stats.values())

        return {
            "total_requests": self.total_requests,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "success_rate": total_successes / max(1, total_successes + total_failures),
            "providers_configured": [p.value for p in self.configs.keys()],
            "custom_proxies_count": len(self.custom_proxies),
            "disabled_proxies_count": len(self.disabled_proxies),
            "active_sessions": len(self.domain_sessions),
            "rotation_strategy": self.rotation_strategy.value,
            "estimated_cost_usd": round(self.total_cost, 4)
        }

    def get_proxy_stats(self, proxy_url: str) -> Dict[str, Any]:
        """Get stats for a specific proxy."""
        stats = self.proxy_stats.get(proxy_url)
        if not stats:
            return {}

        return {
            "requests": stats.requests,
            "successes": stats.successes,
            "failures": stats.failures,
            "success_rate": stats.successes / max(1, stats.requests),
            "avg_response_time": stats.total_time / max(1, stats.successes),
            "last_used": stats.last_used,
            "last_success": stats.last_success,
            "is_disabled": proxy_url in self.disabled_proxies
        }

    async def close(self) -> None:
        """Close the proxy manager."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._session and not self._session.closed:
            await self._session.close()

        logger.info("Proxy manager closed")
