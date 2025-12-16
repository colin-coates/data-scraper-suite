# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Network Sentinel for MJ Data Scraper Suite

Monitors network connectivity, performance, and health metrics.
Provides risk assessments for network-related scraping operations.
"""

import asyncio
import logging
import socket
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import ssl

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    # Fallback for basic connectivity checks
    import urllib.request

from .base import BaseSentinel, SentinelReport

logger = logging.getLogger(__name__)


class NetworkSentinel(BaseSentinel):
    """
    Monitors network conditions and connectivity for scraping operations.

    Probes network health, DNS resolution, SSL certificates, and connectivity
    to provide risk assessments for network-dependent operations.
    """

    name = "network_sentinel"

    def __init__(self):
        super().__init__()

        # Network monitoring thresholds
        self.max_dns_resolution_time = 5.0  # seconds
        self.max_connect_time = 10.0  # seconds
        self.max_response_time = 30.0  # seconds
        self.min_ssl_days_remaining = 7  # days
        self.max_network_errors = 3  # consecutive errors

        # Monitoring state
        self.dns_cache = {}
        self.connectivity_history = []
        self.error_count = 0
        self.history_window = 10

        # Target domains to monitor (can be configured)
        self.monitor_domains = [
            "google.com",  # Basic connectivity
            "linkedin.com",  # Social platform
            "facebook.com",  # Social platform
            "twitter.com",   # Social platform
            "httpbin.org"    # Test endpoint
        ]

    async def probe(self, target: Dict[str, Any]) -> SentinelReport:
        """
        Probe network conditions and connectivity.

        Args:
            target: Target information containing URLs, domains, or network context

        Returns:
            SentinelReport with network health assessment and recommendations
        """
        try:
            # Extract domains/URLs from target
            domains = self._extract_domains(target)

            if not domains:
                domains = self.monitor_domains  # Use default monitoring domains

            # Perform network health checks
            network_metrics = await self._check_network_health(domains)

            # Analyze metrics and determine risk
            findings = {
                "domains_checked": len(domains),
                "network_metrics": network_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }

            issues = []
            risk_score = 0

            # Analyze connectivity issues
            connectivity_failures = sum(1 for m in network_metrics if not m.get("connectivity_ok", False))
            if connectivity_failures > 0:
                issues.append(f"Connectivity failures: {connectivity_failures}/{len(domains)} domains")
                risk_score += connectivity_failures * 2

            # Analyze DNS resolution times
            slow_dns = [m for m in network_metrics if m.get("dns_time", 0) > self.max_dns_resolution_time]
            if slow_dns:
                issues.append(f"Slow DNS resolution: {len(slow_dns)} domains")
                risk_score += len(slow_dns)

            # Analyze connection times
            slow_connections = [m for m in network_metrics if m.get("connect_time", 0) > self.max_connect_time]
            if slow_connections:
                issues.append(f"Slow connections: {len(slow_connections)} domains")
                risk_score += len(slow_connections)

            # Analyze SSL certificate validity
            expiring_ssl = [m for m in network_metrics if m.get("ssl_days_remaining", 999) < self.min_ssl_days_remaining]
            if expiring_ssl:
                issues.append(f"Expiring SSL certificates: {len(expiring_ssl)} domains")
                risk_score += len(expiring_ssl) * 2

            # Analyze response times
            slow_responses = [m for m in network_metrics if m.get("response_time", 0) > self.max_response_time]
            if slow_responses:
                issues.append(f"Slow responses: {len(slow_responses)} domains")
                risk_score += len(slow_responses)

            # Track error history
            if connectivity_failures > 0 or len(issues) > 0:
                self.error_count += 1
            else:
                self.error_count = max(0, self.error_count - 1)

            if self.error_count >= self.max_network_errors:
                issues.append(f"Persistent network issues: {self.error_count} consecutive errors")
                risk_score += 2

            findings["issues"] = issues
            findings["error_count"] = self.error_count

            # Determine risk level and recommended action
            if risk_score >= 6:
                risk_level = "critical"
                recommended_action = "block"
            elif risk_score >= 4:
                risk_level = "high"
                recommended_action = "restrict"
            elif risk_score >= 2:
                risk_level = "medium"
                recommended_action = "delay"
            else:
                risk_level = "low"
                recommended_action = "allow"

            return SentinelReport(
                sentinel_name=self.name,
                domain="network",
                timestamp=datetime.utcnow(),
                risk_level=risk_level,
                findings=findings,
                recommended_action=recommended_action
            )

        except Exception as e:
            logger.error(f"Network probe failed: {e}")
            return SentinelReport(
                sentinel_name=self.name,
                domain="network",
                timestamp=datetime.utcnow(),
                risk_level="critical",
                findings={"error": str(e), "error_type": type(e).__name__},
                recommended_action="block"
            )

    def _extract_domains(self, target: Dict[str, Any]) -> List[str]:
        """Extract domain names from target information."""
        domains = []

        # Extract from URLs
        urls = target.get("urls", []) + target.get("url", "").split(",")
        for url in urls:
            if url.strip():
                try:
                    domain = self._extract_domain_from_url(url.strip())
                    if domain and domain not in domains:
                        domains.append(domain)
                except Exception:
                    continue

        # Extract direct domains
        direct_domains = target.get("domains", [])
        for domain in direct_domains:
            if domain.strip() and domain not in domains:
                domains.append(domain.strip())

        return domains[:5]  # Limit to 5 domains for performance

    def _extract_domain_from_url(self, url: str) -> Optional[str]:
        """Extract domain from URL."""
        if "://" not in url:
            url = "http://" + url

        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain:
                # Remove port if present
                domain = domain.split(":")[0]
                return domain
        except Exception:
            pass
        return None

    async def _check_network_health(self, domains: List[str]) -> List[Dict[str, Any]]:
        """Check network health for given domains."""
        results = []

        for domain in domains:
            domain_metrics = {
                "domain": domain,
                "timestamp": datetime.utcnow().isoformat()
            }

            try:
                # DNS resolution check
                dns_start = time.time()
                try:
                    ip_addresses = await asyncio.get_event_loop().run_in_executor(
                        None, socket.getaddrinfo, domain, 80, socket.AF_UNSPEC,
                        socket.SOCK_STREAM
                    )
                    dns_time = time.time() - dns_start
                    domain_metrics["dns_time"] = dns_time
                    domain_metrics["dns_resolved"] = True
                    domain_metrics["ip_addresses"] = len(ip_addresses)
                except Exception as e:
                    domain_metrics["dns_time"] = time.time() - dns_start
                    domain_metrics["dns_resolved"] = False
                    domain_metrics["dns_error"] = str(e)

                # Basic connectivity check (TCP connect)
                if domain_metrics.get("dns_resolved", False):
                    connect_start = time.time()
                    try:
                        reader, writer = await asyncio.open_connection(domain, 80)
                        connect_time = time.time() - connect_start
                        domain_metrics["connect_time"] = connect_time
                        domain_metrics["connectivity_ok"] = True
                        writer.close()
                        await writer.wait_closed()
                    except Exception as e:
                        domain_metrics["connect_time"] = time.time() - connect_start
                        domain_metrics["connectivity_ok"] = False
                        domain_metrics["connect_error"] = str(e)

                # SSL certificate check (if HTTPS available)
                try:
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE

                    reader, writer = await asyncio.open_connection(
                        domain, 443, ssl=ssl_context
                    )
                    ssl_transport = writer.get_extra_info('ssl_object')
                    if ssl_transport:
                        cert = ssl_transport.getpeercert()
                        if cert:
                            expiry_date = datetime.strptime(
                                cert['notAfter'], '%b %d %H:%M:%S %Y %Z'
                            )
                            days_remaining = (expiry_date - datetime.utcnow()).days
                            domain_metrics["ssl_days_remaining"] = days_remaining
                            domain_metrics["ssl_valid"] = days_remaining > 0

                    writer.close()
                    await writer.wait_closed()
                except Exception as e:
                    domain_metrics["ssl_error"] = str(e)

                # HTTP response check (if aiohttp available)
                if AIOHTTP_AVAILABLE:
                    try:
                        timeout = aiohttp.ClientTimeout(total=10)
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            response_start = time.time()
                            async with session.get(f"http://{domain}") as response:
                                response_time = time.time() - response_start
                                domain_metrics["response_time"] = response_time
                                domain_metrics["response_status"] = response.status
                                domain_metrics["response_ok"] = 200 <= response.status < 400
                    except Exception as e:
                        domain_metrics["response_error"] = str(e)

            except Exception as e:
                domain_metrics["general_error"] = str(e)

            results.append(domain_metrics)

            # Small delay between checks to be respectful
            await asyncio.sleep(0.1)

        return results

    def update_thresholds(self, max_dns_time: float = None, max_connect_time: float = None,
                         max_response_time: float = None, min_ssl_days: int = None) -> None:
        """Update network monitoring thresholds."""
        if max_dns_time is not None:
            self.max_dns_resolution_time = max_dns_time
        if max_connect_time is not None:
            self.max_connect_time = max_connect_time
        if max_response_time is not None:
            self.max_response_time = max_response_time
        if min_ssl_days is not None:
            self.min_ssl_days_remaining = min_ssl_days

        logger.info(f"Updated network thresholds for {self.name}: "
                   f"dns={self.max_dns_resolution_time}s, "
                   f"connect={self.max_connect_time}s, "
                   f"response={self.max_response_time}s, "
                   f"ssl={self.min_ssl_days_remaining} days")

    def set_monitor_domains(self, domains: List[str]) -> None:
        """Set domains to monitor for network health."""
        self.monitor_domains = domains[:10]  # Limit to 10 domains
        logger.info(f"Set monitor domains: {self.monitor_domains}")

    def get_network_history(self) -> list:
        """Get network connectivity history."""
        return self.connectivity_history.copy()


# Factory function for easy instantiation
def create_network_sentinel(monitor_domains: List[str] = None,
                           max_dns_time: float = 5.0,
                           max_connect_time: float = 10.0,
                           max_response_time: float = 30.0,
                           min_ssl_days: int = 7) -> NetworkSentinel:
    """
    Create a network sentinel with sensible defaults.

    Args:
        monitor_domains: List of domains to monitor
        max_dns_time: Maximum acceptable DNS resolution time (seconds)
        max_connect_time: Maximum acceptable connection time (seconds)
        max_response_time: Maximum acceptable response time (seconds)
        min_ssl_days: Minimum SSL certificate validity period (days)

    Returns:
        Configured NetworkSentinel instance
    """
    sentinel = NetworkSentinel()
    sentinel.update_thresholds(
        max_dns_time=max_dns_time,
        max_connect_time=max_connect_time,
        max_response_time=max_response_time,
        min_ssl_days=min_ssl_days
    )

    if monitor_domains:
        sentinel.set_monitor_domains(monitor_domains)

    return sentinel
