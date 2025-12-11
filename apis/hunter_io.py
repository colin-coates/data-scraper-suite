# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Hunter.io API Collector for MJ Data Scraper Suite

Email verification and enrichment using Hunter.io API.
Provides email validation, domain search, and contact discovery.
"""

import asyncio
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from core.base_scraper import BaseScraper, ScraperConfig
from core.retry_utils import retry_async, RetryConfig, retry_on_rate_limits, retry_on_network_errors

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__description__ = "Hunter.io API for email verification and contact discovery"
__author__ = "MJ Intelligence"
__dependencies__ = ["requests"]


class HunterIOConfig(ScraperConfig):
    """Configuration for Hunter.io API integration."""
    api_key: Optional[str] = None
    verify_emails: bool = True
    find_emails: bool = True
    domain_search: bool = True
    max_results_per_call: int = 100
    confidence_threshold: float = 0.8
    cache_ttl_seconds: int = 86400  # 24 hours for email verification


class HunterIOCollector(BaseScraper):
    """
    Hunter.io API collector for email verification and contact discovery.
    Provides comprehensive email validation and business contact finding.
    """

    def __init__(self, config: HunterIOConfig):
        super().__init__(config)
        self.hunter_config = config

        # API configuration
        self.base_url = "https://api.hunter.io/v2"
        self.rate_limit_per_hour = 50  # Hunter.io free tier limit
        self.requests_this_hour = 0
        self.hour_reset_time = datetime.utcnow()

        # Caching for email verification (expensive operation)
        self.verification_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}

        # Email validation patterns
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

        self.api_calls = 0

    async def _execute_scrape(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Hunter.io API data collection.

        Args:
            target: Contains 'email', 'domain', 'company', 'operation' type

        Returns:
            Hunter.io API data (email verification, domain search, etc.)
        """
        operation = target.get('operation', 'verify')
        email = target.get('email')
        domain = target.get('domain')
        company = target.get('company')

        # Ensure API key is available
        if not self.hunter_config.api_key:
            raise RuntimeError("Hunter.io API key required")

        # Check rate limits
        if not await self._check_rate_limits():
            return {
                "error": "Rate limit exceeded",
                "rate_limited": True,
                "reset_time": (self.hour_reset_time + timedelta(hours=1)).isoformat(),
                "scraped_at": datetime.utcnow().isoformat()
            }

        try:
            if operation == 'verify' and email:
                return await self._verify_email(email)
            elif operation == 'domain_search' and domain:
                return await self._search_domain_emails(domain, target.get('type', 'personal'))
            elif operation == 'company_search' and company:
                return await self._search_company_emails(company)
            elif operation == 'find_emails' and (domain or company):
                return await self._find_emails(domain or company, target)
            else:
                raise ValueError(f"Unsupported operation: {operation} or missing required parameters")

        except Exception as e:
            logger.error(f"Hunter.io API collection failed: {e}")
            raise

    async def _verify_email(self, email: str) -> Dict[str, Any]:
        """Verify email address using Hunter.io API."""
        # Validate email format first
        if not self._is_valid_email_format(email):
            return {
                "email": email,
                "verification": {
                    "result": "invalid",
                    "score": 0,
                    "reason": "Invalid email format"
                },
                "scraped_at": datetime.utcnow().isoformat()
            }

        # Check cache first
        cache_key = f"verify_{email}"
        if self._is_cache_valid(cache_key):
            cached_result = self.verification_cache[cache_key].copy()
            cached_result["cached"] = True
            return cached_result

        # Make API call
        params = {
            "email": email,
            "api_key": self.hunter_config.api_key
        }

        data = await self._api_call('GET', "/email-verifier", params)

        result = {
            "email": email,
            "verification": {
                "result": data.get("data", {}).get("result"),
                "score": data.get("data", {}).get("score", 0) / 100,  # Convert to 0-1 scale
                "regexp": data.get("data", {}).get("regexp", False),
                "gibberish": data.get("data", {}).get("gibberish", False),
                "disposable": data.get("data", {}).get("disposable", False),
                "webmail": data.get("data", {}).get("webmail", False),
                "mx_records": data.get("data", {}).get("mx_records", False),
                "smtp_server": data.get("data", {}).get("smtp_server", False),
                "smtp_check": data.get("data", {}).get("smtp_check", False),
                "accept_all": data.get("data", {}).get("accept_all", False),
                "block": data.get("data", {}).get("block", False)
            },
            "meta": data.get("meta", {}),
            "scraped_at": datetime.utcnow().isoformat()
        }

        # Cache the result
        self._cache_verification(cache_key, result)
        self.api_calls += 1

        return result

    async def _search_domain_emails(self, domain: str, email_type: str = "personal") -> Dict[str, Any]:
        """Search for emails associated with a domain."""
        params = {
            "domain": domain,
            "type": email_type,  # personal, generic
            "limit": min(self.hunter_config.max_results_per_call, 100),
            "api_key": self.hunter_config.api_key
        }

        data = await self._api_call('GET', "/domain-search", params)

        # Process and filter results
        emails = []
        for email_data in data.get("data", {}).get("emails", []):
            confidence = email_data.get("confidence", 0) / 100
            if confidence >= self.hunter_config.confidence_threshold:
                emails.append({
                    "email": email_data.get("value"),
                    "type": email_data.get("type"),
                    "confidence": confidence,
                    "first_name": email_data.get("first_name"),
                    "last_name": email_data.get("last_name"),
                    "position": email_data.get("position"),
                    "seniority": email_data.get("seniority"),
                    "department": email_data.get("department"),
                    "linkedin_url": email_data.get("linkedin"),
                    "twitter": email_data.get("twitter"),
                    "phone_number": email_data.get("phone_number")
                })

        return {
            "domain": domain,
            "email_type": email_type,
            "total_emails": len(emails),
            "emails": emails,
            "domain_data": {
                "organization": data.get("data", {}).get("organization"),
                "country": data.get("data", {}).get("country"),
                "state": data.get("data", {}).get("state"),
                "webmail": data.get("data", {}).get("webmail", False),
                "pattern": data.get("data", {}).get("pattern")
            },
            "meta": data.get("meta", {}),
            "scraped_at": datetime.utcnow().isoformat()
        }

    async def _search_company_emails(self, company: str) -> Dict[str, Any]:
        """Search for emails associated with a company name."""
        # This is a more complex operation that might require domain lookup first
        # For now, we'll implement a basic version

        # Try to find company domain first (this would need additional logic)
        # For demonstration, we'll assume the company name might be a domain
        domain = company.lower().replace(' ', '')

        # If it looks like a domain, use domain search
        if '.' in domain:
            return await self._search_domain_emails(domain)

        # Otherwise, we might need to use Hunter's company search (if available)
        # or implement domain discovery logic
        return {
            "company": company,
            "error": "Company name resolution to domain not implemented",
            "suggestion": "Provide domain name instead of company name",
            "scraped_at": datetime.utcnow().isoformat()
        }

    async def _find_emails(self, query: str, target: Dict[str, Any]) -> Dict[str, Any]:
        """Find emails using Hunter.io's email finder API."""
        # Determine if query is domain or company
        if '.' in query and ' ' not in query:
            # Looks like a domain
            return await self._search_domain_emails(query, target.get('type', 'personal'))
        else:
            # Looks like a company name
            return await self._search_company_emails(query)

    @retry_on_network_errors(max_attempts=3)
    @retry_on_rate_limits(max_attempts=3)
    async def _api_call(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make API call to Hunter.io."""
        import aiohttp

        url = f"{self.base_url}{endpoint}"

        # Rate limiting
        await self._apply_rate_limiting()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, params=params) as response:
                    # Update rate limit tracking
                    self.requests_this_hour += 1

                    if response.status == 429:
                        # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 3600))
                        logger.warning(f"Hunter.io rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        return await self._api_call(method, endpoint, params)

                    response.raise_for_status()
                    return await response.json()

        except Exception as e:
            logger.error(f"Hunter.io API call failed: {e}")
            raise

    def _is_valid_email_format(self, email: str) -> bool:
        """Check if email has valid format."""
        return bool(self.email_pattern.match(email))

    async def _check_rate_limits(self) -> bool:
        """Check if we're within Hunter.io rate limits."""
        current_time = datetime.utcnow()

        # Reset counter if it's a new hour
        if current_time >= self.hour_reset_time + timedelta(hours=1):
            self.requests_this_hour = 0
            self.hour_reset_time = current_time.replace(minute=0, second=0, microsecond=0)

        return self.requests_this_hour < self.rate_limit_per_hour

    async def _apply_rate_limiting(self) -> None:
        """Apply rate limiting to prevent hitting API limits."""
        # Hunter.io allows ~50 requests per hour
        # We'll limit to 1 request per minute to be safe
        current_time = datetime.utcnow()
        if hasattr(self, '_last_request_time'):
            time_diff = (current_time - self._last_request_time).total_seconds()
            if time_diff < 72:  # ~50 requests per hour = 1 per ~72 seconds
                await asyncio.sleep(72 - time_diff)

        self._last_request_time = current_time

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached verification is still valid."""
        if cache_key not in self.cache_timestamps:
            return False

        cache_time = self.cache_timestamps[cache_key]
        age = (datetime.utcnow() - cache_time).total_seconds()

        return age < self.hunter_config.cache_ttl_seconds

    def _cache_verification(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache email verification result."""
        self.verification_cache[cache_key] = result
        self.cache_timestamps[cache_key] = datetime.utcnow()

        # Limit cache size to prevent memory issues
        if len(self.verification_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(self.cache_timestamps.keys(),
                               key=lambda k: self.cache_timestamps[k])[:100]
            for key in oldest_keys:
                del self.verification_cache[key]
                del self.cache_timestamps[key]

    async def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate Hunter.io API result."""
        # Check for API errors
        if 'error' in result and not result.get('rate_limited'):
            return False

        # Check for successful data
        has_data = (
            'verification' in result or
            'emails' in result or
            'domain_data' in result or
            result.get('rate_limited')  # Rate limited is still a valid response
        )

        return bool(has_data)

    def bulk_verify_emails(self, emails: List[str]) -> List[Dict[str, Any]]:
        """
        Bulk verify multiple emails (synchronous for simplicity).
        In production, this would be async and batched.
        """
        results = []
        for email in emails[:50]:  # Limit to prevent excessive API calls
            try:
                # This would need to be made async in production
                # For now, return placeholder
                results.append({
                    "email": email,
                    "verification": {"result": "bulk_verification_pending"},
                    "bulk_processed": True
                })
            except Exception as e:
                results.append({
                    "email": email,
                    "error": str(e),
                    "bulk_processed": False
                })

        return results

    def get_hunter_metrics(self) -> Dict[str, Any]:
        """Get Hunter.io API-specific metrics."""
        return {
            **self.get_metrics(),
            'api_calls': self.api_calls,
            'requests_this_hour': self.requests_this_hour,
            'hourly_limit': self.rate_limit_per_hour,
            'cache_size': len(self.verification_cache),
            'api_key_configured': bool(self.hunter_config.api_key),
            'features_enabled': {
                'verify_emails': self.hunter_config.verify_emails,
                'find_emails': self.hunter_config.find_emails,
                'domain_search': self.hunter_config.domain_search
            },
            'confidence_threshold': self.hunter_config.confidence_threshold
        }

    def clear_verification_cache(self) -> None:
        """Clear the email verification cache."""
        self.verification_cache.clear()
        self.cache_timestamps.clear()
        logger.info("Hunter.io verification cache cleared")

    async def cleanup(self) -> None:
        """Cleanup Hunter.io collector resources."""
        await super().cleanup()

        # Clear caches
        self.verification_cache.clear()
        self.cache_timestamps.clear()

        logger.info("Hunter.io collector cleaned up")
