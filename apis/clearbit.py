# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Clearbit API Collector for MJ Data Scraper Suite

Company enrichment using Clearbit API.
Provides comprehensive company data, logos, social media links, and business intelligence.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from core.base_scraper import BaseScraper, ScraperConfig
from core.retry_utils import retry_async, RetryConfig, retry_on_rate_limits, retry_on_network_errors

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__description__ = "Clearbit API for comprehensive company enrichment"
__author__ = "MJ Intelligence"
__dependencies__ = ["requests"]


class ClearbitConfig(ScraperConfig):
    """Configuration for Clearbit API integration."""
    api_key: Optional[str] = None
    enrichment_enabled: bool = True
    logo_retrieval: bool = True
    discovery_enabled: bool = False  # Requires paid plan
    max_results_per_call: int = 50
    cache_ttl_seconds: int = 604800  # 7 days for company data
    include_financial_data: bool = False  # Requires special access


class ClearbitCollector(BaseScraper):
    """
    Clearbit API collector for comprehensive company data enrichment.
    Provides company information, logos, social media, and business intelligence.
    """

    def __init__(self, config: ClearbitConfig):
        super().__init__(config)
        self.clearbit_config = config

        # API configuration
        self.base_url = "https://company.clearbit.com/v2/companies"
        self.logo_base_url = "https://logo.clearbit.com"
        self.discovery_url = "https://discovery.clearbit.com/v1/companies"

        # Rate limiting (Clearbit allows 600 requests per month free tier)
        self.monthly_limit = 600
        self.requests_this_month = 0
        self.month_reset_time = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Caching for company data
        self.company_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}

        self.api_calls = 0

    async def _execute_scrape(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Clearbit API data collection.

        Args:
            target: Contains 'domain', 'company_name', 'operation' type

        Returns:
            Clearbit API company data
        """
        operation = target.get('operation', 'enrich')
        domain = target.get('domain')
        company_name = target.get('company_name')

        # Ensure API key is available
        if not self.clearbit_config.api_key:
            raise RuntimeError("Clearbit API key required")

        # Check rate limits
        if not await self._check_rate_limits():
            return {
                "error": "Rate limit exceeded",
                "rate_limited": True,
                "reset_time": (self.month_reset_time + timedelta(days=30)).isoformat(),
                "scraped_at": datetime.utcnow().isoformat()
            }

        try:
            if operation == 'enrich' and domain:
                return await self._enrich_company(domain)
            elif operation == 'find' and company_name:
                return await self._find_company(company_name)
            elif operation == 'logo' and domain:
                return await self._get_company_logo(domain)
            elif operation == 'autocomplete' and company_name:
                return await self._autocomplete_company(company_name)
            else:
                raise ValueError(f"Unsupported operation: {operation} or missing required parameters")

        except Exception as e:
            logger.error(f"Clearbit API collection failed: {e}")
            raise

    async def _enrich_company(self, domain: str) -> Dict[str, Any]:
        """Enrich company data using Clearbit API."""
        # Clean domain
        domain = domain.lower().strip()
        if domain.startswith('http'):
            from urllib.parse import urlparse
            domain = urlparse(domain).netloc

        # Check cache first
        cache_key = f"enrich_{domain}"
        if self._is_cache_valid(cache_key):
            cached_result = self.company_cache[cache_key].copy()
            cached_result["cached"] = True
            return cached_result

        # Make API call
        url = f"{self.base_url}/find"
        params = {
            "domain": domain
        }

        data = await self._api_call('GET', url, params)

        # Process and enhance the result
        result = {
            "domain": domain,
            "company_data": data,
            "enriched_at": datetime.utcnow().isoformat(),
            "data_source": "clearbit"
        }

        # Add derived insights
        result.update(self._derive_company_insights(data))

        # Cache the result
        self._cache_company(cache_key, result)
        self.api_calls += 1

        return result

    async def _find_company(self, company_name: str) -> Dict[str, Any]:
        """Find company by name using Clearbit Discovery API."""
        if not self.clearbit_config.discovery_enabled:
            return {
                "company_name": company_name,
                "error": "Discovery API not enabled (requires paid plan)",
                "suggestion": "Use domain-based enrichment instead",
                "scraped_at": datetime.utcnow().isoformat()
            }

        url = f"{self.discovery_url}/search"
        params = {
            "query": company_name,
            "limit": min(self.clearbit_config.max_results_per_call, 10)
        }

        data = await self._api_call('GET', url, params)

        return {
            "search_query": company_name,
            "companies": data.get("results", []),
            "total_results": len(data.get("results", [])),
            "scraped_at": datetime.utcnow().isoformat()
        }

    async def _get_company_logo(self, domain: str) -> Dict[str, Any]:
        """Get company logo using Clearbit Logo API."""
        if not self.clearbit_config.logo_retrieval:
            return {
                "domain": domain,
                "error": "Logo retrieval not enabled",
                "scraped_at": datetime.utcnow().isoformat()
            }

        # Clearbit Logo API is a simple URL-based service
        logo_url = f"{self.logo_base_url}/{domain}"

        # Test if logo exists (basic check)
        logo_exists = await self._check_logo_exists(logo_url)

        return {
            "domain": domain,
            "logo_url": logo_url if logo_exists else None,
            "logo_exists": logo_exists,
            "logo_service": "clearbit",
            "scraped_at": datetime.utcnow().isoformat()
        }

    async def _autocomplete_company(self, partial_name: str) -> Dict[str, Any]:
        """Autocomplete company name using Clearbit."""
        url = f"{self.base_url}/suggest"
        params = {
            "query": partial_name,
            "limit": min(self.clearbit_config.max_results_per_call, 20)
        }

        data = await self._api_call('GET', url, params)

        return {
            "query": partial_name,
            "suggestions": data.get("results", []),
            "total_suggestions": len(data.get("results", [])),
            "scraped_at": datetime.utcnow().isoformat()
        }

    @retry_on_network_errors(max_attempts=3)
    @retry_on_rate_limits(max_attempts=3)
    async def _api_call(self, method: str, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make authenticated API call to Clearbit."""
        import aiohttp

        headers = {
            'Authorization': f'Bearer {self.clearbit_config.api_key}',
            'Content-Type': 'application/json'
        }

        # Rate limiting
        await self._apply_rate_limiting()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, headers=headers, params=params) as response:
                    # Update rate limit tracking
                    self.requests_this_month += 1

                    if response.status == 429:
                        # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 86400))  # Default to 1 day
                        logger.warning(f"Clearbit rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        return await self._api_call(method, url, params)

                    if response.status == 404:
                        # Company not found
                        return {}

                    response.raise_for_status()
                    return await response.json()

        except Exception as e:
            logger.error(f"Clearbit API call failed: {e}")
            raise

    async def _check_logo_exists(self, logo_url: str) -> bool:
        """Check if a Clearbit logo exists."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.head(logo_url) as response:
                    return response.status == 200
        except:
            return False

    def _derive_company_insights(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Derive additional insights from company data."""
        insights = {}

        if not company_data:
            return insights

        # Company size insights
        employees = company_data.get('metrics', {}).get('employees')
        if employees:
            if employees < 10:
                insights['company_size'] = 'startup'
            elif employees < 100:
                insights['company_size'] = 'small'
            elif employees < 1000:
                insights['company_size'] = 'medium'
            elif employees < 10000:
                insights['company_size'] = 'large'
            else:
                insights['company_size'] = 'enterprise'

        # Industry classification
        category = company_data.get('category', {}).get('sector')
        if category:
            insights['industry'] = category.lower().replace(' ', '_')

        # Geographic insights
        location = company_data.get('geo', {})
        if location:
            insights['headquarters_country'] = location.get('country')
            insights['headquarters_state'] = location.get('state')

        # Social media completeness
        social_links = []
        if company_data.get('facebook', {}).get('handle'):
            social_links.append('facebook')
        if company_data.get('twitter', {}).get('handle'):
            social_links.append('twitter')
        if company_data.get('linkedin', {}).get('handle'):
            social_links.append('linkedin')

        insights['social_media_presence'] = social_links
        insights['social_completeness_score'] = len(social_links) / 3  # Max 3 major platforms

        # Technology stack insights
        tech_tags = company_data.get('tags', [])
        insights['technology_tags'] = tech_tags[:10]  # Limit to top 10

        return insights

    async def _check_rate_limits(self) -> bool:
        """Check if we're within Clearbit rate limits."""
        current_time = datetime.utcnow()

        # Reset counter if it's a new month
        if current_time >= self.month_reset_time + timedelta(days=30):
            self.requests_this_month = 0
            # Calculate next month reset
            next_month = current_time.replace(day=1) + timedelta(days=32)
            self.month_reset_time = next_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        return self.requests_this_month < self.monthly_limit

    async def _apply_rate_limiting(self) -> None:
        """Apply rate limiting to prevent hitting API limits."""
        # Clearbit allows 600 requests per month = ~20 per day = ~1 per hour
        # We'll be conservative and limit to 1 request per 2 hours
        current_time = datetime.utcnow()
        if hasattr(self, '_last_request_time'):
            time_diff = (current_time - self._last_request_time).total_seconds()
            if time_diff < 7200:  # 2 hours
                wait_time = 7200 - time_diff
                logger.debug(f"Rate limiting: waiting {wait_time:.0f}s")
                await asyncio.sleep(wait_time)

        self._last_request_time = current_time

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached company data is still valid."""
        if cache_key not in self.cache_timestamps:
            return False

        cache_time = self.cache_timestamps[cache_key]
        age = (datetime.utcnow() - cache_time).total_seconds()

        return age < self.clearbit_config.cache_ttl_seconds

    def _cache_company(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Cache company data."""
        self.company_cache[cache_key] = data
        self.cache_timestamps[cache_key] = datetime.utcnow()

        # Limit cache size
        if len(self.company_cache) > 500:
            # Remove oldest entries
            oldest_keys = sorted(self.cache_timestamps.keys(),
                               key=lambda k: self.cache_timestamps[k])[:50]
            for key in oldest_keys:
                del self.company_cache[key]
                del self.cache_timestamps[key]

    async def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate Clearbit API result."""
        # Check for API errors
        if 'error' in result and not result.get('rate_limited'):
            return False

        # Check for successful data
        has_data = (
            'company_data' in result or
            'companies' in result or
            'suggestions' in result or
            'logo_url' in result or
            result.get('rate_limited')  # Rate limited is still a valid response
        )

        return bool(has_data)

    def get_clearbit_metrics(self) -> Dict[str, Any]:
        """Get Clearbit API-specific metrics."""
        return {
            **self.get_metrics(),
            'api_calls': self.api_calls,
            'requests_this_month': self.requests_this_month,
            'monthly_limit': self.monthly_limit,
            'cache_size': len(self.company_cache),
            'api_key_configured': bool(self.clearbit_config.api_key),
            'features_enabled': {
                'enrichment': self.clearbit_config.enrichment_enabled,
                'logo_retrieval': self.clearbit_config.logo_retrieval,
                'discovery': self.clearbit_config.discovery_enabled
            },
            'next_reset': self.month_reset_time.isoformat()
        }

    def clear_company_cache(self) -> None:
        """Clear the company data cache."""
        self.company_cache.clear()
        self.cache_timestamps.clear()
        logger.info("Clearbit company cache cleared")

    async def cleanup(self) -> None:
        """Cleanup Clearbit collector resources."""
        await super().cleanup()

        # Clear caches
        self.company_cache.clear()
        self.cache_timestamps.clear()

        logger.info("Clearbit collector cleaned up")
