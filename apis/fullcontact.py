# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
FullContact API Collector for MJ Data Scraper Suite

Contact graph enrichment using FullContact API.
Provides social media profiles, contact information, and relationship mapping.
"""

import asyncio
import logging
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from core.base_scraper import BaseScraper, ScraperConfig
from core.retry_utils import retry_async, RetryConfig, retry_on_rate_limits, retry_on_network_errors

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__description__ = "FullContact API for contact graph enrichment and social profiling"
__author__ = "MJ Intelligence"
__dependencies__ = ["requests"]


class FullContactConfig(ScraperConfig):
    """Configuration for FullContact API integration."""
    api_key: Optional[str] = None
    person_enrichment: bool = True
    company_enrichment: bool = True
    audience_enrichment: bool = False  # Requires paid plan
    webhooks_enabled: bool = False
    max_results_per_call: int = 50
    cache_ttl_seconds: int = 2592000  # 30 days for contact data
    include_social_profiles: bool = True
    include_demographics: bool = True
    include_photos: bool = True


class FullContactCollector(BaseScraper):
    """
    FullContact API collector for comprehensive contact enrichment.
    Provides social profiles, contact data, and relationship intelligence.
    """

    def __init__(self, config: FullContactConfig):
        super().__init__(config)
        self.fullcontact_config = config

        # API configuration
        self.base_url = "https://api.fullcontact.com/v3"
        self.audience_url = "https://audience.fullcontact.com/v2"
        self.webhook_url = "https://webhook.fullcontact.com/v1"

        # Rate limiting (FullContact allows 500 requests per month free tier)
        self.monthly_limit = 500
        self.requests_this_month = 0
        self.month_reset_time = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Caching for contact data
        self.contact_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}

        self.api_calls = 0

    async def _execute_scrape(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute FullContact API data collection.

        Args:
            target: Contains 'email', 'phone', 'twitter', 'operation' type

        Returns:
            FullContact API enriched contact data
        """
        operation = target.get('operation', 'enrich')
        email = target.get('email')
        phone = target.get('phone')
        twitter = target.get('twitter')
        linkedin = target.get('linkedin')

        # Ensure API key is available
        if not self.fullcontact_config.api_key:
            raise RuntimeError("FullContact API key required")

        # Check rate limits
        if not await self._check_rate_limits():
            return {
                "error": "Rate limit exceeded",
                "rate_limited": True,
                "reset_time": (self.month_reset_time + timedelta(days=30)).isoformat(),
                "scraped_at": datetime.utcnow().isoformat()
            }

        try:
            if operation == 'enrich_person' and (email or phone or twitter):
                return await self._enrich_person(email, phone, twitter, linkedin)
            elif operation == 'enrich_company' and email:
                return await self._enrich_company(email)
            elif operation == 'verify_email' and email:
                return await self._verify_email(email)
            elif operation == 'audience' and self.fullcontact_config.audience_enrichment:
                return await self._get_audience_insights(target)
            else:
                raise ValueError(f"Unsupported operation: {operation} or missing required parameters")

        except Exception as e:
            logger.error(f"FullContact API collection failed: {e}")
            raise

    async def _enrich_person(self, email: Optional[str] = None,
                           phone: Optional[str] = None,
                           twitter: Optional[str] = None,
                           linkedin: Optional[str] = None) -> Dict[str, Any]:
        """Enrich person data using FullContact Person API."""
        # Create cache key from available identifiers
        identifiers = [email, phone, twitter, linkedin]
        cache_key = f"person_{hashlib.md5('_'.join(str(i) for i in identifiers if i).encode()).hexdigest()[:16]}"

        # Check cache first
        if self._is_cache_valid(cache_key):
            cached_result = self.contact_cache[cache_key].copy()
            cached_result["cached"] = True
            return cached_result

        # Prepare request data
        request_data = {}

        if email:
            request_data["emails"] = [{"value": email}]
        if phone:
            request_data["phones"] = [{"value": phone}]
        if twitter:
            request_data["profiles"] = [{"service": "twitter", "username": twitter}]
        if linkedin:
            if not request_data.get("profiles"):
                request_data["profiles"] = []
            request_data["profiles"].append({"service": "linkedin", "url": linkedin})

        # Make API call
        data = await self._api_call('POST', "/person.enrich", json_data=request_data)

        # Process and structure the result
        result = {
            "input_identifiers": {
                "email": email,
                "phone": phone,
                "twitter": twitter,
                "linkedin": linkedin
            },
            "person_data": self._process_person_data(data),
            "enriched_at": datetime.utcnow().isoformat(),
            "data_source": "fullcontact"
        }

        # Cache the result
        self._cache_contact(cache_key, result)
        self.api_calls += 1

        return result

    async def _enrich_company(self, email: str) -> Dict[str, Any]:
        """Enrich company data from email domain using FullContact."""
        # Extract domain from email
        domain = email.split('@')[1] if '@' in email else None

        if not domain:
            return {
                "email": email,
                "error": "Invalid email format for company enrichment",
                "scraped_at": datetime.utcnow().isoformat()
            }

        cache_key = f"company_{domain}"

        # Check cache first
        if self._is_cache_valid(cache_key):
            cached_result = self.contact_cache[cache_key].copy()
            cached_result["cached"] = True
            return cached_result

        # FullContact doesn't have a direct company API, but we can use person enrichment
        # and extract company information from the results
        person_result = await self._enrich_person(email=email)

        # Extract company information
        company_info = self._extract_company_from_person(person_result.get('person_data', {}))

        result = {
            "domain": domain,
            "email": email,
            "company_info": company_info,
            "enriched_at": datetime.utcnow().isoformat(),
            "data_source": "fullcontact"
        }

        self._cache_contact(cache_key, result)
        return result

    async def _verify_email(self, email: str) -> Dict[str, Any]:
        """Verify email address using FullContact."""
        # This would use a verification service if available
        # For now, we'll use the person enrichment as a proxy
        result = await self._enrich_person(email=email)

        # Extract verification-like information
        person_data = result.get('person_data', {})
        contact_info = person_data.get('contactInfo', {})

        verification = {
            "email": email,
            "verified": bool(contact_info.get('emails')),
            "confidence": 0.8 if contact_info.get('emails') else 0.0,
            "deliverable": bool(contact_info.get('emails')),
            "fullcontact_enriched": bool(person_data)
        }

        return {
            "email": email,
            "verification": verification,
            "enriched_data_available": bool(person_data),
            "scraped_at": datetime.utcnow().isoformat()
        }

    async def _get_audience_insights(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """Get audience insights (requires paid plan)."""
        if not self.fullcontact_config.audience_enrichment:
            return {
                "error": "Audience enrichment not enabled (requires paid plan)",
                "scraped_at": datetime.utcnow().isoformat()
            }

        # This would implement audience insights API calls
        # For now, return placeholder
        return {
            "audience_insights": "Not implemented - requires FullContact Audience API",
            "target": target,
            "scraped_at": datetime.utcnow().isoformat()
        }

    @retry_on_network_errors(max_attempts=3)
    @retry_on_rate_limits(max_attempts=3)
    async def _api_call(self, method: str, endpoint: str,
                       params: Optional[Dict[str, Any]] = None,
                       json_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make authenticated API call to FullContact."""
        import aiohttp

        url = f"{self.base_url}{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.fullcontact_config.api_key}',
            'Content-Type': 'application/json'
        }

        # Rate limiting
        await self._apply_rate_limiting()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, headers=headers,
                                         params=params, json=json_data) as response:
                    # Update rate limit tracking
                    self.requests_this_month += 1

                    if response.status == 429:
                        # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 86400))
                        logger.warning(f"FullContact rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        return await self._api_call(method, endpoint, params, json_data)

                    if response.status == 404:
                        # No data found
                        return {}

                    response.raise_for_status()
                    return await response.json()

        except Exception as e:
            logger.error(f"FullContact API call failed: {e}")
            raise

    def _process_person_data(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and structure person data from FullContact API."""
        if not api_data:
            return {}

        processed = {
            "full_name": api_data.get('fullName'),
            "given_name": api_data.get('givenName'),
            "family_name": api_data.get('familyName'),
            "age_range": api_data.get('ageRange'),
            "gender": api_data.get('gender'),
            "location": api_data.get('location'),
            "title": api_data.get('title'),
            "organization": api_data.get('organization')
        }

        # Process contact information
        contact_info = api_data.get('contactInfo', {})
        if contact_info:
            processed['contact_info'] = {
                'emails': contact_info.get('emails', []),
                'phones': contact_info.get('phones', [])
            }

        # Process social profiles
        profiles = api_data.get('profiles', [])
        if profiles and self.fullcontact_config.include_social_profiles:
            processed['social_profiles'] = {}
            for profile in profiles:
                service = profile.get('service')
                if service:
                    processed['social_profiles'][service] = {
                        'username': profile.get('username'),
                        'userid': profile.get('userid'),
                        'url': profile.get('url'),
                        'bio': profile.get('bio')
                    }

        # Process demographics
        demographics = api_data.get('demographics', {})
        if demographics and self.fullcontact_config.include_demographics:
            processed['demographics'] = {
                'location_general': demographics.get('locationGeneral'),
                'gender': demographics.get('gender'),
                'age': demographics.get('age')
            }

        # Process photos
        photos = api_data.get('photos', [])
        if photos and self.fullcontact_config.include_photos:
            processed['photos'] = [
                {
                    'url': photo.get('url'),
                    'type': photo.get('type'),
                    'is_primary': photo.get('isPrimary', False)
                }
                for photo in photos
            ]

        return processed

    def _extract_company_from_person(self, person_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract company information from person enrichment data."""
        organization = person_data.get('organization', {})

        if not organization:
            return {}

        return {
            "name": organization.get('name'),
            "title": organization.get('title'),
            "start_date": organization.get('startDate'),
            "end_date": organization.get('endDate'),
            "current": organization.get('current', False),
            "domain": self._extract_domain_from_org(organization)
        }

    def _extract_domain_from_org(self, organization: Dict[str, Any]) -> Optional[str]:
        """Extract domain from organization data."""
        # This would try to extract domain from various fields
        # For now, return None as FullContact may not always provide domain info
        return None

    async def _check_rate_limits(self) -> bool:
        """Check if we're within FullContact rate limits."""
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
        # FullContact allows 500 requests per month = ~16.7 per day = ~1 per 1.7 hours
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
        """Check if cached contact data is still valid."""
        if cache_key not in self.cache_timestamps:
            return False

        cache_time = self.cache_timestamps[cache_key]
        age = (datetime.utcnow() - cache_time).total_seconds()

        return age < self.fullcontact_config.cache_ttl_seconds

    def _cache_contact(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Cache contact data."""
        self.contact_cache[cache_key] = data
        self.cache_timestamps[cache_key] = datetime.utcnow()

        # Limit cache size
        if len(self.contact_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(self.cache_timestamps.keys(),
                               key=lambda k: self.cache_timestamps[k])[:100]
            for key in oldest_keys:
                del self.contact_cache[key]
                del self.cache_timestamps[key]

    async def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate FullContact API result."""
        # Check for API errors
        if 'error' in result and not result.get('rate_limited'):
            return False

        # Check for successful data
        has_data = (
            'person_data' in result or
            'company_info' in result or
            'verification' in result or
            'audience_insights' in result or
            result.get('rate_limited')  # Rate limited is still a valid response
        )

        return bool(has_data)

    def get_fullcontact_metrics(self) -> Dict[str, Any]:
        """Get FullContact API-specific metrics."""
        return {
            **self.get_metrics(),
            'api_calls': self.api_calls,
            'requests_this_month': self.requests_this_month,
            'monthly_limit': self.monthly_limit,
            'cache_size': len(self.contact_cache),
            'api_key_configured': bool(self.fullcontact_config.api_key),
            'features_enabled': {
                'person_enrichment': self.fullcontact_config.person_enrichment,
                'company_enrichment': self.fullcontact_config.company_enrichment,
                'audience_enrichment': self.fullcontact_config.audience_enrichment,
                'social_profiles': self.fullcontact_config.include_social_profiles,
                'demographics': self.fullcontact_config.include_demographics,
                'photos': self.fullcontact_config.include_photos
            },
            'next_reset': (self.month_reset_time + timedelta(days=30)).isoformat()
        }

    def clear_contact_cache(self) -> None:
        """Clear the contact data cache."""
        self.contact_cache.clear()
        self.cache_timestamps.clear()
        logger.info("FullContact contact cache cleared")

    async def cleanup(self) -> None:
        """Cleanup FullContact collector resources."""
        await super().cleanup()

        # Clear caches
        self.contact_cache.clear()
        self.cache_timestamps.clear()

        logger.info("FullContact collector cleaned up")
