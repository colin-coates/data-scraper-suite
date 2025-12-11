# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
LinkedIn API Collector for MJ Data Scraper Suite

Official LinkedIn API integration for professional data enrichment.
Provides access to LinkedIn's official APIs for organizations, people, and jobs.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from urllib.parse import urlencode

from core.base_scraper import BaseScraper, ScraperConfig
from core.retry_utils import retry_async, RetryConfig, retry_on_rate_limits, retry_on_network_errors

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__description__ = "Official LinkedIn API integration for professional data"
__author__ = "MJ Intelligence"
__dependencies__ = ["requests", "oauthlib", "requests_oauthlib"]


class LinkedInAPIConfig(ScraperConfig):
    """Configuration for LinkedIn API integration."""
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    redirect_uri: str = "http://localhost:8080/callback"
    token_expires_at: Optional[datetime] = None
    extract_organizations: bool = True
    extract_people: bool = True
    extract_jobs: bool = False
    max_results_per_call: int = 50
    cache_ttl_seconds: int = 3600  # 1 hour


class LinkedInAPICollector(BaseScraper):
    """
    Official LinkedIn API collector with OAuth2 authentication.
    Provides access to LinkedIn's professional data APIs.
    """

    def __init__(self, config: LinkedInAPIConfig):
        super().__init__(config)
        self.linkedin_config = config

        # API endpoints
        self.base_url = "https://api.linkedin.com/v2"
        self.auth_url = "https://www.linkedin.com/oauth/v2/authorization"
        self.token_url = "https://www.linkedin.com/oauth/v2/accessToken"

        # Rate limiting (LinkedIn allows 100 requests per day for most endpoints)
        self.daily_limit = 100
        self.requests_today = 0
        self.last_reset_date = datetime.utcnow().date()

        # Caching
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}

        # OAuth2 state
        self.auth_state = None

        self.api_calls = 0

    async def _execute_scrape(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute LinkedIn API data collection.

        Args:
            target: Contains 'entity_type', 'entity_id', 'search_query', etc.

        Returns:
            LinkedIn API data
        """
        entity_type = target.get('entity_type', 'organization')
        entity_id = target.get('entity_id')
        search_query = target.get('search_query')

        # Ensure we have valid authentication
        await self._ensure_authenticated()

        # Check rate limits
        if not await self._check_rate_limits():
            return {
                "error": "Rate limit exceeded",
                "rate_limited": True,
                "reset_date": (datetime.utcnow().date() + timedelta(days=1)).isoformat(),
                "scraped_at": datetime.utcnow().isoformat()
            }

        try:
            if entity_type == 'organization' and entity_id:
                return await self._get_organization_data(entity_id)
            elif entity_type == 'person' and entity_id:
                return await self._get_person_data(entity_id)
            elif entity_type == 'search' and search_query:
                return await self._search_entities(search_query, target.get('entity_types', ['organization']))
            elif entity_type == 'jobs':
                return await self._get_jobs_data(target)
            else:
                raise ValueError(f"Unsupported entity_type: {entity_type}")

        except Exception as e:
            logger.error(f"LinkedIn API collection failed: {e}")
            raise

    async def _get_organization_data(self, org_id: str) -> Dict[str, Any]:
        """Get organization data from LinkedIn API."""
        cache_key = f"org_{org_id}"

        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        # Make API call
        endpoint = f"/organizations/{org_id}"
        params = {
            'projection': '(id,localizedName,vanityName,localizedWebsite,foundedOn,locations,organizationTypes,industries,staffCountRange,specialities,description)'
        }

        data = await self._api_call('GET', endpoint, params=params)

        # Get additional data
        try:
            # Get follower count
            follower_data = await self._api_call('GET', f"/organizations/{org_id}/organizationFollowers",
                                                params={'q': 'organization'})
            if 'elements' in follower_data and follower_data['elements']:
                data['follower_count'] = follower_data['elements'][0].get('followerCount', 0)
        except Exception as e:
            logger.debug(f"Could not get follower count for org {org_id}: {e}")

        # Cache the result
        self._cache_result(cache_key, data)
        self.api_calls += 1

        return data

    async def _get_person_data(self, person_id: str) -> Dict[str, Any]:
        """Get person data from LinkedIn API (limited by privacy settings)."""
        cache_key = f"person_{person_id}"

        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        # LinkedIn Person API has significant privacy restrictions
        # This will only work for authenticated user's connections or public data
        endpoint = f"/people/{person_id}"
        params = {
            'projection': '(id,localizedFirstName,localizedLastName,vanityName,profilePicture,headline,publicProfileUrl)'
        }

        try:
            data = await self._api_call('GET', endpoint, params=params)
            self._cache_result(cache_key, data)
            self.api_calls += 1
            return data
        except Exception as e:
            logger.warning(f"Person data access limited by LinkedIn privacy: {e}")
            return {
                "id": person_id,
                "error": "Person data not accessible due to LinkedIn privacy restrictions",
                "public_only": True
            }

    async def _search_entities(self, query: str, entity_types: List[str]) -> Dict[str, Any]:
        """Search for entities using LinkedIn Search API."""
        results = {}

        for entity_type in entity_types:
            try:
                if entity_type == 'organization':
                    search_results = await self._search_organizations(query)
                    results['organizations'] = search_results
                elif entity_type == 'person':
                    search_results = await self._search_people(query)
                    results['people'] = search_results
            except Exception as e:
                logger.error(f"Search failed for {entity_type}: {e}")
                results[f'{entity_type}s'] = {"error": str(e)}

        return {
            "query": query,
            "entity_types": entity_types,
            "results": results,
            "total_results": sum(len(v) if isinstance(v, list) else 0 for v in results.values()),
            "scraped_at": datetime.utcnow().isoformat()
        }

    async def _search_organizations(self, query: str) -> List[Dict[str, Any]]:
        """Search for organizations."""
        endpoint = "/organizationSearch"
        params = {
            'q': 'companies',
            'keywords': query,
            'count': min(self.linkedin_config.max_results_per_call, 50)
        }

        data = await self._api_call('GET', endpoint, params=params)
        self.api_calls += 1

        return data.get('elements', [])

    async def _search_people(self, query: str) -> List[Dict[str, Any]]:
        """Search for people (limited results due to privacy)."""
        endpoint = "/peopleSearch"
        params = {
            'keywords': query,
            'count': min(self.linkedin_config.max_results_per_call, 25)  # More restricted for people
        }

        data = await self._api_call('GET', endpoint, params=params)
        self.api_calls += 1

        return data.get('elements', [])

    async def _get_jobs_data(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """Get job postings data."""
        # Note: Jobs API requires special permissions and has restrictions
        endpoint = "/jobs"
        params = {
            'q': 'search',
            'keywords': target.get('keywords', ''),
            'location': target.get('location', ''),
            'count': min(target.get('limit', self.linkedin_config.max_results_per_call), 50)
        }

        # Remove empty parameters
        params = {k: v for k, v in params.items() if v}

        try:
            data = await self._api_call('GET', endpoint, params=params)
            self.api_calls += 1

            return {
                "jobs": data.get('elements', []),
                "total": len(data.get('elements', [])),
                "keywords": target.get('keywords'),
                "location": target.get('location'),
                "scraped_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.warning(f"Jobs API access failed: {e}")
            return {
                "error": "Jobs API access requires special permissions",
                "details": str(e)
            }

    @retry_on_network_errors(max_attempts=3)
    @retry_on_rate_limits(max_attempts=5)
    async def _api_call(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None,
                       data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make authenticated API call to LinkedIn."""
        import aiohttp

        url = f"{self.base_url}{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.linkedin_config.access_token}',
            'X-Restli-Protocol-Version': '2.0.0',
            'Content-Type': 'application/json'
        }

        # Rate limiting
        await self._apply_rate_limiting()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, headers=headers, params=params, json=data) as response:
                    # Update rate limit tracking
                    self.requests_today += 1

                    if response.status == 401:
                        # Token expired, try to refresh
                        await self._refresh_token()
                        # Retry once with new token
                        headers['Authorization'] = f'Bearer {self.linkedin_config.access_token}'
                        async with session.request(method, url, headers=headers, params=params, json=data) as retry_response:
                            retry_response.raise_for_status()
                            return await retry_response.json()

                    response.raise_for_status()
                    return await response.json()

        except Exception as e:
            logger.error(f"LinkedIn API call failed: {e}")
            raise

    async def _ensure_authenticated(self) -> None:
        """Ensure we have valid authentication."""
        if not self.linkedin_config.access_token:
            raise RuntimeError("LinkedIn API requires access_token")

        # Check if token is expired
        if self.linkedin_config.token_expires_at:
            if datetime.utcnow() >= self.linkedin_config.token_expires_at:
                await self._refresh_token()

    async def _refresh_token(self) -> None:
        """Refresh OAuth2 access token."""
        if not self.linkedin_config.refresh_token:
            raise RuntimeError("No refresh token available")

        import aiohttp

        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.linkedin_config.refresh_token,
            'client_id': self.linkedin_config.client_id,
            'client_secret': self.linkedin_config.client_secret
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.token_url, data=data) as response:
                    response.raise_for_status()
                    token_data = await response.json()

                    self.linkedin_config.access_token = token_data['access_token']
                    if 'refresh_token' in token_data:
                        self.linkedin_config.refresh_token = token_data['refresh_token']

                    expires_in = token_data.get('expires_in', 5184000)  # 60 days default
                    self.linkedin_config.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

                    logger.info("LinkedIn access token refreshed")

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise

    def generate_authorization_url(self, scopes: List[str] = None) -> str:
        """Generate OAuth2 authorization URL."""
        if not self.linkedin_config.client_id:
            raise RuntimeError("client_id required for authorization")

        scopes = scopes or ['r_liteprofile', 'r_emailaddress', 'w_member_social']
        scope_string = ' '.join(scopes)

        params = {
            'response_type': 'code',
            'client_id': self.linkedin_config.client_id,
            'redirect_uri': self.linkedin_config.redirect_uri,
            'state': self._generate_state(),
            'scope': scope_string
        }

        return f"{self.auth_url}?{urlencode(params)}"

    async def exchange_code_for_token(self, authorization_code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        import aiohttp

        data = {
            'grant_type': 'authorization_code',
            'code': authorization_code,
            'redirect_uri': self.linkedin_config.redirect_uri,
            'client_id': self.linkedin_config.client_id,
            'client_secret': self.linkedin_config.client_secret
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.token_url, data=data) as response:
                response.raise_for_status()
                token_data = await response.json()

                # Update configuration
                self.linkedin_config.access_token = token_data['access_token']
                if 'refresh_token' in token_data:
                    self.linkedin_config.refresh_token = token_data['refresh_token']

                expires_in = token_data.get('expires_in', 5184000)
                self.linkedin_config.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

                return token_data

    async def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits."""
        # Reset counter if it's a new day
        current_date = datetime.utcnow().date()
        if current_date != self.last_reset_date:
            self.requests_today = 0
            self.last_reset_date = current_date

        return self.requests_today < self.daily_limit

    async def _apply_rate_limiting(self) -> None:
        """Apply rate limiting to prevent hitting limits."""
        # Simple rate limiting: max 2 requests per second
        current_time = time.time()
        if hasattr(self, '_last_request_time'):
            time_diff = current_time - self._last_request_time
            if time_diff < 0.5:  # Less than 500ms since last request
                await asyncio.sleep(0.5 - time_diff)

        self._last_request_time = current_time

    def _generate_state(self) -> str:
        """Generate OAuth2 state parameter."""
        import secrets
        self.auth_state = secrets.token_urlsafe(32)
        return self.auth_state

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache_timestamps:
            return False

        cache_time = self.cache_timestamps[cache_key]
        age = (datetime.utcnow() - cache_time).total_seconds()

        return age < self.linkedin_config.cache_ttl_seconds

    def _cache_result(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Cache API result."""
        self.cache[cache_key] = data
        self.cache_timestamps[cache_key] = datetime.utcnow()

        # Limit cache size
        if len(self.cache) > 100:
            oldest_key = min(self.cache_timestamps.keys(),
                           key=lambda k: self.cache_timestamps[k])
            del self.cache[oldest_key]
            del self.cache_timestamps[oldest_key]

    async def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate LinkedIn API result."""
        # Check for API errors
        if 'error' in result:
            return False

        # Check for successful data
        has_data = (
            'id' in result or
            'elements' in result or
            'organizations' in result or
            'people' in result or
            'jobs' in result
        )

        return bool(has_data)

    def get_linkedin_api_metrics(self) -> Dict[str, Any]:
        """Get LinkedIn API-specific metrics."""
        return {
            **self.get_metrics(),
            'api_calls': self.api_calls,
            'requests_today': self.requests_today,
            'daily_limit': self.daily_limit,
            'cache_size': len(self.cache),
            'authenticated': bool(self.linkedin_config.access_token),
            'token_expires_soon': (
                self.linkedin_config.token_expires_at and
                (self.linkedin_config.token_expires_at - datetime.utcnow()).days < 7
            ),
            'features_enabled': {
                'organizations': self.linkedin_config.extract_organizations,
                'people': self.linkedin_config.extract_people,
                'jobs': self.linkedin_config.extract_jobs
            }
        }

    async def cleanup(self) -> None:
        """Cleanup LinkedIn API collector resources."""
        await super().cleanup()

        # Clear sensitive data
        self.cache.clear()
        self.cache_timestamps.clear()
        self.auth_state = None

        logger.info("LinkedIn API collector cleaned up")
