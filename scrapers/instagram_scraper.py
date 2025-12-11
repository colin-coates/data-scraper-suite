# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Instagram Scraper Plugin for MJ Data Scraper Suite

Scrapes Instagram profiles, posts, stories, and reels using Graph API and web scraping fallback.
Extracts user profiles, media content, engagement metrics, and follower data.
"""

import asyncio
import logging
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse

from core.base_scraper import BaseScraper, ScraperConfig

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__description__ = "Instagram scraper with Graph API and web scraping capabilities"
__author__ = "MJ Intelligence"
__dependencies__ = ["requests", "beautifulsoup4", "lxml"]


class InstagramConfig(ScraperConfig):
    """Configuration specific to Instagram scraping."""
    app_id: Optional[str] = None
    app_secret: Optional[str] = None
    access_token: Optional[str] = None
    user_token: Optional[str] = None
    use_api_first: bool = True  # Try Graph API first, fallback to scraping
    extract_posts: bool = True
    extract_stories: bool = False  # Requires business account
    extract_reels: bool = True
    extract_igtv: bool = False
    extract_highlights: bool = True
    max_media_items: int = 50
    media_lookback_days: int = 90
    include_comments: bool = False
    include_insights: bool = False  # Requires business/creator account
    extract_followers: bool = False  # Limited by API
    extract_following: bool = False  # Limited by API


class InstagramScraper(BaseScraper):
    """
    Instagram scraper with Graph API and web scraping fallback.
    Handles authentication, rate limiting, and comprehensive content extraction.
    """

    def __init__(self, config: InstagramConfig):
        super().__init__(config)
        self.instagram_config = config

        # API state
        self.api_available = bool(config.access_token or (config.app_id and config.app_secret))
        self.long_lived_token = None
        self.token_expires_at = None

        # Rate limiting
        self.api_call_times = []
        self.scraping_call_times = []

        # Base URLs
        self.graph_api_base = "https://graph.instagram.com"
        self.business_api_base = "https://graph.facebook.com/v18.0"

        # Web scraping URLs
        self.web_base = "https://www.instagram.com"

        # Content type mappings
        self.media_fields = "id,media_type,media_url,permalink,thumbnail_url,caption,timestamp,like_count,comments_count"

        self.profile_count = 0

    async def _execute_scrape(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Instagram data scraping.

        Args:
            target: Contains 'username', 'user_id', 'url', or account handle

        Returns:
            Instagram profile and content data
        """
        username = target.get('username') or target.get('handle')
        user_id = target.get('user_id')
        url = target.get('url')

        # Extract username from various input types
        if not username:
            if url and 'instagram.com' in url:
                username = self._extract_username_from_url(url)
            elif user_id:
                # Resolve user_id to username
                username = await self._resolve_user_id(user_id)

        if not username:
            raise ValueError("Instagram scraper requires 'username', 'handle', 'user_id', or 'url'")

        logger.info(f"Scraping Instagram profile: @{username}")

        # Try Graph API first if configured and available
        if self.instagram_config.use_api_first and self.api_available:
            try:
                return await self._scrape_with_api(username)
            except Exception as e:
                logger.warning(f"Instagram API scraping failed for @{username}: {e}. Falling back to web scraping.")
                if not self.instagram_config.use_api_first:
                    raise  # If API-only mode, fail

        # Fallback to web scraping
        return await self._scrape_with_web_scraping(username)

    async def _scrape_with_api(self, username: str) -> Dict[str, Any]:
        """Scrape Instagram data using Graph API."""
        if not self.api_available:
            raise RuntimeError("Instagram API not configured")

        # Ensure valid token
        await self._ensure_valid_token()

        # Get user ID from username (if we have it)
        user_id = await self._get_user_id_from_username(username)
        if not user_id:
            raise ValueError(f"Could not resolve Instagram username @{username}")

        # Get user profile info
        profile_data = await self._api_get_user_profile(user_id)

        # Build comprehensive profile data
        result = {
            "user_id": user_id,
            "username": username,
            "full_name": profile_data.get('name'),
            "biography": profile_data.get('biography'),
            "website": profile_data.get('website'),
            "profile_picture_url": profile_data.get('profile_picture_url'),
            "is_business_account": profile_data.get('is_business_account', False),
            "is_verified": profile_data.get('is_verified', False),
            "followers_count": profile_data.get('followers_count'),
            "following_count": profile_data.get('follows_count'),
            "media_count": profile_data.get('media_count'),
            "api_used": True,
            "scraped_at": datetime.utcnow().isoformat()
        }

        # Get media content
        if self.instagram_config.extract_posts:
            posts = await self._api_get_user_media(user_id)
            result['recent_media'] = posts

        # Get insights if available (business accounts only)
        if self.instagram_config.include_insights and profile_data.get('is_business_account'):
            try:
                insights = await self._api_get_insights(user_id)
                result['insights'] = insights
            except Exception as e:
                logger.warning(f"Could not get insights for @{username}: {e}")

        # Get stories if available
        if self.instagram_config.extract_stories:
            try:
                stories = await self._api_get_stories(user_id)
                result['stories'] = stories
            except Exception as e:
                logger.warning(f"Could not get stories for @{username}: {e}")

        self.profile_count += 1
        return result

    async def _api_get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get Instagram user profile via Graph API."""
        # For basic display API
        if self.instagram_config.app_id:
            url = f"{self.graph_api_base}/me"
            params = {
                'fields': 'id,username,account_type,media_count',
                'access_token': self.long_lived_token or self.instagram_config.access_token
            }
        else:
            # For business/creator accounts via Facebook Graph API
            url = f"{self.business_api_base}/{user_id}"
            params = {
                'fields': 'biography,followers_count,follows_count,id,ig_id,media_count,name,profile_picture_url,username,website,is_verified,is_business_account',
                'access_token': self.long_lived_token or self.instagram_config.access_token
            }

        return await self._api_request('GET', url, params=params)

    async def _api_get_user_media(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's media content via Graph API."""
        url = f"{self.business_api_base}/{user_id}/media"
        params = {
            'fields': self.media_fields,
            'limit': min(self.instagram_config.max_media_items, 100),
            'since': int((datetime.utcnow() - timedelta(days=self.instagram_config.media_lookback_days)).timestamp()),
            'access_token': self.long_lived_token or self.instagram_config.access_token
        }

        response = await self._api_request('GET', url, params=params)
        return self._process_media_items(response.get('data', []))

    async def _api_get_stories(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user stories via Graph API."""
        url = f"{self.business_api_base}/{user_id}/stories"
        params = {
            'fields': 'id,media_type,media_url,permalink,timestamp',
            'access_token': self.long_lived_token or self.instagram_config.access_token
        }

        response = await self._api_request('GET', url, params=params)
        return response.get('data', [])

    async def _api_get_insights(self, user_id: str) -> Dict[str, Any]:
        """Get account insights via Graph API (business accounts only)."""
        url = f"{self.business_api_base}/{user_id}/insights"
        params = {
            'metric': 'follower_count,impressions,reach,profile_views',
            'period': 'day',
            'access_token': self.long_lived_token or self.instagram_config.access_token
        }

        response = await self._api_request('GET', url, params=params)
        return self._process_insights(response.get('data', []))

    def _process_media_items(self, media_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean media item data."""
        processed = []

        for item in media_items:
            processed_item = {
                "id": item.get('id'),
                "media_type": item.get('media_type'),
                "media_url": item.get('media_url'),
                "permalink": item.get('permalink'),
                "thumbnail_url": item.get('thumbnail_url'),
                "caption": item.get('caption'),
                "timestamp": item.get('timestamp'),
                "like_count": item.get('like_count', 0),
                "comments_count": item.get('comments_count', 0)
            }

            # Filter by content type
            media_type = item.get('media_type')
            if media_type == 'VIDEO' and not self.instagram_config.extract_reels:
                continue
            elif media_type == 'IMAGE' and not self.instagram_config.extract_posts:
                continue

            processed.append(processed_item)

        return processed

    def _process_insights(self, insights_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process insights data."""
        processed = {}

        for metric in insights_data:
            metric_name = metric.get('name')
            values = metric.get('values', [])
            if values:
                latest_value = values[-1].get('value', 0)
                processed[metric_name] = latest_value

        return processed

    async def _api_request(self, method: str, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make authenticated request to Instagram Graph API."""
        import aiohttp

        # Check rate limits
        await self._check_api_rate_limits()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, params=params) as response:
                    # Update rate limit tracking
                    self._update_api_rate_limits(response.headers)

                    if response.status == 429:
                        # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Instagram API rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        return await self._api_request(method, url, params)

                    response.raise_for_status()
                    return await response.json()

        except Exception as e:
            logger.error(f"Instagram API request failed: {e}")
            raise

    async def _scrape_with_web_scraping(self, username: str) -> Dict[str, Any]:
        """Scrape Instagram data using web scraping fallback."""
        logger.info(f"Using web scraping fallback for @{username}")

        # This is a mock implementation
        # In production, this would:
        # 1. Make HTTP requests to instagram.com
        # 2. Parse JSON data from HTML/script tags
        # 3. Handle dynamic content loading
        # 4. Extract structured data

        return {
            "username": username,
            "full_name": f"User {username}",
            "biography": "Profile data extracted via web scraping",
            "is_verified": False,
            "followers_count": None,
            "following_count": None,
            "media_count": None,
            "api_used": False,
            "fallback_mode": True,
            "scraped_at": datetime.utcnow().isoformat(),
            "limited_data": True,
            "error": "API not available, limited data extracted via web scraping"
        }

    async def _get_user_id_from_username(self, username: str) -> Optional[str]:
        """Resolve Instagram username to user ID."""
        if not self.api_available:
            return None

        try:
            # Try to get user by username (limited API support)
            # This might require web scraping or search API
            logger.warning("Instagram username to ID resolution limited by API")
            return None
        except Exception as e:
            logger.warning(f"Failed to resolve username {username}: {e}")
            return None

    async def _resolve_user_id(self, user_id: str) -> Optional[str]:
        """Resolve Instagram user ID to username."""
        if not self.api_available:
            return None

        try:
            # Get user info by ID
            user_data = await self._api_get_user_profile(user_id)
            return user_data.get('username')
        except Exception as e:
            logger.warning(f"Failed to resolve user ID {user_id}: {e}")
            return None

    def _extract_username_from_url(self, url: str) -> str:
        """Extract username from Instagram URL."""
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')

        # Handle different Instagram URL formats
        if path_parts and not path_parts[0].startswith('@'):
            return path_parts[0].lstrip('@')

        return "unknown"

    async def _ensure_valid_token(self) -> None:
        """Ensure we have a valid access token."""
        if not self.instagram_config.access_token and not self.long_lived_token:
            if self.instagram_config.app_id and self.instagram_config.app_secret:
                # Generate app access token for basic display API
                self.instagram_config.access_token = f"{self.instagram_config.app_id}|{self.instagram_config.app_secret}"
            else:
                raise RuntimeError("Instagram API requires access_token or app_id + app_secret")

        # Check if we need to exchange for long-lived token
        if (self.instagram_config.user_token and
            (not self.long_lived_token or
             (self.token_expires_at and datetime.utcnow() > self.token_expires_at))):

            await self._exchange_for_long_lived_token()

    async def _exchange_for_long_lived_token(self) -> None:
        """Exchange short-lived token for long-lived token."""
        import aiohttp

        params = {
            'grant_type': 'ig_exchange_token',
            'client_secret': self.instagram_config.app_secret,
            'access_token': self.instagram_config.user_token
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.graph_api_base}/access_token", params=params) as response:
                    response.raise_for_status()
                    token_data = await response.json()

                    self.long_lived_token = token_data['access_token']
                    expires_in = token_data.get('expires_in', 5184000)  # 60 days default
                    self.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

                    logger.info("Exchanged for long-lived Instagram token")

        except Exception as e:
            logger.warning(f"Failed to exchange for long-lived token: {e}")
            # Continue with short-lived token

    async def _check_api_rate_limits(self) -> None:
        """Check and handle API rate limits."""
        current_time = datetime.utcnow().timestamp()

        # Clean old API call times (keep last hour)
        self.api_call_times = [t for t in self.api_call_times if current_time - t < 3600]

        # Instagram allows ~200 calls per hour for most endpoints
        if len(self.api_call_times) >= 180:  # Be conservative
            oldest_call = min(self.api_call_times)
            wait_time = 3600 - (current_time - oldest_call)
            if wait_time > 0:
                logger.info(f"Approaching Instagram API rate limit, waiting {wait_time:.0f}s")
                await asyncio.sleep(wait_time)

        # Track this call
        self.api_call_times.append(current_time)

    def _update_api_rate_limits(self, headers: Dict[str, str]) -> None:
        """Update rate limit tracking from response headers."""
        # Instagram provides rate limit info in headers
        # This is a simplified implementation
        pass

    async def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate Instagram scraping result."""
        # At minimum, we need a username
        if not result.get('username'):
            return False

        # If in fallback mode, accept limited data
        if result.get('fallback_mode'):
            return True

        # For API results, check for basic profile info
        has_basic_info = result.get('full_name') or result.get('biography')
        return bool(has_basic_info)

    def get_instagram_metrics(self) -> Dict[str, Any]:
        """Get Instagram-specific metrics."""
        return {
            **self.get_metrics(),
            'profiles_scraped': self.profile_count,
            'api_available': self.api_available,
            'token_type': 'long_lived' if self.long_lived_token else 'short_lived',
            'token_expires_soon': self.token_expires_at and (self.token_expires_at - datetime.utcnow()).days < 7,
            'features_enabled': {
                'posts': self.instagram_config.extract_posts,
                'stories': self.instagram_config.extract_stories,
                'reels': self.instagram_config.extract_reels,
                'igtv': self.instagram_config.extract_igtv,
                'highlights': self.instagram_config.extract_highlights,
                'insights': self.instagram_config.include_insights
            }
        }

    async def cleanup(self) -> None:
        """Cleanup Instagram scraper resources."""
        await super().cleanup()

        # Clear tokens and sensitive data
        self.long_lived_token = None
        self.token_expires_at = None
        self.api_call_times.clear()
        self.scraping_call_times.clear()
        self.profile_count = 0

        logger.info("Instagram scraper cleaned up")
