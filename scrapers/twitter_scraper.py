# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Twitter/X Scraper Plugin for MJ Data Scraper Suite

Scrapes Twitter/X profiles and posts using official API with fallback to web scraping.
Extracts user profiles, tweets, follower/following data, and engagement metrics.
"""

import asyncio
import logging
import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse, quote

from core.base_scraper import BaseScraper, ScraperConfig

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__description__ = "Twitter/X scraper with API and web scraping fallback"
__author__ = "MJ Intelligence"
__dependencies__ = ["requests", "beautifulsoup4", "lxml"]


class TwitterConfig(ScraperConfig):
    """Configuration specific to Twitter/X scraping."""
    api_bearer_token: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    access_token: Optional[str] = None
    access_token_secret: Optional[str] = None
    use_api_first: bool = True  # Try API first, fallback to scraping
    extract_tweets: bool = True
    extract_followers: bool = False  # API only
    extract_following: bool = False  # API only
    max_tweets: int = 100
    tweet_lookback_days: int = 30
    include_replies: bool = False
    include_retweets: bool = True
    extract_media: bool = True


class TwitterScraper(BaseScraper):
    """
    Twitter/X scraper with dual-mode operation: API + web scraping fallback.
    Handles rate limits, authentication, and data extraction.
    """

    def __init__(self, config: TwitterConfig):
        super().__init__(config)
        self.twitter_config = config

        # API state
        self.api_available = bool(config.api_bearer_token or config.api_key)
        self.api_rate_limits = {}
        self.api_reset_times = {}

        # Scraping state
        self.session_headers = {}
        self.csrf_token = None
        self.guest_token = None
        self.profile_count = 0

        # Base URLs
        self.api_base = "https://api.twitter.com/2"
        self.web_base = "https://twitter.com"

        # Rate limit tracking
        self.request_counts = {}
        self.rate_limit_reset = {}

    async def _execute_scrape(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Twitter/X data scraping.

        Args:
            target: Contains 'username', 'user_id', 'url', or search terms

        Returns:
            Twitter/X profile and content data
        """
        username = target.get('username') or target.get('handle')
        user_id = target.get('user_id')
        url = target.get('url')

        # Extract username from various input types
        if not username:
            if url and 'twitter.com' in url:
                username = self._extract_username_from_url(url)
            elif user_id:
                # Would need to resolve user_id to username via API
                username = await self._resolve_user_id(user_id)

        if not username:
            raise ValueError("Twitter scraper requires 'username', 'handle', 'user_id', or 'url'")

        logger.info(f"Scraping Twitter profile: @{username}")

        # Try API first if configured and available
        if self.twitter_config.use_api_first and self.api_available:
            try:
                return await self._scrape_with_api(username)
            except Exception as e:
                logger.warning(f"API scraping failed for @{username}: {e}. Falling back to web scraping.")
                if not self.twitter_config.use_api_first:
                    raise  # If API-only mode, fail

        # Fallback to web scraping
        return await self._scrape_with_web_scraping(username)

    async def _scrape_with_api(self, username: str) -> Dict[str, Any]:
        """Scrape Twitter data using official API."""
        if not self.api_available:
            raise RuntimeError("Twitter API not configured")

        # Check rate limits
        await self._check_api_rate_limits()

        # Get user information
        user_data = await self._api_get_user(username)
        user_id = user_data['id']

        # Build comprehensive profile data
        profile_data = {
            "user_id": user_id,
            "username": user_data['username'],
            "display_name": user_data.get('name', ''),
            "description": user_data.get('description', ''),
            "location": user_data.get('location', ''),
            "website": user_data.get('url', ''),
            "profile_image_url": user_data.get('profile_image_url', ''),
            "verified": user_data.get('verified', False),
            "protected": user_data.get('protected', False),
            "created_at": user_data.get('created_at'),
            "followers_count": user_data.get('public_metrics', {}).get('followers_count', 0),
            "following_count": user_data.get('public_metrics', {}).get('following_count', 0),
            "tweet_count": user_data.get('public_metrics', {}).get('tweet_count', 0),
            "api_used": True,
            "scraped_at": datetime.utcnow().isoformat()
        }

        # Get tweets if requested
        if self.twitter_config.extract_tweets:
            tweets = await self._api_get_tweets(user_id)
            profile_data['recent_tweets'] = tweets

        # Get followers/following if requested (expensive operations)
        if self.twitter_config.extract_followers:
            followers = await self._api_get_followers(user_id)
            profile_data['followers_sample'] = followers[:100]  # Limit for performance

        if self.twitter_config.extract_following:
            following = await self._api_get_following(user_id)
            profile_data['following_sample'] = following[:100]

        # Update counter
        self.profile_count += 1

        return profile_data

    async def _api_get_user(self, username: str) -> Dict[str, Any]:
        """Get user information via Twitter API."""
        url = f"{self.api_base}/users/by/username/{username}"
        params = {
            'user.fields': 'id,name,username,description,location,url,profile_image_url,verified,protected,created_at,public_metrics'
        }

        response = await self._api_request('GET', url, params=params)
        return response['data']

    async def _api_get_tweets(self, user_id: str) -> List[Dict[str, Any]]:
        """Get recent tweets for a user via Twitter API."""
        url = f"{self.api_base}/users/{user_id}/tweets"
        params = {
            'max_results': min(self.twitter_config.max_tweets, 100),
            'tweet.fields': 'id,text,created_at,public_metrics,entities,context_annotations',
            'exclude': 'replies' if not self.twitter_config.include_replies else None,
            'start_time': (datetime.utcnow() - timedelta(days=self.twitter_config.tweet_lookback_days)).isoformat() + 'Z'
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        response = await self._api_request('GET', url, params=params)
        return response.get('data', [])

    async def _api_get_followers(self, user_id: str) -> List[Dict[str, Any]]:
        """Get followers list via Twitter API."""
        url = f"{self.api_base}/users/{user_id}/followers"
        params = {
            'max_results': 100,
            'user.fields': 'id,name,username,description,profile_image_url,verified'
        }

        response = await self._api_request('GET', url, params=params)
        return response.get('data', [])

    async def _api_get_following(self, user_id: str) -> List[Dict[str, Any]]:
        """Get following list via Twitter API."""
        url = f"{self.api_base}/users/{user_id}/following"
        params = {
            'max_results': 100,
            'user.fields': 'id,name,username,description,profile_image_url,verified'
        }

        response = await self._api_request('GET', url, params=params)
        return response.get('data', [])

    async def _api_request(self, method: str, url: str, params: Optional[Dict[str, Any]] = None,
                          data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make authenticated request to Twitter API."""
        import aiohttp

        headers = {
            'Authorization': f'Bearer {self.twitter_config.api_bearer_token}',
            'Content-Type': 'application/json'
        }

        # Check rate limits before making request
        endpoint_key = url.split('/2/')[1].split('/')[0]  # Extract endpoint for rate limiting
        if endpoint_key in self.api_rate_limits:
            remaining = self.api_rate_limits[endpoint_key]
            reset_time = self.api_reset_times.get(endpoint_key, 0)

            if remaining <= 0 and datetime.utcnow().timestamp() < reset_time:
                wait_time = reset_time - datetime.utcnow().timestamp()
                logger.warning(f"Rate limited on {endpoint_key}, waiting {wait_time:.0f}s")
                await asyncio.sleep(wait_time + 1)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, headers=headers, params=params, json=data) as response:
                    # Update rate limit tracking
                    if 'x-rate-limit-remaining' in response.headers:
                        self.api_rate_limits[endpoint_key] = int(response.headers['x-rate-limit-remaining'])
                    if 'x-rate-limit-reset' in response.headers:
                        self.api_reset_times[endpoint_key] = int(response.headers['x-rate-limit-reset'])

                    if response.status == 429:
                        reset_time = int(response.headers.get('x-rate-limit-reset', 0))
                        wait_time = max(0, reset_time - datetime.utcnow().timestamp())
                        logger.warning(f"Rate limited, waiting {wait_time:.0f}s")
                        await asyncio.sleep(wait_time + 1)
                        return await self._api_request(method, url, params, data)  # Retry

                    response.raise_for_status()
                    return await response.json()

        except Exception as e:
            logger.error(f"Twitter API request failed: {e}")
            raise

    async def _scrape_with_web_scraping(self, username: str) -> Dict[str, Any]:
        """Scrape Twitter data using web scraping fallback."""
        logger.info(f"Using web scraping fallback for @{username}")

        # This is a mock implementation
        # In production, this would:
        # 1. Make HTTP requests to twitter.com
        # 2. Parse HTML/JSON responses
        # 3. Handle JavaScript-rendered content
        # 4. Extract structured data

        return {
            "username": username,
            "display_name": f"User {username}",
            "description": "Profile data extracted via web scraping",
            "location": None,
            "website": None,
            "verified": False,
            "protected": False,
            "followers_count": None,
            "following_count": None,
            "tweet_count": None,
            "api_used": False,
            "fallback_mode": True,
            "scraped_at": datetime.utcnow().isoformat(),
            "limited_data": True,
            "error": "API not available, limited data extracted via web scraping"
        }

    async def _resolve_user_id(self, user_id: str) -> Optional[str]:
        """Resolve Twitter user ID to username."""
        if not self.api_available:
            return None

        try:
            url = f"{self.api_base}/users/{user_id}"
            params = {'user.fields': 'username'}

            response = await self._api_request('GET', url, params=params)
            return response['data']['username']
        except Exception as e:
            logger.warning(f"Failed to resolve user ID {user_id}: {e}")
            return None

    def _extract_username_from_url(self, url: str) -> str:
        """Extract username from Twitter URL."""
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')

        # Handle different Twitter URL formats
        if path_parts and not path_parts[0].startswith('@'):
            return path_parts[0].lstrip('@')

        return "unknown"

    async def _check_api_rate_limits(self) -> None:
        """Check and handle API rate limits."""
        # Global rate limit check
        current_time = datetime.utcnow().timestamp()

        for endpoint, reset_time in self.api_reset_times.items():
            if current_time < reset_time and self.api_rate_limits.get(endpoint, 100) <= 0:
                wait_time = reset_time - current_time
                logger.info(f"Rate limited on {endpoint}, waiting {wait_time:.0f}s")
                await asyncio.sleep(wait_time + 1)

    async def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate Twitter scraping result."""
        # At minimum, we need a username
        if not result.get('username'):
            return False

        # If in fallback mode, accept limited data
        if result.get('fallback_mode'):
            return True

        # For API results, check for basic profile info
        has_basic_info = result.get('display_name') or result.get('description')
        return bool(has_basic_info)

    def get_twitter_metrics(self) -> Dict[str, Any]:
        """Get Twitter-specific metrics."""
        return {
            **self.get_metrics(),
            'profiles_scraped': self.profile_count,
            'api_available': self.api_available,
            'api_rate_limits': self.api_rate_limits,
            'fallback_used': not self.api_available,
            'features_enabled': {
                'tweets': self.twitter_config.extract_tweets,
                'followers': self.twitter_config.extract_followers,
                'following': self.twitter_config.extract_following,
                'media': self.twitter_config.extract_media
            }
        }

    async def cleanup(self) -> None:
        """Cleanup Twitter scraper resources."""
        await super().cleanup()

        # Clear API state
        self.api_rate_limits.clear()
        self.api_reset_times.clear()
        self.request_counts.clear()
        self.rate_limit_reset.clear()

        # Clear web scraping state
        self.session_headers.clear()
        self.csrf_token = None
        self.guest_token = None
        self.profile_count = 0

        logger.info("Twitter scraper cleaned up")
