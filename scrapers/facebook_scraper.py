# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Facebook Scraper Plugin for MJ Data Scraper Suite

Scrapes Facebook profiles and pages using Graph API with proper authentication.
Extracts user profiles, posts, events, and engagement metrics.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse

from core.base_scraper import BaseScraper, ScraperConfig

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__description__ = "Facebook Graph API scraper for profiles and pages"
__author__ = "MJ Intelligence"
__dependencies__ = ["requests", "facebook-sdk"]


class FacebookConfig(ScraperConfig):
    """Configuration specific to Facebook scraping."""
    app_id: Optional[str] = None
    app_secret: Optional[str] = None
    access_token: Optional[str] = None
    user_token: Optional[str] = None  # For user-specific data
    page_token: Optional[str] = None  # For page access
    extract_posts: bool = True
    extract_events: bool = True
    extract_photos: bool = False
    extract_friends: bool = False  # Requires user_friends permission
    extract_groups: bool = False
    max_posts: int = 100
    post_lookback_days: int = 90
    include_comments: bool = False
    include_reactions: bool = True
    extract_page_insights: bool = False  # Requires page access token


class FacebookScraper(BaseScraper):
    """
    Facebook scraper using Graph API with comprehensive data extraction.
    Handles authentication, rate limiting, and structured data retrieval.
    """

    def __init__(self, config: FacebookConfig):
        super().__init__(config)
        self.facebook_config = config

        # API state
        self.api_available = bool(config.access_token or (config.app_id and config.app_secret))
        self.long_lived_token = None
        self.token_expires_at = None

        # Rate limiting
        self.request_count = 0
        self.rate_limit_reset = 0
        self.api_call_times = []

        # Base URLs
        self.graph_api_base = "https://graph.facebook.com/v18.0"

        # Field mappings for different object types
        self.profile_fields = "id,name,about,birthday,email,education,work,location,website,hometown,relationship_status"
        self.post_fields = "id,message,created_time,type,permalink_url,attachments,comments.summary(true),reactions.summary(true)"
        self.page_fields = "id,name,about,category,description,website,location,phone,emails,fan_count"

        self.profile_count = 0

    async def _execute_scrape(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Facebook data scraping via Graph API.

        Args:
            target: Contains 'profile_id', 'page_id', 'url', or 'username'

        Returns:
            Facebook profile/page data
        """
        profile_id = target.get('profile_id') or target.get('page_id')
        url = target.get('url')
        username = target.get('username')

        # Extract ID from various input types
        if not profile_id:
            if url and 'facebook.com' in url:
                profile_id = self._extract_id_from_url(url)
            elif username:
                # Try to resolve username to ID
                profile_id = await self._resolve_username(username)

        if not profile_id:
            raise ValueError("Facebook scraper requires 'profile_id', 'page_id', 'url', or 'username'")

        logger.info(f"Scraping Facebook entity: {profile_id}")

        # Ensure we have a valid token
        await self._ensure_valid_token()

        # Determine if this is a user profile or page
        entity_type = await self._determine_entity_type(profile_id)

        if entity_type == 'page':
            return await self._scrape_page(profile_id)
        else:
            return await self._scrape_profile(profile_id)

    async def _scrape_profile(self, profile_id: str) -> Dict[str, Any]:
        """Scrape Facebook user profile (limited by API permissions)."""
        # Note: User profiles have significant privacy restrictions
        # This will only work for the authenticated user or approved apps

        try:
            # Get basic profile info
            profile_data = await self._api_get(f"/{profile_id}", fields=self.profile_fields)

            # Structure the response
            result = {
                "entity_id": profile_id,
                "entity_type": "profile",
                "name": profile_data.get('name'),
                "about": profile_data.get('about'),
                "birthday": profile_data.get('birthday'),
                "email": profile_data.get('email'),
                "location": profile_data.get('location', {}).get('name') if profile_data.get('location') else None,
                "hometown": profile_data.get('hometown', {}).get('name') if profile_data.get('hometown') else None,
                "website": profile_data.get('website'),
                "relationship_status": profile_data.get('relationship_status'),
                "education": profile_data.get('education', []),
                "work": profile_data.get('work', []),
                "api_used": True,
                "scraped_at": datetime.utcnow().isoformat()
            }

            # Get posts if requested and permitted
            if self.facebook_config.extract_posts:
                try:
                    posts = await self._get_user_posts(profile_id)
                    result['recent_posts'] = posts
                except Exception as e:
                    logger.warning(f"Could not get posts for profile {profile_id}: {e}")
                    result['posts_error'] = str(e)

            self.profile_count += 1
            return result

        except Exception as e:
            logger.error(f"Failed to scrape Facebook profile {profile_id}: {e}")
            raise

    async def _scrape_page(self, page_id: str) -> Dict[str, Any]:
        """Scrape Facebook page with comprehensive data extraction."""
        try:
            # Get page info
            page_data = await self._api_get(f"/{page_id}", fields=self.page_fields)

            result = {
                "entity_id": page_id,
                "entity_type": "page",
                "name": page_data.get('name'),
                "about": page_data.get('about'),
                "category": page_data.get('category'),
                "description": page_data.get('description'),
                "website": page_data.get('website'),
                "location": page_data.get('location'),
                "phone": page_data.get('phone'),
                "emails": page_data.get('emails', []),
                "fan_count": page_data.get('fan_count'),
                "api_used": True,
                "scraped_at": datetime.utcnow().isoformat()
            }

            # Get posts if requested
            if self.facebook_config.extract_posts:
                posts = await self._get_page_posts(page_id)
                result['recent_posts'] = posts

            # Get events if requested
            if self.facebook_config.extract_events:
                events = await self._get_page_events(page_id)
                result['upcoming_events'] = events

            # Get insights if page token available
            if self.facebook_config.extract_page_insights and self.facebook_config.page_token:
                insights = await self._get_page_insights(page_id)
                result['insights'] = insights

            self.profile_count += 1
            return result

        except Exception as e:
            logger.error(f"Failed to scrape Facebook page {page_id}: {e}")
            raise

    async def _get_user_posts(self, user_id: str) -> List[Dict[str, Any]]:
        """Get posts from a user profile."""
        since_date = datetime.utcnow() - timedelta(days=self.facebook_config.post_lookback_days)

        params = {
            'fields': self.post_fields,
            'since': int(since_date.timestamp()),
            'limit': min(self.facebook_config.max_posts, 100)
        }

        posts_data = await self._api_get(f"/{user_id}/posts", **params)
        return self._process_posts(posts_data.get('data', []))

    async def _get_page_posts(self, page_id: str) -> List[Dict[str, Any]]:
        """Get posts from a Facebook page."""
        since_date = datetime.utcnow() - timedelta(days=self.facebook_config.post_lookback_days)

        params = {
            'fields': self.post_fields,
            'since': int(since_date.timestamp()),
            'limit': min(self.facebook_config.max_posts, 100)
        }

        posts_data = await self._api_get(f"/{page_id}/posts", **params)
        return self._process_posts(posts_data.get('data', []))

    async def _get_page_events(self, page_id: str) -> List[Dict[str, Any]]:
        """Get events from a Facebook page."""
        params = {
            'fields': 'id,name,description,start_time,end_time,place,attending_count,interested_count',
            'since': int(datetime.utcnow().timestamp()),
            'limit': 50
        }

        events_data = await self._api_get(f"/{page_id}/events", **params)
        return events_data.get('data', [])

    async def _get_page_insights(self, page_id: str) -> Dict[str, Any]:
        """Get page insights (requires page access token)."""
        # This requires special permissions and page access token
        params = {
            'metric': 'page_fans,page_impressions,page_engaged_users',
            'period': 'day',
            'date_preset': 'last_30d'
        }

        insights_data = await self._api_get(f"/{page_id}/insights", **params)
        return self._process_insights(insights_data.get('data', []))

    def _process_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean post data."""
        processed = []

        for post in posts:
            processed_post = {
                "id": post.get('id'),
                "message": post.get('message'),
                "created_time": post.get('created_time'),
                "type": post.get('type'),
                "permalink_url": post.get('permalink_url'),
                "attachments": post.get('attachments', {}).get('data', []),
                "reaction_count": post.get('reactions', {}).get('summary', {}).get('total_count', 0),
                "comment_count": post.get('comments', {}).get('summary', {}).get('total_count', 0) if self.facebook_config.include_comments else 0
            }
            processed.append(processed_post)

        return processed

    def _process_insights(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process page insights data."""
        processed = {}

        for metric in insights:
            metric_name = metric.get('name')
            values = metric.get('values', [])
            if values:
                latest_value = values[-1].get('value', 0)
                processed[metric_name] = latest_value

        return processed

    async def _api_get(self, endpoint: str, **params) -> Dict[str, Any]:
        """Make authenticated GET request to Facebook Graph API."""
        import aiohttp

        # Check rate limits
        await self._check_rate_limits()

        url = f"{self.graph_api_base}{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.long_lived_token or self.facebook_config.access_token}'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    # Update rate limit tracking
                    self._update_rate_limits(response.headers)

                    if response.status == 429:
                        # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Facebook API rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        return await self._api_get(endpoint, **params)

                    response.raise_for_status()
                    return await response.json()

        except Exception as e:
            logger.error(f"Facebook API request failed: {e}")
            raise

    async def _ensure_valid_token(self) -> None:
        """Ensure we have a valid access token."""
        if not self.facebook_config.access_token and not self.long_lived_token:
            if self.facebook_config.app_id and self.facebook_config.app_secret:
                # Generate app access token
                self.facebook_config.access_token = f"{self.facebook_config.app_id}|{self.facebook_config.app_secret}"
            else:
                raise RuntimeError("Facebook API requires access_token or app_id + app_secret")

        # Check if we need to exchange for long-lived token
        if (self.facebook_config.user_token and
            (not self.long_lived_token or
             (self.token_expires_at and datetime.utcnow() > self.token_expires_at))):

            await self._exchange_for_long_lived_token()

    async def _exchange_for_long_lived_token(self) -> None:
        """Exchange short-lived token for long-lived token."""
        import aiohttp

        params = {
            'grant_type': 'fb_exchange_token',
            'client_id': self.facebook_config.app_id,
            'client_secret': self.facebook_config.app_secret,
            'fb_exchange_token': self.facebook_config.user_token
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://graph.facebook.com/oauth/access_token", params=params) as response:
                    response.raise_for_status()
                    token_data = await response.json()

                    self.long_lived_token = token_data['access_token']
                    expires_in = token_data.get('expires_in', 5184000)  # 60 days default
                    self.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

                    logger.info("Exchanged for long-lived Facebook token")

        except Exception as e:
            logger.warning(f"Failed to exchange for long-lived token: {e}")
            # Continue with short-lived token

    async def _determine_entity_type(self, entity_id: str) -> str:
        """Determine if entity is a user profile or page."""
        try:
            # Try to get as page first
            page_data = await self._api_get(f"/{entity_id}", fields="id,name,fan_count")
            if 'fan_count' in page_data:
                return 'page'
            else:
                return 'profile'
        except Exception:
            # If page query fails, assume it's a profile
            return 'profile'

    async def _resolve_username(self, username: str) -> Optional[str]:
        """Resolve Facebook username to ID."""
        try:
            # This is tricky as Facebook doesn't have a direct username resolution API
            # We'd need to use search or web scraping fallback
            logger.warning("Facebook username resolution not implemented - use profile/page IDs directly")
            return None
        except Exception as e:
            logger.warning(f"Failed to resolve username {username}: {e}")
            return None

    def _extract_id_from_url(self, url: str) -> str:
        """Extract Facebook ID from URL."""
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')

        # Handle different Facebook URL formats
        if 'pages' in path_parts:
            # /pages/PageName/PageID
            pages_index = path_parts.index('pages')
            if pages_index + 2 < len(path_parts):
                return path_parts[pages_index + 2]
        elif 'profile.php' in parsed.query:
            # profile.php?id=12345
            from urllib.parse import parse_qs
            query_params = parse_qs(parsed.query)
            if 'id' in query_params:
                return query_params['id'][0]
        elif path_parts:
            # Direct username or ID
            return path_parts[0]

        return "unknown"

    async def _check_rate_limits(self) -> None:
        """Check and handle API rate limits."""
        current_time = datetime.utcnow().timestamp()

        # Clean old API call times (keep last hour)
        self.api_call_times = [t for t in self.api_call_times if current_time - t < 3600]

        # Facebook allows ~200 calls per hour for most endpoints
        if len(self.api_call_times) >= 180:  # Be conservative
            oldest_call = min(self.api_call_times)
            wait_time = 3600 - (current_time - oldest_call)
            if wait_time > 0:
                logger.info(f"Approaching Facebook API rate limit, waiting {wait_time:.0f}s")
                await asyncio.sleep(wait_time)

        # Track this call
        self.api_call_times.append(current_time)

    def _update_rate_limits(self, headers: Dict[str, str]) -> None:
        """Update rate limit tracking from response headers."""
        # Facebook provides rate limit info in headers
        # This is a simplified implementation
        pass

    async def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate Facebook scraping result."""
        # At minimum, we need an entity ID
        if not result.get('entity_id'):
            return False

        # Check for basic entity info
        has_basic_info = result.get('name') or result.get('about') or result.get('description')
        return bool(has_basic_info)

    def get_facebook_metrics(self) -> Dict[str, Any]:
        """Get Facebook-specific metrics."""
        return {
            **self.get_metrics(),
            'entities_scraped': self.profile_count,
            'api_available': self.api_available,
            'token_type': 'long_lived' if self.long_lived_token else 'short_lived',
            'token_expires_soon': self.token_expires_at and (self.token_expires_at - datetime.utcnow()).days < 7,
            'features_enabled': {
                'posts': self.facebook_config.extract_posts,
                'events': self.facebook_config.extract_events,
                'photos': self.facebook_config.extract_photos,
                'friends': self.facebook_config.extract_friends,
                'groups': self.facebook_config.extract_groups,
                'insights': self.facebook_config.extract_page_insights
            }
        }

    async def cleanup(self) -> None:
        """Cleanup Facebook scraper resources."""
        await super().cleanup()

        # Clear tokens and sensitive data
        self.long_lived_token = None
        self.token_expires_at = None
        self.api_call_times.clear()
        self.profile_count = 0

        logger.info("Facebook scraper cleaned up")
