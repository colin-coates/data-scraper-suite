"""
Instagram Scraper - Migrated from leadintel.

Scrapes Instagram profiles for lead generation.
"""

import logging
import asyncio
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class InstagramProfile:
    """Instagram profile data."""
    username: str
    full_name: Optional[str] = None
    bio: Optional[str] = None
    follower_count: Optional[int] = None
    following_count: Optional[int] = None
    post_count: Optional[int] = None
    is_business: bool = False
    business_category: Optional[str] = None
    external_url: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    is_verified: bool = False
    profile_pic_url: Optional[str] = None
    scraped_at: datetime = None
    
    def __post_init__(self):
        if self.scraped_at is None:
            self.scraped_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_platform": "instagram",
            "username": self.username,
            "full_name": self.full_name,
            "bio": self.bio,
            "follower_count": self.follower_count,
            "following_count": self.following_count,
            "post_count": self.post_count,
            "is_business": self.is_business,
            "business_category": self.business_category,
            "website": self.external_url,
            "email": self.email,
            "phone": self.phone,
            "location": self.location,
            "is_verified": self.is_verified,
            "scraped_at": self.scraped_at.isoformat() if self.scraped_at else None,
        }


class InstagramScraper:
    """
    Instagram profile scraper.
    
    Migrated from leadintel Django app.
    Now part of the data-scraper-suite plugin architecture.
    
    Features:
    - Profile data extraction
    - Email/phone extraction from bio
    - Business account detection
    - Rate limiting and anti-detection
    """
    
    # Email regex pattern
    EMAIL_PATTERN = re.compile(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    )
    
    # Phone regex pattern (various formats)
    PHONE_PATTERN = re.compile(
        r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}'
    )
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        rate_limit_delay: float = 2.0,
    ):
        self.session_id = session_id
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: Optional[datetime] = None
    
    async def scrape_profile(self, username: str) -> Optional[InstagramProfile]:
        """
        Scrape an Instagram profile.
        
        Args:
            username: Instagram username to scrape
            
        Returns:
            InstagramProfile or None if failed
        """
        # Rate limiting
        await self._rate_limit()
        
        try:
            # Use Instagram's public API endpoint
            url = f"https://www.instagram.com/api/v1/users/web_profile_info/?username={username}"
            
            headers = self._get_headers()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 404:
                        logger.warning(f"Profile not found: {username}")
                        return None
                    
                    if response.status != 200:
                        logger.error(f"Failed to fetch profile: {response.status}")
                        return None
                    
                    data = await response.json()
                    return self._parse_profile(data, username)
                    
        except Exception as e:
            logger.error(f"Error scraping {username}: {e}")
            return None
    
    async def scrape_profiles(
        self,
        usernames: List[str],
        max_concurrent: int = 3,
    ) -> List[InstagramProfile]:
        """
        Scrape multiple profiles with concurrency control.
        
        Args:
            usernames: List of usernames to scrape
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of scraped profiles
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(username: str):
            async with semaphore:
                return await self.scrape_profile(username)
        
        tasks = [scrape_with_semaphore(u) for u in usernames]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        profiles = []
        for result in results:
            if isinstance(result, InstagramProfile):
                profiles.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Scrape failed: {result}")
        
        return profiles
    
    def _parse_profile(self, data: Dict[str, Any], username: str) -> InstagramProfile:
        """Parse Instagram API response into profile."""
        try:
            user = data.get("data", {}).get("user", {})
            
            bio = user.get("biography", "")
            
            # Extract email from bio
            email = None
            email_match = self.EMAIL_PATTERN.search(bio)
            if email_match:
                email = email_match.group()
            
            # Extract phone from bio
            phone = None
            phone_match = self.PHONE_PATTERN.search(bio)
            if phone_match:
                phone = phone_match.group()
            
            return InstagramProfile(
                username=username,
                full_name=user.get("full_name"),
                bio=bio,
                follower_count=user.get("edge_followed_by", {}).get("count"),
                following_count=user.get("edge_follow", {}).get("count"),
                post_count=user.get("edge_owner_to_timeline_media", {}).get("count"),
                is_business=user.get("is_business_account", False),
                business_category=user.get("business_category_name"),
                external_url=user.get("external_url"),
                email=email,
                phone=phone,
                is_verified=user.get("is_verified", False),
                profile_pic_url=user.get("profile_pic_url_hd"),
            )
            
        except Exception as e:
            logger.error(f"Error parsing profile data: {e}")
            return InstagramProfile(username=username)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with anti-detection measures."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "X-IG-App-ID": "936619743392459",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://www.instagram.com/",
            "Origin": "https://www.instagram.com",
        }
        
        if self.session_id:
            headers["Cookie"] = f"sessionid={self.session_id}"
        
        return headers
    
    async def _rate_limit(self):
        """Apply rate limiting between requests."""
        if self._last_request_time:
            elapsed = (datetime.utcnow() - self._last_request_time).total_seconds()
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
        
        self._last_request_time = datetime.utcnow()


# Plugin registration for data-scraper-suite
def register_scraper(engine):
    """Register Instagram scraper with the scraper engine."""
    engine.register_scraper("instagram", InstagramScraper)
