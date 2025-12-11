# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Generic Web Scraper Plugin for MJ Data Scraper Suite

Scrapes general web pages for content, metadata, and structured data extraction.
"""

import asyncio
import logging
from typing import Dict, Any, List
from urllib.parse import urljoin, urlparse

from core.base_scraper import BaseScraper, ScraperConfig

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__description__ = "Generic web scraper for content and metadata extraction"
__author__ = "MJ Intelligence"
__dependencies__ = ["beautifulsoup4", "lxml"]


class WebScraperConfig(ScraperConfig):
    """Configuration specific to web scraping."""
    extract_metadata: bool = True
    extract_images: bool = False
    extract_links: bool = True
    follow_redirects: bool = True
    extract_structured_data: bool = True
    max_content_length: int = 1024 * 1024  # 1MB
    allowed_domains: List[str] = None


class WebScraper(BaseScraper):
    """
    Generic web scraper for content extraction with configurable features.
    """

    def __init__(self, config: WebScraperConfig):
        super().__init__(config)
        self.web_config = config

        # Web scraping state
        self.visited_urls = set()
        self.domain_stats = {}

    async def _execute_scrape(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute web page scraping.

        Args:
            target: Contains 'url' and optional scraping parameters

        Returns:
            Scraped web page data
        """
        url = target.get('url')
        if not url:
            raise ValueError("Web scraper requires 'url' in target")

        # Domain validation
        if self.web_config.allowed_domains:
            domain = urlparse(url).netloc
            if not any(domain.endswith(allowed) for allowed in self.web_config.allowed_domains):
                raise ValueError(f"Domain {domain} not in allowed domains")

        # Check if already visited (basic duplicate prevention)
        if url in self.visited_urls:
            logger.warning(f"URL already visited: {url}")
            return {"url": url, "status": "duplicate"}

        logger.info(f"Scraping web page: {url}")

        # Mark as visited
        self.visited_urls.add(url)

        # Update domain stats
        domain = urlparse(url).netloc
        self.domain_stats[domain] = self.domain_stats.get(domain, 0) + 1

        # Simulate scraping delay (replace with actual scraping)
        await asyncio.sleep(1.5)

        # Mock page data (replace with real scraping)
        page_data = await self._extract_page_data(url, target)

        return page_data

    async def _extract_page_data(self, url: str, target: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from web page."""
        # This is a mock implementation
        # In production, this would:
        # 1. Make HTTP request with proper headers
        # 2. Parse HTML with BeautifulSoup
        # 3. Extract title, meta tags, content
        # 4. Handle different content types

        return {
            "url": url,
            "title": "Example Web Page Title",
            "description": "This is a meta description of the page",
            "content_type": "text/html",
            "status_code": 200,
            "content_length": 15432,
            "language": "en",
            "canonical_url": url,
            "metadata": {
                "keywords": ["example", "web", "page"],
                "author": "Example Author",
                "published": "2024-01-15",
                "modified": "2024-01-15"
            },
            "structured_data": {
                "type": "WebPage",
                "name": "Example Page",
                "description": "Example page description"
            },
            "links": [
                {"url": "/about", "text": "About Us", "internal": True},
                {"url": "https://external.com", "text": "External Link", "internal": False}
            ],
            "images": [
                {"src": "/image1.jpg", "alt": "Example Image", "width": 800, "height": 600}
            ],
            "scraped_at": "2024-01-15T10:30:00Z",
            "scraper_config": {
                "extract_metadata": self.web_config.extract_metadata,
                "extract_images": self.web_config.extract_images,
                "extract_links": self.web_config.extract_links
            }
        }

    async def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate web scraping result."""
        if result.get("status") == "duplicate":
            return True  # Duplicates are valid results

        required_fields = ['url', 'title', 'status_code']
        return all(field in result for field in required_fields)

    def get_web_metrics(self) -> Dict[str, Any]:
        """Get web scraping specific metrics."""
        return {
            **self.get_metrics(),
            'urls_visited': len(self.visited_urls),
            'domain_stats': self.domain_stats,
            'config': {
                'extract_metadata': self.web_config.extract_metadata,
                'extract_images': self.web_config.extract_images,
                'extract_links': self.web_config.extract_links,
                'follow_redirects': self.web_config.follow_redirects,
                'max_content_length': self.web_config.max_content_length,
                'allowed_domains': self.web_config.allowed_domains
            }
        }

    def reset_visited_urls(self) -> None:
        """Reset the visited URLs set."""
        self.visited_urls.clear()
        logger.info("Visited URLs cache cleared")

    def get_domain_stats(self) -> Dict[str, int]:
        """Get statistics about scraped domains."""
        return self.domain_stats.copy()

    async def cleanup(self) -> None:
        """Cleanup web scraper resources."""
        await super().cleanup()
        self.visited_urls.clear()
        self.domain_stats.clear()
        logger.info("Web scraper cleaned up")
