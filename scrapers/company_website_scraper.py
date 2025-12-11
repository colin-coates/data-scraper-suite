# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Company Website Scraper Plugin for MJ Data Scraper Suite

Scrapes corporate websites for company information, team members, products, news,
and contact details. Uses intelligent crawling with content analysis.
"""

import asyncio
import logging
import re
import json
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

from core.base_scraper import BaseScraper, ScraperConfig

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__description__ = "Intelligent company website scraper with content analysis"
__author__ = "MJ Intelligence"
__dependencies__ = ["requests", "beautifulsoup4", "lxml", "newspaper3k"]


class CompanyWebsiteConfig(ScraperConfig):
    """Configuration specific to company website scraping."""
    extract_contact_info: bool = True
    extract_team_members: bool = True
    extract_products_services: bool = True
    extract_news_press: bool = True
    extract_about_info: bool = True
    extract_careers: bool = True
    extract_locations: bool = True
    respect_robots_txt: bool = True
    max_pages_per_domain: int = 50
    max_crawl_depth: int = 3
    crawl_delay: float = 1.0
    allowed_file_types: List[str] = ['html', 'htm', 'php', 'asp', 'aspx']
    follow_external_links: bool = False
    extract_social_links: bool = True
    extract_structured_data: bool = True


class CompanyWebsiteScraper(BaseScraper):
    """
    Intelligent company website scraper with content analysis and structured data extraction.
    Respects robots.txt and implements ethical crawling practices.
    """

    def __init__(self, config: CompanyWebsiteConfig):
        super().__init__(config)
        self.company_config = config

        # Crawling state
        self.visited_urls: Set[str] = set()
        self.url_queue: asyncio.Queue[str] = asyncio.Queue()
        self.domain_info: Dict[str, Dict[str, Any]] = {}
        self.robots_parser: Optional[RobotFileParser] = None

        # Content extraction patterns
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b')
        self.social_patterns = {
            'linkedin': re.compile(r'linkedin\.com/company/([^/\s]+)'),
            'twitter': re.compile(r'twitter\.com/([^/\s]+)|x\.com/([^/\s]+)'),
            'facebook': re.compile(r'facebook\.com/([^/\s]+)'),
            'instagram': re.compile(r'instagram\.com/([^/\s]+)'),
            'youtube': re.compile(r'youtube\.com/(?:channel/|user/|@)?([^/\s]+)')
        }

        self.site_count = 0

    async def _execute_scrape(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute company website scraping with intelligent crawling.

        Args:
            target: Contains 'website_url', 'company_name', or 'domain'

        Returns:
            Comprehensive company website data
        """
        website_url = target.get('website_url') or target.get('url')
        company_name = target.get('company_name')

        if not website_url:
            raise ValueError("Company website scraper requires 'website_url' or 'url'")

        # Normalize URL
        website_url = self._normalize_website_url(website_url)
        domain = self._extract_domain(website_url)

        logger.info(f"Scraping company website: {website_url}")

        # Check robots.txt if enabled
        if self.company_config.respect_robots_txt:
            can_crawl = await self._check_robots_txt(website_url)
            if not can_crawl:
                logger.warning(f"Robots.txt disallows crawling: {website_url}")
                return {
                    "website_url": website_url,
                    "domain": domain,
                    "company_name": company_name,
                    "robots_blocked": True,
                    "error": "Blocked by robots.txt",
                    "scraped_at": datetime.utcnow().isoformat()
                }

        # Initialize crawling
        await self._init_crawling(website_url, domain)

        # Perform intelligent crawling
        crawled_data = await self._perform_crawling(domain)

        # Extract structured company information
        company_info = await self._extract_company_info(crawled_data, company_name)

        # Add metadata
        company_info.update({
            "website_url": website_url,
            "domain": domain,
            "pages_crawled": len(crawled_data),
            "crawl_depth": self.company_config.max_crawl_depth,
            "scraped_at": datetime.utcnow().isoformat()
        })

        self.site_count += 1
        return company_info

    async def _init_crawling(self, start_url: str, domain: str) -> None:
        """Initialize crawling session for a domain."""
        # Reset state for new domain
        self.visited_urls.clear()
        while not self.url_queue.empty():
            try:
                self.url_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Initialize domain info
        self.domain_info[domain] = {
            'start_url': start_url,
            'pages_crawled': 0,
            'last_crawl': datetime.utcnow(),
            'response_times': []
        }

        # Add initial URL to queue
        await self.url_queue.put(start_url)
        self.visited_urls.add(start_url)

    async def _check_robots_txt(self, url: str) -> bool:
        """Check if crawling is allowed by robots.txt."""
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(robots_url, timeout=10) as response:
                    if response.status == 200:
                        robots_content = await response.text()
                        self.robots_parser = RobotFileParser()
                        self.robots_parser.parse(robots_content)

                        # Check if our user agent can fetch the URL
                        return self.robots_parser.can_fetch('*', url)

        except Exception as e:
            logger.warning(f"Could not check robots.txt for {url}: {e}")
            # Default to allowing if robots.txt check fails
            return True

        return True  # Allow by default if no robots.txt

    async def _perform_crawling(self, domain: str) -> Dict[str, Dict[str, Any]]:
        """Perform intelligent crawling of the website."""
        crawled_data = {}
        pages_crawled = 0

        while (pages_crawled < self.company_config.max_pages_per_domain and
               not self.url_queue.empty()):

            try:
                current_url = await self.url_queue.get()

                # Skip if already visited or external domain
                if current_url in self.visited_urls:
                    continue

                current_domain = self._extract_domain(current_url)
                if (not self.company_config.follow_external_links and
                    current_domain != domain):
                    continue

                self.visited_urls.add(current_url)

                # Crawl the page
                page_data = await self._crawl_page(current_url, domain)

                if page_data:
                    crawled_data[current_url] = page_data
                    pages_crawled += 1

                    # Extract and queue new URLs
                    if pages_crawled < self.company_config.max_pages_per_domain:
                        await self._extract_and_queue_urls(page_data, current_url, domain)

                # Respect crawl delay
                await asyncio.sleep(self.company_config.crawl_delay)

            except Exception as e:
                logger.error(f"Error crawling page: {e}")
                continue

        return crawled_data

    async def _crawl_page(self, url: str, domain: str) -> Optional[Dict[str, Any]]:
        """Crawl a single page and extract its content."""
        try:
            import aiohttp
            from bs4 import BeautifulSoup

            # Make request with timeout
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                start_time = asyncio.get_event_loop().time()

                async with session.get(url, headers=self._get_request_headers()) as response:
                    if response.status != 200:
                        logger.debug(f"HTTP {response.status} for {url}")
                        return None

                    content_type = response.headers.get('content-type', '')
                    if not content_type.startswith('text/html'):
                        return None

                    html_content = await response.text()
                    response_time = asyncio.get_event_loop().time() - start_time

                    # Update domain stats
                    self.domain_info[domain]['response_times'].append(response_time)

                    # Parse HTML
                    soup = BeautifulSoup(html_content, 'lxml')

                    # Extract page data
                    page_data = {
                        'url': url,
                        'title': self._extract_title(soup),
                        'meta_description': self._extract_meta_description(soup),
                        'text_content': self._extract_text_content(soup),
                        'links': self._extract_links(soup, url),
                        'structured_data': self._extract_structured_data(soup),
                        'response_time': response_time,
                        'content_length': len(html_content)
                    }

                    # Extract page-specific content
                    page_data.update(self._analyze_page_content(soup, url))

                    return page_data

        except Exception as e:
            logger.debug(f"Failed to crawl {url}: {e}")
            return None

    def _get_request_headers(self) -> Dict[str, str]:
        """Get appropriate headers for web requests."""
        return {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

    def _extract_title(self, soup) -> Optional[str]:
        """Extract page title."""
        title_tag = soup.find('title')
        return title_tag.get_text().strip() if title_tag else None

    def _extract_meta_description(self, soup) -> Optional[str]:
        """Extract meta description."""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        return meta_desc.get('content') if meta_desc else None

    def _extract_text_content(self, soup) -> str:
        """Extract main text content from page."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text content
        text = soup.get_text(separator=' ', strip=True)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text[:10000]  # Limit text length

    def _extract_links(self, soup, base_url: str) -> List[Dict[str, str]]:
        """Extract all links from the page."""
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href')
            text = a_tag.get_text().strip()

            if href:
                full_url = urljoin(base_url, href)
                # Remove fragment
                full_url = urldefrag(full_url)[0]

                links.append({
                    'url': full_url,
                    'text': text,
                    'internal': self._is_internal_link(full_url, base_url)
                })

        return links

    def _extract_structured_data(self, soup) -> List[Dict[str, Any]]:
        """Extract JSON-LD structured data."""
        structured_data = []

        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                structured_data.append(data)
            except (json.JSONDecodeError, TypeError):
                continue

        return structured_data

    def _analyze_page_content(self, soup, url: str) -> Dict[str, Any]:
        """Analyze page content for company-specific information."""
        content_analysis = {
            'emails': [],
            'phones': [],
            'social_links': {},
            'page_type': self._classify_page_type(soup, url)
        }

        text_content = self._extract_text_content(soup)

        # Extract emails
        if self.company_config.extract_contact_info:
            content_analysis['emails'] = list(set(self.email_pattern.findall(text_content)))

        # Extract phone numbers
        if self.company_config.extract_contact_info:
            content_analysis['phones'] = list(set(self.phone_pattern.findall(text_content)))

        # Extract social media links
        if self.company_config.extract_social_links:
            for platform, pattern in self.social_patterns.items():
                matches = pattern.findall(text_content)
                if matches:
                    content_analysis['social_links'][platform] = list(set(matches))

        return content_analysis

    def _classify_page_type(self, soup, url: str) -> str:
        """Classify the type of page based on content and URL."""
        url_lower = url.lower()
        title = self._extract_title(soup) or ""
        title_lower = title.lower()

        # Check URL patterns
        if any(keyword in url_lower for keyword in ['about', 'about-us', 'company']):
            return 'about'
        elif any(keyword in url_lower for keyword in ['team', 'leadership', 'executives']):
            return 'team'
        elif any(keyword in url_lower for keyword in ['product', 'products', 'solutions']):
            return 'products'
        elif any(keyword in url_lower for keyword in ['news', 'press', 'blog', 'media']):
            return 'news'
        elif any(keyword in url_lower for keyword in ['career', 'jobs', 'hiring']):
            return 'careers'
        elif any(keyword in url_lower for keyword in ['contact', 'contact-us']):
            return 'contact'

        # Check title patterns
        if any(keyword in title_lower for keyword in ['about', 'company', 'our story']):
            return 'about'
        elif any(keyword in title_lower for keyword in ['team', 'leadership', 'meet']):
            return 'team'
        elif any(keyword in title_lower for keyword in ['product', 'solution']):
            return 'products'

        return 'general'

    async def _extract_and_queue_urls(self, page_data: Dict[str, Any], current_url: str, domain: str) -> None:
        """Extract new URLs from page data and add to queue."""
        if 'links' not in page_data:
            return

        for link in page_data['links']:
            url = link['url']

            # Skip if already visited
            if url in self.visited_urls:
                continue

            # Check if URL should be crawled
            if self._should_crawl_url(url, domain):
                try:
                    self.url_queue.put_nowait(url)
                    self.visited_urls.add(url)
                except asyncio.QueueFull:
                    break

    def _should_crawl_url(self, url: str, domain: str) -> bool:
        """Determine if a URL should be crawled."""
        try:
            parsed = urlparse(url)

            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                return False

            # Check file extension
            path = parsed.path.lower()
            if any(path.endswith(f'.{ext}') for ext in ['pdf', 'doc', 'docx', 'jpg', 'png', 'gif', 'css', 'js']):
                return False

            # Check if file type is allowed
            if '.' in parsed.path:
                ext = parsed.path.split('.')[-1].lower()
                if ext not in self.company_config.allowed_file_types:
                    return False

            # Check robots.txt if available
            if self.robots_parser and not self.robots_parser.can_fetch('*', url):
                return False

            return True

        except Exception:
            return False

    async def _extract_company_info(self, crawled_data: Dict[str, Dict[str, Any]],
                                   company_name: Optional[str]) -> Dict[str, Any]:
        """Extract structured company information from crawled data."""
        company_info = {
            'company_name': company_name,
            'contact_info': {},
            'team_members': [],
            'products_services': [],
            'news_press': [],
            'locations': [],
            'social_media': {}
        }

        # Process each crawled page
        for url, page_data in crawled_data.items():
            page_type = page_data.get('page_type', 'general')
            content_analysis = page_data.get('content_analysis', {})

            # Extract contact information
            if self.company_config.extract_contact_info:
                company_info['contact_info'].update(self._extract_contact_info(content_analysis, page_type))

            # Extract team members
            if self.company_config.extract_team_members and page_type == 'team':
                team_members = await self._extract_team_members(page_data)
                company_info['team_members'].extend(team_members)

            # Extract products/services
            if self.company_config.extract_products_services and page_type == 'products':
                products = await self._extract_products_services(page_data)
                company_info['products_services'].extend(products)

            # Extract news/press
            if self.company_config.extract_news_press and page_type == 'news':
                news_items = await self._extract_news_press(page_data)
                company_info['news_press'].extend(news_items)

            # Extract locations
            if self.company_config.extract_locations:
                locations = await self._extract_locations(page_data)
                company_info['locations'].extend(locations)

            # Extract social media links
            if self.company_config.extract_social_links:
                company_info['social_media'].update(content_analysis.get('social_links', {}))

        # Deduplicate and clean data
        company_info = self._clean_company_data(company_info)

        return company_info

    def _extract_contact_info(self, content_analysis: Dict[str, Any], page_type: str) -> Dict[str, Any]:
        """Extract contact information from content analysis."""
        contact_info = {}

        # Emails
        emails = content_analysis.get('emails', [])
        if emails:
            contact_info['emails'] = list(set(emails))

        # Phone numbers
        phones = content_analysis.get('phones', [])
        if phones:
            contact_info['phones'] = list(set(phones))

        return contact_info

    async def _extract_team_members(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract team member information from team page."""
        # This would implement team member extraction logic
        # For now, return mock data structure
        return []

    async def _extract_products_services(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract products and services information."""
        # This would implement product/service extraction logic
        return []

    async def _extract_news_press(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract news and press release information."""
        # This would implement news extraction logic
        return []

    async def _extract_locations(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract office locations and addresses."""
        # This would implement location extraction logic
        return []

    def _clean_company_data(self, company_info: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and deduplicate extracted company data."""
        # Remove duplicates from lists
        for key, value in company_info.items():
            if isinstance(value, list):
                # Deduplicate based on content
                seen = set()
                deduplicated = []
                for item in value:
                    item_str = json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item)
                    if item_str not in seen:
                        seen.add(item_str)
                        deduplicated.append(item)
                company_info[key] = deduplicated

        return company_info

    def _normalize_website_url(self, url: str) -> str:
        """Normalize website URL to standard format."""
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'

        # Remove trailing slash
        url = url.rstrip('/')

        return url

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc.lower()

    def _is_internal_link(self, url: str, base_url: str) -> bool:
        """Check if a URL is internal to the domain."""
        try:
            url_domain = self._extract_domain(url)
            base_domain = self._extract_domain(base_url)
            return url_domain == base_domain
        except:
            return False

    async def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate company website scraping result."""
        # At minimum, we need a website URL and domain
        if not result.get('website_url') or not result.get('domain'):
            return False

        # Check if robots.txt blocked access
        if result.get('robots_blocked'):
            return True  # Still a valid result

        # Check for some extracted content
        has_content = (result.get('contact_info') or
                      result.get('team_members') or
                      result.get('products_services') or
                      result.get('pages_crawled', 0) > 0)

        return bool(has_content)

    def get_company_website_metrics(self) -> Dict[str, Any]:
        """Get company website scraping specific metrics."""
        return {
            **self.get_metrics(),
            'sites_scraped': self.site_count,
            'domains_tracked': len(self.domain_info),
            'total_pages_crawled': sum(info.get('pages_crawled', 0) for info in self.domain_info.values()),
            'features_enabled': {
                'contact_info': self.company_config.extract_contact_info,
                'team_members': self.company_config.extract_team_members,
                'products_services': self.company_config.extract_products_services,
                'news_press': self.company_config.extract_news_press,
                'careers': self.company_config.extract_careers,
                'locations': self.company_config.extract_locations,
                'social_links': self.company_config.extract_social_links
            },
            'crawl_settings': {
                'max_pages_per_domain': self.company_config.max_pages_per_domain,
                'max_crawl_depth': self.company_config.max_crawl_depth,
                'crawl_delay': self.company_config.crawl_delay,
                'respect_robots_txt': self.company_config.respect_robots_txt
            }
        }

    async def cleanup(self) -> None:
        """Cleanup company website scraper resources."""
        await super().cleanup()

        # Clear crawling state
        self.visited_urls.clear()
        while not self.url_queue.empty():
            try:
                self.url_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self.domain_info.clear()
        self.robots_parser = None
        self.site_count = 0

        logger.info("Company website scraper cleaned up")
