# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
LinkedIn Scraper Plugin for MJ Data Scraper Suite

Advanced LinkedIn profile scraper using Playwright for JavaScript-heavy pages.
Extracts professional information including work history, education, skills,
connections, and contact details with anti-detection measures.
"""

import asyncio
import logging
import json
import re
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin, urlparse, parse_qs
from datetime import datetime

from core.base_scraper import BaseScraper, ScraperConfig

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__description__ = "Advanced LinkedIn profile scraper using Playwright"
__author__ = "MJ Intelligence"
__dependencies__ = ["playwright", "beautifulsoup4", "lxml"]


class LinkedInConfig(ScraperConfig):
    """Configuration specific to LinkedIn scraping."""
    login_email: Optional[str] = None
    login_password: Optional[str] = None
    use_browser: bool = True  # Use Playwright for JS-heavy pages
    headless: bool = True
    extract_connections: bool = True
    extract_recommendations: bool = True
    extract_skills: bool = True
    extract_experience: bool = True
    extract_education: bool = True
    extract_posts: bool = False
    max_scroll_attempts: int = 5
    wait_for_load: float = 3.0
    profile_cache_ttl: int = 3600  # 1 hour


class LinkedInScraper(BaseScraper):
    """
    Advanced LinkedIn profile scraper using Playwright for JavaScript rendering.
    Includes sophisticated anti-detection measures and session management.
    """

    def __init__(self, config: LinkedInConfig):
        super().__init__(config)
        self.linkedin_config = config

        # LinkedIn specific state
        self.browser = None
        self.context = None
        self.page = None
        self.is_logged_in = False
        self.profile_cache = {}
        self.session_cookies = {}
        self.profile_count = 0

        # Playwright selectors for LinkedIn elements
        self.selectors = {
            'name': 'h1.text-heading-xlarge',
            'headline': 'div.text-body-medium',
            'location': 'span.text-body-small.inline.t-black--light.break-words',
            'about': 'div.pv-about__summary-text',
            'experience_section': 'section[data-section="experience"]',
            'education_section': 'section[data-section="education"]',
            'skills_section': 'section[data-section="skills"]',
            'connections_count': 'span[data-test-id="connections-count"]',
            'recommendations_section': 'section[data-section="recommendations"]'
        }

    async def _execute_scrape(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute LinkedIn profile scraping using Playwright.

        Args:
            target: Contains 'profile_url', 'profile_id', or search terms

        Returns:
            Comprehensive LinkedIn profile data
        """
        profile_url = target.get('profile_url') or target.get('url')

        # Handle different input types
        if not profile_url:
            profile_id = target.get('profile_id')
            if profile_id:
                profile_url = f"https://www.linkedin.com/in/{profile_id}"
            else:
                # Handle search-based scraping
                search_query = target.get('search_query')
                if search_query:
                    return await self._search_and_scrape(search_query, target)
                else:
                    raise ValueError("LinkedIn scraper requires 'profile_url', 'profile_id', or 'search_query'")

        # Validate and normalize URL
        profile_url = self._normalize_linkedin_url(profile_url)

        logger.info(f"Scraping LinkedIn profile: {profile_url}")

        # Check cache first
        cache_key = self._get_cache_key(profile_url)
        if cache_key in self.profile_cache:
            cached_data = self.profile_cache[cache_key]
            if self._is_cache_valid(cached_data):
                logger.info("Returning cached LinkedIn profile data")
                return cached_data['data']

        # Initialize browser if needed
        if self.linkedin_config.use_browser and not self.browser:
            await self._init_browser()

        # Login if required and not logged in
        if self.linkedin_config.login_email and not self.is_logged_in:
            await self._login()

        # Scrape the profile
        try:
            profile_data = await self._scrape_profile_with_playwright(profile_url)

            # Cache the result
            self._cache_profile(cache_key, profile_data)

            # Update session counter
            self.profile_count += 1

            return profile_data

        except Exception as e:
            logger.error(f"Failed to scrape LinkedIn profile {profile_url}: {e}")
            raise

    async def _init_browser(self) -> None:
        """Initialize Playwright browser with anti-detection settings."""
        try:
            from playwright.async_api import async_playwright

            playwright = await async_playwright().start()

            # Launch browser with stealth settings
            self.browser = await playwright.chromium.launch(
                headless=self.linkedin_config.headless,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--single-process',  # <- this one doesn't work in Windows
                    '--disable-gpu'
                ]
            )

            # Create context with realistic settings
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                locale='en-US',
                timezone_id='America/New_York',
                permissions=['geolocation'],
                geolocation={'latitude': 40.7128, 'longitude': -74.0060},  # NYC coordinates
                extra_http_headers={
                    'Accept-Language': 'en-US,en;q=0.9',
                    'DNT': '1',
                    'Upgrade-Insecure-Requests': '1'
                }
            )

            # Add anti-detection scripts
            await self.context.add_init_script("""
                // Remove webdriver property
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });

                // Mock plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });

                // Mock languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                });
            """)

            self.page = await self.context.new_page()

            logger.info("Playwright browser initialized with anti-detection measures")

        except ImportError:
            logger.warning("Playwright not available, falling back to HTTP scraping")
            self.linkedin_config.use_browser = False

    async def _login(self) -> None:
        """Login to LinkedIn if credentials are provided."""
        if not self.page or not self.linkedin_config.login_email:
            return

        try:
            logger.info("Attempting LinkedIn login...")

            await self.page.goto("https://www.linkedin.com/login")
            await self.page.wait_for_load_state('networkidle')

            # Fill login form
            await self.page.fill('input[name="session_key"]', self.linkedin_config.login_email)
            await self.page.fill('input[name="session_password"]', self.linkedin_config.login_password or "")

            # Click login button
            await self.page.click('button[type="submit"]')

            # Wait for navigation or error
            try:
                await self.page.wait_for_url("**/feed", timeout=10000)
                self.is_logged_in = True
                logger.info("LinkedIn login successful")

                # Save session cookies
                cookies = await self.context.cookies()
                self.session_cookies = {cookie['name']: cookie['value'] for cookie in cookies}

            except Exception:
                logger.warning("LinkedIn login may have failed or requires 2FA")
                self.is_logged_in = False

        except Exception as e:
            logger.error(f"LinkedIn login failed: {e}")
            self.is_logged_in = False

    async def _scrape_profile_with_playwright(self, profile_url: str) -> Dict[str, Any]:
        """Scrape LinkedIn profile using Playwright."""
        if not self.page:
            raise RuntimeError("Browser not initialized")

        try:
            # Navigate to profile
            await self.page.goto(profile_url, wait_until='domcontentloaded')

            # Wait for page to load
            await asyncio.sleep(self.linkedin_config.wait_for_load)

            # Scroll to load dynamic content
            await self._scroll_to_load_content()

            # Extract profile data
            profile_data = await self.page.evaluate(self._get_profile_extraction_script())

            # Enhance with additional data if logged in
            if self.is_logged_in:
                profile_data.update(await self._extract_private_data())

            # Add metadata
            profile_data.update({
                "profile_url": profile_url,
                "profile_id": self._extract_profile_id(profile_url),
                "scraped_at": datetime.utcnow().isoformat(),
                "scraper_version": __version__,
                "login_used": self.is_logged_in
            })

            return profile_data

        except Exception as e:
            logger.error(f"Playwright scraping failed: {e}")
            # Fallback to HTTP scraping
            return await self._scrape_profile_http(profile_url)

    def _get_profile_extraction_script(self) -> str:
        """JavaScript code to extract profile data from LinkedIn page."""
        return """
        () => {
            const getTextContent = (selector) => {
                const element = document.querySelector(selector);
                return element ? element.textContent.trim() : null;
            };

            const getMultipleElements = (selector) => {
                return Array.from(document.querySelectorAll(selector))
                    .map(el => el.textContent.trim())
                    .filter(text => text.length > 0);
            };

            // Basic profile information
            const profile = {
                full_name: getTextContent('h1.text-heading-xlarge') ||
                          getTextContent('h1[data-test-id="hero-name"]') ||
                          getTextContent('.pv-top-card--list li:first-child'),

                headline: getTextContent('div.text-body-medium.break-words') ||
                         getTextContent('.pv-about__title'),

                location: getTextContent('span.text-body-small.inline.t-black--light.break-words') ||
                         getTextContent('.pv-top-card--list-bullet li:last-child'),

                about: getTextContent('div.pv-about__summary-text') ||
                      getTextContent('.pv-about-panel .pv-about__summary-text'),

                connections_count: (() => {
                    const connElement = document.querySelector('span[data-test-id="connections-count"]') ||
                                       document.querySelector('span[data-test-id="network-count"]');
                    if (connElement) {
                        const match = connElement.textContent.match(/(\\d+)/);
                        return match ? parseInt(match[1]) : null;
                    }
                    return null;
                })()
            };

            // Experience section
            const experienceSection = document.querySelector('section[data-section="experience"]');
            if (experienceSection) {
                profile.experience = Array.from(experienceSection.querySelectorAll('.pv-entity__summary-info'))
                    .map(exp => ({
                        company: exp.querySelector('p.pv-entity__secondary-title')?.textContent?.trim(),
                        title: exp.querySelector('h3')?.textContent?.trim(),
                        duration: exp.querySelector('.pv-entity__date-range span:last-child')?.textContent?.trim(),
                        location: exp.querySelector('.pv-entity__location span:last-child')?.textContent?.trim(),
                        description: exp.querySelector('.pv-entity__description')?.textContent?.trim()
                    }))
                    .filter(exp => exp.title || exp.company);
            }

            // Education section
            const educationSection = document.querySelector('section[data-section="education"]');
            if (educationSection) {
                profile.education = Array.from(educationSection.querySelectorAll('.pv-entity__summary-info'))
                    .map(edu => ({
                        school: edu.querySelector('h3')?.textContent?.trim(),
                        degree: edu.querySelector('.pv-entity__degree-name span:last-child')?.textContent?.trim(),
                        field: edu.querySelector('.pv-entity__fos span:last-child')?.textContent?.trim(),
                        year: edu.querySelector('.pv-entity__dates span:last-child')?.textContent?.trim()
                    }))
                    .filter(edu => edu.school);
            }

            // Skills section
            const skillsSection = document.querySelector('section[data-section="skills"]');
            if (skillsSection) {
                profile.skills = Array.from(skillsSection.querySelectorAll('.pv-skill-category-entity__name-text'))
                    .map(skill => skill.textContent.trim())
                    .filter(skill => skill.length > 0);
            }

            return profile;
        }
        """

    async def _extract_private_data(self) -> Dict[str, Any]:
        """Extract data only available to logged-in users."""
        if not self.page or not self.is_logged_in:
            return {}

        try:
            # Click "Show more" buttons to reveal additional content
            show_more_buttons = await self.page.query_selector_all('button[data-test-id="show-more-button"]')
            for button in show_more_buttons:
                try:
                    await button.click()
                    await asyncio.sleep(0.5)
                except:
                    pass

            # Extract recommendations if enabled
            recommendations = {}
            if self.linkedin_config.extract_recommendations:
                recommendations = await self.page.evaluate("""
                () => {
                    const recSection = document.querySelector('section[data-section="recommendations"]');
                    if (!recSection) return [];

                    return Array.from(recSection.querySelectorAll('.pv-recommendation-entity'))
                        .map(rec => ({
                            recommender: rec.querySelector('.pv-recommendation-entity__member')?.textContent?.trim(),
                            relationship: rec.querySelector('.pv-recommendation-entity__detail')?.textContent?.trim(),
                            text: rec.querySelector('.pv-recommendation-entity__text')?.textContent?.trim()
                        }))
                        .filter(rec => rec.recommender);
                }
                """)

            return {
                "recommendations": recommendations,
                "private_data_accessed": True
            }

        except Exception as e:
            logger.warning(f"Failed to extract private data: {e}")
            return {}

    async def _scroll_to_load_content(self) -> None:
        """Scroll page to load dynamic content."""
        if not self.page:
            return

        try:
            for _ in range(self.linkedin_config.max_scroll_attempts):
                # Scroll down
                await self.page.evaluate("""
                window.scrollTo(0, document.body.scrollHeight);
                """)

                # Wait for content to load
                await asyncio.sleep(1)

                # Check if new content loaded
                new_height = await self.page.evaluate("document.body.scrollHeight")
                await asyncio.sleep(0.5)

        except Exception as e:
            logger.warning(f"Scroll loading failed: {e}")

    async def _scrape_profile_http(self, profile_url: str) -> Dict[str, Any]:
        """Fallback HTTP-based scraping without browser."""
        # This would implement HTTP-based scraping as a fallback
        # For now, return basic structure
        logger.warning("Using fallback HTTP scraping (limited functionality)")

        profile_id = self._extract_profile_id(profile_url)
        return {
            "profile_id": profile_id,
            "full_name": None,
            "headline": None,
            "location": None,
            "about": None,
            "experience": [],
            "education": [],
            "skills": [],
            "connections_count": None,
            "profile_url": profile_url,
            "scraped_at": datetime.utcnow().isoformat(),
            "scraper_version": __version__,
            "fallback_mode": True,
            "error": "Browser-based scraping failed, limited data available"
        }

    async def _search_and_scrape(self, search_query: str, target: Dict[str, Any]) -> Dict[str, Any]:
        """Search for profiles and scrape the first result."""
        # This would implement search functionality
        # For now, raise NotImplementedError
        raise NotImplementedError("Search-based scraping not yet implemented")

    def _normalize_linkedin_url(self, url: str) -> str:
        """Normalize LinkedIn URL to standard format."""
        if not url.startswith('http'):
            url = f"https://www.linkedin.com/in/{url}"

        # Remove query parameters and fragments
        parsed = urlparse(url)
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        return clean_url

    def _extract_profile_id(self, url: str) -> str:
        """Extract profile ID from LinkedIn URL."""
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')

        if 'in' in path_parts:
            in_index = path_parts.index('in')
            if in_index + 1 < len(path_parts):
                profile_id = path_parts[in_index + 1]
                # Remove any remaining path components
                return profile_id.split('/')[0]

        return "unknown"

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for profile URL."""
        return f"linkedin_{self._extract_profile_id(url)}"

    def _is_cache_valid(self, cached_data: Dict[str, Any]) -> bool:
        """Check if cached data is still valid."""
        if 'timestamp' not in cached_data:
            return False

        cache_time = datetime.fromisoformat(cached_data['timestamp'])
        age = (datetime.utcnow() - cache_time).total_seconds()

        return age < self.linkedin_config.profile_cache_ttl

    def _cache_profile(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Cache profile data."""
        self.profile_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Limit cache size
        if len(self.profile_cache) > 100:
            # Remove oldest entries (simple FIFO)
            oldest_key = min(self.profile_cache.keys(),
                           key=lambda k: self.profile_cache[k]['timestamp'])
            del self.profile_cache[oldest_key]

    async def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate LinkedIn scraping result."""
        # At minimum, we need a profile ID
        if not result.get('profile_id') or result['profile_id'] == 'unknown':
            return False

        # If in fallback mode, we accept minimal data
        if result.get('fallback_mode'):
            return True

        # For full scraping, check for basic profile info
        has_basic_info = result.get('full_name') or result.get('headline')
        return bool(has_basic_info)

    def get_linkedin_metrics(self) -> Dict[str, Any]:
        """Get LinkedIn-specific metrics."""
        return {
            **self.get_metrics(),
            'profiles_scraped': self.profile_count,
            'browser_mode': self.linkedin_config.use_browser,
            'logged_in': self.is_logged_in,
            'cache_size': len(self.profile_cache),
            'features_enabled': {
                'connections': self.linkedin_config.extract_connections,
                'recommendations': self.linkedin_config.extract_recommendations,
                'skills': self.linkedin_config.extract_skills,
                'experience': self.linkedin_config.extract_experience,
                'education': self.linkedin_config.extract_education,
                'posts': self.linkedin_config.extract_posts
            }
        }

    async def cleanup(self) -> None:
        """Cleanup LinkedIn scraper resources."""
        await super().cleanup()

        # Close browser context and page
        if self.page:
            try:
                await self.page.close()
            except:
                pass
            self.page = None

        if self.context:
            try:
                await self.context.close()
            except:
                pass
            self.context = None

        if self.browser:
            try:
                await self.browser.close()
            except:
                pass
            self.browser = None

        # Clear caches and session data
        self.profile_cache.clear()
        self.session_cookies.clear()
        self.is_logged_in = False
        self.profile_count = 0

        logger.info("LinkedIn scraper cleaned up")
