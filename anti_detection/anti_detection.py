# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Anti-Detection Layer for MJ Data Scraper Suite

Implements dynamic headers, human-behavior simulation, and cookie persistence
to avoid detection by modern websites and APIs.
"""

import asyncio
import json
import logging
import random
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse
import http.cookies

from fake_useragent import UserAgent
from core.base_scraper import ScraperResult

logger = logging.getLogger(__name__)


class AntiDetectionLayer:
    """Anti-detection layer with dynamic headers, behavior simulation, and cookie management."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.ua = UserAgent()

        # Cookie management per domain
        self.cookie_jars: Dict[str, Dict[str, str]] = {}

        # Request history for behavior simulation
        self.request_history: List[Dict[str, Any]] = []

        # Dynamic headers pool
        self.header_templates = self._load_header_templates()

        # Behavior simulation
        self.behavior_patterns = self._load_behavior_patterns()

        # Rate limiting state
        self.last_request_time = 0.0
        self.request_count = 0

        logger.info("Anti-detection layer initialized")

    def _load_header_templates(self) -> List[Dict[str, str]]:
        """Load diverse header templates for rotation."""
        return [
            {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Cache-Control": "max-age=0"
            },
            {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Pragma": "no-cache",
                "Cache-Control": "no-cache"
            },
            {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "cross-site",
                "Sec-Fetch-User": "?1"
            }
        ]

    def _load_behavior_patterns(self) -> Dict[str, Any]:
        """Load human behavior patterns for simulation."""
        return {
            "reading_time": {
                "min": 2.0,
                "max": 8.0,
                "distribution": "normal"
            },
            "scroll_behavior": {
                "patterns": ["smooth", "jumpy", "gradual"],
                "pause_probability": 0.3
            },
            "mouse_movement": {
                "speed_variation": 0.2,
                "curve_probability": 0.6
            },
            "typing_patterns": {
                "keystroke_delay": (0.05, 0.15),
                "word_pause": (0.2, 0.8)
            }
        }

    async def prepare_for_request(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare headers, cookies, and timing for a request.

        Args:
            target: Target information including URL

        Returns:
            Enhanced headers and cookies for the request
        """
        url = target.get('url', '')
        domain = self._extract_domain(url)

        # Generate dynamic headers
        headers = await self._generate_dynamic_headers(url)

        # Get persistent cookies for domain
        cookies = self._get_domain_cookies(domain)

        # Simulate human timing
        await self._simulate_human_timing()

        # Update request history
        self._update_request_history({
            'url': url,
            'domain': domain,
            'timestamp': datetime.utcnow(),
            'headers_count': len(headers)
        })

        return {
            'headers': headers,
            'cookies': cookies,
            'user_agent': headers.get('User-Agent', ''),
            'domain': domain
        }

    async def _generate_dynamic_headers(self, url: str) -> Dict[str, str]:
        """Generate dynamic headers with rotation and randomization."""
        # Select random header template
        template = random.choice(self.header_templates).copy()

        # Rotate User-Agent
        try:
            template['User-Agent'] = self.ua.random
        except:
            template['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

        # Add dynamic elements
        template['Referer'] = self._generate_referer(url)

        # Add timestamp-based variation
        timestamp = str(int(time.time() * 1000))
        template['X-Requested-With'] = f'XMLHttpRequest-{timestamp[-4:]}'

        # Add viewport variation
        viewports = ['1920x1080', '1366x768', '1536x864', '1440x900']
        template['Viewport-Width'] = random.choice(viewports).split('x')[0]

        return template

    def _generate_referer(self, url: str) -> str:
        """Generate realistic referer headers."""
        domain = self._extract_domain(url)

        # Common referer patterns
        referers = [
            f"https://www.google.com/search?q={domain}",
            f"https://www.bing.com/search?q={domain}",
            f"https://search.yahoo.com/search?p={domain}",
            f"https://duckduckgo.com/?q={domain}",
            ""  # No referer sometimes
        ]

        return random.choice(referers)

    def _get_domain_cookies(self, domain: str) -> Dict[str, str]:
        """Get persistent cookies for a domain."""
        if domain not in self.cookie_jars:
            self.cookie_jars[domain] = {}

        cookies = self.cookie_jars[domain].copy()

        # Add session cookie if not present
        if 'session_id' not in cookies:
            cookies['session_id'] = f"session_{random.randint(1000000, 9999999)}"

        # Add timestamp cookie
        cookies['__timestamp__'] = str(int(time.time()))

        return cookies

    async def _simulate_human_timing(self) -> None:
        """Simulate human-like timing between requests."""
        # Base delay
        base_delay = random.uniform(1.0, 3.0)

        # Add variation based on time of day (simulate human patterns)
        hour = datetime.utcnow().hour
        if 9 <= hour <= 17:  # Business hours - faster
            base_delay *= 0.7
        elif 23 <= hour or hour <= 6:  # Late night - slower
            base_delay *= 1.5

        # Add random variation
        variation = random.uniform(-0.5, 0.5)
        delay = max(0.5, base_delay + variation)

        # Simulate reading/thinking time
        if random.random() < 0.3:  # 30% chance of longer pause
            reading_time = random.uniform(2.0, 6.0)
            delay += reading_time

        await asyncio.sleep(delay)

    async def post_request_update(self, result: ScraperResult) -> None:
        """
        Update anti-detection state based on request result.

        Args:
            result: Result of the scraping operation
        """
        if result.success:
            # Store successful cookies
            domain = self._extract_domain(result.target_url)
            if domain and hasattr(result, 'metadata') and 'cookies' in result.metadata:
                self.cookie_jars[domain].update(result.metadata['cookies'])
        else:
            # Handle failures - might need proxy rotation, etc.
            logger.warning(f"Request failed for {result.target_url}: {result.error_message}")

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return "unknown"

    def _update_request_history(self, request_info: Dict[str, Any]) -> None:
        """Update request history for behavior analysis."""
        self.request_history.append(request_info)

        # Keep only recent history (last 100 requests)
        if len(self.request_history) > 100:
            self.request_history = self.request_history[-100:]

    def simulate_mouse_movement(self, start_x: int, start_y: int, end_x: int, end_y: int) -> List[Tuple[int, int]]:
        """
        Simulate human mouse movement with curves.

        Returns:
            List of (x, y) coordinates for smooth mouse movement
        """
        points = []
        steps = random.randint(10, 20)

        for i in range(steps + 1):
            t = i / steps

            # Add curve variation
            if random.random() < self.behavior_patterns['mouse_movement']['curve_probability']:
                # Bezier curve variation
                control_x = (start_x + end_x) / 2 + random.randint(-50, 50)
                control_y = (start_y + end_y) / 2 + random.randint(-50, 50)

                x = (1 - t) ** 2 * start_x + 2 * (1 - t) * t * control_x + t ** 2 * end_x
                y = (1 - t) ** 2 * start_y + 2 * (1 - t) * t * control_y + t ** 2 * end_y
            else:
                # Linear movement
                x = start_x + (end_x - start_x) * t
                y = start_y + (end_y - start_y) * t

            points.append((int(x), int(y)))

        return points

    def simulate_typing(self, text: str) -> List[Tuple[str, float]]:
        """
        Simulate human typing patterns.

        Returns:
            List of (character, delay) tuples
        """
        typing_events = []

        for char in text:
            # Base keystroke delay
            min_delay, max_delay = self.behavior_patterns['typing_patterns']['keystroke_delay']
            delay = random.uniform(min_delay, max_delay)

            # Add word pauses
            if char in [' ', '.', ',', '!', '?']:
                word_pause = random.uniform(*self.behavior_patterns['typing_patterns']['word_pause'])
                delay += word_pause

            typing_events.append((char, delay))

        return typing_events

    def simulate_scrolling(self, total_height: int) -> List[Tuple[int, float]]:
        """
        Simulate human scrolling behavior.

        Returns:
            List of (scroll_position, delay) tuples
        """
        scroll_events = []
        current_position = 0

        while current_position < total_height:
            # Choose scroll pattern
            pattern = random.choice(self.behavior_patterns['scroll_behavior']['patterns'])

            if pattern == "smooth":
                scroll_amount = random.randint(100, 300)
            elif pattern == "jumpy":
                scroll_amount = random.randint(300, 600)
            else:  # gradual
                scroll_amount = random.randint(50, 150)

            current_position = min(total_height, current_position + scroll_amount)

            # Scroll delay
            delay = random.uniform(0.5, 2.0)

            # Pause sometimes
            if random.random() < self.behavior_patterns['scroll_behavior']['pause_probability']:
                delay += random.uniform(1.0, 3.0)

            scroll_events.append((current_position, delay))

        return scroll_events

    def update_cookies(self, domain: str, cookies: Dict[str, str]) -> None:
        """Update stored cookies for a domain."""
        if domain not in self.cookie_jars:
            self.cookie_jars[domain] = {}

        self.cookie_jars[domain].update(cookies)
        logger.debug(f"Updated cookies for {domain}: {len(cookies)} cookies")

    def clear_cookies(self, domain: Optional[str] = None) -> None:
        """Clear cookies for a domain or all domains."""
        if domain:
            self.cookie_jars.pop(domain, None)
            logger.info(f"Cleared cookies for {domain}")
        else:
            self.cookie_jars.clear()
            logger.info("Cleared all cookies")

    def get_detection_score(self) -> float:
        """
        Calculate a detection risk score based on behavior patterns.

        Returns:
            Score between 0.0 (low risk) and 1.0 (high risk)
        """
        score = 0.0

        # Check request frequency
        if len(self.request_history) > 10:
            recent_requests = [r for r in self.request_history[-10:]
                             if (datetime.utcnow() - r['timestamp']).seconds < 60]

            if len(recent_requests) > 5:
                score += 0.3  # Too many requests per minute

        # Check cookie consistency
        total_cookies = sum(len(cookies) for cookies in self.cookie_jars.values())
        if total_cookies > 50:
            score += 0.2  # Too many cookies stored

        # Check header variation
        header_variations = len(set(r.get('headers_count', 0) for r in self.request_history))
        if header_variations < 3:
            score += 0.2  # Insufficient header variation

        return min(1.0, score)

    def get_stats(self) -> Dict[str, Any]:
        """Get anti-detection layer statistics."""
        return {
            'domains_tracked': len(self.cookie_jars),
            'total_cookies': sum(len(cookies) for cookies in self.cookie_jars.values()),
            'requests_tracked': len(self.request_history),
            'detection_score': self.get_detection_score(),
            'header_templates': len(self.header_templates),
            'active_since': self.request_history[0]['timestamp'] if self.request_history else None
        }

    async def cleanup(self) -> None:
        """Cleanup anti-detection resources."""
        self.cookie_jars.clear()
        self.request_history.clear()
        logger.info("Anti-detection layer cleaned up")
