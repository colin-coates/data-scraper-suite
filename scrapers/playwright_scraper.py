# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Playwright Browser Scraper for MJ Data Scraper Suite

Handles JavaScript-heavy sites using headless browser automation.
Supports dynamic content, SPAs, and sites requiring user interaction.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from core.base_scraper import BaseScraper, ScraperConfig

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__description__ = "Playwright-based browser scraper for JS-heavy sites"
__author__ = "MJ Intelligence"
__dependencies__ = ["playwright"]


@dataclass
class PlaywrightConfig(ScraperConfig):
    """Configuration for Playwright browser scraping."""
    headless: bool = True
    browser_type: str = "chromium"  # chromium, firefox, webkit
    viewport_width: int = 1920
    viewport_height: int = 1080
    wait_for_selector: Optional[str] = None
    wait_for_load_state: str = "networkidle"  # load, domcontentloaded, networkidle
    wait_timeout: int = 30000  # milliseconds
    screenshot: bool = False
    screenshot_path: Optional[str] = None
    block_resources: List[str] = field(default_factory=lambda: ["image", "media", "font"])
    javascript_enabled: bool = True
    locale: str = "en-US"
    timezone: str = "America/New_York"
    geolocation: Optional[Dict[str, float]] = None
    permissions: List[str] = field(default_factory=list)
    extra_http_headers: Dict[str, str] = field(default_factory=dict)
    ignore_https_errors: bool = True
    slow_mo: int = 0  # Slow down operations by ms (for debugging)
    record_video: bool = False
    stealth_mode: bool = True  # Enable anti-detection measures


class PlaywrightScraper(BaseScraper):
    """
    Browser-based scraper using Playwright for JavaScript-heavy sites.
    
    Features:
    - Headless browser automation
    - JavaScript execution and waiting
    - Screenshot capture
    - Resource blocking for performance
    - Stealth mode for anti-detection
    - Multiple browser engine support
    """

    ROLE = "browser"
    TIER = 3  # Requires human approval for browser scraping
    SUPPORTED_EVENTS = ["weddings", "corporate", "social", "professional"]

    def __init__(self, config: PlaywrightConfig):
        super().__init__(config)
        self.pw_config = config
        self._browser = None
        self._context = None
        self._playwright = None

    async def _init_browser(self):
        """Initialize Playwright browser instance."""
        if self._browser is not None:
            return

        try:
            from playwright.async_api import async_playwright
            
            self._playwright = await async_playwright().start()
            
            # Select browser type
            if self.pw_config.browser_type == "firefox":
                browser_type = self._playwright.firefox
            elif self.pw_config.browser_type == "webkit":
                browser_type = self._playwright.webkit
            else:
                browser_type = self._playwright.chromium

            # Launch browser with stealth options
            launch_options = {
                "headless": self.pw_config.headless,
                "slow_mo": self.pw_config.slow_mo,
            }

            if self.pw_config.stealth_mode:
                launch_options["args"] = [
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-infobars",
                    "--window-position=0,0",
                    "--ignore-certifcate-errors",
                    "--ignore-certifcate-errors-spki-list",
                ]

            self._browser = await browser_type.launch(**launch_options)

            # Create context with anti-detection settings
            context_options = {
                "viewport": {
                    "width": self.pw_config.viewport_width,
                    "height": self.pw_config.viewport_height
                },
                "locale": self.pw_config.locale,
                "timezone_id": self.pw_config.timezone,
                "ignore_https_errors": self.pw_config.ignore_https_errors,
                "java_script_enabled": self.pw_config.javascript_enabled,
            }

            if self.pw_config.geolocation:
                context_options["geolocation"] = self.pw_config.geolocation
                context_options["permissions"] = ["geolocation"]

            if self.pw_config.extra_http_headers:
                context_options["extra_http_headers"] = self.pw_config.extra_http_headers

            if self.pw_config.stealth_mode:
                context_options["user_agent"] = (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )

            if self.pw_config.record_video:
                context_options["record_video_dir"] = "./videos"

            self._context = await self._browser.new_context(**context_options)

            # Add stealth scripts
            if self.pw_config.stealth_mode:
                await self._context.add_init_script("""
                    // Override webdriver detection
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                    
                    // Override plugins
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [1, 2, 3, 4, 5]
                    });
                    
                    // Override languages
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['en-US', 'en']
                    });
                    
                    // Override chrome
                    window.chrome = {
                        runtime: {}
                    };
                    
                    // Override permissions
                    const originalQuery = window.navigator.permissions.query;
                    window.navigator.permissions.query = (parameters) => (
                        parameters.name === 'notifications' ?
                            Promise.resolve({ state: Notification.permission }) :
                            originalQuery(parameters)
                    );
                """)

            logger.info(f"Playwright browser initialized: {self.pw_config.browser_type}")

        except ImportError:
            raise ImportError("Playwright not installed. Run: pip install playwright && playwright install")
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            raise

    async def _execute_scrape(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute browser-based scraping.

        Args:
            target: Contains 'url' and optional parameters

        Returns:
            Scraped page data including rendered HTML
        """
        url = target.get('url')
        if not url:
            raise ValueError("Playwright scraper requires 'url' in target")

        # Initialize browser if needed
        await self._init_browser()

        logger.info(f"Scraping with Playwright: {url}")

        page = await self._context.new_page()

        try:
            # Block unnecessary resources for performance
            if self.pw_config.block_resources:
                await page.route("**/*", lambda route: (
                    route.abort() if route.request.resource_type in self.pw_config.block_resources
                    else route.continue_()
                ))

            # Navigate to page
            response = await page.goto(
                url,
                wait_until=self.pw_config.wait_for_load_state,
                timeout=self.pw_config.wait_timeout
            )

            # Wait for specific selector if configured
            if self.pw_config.wait_for_selector:
                await page.wait_for_selector(
                    self.pw_config.wait_for_selector,
                    timeout=self.pw_config.wait_timeout
                )

            # Additional wait for dynamic content
            await page.wait_for_load_state("networkidle", timeout=self.pw_config.wait_timeout)

            # Execute any custom JavaScript
            custom_js = target.get('execute_js')
            js_result = None
            if custom_js:
                js_result = await page.evaluate(custom_js)

            # Extract page data
            title = await page.title()
            content = await page.content()
            
            # Extract text content
            text_content = await page.evaluate("() => document.body.innerText")

            # Extract metadata
            metadata = await page.evaluate("""
                () => {
                    const meta = {};
                    document.querySelectorAll('meta').forEach(m => {
                        const name = m.getAttribute('name') || m.getAttribute('property');
                        const content = m.getAttribute('content');
                        if (name && content) meta[name] = content;
                    });
                    return meta;
                }
            """)

            # Extract links
            links = await page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                    href: a.href,
                    text: a.innerText.trim(),
                    rel: a.rel
                })).slice(0, 100)
            """)

            # Extract structured data (JSON-LD)
            structured_data = await page.evaluate("""
                () => {
                    const scripts = document.querySelectorAll('script[type="application/ld+json"]');
                    return Array.from(scripts).map(s => {
                        try { return JSON.parse(s.textContent); }
                        catch { return null; }
                    }).filter(Boolean);
                }
            """)

            # Take screenshot if configured
            screenshot_data = None
            if self.pw_config.screenshot:
                screenshot_path = self.pw_config.screenshot_path or f"./screenshots/{url.replace('/', '_')}.png"
                await page.screenshot(path=screenshot_path, full_page=True)
                screenshot_data = screenshot_path

            # Get cookies
            cookies = await self._context.cookies()

            # Get console logs
            console_logs = []
            page.on("console", lambda msg: console_logs.append({
                "type": msg.type,
                "text": msg.text
            }))

            result = {
                "url": url,
                "final_url": page.url,
                "title": title,
                "status_code": response.status if response else None,
                "content_type": response.headers.get("content-type") if response else None,
                "html_length": len(content),
                "text_content": text_content[:10000] if text_content else None,  # Limit text
                "metadata": metadata,
                "links": links,
                "structured_data": structured_data,
                "cookies": [{"name": c["name"], "domain": c["domain"]} for c in cookies],
                "screenshot": screenshot_data,
                "js_result": js_result,
                "console_logs": console_logs[:50],  # Limit logs
                "browser_type": self.pw_config.browser_type,
                "viewport": f"{self.pw_config.viewport_width}x{self.pw_config.viewport_height}",
                "scraped_at": asyncio.get_event_loop().time()
            }

            return result

        finally:
            await page.close()

    async def execute_actions(self, url: str, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a sequence of browser actions.

        Args:
            url: Starting URL
            actions: List of actions like click, type, scroll, wait

        Returns:
            Result after executing all actions
        """
        await self._init_browser()
        page = await self._context.new_page()

        try:
            await page.goto(url, wait_until="networkidle")

            results = []
            for action in actions:
                action_type = action.get("type")
                selector = action.get("selector")
                value = action.get("value")

                if action_type == "click":
                    await page.click(selector)
                    results.append({"action": "click", "selector": selector, "success": True})

                elif action_type == "type":
                    await page.fill(selector, value)
                    results.append({"action": "type", "selector": selector, "success": True})

                elif action_type == "scroll":
                    await page.evaluate(f"window.scrollBy(0, {value or 500})")
                    results.append({"action": "scroll", "pixels": value, "success": True})

                elif action_type == "wait":
                    await page.wait_for_timeout(value or 1000)
                    results.append({"action": "wait", "ms": value, "success": True})

                elif action_type == "wait_for_selector":
                    await page.wait_for_selector(selector, timeout=value or 10000)
                    results.append({"action": "wait_for_selector", "selector": selector, "success": True})

                elif action_type == "screenshot":
                    path = value or f"./screenshots/action_{len(results)}.png"
                    await page.screenshot(path=path)
                    results.append({"action": "screenshot", "path": path, "success": True})

                elif action_type == "evaluate":
                    js_result = await page.evaluate(value)
                    results.append({"action": "evaluate", "result": js_result, "success": True})

            # Get final page state
            final_content = await page.content()
            final_url = page.url

            return {
                "start_url": url,
                "final_url": final_url,
                "actions_executed": results,
                "final_html_length": len(final_content)
            }

        finally:
            await page.close()

    async def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate Playwright scraping result."""
        required_fields = ['url', 'title', 'status_code']
        return all(field in result for field in required_fields)

    async def cleanup(self) -> None:
        """Cleanup browser resources."""
        await super().cleanup()
        
        if self._context:
            await self._context.close()
            self._context = None
            
        if self._browser:
            await self._browser.close()
            self._browser = None
            
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
            
        logger.info("Playwright browser cleaned up")

    def get_browser_metrics(self) -> Dict[str, Any]:
        """Get browser-specific metrics."""
        return {
            **self.get_metrics(),
            "browser_type": self.pw_config.browser_type,
            "headless": self.pw_config.headless,
            "stealth_mode": self.pw_config.stealth_mode,
            "viewport": f"{self.pw_config.viewport_width}x{self.pw_config.viewport_height}"
        }
