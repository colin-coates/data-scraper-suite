# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
CAPTCHA Solving Integration for MJ Data Scraper Suite

Supports multiple CAPTCHA solving services:
- 2Captcha
- Anti-Captcha
- CapMonster
- hCaptcha Solver

Handles reCAPTCHA v2/v3, hCaptcha, image CAPTCHAs, and more.
"""

import asyncio
import aiohttp
import base64
import logging
import time
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CaptchaType(Enum):
    """Supported CAPTCHA types."""
    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V3 = "recaptcha_v3"
    HCAPTCHA = "hcaptcha"
    IMAGE = "image"
    FUNCAPTCHA = "funcaptcha"
    TURNSTILE = "turnstile"  # Cloudflare Turnstile


class CaptchaProvider(Enum):
    """Supported CAPTCHA solving providers."""
    TWO_CAPTCHA = "2captcha"
    ANTI_CAPTCHA = "anti-captcha"
    CAP_MONSTER = "capmonster"


@dataclass
class CaptchaSolveResult:
    """Result of a CAPTCHA solve attempt."""
    success: bool
    solution: Optional[str] = None
    task_id: Optional[str] = None
    cost: float = 0.0
    solve_time: float = 0.0
    error: Optional[str] = None
    provider: Optional[str] = None


class CaptchaSolver:
    """
    Multi-provider CAPTCHA solving service.
    
    Supports automatic failover between providers and
    tracks solve rates and costs.
    """

    # Provider API endpoints
    ENDPOINTS = {
        CaptchaProvider.TWO_CAPTCHA: {
            "submit": "https://2captcha.com/in.php",
            "result": "https://2captcha.com/res.php"
        },
        CaptchaProvider.ANTI_CAPTCHA: {
            "submit": "https://api.anti-captcha.com/createTask",
            "result": "https://api.anti-captcha.com/getTaskResult"
        },
        CaptchaProvider.CAP_MONSTER: {
            "submit": "https://api.capmonster.cloud/createTask",
            "result": "https://api.capmonster.cloud/getTaskResult"
        }
    }

    # Approximate costs per solve (USD)
    COSTS = {
        CaptchaType.RECAPTCHA_V2: 0.003,
        CaptchaType.RECAPTCHA_V3: 0.004,
        CaptchaType.HCAPTCHA: 0.003,
        CaptchaType.IMAGE: 0.001,
        CaptchaType.FUNCAPTCHA: 0.005,
        CaptchaType.TURNSTILE: 0.003,
    }

    def __init__(
        self,
        api_keys: Dict[CaptchaProvider, str],
        primary_provider: CaptchaProvider = CaptchaProvider.TWO_CAPTCHA,
        timeout: int = 120,
        poll_interval: int = 5,
        max_retries: int = 3
    ):
        """
        Initialize CAPTCHA solver.

        Args:
            api_keys: Dict mapping providers to API keys
            primary_provider: Preferred provider to use first
            timeout: Max time to wait for solution (seconds)
            poll_interval: Time between status checks (seconds)
            max_retries: Max retry attempts per provider
        """
        self.api_keys = api_keys
        self.primary_provider = primary_provider
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_retries = max_retries

        # Metrics
        self.total_solves = 0
        self.successful_solves = 0
        self.failed_solves = 0
        self.total_cost = 0.0
        self.total_time = 0.0

        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def solve_recaptcha_v2(
        self,
        site_key: str,
        page_url: str,
        invisible: bool = False,
        data_s: Optional[str] = None
    ) -> CaptchaSolveResult:
        """
        Solve reCAPTCHA v2.

        Args:
            site_key: The site key from the reCAPTCHA element
            page_url: URL of the page with the CAPTCHA
            invisible: Whether it's an invisible reCAPTCHA
            data_s: Optional data-s parameter for some sites

        Returns:
            CaptchaSolveResult with the g-recaptcha-response token
        """
        return await self._solve(
            captcha_type=CaptchaType.RECAPTCHA_V2,
            site_key=site_key,
            page_url=page_url,
            extra_params={"invisible": invisible, "data_s": data_s}
        )

    async def solve_recaptcha_v3(
        self,
        site_key: str,
        page_url: str,
        action: str = "verify",
        min_score: float = 0.7
    ) -> CaptchaSolveResult:
        """
        Solve reCAPTCHA v3.

        Args:
            site_key: The site key from the reCAPTCHA element
            page_url: URL of the page with the CAPTCHA
            action: The action parameter for v3
            min_score: Minimum score required (0.1-0.9)

        Returns:
            CaptchaSolveResult with the token
        """
        return await self._solve(
            captcha_type=CaptchaType.RECAPTCHA_V3,
            site_key=site_key,
            page_url=page_url,
            extra_params={"action": action, "min_score": min_score}
        )

    async def solve_hcaptcha(
        self,
        site_key: str,
        page_url: str
    ) -> CaptchaSolveResult:
        """
        Solve hCaptcha.

        Args:
            site_key: The site key from the hCaptcha element
            page_url: URL of the page with the CAPTCHA

        Returns:
            CaptchaSolveResult with the h-captcha-response token
        """
        return await self._solve(
            captcha_type=CaptchaType.HCAPTCHA,
            site_key=site_key,
            page_url=page_url
        )

    async def solve_image(
        self,
        image_data: bytes,
        case_sensitive: bool = False,
        numeric_only: bool = False,
        min_length: int = 0,
        max_length: int = 0
    ) -> CaptchaSolveResult:
        """
        Solve image-based CAPTCHA.

        Args:
            image_data: Raw image bytes
            case_sensitive: Whether solution is case-sensitive
            numeric_only: Whether solution contains only numbers
            min_length: Minimum solution length
            max_length: Maximum solution length

        Returns:
            CaptchaSolveResult with the text solution
        """
        image_base64 = base64.b64encode(image_data).decode()
        return await self._solve(
            captcha_type=CaptchaType.IMAGE,
            image_base64=image_base64,
            extra_params={
                "case_sensitive": case_sensitive,
                "numeric_only": numeric_only,
                "min_length": min_length,
                "max_length": max_length
            }
        )

    async def solve_turnstile(
        self,
        site_key: str,
        page_url: str
    ) -> CaptchaSolveResult:
        """
        Solve Cloudflare Turnstile.

        Args:
            site_key: The site key from the Turnstile element
            page_url: URL of the page with the CAPTCHA

        Returns:
            CaptchaSolveResult with the cf-turnstile-response token
        """
        return await self._solve(
            captcha_type=CaptchaType.TURNSTILE,
            site_key=site_key,
            page_url=page_url
        )

    async def _solve(
        self,
        captcha_type: CaptchaType,
        site_key: Optional[str] = None,
        page_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> CaptchaSolveResult:
        """
        Internal solve method with provider failover.
        """
        start_time = time.time()
        self.total_solves += 1

        # Try primary provider first, then others
        providers_to_try = [self.primary_provider]
        for provider in CaptchaProvider:
            if provider != self.primary_provider and provider in self.api_keys:
                providers_to_try.append(provider)

        last_error = None
        for provider in providers_to_try:
            if provider not in self.api_keys:
                continue

            try:
                result = await self._solve_with_provider(
                    provider=provider,
                    captcha_type=captcha_type,
                    site_key=site_key,
                    page_url=page_url,
                    image_base64=image_base64,
                    extra_params=extra_params or {}
                )

                if result.success:
                    solve_time = time.time() - start_time
                    result.solve_time = solve_time
                    result.cost = self.COSTS.get(captcha_type, 0.003)
                    result.provider = provider.value

                    self.successful_solves += 1
                    self.total_cost += result.cost
                    self.total_time += solve_time

                    logger.info(f"CAPTCHA solved via {provider.value} in {solve_time:.1f}s")
                    return result

                last_error = result.error

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Provider {provider.value} failed: {e}")
                continue

        # All providers failed
        self.failed_solves += 1
        return CaptchaSolveResult(
            success=False,
            error=last_error or "All providers failed",
            solve_time=time.time() - start_time
        )

    async def _solve_with_provider(
        self,
        provider: CaptchaProvider,
        captcha_type: CaptchaType,
        site_key: Optional[str],
        page_url: Optional[str],
        image_base64: Optional[str],
        extra_params: Dict[str, Any]
    ) -> CaptchaSolveResult:
        """
        Solve CAPTCHA using a specific provider.
        """
        session = await self._get_session()
        api_key = self.api_keys[provider]

        if provider == CaptchaProvider.TWO_CAPTCHA:
            return await self._solve_2captcha(
                session, api_key, captcha_type, site_key, page_url, image_base64, extra_params
            )
        elif provider in (CaptchaProvider.ANTI_CAPTCHA, CaptchaProvider.CAP_MONSTER):
            return await self._solve_anticaptcha_style(
                session, api_key, provider, captcha_type, site_key, page_url, image_base64, extra_params
            )

        return CaptchaSolveResult(success=False, error=f"Unknown provider: {provider}")

    async def _solve_2captcha(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        captcha_type: CaptchaType,
        site_key: Optional[str],
        page_url: Optional[str],
        image_base64: Optional[str],
        extra_params: Dict[str, Any]
    ) -> CaptchaSolveResult:
        """Solve using 2Captcha API."""
        endpoints = self.ENDPOINTS[CaptchaProvider.TWO_CAPTCHA]

        # Build submit request
        params = {"key": api_key, "json": 1}

        if captcha_type == CaptchaType.RECAPTCHA_V2:
            params.update({
                "method": "userrecaptcha",
                "googlekey": site_key,
                "pageurl": page_url,
                "invisible": 1 if extra_params.get("invisible") else 0
            })
        elif captcha_type == CaptchaType.RECAPTCHA_V3:
            params.update({
                "method": "userrecaptcha",
                "version": "v3",
                "googlekey": site_key,
                "pageurl": page_url,
                "action": extra_params.get("action", "verify"),
                "min_score": extra_params.get("min_score", 0.7)
            })
        elif captcha_type == CaptchaType.HCAPTCHA:
            params.update({
                "method": "hcaptcha",
                "sitekey": site_key,
                "pageurl": page_url
            })
        elif captcha_type == CaptchaType.IMAGE:
            params.update({
                "method": "base64",
                "body": image_base64
            })
        elif captcha_type == CaptchaType.TURNSTILE:
            params.update({
                "method": "turnstile",
                "sitekey": site_key,
                "pageurl": page_url
            })

        # Submit task
        async with session.post(endpoints["submit"], data=params) as resp:
            result = await resp.json()

        if result.get("status") != 1:
            return CaptchaSolveResult(success=False, error=result.get("request", "Submit failed"))

        task_id = result["request"]

        # Poll for result
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            await asyncio.sleep(self.poll_interval)

            async with session.get(
                endpoints["result"],
                params={"key": api_key, "action": "get", "id": task_id, "json": 1}
            ) as resp:
                result = await resp.json()

            if result.get("status") == 1:
                return CaptchaSolveResult(
                    success=True,
                    solution=result["request"],
                    task_id=task_id
                )
            elif result.get("request") != "CAPCHA_NOT_READY":
                return CaptchaSolveResult(success=False, error=result.get("request"))

        return CaptchaSolveResult(success=False, error="Timeout waiting for solution", task_id=task_id)

    async def _solve_anticaptcha_style(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        provider: CaptchaProvider,
        captcha_type: CaptchaType,
        site_key: Optional[str],
        page_url: Optional[str],
        image_base64: Optional[str],
        extra_params: Dict[str, Any]
    ) -> CaptchaSolveResult:
        """Solve using Anti-Captcha style API (Anti-Captcha, CapMonster)."""
        endpoints = self.ENDPOINTS[provider]

        # Build task
        task = {}
        if captcha_type == CaptchaType.RECAPTCHA_V2:
            task = {
                "type": "RecaptchaV2TaskProxyless",
                "websiteURL": page_url,
                "websiteKey": site_key,
                "isInvisible": extra_params.get("invisible", False)
            }
        elif captcha_type == CaptchaType.RECAPTCHA_V3:
            task = {
                "type": "RecaptchaV3TaskProxyless",
                "websiteURL": page_url,
                "websiteKey": site_key,
                "minScore": extra_params.get("min_score", 0.7),
                "pageAction": extra_params.get("action", "verify")
            }
        elif captcha_type == CaptchaType.HCAPTCHA:
            task = {
                "type": "HCaptchaTaskProxyless",
                "websiteURL": page_url,
                "websiteKey": site_key
            }
        elif captcha_type == CaptchaType.IMAGE:
            task = {
                "type": "ImageToTextTask",
                "body": image_base64
            }
        elif captcha_type == CaptchaType.TURNSTILE:
            task = {
                "type": "TurnstileTaskProxyless",
                "websiteURL": page_url,
                "websiteKey": site_key
            }

        # Submit task
        async with session.post(
            endpoints["submit"],
            json={"clientKey": api_key, "task": task}
        ) as resp:
            result = await resp.json()

        if result.get("errorId", 0) != 0:
            return CaptchaSolveResult(success=False, error=result.get("errorDescription", "Submit failed"))

        task_id = result["taskId"]

        # Poll for result
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            await asyncio.sleep(self.poll_interval)

            async with session.post(
                endpoints["result"],
                json={"clientKey": api_key, "taskId": task_id}
            ) as resp:
                result = await resp.json()

            if result.get("status") == "ready":
                solution = result.get("solution", {})
                token = (
                    solution.get("gRecaptchaResponse") or
                    solution.get("token") or
                    solution.get("text") or
                    str(solution)
                )
                return CaptchaSolveResult(
                    success=True,
                    solution=token,
                    task_id=str(task_id)
                )
            elif result.get("errorId", 0) != 0:
                return CaptchaSolveResult(success=False, error=result.get("errorDescription"))

        return CaptchaSolveResult(success=False, error="Timeout waiting for solution", task_id=str(task_id))

    def get_metrics(self) -> Dict[str, Any]:
        """Get CAPTCHA solving metrics."""
        return {
            "total_solves": self.total_solves,
            "successful_solves": self.successful_solves,
            "failed_solves": self.failed_solves,
            "success_rate": self.successful_solves / max(1, self.total_solves),
            "total_cost_usd": round(self.total_cost, 4),
            "total_time_seconds": round(self.total_time, 2),
            "avg_solve_time": round(self.total_time / max(1, self.successful_solves), 2),
            "providers_configured": list(self.api_keys.keys())
        }

    async def get_balance(self, provider: CaptchaProvider) -> Optional[float]:
        """Get account balance for a provider."""
        if provider not in self.api_keys:
            return None

        session = await self._get_session()
        api_key = self.api_keys[provider]

        try:
            if provider == CaptchaProvider.TWO_CAPTCHA:
                async with session.get(
                    "https://2captcha.com/res.php",
                    params={"key": api_key, "action": "getbalance", "json": 1}
                ) as resp:
                    result = await resp.json()
                    return float(result.get("request", 0))

            elif provider in (CaptchaProvider.ANTI_CAPTCHA, CaptchaProvider.CAP_MONSTER):
                endpoint = "https://api.anti-captcha.com/getBalance" if provider == CaptchaProvider.ANTI_CAPTCHA else "https://api.capmonster.cloud/getBalance"
                async with session.post(endpoint, json={"clientKey": api_key}) as resp:
                    result = await resp.json()
                    return result.get("balance")

        except Exception as e:
            logger.error(f"Failed to get balance for {provider.value}: {e}")
            return None

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
