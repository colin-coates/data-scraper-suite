# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Base Scraper Class for MJ Data Scraper Suite

Provides common functionality for all scrapers including error handling,
logging hooks, proxy rotation, and rate-limit callbacks.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .retry_utils import retry_async, RetryConfig, retry_on_network_errors, retry_on_rate_limits
from .control_models import ScrapeControlContract

logger = logging.getLogger(__name__)


class AbortScrape(Exception):
    """Exception raised when scraping should be aborted due to governance violations."""
    pass


@dataclass
class ScraperResult:
    """Standardized result structure for all scrapers."""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    scraper_name: str = ""
    target_url: str = ""
    response_time: float = 0.0
    retry_count: int = 0


@dataclass
class ScraperConfig:
    """Configuration for scraper instances."""
    name: str
    user_agent_rotation: bool = True
    proxy_rotation: bool = True
    rate_limit_delay: float = 1.0
    max_retries: int = 3
    timeout: int = 30
    headers: Dict[str, str] = field(default_factory=dict)
    cookies: Dict[str, str] = field(default_factory=dict)
    proxies: List[str] = field(default_factory=list)
    retry_config: Optional[RetryConfig] = None  # Custom retry configuration
    enable_retry: bool = True  # Enable retry logic


class BaseScraper(ABC):
    """Base class for all scrapers with common functionality."""

    # Class attributes for scraper classification
    ROLE = None  # discovery / verification / enrichment / browser
    TIER = None  # 1, 2, 3 (complexity/compliance level)
    SUPPORTED_EVENTS = []  # List of supported event types (weddings/corporate/social/professional)

    def __init__(self, config: ScraperConfig, control: Optional[ScrapeControlContract] = None):
        self.config = config
        self.control = control
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self._session_cookies = {}
        self._rate_limit_until = 0.0
        self._error_count = 0
        self._success_count = 0

        # Governance metrics
        self.pages = 0
        self.records = 0
        self.start_time = time.time()

        # Callbacks
        self.on_error: Optional[Callable[[Exception, Dict[str, Any]], None]] = None
        self.on_rate_limit: Optional[Callable[[float], None]] = None
        self.on_success: Optional[Callable[[ScraperResult], None]] = None
        self.on_retry: Optional[Callable[[int, Exception], None]] = None

        # Anti-detection integration
        self.anti_detection = None

    async def scrape(self, target: Dict[str, Any]) -> ScraperResult:
        """
        Main scraping method with error handling, retries, and runtime governance.

        Args:
            target: Target information (URL, ID, search terms, etc.)

        Returns:
            ScraperResult with scraping outcome
        """
        start_time = time.time()
        result = ScraperResult(
            success=False,
            scraper_name=self.config.name,
            target_url=target.get('url', target.get('target', ''))
        )

        try:
            # Pre-flight governance check
            if self.control:
                self.enforce()

            # Check rate limiting
            await self._check_rate_limit()

            # Pre-scrape hooks
            await self._pre_scrape_hook(target)

            # Execute scrape with retries and runtime monitoring
            result = await self._scrape_with_retries(target, result)

            # Post-scrape hooks
            await self._post_scrape_hook(result)

            # Update final metrics
            result.response_time = time.time() - start_time

            if result.success:
                self._success_count += 1
                # Governance metrics are now updated in _execute_scrape_with_monitoring
                if self.on_success:
                    self.on_success(result)
            else:
                self._error_count += 1

        except AbortScrape as e:
            # Handle governance violations during execution
            result.error_message = f"Governance violation: {e}"
            result.response_time = time.time() - start_time
            self._error_count += 1
            self.logger.error(f"Governance violation during scraping: {e}")

        except Exception as e:
            result.error_message = str(e)
            result.response_time = time.time() - start_time
            self._error_count += 1

            self.logger.error(f"Scraping failed for target {target}: {e}")

            if self.on_error:
                self.on_error(e, target)

        return result

    async def _scrape_with_retries(self, target: Dict[str, Any], result: ScraperResult) -> ScraperResult:
        """Execute scraping with retry logic using the retry utility."""
        if not self.config.enable_retry:
            # Fallback to single attempt if retry is disabled
            try:
                result.retry_count = 0
                scrape_result = await self._execute_scrape_with_monitoring(target)

                if await self._validate_result(scrape_result):
                    result.success = True
                    result.data = scrape_result
                    return result
                else:
                    result.error_message = "Result validation failed"
                    return result

            except Exception as e:
                result.error_message = str(e)
                return result

        # Use retry utility for enhanced retry logic
        retry_config = self.config.retry_config or RetryConfig(
            max_attempts=self.config.max_retries + 1,
            base_delay=self.config.rate_limit_delay,
            max_delay=60.0,  # Max 1 minute between retries
            backoff_factor=2.0,
            success_hook=lambda attempt: self.logger.info(f"Scrape succeeded on attempt {attempt}"),
            failure_hook=lambda e, attempt: self.logger.error(f"Scrape failed after {attempt} attempts: {e}")
        )

        # Create retry-enabled scrape function
        @retry_async(retry_config)
        async def scrape_with_retry():
            # Execute the actual scraping logic with monitoring
            scrape_result = await self._execute_scrape_with_monitoring(target)

            # Validate result
            if await self._validate_result(scrape_result):
                return scrape_result
            else:
                raise ValueError("Result validation failed")

        try:
            # Execute with retry logic
            scrape_result = await scrape_with_retry()
            result.success = True
            result.data = scrape_result
            result.retry_count = retry_config.max_attempts - 1  # This would need to be tracked better

        except Exception as e:
            result.error_message = str(e)
            result.retry_count = retry_config.max_attempts

            # Call retry callback if available
            if self.on_retry:
                self.on_retry(result.retry_count, e)

        return result

    async def _execute_scrape_with_monitoring(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute scraping with runtime governance monitoring.

        This wrapper provides continuous compliance checking during execution.
        """
        # Runtime governance checks (before execution)
        if self.control:
            # Check runtime against budget continuously
            time_elapsed = time.time() - self.start_time
            if hasattr(self.control.budget, 'max_runtime_minutes') and time_elapsed > (self.control.budget.max_runtime_minutes * 60):
                self.logger.warning(f"Runtime budget exceeded: {time_elapsed:.1f}s > {self.control.budget.max_runtime_minutes * 60}s")
                raise AbortScrape(f"Runtime budget exceeded: {time_elapsed:.1f}s used")

            # Periodic governance re-validation (every 10 pages or configurable interval)
            if self.pages > 0 and self.pages % 10 == 0:
                try:
                    self.enforce()
                    self.logger.debug(f"Governance check passed at page {self.pages}")
                except AbortScrape as e:
                    self.logger.error(f"Governance violation during execution: {e}")
                    raise

        # Execute the actual scraping logic
        result = await self._execute_scrape(target)

        # Update page counter after successful execution
        if self.control:
            self.pages += 1

        return result

    @abstractmethod
    async def _execute_scrape(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the actual scraping logic.

        This method must be implemented by subclasses.

        Args:
            target: Target information specific to the scraper type

        Returns:
            Dict containing scraped data
        """
        pass

    async def _validate_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate scraping result.

        Override in subclasses for scraper-specific validation.

        Args:
            result: Scraped data to validate

        Returns:
            True if result is valid
        """
        return bool(result and isinstance(result, dict))

    async def _pre_scrape_hook(self, target: Dict[str, Any]) -> None:
        """Hook called before scraping begins."""
        self.logger.info(f"Starting scrape for target: {target.get('url', target.get('target', 'unknown'))}")

        # Update anti-detection layer if available
        if self.anti_detection:
            await self.anti_detection.prepare_for_request(target)

    async def _post_scrape_hook(self, result: ScraperResult) -> None:
        """Hook called after scraping completes."""
        status = "SUCCESS" if result.success else "FAILED"
        self.logger.info(f"Scrape {status}: {result.response_time:.2f}s, "
                        f"{len(result.data) if result.data else 0} items collected")

        # Update anti-detection layer
        if self.anti_detection:
            await self.anti_detection.post_request_update(result)

    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        current_time = time.time()

        if current_time < self._rate_limit_until:
            wait_time = self._rate_limit_until - current_time
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")

            if self.on_rate_limit:
                self.on_rate_limit(wait_time)

            await asyncio.sleep(wait_time)

        # Update rate limit
        self._rate_limit_until = current_time + self.config.rate_limit_delay

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update scraper configuration."""
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated config {key}: {value}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get scraper performance metrics."""
        total_requests = self._success_count + self._error_count
        runtime = time.time() - self.start_time

        return {
            # Basic metrics
            'scraper_name': self.config.name,
            'success_count': self._success_count,
            'error_count': self._error_count,
            'total_requests': total_requests,
            'success_rate': self._success_count / max(1, total_requests),

            # Governance metrics
            'pages_scraped': self.pages,
            'records_collected': self.records,
            'runtime_seconds': runtime,
            'pages_per_second': self.pages / max(1, runtime),
            'records_per_second': self.records / max(1, runtime),

            # Configuration
            'rate_limit_delay': self.config.rate_limit_delay,
            'max_retries': self.config.max_retries,
            'timeout': self.config.timeout,

            # Classification
            'role': self.ROLE,
            'tier': self.TIER
        }

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._success_count = 0
        self._error_count = 0
        self.logger.info("Metrics reset")

    def set_anti_detection(self, anti_detection_layer) -> None:
        """Set anti-detection layer for this scraper."""
        self.anti_detection = anti_detection_layer
        self.logger.info("Anti-detection layer configured")

    def enforce(self) -> None:
        """
        Enforce governance controls before scraping operations.

        Validates time windows, budget constraints, scope limitations,
        role compatibility, and event type support.
        Should be called before starting scraping operations.
        """
        if not self.control:
            self.logger.warning("No control contract provided - skipping governance enforcement")
            return

        # Core governance checks
        enforce_time(self.control)
        enforce_budget(self.control)
        enforce_scope(self.control)

        # AI precheck validation (first line of defense)
        if not ai_precheck(self.control):
            raise AbortScrape("AI rejected scrape")

        # Role and capability validation
        if self.ROLE and self.control.intent.allowed_role and self.ROLE != self.control.intent.allowed_role:
            raise AbortScrape(f"Scraper role mismatch: {self.ROLE} != {self.control.intent.allowed_role}")

        if not self.control.intent.geography:
            raise AbortScrape("Geography required")

        if self.control.intent.event_type and self.control.intent.event_type not in self.SUPPORTED_EVENTS:
            raise AbortScrape(f"Unsupported event type: {self.control.intent.event_type}. Supported: {self.SUPPORTED_EVENTS}")

        # Tier 3 human approval requirement
        if self.TIER == 3 and not self.control.human_override:
            raise AbortScrape("Browser scraping requires human approval")

        # AI-based precheck for intelligent governance
        if not ai_precheck(self.control):
            raise AbortScrape("AI rejected scrape")

    async def cleanup(self) -> None:
        """Cleanup scraper resources."""
        self.logger.info(f"Cleaning up scraper {self.config.name}")
        self._session_cookies.clear()
        self._rate_limit_until = 0.0

    def __str__(self) -> str:
        role_info = f", role={self.ROLE}" if self.ROLE else ""
        tier_info = f", tier={self.TIER}" if self.TIER else ""
        return f"{self.__class__.__name__}(name={self.config.name}, success_rate={self.get_metrics()['success_rate']:.2f}{role_info}{tier_info})"


# AI-based governance functions
def ai_precheck(control: ScrapeControlContract) -> bool:
    """
    AI-powered precheck for scraping operations using machine learning and heuristics.

    Evaluates multiple factors to determine if scraping operation should proceed:
    - Risk assessment of target sources and data types
    - Historical success rates and performance patterns
    - Legal and compliance considerations
    - Resource utilization predictions
    - Business value and ROI assessment
    - Anti-detection and blocking probability

    Args:
        control: Complete control contract with intent, budget, and constraints

    Returns:
        True if operation should proceed, False if rejected by AI
    """
    intent = control.intent
    budget = control.budget

    # Factor 1: Source Risk Assessment
    high_risk_sources = {"facebook", "instagram", "twitter"}
    medium_risk_sources = {"linkedin", "news"}
    low_risk_sources = {"web", "company_websites", "business_directories", "public_records"}

    source_risk_score = 0
    high_risk_count = 0
    for source in intent.sources:
        if source in high_risk_sources:
            source_risk_score += 3
            high_risk_count += 1
        elif source in medium_risk_sources:
            source_risk_score += 2
        elif source in low_risk_sources:
            source_risk_score += 1

    # Reject if multiple high-risk sources are combined
    if high_risk_count > 1:
        logger.warning(f"AI precheck: Multiple high-risk sources ({high_risk_count}) - rejecting")
        return False

    # Factor 2: Budget Efficiency Check
    estimated_records = _estimate_record_yield(intent)
    estimated_cost = _estimate_scraping_cost(intent, budget)
    efficiency_score = estimated_records / max(estimated_cost, 0.01)

    # Reject if efficiency is too low (< 10 records per dollar for high-risk sources)
    efficiency_threshold = 5.0 if high_risk_count > 0 else 3.0
    if efficiency_score < efficiency_threshold:
        logger.warning(f"AI precheck: Low efficiency score {efficiency_score:.2f} - rejecting")
        return False

    # Factor 3: Geography Scope Check
    geography_score = _assess_geography_risk(intent.geography)

    # Factor 4: Historical Performance (simplified heuristic)
    # In production, this would query actual historical telemetry data
    historical_success_rate = _get_historical_success_rate(intent.sources, intent.event_type)

    # Factor 5: Time Window Optimization
    time_score = _assess_time_window_suitability(control.deployment_window, intent.sources)

    # Factor 6: Compliance and Legal Assessment
    compliance_score = _assess_compliance_risk(intent.sources, intent.geography)

    # AI Decision Engine (weighted scoring system)
    total_score = (
        source_risk_score * 0.25 +          # 25% weight on source risk
        (efficiency_score / 10) * 0.30 +    # 30% weight on efficiency (normalized)
        geography_score * 0.15 +            # 15% weight on geography
        historical_success_rate * 0.15 +     # 15% weight on history
        time_score * 0.10 +                  # 10% weight on timing
        compliance_score * 0.05              # 5% weight on compliance
    )

    # Decision threshold (70% confidence required)
    confidence_threshold = 7.0
    decision = total_score >= confidence_threshold

    logger.info(
        f"AI precheck result: {'APPROVED' if decision else 'REJECTED'} "
        f"(score: {total_score:.2f}/{confidence_threshold}, "
        f"efficiency: {efficiency_score:.2f}, "
        f"sources: {intent.sources})"
    )

    return decision


def _estimate_record_yield(intent) -> int:
    """Estimate expected record yield based on intent parameters."""
    base_yield = {
        "linkedin": 150,
        "facebook": 80,
        "twitter": 200,
        "instagram": 120,
        "web": 50,
        "company_websites": 30,
        "news": 100,
        "business_directories": 75,
        "public_records": 25
    }

    total_yield = 0
    for source in intent.sources:
        total_yield += base_yield.get(source, 10)

    # Adjust for event type
    if intent.event_type == "weddings":
        total_yield *= 0.8  # More specific, potentially lower yield
    elif intent.event_type == "corporate":
        total_yield *= 1.2  # Broader scope, higher yield

    return int(total_yield)


def _estimate_scraping_cost(intent, budget) -> float:
    """Estimate scraping cost based on sources and constraints."""
    cost_per_source = {
        "linkedin": 0.15,
        "facebook": 0.25,
        "twitter": 0.08,
        "instagram": 0.20,
        "web": 0.02,
        "company_websites": 0.05,
        "news": 0.03,
        "business_directories": 0.04,
        "public_records": 0.01
    }

    estimated_cost = sum(cost_per_source.get(source, 0.10) for source in intent.sources)
    estimated_cost *= len(intent.sources)  # Scale with number of sources

    return max(estimated_cost, 0.01)


def _assess_geography_risk(geography: Dict[str, Any]) -> float:
    """Assess risk level of geographic targeting."""
    if not geography:
        return 1.0  # Low score for unspecified geography

    # Prefer specific geographic targeting over global
    if "country" in geography and len(geography) == 1:
        return 8.0  # High score for country-specific
    elif "state" in geography or "city" in geography:
        return 9.0  # Very high score for granular targeting
    else:
        return 5.0  # Medium score for broad targeting


def _get_historical_success_rate(sources: List[str], event_type: Optional[str]) -> float:
    """Get historical success rate for similar operations (simplified heuristic)."""
    # In production, this would query actual historical telemetry data
    base_rates = {
        "linkedin": 0.85,
        "facebook": 0.60,
        "twitter": 0.75,
        "instagram": 0.55,
        "web": 0.95,
        "company_websites": 0.80,
        "news": 0.90,
        "business_directories": 0.85,
        "public_records": 0.98
    }

    avg_rate = sum(base_rates.get(source, 0.70) for source in sources) / len(sources)

    # Adjust for event type specificity
    if event_type:
        avg_rate *= 0.9  # Slightly lower for specific events

    return avg_rate


def _assess_time_window_suitability(deployment_window, sources: List[str]) -> float:
    """Assess suitability of deployment time window."""
    # Simple heuristic: prefer business hours for professional data
    # and off-hours for social media to avoid detection
    professional_sources = {"linkedin", "company_websites"}
    social_sources = {"facebook", "instagram", "twitter"}

    has_professional = any(s in professional_sources for s in sources)
    has_social = any(s in social_sources for s in sources)

    if has_professional and not has_social:
        return 8.0  # Good for professional data
    elif has_social and not has_professional:
        return 7.0  # Decent for social data
    else:
        return 6.0  # Neutral for mixed sources


def _assess_compliance_risk(sources: List[str], geography: Dict[str, Any]) -> float:
    """Assess compliance and legal risk factors."""
    # High compliance risk sources
    high_risk = {"facebook", "instagram"}
    medium_risk = {"linkedin", "twitter"}

    risk_score = 0
    for source in sources:
        if source in high_risk:
            risk_score += 3
        elif source in medium_risk:
            risk_score += 2
        else:
            risk_score += 1

    # Geography affects compliance (GDPR, CCPA, etc.)
    if "country" in geography:
        country = geography["country"]
        if country in ["EU", "UK"]:  # High privacy regulation
            risk_score += 2
        elif country == "US":  # Moderate regulation
            risk_score += 1

    # Convert risk score to compliance score (inverse relationship)
    compliance_score = max(0, 10 - risk_score)
    return compliance_score / 10.0  # Normalize to 0-1


def abort(message: str) -> None:
    """
    Abort scraping operation with specified message.

    This is a convenience function for clean operation abortion
    with consistent error handling.

    Args:
        message: Abort reason message

    Raises:
        AbortScrape: Always raised with the provided message
    """
    logger.warning(f"Scraping operation aborted: {message}")
    raise AbortScrape(message)


def ai_precheck(control: ScrapeControlContract) -> bool:
    """
    AI-powered precheck for scraping operations.

    Evaluates scraping intent, risks, compliance, and operational factors
    to determine if the operation should proceed.

    Args:
        control: Control contract containing scraping parameters and intent

    Returns:
        bool: True if operation should proceed, False if rejected
    """
    import random  # For simulation - replace with actual AI model
    from datetime import datetime

    intent = control.intent
    budget = control.budget

    # Risk assessment factors
    risk_score = 0.0
    risk_factors = []

    # 1. Source type risk assessment
    high_risk_sources = ["facebook", "instagram", "twitter"]
    medium_risk_sources = ["company_websites", "news"]

    for source in intent.sources:
        if source in high_risk_sources:
            risk_score += 0.8
            risk_factors.append(f"high_risk_source_{source}")
        elif source in medium_risk_sources:
            risk_score += 0.4
            risk_factors.append(f"medium_risk_source_{source}")

    # 2. Geographic scope assessment
    if not intent.geography:
        risk_score += 1.0  # No geography = maximum risk
        risk_factors.append("no_geography_specified")
    elif len(intent.geography) > 5:
        risk_score += 0.6  # Too broad geographic scope
        risk_factors.append("overly_broad_geography")

    # 3. Event type risk assessment
    if intent.event_type:
        sensitive_events = ["corporate", "professional"]
        if intent.event_type in sensitive_events:
            risk_score += 0.5
            risk_factors.append(f"sensitive_event_type_{intent.event_type}")

    # 4. Budget assessment
    if budget.max_records > 10000:
        risk_score += 0.3
        risk_factors.append("large_record_volume")

    if budget.max_runtime_minutes > 120:
        risk_score += 0.4
        risk_factors.append("extended_runtime")

    # 5. Time-based risk assessment
    now = datetime.utcnow()
    current_hour = now.hour
    current_day = now.weekday()

    # Higher risk during business hours (potential blocking)
    if 9 <= current_hour <= 17 and current_day < 5:  # Business hours, weekdays
        risk_score += 0.2
        risk_factors.append("business_hours_execution")

    # 6. Authorization assessment
    if not control.authorization:
        risk_score += 2.0  # No authorization = critical rejection
        risk_factors.append("no_authorization")
    elif control.authorization.expires_at <= now:
        risk_score += 2.0  # Expired authorization = critical rejection
        risk_factors.append("expired_authorization")

    # 7. Scraper tier assessment (from intent if specified)
    if intent.allowed_role == "browser":
        risk_score += 0.7  # Browser scraping is inherently riskier
        risk_factors.append("browser_scraping_tier")

    # AI Decision Logic
    # For simulation: reject based on risk thresholds and critical factors
    # In production: replace with actual ML model inference

    # Critical rejection criteria (always reject)
    if "no_authorization" in risk_factors or "expired_authorization" in risk_factors:
        logger.warning(f"AI precheck rejected: Authorization issues, factors: {risk_factors}")
        return False

    if "no_geography_specified" in risk_factors:
        logger.warning(f"AI precheck rejected: No geographic targeting, factors: {risk_factors}")
        return False

    # High-risk rejection criteria
    if risk_score >= 2.0:  # Very high risk score
        logger.warning(f"AI precheck rejected: Critical risk score {risk_score:.2f}, factors: {risk_factors}")
        return False

    # Probabilistic rejection for medium risk (simulate AI uncertainty)
    if risk_score > 0.6:
        rejection_probability = min(0.4, risk_score * 0.3)  # Up to 40% rejection for high medium risk
        if random.random() < rejection_probability:
            logger.warning(f"AI precheck rejected: Medium risk assessment {risk_score:.2f}, factors: {risk_factors}")
            return False

    # Accept if risk score is low enough
    if risk_score <= 0.3:
        logger.info(f"AI precheck approved: Low risk operation {risk_score:.2f}")
        return True

    # For medium risk, additional checks
    # Check if human override is available for high-risk operations
    if risk_score > 0.8 and control.human_override:
        logger.info(f"AI precheck approved with human override: {risk_score:.2f}, factors: {risk_factors}")
        return True

    # Default: approve for now (conservative approach)
    # In production, this would be more sophisticated
    logger.info(f"AI precheck approved: Acceptable risk {risk_score:.2f}, factors: {risk_factors}")
    return True


# Governance enforcement functions
def enforce_time(control: ScrapeControlContract) -> None:
    """
    Enforce time-based governance controls.

    Args:
        control: Control contract with deployment window

    Raises:
        RuntimeError: If current time is outside deployment window
    """
    from datetime import datetime
    now = datetime.utcnow()

    if now < control.deployment_window.earliest_start:
        wait_seconds = (control.deployment_window.earliest_start - now).total_seconds()
        raise RuntimeError(f"Deployment window not open yet. Wait {wait_seconds:.0f} seconds.")

    if now > control.deployment_window.latest_start:
        raise RuntimeError("Deployment window has closed.")


def enforce_budget(control: ScrapeControlContract) -> None:
    """
    Enforce budget-based governance controls.

    Args:
        control: Control contract with budget limits

    Raises:
        RuntimeError: If budget constraints are violated
    """
    budget = control.budget

    # Check runtime budget (would need current runtime tracking)
    # This is a placeholder - actual implementation would track runtime

    # Check page budget
    if hasattr(budget, 'max_pages') and budget.max_pages <= 0:
        raise RuntimeError("Page budget exhausted")

    # Check record budget
    if budget.max_records <= 0:
        raise RuntimeError("Record budget exhausted")

    # Check browser instance budget
    if hasattr(budget, 'max_browser_instances') and budget.max_browser_instances <= 0:
        raise RuntimeError("Browser instance budget exhausted")


def enforce_scope(control: ScrapeControlContract) -> None:
    """
    Enforce scope-based governance controls.

    Args:
        control: Control contract with intent specifications

    Raises:
        RuntimeError: If scope constraints are violated
    """
    intent = control.intent

    # Validate data sources
    if not intent.sources:
        raise RuntimeError("No data sources specified")

    # Additional scope validations could be added here
    # (e.g., industry restrictions, data type limitations, etc.)
