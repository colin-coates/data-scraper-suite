# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Enhanced Base Scraper Class for MJ Data Scraper Suite

Enterprise-grade scraper base class with full intelligence integration,
governance enforcement, runtime monitoring, and advanced analytics.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum

from core.control_models import ScrapeControlContract, ScrapeTempo
from core.base_scraper import ai_precheck
from core.scrape_telemetry import emit_telemetry
from core.sentinels.sentinel_orchestrator import run_sentinels
from core.safety_verdict import safety_verdict
from core.cost_predictor import predict_scraping_cost
from core.intent_classifier import IntentRiskLevel
from core.execution_mode_classifier import ExecutionProfile, ExecutionMode
from core.models.asset_signal import AssetType, SignalType

logger = logging.getLogger(__name__)


class ScraperRole(Enum):
    """Scraper classification by primary function."""
    DISCOVERY = "discovery"      # Finding new targets/data sources
    VERIFICATION = "verification"  # Validating existing data
    ENRICHMENT = "enrichment"    # Adding additional information
    BROWSER = "browser"          # Browser-based scraping (Tier 3)


class ScraperTier(Enum):
    """Scraper classification by operational requirements."""
    TIER_1 = 1  # Standard scraping (basic validation, low risk)
    TIER_2 = 2  # Enhanced scraping (compliance required, medium risk)
    TIER_3 = 3  # Critical scraping (human approval required, high risk)


class ScraperResult:
    """Enhanced result container with intelligence data."""
    def __init__(self, target: str, data: List[Dict[str, Any]], success: bool = True,
                 error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.target = target
        self.data = data
        self.success = success
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
        self.scraper_name = None  # Will be set by scraper
        self.execution_time = None
        self.cost_estimate = None
        self.risk_assessment = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "target": self.target,
            "data_count": len(self.data),
            "data": self.data,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "scraper_name": self.scraper_name,
            "execution_time": self.execution_time,
            "cost_estimate": self.cost_estimate,
            "risk_assessment": self.risk_assessment
        }


class ScraperConfig:
    """Enhanced scraper configuration with intelligence settings."""
    def __init__(self,
                 name: str,
                 role: ScraperRole,
                 tier: ScraperTier,
                 supported_signals: Optional[List[SignalType]] = None,
                 supported_assets: Optional[List[AssetType]] = None,
                 rate_limits: Optional[Dict[str, Any]] = None,
                 timeout_seconds: int = 30,
                 max_retries: int = 3,
                 enable_ai_precheck: bool = True,
                 enable_sentinel_monitoring: bool = True,
                 enable_cost_tracking: bool = True,
                 enable_telemetry: bool = True,
                 custom_config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.role = role
        self.tier = tier
        self.supported_signals = supported_signals or []
        self.supported_assets = supported_assets or []
        self.rate_limits = rate_limits or {}
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.enable_ai_precheck = enable_ai_precheck
        self.enable_sentinel_monitoring = enable_sentinel_monitoring
        self.enable_cost_tracking = enable_cost_tracking
        self.enable_telemetry = enable_telemetry
        self.custom_config = custom_config or {}


class EnhancedBaseScraper(ABC):
    """Enterprise-grade base scraper with full intelligence integration."""

    # Scraper classification - must be overridden in subclasses
    ROLE: ScraperRole = ScraperRole.DISCOVERY
    TIER: ScraperTier = ScraperTier.TIER_1

    # Supported capabilities
    SUPPORTED_SIGNALS: List[SignalType] = []
    SUPPORTED_ASSETS: List[AssetType] = []

    def __init__(self, config: ScraperConfig, control: Optional[ScrapeControlContract] = None):
        self.config = config
        self.control = control
        self.logger = logging.getLogger(f"{__name__}.{config.name}")

        # Operational state
        self.is_running = False
        self.is_initialized = False

        # Performance tracking
        self.start_time = None
        self.pages_scraped = 0
        self.records_collected = 0
        self.requests_made = 0
        self.errors_encountered = 0

        # Intelligence data
        self.intelligence_assessment = {}
        self.cost_tracking = {}
        self.risk_assessment = {}
        self.execution_profile = None

        # Supported events/types (can be overridden in subclasses)
        self.SUPPORTED_EVENTS = []

        logger.info(f"EnhancedBaseScraper initialized: {config.name} ({self.ROLE.value}, Tier {self.TIER.value})")

    @abstractmethod
    async def scrape(self, target: str, **kwargs) -> ScraperResult:
        """Enhanced abstract method to perform scraping with intelligence.

        Args:
            target: The target to scrape (URL, search term, etc.)
            **kwargs: Additional parameters

        Returns:
            Enhanced ScraperResult with intelligence data
        """
        pass

    @abstractmethod
    def validate_target(self, target: str) -> bool:
        """Validate if target is suitable for this scraper.

        Args:
            target: Target to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    def enforce(self) -> None:
        """Enforce governance policies and operational controls."""
        if not self.control:
            return

        # Time-based enforcement
        self._enforce_time_window()

        # Budget enforcement
        self._enforce_budget_limits()

        # Scope enforcement
        self._enforce_scope_limits()

        # Role/geography/event validation
        self._validate_role_compatibility()
        self._validate_geography_scope()
        self._validate_supported_events()

    def _enforce_time_window(self) -> None:
        """Enforce deployment time window restrictions."""
        if not self.control or not self.control.deployment_window:
            return

        now = datetime.utcnow()
        window = self.control.deployment_window

        if now < window.earliest_start:
            raise RuntimeError(f"Deployment too early. Earliest start: {window.earliest_start}")
        if now > window.latest_start:
            raise RuntimeError(f"Deployment window expired. Latest start: {window.latest_start}")

    def _enforce_budget_limits(self) -> None:
        """Enforce budget limits and runtime constraints."""
        if not self.control or not self.control.budget:
            return

        budget = self.control.budget
        current_time = time.time()

        # Check runtime limits
        if hasattr(self, 'start_time') and self.start_time:
            elapsed = current_time - self.start_time
            if elapsed > budget.max_runtime_minutes * 60:
                raise RuntimeError(f"Runtime budget exceeded: {elapsed/60:.1f}min > {budget.max_runtime_minutes}min")

        # Check page limits
        if self.pages_scraped > budget.max_pages:
            raise RuntimeError(f"Page budget exceeded: {self.pages_scraped} > {budget.max_pages}")

        # Check record limits
        if self.records_collected > budget.max_records:
            raise RuntimeError(f"Record budget exceeded: {self.records_collected} > {budget.max_records}")

    def _enforce_scope_limits(self) -> None:
        """Enforce operational scope limits."""
        if not self.control or not self.control.budget:
            return

        # Additional scope validations can be added here
        pass

    def _validate_role_compatibility(self) -> None:
        """Validate scraper role compatibility with control intent."""
        if not self.control or not self.control.intent:
            return

        allowed_role = getattr(self.control.intent, 'allowed_role', None)
        if allowed_role and self.ROLE.value != allowed_role:
            raise RuntimeError(f"Scraper role mismatch: {self.ROLE.value} != {allowed_role}")

    def _validate_geography_scope(self) -> None:
        """Validate geography scope compatibility."""
        if not self.control or not self.control.intent or not self.control.intent.geography:
            raise RuntimeError("Geography scope required but not specified")

        # Geography validation logic can be enhanced
        pass

    def _validate_supported_events(self) -> None:
        """Validate supported event types."""
        if not self.control or not self.control.intent:
            return

        event_type = getattr(self.control.intent, 'event_type', None)
        if event_type and event_type not in self.SUPPORTED_EVENTS:
            raise RuntimeError(f"Unsupported event type: {event_type}")

    async def _execute_scrape_with_monitoring(self, target: str, **kwargs) -> ScraperResult:
        """Execute scraping with continuous monitoring and governance."""
        start_time = time.time()

        # Pre-scrape governance check
        self.enforce()

        # Initialize result
        result = ScraperResult(target=target, data=[])
        result.scraper_name = self.config.name

        try:
            # AI precheck if enabled
            if self.config.enable_ai_precheck and self.control:
                ai_approved = await ai_precheck(self.control)
                if not ai_approved:
                    raise RuntimeError("AI precheck rejected scrape operation")

            # Sentinel monitoring if enabled
            if self.config.enable_sentinel_monitoring:
                target_info = {"domain": target, "target_type": "scrape_target"}
                sentinels_result = await run_sentinels(target_info)
                verdict = safety_verdict(sentinels_result, self.control)

                if verdict.action in ["block", "delay"]:
                    raise RuntimeError(f"Sentinel verdict: {verdict.reason}")

            # Execute the actual scrape
            scrape_result = await self.scrape(target, **kwargs)

            # Update result with scrape data
            result.data = scrape_result.data
            result.success = scrape_result.success
            result.error = scrape_result.error
            result.metadata = scrape_result.metadata

            # Update performance counters
            self.pages_scraped += getattr(scrape_result, 'pages_scraped', 1)
            self.records_collected += len(result.data)
            self.requests_made += getattr(scrape_result, 'requests_made', 1)

            # Runtime governance check
            self.enforce()

        except Exception as e:
            result.success = False
            result.error = str(e)
            self.errors_encountered += 1
            self.logger.error(f"Scraping failed for {target}: {e}")
            raise

        finally:
            # Record execution time
            execution_time = time.time() - start_time
            result.execution_time = execution_time

            # Cost estimation if enabled
            if self.config.enable_cost_tracking and self.control:
                try:
                    cost_prediction = await predict_scraping_cost(
                        asset_type=self._infer_asset_type(),
                        signal_type=self.SUPPORTED_SIGNALS[0] if self.SUPPORTED_SIGNALS else None,
                        execution_mode=None,
                        scope_size=1,  # Single target
                        control=self.control
                    )
                    result.cost_estimate = cost_prediction.predicted_cost
                    self.cost_tracking[target] = cost_prediction.predicted_cost
                except Exception as e:
                    self.logger.debug(f"Cost estimation failed: {e}")

            # Telemetry emission if enabled
            if self.config.enable_telemetry:
                try:
                    await emit_telemetry(
                        scraper=self.config.name,
                        role=self.ROLE.value,
                        cost_estimate=result.cost_estimate,
                        records_found=len(result.data),
                        blocked_reason=result.error if not result.success else None,
                        runtime=execution_time
                    )
                except Exception as e:
                    self.logger.debug(f"Telemetry emission failed: {e}")

        return result

    def _infer_asset_type(self) -> AssetType:
        """Infer asset type from supported signals."""
        if any(s in [SignalType.LIEN, SignalType.MORTGAGE, SignalType.DEED] for s in self.SUPPORTED_SIGNALS):
            return AssetType.SINGLE_FAMILY_HOME
        elif any(s in [SignalType.COURT_CASE, SignalType.JUDGMENT] for s in self.SUPPORTED_SIGNALS):
            return AssetType.COMPANY
        return AssetType.PERSON

    async def run_enhanced(self, targets: List[str], **kwargs) -> Dict[str, Any]:
        """Enhanced run method with full intelligence integration.

        Args:
            targets: List of targets to scrape
            **kwargs: Additional parameters

        Returns:
            Enhanced results summary with intelligence data
        """
        self.is_running = True
        self.is_initialized = True
        self.start_time = time.time()

        results = {
            "scraper": self.config.name,
            "role": self.ROLE.value,
            "tier": self.TIER.value,
            "targets_processed": 0,
            "data_collected": 0,
            "errors": 0,
            "pages_scraped": 0,
            "requests_made": 0,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "results": [],
            "intelligence_summary": {
                "cost_tracking": {},
                "risk_assessment": {},
                "performance_metrics": {}
            }
        }

        try:
            # Preflight checks if control provided
            if self.control and kwargs.get('enable_preflight', True):
                try:
                    preflight = await preflight_cost_check(self.control)
                    results["preflight_assessment"] = preflight

                    if preflight['overall_readiness'] == 'blocked':
                        raise RuntimeError("Preflight checks blocked execution")

                except Exception as e:
                    self.logger.error(f"Preflight check failed: {e}")
                    if kwargs.get('strict_preflight', True):
                        raise

            for target in targets:
                if not self.validate_target(target):
                    self.logger.warning(f"Invalid target: {target}")
                    results["errors"] += 1
                    results["results"].append({
                        "target": target,
                        "error": "Invalid target",
                        "success": False
                    })
                    continue

                try:
                    self.logger.info(f"Enhanced scraping target: {target}")

                    # Use enhanced scraping with monitoring
                    scrape_result = await self._execute_scrape_with_monitoring(target, **kwargs)

                    result_entry = scrape_result.to_dict()
                    results["results"].append(result_entry)

                    results["targets_processed"] += 1
                    results["data_collected"] += len(scrape_result.data)
                    results["pages_scraped"] += self.pages_scraped
                    results["requests_made"] += self.requests_made

                    # Update intelligence summary
                    if scrape_result.cost_estimate:
                        results["intelligence_summary"]["cost_tracking"][target] = scrape_result.cost_estimate

                except Exception as e:
                    self.logger.error(f"Enhanced scraping failed for {target}: {e}")
                    results["results"].append({
                        "target": target,
                        "error": str(e),
                        "success": False
                    })
                    results["errors"] += 1

        finally:
            self.is_running = False
            end_time = time.time()
            results["end_time"] = datetime.fromtimestamp(end_time).isoformat()
            results["duration_seconds"] = end_time - self.start_time

            # Calculate performance metrics
            results["intelligence_summary"]["performance_metrics"] = self._calculate_performance_metrics(results)

        self.logger.info(f"Enhanced scraping completed: {results['data_collected']} items from {results['targets_processed']} targets")
        return results

    def _calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {
            "success_rate": 0.0,
            "error_rate": 0.0,
            "avg_cost_per_target": 0.0,
            "avg_records_per_target": 0.0,
            "throughput_records_per_second": 0.0,
            "cost_efficiency": 0.0
        }

        if results["targets_processed"] > 0:
            metrics["success_rate"] = (results["targets_processed"] - results["errors"]) / len(results["results"])
            metrics["error_rate"] = results["errors"] / len(results["results"])
            metrics["avg_records_per_target"] = results["data_collected"] / results["targets_processed"]

            # Cost efficiency
            total_cost = sum(results["intelligence_summary"]["cost_tracking"].values())
            if total_cost > 0:
                metrics["avg_cost_per_target"] = total_cost / len(results["intelligence_summary"]["cost_tracking"])
                metrics["cost_efficiency"] = results["data_collected"] / total_cost

            # Throughput
            duration = results["duration_seconds"]
            if duration > 0:
                metrics["throughput_records_per_second"] = results["data_collected"] / duration

        return metrics

    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get comprehensive scraper status with intelligence data."""
        base_status = {
            "name": self.config.name,
            "role": self.ROLE.value,
            "tier": self.TIER.value,
            "is_running": self.is_running,
            "is_initialized": self.is_initialized,
            "supported_signals": [s.value for s in self.SUPPORTED_SIGNALS],
            "supported_assets": [a.value for a in self.SUPPORTED_ASSETS],
            "supported_events": self.SUPPORTED_EVENTS
        }

        # Performance metrics
        if hasattr(self, 'start_time') and self.start_time:
            base_status["runtime_seconds"] = time.time() - self.start_time
        else:
            base_status["runtime_seconds"] = 0

        base_status.update({
            "pages_scraped": self.pages_scraped,
            "records_collected": self.records_collected,
            "requests_made": self.requests_made,
            "errors_encountered": self.errors_encountered,
            "intelligence_data": {
                "cost_tracking_count": len(self.cost_tracking),
                "risk_assessment_count": len(self.risk_assessment),
                "intelligence_assessment_count": len(self.intelligence_assessment)
            }
        })

        return base_status

    async def initialize_intelligence(self) -> None:
        """Initialize intelligence systems and profiles."""
        if not self.control:
            return

        try:
            # Get execution profile
            from core.execution_mode_classifier import classify_execution_mode
            self.execution_profile = await classify_execution_mode(
                asset_type=self._infer_asset_type(),
                scope_size=10,  # Default scope estimate
                control=self.control
            )

            # Initialize intelligence assessment
            self.intelligence_assessment = {
                "execution_mode": self.execution_profile.mode.value if self.execution_profile else None,
                "risk_level": "unknown",
                "cost_efficiency": "unknown",
                "initialized_at": datetime.utcnow().isoformat()
            }

            self.logger.info(f"Intelligence initialized for {self.config.name}")

        except Exception as e:
            self.logger.warning(f"Intelligence initialization failed: {e}")

    async def cleanup_enhanced(self) -> None:
        """Enhanced cleanup with intelligence data preservation."""
        self.logger.info(f"Enhanced cleanup for {self.config.name}")

        # Preserve intelligence data for analysis
        cleanup_summary = {
            "scraper": self.config.name,
            "cleanup_time": datetime.utcnow().isoformat(),
            "final_metrics": self._calculate_performance_metrics({
                "targets_processed": self.requests_made,
                "data_collected": self.records_collected,
                "errors": self.errors_encountered,
                "intelligence_summary": {"cost_tracking": self.cost_tracking}
            })
        }

        # Reset operational state
        self.is_running = False
        self.is_initialized = False
        self.start_time = None
        self.pages_scraped = 0
        self.records_collected = 0
        self.requests_made = 0
        self.errors_encountered = 0

        # Preserve intelligence data (don't reset)
        # self.intelligence_assessment = {}
        # self.cost_tracking = {}
        # self.risk_assessment = {}

        self.logger.info(f"Enhanced cleanup completed: {cleanup_summary}")

    # Backward compatibility methods
    async def run(self, targets: List[str], **kwargs) -> Dict[str, Any]:
        """Backward compatible run method."""
        return await self.run_enhanced(targets, **kwargs)

    def get_status(self) -> Dict[str, Any]:
        """Backward compatible status method."""
        return self.get_enhanced_status()

    async def cleanup(self) -> None:
        """Backward compatible cleanup method."""
        return await self.cleanup_enhanced()