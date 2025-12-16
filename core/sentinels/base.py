# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Sentinel Base Classes for MJ Data Scraper Suite

Provides monitoring, alerting, and automated response systems for scraping operations.
Sentinels probe targets and return structured reports with risk assessments and recommendations.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, Union, Callable, List
from collections import defaultdict, deque
from dataclasses import dataclass, field

try:
    from pydantic import BaseModel
except ImportError:
    # Fallback for environments without pydantic
    class BaseModel:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

        def dict(self):
            return self.__dict__

from ..scrape_telemetry import get_global_telemetry_collector, emit_telemetry
from ..retry_utils import retry_async, RetryConfig, retry_on_network_errors
from ..control_models import DeploymentWindow

logger = logging.getLogger(__name__)


@dataclass
class SentinelAuditEntry:
    """Audit entry for sentinel probe operations."""
    sentinel_name: str
    target: Dict[str, Any]
    probe_time: datetime
    result: SentinelReport
    execution_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sentinel_name": self.sentinel_name,
            "target": self.target,
            "probe_time": self.probe_time.isoformat(),
            "result": self.result.dict(),
            "execution_context": self.execution_context
        }


@dataclass
class SentinelMetrics:
    """Comprehensive metrics for sentinel operations."""
    probes_attempted: int = 0
    probes_succeeded: int = 0
    probes_failed: int = 0
    total_response_time: float = 0.0
    risk_distribution: Dict[str, int] = field(default_factory=lambda: {"low": 0, "medium": 0, "high": 0, "critical": 0})
    action_distribution: Dict[str, int] = field(default_factory=lambda: {"allow": 0, "delay": 0, "restrict": 0, "block": 0})

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.probes_succeeded + self.probes_failed
        return self.probes_succeeded / max(1, total)

    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        return self.total_response_time / max(1, self.probes_attempted)

    def update_from_report(self, report: SentinelReport) -> None:
        """Update metrics from a sentinel report."""
        self.probes_attempted += 1
        if report.success:
            self.probes_succeeded += 1
        else:
            self.probes_failed += 1

        self.total_response_time += report.response_time
        self.risk_distribution[report.risk_level] = self.risk_distribution.get(report.risk_level, 0) + 1
        self.action_distribution[report.recommended_action] = self.action_distribution.get(report.recommended_action, 0) + 1


class SentinelReport(BaseModel):
    """Structured report from sentinel probe operations."""
    sentinel_name: str
    domain: str  # e.g., "performance", "security", "compliance", "network"
    timestamp: datetime
    risk_level: str  # "low" | "medium" | "high" | "critical"
    findings: Dict[str, Any]
    recommended_action: str  # "allow" | "restrict" | "delay" | "block"
    request_id: str = ""  # Unique request identifier
    response_time: float = 0.0  # Time taken for probe
    retry_count: int = 0  # Number of retries attempted
    success: bool = True  # Whether probe completed successfully
    error_message: Optional[str] = None  # Error details if probe failed

    async def emit_telemetry_async(self) -> None:
        """Emit telemetry data for this sentinel probe."""
        try:
            await emit_telemetry(
                scraper=self.sentinel_name,
                role=self.domain,
                cost_estimate=self.response_time * 0.001,  # Rough cost estimate
                records_found=len(self.findings) if self.findings else 0,
                runtime=self.response_time,
                blocked_reason=self.recommended_action if self.recommended_action in ["restrict", "block"] else ""
            )
        except Exception as e:
            logger.warning(f"Failed to emit telemetry for sentinel {self.sentinel_name}: {e}")

    def emit_telemetry(self) -> None:
        """Synchronous wrapper for telemetry emission."""
        # For synchronous contexts, we create a task to emit telemetry
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a task
                loop.create_task(self.emit_telemetry_async())
            else:
                # If no loop is running, run it synchronously
                loop.run_until_complete(self.emit_telemetry_async())
        except Exception as e:
            logger.warning(f"Failed to emit telemetry for sentinel {self.sentinel_name}: {e}")


class BaseSentinel(ABC):
    """Base class for all sentinels in the MJ Data Scraper Suite with enterprise features."""

    name: str

    def __init__(self):
        # Enhanced metrics tracking (inspired by scraper engine)
        self.metrics = SentinelMetrics()
        self.start_time = time.time()
        self.last_probe_time = 0.0

        # Audit logging (inspired by authorization module)
        self.audit_log: List[SentinelAuditEntry] = []
        self.max_audit_entries = 1000  # Limit audit log size

        # Callbacks (similar to BaseScraper)
        self.on_error: Optional[Callable[[Exception, Dict[str, Any]], None]] = None
        self.on_success: Optional[Callable[[SentinelReport], None]] = None
        self.on_retry: Optional[Callable[[int, Exception], None]] = None
        self.on_audit: Optional[Callable[[SentinelAuditEntry], None]] = None

        # Retry configuration (enhanced)
        self.retry_config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=10.0,
            backoff_factor=2.0,
            retry_on_exceptions=(ConnectionError, TimeoutError, OSError, asyncio.TimeoutError),
            retry_on_status_codes=(429, 500, 502, 503, 504)
        )

        # Time-based execution controls (inspired by deployment timer)
        self.execution_window: Optional[DeploymentWindow] = None
        self.timezone_aware = True

    @abstractmethod
    async def probe(self, target: Dict[str, Any]) -> SentinelReport:
        """
        Probe a target and return a structured sentinel report.

        Args:
            target: Target information to probe (URL, domain, operation context, etc.)

        Returns:
            SentinelReport with findings, risk assessment, and recommendations
        """
        pass

    async def probe_with_retry(self, target: Dict[str, Any]) -> SentinelReport:
        """
        Probe with automatic retry logic, time-based controls, and comprehensive audit logging.

        Args:
            target: Target information to probe

        Returns:
            SentinelReport with retry tracking and error details if needed
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        retry_count = 0

        # Check execution window (inspired by deployment timer)
        if self.execution_window:
            current_time = datetime.utcnow()
            if not self.execution_window.is_within_window(current_time):
                logger.warning(f"Sentinel {self.name} execution outside allowed window")
                # Could delay until window opens, but for now just log

        # Implement retry logic with enhanced error handling
        max_attempts = self.retry_config.max_attempts
        base_delay = self.retry_config.base_delay

        for attempt in range(max_attempts):
            try:
                self.metrics.probes_attempted += 1
                self.last_probe_time = start_time

                # Execute the probe
                report = await self.probe(target)

                # Update metrics
                self.metrics.probes_succeeded += 1
                response_time = time.time() - start_time

                # Enhance report with additional metadata
                report.request_id = request_id
                report.response_time = response_time
                report.retry_count = retry_count
                report.success = True

                # Update metrics from successful report
                self.metrics.update_from_report(report)

                # Create audit entry (inspired by authorization module)
                audit_entry = SentinelAuditEntry(
                    sentinel_name=self.name,
                    target=target,
                    probe_time=datetime.utcnow(),
                    result=report,
                    execution_context={
                        "attempt": attempt + 1,
                        "response_time": response_time,
                        "retry_count": retry_count
                    }
                )

                # Add to audit log with size limit
                self.audit_log.append(audit_entry)
                if len(self.audit_log) > self.max_audit_entries:
                    self.audit_log.pop(0)  # Remove oldest

                # Emit telemetry
                report.emit_telemetry()

                # Call success callback
                if self.on_success:
                    try:
                        self.on_success(report)
                    except Exception as e:
                        logger.warning(f"Success callback failed for sentinel {self.name}: {e}")

                # Call audit callback
                if self.on_audit:
                    try:
                        self.on_audit(audit_entry)
                    except Exception as e:
                        logger.warning(f"Audit callback failed for sentinel {self.name}: {e}")

                return report

            except Exception as e:
                retry_count = attempt
                response_time = time.time() - start_time

                # Check if we should retry
                if attempt < max_attempts - 1 and self._should_retry(e):
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Sentinel {self.name} probe attempt {attempt + 1} failed, retrying in {delay}s: {e}")

                    # Call retry callback
                    if self.on_retry:
                        try:
                            self.on_retry(attempt + 1, e)
                        except Exception as callback_error:
                            logger.warning(f"Retry callback failed for sentinel {self.name}: {callback_error}")

                    await asyncio.sleep(delay)
                    continue
                else:
                    # Final failure
                    self.metrics.probes_failed += 1
                    logger.error(f"Sentinel {self.name} probe failed after {attempt + 1} attempts for target {target}: {e}")

                    # Create error report
                    error_report = SentinelReport(
                        sentinel_name=self.name,
                        domain="error",
                        timestamp=datetime.utcnow(),
                        risk_level="critical",
                        findings={"error": str(e), "target": target, "error_type": type(e).__name__},
                        recommended_action="block",
                        request_id=request_id,
                        response_time=response_time,
                        retry_count=retry_count,
                        success=False,
                        error_message=str(e)
                    )

                    # Create audit entry for failed probe
                    audit_entry = SentinelAuditEntry(
                        sentinel_name=self.name,
                        target=target,
                        probe_time=datetime.utcnow(),
                        result=error_report,
                        execution_context={
                            "attempt": attempt + 1,
                            "response_time": response_time,
                            "retry_count": retry_count,
                            "final_error": str(e)
                        }
                    )

                    # Add to audit log
                    self.audit_log.append(audit_entry)
                    if len(self.audit_log) > self.max_audit_entries:
                        self.audit_log.pop(0)

                    # Call error callback
                    if self.on_error:
                        try:
                            self.on_error(e, target)
                        except Exception as callback_error:
                            logger.warning(f"Error callback failed for sentinel {self.name}: {callback_error}")

                    # Call audit callback for failed probes too
                    if self.on_audit:
                        try:
                            self.on_audit(audit_entry)
                        except Exception as audit_error:
                            logger.warning(f"Audit callback failed for sentinel {self.name}: {audit_error}")

                    return error_report

    def _should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry."""
        exception_types = tuple(self.retry_config.retry_on_exceptions)
        return isinstance(exception, exception_types)

    def set_execution_window(self, window: DeploymentWindow) -> None:
        """Set execution window for time-based controls."""
        self.execution_window = window
        logger.info(f"Execution window set for sentinel {self.name}")

    def clear_execution_window(self) -> None:
        """Clear execution window restrictions."""
        self.execution_window = None
        logger.info(f"Execution window cleared for sentinel {self.name}")

    def get_audit_log(self, limit: Optional[int] = None) -> List[SentinelAuditEntry]:
        """Get audit log entries."""
        if limit:
            return self.audit_log[-limit:]
        return self.audit_log.copy()

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self.audit_log.clear()
        logger.info(f"Audit log cleared for sentinel {self.name}")

    def reset_metrics(self) -> None:
        """Reset sentinel metrics."""
        self.metrics = SentinelMetrics()
        self.start_time = time.time()
        logger.info(f"Metrics reset for sentinel {self.name}")

    async def probe_with_fallback(self, target: Dict[str, Any]) -> SentinelReport:
        """
        Legacy method for backward compatibility.
        Use probe_with_retry for enhanced functionality.
        """
        return await self.probe_with_retry(target)

    def get_sentinel_info(self) -> Dict[str, Any]:
        """Get comprehensive information about this sentinel (enhanced with enterprise features)."""
        uptime = time.time() - self.start_time
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
            "metrics": {
                "probes_attempted": self.metrics.probes_attempted,
                "probes_succeeded": self.metrics.probes_succeeded,
                "probes_failed": self.metrics.probes_failed,
                "success_rate": self.metrics.success_rate,
                "average_response_time": self.metrics.average_response_time,
                "risk_distribution": self.metrics.risk_distribution,
                "action_distribution": self.metrics.action_distribution,
                "uptime_seconds": uptime,
                "last_probe_time": self.last_probe_time
            },
            "audit": {
                "total_entries": len(self.audit_log),
                "max_entries": self.max_audit_entries,
                "recent_entries": [entry.to_dict() for entry in self.audit_log[-5:]]  # Last 5 entries
            },
            "configuration": {
                "retry_config": {
                    "max_attempts": self.retry_config.max_attempts,
                    "base_delay": self.retry_config.base_delay,
                    "max_delay": self.retry_config.max_delay,
                    "exceptions": [e.__name__ for e in self.retry_config.retry_on_exceptions],
                    "status_codes": list(self.retry_config.retry_on_status_codes)
                },
                "execution_window": self.execution_window.dict() if self.execution_window else None,
                "timezone_aware": self.timezone_aware
            }
        }


class SentinelSeverity:
    """Severity levels for sentinel alerts."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SentinelStatus(Enum):
    """Status of sentinel monitoring."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ALERTING = "alerting"
    MAINTENANCE = "maintenance"


class SentinelAlert:
    """Alert generated by a sentinel."""
    def __init__(self, sentinel_name: str, severity: Union[SentinelSeverity, str], message: str,
                 details: Optional[Dict[str, Any]] = None, timestamp: Optional[datetime] = None,
                 resolved: bool = False, resolved_at: Optional[datetime] = None,
                 alert_id: Optional[str] = None):
        self.sentinel_name = sentinel_name
        # Convert string severity to enum if needed
        if isinstance(severity, str):
            severity_map = {
                "info": SentinelSeverity.INFO,
                "warning": SentinelSeverity.WARNING,
                "error": SentinelSeverity.ERROR,
                "critical": SentinelSeverity.CRITICAL
            }
            self.severity = severity_map.get(severity.lower(), SentinelSeverity.WARNING)
        else:
            self.severity = severity
        self.message = message
        self.details = details or {}
        self.timestamp = timestamp or datetime.utcnow()
        self.resolved = resolved
        self.resolved_at = resolved_at
        self.alert_id = alert_id or f"alert_{int(time.time() * 1000)}"

    def resolve(self) -> None:
        """Mark the alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.utcnow()
        logger.info(f"Alert {self.alert_id} resolved: {self.message}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "sentinel_name": self.sentinel_name,
            "severity": self.severity.name if hasattr(self.severity, 'name') else str(self.severity),
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


class SentinelConfig:
    """Configuration for sentinel instances."""
    def __init__(self, name: str, enabled: bool = True, check_interval: float = 60.0,
                 alert_threshold: int = 3, cooldown_period: float = 300.0,
                 severity: SentinelSeverity = SentinelSeverity.WARNING,
                 auto_resolve: bool = True, alert_channels: Optional[list] = None):
        self.name = name
        self.enabled = enabled
        self.check_interval = check_interval
        self.alert_threshold = alert_threshold
        self.cooldown_period = cooldown_period
        self.severity = severity
        self.auto_resolve = auto_resolve
        self.alert_channels = alert_channels or ["log"]

    def validate(self) -> None:
        """Validate sentinel configuration."""
        if self.check_interval <= 0:
            raise ValueError("check_interval must be positive")
        if self.alert_threshold < 1:
            raise ValueError("alert_threshold must be at least 1")
        if self.cooldown_period < 0:
            raise ValueError("cooldown_period cannot be negative")


class SentinelRunner:
    """
    Runner for sentinel probes with monitoring and alerting capabilities.

    Wraps BaseSentinel instances to provide monitoring infrastructure,
    alert generation, and automated response handling.
    """

    def __init__(self, sentinel: BaseSentinel, config: Optional[SentinelConfig] = None):
        self.sentinel = sentinel
        self.config = config or SentinelConfig(name=f"{sentinel.name}_runner")

        self.status = SentinelStatus.ACTIVE
        self.last_check_time = 0.0
        self.consecutive_failures = 0
        self.last_alert_time = 0.0
        self.active_alerts: List[SentinelAlert] = []
        self.alert_history: deque[SentinelAlert] = deque(maxlen=1000)
        self.last_report: Optional[SentinelReport] = None

        # Enhanced audit logging (inspired by authorization module)
        self.audit_log: List[SentinelAuditEntry] = []
        self.max_audit_entries = 500  # Smaller than individual sentinels

        # Callbacks
        self.on_alert: Optional[Callable[[SentinelAlert], None]] = None
        self.on_resolve: Optional[Callable[[SentinelAlert], None]] = None
        self.on_status_change: Optional[Callable[[SentinelStatus, SentinelStatus], None]] = None
        self.on_report: Optional[Callable[[SentinelReport], None]] = None
        self.on_audit: Optional[Callable[[SentinelAuditEntry], None]] = None

        # Telemetry integration
        self.telemetry_collector = get_global_telemetry_collector()

        logger.info(f"SentinelRunner for {sentinel.name} initialized")

    async def probe_target(self, target: Dict[str, Any]) -> SentinelReport:
        """
        Execute a probe with retry logic, audit logging, and result handling.

        Args:
            target: Target to probe

        Returns:
            SentinelReport from the probe with enhanced metadata
        """
        report = await self.sentinel.probe_with_retry(target)
        self.last_report = report

        # Create audit entry for runner-level tracking
        audit_entry = SentinelAuditEntry(
            sentinel_name=f"{self.sentinel.name}_runner",
            target=target,
            probe_time=datetime.utcnow(),
            result=report,
            execution_context={
                "runner": self.config.name,
                "status": self.status.value
            }
        )

        # Add to runner audit log
        self.audit_log.append(audit_entry)
        if len(self.audit_log) > self.max_audit_entries:
            self.audit_log.pop(0)

        # Call audit callback
        if self.on_audit:
            try:
                self.on_audit(audit_entry)
            except Exception as e:
                logger.error(f"Runner audit callback failed for {self.sentinel.name}: {e}")

        # Handle report based on risk level and recommended action
        await self._handle_report(report, target)

        # Call report callback
        if self.on_report:
            try:
                self.on_report(report)
            except Exception as e:
                logger.error(f"Report callback failed for {self.sentinel.name}: {e}")

        return report

    async def _handle_report(self, report: SentinelReport, target: Dict[str, Any]):
        """Handle sentinel report and take appropriate actions."""
        # Check if this constitutes an alert condition
        should_alert = self._should_alert_on_report(report)

        if should_alert:
            await self._generate_report_alert(report, target)
        else:
            # Clear any existing alerts if risk is now acceptable
            if self.active_alerts and report.risk_level in ["low", "medium"]:
                await self._resolve_active_alerts()

    def _should_alert_on_report(self, report: SentinelReport) -> bool:
        """Determine if a report should trigger an alert."""
        # Alert on high-risk or critical findings
        if report.risk_level in ["high", "critical"]:
            return True

        # Alert on restrictive recommendations
        if report.recommended_action in ["restrict", "delay", "block"]:
            return True

        # Could add more sophisticated logic here
        return False

    async def _generate_report_alert(self, report: SentinelReport, target: Dict[str, Any]):
        """Generate an alert based on sentinel report."""
        # Map risk level to severity
        severity_map = {
            "low": SentinelSeverity.INFO,
            "medium": SentinelSeverity.WARNING,
            "high": SentinelSeverity.ERROR,
            "critical": SentinelSeverity.CRITICAL
        }

        severity = severity_map.get(report.risk_level, SentinelSeverity.WARNING)

        alert = SentinelAlert(
            sentinel_name=self.sentinel.name,
            severity=severity,
            message=f"Risk assessment: {report.risk_level} - {report.recommended_action}",
            details={
                "report": report.dict(),
                "target": target,
                "domain": report.domain
            }
        )

        self.active_alerts.append(alert)
        self.alert_history.append(alert)

        # Update status if this is our first active alert
        if len(self.active_alerts) == 1:
            await self._change_status(SentinelStatus.ALERTING)

        # Call alert callback
        if self.on_alert:
            try:
                self.on_alert(alert)
            except Exception as e:
                logger.error(f"Alert callback failed for {self.sentinel.name}: {e}")

        # Send alert to configured channels
        await self._send_alert_to_channels(alert)

        severity_str = alert.severity.name if hasattr(alert.severity, 'name') else str(alert.severity)
        logger.warning(f"Sentinel {self.sentinel.name} generated {severity_str} alert: {alert.message}")

    async def _resolve_active_alerts(self):
        """Resolve all active alerts."""
        resolved_alerts = []
        for alert in self.active_alerts:
            if not alert.resolved:
                alert.resolve()
                resolved_alerts.append(alert)

                if self.on_resolve:
                    try:
                        self.on_resolve(alert)
                    except Exception as e:
                        logger.error(f"Resolve callback failed for {self.sentinel.name}: {e}")

        self.active_alerts.clear()

        if resolved_alerts and self.status == SentinelStatus.ALERTING:
            await self._change_status(SentinelStatus.ACTIVE)

        if resolved_alerts:
            logger.info(f"Sentinel {self.sentinel.name} resolved {len(resolved_alerts)} alerts")

    async def _change_status(self, new_status: SentinelStatus):
        """Change sentinel status and notify callbacks."""
        old_status = self.status
        self.status = new_status

        if self.on_status_change and old_status != new_status:
            try:
                self.on_status_change(old_status, new_status)
            except Exception as e:
                logger.error(f"Status change callback failed for {self.sentinel.name}: {e}")

        logger.info(f"Sentinel {self.sentinel.name} status changed: {old_status.value} -> {new_status.value}")

    async def _send_alert_to_channels(self, alert: SentinelAlert):
        """Send alert to configured channels."""
        for channel in self.config.alert_channels:
            try:
                if channel == "log":
                    # Already logged above
                    pass
                elif channel == "email":
                    await self._send_email_alert(alert)
                elif channel == "slack":
                    await self._send_slack_alert(alert)
                elif channel == "webhook":
                    await self._send_webhook_alert(alert)
                else:
                    logger.warning(f"Unknown alert channel: {channel}")
            except Exception as e:
                logger.error(f"Failed to send alert to {channel}: {e}")

    async def _send_email_alert(self, alert: SentinelAlert):
        """Send alert via email."""
        logger.debug(f"Would send email alert: {alert.message}")

    async def _send_slack_alert(self, alert: SentinelAlert):
        """Send alert via Slack."""
        logger.debug(f"Would send Slack alert: {alert.message}")

    async def _send_webhook_alert(self, alert: SentinelAlert):
        """Send alert via webhook."""
        logger.debug(f"Would send webhook alert: {alert.message}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive sentinel runner metrics."""
        return {
            "sentinel_name": self.sentinel.name,
            "sentinel_type": self.sentinel.__class__.__name__,
            "status": self.status.value,
            "active_alerts": len(self.active_alerts),
            "total_alerts": len(self.alert_history),
            "audit_entries": len(self.audit_log),
            "last_check_time": self.last_check_time,
            "last_alert_time": self.last_alert_time,
            "last_report": self.last_report.dict() if self.last_report else None,
            # Include sentinel metrics
            "sentinel_metrics": self.sentinel.get_sentinel_info()["metrics"]
        }

    def get_audit_log(self, limit: Optional[int] = None) -> List[SentinelAuditEntry]:
        """Get runner audit log entries."""
        if limit:
            return self.audit_log[-limit:]
        return self.audit_log.copy()

    def clear_audit_log(self) -> None:
        """Clear the runner audit log."""
        self.audit_log.clear()
        logger.info(f"Runner audit log cleared for {self.sentinel.name}")

    def reset_metrics(self) -> None:
        """Reset runner metrics and underlying sentinel metrics."""
        self.active_alerts.clear()
        self.alert_history.clear()
        self.audit_log.clear()
        self.last_check_time = 0.0
        self.last_alert_time = 0.0
        self.last_report = None
        self.consecutive_failures = 0

        # Reset underlying sentinel metrics
        self.sentinel.reset_metrics()

        logger.info(f"Runner metrics reset for {self.sentinel.name}")

    def get_active_alerts(self) -> List[SentinelAlert]:
        """Get list of currently active alerts."""
        return self.active_alerts.copy()

    def get_last_report(self) -> Optional[SentinelReport]:
        """Get the last sentinel report."""
        return self.last_report

    async def execute_check(self) -> None:
        """
        Execute a single check cycle.

        This method handles the check execution, alert generation, and state management.
        """
        if self.status != SentinelStatus.ACTIVE:
            return

        try:
            current_time = time.time()

            # Check if it's time for another check
            if current_time - self.last_check_time < self.config.check_interval:
                return

            self.last_check_time = current_time

            # Execute the specific condition check
            is_healthy, message, details = await self.check_condition()

            if is_healthy:
                # Reset consecutive failures on success
                if self.consecutive_failures > 0:
                    logger.info(f"Sentinel {self.config.name} recovered: {message}")
                    await self._resolve_active_alerts()
                self.consecutive_failures = 0
            else:
                # Increment consecutive failures
                self.consecutive_failures += 1
                logger.warning(f"Sentinel {self.config.name} condition failed ({self.consecutive_failures}/{self.config.alert_threshold}): {message}")

                # Check if we should alert
                if (self.consecutive_failures >= self.config.alert_threshold and
                    current_time - self.last_alert_time >= self.config.cooldown_period):
                    await self._generate_alert(message, details)
                    self.last_alert_time = current_time

        except Exception as e:
            logger.error(f"Sentinel {self.config.name} check failed with exception: {e}")
            self.consecutive_failures += 1

            if self.consecutive_failures >= self.config.alert_threshold:
                await self._generate_alert(f"Check execution failed: {str(e)}", {"exception": str(e)})

    async def _generate_alert(self, message: str, details: Dict[str, Any]) -> None:
        """Generate and handle a new alert."""
        alert = SentinelAlert(
            sentinel_name=self.config.name,
            severity=self.config.severity,
            message=message,
            details=details
        )

        self.active_alerts.append(alert)
        self.alert_history.append(alert)

        # Update status if this is our first active alert
        if len(self.active_alerts) == 1:
            await self._change_status(SentinelStatus.ALERTING)

        # Call alert callback
        if self.on_alert:
            try:
                self.on_alert(alert)
            except Exception as e:
                logger.error(f"Alert callback failed for {self.config.name}: {e}")

        # Send alert to configured channels
        await self._send_alert_to_channels(alert)

        logger.warning(f"Sentinel {self.config.name} generated {alert.severity.value} alert: {alert.message}")

    async def _resolve_active_alerts(self) -> None:
        """Resolve all active alerts if auto-resolve is enabled."""
        if not self.config.auto_resolve:
            return

        resolved_alerts = []
        for alert in self.active_alerts:
            if not alert.resolved:
                alert.resolve()
                resolved_alerts.append(alert)

                if self.on_resolve:
                    try:
                        self.on_resolve(alert)
                    except Exception as e:
                        logger.error(f"Resolve callback failed for {self.config.name}: {e}")

        self.active_alerts.clear()

        if resolved_alerts and self.status == SentinelStatus.ALERTING:
            await self._change_status(SentinelStatus.ACTIVE)

        if resolved_alerts:
            logger.info(f"Sentinel {self.config.name} resolved {len(resolved_alerts)} alerts")

    async def _change_status(self, new_status: SentinelStatus) -> None:
        """Change sentinel status and notify callbacks."""
        old_status = self.status
        self.status = new_status

        if self.on_status_change and old_status != new_status:
            try:
                self.on_status_change(old_status, new_status)
            except Exception as e:
                logger.error(f"Status change callback failed for {self.config.name}: {e}")

        logger.info(f"Sentinel {self.config.name} status changed: {old_status.value} -> {new_status.value}")

    async def _send_alert_to_channels(self, alert: SentinelAlert) -> None:
        """Send alert to configured channels."""
        for channel in self.config.alert_channels:
            try:
                if channel == "log":
                    # Already logged above
                    pass
                elif channel == "email":
                    await self._send_email_alert(alert)
                elif channel == "slack":
                    await self._send_slack_alert(alert)
                elif channel == "webhook":
                    await self._send_webhook_alert(alert)
                else:
                    logger.warning(f"Unknown alert channel: {channel}")
            except Exception as e:
                logger.error(f"Failed to send alert to {channel}: {e}")

    async def _send_email_alert(self, alert: SentinelAlert) -> None:
        """Send alert via email (placeholder for implementation)."""
        # TODO: Implement email integration
        logger.debug(f"Would send email alert: {alert.message}")

    async def _send_slack_alert(self, alert: SentinelAlert) -> None:
        """Send alert via Slack (placeholder for implementation)."""
        # TODO: Implement Slack integration
        logger.debug(f"Would send Slack alert: {alert.message}")

    async def _send_webhook_alert(self, alert: SentinelAlert) -> None:
        """Send alert via webhook (placeholder for implementation)."""
        # TODO: Implement webhook integration
        logger.debug(f"Would send webhook alert: {alert.message}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get sentinel performance metrics."""
        return {
            "sentinel_name": self.config.name,
            "sentinel_type": self.__class__.__name__,
            "status": self.status.value,
            "check_interval": self.config.check_interval,
            "consecutive_failures": self.consecutive_failures,
            "active_alerts": len(self.active_alerts),
            "total_alerts": len(self.alert_history),
            "last_check_time": self.last_check_time,
            "last_alert_time": self.last_alert_time,
            "severity": self.config.severity
        }

    def get_active_alerts(self) -> List[SentinelAlert]:
        """Get list of currently active alerts."""
        return self.active_alerts.copy()

    def get_alert_history(self, limit: Optional[int] = None) -> List[SentinelAlert]:
        """Get alert history, optionally limited to most recent."""
        alerts = list(self.alert_history)
        if limit:
            alerts = alerts[-limit:]
        return alerts

    def reset_metrics(self) -> None:
        """Reset sentinel metrics (useful for testing)."""
        self.consecutive_failures = 0
        self.last_alert_time = 0.0
        self.active_alerts.clear()
        logger.info(f"Metrics reset for sentinel {self.config.name}")

    async def start_monitoring(self) -> None:
        """Start the sentinel monitoring loop."""
        self.status = SentinelStatus.ACTIVE
        logger.info(f"Started monitoring for sentinel {self.config.name}")

        while self.status == SentinelStatus.ACTIVE:
            try:
                await self.execute_check()
                await asyncio.sleep(min(self.config.check_interval, 1.0))  # Don't sleep too long
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error for {self.config.name}: {e}")
                await asyncio.sleep(5.0)  # Brief pause on error

        logger.info(f"Stopped monitoring for sentinel {self.config.name}")

    def stop_monitoring(self) -> None:
        """Stop the sentinel monitoring."""
        self.status = SentinelStatus.INACTIVE
        logger.info(f"Monitoring stopped for sentinel {self.config.name}")

    def set_maintenance_mode(self) -> None:
        """Put sentinel in maintenance mode (suppresses alerts)."""
        self.status = SentinelStatus.MAINTENANCE
        logger.info(f"Maintenance mode enabled for sentinel {self.config.name}")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name}, status={self.status.value}, alerts={len(self.active_alerts)})"


# Global sentinel registry
_sentinel_registry: Dict[str, SentinelBase] = {}


def register_sentinel(sentinel: SentinelBase) -> None:
    """Register a sentinel in the global registry."""
    _sentinel_registry[sentinel.config.name] = sentinel
    logger.info(f"Registered sentinel: {sentinel.config.name}")


def unregister_sentinel(name: str) -> None:
    """Unregister a sentinel from the global registry."""
    if name in _sentinel_registry:
        del _sentinel_registry[name]
        logger.info(f"Unregistered sentinel: {name}")


def get_registered_sentinels() -> Dict[str, SentinelBase]:
    """Get all registered sentinels."""
    return _sentinel_registry.copy()


def get_sentinel(name: str) -> Optional[SentinelBase]:
    """Get a specific registered sentinel by name."""
    return _sentinel_registry.get(name)


async def start_all_sentinels() -> None:
    """Start monitoring for all registered sentinels."""
    tasks = []
    for sentinel in _sentinel_registry.values():
        if sentinel.config.enabled:
            task = asyncio.create_task(sentinel.start_monitoring())
            tasks.append(task)

    if tasks:
        logger.info(f"Started monitoring for {len(tasks)} sentinels")
        await asyncio.gather(*tasks, return_exceptions=True)
    else:
        logger.info("No sentinels to start")


def stop_all_sentinels() -> None:
    """Stop monitoring for all registered sentinels."""
    for sentinel in _sentinel_registry.values():
        sentinel.stop_monitoring()
    logger.info("Stopped all sentinels")
