# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Telemetry Models for MJ Data Scraper Suite

Comprehensive data models for telemetry events, metrics, and monitoring
across all scraper operations with enterprise-grade observability.
"""

import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
import uuid

try:
    from pydantic import BaseModel, Field, validator
except ImportError:
    # Fallback for environments without pydantic
    class BaseModel:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

        def dict(self):
            return self.__dict__

    def Field(default=None, description=None, **kwargs):
        return default

    def validator(method_name):
        def decorator(func):
            return func
        return decorator


class TelemetryEventType(Enum):
    """Types of telemetry events."""
    SCRAPER_OPERATION = "scraper_operation"
    SENTINEL_CHECK = "sentinel_check"
    SAFETY_VERDICT = "safety_verdict"
    CONSTRAINT_APPLICATION = "constraint_application"
    WORKFLOW_EXECUTION = "workflow_execution"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"
    SECURITY_EVENT = "security_event"
    AUTHORIZATION_CHECK = "authorization_check"
    AUDIT_LOG = "audit_log"


class TelemetrySeverity(Enum):
    """Severity levels for telemetry events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class BaseTelemetryEvent(BaseModel):
    """Base model for all telemetry events."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique event identifier")
    event_type: TelemetryEventType = Field(..., description="Type of telemetry event")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When event occurred")
    severity: TelemetrySeverity = Field(TelemetrySeverity.INFO, description="Event severity level")

    # Source identification
    source_component: str = Field(..., description="Component that generated the event")
    source_workflow: Optional[str] = Field(None, description="Associated workflow identifier")
    source_operation: Optional[str] = Field(None, description="Associated operation identifier")

    # Contextual information
    correlation_id: Optional[str] = Field(None, description="Correlation ID for request tracing")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier if applicable")

    # Environment context
    environment: str = Field("production", description="Deployment environment")
    version: Optional[str] = Field(None, description="Component version")
    region: Optional[str] = Field(None, description="Geographic region")

    # Custom metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")
    tags: List[str] = Field(default_factory=list, description="Event classification tags")


class ScraperOperationEvent(BaseTelemetryEvent):
    """Telemetry event for scraper operations."""

    event_type: TelemetryEventType = Field(TelemetryEventType.SCRAPER_OPERATION, const=True)

    # Operation details
    scraper_type: str = Field(..., description="Type of scraper used")
    scraper_role: Optional[str] = Field(None, description="Scraper role (discovery, verification, enrichment, browser)")
    scraper_tier: Optional[int] = Field(None, description="Scraper security tier (1-3)")

    # Target information
    target_url: Optional[str] = Field(None, description="Target URL being scraped")
    target_domain: Optional[str] = Field(None, description="Target domain")
    target_type: Optional[str] = Field(None, description="Type of target (company, person, etc.)")

    # Performance metrics
    records_found: int = Field(0, description="Number of records extracted")
    records_processed: int = Field(0, description="Number of records successfully processed")
    processing_time: float = Field(0.0, description="Total processing time in seconds")
    cost_estimate: float = Field(0.0, description="Estimated operation cost")

    # Status and outcome
    operation_status: str = Field(..., description="Operation status (success, failed, blocked, etc.)")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")
    blocked_reason: Optional[str] = Field(None, description="Reason for blocking if applicable")

    # Resource usage
    memory_used_mb: Optional[float] = Field(None, description="Memory usage in MB")
    network_requests: Optional[int] = Field(None, description="Number of network requests made")
    rate_limit_hits: Optional[int] = Field(None, description="Number of rate limit encounters")

    # Business metrics
    data_quality_score: Optional[float] = Field(None, description="Data quality assessment (0.0-1.0)")
    business_value: Optional[float] = Field(None, description="Business value generated")


class SentinelCheckEvent(BaseTelemetryEvent):
    """Telemetry event for sentinel security checks."""

    event_type: TelemetryEventType = Field(TelemetryEventType.SENTINEL_CHECK, const=True)

    # Sentinel details
    sentinel_name: str = Field(..., description="Name of sentinel that performed check")
    sentinel_type: str = Field(..., description="Type of sentinel (network, waf, malware, performance)")

    # Target information
    target_domain: Optional[str] = Field(None, description="Domain being checked")
    target_urls: List[str] = Field(default_factory=list, description="URLs being checked")

    # Check results
    risk_level: str = Field(..., description="Risk level assessment (low, medium, high, critical)")
    risk_score: float = Field(0.0, description="Numerical risk score (0.0-1.0)")
    confidence_score: float = Field(0.0, description="Confidence in assessment (0.0-1.0)")

    # Findings
    findings_count: int = Field(0, description="Number of security findings")
    critical_findings: List[str] = Field(default_factory=list, description="Critical security issues found")
    recommended_action: str = Field(..., description="Recommended action (allow, delay, restrict, block)")

    # Performance
    check_duration: float = Field(0.0, description="Time taken for check in seconds")
    probes_attempted: int = Field(0, description="Number of probes attempted")
    probes_successful: int = Field(0, description="Number of successful probes")

    # Context
    check_trigger: str = Field(..., description="What triggered this check (pre_job, continuous, post_job)")


class SafetyVerdictEvent(BaseTelemetryEvent):
    """Telemetry event for safety verdict generation."""

    event_type: TelemetryEventType = Field(TelemetryEventType.SAFETY_VERDICT, const=True)

    # Verdict details
    verdict_action: str = Field(..., description="Verdict action (allow, restrict, delay, block, human_required)")
    verdict_reason: str = Field(..., description="Reason for verdict")
    risk_level: str = Field(..., description="Overall risk level assessment")

    # Analysis details
    reports_analyzed: int = Field(0, description="Number of sentinel reports analyzed")
    sentinels_involved: List[str] = Field(default_factory=list, description="Sentinels that contributed")
    processing_duration: float = Field(0.0, description="Time to generate verdict")

    # Risk analysis
    risk_distribution: Dict[str, int] = Field(default_factory=dict, description="Risk level distribution")
    confidence_score: float = Field(0.0, description="Confidence in verdict")
    algorithm_used: str = Field(..., description="Algorithm used for decision making")

    # Constraints applied
    constraints_applied: List[str] = Field(default_factory=list, description="Constraints enforced")
    human_override_used: bool = Field(False, description="Whether human override was used")

    # Business impact
    operation_blocked: bool = Field(False, description="Whether operation was blocked")
    delay_applied: Optional[int] = Field(None, description="Delay applied in minutes")


class ConstraintApplicationEvent(BaseTelemetryEvent):
    """Telemetry event for constraint application."""

    event_type: TelemetryEventType = Field(TelemetryEventType.CONSTRAINT_APPLICATION, const=True)

    # Constraint details
    verdict_action: str = Field(..., description="Original verdict action")
    constraints_requested: List[str] = Field(default_factory=list, description="Constraints that were requested")
    constraints_applied: List[str] = Field(default_factory=list, description="Constraints successfully applied")
    constraints_failed: List[str] = Field(default_factory=list, description="Constraints that failed")

    # Application results
    application_status: str = Field(..., description="Overall application status")
    application_duration: float = Field(0.0, description="Time taken to apply constraints")

    # Specific constraint details
    delay_applied: Optional[int] = Field(None, description="Delay applied in minutes")
    rate_limit_reduction: Optional[float] = Field(None, description="Rate limit reduction factor")
    monitoring_enabled: bool = Field(False, description="Whether enhanced monitoring was enabled")
    audit_enabled: bool = Field(False, description="Whether enhanced audit was enabled")

    # Error handling
    error_messages: List[str] = Field(default_factory=list, description="Error messages from failed constraints")


class WorkflowExecutionEvent(BaseTelemetryEvent):
    """Telemetry event for workflow execution."""

    event_type: TelemetryEventType = Field(TelemetryEventType.WORKFLOW_EXECUTION, const=True)

    # Workflow details
    workflow_type: str = Field(..., description="Type of workflow (scraping, sentinel_check, etc.)")
    workflow_steps: List[str] = Field(default_factory=list, description="Steps executed in workflow")

    # Execution metrics
    total_duration: float = Field(0.0, description="Total workflow execution time")
    steps_completed: int = Field(0, description="Number of steps completed successfully")
    steps_failed: int = Field(0, description="Number of steps that failed")

    # Resource usage
    memory_peak_mb: Optional[float] = Field(None, description="Peak memory usage")
    network_requests_total: Optional[int] = Field(None, description="Total network requests")

    # Business outcomes
    records_generated: int = Field(0, description="Records generated by workflow")
    value_generated: Optional[float] = Field(None, description="Business value generated")

    # Status
    workflow_status: str = Field(..., description="Final workflow status")
    error_summary: Optional[str] = Field(None, description="Summary of errors if any")


class ErrorEvent(BaseTelemetryEvent):
    """Telemetry event for errors and exceptions."""

    event_type: TelemetryEventType = Field(TelemetryEventType.ERROR_OCCURRED, const=True)
    severity: TelemetrySeverity = Field(TelemetrySeverity.ERROR, const=True)

    # Error details
    error_type: str = Field(..., description="Type of error (exception class)")
    error_message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code if applicable")

    # Stack trace (truncated for security)
    stack_trace: Optional[str] = Field(None, description="Stack trace (sanitized)")
    error_location: Optional[str] = Field(None, description="File and line where error occurred")

    # Context
    operation_context: Optional[str] = Field(None, description="Operation context when error occurred")
    user_action: Optional[str] = Field(None, description="User action that triggered error")

    # Impact assessment
    operation_impacted: bool = Field(False, description="Whether operation was impacted")
    data_loss: bool = Field(False, description="Whether data loss occurred")
    security_impacted: bool = Field(False, description="Whether security was impacted")

    # Recovery
    recovery_attempted: bool = Field(False, description="Whether recovery was attempted")
    recovery_successful: bool = Field(False, description="Whether recovery succeeded")


class PerformanceMetricEvent(BaseTelemetryEvent):
    """Telemetry event for performance metrics."""

    event_type: TelemetryEventType = Field(TelemetryEventType.PERFORMANCE_METRIC, const=True)

    # Metric details
    metric_name: str = Field(..., description="Name of performance metric")
    metric_value: Union[int, float] = Field(..., description="Metric value")
    metric_unit: str = Field(..., description="Unit of measurement")

    # Context
    component_name: str = Field(..., description="Component being measured")
    operation_type: Optional[str] = Field(None, description="Type of operation measured")

    # Thresholds
    threshold_warning: Optional[float] = Field(None, description="Warning threshold")
    threshold_critical: Optional[float] = Field(None, description="Critical threshold")

    # Trends
    value_trend: Optional[str] = Field(None, description="Trend direction (up, down, stable)")
    baseline_value: Optional[float] = Field(None, description="Baseline value for comparison")

    # Aggregation
    aggregation_window: Optional[str] = Field(None, description="Time window for aggregation")


class SecurityEvent(BaseTelemetryEvent):
    """Telemetry event for security-related events."""

    event_type: TelemetryEventType = Field(TelemetryEventType.SECURITY_EVENT, const=True)

    # Security details
    security_event_type: str = Field(..., description="Type of security event")
    threat_level: str = Field(..., description="Threat level (low, medium, high, critical)")
    attack_vector: Optional[str] = Field(None, description="Attack vector used")

    # Target information
    target_resource: Optional[str] = Field(None, description="Resource that was targeted")
    target_ip: Optional[str] = Field(None, description="IP address involved")
    target_user: Optional[str] = Field(None, description="User account involved")

    # Incident details
    indicators_compromise: List[str] = Field(default_factory=list, description="Indicators of compromise")
    remediation_actions: List[str] = Field(default_factory=list, description="Remediation actions taken")

    # Impact
    confidentiality_impacted: bool = Field(False, description="Whether confidentiality was impacted")
    integrity_impacted: bool = Field(False, description="Whether integrity was impacted")
    availability_impacted: bool = Field(False, description="Whether availability was impacted")


class AuthorizationCheckEvent(BaseTelemetryEvent):
    """Telemetry event for authorization checks."""

    event_type: TelemetryEventType = Field(TelemetryEventType.AUTHORIZATION_CHECK, const=True)

    # Authorization details
    authorization_type: str = Field(..., description="Type of authorization checked")
    resource_requested: str = Field(..., description="Resource being requested")
    permission_required: str = Field(..., description="Permission level required")

    # Check results
    authorization_granted: bool = Field(False, description="Whether authorization was granted")
    authorization_denied_reason: Optional[str] = Field(None, description="Reason for denial")

    # Context
    user_role: Optional[str] = Field(None, description="User role making request")
    override_used: bool = Field(False, description="Whether override was used")

    # Audit
    approval_required: bool = Field(False, description="Whether approval was required")
    approval_obtained: bool = Field(False, description="Whether approval was obtained")


class AuditLogEvent(BaseTelemetryEvent):
    """Telemetry event for audit log entries."""

    event_type: TelemetryEventType = Field(TelemetryEventType.AUDIT_LOG, const=True)

    # Audit details
    audit_event_type: str = Field(..., description="Type of audit event")
    audit_action: str = Field(..., description="Action being audited")
    audit_resource: str = Field(..., description="Resource affected")

    # Change tracking
    previous_state: Optional[Dict[str, Any]] = Field(None, description="Previous state")
    new_state: Optional[Dict[str, Any]] = Field(None, description="New state")
    changes_made: List[str] = Field(default_factory=list, description="List of changes made")

    # Compliance
    compliance_requirement: Optional[str] = Field(None, description="Compliance requirement satisfied")
    retention_period: Optional[str] = Field(None, description="Audit retention period")


class SentinelOutcome(BaseModel):
    """
    Enhanced enterprise-grade sentinel outcome model.

    Comprehensive sentinel analysis results with advanced intelligence,
    risk assessment, compliance tracking, and operational insights.
    """

    # Core sentinel outcome data
    outcome_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique outcome identifier")
    domain: str = Field(..., description="Target domain analyzed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When analysis was performed")

    # Temporal context
    hour_of_day: int = Field(..., description="Hour of day (0-23) when analysis occurred")
    day_of_week: int = Field(..., description="Day of week (0=Monday, 6=Sunday)")

    # Risk assessment
    risk_level: str = Field(..., description="Risk level assessment (low, medium, high, critical)")
    risk_score: float = Field(0.0, description="Numerical risk score (0.0-1.0)")
    confidence_score: float = Field(0.0, description="Confidence in risk assessment (0.0-1.0)")

    # Action and decision
    action: str = Field(..., description="Recommended action (allow, delay, restrict, block, human_required)")
    action_reason: str = Field(..., description="Reason for recommended action")
    blocked: bool = Field(False, description="Whether operation was blocked")

    # Performance metrics
    latency_ms: int = Field(0, description="Analysis latency in milliseconds")
    processing_duration: float = Field(0.0, description="Total processing time in seconds")

    # Resource allocation
    proxy_pool: str = Field("", description="Proxy pool used for analysis")
    proxy_effectiveness: Optional[float] = Field(None, description="Proxy effectiveness score (0.0-1.0)")

    # Findings and analysis
    findings: Dict[str, Any] = Field(default_factory=dict, description="Detailed security findings")
    findings_count: int = Field(0, description="Total number of findings")
    critical_findings: List[str] = Field(default_factory=list, description="Critical security issues found")
    warning_findings: List[str] = Field(default_factory=list, description="Warning-level findings")

    # Sentinel intelligence
    sentinel_name: str = Field(..., description="Name of sentinel that performed analysis")
    sentinel_version: str = Field("", description="Version of sentinel used")
    analysis_method: str = Field(..., description="Analysis method employed")

    # Threat intelligence
    threat_indicators: List[str] = Field(default_factory=list, description="Detected threat indicators")
    threat_categories: List[str] = Field(default_factory=list, description="Threat categories identified")
    malware_signatures: List[str] = Field(default_factory=list, description="Malware signatures detected")

    # Network intelligence
    connectivity_status: str = Field("unknown", description="Network connectivity status")
    dns_resolution_time: Optional[float] = Field(None, description="DNS resolution time in seconds")
    ssl_validity_days: Optional[int] = Field(None, description="SSL certificate validity remaining in days")
    response_time_ms: Optional[int] = Field(None, description="HTTP response time in milliseconds")

    # WAF and anti-detection
    waf_detected: bool = Field(False, description="Whether WAF was detected")
    bot_protection_level: str = Field("none", description="Level of bot protection detected")
    rate_limiting_detected: bool = Field(False, description="Whether rate limiting was detected")
    session_tracking: bool = Field(False, description="Whether session tracking was detected")

    # Trend analysis
    historical_risk_trend: Optional[str] = Field(None, description="Risk trend compared to historical data")
    baseline_comparison: Optional[float] = Field(None, description="Comparison to baseline risk score")
    anomaly_score: Optional[float] = Field(None, description="Anomaly detection score")

    # Operational intelligence
    operational_recommendations: List[str] = Field(default_factory=list, description="Operational recommendations")
    alternative_strategies: List[str] = Field(default_factory=list, description="Alternative scraping strategies")
    retry_recommendations: Optional[Dict[str, Any]] = Field(None, description="Retry strategy recommendations")

    # Cost and efficiency
    estimated_cost: Optional[float] = Field(None, description="Estimated cost of proceeding")
    efficiency_score: Optional[float] = Field(None, description="Efficiency score for this approach")
    resource_intensity: str = Field("medium", description="Resource intensity assessment")

    # Compliance and governance
    compliance_flags: List[str] = Field(default_factory=list, description="Compliance flags raised")
    regulatory_requirements: List[str] = Field(default_factory=list, description="Regulatory requirements identified")
    data_residency_compliant: bool = Field(True, description="Whether analysis respects data residency")

    # Metadata and context
    correlation_id: Optional[str] = Field(None, description="Correlation ID for request tracing")
    session_id: Optional[str] = Field(None, description="Session identifier")
    workflow_id: Optional[str] = Field(None, description="Associated workflow identifier")

    # Environment context
    environment: str = Field("production", description="Deployment environment")
    region: str = Field("", description="Geographic region of analysis")
    network_segment: str = Field("", description="Network segment used")

    # Business impact
    business_impact_assessment: str = Field("low", description="Business impact assessment")
    priority_level: str = Field("normal", description="Priority level for follow-up")
    escalation_required: bool = Field(False, description="Whether escalation is required")

    # Audit trail
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list, description="Audit trail of analysis steps")
    analysis_steps: List[str] = Field(default_factory=list, description="Steps performed during analysis")
    decision_factors: Dict[str, Any] = Field(default_factory=dict, description="Factors influencing final decision")

    # Performance monitoring
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage during analysis")
    cpu_usage_percent: Optional[float] = Field(None, description="CPU usage during analysis")
    network_requests: Optional[int] = Field(None, description="Number of network requests made")

    # Quality metrics
    false_positive_probability: Optional[float] = Field(None, description="Probability of false positive")
    analysis_completeness: float = Field(1.0, description="Completeness of analysis (0.0-1.0)")

    # Future predictions
    predicted_risk_trend: Optional[str] = Field(None, description="Predicted future risk trend")
    recommended_monitoring_interval: Optional[int] = Field(None, description="Recommended monitoring interval in minutes")

    # Custom metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional analysis metadata")
    tags: List[str] = Field(default_factory=list, description="Classification tags")

    # Validation
    @validator("hour_of_day")
    def validate_hour(cls, v):
        if not (0 <= v <= 23):
            raise ValueError("hour_of_day must be between 0 and 23")
        return v

    @validator("day_of_week")
    def validate_day(cls, v):
        if not (0 <= v <= 6):
            raise ValueError("day_of_week must be between 0 (Monday) and 6 (Sunday)")
        return v

    @validator("risk_score")
    def validate_risk_score(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("risk_score must be between 0.0 and 1.0")
        return v

    @validator("confidence_score")
    def validate_confidence_score(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        return v

    @validator("proxy_effectiveness")
    def validate_proxy_effectiveness(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("proxy_effectiveness must be between 0.0 and 1.0")
        return v

    @validator("efficiency_score")
    def validate_efficiency_score(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("efficiency_score must be between 0.0 and 1.0")
        return v

    @validator("analysis_completeness")
    def validate_analysis_completeness(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("analysis_completeness must be between 0.0 and 1.0")
        return v

    def get_risk_category(self) -> str:
        """Get human-readable risk category."""
        if self.risk_score >= 0.8:
            return "critical"
        elif self.risk_score >= 0.6:
            return "high"
        elif self.risk_score >= 0.4:
            return "medium"
        elif self.risk_score >= 0.2:
            return "low"
        else:
            return "minimal"

    def get_action_priority(self) -> str:
        """Get action priority based on risk and impact."""
        if self.risk_level == "critical" or self.business_impact_assessment == "high":
            return "urgent"
        elif self.risk_level == "high" or self.business_impact_assessment == "medium":
            return "high"
        elif self.risk_level == "medium":
            return "normal"
        else:
            return "low"

    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance summary for reporting."""
        return {
            "compliant": len(self.compliance_flags) == 0,
            "flags": self.compliance_flags,
            "regulatory_requirements": self.regulatory_requirements,
            "data_residency_compliant": self.data_residency_compliant,
            "audit_trail_complete": len(self.audit_trail or []) > 0
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring."""
        return {
            "latency_ms": self.latency_ms,
            "processing_duration": self.processing_duration,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "network_requests": self.network_requests,
            "efficiency_score": self.efficiency_score
        }

    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy format for backward compatibility."""
        return {
            "domain": self.domain,
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
            "risk_level": self.risk_level,
            "action": self.action,
            "latency_ms": self.latency_ms,
            "blocked": self.blocked,
            "proxy_pool": self.proxy_pool,
            "timestamp": self.timestamp,
            "findings": self.findings
        }


# Union type for all telemetry events
TelemetryEvent = Union[
    ScraperOperationEvent,
    SentinelCheckEvent,
    SafetyVerdictEvent,
    ConstraintApplicationEvent,
    WorkflowExecutionEvent,
    ErrorEvent,
    PerformanceMetricEvent,
    SecurityEvent,
    AuthorizationCheckEvent,
    AuditLogEvent
]

# Additional outcome models
TelemetryOutcome = Union[
    SentinelOutcome
]


class TelemetryBatch(BaseModel):
    """Batch of telemetry events for efficient transmission."""

    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique batch identifier")
    events: List[TelemetryEvent] = Field(..., description="Telemetry events in batch")
    batch_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When batch was created")
    batch_size: int = Field(..., description="Number of events in batch")

    # Transmission metadata
    transmission_attempts: int = Field(0, description="Number of transmission attempts")
    last_attempt_timestamp: Optional[datetime] = Field(None, description="Last transmission attempt")
    transmission_success: bool = Field(False, description="Whether transmission succeeded")

    # Compression and optimization
    compressed: bool = Field(False, description="Whether batch is compressed")
    compression_ratio: Optional[float] = Field(None, description="Compression ratio achieved")


class TelemetryConfiguration(BaseModel):
    """Configuration for telemetry collection and transmission."""

    enabled: bool = Field(True, description="Whether telemetry is enabled")
    event_types_enabled: List[TelemetryEventType] = Field(
        default_factory=lambda: list(TelemetryEventType),
        description="Types of events to collect"
    )

    # Collection settings
    sampling_rate: float = Field(1.0, description="Sampling rate (0.0-1.0)")
    batch_size: int = Field(100, description="Events per batch")
    flush_interval_seconds: int = Field(300, description="Batch flush interval")

    # Transmission settings
    endpoint_url: Optional[str] = Field(None, description="Telemetry endpoint URL")
    api_key: Optional[str] = Field(None, description="API key for transmission")
    retry_attempts: int = Field(3, description="Number of transmission retries")
    timeout_seconds: int = Field(30, description="Transmission timeout")

    # Privacy and compliance
    pii_filtering_enabled: bool = Field(True, description="Whether PII filtering is enabled")
    retention_days: int = Field(90, description="Telemetry retention period in days")

    # Performance
    async_transmission: bool = Field(True, description="Whether to transmit asynchronously")
    compression_enabled: bool = Field(True, description="Whether to compress telemetry data")


# Utility functions for telemetry event creation
def create_scraper_operation_event(
    scraper_type: str,
    records_found: int,
    processing_time: float,
    operation_status: str,
    **kwargs
) -> ScraperOperationEvent:
    """Create a scraper operation telemetry event."""
    return ScraperOperationEvent(
        source_component="scraper_engine",
        scraper_type=scraper_type,
        records_found=records_found,
        processing_time=processing_time,
        operation_status=operation_status,
        **kwargs
    )


def create_sentinel_check_event(
    sentinel_name: str,
    risk_level: str,
    recommended_action: str,
    **kwargs
) -> SentinelCheckEvent:
    """Create a sentinel check telemetry event."""
    return SentinelCheckEvent(
        source_component="sentinel_orchestrator",
        sentinel_name=sentinel_name,
        risk_level=risk_level,
        recommended_action=recommended_action,
        **kwargs
    )


def create_safety_verdict_event(
    verdict_action: str,
    verdict_reason: str,
    risk_level: str,
    **kwargs
) -> SafetyVerdictEvent:
    """Create a safety verdict telemetry event."""
    return SafetyVerdictEvent(
        source_component="safety_verdict",
        verdict_action=verdict_action,
        verdict_reason=verdict_reason,
        risk_level=risk_level,
        **kwargs
    )


def create_error_event(
    error_type: str,
    error_message: str,
    severity: TelemetrySeverity = TelemetrySeverity.ERROR,
    **kwargs
) -> ErrorEvent:
    """Create an error telemetry event."""
    return ErrorEvent(
        severity=severity,
        error_type=error_type,
        error_message=error_message,
        **kwargs
    )


def create_performance_metric_event(
    metric_name: str,
    metric_value: Union[int, float],
    metric_unit: str,
    **kwargs
) -> PerformanceMetricEvent:
    """Create a performance metric telemetry event."""
    return PerformanceMetricEvent(
        metric_name=metric_name,
        metric_value=metric_value,
        metric_unit=metric_unit,
        **kwargs
    )


# Factory functions for outcome models
def create_sentinel_outcome(
    domain: str,
    risk_level: str,
    action: str,
    sentinel_name: str,
    findings: Optional[Dict[str, Any]] = None,
    **kwargs
) -> SentinelOutcome:
    """Create a comprehensive sentinel outcome."""
    # Calculate temporal context with defaults
    now = kwargs.pop('timestamp', datetime.utcnow())
    hour_of_day = kwargs.pop('hour_of_day', now.hour)
    day_of_week = kwargs.pop('day_of_week', now.weekday())  # 0=Monday, 6=Sunday

    # Set defaults for required fields not in kwargs
    defaults = {
        'latency_ms': 0,
        'blocked': False,
        'proxy_pool': '',
        'findings': findings or {},
    }

    # Merge defaults with provided kwargs (kwargs take precedence)
    for key, default_value in defaults.items():
        if key not in kwargs:
            kwargs[key] = default_value

    return SentinelOutcome(
        domain=domain,
        hour_of_day=hour_of_day,
        day_of_week=day_of_week,
        risk_level=risk_level,
        action=action,
        timestamp=now,
        sentinel_name=sentinel_name,
        **kwargs
    )
