# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Control Models for MJ Data Scraper Suite

DEPRECATION NOTICE: Core scraping contracts have been moved to mj-shared-lib.
Import from mj_shared.contracts instead:

    from mj_shared.contracts import (
        ScrapeTempo,
        ScrapeBudget,
        ScrapeIntent,
        ScrapeAuthorization,
        ScrapeControlContract,
        IntentClassification,
        CostPrediction,
    )

This file is maintained for backward compatibility only.
New code should use mj-shared-lib contracts directly.
"""

import uuid
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum

# Pydantic replacement using dataclasses for validation
from .retry_utils import RetryConfig

# Re-export shared contracts for backward compatibility
try:
    from mj_shared.contracts import (
        ScrapeTempo as SharedScrapeTempo,
        ScrapeBudget as SharedScrapeBudget,
        ScrapeIntent as SharedScrapeIntent,
        ScrapeAuthorization as SharedScrapeAuthorization,
        ScrapeControlContract as SharedScrapeControlContract,
        IntentClassification as SharedIntentClassification,
        CostPrediction as SharedCostPrediction,
        JobPriority,
        JobStatus,
        IntentRiskLevel,
        IntentCategory,
        GovernanceRequirement,
    )
    _SHARED_LIB_AVAILABLE = True
except ImportError:
    _SHARED_LIB_AVAILABLE = False
    warnings.warn(
        "mj-shared-lib not installed. Using local definitions. "
        "Install mj-shared-lib to prevent drift.",
        DeprecationWarning
    )


# Dataclass-based Models for Advanced Control Contracts

class ScrapeTempo(str, Enum):
    """Scraping tempo levels for rate control."""
    FORENSIC = "forensic"  # Very slow, minimal detection risk
    HUMAN = "human"       # Human-like timing
    AGGRESSIVE = "aggressive"  # Fast, higher detection risk


@dataclass
class ScrapeBudget:
    """Resource and time budget constraints for scraping operations."""
    max_runtime_minutes: int
    max_pages: int
    max_records: int
    max_browser_instances: int
    max_memory_mb: int

    def __post_init__(self):
        """Validate budget constraints."""
        if self.max_runtime_minutes <= 0:
            raise ValueError("max_runtime_minutes must be greater than 0")
        if self.max_pages <= 0:
            raise ValueError("max_pages must be greater than 0")
        if self.max_records <= 0:
            raise ValueError("max_records must be greater than 0")
        if self.max_browser_instances <= 0:
            raise ValueError("max_browser_instances must be greater than 0")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be greater than 0")


@dataclass
class DeploymentWindow:
    """Time window constraints for scraping deployment."""
    earliest_start: datetime
    latest_start: datetime
    max_duration_minutes: int
    timezone: str

    def __post_init__(self):
        """Validate deployment window."""
        if self.max_duration_minutes <= 0:
            raise ValueError("max_duration_minutes must be greater than 0")
        if self.latest_start <= self.earliest_start:
            raise ValueError("latest_start must be after earliest_start")

    def is_within_window(self, check_time: Optional[datetime] = None) -> bool:
        """Check if given time is within deployment window."""
        check_time = check_time or datetime.utcnow()
        return self.earliest_start <= check_time <= self.latest_start


@dataclass
class ScrapeIntent:
    """Intent specification for what data to collect."""
    geography: Dict[str, Any]
    events: Dict[str, Any]
    demographics: Optional[Dict[str, Any]] = None
    sources: List[str] = field(default_factory=list)
    allowed_role: Optional[str] = None  # discovery/verification/enrichment/browser
    event_type: Optional[str] = None     # weddings/corporate/social/professional

    def get_target_criteria(self) -> Dict[str, Any]:
        """Get combined targeting criteria."""
        criteria = {
            "geography": self.geography,
            "events": self.events,
            "sources": self.sources
        }
        if self.demographics:
            criteria["demographics"] = self.demographics
        return criteria


@dataclass
class ScrapeAuthorization:
    """Authorization and approval details for scraping operations."""
    approved_by: str
    purpose: str
    approval_timestamp: datetime
    expires_at: datetime

    def __post_init__(self):
        """Validate authorization timestamps."""
        if self.expires_at <= self.approval_timestamp:
            raise ValueError("Authorization expiry must be after approval time")

    def is_valid(self, check_time: Optional[datetime] = None) -> bool:
        """Check if authorization is still valid."""
        check_time = check_time or datetime.utcnow()
        return check_time <= self.expires_at


@dataclass
class ScrapeControlContract:
    """Master control contract combining all scraping governance elements."""
    intent: ScrapeIntent
    budget: ScrapeBudget
    tempo: ScrapeTempo = ScrapeTempo.HUMAN
    deployment_window: Optional[DeploymentWindow] = None
    authorization: Optional[ScrapeAuthorization] = None
    human_override: bool = False  # Allows bypassing Tier 3 human approval requirement

    def __post_init__(self):
        """Validate contract components."""
        if self.deployment_window is None:
            # Create default deployment window if not provided
            now = datetime.utcnow()
            self.deployment_window = DeploymentWindow(
                earliest_start=now,
                latest_start=now + timedelta(hours=24),
                max_duration_minutes=60,
                timezone="UTC"
            )

        if self.authorization is None:
            # Create default authorization if not provided
            now = datetime.utcnow()
            self.authorization = ScrapeAuthorization(
                approved_by="system",
                purpose="Default scraping operation",
                approval_timestamp=now,
                expires_at=now + timedelta(days=1)
            )

    def can_deploy(self, deploy_time: Optional[datetime] = None) -> bool:
        """Check if deployment is allowed at given time."""
        deploy_time = deploy_time or datetime.utcnow()
        return (self.authorization.is_valid(deploy_time) and
                self.deployment_window.is_within_window(deploy_time))

    def get_tempo_settings(self) -> Dict[str, Any]:
        """Get tempo-specific settings."""
        tempo_configs = {
            ScrapeTempo.FORENSIC: {
                "base_delay": 30.0,  # 30 seconds between requests
                "random_delay": (10, 60),
                "max_concurrent": 1,
                "respect_robots": True,
                "user_agent_rotation": True
            },
            ScrapeTempo.HUMAN: {
                "base_delay": 3.0,   # 3 seconds between requests
                "random_delay": (1, 8),
                "max_concurrent": 3,
                "respect_robots": True,
                "user_agent_rotation": True
            },
            ScrapeTempo.AGGRESSIVE: {
                "base_delay": 0.5,   # 0.5 seconds between requests
                "random_delay": (0.1, 2),
                "max_concurrent": 10,
                "respect_robots": False,
                "user_agent_rotation": False
            }
        }
        return tempo_configs[self.tempo]


class JobPriority(Enum):
    """Job priority levels for queue management."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ScraperType(Enum):
    """Available scraper types in the system."""
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    WEB = "web"
    COMPANY_WEBSITE = "company_website"
    NEWS = "news"
    PUBLIC_RECORDS = "public_records"
    SOCIAL_MEDIA = "social_media"


class DataType(Enum):
    """Data types that can be scraped and processed."""
    PERSON = "person"
    EVENT = "event"
    COMPANY = "company"
    NEWS_ARTICLE = "news_article"
    SOCIAL_POST = "social_post"
    PUBLIC_RECORD = "public_record"


@dataclass
class ControlMetadata:
    """Metadata for control operations."""
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "1.0"
    tags: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScraperControl:
    """Control model for scraper configuration and management."""
    scraper_id: str
    scraper_type: ScraperType
    name: str
    enabled: bool = True
    priority: int = 1
    max_concurrent_jobs: int = 5
    rate_limit: float = 1.0  # requests per second
    timeout: int = 30  # seconds
    max_retries: int = 3
    requires_auth: bool = False
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    user_agent_rotation: bool = True
    proxy_required: bool = False
    data_types: List[DataType] = field(default_factory=list)

    # Anti-detection settings
    anti_detection_enabled: bool = True
    human_behavior_simulation: bool = True
    cookie_persistence: bool = True

    # Monitoring
    health_check_interval: int = 60  # seconds
    failure_threshold: int = 5
    circuit_breaker_timeout: int = 300  # seconds

    # Metadata
    metadata: ControlMetadata = field(default_factory=ControlMetadata)

    def __post_init__(self):
        if not self.scraper_id:
            self.scraper_id = f"{self.scraper_type.value}_{uuid.uuid4().hex[:8]}"

    def is_healthy(self, current_failures: int = 0) -> bool:
        """Check if scraper is healthy based on failure threshold."""
        return current_failures < self.failure_threshold

    def get_effective_rate_limit(self, global_limit: float) -> float:
        """Get the most restrictive rate limit."""
        return min(self.rate_limit, global_limit)


@dataclass
class JobControl:
    """Control model for job execution and management."""
    job_id: str
    scraper_type: ScraperType
    target: Dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING

    # Timing controls
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: int = 300

    # Execution controls
    max_attempts: int = 3
    retry_config: Optional[RetryConfig] = None
    dependencies: List[str] = field(default_factory=list)  # job IDs this depends on

    # Data controls
    data_types: List[DataType] = field(default_factory=list)
    enrichment_required: bool = True
    validation_required: bool = True

    # Resource controls
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    cost_limit: Optional[float] = None

    # Governance controls (Pydantic integration)
    control_contract: Optional[ScrapeControlContract] = None

    # Result controls
    result_storage_enabled: bool = True
    result_retention_days: int = 30
    publish_to_queue: bool = True

    # Metadata
    metadata: ControlMetadata = field(default_factory=ControlMetadata)

    def __post_init__(self):
        if not self.job_id:
            self.job_id = f"job_{uuid.uuid4().hex}"

    @property
    def is_expired(self) -> bool:
        """Check if job has exceeded timeout."""
        if not self.started_at:
            return False
        return (datetime.utcnow() - self.started_at) > timedelta(seconds=self.timeout_seconds)

    @property
    def duration(self) -> Optional[float]:
        """Get job execution duration in seconds."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()

    def can_start(self, completed_job_ids: List[str]) -> bool:
        """Check if job can start based on dependencies."""
        return all(dep_id in completed_job_ids for dep_id in self.dependencies)

    def can_deploy(self, deploy_time: Optional[datetime] = None) -> bool:
        """Check if job can deploy based on control contract."""
        if not self.control_contract:
            return True  # No contract = no restrictions
        return self.control_contract.can_deploy(deploy_time)

    def get_tempo_settings(self) -> Dict[str, Any]:
        """Get tempo settings from control contract."""
        if not self.control_contract:
            return {}  # Default settings
        return self.control_contract.get_tempo_settings()

    def get_resource_limits_from_contract(self) -> Dict[str, Any]:
        """Get resource limits from control contract."""
        if not self.control_contract:
            return {}
        budget = self.control_contract.budget
        return {
            "max_runtime_minutes": budget.max_runtime_minutes,
            "max_pages": budget.max_pages,
            "max_records": budget.max_records,
            "max_browser_instances": budget.max_browser_instances,
            "max_memory_mb": budget.max_memory_mb
        }

    def validate_target_against_intent(self) -> bool:
        """Validate job target against scrape intent."""
        if not self.control_contract:
            return True
        # Implement intent validation logic here
        # For now, return True as placeholder
        return True


@dataclass
class SystemControl:
    """Control model for system-wide configuration."""
    system_id: str = field(default_factory=lambda: f"mj_scraper_{uuid.uuid4().hex[:8]}")

    # Engine controls
    max_concurrent_jobs: int = 10
    max_queue_size: int = 1000
    job_timeout_default: int = 300
    enable_metrics: bool = True

    # Global rate limiting
    global_rate_limit: float = 10.0  # requests per second across all scrapers
    burst_limit: int = 50

    # Resource controls
    max_memory_mb: int = 1024
    max_cpu_percent: float = 80.0
    thread_pool_size: int = 20

    # Anti-detection controls
    anti_detection_enabled: bool = True
    proxy_rotation_enabled: bool = False
    user_agent_rotation_enabled: bool = True

    # Monitoring controls
    metrics_interval: int = 30  # seconds
    health_check_interval: int = 60  # seconds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate": 0.1,
        "avg_response_time": 30.0,
        "queue_size": 500
    })

    # Integration controls
    azure_service_bus_enabled: bool = True
    azure_blob_storage_enabled: bool = True
    mj_ingestion_enabled: bool = True

    # Metadata
    metadata: ControlMetadata = field(default_factory=ControlMetadata)

    def get_resource_limits(self) -> Dict[str, Any]:
        """Get current resource limits."""
        return {
            "memory_mb": self.max_memory_mb,
            "cpu_percent": self.max_cpu_percent,
            "threads": self.thread_pool_size
        }


@dataclass
class QueueControl:
    """Control model for queue management and publishing."""
    queue_name: str
    connection_string: str
    enabled: bool = True

    # Batch controls
    max_batch_size: int = 10
    batch_timeout_seconds: int = 30
    max_message_size_kb: int = 256

    # Retry controls
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True

    # Monitoring
    dead_letter_enabled: bool = True
    dead_letter_queue: Optional[str] = None

    # Throughput controls
    max_throughput_per_second: int = 100
    enable_compression: bool = True

    # Metadata
    metadata: ControlMetadata = field(default_factory=ControlMetadata)

    def get_batch_config(self) -> Dict[str, Any]:
        """Get batch configuration for publishing."""
        return {
            "max_size": self.max_batch_size,
            "timeout": self.batch_timeout_seconds,
            "max_message_size": self.max_message_size_kb * 1024,
            "compression": self.enable_compression
        }


@dataclass
class DataControl:
    """Control model for data processing and validation."""
    data_type: DataType
    schema_version: str = "1.0"

    # Validation controls
    validation_enabled: bool = True
    schema_validation: bool = True
    data_quality_checks: bool = True

    # Enrichment controls
    enrichment_enabled: bool = True
    enrichment_pipeline: List[str] = field(default_factory=list)

    # Storage controls
    storage_format: str = "json"  # json, parquet, avro
    compression: str = "gzip"
    retention_days: int = 365

    # Privacy controls
    pii_detection: bool = True
    data_anonymization: bool = False
    gdpr_compliance: bool = True

    # Quality thresholds
    min_confidence_score: float = 0.7
    max_error_rate: float = 0.05

    # Metadata
    metadata: ControlMetadata = field(default_factory=ControlMetadata)

    def validate_data_quality(self, data: Dict[str, Any]) -> bool:
        """Validate data meets quality thresholds."""
        if not self.validation_enabled:
            return True

        confidence = data.get("confidence", 0.0)
        return confidence >= self.min_confidence_score


@dataclass
class ControlCommand:
    """Control command for system management operations."""
    command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    command_type: str = ""
    target: str = ""  # scraper_id, job_id, or "system"
    action: str = ""  # start, stop, pause, resume, restart, update
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Execution tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, executing, completed, failed

    # Result tracking
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    # Metadata
    metadata: ControlMetadata = field(default_factory=ControlMetadata)

    def mark_executed(self):
        """Mark command as executed."""
        self.executed_at = datetime.utcnow()
        self.status = "executing"

    def mark_completed(self, result: Optional[Dict[str, Any]] = None):
        """Mark command as completed."""
        self.completed_at = datetime.utcnow()
        self.status = "completed"
        self.result = result

    def mark_failed(self, error: str):
        """Mark command as failed."""
        self.completed_at = datetime.utcnow()
        self.status = "failed"
        self.error_message = error


# Utility functions for control model management

def create_scraper_control(
    scraper_type: ScraperType,
    name: str,
    **kwargs
) -> ScraperControl:
    """Factory function for creating scraper controls."""
    return ScraperControl(
        scraper_id="",
        scraper_type=scraper_type,
        name=name,
        **kwargs
    )


def create_job_control(
    scraper_type: ScraperType,
    target: Dict[str, Any],
    **kwargs
) -> JobControl:
    """Factory function for creating job controls."""
    return JobControl(
        job_id="",
        scraper_type=scraper_type,
        target=target,
        **kwargs
    )


def validate_control_model(model: Any) -> bool:
    """Validate a control model has required fields."""
    if hasattr(model, 'metadata') and isinstance(model.metadata, ControlMetadata):
        return True
    return False


def serialize_control_model(model: Any) -> Dict[str, Any]:
    """Serialize a control model to dictionary."""
    data = asdict(model)

    # Convert enums to values
    for key, value in data.items():
        if isinstance(value, Enum):
            data[key] = value.value
        elif isinstance(value, datetime):
            data[key] = value.isoformat()
        elif isinstance(value, set):
            data[key] = list(value)

    return data


def deserialize_control_model(data: Dict[str, Any], model_class: type) -> Any:
    """Deserialize dictionary to control model."""
    # Convert string values back to enums
    for field_name, field_type in model_class.__annotations__.items():
        if hasattr(field_type, '__members__'):  # Enum type
            if field_name in data and data[field_name] in field_type.__members__:
                data[field_name] = field_type(data[field_name])
        elif field_name == 'created_at' and field_name in data:
            data[field_name] = datetime.fromisoformat(data[field_name])

    return model_class(**data)
