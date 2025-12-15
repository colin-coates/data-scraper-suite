# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Core Scraper Engine - Advanced Orchestration Engine

Enhanced job dispatching, scraper management, and governance integration
for the MJ Data Scraper Suite with control contract enforcement.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Type, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import json

from azure.servicebus import ServiceBusClient
from azure.servicebus.aio import ServiceBusClient as AsyncServiceBusClient
from azure.storage.blob import BlobServiceClient

from .base_scraper import BaseScraper, ScraperConfig, ScraperResult
from .queue_publisher import QueuePublisher, QueueConfig, PublishResult
from .mj_envelope import MJMessageEnvelope
from .mj_payload_builder import build_person_payload, build_event_payload
from .anti_detection.anti_detection import AntiDetectionLayer
from .control_models import (
    JobPriority, JobStatus, ScraperType, DataType,
    ScraperControl, JobControl, SystemControl, QueueControl,
    ControlMetadata, ScrapeControlContract, ScrapeTempo
)

logger = logging.getLogger(__name__)


@dataclass
class EngineMetrics:
    """Real-time engine performance metrics."""
    jobs_processed: int = 0
    jobs_succeeded: int = 0
    jobs_failed: int = 0
    total_processing_time: float = 0.0
    active_jobs: int = 0
    queued_jobs: int = 0
    scraper_usage: Dict[str, int] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.utcnow)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.jobs_succeeded + self.jobs_failed
        return self.jobs_succeeded / max(1, total)

    @property
    def avg_processing_time(self) -> float:
        """Calculate average processing time."""
        return self.total_processing_time / max(1, self.jobs_processed)


@dataclass
class EngineConfig:
    """Configuration for the scraper engine with governance controls."""
    max_concurrent_jobs: int = 10
    job_queue_size: int = 1000
    enable_metrics: bool = True
    enable_anti_detection: bool = True

    # Azure integration
    azure_service_bus_connection: str = ""
    azure_blob_connection: str = ""
    azure_queue_name: str = "scraping-jobs"
    azure_blob_container: str = "scraping-results"

    # Governance settings
    require_control_contracts: bool = False  # Make contracts mandatory
    enforce_deployment_windows: bool = True
    validate_authorizations: bool = True

    # Output settings
    output_queue_name: str = "scraping-results"
    enable_result_publishing: bool = True
    default_rate_limit: float = 1.0


class CoreScraperEngine:
    """
    Advanced scraper engine with governance and control contract integration.

    Provides enterprise-grade job orchestration with compliance, resource management,
    and real-time governance enforcement.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.logger = logging.getLogger(f"{__name__}.CoreScraperEngine")

        # Core registries
        self.scrapers: Dict[str, Type[BaseScraper]] = {}
        self.active_scrapers: Dict[str, BaseScraper] = {}
        self.scraper_controls: Dict[str, ScraperControl] = {}

        # Job management with priority queues
        self.job_queues: Dict[JobPriority, asyncio.Queue[JobControl]] = {
            priority: asyncio.Queue(maxsize=self.config.job_queue_size)
            for priority in JobPriority
        }
        self.active_jobs: Dict[str, JobControl] = {}
        self.completed_jobs: List[JobControl] = []
        self.job_dependencies: Dict[str, Set[str]] = {}  # job_id -> set of blocking job_ids

        # Azure clients
        self.service_bus_client: Optional[AsyncServiceBusClient] = None
        self.blob_service_client: Optional[BlobServiceClient] = None
        self.queue_publisher: Optional[QueuePublisher] = None

        # Governance and control
        self.anti_detection = AntiDetectionLayer() if self.config.enable_anti_detection else None
        self.system_control: Optional[SystemControl] = None

        # Control and metrics
        self.running = False
        self.shutdown_event = asyncio.Event()
        self.metrics = EngineMetrics()

        # Resource management
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_jobs)

        self.logger.info("CoreScraperEngine initialized with governance controls")

    async def initialize(self, system_control: Optional[SystemControl] = None) -> None:
        """Initialize the scraper engine with system controls."""
        try:
            self.system_control = system_control or SystemControl()

            # Initialize Azure clients
            if self.config.azure_service_bus_connection:
                self.service_bus_client = AsyncServiceBusClient.from_connection_string(
                    self.config.azure_service_bus_connection
                )
                self.logger.info("Azure Service Bus client initialized")

            if self.config.azure_blob_connection:
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    self.config.azure_blob_connection
                )
                self.logger.info("Azure Blob Storage client initialized")

            # Initialize queue publisher
            if self.config.enable_result_publishing and self.config.azure_service_bus_connection:
                queue_config = QueueConfig(
                    connection_string=self.config.azure_service_bus_connection,
                    queue_name=self.config.output_queue_name
                )
                self.queue_publisher = QueuePublisher(queue_config)
                await self.queue_publisher.initialize()
                self.logger.info("Queue publisher initialized")

            # Initialize anti-detection
            if self.anti_detection:
                await self.anti_detection.initialize()
                self.logger.info("Anti-detection layer initialized")

            self.logger.info("CoreScraperEngine initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize CoreScraperEngine: {e}")
            raise

    def register_scraper(
        self,
        scraper_type: ScraperType,
        scraper_class: Type[BaseScraper],
        scraper_control: Optional[ScraperControl] = None
    ) -> None:
        """
        Register a scraper with governance controls.

        Args:
            scraper_type: Type of scraper to register
            scraper_class: The scraper class implementation
            scraper_control: Optional governance controls for this scraper
        """
        scraper_key = scraper_type.value

        if scraper_key in self.scrapers:
            self.logger.warning(f"Scraper type '{scraper_key}' already registered, overwriting")

        self.scrapers[scraper_key] = scraper_class

        # Create or use provided scraper control
        if scraper_control is None:
            scraper_control = ScraperControl(
                scraper_id="",
                scraper_type=scraper_type,
                name=f"{scraper_type.value}_scraper"
            )

        self.scraper_controls[scraper_key] = scraper_control

        # Instantiate the scraper
        scraper_instance = scraper_class(scraper_control.__dict__)

        # Attach anti-detection if available
        if self.anti_detection:
            scraper_instance.set_anti_detection(self.anti_detection)

        self.active_scrapers[scraper_key] = scraper_instance

        self.logger.info(f"Registered scraper: {scraper_key} with governance controls")

    def unregister_scraper(self, scraper_type: ScraperType) -> None:
        """Unregister a scraper and cleanup resources."""
        scraper_key = scraper_type.value

        if scraper_key in self.active_scrapers:
            scraper = self.active_scrapers[scraper_key]
            asyncio.create_task(scraper.cleanup())

            del self.active_scrapers[scraper_key]

        if scraper_key in self.scrapers:
            del self.scrapers[scraper_key]

        if scraper_key in self.scraper_controls:
            del self.scraper_controls[scraper_key]

        self.logger.info(f"Unregistered scraper: {scraper_key}")

    async def submit_job(self, job_control: JobControl) -> str:
        """
        Submit a job with governance validation.

        Args:
            job_control: Complete job control with governance contract

        Returns:
            Job ID for tracking

        Raises:
            ValueError: If job fails governance checks
        """
        # Validate job against governance rules
        await self._validate_job_governance(job_control)

        # Generate job ID if not provided
        if not job_control.job_id:
            job_control.job_id = f"job_{int(time.time() * 1000)}_{hash(str(job_control.target)) % 10000}"

        # Check dependencies
        if job_control.dependencies and not await self._check_dependencies(job_control):
            raise ValueError(f"Job dependencies not satisfied: {job_control.dependencies}")

        # Add to appropriate priority queue
        try:
            await self.job_queues[job_control.priority].put(job_control)
            self.active_jobs[job_control.job_id] = job_control
            self.metrics.queued_jobs += 1

            self.logger.info(f"Job submitted: {job_control.job_id} ({job_control.scraper_type.value})")
            return job_control.job_id

        except asyncio.QueueFull:
            raise RuntimeError("Job queue is full - system at capacity")

    async def _validate_job_governance(self, job_control: JobControl) -> None:
        """Validate job against governance rules."""
        # Check if control contracts are required
        if self.config.require_control_contracts and job_control.control_contract is None:
            raise ValueError("Control contract required but not provided")

        # Validate contract if present
        if job_control.control_contract:
            contract = job_control.control_contract

            # Check deployment window
            if self.config.enforce_deployment_windows and not contract.can_deploy():
                raise ValueError("Job deployment not permitted within current time window")

            # Validate authorization
            if self.config.validate_authorizations and not contract.authorization.is_valid():
                raise ValueError("Job authorization has expired")

            # Check resource limits against system controls
            if self.system_control:
                contract_limits = job_control.get_resource_limits_from_contract()
                system_limits = self.system_control.get_resource_limits()

                for resource, limit in contract_limits.items():
                    if resource in system_limits and limit > system_limits[resource]:
                        raise ValueError(f"Contract resource limit exceeds system limit: {resource}")

        # Validate scraper is available and healthy
        scraper_key = job_control.scraper_type.value
        if scraper_key not in self.active_scrapers:
            raise ValueError(f"Scraper not available: {scraper_key}")

        scraper_control = self.scraper_controls.get(scraper_key)
        if scraper_control and not scraper_control.is_healthy():
            raise ValueError(f"Scraper not healthy: {scraper_key}")

    async def _check_dependencies(self, job_control: JobControl) -> bool:
        """Check if all job dependencies are satisfied."""
        for dep_id in job_control.dependencies:
            if dep_id not in self.active_jobs:
                # Check completed jobs
                completed_job = next(
                    (j for j in self.completed_jobs if j.job_id == dep_id), None
                )
                if not completed_job or completed_job.status != JobStatus.COMPLETED:
                    return False
        return True

    async def start_processing(self) -> None:
        """Start the job processing engine."""
        if self.running:
            self.logger.warning("Engine already running")
            return

        self.running = True
        self.shutdown_event.clear()

        self.logger.info(f"Starting CoreScraperEngine with {self.config.max_concurrent_jobs} concurrent jobs")

        try:
            # Start job processors for each priority level
            processors = []
            for priority in JobPriority:
                priority_processors = [
                    asyncio.create_task(self._job_processor(priority))
                    for _ in range(self.config.max_concurrent_jobs // len(JobPriority))
                ]
                processors.extend(priority_processors)

            # Start queue monitor
            monitor = asyncio.create_task(self._queue_monitor())

            # Wait for shutdown
            await self.shutdown_event.wait()

            # Cancel processors
            for processor in processors:
                processor.cancel()
            monitor.cancel()

            # Wait for cleanup
            await asyncio.gather(*processors, monitor, return_exceptions=True)

        except Exception as e:
            self.logger.error(f"Engine error: {e}")
        finally:
            self.running = False
            self.logger.info("CoreScraperEngine stopped")

    async def stop_processing(self) -> None:
        """Stop the job processing engine gracefully."""
        self.logger.info("Stopping CoreScraperEngine...")
        self.shutdown_event.set()

    async def _job_processor(self, priority: JobPriority) -> None:
        """Process jobs from a specific priority queue."""
        while not self.shutdown_event.is_set():
            try:
                # Get job from queue with timeout
                job_control = await asyncio.wait_for(
                    self.job_queues[priority].get(),
                    timeout=1.0
                )

                # Process the job
                await self._process_job(job_control)

                # Mark queue task as done
                self.job_queues[priority].task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Job processor error: {e}")
                await asyncio.sleep(1)

    async def _process_job(self, job_control: JobControl) -> None:
        """Process a single scraping job with governance enforcement."""
        start_time = time.time()
        job_control.status = JobStatus.RUNNING
        job_control.started_at = datetime.utcnow()

        self.metrics.active_jobs += 1
        self.metrics.queued_jobs -= 1

        try:
            self.logger.info(f"Processing job {job_control.job_id} with {job_control.scraper_type.value}")

            # Get scraper instance
            scraper = self.active_scrapers.get(job_control.scraper_type.value)
            if not scraper:
                raise RuntimeError(f"Scraper {job_control.scraper_type.value} not available")

            # Apply tempo settings from contract if available
            if job_control.control_contract:
                tempo_settings = job_control.control_contract.get_tempo_settings()
                # Apply tempo settings to scraper (would need scraper API for this)

            # Execute scraping
            result = await scraper.scrape(job_control.target)

            # Update job
            job_control.result = result
            job_control.status = JobStatus.COMPLETED if result.success else JobStatus.FAILED
            job_control.completed_at = datetime.utcnow()

            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.jobs_processed += 1
            self.metrics.total_processing_time += processing_time

            if result.success:
                self.metrics.jobs_succeeded += 1
            else:
                self.metrics.jobs_failed += 1

            # Update scraper usage
            scraper_key = job_control.scraper_type.value
            self.metrics.scraper_usage[scraper_key] = \
                self.metrics.scraper_usage.get(scraper_key, 0) + 1

            # Publish result
            await self._publish_result(job_control)

            # Auto-publish to queue if enabled
            if self.queue_publisher and result.success and self.config.enable_result_publishing:
                try:
                    await self._publish_to_queue_publisher(job_control)
                except Exception as e:
                    self.logger.error(f"Error auto-publishing job {job_control.job_id}: {e}")

            self.logger.info(
                f"Job {job_control.job_id} completed in {processing_time:.2f}s: "
                f"{'SUCCESS' if result.success else 'FAILED'}"
            )

        except Exception as e:
            job_control.status = JobStatus.FAILED
            job_control.result = ScraperResult(success=False, error_message=str(e))
            job_control.completed_at = datetime.utcnow()

            self.metrics.jobs_failed += 1
            self.logger.error(f"Job {job_control.job_id} failed: {e}")

        finally:
            # Move to completed jobs (keep last 1000)
            self.completed_jobs.append(job_control)
            if len(self.completed_jobs) > 1000:
                self.completed_jobs.pop(0)

            # Remove from active jobs
            self.active_jobs.pop(job_control.job_id, None)
            self.metrics.active_jobs -= 1

    async def _publish_result(self, job_control: JobControl) -> None:
        """Publish job results to configured outputs."""
        if not job_control.result:
            return

        try:
            result_data = {
                'job_id': job_control.job_id,
                'scraper_type': job_control.scraper_type.value,
                'target': job_control.target,
                'success': job_control.result.success,
                'data': job_control.result.data,
                'error_message': job_control.result.error_message,
                'metadata': job_control.result.metadata,
                'timestamp': job_control.result.timestamp.isoformat(),
                'processing_time': job_control.result.response_time,
                'retry_count': job_control.result.retry_count,
                'control_contract': job_control.control_contract.__dict__ if job_control.control_contract else None
            }

            # MJ ingestion integration
            if job_control.result.success and job_control.result.data and job_control.control_contract:
                data_type = job_control.result.data.get("data_type")
                if data_type in ["person", "event"]:
                    payload = self._build_mj_payload(data_type, job_control.result.data)

                    envelope = MJMessageEnvelope(
                        data_type=data_type,
                        payload=payload,
                        correlation_id=job_control.job_id
                    )

                    if self.queue_publisher:
                        await self.queue_publisher.publish_to_queue(envelope.to_dict())

            # Publish to Azure Service Bus
            if self.service_bus_client:
                await self._publish_to_service_bus(result_data)

            # Store in Azure Blob Storage
            if self.blob_service_client:
                await self._store_in_blob_storage(job_control.job_id, result_data)

        except Exception as e:
            self.logger.error(f"Failed to publish result for job {job_control.job_id}: {e}")

    def _build_mj_payload(self, data_type: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build MJ payload based on data type."""
        if data_type == "person":
            return build_person_payload(raw_data)
        elif data_type == "event":
            return build_event_payload(raw_data)
        else:
            return raw_data

    async def _publish_to_service_bus(self, result_data: Dict[str, Any]) -> None:
        """Publish result to Azure Service Bus."""
        try:
            async with self.service_bus_client.get_queue_sender(
                queue_name=self.config.azure_queue_name
            ) as sender:
                message = json.dumps(result_data)
                await sender.send_messages(message)
        except Exception as e:
            self.logger.error(f"Service Bus publish failed: {e}")

    async def _store_in_blob_storage(self, job_id: str, result_data: Dict[str, Any]) -> None:
        """Store result in Azure Blob Storage."""
        try:
            blob_name = f"{job_id}.json"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.config.azure_blob_container,
                blob=blob_name
            )

            data = json.dumps(result_data, indent=2)
            await blob_client.upload_blob(data, overwrite=True)
        except Exception as e:
            self.logger.error(f"Blob storage upload failed: {e}")

    async def _publish_to_queue_publisher(self, job_control: JobControl) -> None:
        """Publish job result via queue publisher."""
        if not self.queue_publisher or not job_control.result:
            return

        queue_data = {
            "job_id": job_control.job_id,
            "scraper_type": job_control.scraper_type.value,
            "target": job_control.target,
            "result": {
                "success": job_control.result.success,
                "data": job_control.result.data,
                "error_message": job_control.result.error_message,
                "response_time": job_control.result.response_time,
                "retry_count": job_control.result.retry_count
            },
            "control_contract": job_control.control_contract.__dict__ if job_control.control_contract else None,
            "metadata": job_control.metadata.__dict__ if job_control.metadata else None
        }

        await self.queue_publisher.publish_to_queue(queue_data)

    async def _queue_monitor(self) -> None:
        """Monitor queue status and log metrics."""
        while not self.shutdown_event.is_set():
            try:
                total_queued = sum(queue.qsize() for queue in self.job_queues.values())

                self.logger.debug(
                    f"Queue status: {total_queued} queued, "
                    f"{self.metrics.active_jobs} active, "
                    f"{len(self.completed_jobs)} completed"
                )

                if self.config.enable_metrics:
                    # Log metrics every 30 seconds
                    pass

                await asyncio.sleep(10)

            except Exception as e:
                self.logger.error(f"Queue monitor error: {e}")
                await asyncio.sleep(5)

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        job = self.active_jobs.get(job_id) or next(
            (j for j in self.completed_jobs if j.job_id == job_id), None
        )

        if job:
            return {
                'job_id': job.job_id,
                'status': job.status.value,
                'scraper_type': job.scraper_type.value,
                'created_at': job.created_at.isoformat(),
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                'priority': job.priority.value,
                'duration': job.duration,
                'result': {
                    'success': job.result.success if job.result else False,
                    'error_message': job.result.error_message if job.result else None,
                    'response_time': job.result.response_time if job.result else 0.0
                } if job.result else None,
                'control_contract': job.control_contract.__dict__ if job.control_contract else None
            }

        return None

    def get_engine_metrics(self) -> Dict[str, Any]:
        """Get comprehensive engine performance metrics."""
        uptime = (datetime.utcnow() - self.metrics.start_time).total_seconds()

        return {
            'uptime_seconds': uptime,
            'jobs_processed': self.metrics.jobs_processed,
            'jobs_succeeded': self.metrics.jobs_succeeded,
            'jobs_failed': self.metrics.jobs_failed,
            'success_rate': self.metrics.success_rate,
            'average_processing_time': self.metrics.avg_processing_time,
            'active_jobs': self.metrics.active_jobs,
            'queued_jobs': sum(queue.qsize() for queue in self.job_queues.values()),
            'completed_jobs': len(self.completed_jobs),
            'scraper_usage': self.metrics.scraper_usage,
            'registered_scrapers': list(self.scrapers.keys()),
            'active_scrapers': list(self.active_scrapers.keys()),
            'anti_detection_enabled': self.anti_detection is not None,
            'result_publishing_enabled': self.config.enable_result_publishing,
            'governance_enabled': self.config.require_control_contracts,
            'system_control': self.system_control.__dict__ if self.system_control else None
        }

    def get_registered_scrapers(self) -> List[str]:
        """Get list of registered scraper types."""
        return list(self.scrapers.keys())

    def get_scraper_status(self, scraper_type: ScraperType) -> Optional[Dict[str, Any]]:
        """Get status and metrics for a specific scraper."""
        scraper_key = scraper_type.value
        scraper = self.active_scrapers.get(scraper_key)
        scraper_control = self.scraper_controls.get(scraper_key)

        if scraper and scraper_control:
            return {
                'scraper_type': scraper_key,
                'control': scraper_control.__dict__,
                'metrics': scraper.get_metrics(),
                'is_healthy': scraper_control.is_healthy(),
                'usage_count': self.metrics.scraper_usage.get(scraper_key, 0)
            }
        return None

    async def cleanup(self) -> None:
        """Cleanup engine resources."""
        self.logger.info("Cleaning up CoreScraperEngine...")

        # Cleanup scrapers
        for scraper in self.active_scrapers.values():
            try:
                await scraper.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up scraper {scraper.config.name}: {e}")

        # Cleanup anti-detection
        if self.anti_detection:
            await self.anti_detection.cleanup()

        # Close clients
        if self.service_bus_client:
            await self.service_bus_client.close()

        # Cleanup queue publisher
        if self.queue_publisher:
            await self.queue_publisher.cleanup()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        self.logger.info("CoreScraperEngine cleanup complete")

    def __str__(self) -> str:
        return (
            f"CoreScraperEngine("
            f"jobs_processed={self.metrics.jobs_processed}, "
            f"active_jobs={self.metrics.active_jobs}, "
            f"registered_scrapers={len(self.scrapers)}, "
            f"governance={'enabled' if self.config.require_control_contracts else 'optional'})"
        )
