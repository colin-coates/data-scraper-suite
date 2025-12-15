# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Scraper Engine - Centralized Orchestration Engine

Manages job dispatching, scraper registration, plugin architecture,
and provides the main interface for the MJ Data Scraper Suite.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import json

from azure.servicebus import ServiceBusClient
from azure.servicebus.aio import ServiceBusClient as AsyncServiceBusClient
from azure.storage.blob import BlobServiceClient

from .core.base_scraper import BaseScraper, ScraperConfig, ScraperResult
from .core.queue_publisher import QueuePublisher, QueueConfig, PublishResult
from .core.mj_envelope import MJMessageEnvelope
from .core.mj_payload_builder import build_person_payload, build_event_payload
from .core.scraper_engine import CoreScraperEngine, EngineConfig
from .anti_detection.anti_detection import AntiDetectionLayer

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """Represents a scraping job."""
    job_id: str
    scraper_type: str
    target: Dict[str, Any]
    priority: str = "normal"  # low, normal, high, urgent
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[ScraperResult] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineConfig:
    """Configuration for the scraper engine."""
    max_concurrent_jobs: int = 10
    job_queue_size: int = 1000
    enable_metrics: bool = True
    azure_service_bus_connection: str = ""
    azure_blob_connection: str = ""
    azure_queue_name: str = "scraping-jobs"
    azure_blob_container: str = "scraping-results"
    output_queue_name: str = "scraping-results"  # Queue for publishing results
    enable_result_publishing: bool = True  # Enable publishing results to queue
    default_rate_limit: float = 1.0
    enable_anti_detection: bool = True


class ScraperEngine:
    """
    Legacy compatibility layer for the MJ Data Scraper Suite.

    This class provides backward compatibility while delegating to the
    enhanced CoreScraperEngine with governance and control contract support.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.logger = logging.getLogger(f"{__name__}.ScraperEngine")

        # Initialize core engine
        self.core_engine = CoreScraperEngine(self.config)

        # Legacy compatibility attributes
        self.scrapers = self.core_engine.scrapers
        self.active_scrapers = self.core_engine.active_scrapers
        self.active_jobs = self.core_engine.active_jobs
        self.completed_jobs = self.core_engine.completed_jobs

        self.logger.info("ScraperEngine initialized (with CoreScraperEngine backend)")

    async def initialize(self) -> None:
        """Initialize the scraper engine with system controls."""
        await self.core_engine.initialize()

    def register_scraper(self, scraper_type: str, scraper_class: Type[BaseScraper],
                        config: Optional[ScraperConfig] = None) -> None:
        """
        Register a scraper class with the engine (legacy compatibility).

        Args:
            scraper_type: Unique identifier for the scraper type
            scraper_class: The scraper class to register
            config: Optional default configuration for the scraper
        """
        # Convert legacy registration to new control model system
        from .core.control_models import ScraperType as ControlScraperType, ScraperControl

        try:
            control_scraper_type = ControlScraperType(scraper_type)
        except ValueError:
            self.logger.warning(f"Unknown scraper type '{scraper_type}', registering as generic")
            # For backward compatibility, allow any string
            control_scraper_type = ControlScraperType.WEB  # Default fallback

        # Create scraper control from legacy config
        scraper_control = ScraperControl(
            scraper_id="",
            scraper_type=control_scraper_type,
            name=scraper_type
        )

        # Apply legacy config if provided
        if config:
            scraper_control.rate_limit = config.rate_limit_delay
            scraper_control.max_retries = config.max_retries
            scraper_control.timeout = config.timeout

        # Register with core engine
        self.core_engine.register_scraper(control_scraper_type, scraper_class, scraper_control)

    def unregister_scraper(self, scraper_type: str) -> None:
        """Unregister a scraper type."""
        if scraper_type in self.active_scrapers:
            scraper = self.active_scrapers[scraper_type]
            asyncio.create_task(scraper.cleanup())
            del self.active_scrapers[scraper_type]

        if scraper_type in self.scrapers:
            del self.scrapers[scraper_type]

        self.logger.info(f"Unregistered scraper: {scraper_type}")

    async def dispatch_job(self, job_data: Dict[str, Any]) -> str:
        """
        Dispatch a scraping job to the engine (legacy compatibility).

        Args:
            job_data: Job specification including scraper_type, target, etc.

        Returns:
            Job ID for tracking
        """
        from .core.control_models import JobControl, ScraperType as ControlScraperType, JobPriority

        # Convert legacy job data to JobControl
        scraper_type_str = job_data.get('scraper_type', '')
        try:
            scraper_type = ControlScraperType(scraper_type_str)
        except ValueError:
            raise ValueError(f"Unknown scraper type: {scraper_type_str}")

        # Convert priority string to enum
        priority_str = job_data.get('priority', 'normal')
        try:
            priority = JobPriority(priority_str)
        except ValueError:
            priority = JobPriority.NORMAL

        # Create JobControl with optional contract
        job_control = JobControl(
            job_id="",  # Will be generated
            scraper_type=scraper_type,
            target=job_data.get('target', {}),
            priority=priority,
            metadata=job_data.get('metadata', {}),
            control_contract=job_data.get('control_contract')  # Optional governance
        )

        # Submit to core engine
        return await self.core_engine.submit_job(job_control)

    async def start(self) -> None:
        """Start the scraper engine processing loop."""
        await self.core_engine.start_processing()

    async def stop(self) -> None:
        """Stop the scraper engine gracefully."""
        await self.core_engine.stop_processing()

    async def _job_processor(self) -> None:
        """Process jobs from the queue."""
        while not self.shutdown_event.is_set():
            try:
                # Get job from queue with timeout
                job = await asyncio.wait_for(
                    self.job_queue.get(),
                    timeout=1.0
                )

                # Process the job
                await self._process_job(job)

                # Mark queue task as done
                self.job_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Job processor error: {e}")
                await asyncio.sleep(1)

    async def _process_job(self, job: Job) -> None:
        """Process a single scraping job."""
        start_time = time.time()
        job.status = "running"

        try:
            self.logger.info(f"Processing job {job.job_id} with {job.scraper_type}")

            # Get scraper instance
            scraper = self.active_scrapers.get(job.scraper_type)
            if not scraper:
                raise RuntimeError(f"Scraper {job.scraper_type} not available")

            # Execute scraping
            result = await scraper.scrape(job.target)

            # Update job
            job.result = result
            job.status = "completed" if result.success else "failed"

            # Update metrics
            processing_time = time.time() - start_time
            self.metrics['jobs_processed'] += 1
            self.metrics['total_processing_time'] += processing_time

            if result.success:
                self.metrics['jobs_succeeded'] += 1
            else:
                self.metrics['jobs_failed'] += 1

            # Update scraper usage
            self.metrics['scraper_usage'][job.scraper_type] = \
                self.metrics['scraper_usage'].get(job.scraper_type, 0) + 1

            # Publish result
            await self._publish_result(job)

            # Auto-publish to queue if enabled
            if self.queue_publisher and result.success:
                try:
                    queue_data = {
                        "job_id": job.job_id,
                        "scraper_type": job.scraper_type,
                        "target": job.target,
                        "result": {
                            "success": result.success,
                            "data": result.data,
                            "error_message": result.error_message,
                            "response_time": result.response_time,
                            "retry_count": result.retry_count
                        },
                        "metadata": job.metadata
                    }

                    publish_result = await self.queue_publisher.publish_to_queue(queue_data)
                    if not publish_result.success:
                        self.logger.warning(f"Failed to publish job {job.job_id} result to queue: {publish_result.errors}")

                except Exception as e:
                    self.logger.error(f"Error auto-publishing job {job.job_id} to queue: {e}")

            self.logger.info(f"Job {job.job_id} completed in {processing_time:.2f}s: "
                           f"{'SUCCESS' if result.success else 'FAILED'}")

        except Exception as e:
            job.status = "failed"
            job.result = ScraperResult(success=False, error_message=str(e))
            self.metrics['jobs_failed'] += 1
            self.logger.error(f"Job {job.job_id} failed: {e}")

        finally:
            # Move to completed jobs (keep last 1000)
            self.completed_jobs.append(job)
            if len(self.completed_jobs) > 1000:
                self.completed_jobs.pop(0)

            # Remove from active jobs
            self.active_jobs.pop(job.job_id, None)

    async def publish_to_queue(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> PublishResult:
        """
        Async wrapper for publishing data to the results queue.

        Args:
            data: Data to publish (dict, list, or any serializable object)
            metadata: Optional metadata to attach

        Returns:
            PublishResult with operation status
        """
        if not self.queue_publisher:
            raise RuntimeError("Queue publisher not initialized. Check configuration.")

        return await self.queue_publisher.publish_to_queue(data, metadata)

    async def flush_queue(self) -> PublishResult:
        """
        Flush any pending batched messages in the queue publisher.

        Returns:
            PublishResult with flush operation status
        """
        if not self.queue_publisher:
            raise RuntimeError("Queue publisher not initialized.")

        return await self.queue_publisher.flush()

    async def _publish_result(self, job: Job) -> None:
        """Publish job results to configured outputs."""
        if not job.result:
            return

        try:
            result_data = {
                'job_id': job.job_id,
                'scraper_type': job.scraper_type,
                'target': job.target,
                'success': job.result.success,
                'data': job.result.data,
                'error_message': job.result.error_message,
                'metadata': job.result.metadata,
                'timestamp': job.result.timestamp.isoformat(),
                'processing_time': job.result.response_time,
                'retry_count': job.result.retry_count
            }

            # MJ ingestion integration - detect person/event and wrap in envelope
            if job.result.success and job.result.data:
                data_type = job.result.data.get("data_type")
                if data_type in ["person", "event"]:
                    # Build payload using mj_payload_builder
                    if data_type == "person":
                        payload = build_person_payload(job.result.data)
                    elif data_type == "event":
                        payload = build_event_payload(job.result.data)

                    # Wrap using MJMessageEnvelope
                    envelope = MJMessageEnvelope(
                        data_type=data_type,
                        payload=payload
                    )

                    # Publish via queue publisher
                    if self.queue_publisher:
                        await self.queue_publisher.publish_to_queue(envelope.to_dict())

            # Publish to Azure Service Bus
            if self.service_bus_client:
                await self._publish_to_service_bus(result_data)

            # Store in Azure Blob Storage
            if self.blob_service_client:
                await self._store_in_blob_storage(job.job_id, result_data)

        except Exception as e:
            self.logger.error(f"Failed to publish result for job {job.job_id}: {e}")

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

    async def _queue_monitor(self) -> None:
        """Monitor queue status and log metrics."""
        while not self.shutdown_event.is_set():
            try:
                queue_size = self.job_queue.qsize()
                active_count = len(self.active_jobs)

                self.logger.debug(f"Queue status: {queue_size} queued, "
                                f"{active_count} active, "
                                f"{len(self.completed_jobs)} completed")

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
                'status': job.status,
                'scraper_type': job.scraper_type,
                'created_at': job.created_at.isoformat(),
                'priority': job.priority,
                'result': {
                    'success': job.result.success if job.result else False,
                    'error_message': job.result.error_message if job.result else None,
                    'response_time': job.result.response_time if job.result else 0.0
                } if job.result else None
            }

        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics (legacy compatibility)."""
        return self.core_engine.get_engine_metrics()

    def get_registered_scrapers(self) -> List[str]:
        """Get list of registered scraper types."""
        return list(self.scrapers.keys())

    def get_scraper_metrics(self, scraper_type: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific scraper."""
        scraper = self.active_scrapers.get(scraper_type)
        if scraper:
            return scraper.get_metrics()
        return None

    async def cleanup(self) -> None:
        """Cleanup engine resources."""
        await self.core_engine.cleanup()

    def __str__(self) -> str:
        return str(self.core_engine)
