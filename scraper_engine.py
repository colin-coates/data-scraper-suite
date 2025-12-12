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
    """Centralized orchestration engine for web scraping operations."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.logger = logging.getLogger(f"{__name__}.ScraperEngine")

        # Component registries
        self.scrapers: Dict[str, Type[BaseScraper]] = {}
        self.active_scrapers: Dict[str, BaseScraper] = {}

        # Job management
        self.job_queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=self.config.job_queue_size)
        self.active_jobs: Dict[str, Job] = {}
        self.completed_jobs: List[Job] = []

        # Azure clients
        self.service_bus_client: Optional[AsyncServiceBusClient] = None
        self.blob_service_client: Optional[BlobServiceClient] = None

        # Queue publisher for results
        self.queue_publisher: Optional[QueuePublisher] = None

        # Anti-detection layer
        self.anti_detection = AntiDetectionLayer() if self.config.enable_anti_detection else None

        # Control flags
        self.running = False
        self.shutdown_event = asyncio.Event()

        # Metrics
        self.metrics = {
            'jobs_processed': 0,
            'jobs_succeeded': 0,
            'jobs_failed': 0,
            'total_processing_time': 0.0,
            'scraper_usage': {},
            'start_time': datetime.utcnow()
        }

        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_jobs)

        self.logger.info("ScraperEngine initialized")

    async def initialize(self) -> None:
        """Initialize the scraper engine and its components."""
        try:
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

            # Initialize queue publisher if enabled
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

            self.logger.info("ScraperEngine initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize ScraperEngine: {e}")
            raise

    def register_scraper(self, scraper_type: str, scraper_class: Type[BaseScraper],
                        config: Optional[ScraperConfig] = None) -> None:
        """
        Register a scraper class with the engine.

        Args:
            scraper_type: Unique identifier for the scraper type
            scraper_class: The scraper class to register
            config: Optional default configuration for the scraper
        """
        if scraper_type in self.scrapers:
            self.logger.warning(f"Scraper type '{scraper_type}' already registered, overwriting")

        self.scrapers[scraper_type] = scraper_class

        # Create default config if not provided
        if config is None:
            config = ScraperConfig(
                name=scraper_type,
                rate_limit_delay=self.config.default_rate_limit
            )

        # Instantiate the scraper
        scraper_instance = scraper_class(config)

        # Attach anti-detection if available
        if self.anti_detection:
            scraper_instance.set_anti_detection(self.anti_detection)

        self.active_scrapers[scraper_type] = scraper_instance

        self.logger.info(f"Registered scraper: {scraper_type}")

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
        Dispatch a scraping job to the engine.

        Args:
            job_data: Job specification including scraper_type, target, etc.

        Returns:
            Job ID for tracking
        """
        job = Job(
            job_id=f"job_{int(time.time() * 1000)}_{hash(str(job_data)) % 10000}",
            scraper_type=job_data.get('scraper_type', ''),
            target=job_data.get('target', {}),
            priority=job_data.get('priority', 'normal'),
            metadata=job_data.get('metadata', {})
        )

        # Validate job
        if not job.scraper_type or job.scraper_type not in self.active_scrapers:
            raise ValueError(f"Unknown or unregistered scraper type: {job.scraper_type}")

        # Add to queue
        try:
            self.job_queue.put_nowait(job)
            self.active_jobs[job.job_id] = job
            self.logger.info(f"Job dispatched: {job.job_id} ({job.scraper_type})")
            return job.job_id

        except asyncio.QueueFull:
            raise RuntimeError("Job queue is full")

    async def start(self) -> None:
        """Start the scraper engine processing loop."""
        if self.running:
            self.logger.warning("Engine already running")
            return

        self.running = True
        self.shutdown_event.clear()

        self.logger.info("Starting ScraperEngine with "
                        f"{self.config.max_concurrent_jobs} concurrent jobs")

        try:
            # Start job processors
            processors = [
                asyncio.create_task(self._job_processor())
                for _ in range(self.config.max_concurrent_jobs)
            ]

            # Start queue monitor
            monitor = asyncio.create_task(self._queue_monitor())

            # Wait for shutdown
            await self.shutdown_event.wait()

            # Cancel processors
            for processor in processors:
                processor.cancel()
            for monitor_task in [monitor]:
                monitor_task.cancel()

            # Wait for cleanup
            await asyncio.gather(*processors, *monitor_task, return_exceptions=True)

        except Exception as e:
            self.logger.error(f"Engine error: {e}")
        finally:
            self.running = False
            self.logger.info("ScraperEngine stopped")

    async def stop(self) -> None:
        """Stop the scraper engine gracefully."""
        self.logger.info("Stopping ScraperEngine...")
        self.shutdown_event.set()

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
        """Get engine performance metrics."""
        uptime = (datetime.utcnow() - self.metrics['start_time']).total_seconds()

        return {
            'uptime_seconds': uptime,
            'jobs_processed': self.metrics['jobs_processed'],
            'jobs_succeeded': self.metrics['jobs_succeeded'],
            'jobs_failed': self.metrics['jobs_failed'],
            'success_rate': self.metrics['jobs_succeeded'] / max(1, self.metrics['jobs_processed']),
            'average_processing_time': self.metrics['total_processing_time'] / max(1, self.metrics['jobs_processed']),
            'active_jobs': len(self.active_jobs),
            'queued_jobs': self.job_queue.qsize(),
            'completed_jobs': len(self.completed_jobs),
            'scraper_usage': self.metrics['scraper_usage'],
            'registered_scrapers': list(self.scrapers.keys()),
            'active_scrapers': list(self.active_scrapers.keys()),
            'anti_detection_enabled': self.anti_detection is not None,
            'result_publishing_enabled': self.config.enable_result_publishing,
            'queue_publisher_metrics': self.queue_publisher.get_metrics() if self.queue_publisher else None
        }

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
        self.logger.info("Cleaning up ScraperEngine...")

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

        self.logger.info("ScraperEngine cleanup complete")

    def __str__(self) -> str:
        return f"ScraperEngine(jobs_processed={self.metrics['jobs_processed']}, " \
               f"active_jobs={len(self.active_jobs)}, " \
               f"registered_scrapers={len(self.scrapers)})"
