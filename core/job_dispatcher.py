# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Async Job Dispatcher for MJ Data Scraper Suite

Provides intelligent job queuing, prioritization, and concurrent execution
with load balancing and resource management.
"""

import asyncio
import heapq
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass(order=True)
class PrioritizedJob:
    """Job with priority ordering."""
    priority_value: int
    created_at: float
    job: Any = field(compare=False)


class JobDispatcher:
    """
    Async job dispatcher with priority queuing and intelligent resource management.
    """

    def __init__(self,
                 max_concurrent: int = 10,
                 queue_size: int = 1000,
                 enable_priority: bool = True):
        self.max_concurrent = max_concurrent
        self.queue_size = queue_size
        self.enable_priority = enable_priority

        # Priority queue for jobs
        self.job_queue: List[PrioritizedJob] = []
        self.job_map: Dict[str, PrioritizedJob] = {}

        # Execution control
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.completed_jobs: Dict[str, Any] = {}
        self.failed_jobs: Dict[str, Any] = {}

        # Semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Control events
        self.running = False
        self.shutdown_event = asyncio.Event()

        # Callbacks
        self.on_job_started: Optional[Callable[[str, Any], None]] = None
        self.on_job_completed: Optional[Callable[[str, Any, Any], None]] = None
        self.on_job_failed: Optional[Callable[[str, Any, Exception], None]] = None
        self.on_queue_full: Optional[Callable[[], None]] = None

        # Metrics
        self.metrics = {
            'jobs_dispatched': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'total_queue_time': 0.0,
            'total_execution_time': 0.0,
            'priority_distribution': {p.name: 0 for p in Priority}
        }

        logger.info(f"JobDispatcher initialized with max_concurrent={max_concurrent}")

    def dispatch(self, job_id: str, job_data: Any, priority: Priority = Priority.NORMAL) -> bool:
        """
        Dispatch a job to the queue.

        Args:
            job_id: Unique job identifier
            job_data: Job data/payload
            priority: Job priority level

        Returns:
            True if job was queued successfully
        """
        if not self.running:
            raise RuntimeError("Dispatcher is not running")

        if job_id in self.job_map:
            logger.warning(f"Job {job_id} already exists in queue")
            return False

        if len(self.job_queue) >= self.queue_size:
            if self.on_queue_full:
                self.on_queue_full()
            logger.warning("Job queue is full")
            return False

        # Create prioritized job
        priority_value = priority.value if self.enable_priority else Priority.NORMAL.value
        created_at = time.time()

        prioritized_job = PrioritizedJob(
            priority_value=priority_value,
            created_at=created_at,
            job={'job_id': job_id, 'data': job_data, 'priority': priority.name}
        )

        # Add to queue and map
        heapq.heappush(self.job_queue, prioritized_job)
        self.job_map[job_id] = prioritized_job

        # Update metrics
        self.metrics['jobs_dispatched'] += 1
        self.metrics['priority_distribution'][priority.name] += 1

        logger.info(f"Job dispatched: {job_id} (priority: {priority.name})")
        return True

    async def start(self) -> None:
        """Start the job dispatcher."""
        if self.running:
            logger.warning("Dispatcher already running")
            return

        self.running = True
        self.shutdown_event.clear()

        logger.info("Starting JobDispatcher")

        try:
            # Start worker tasks
            workers = [
                asyncio.create_task(self._worker())
                for _ in range(self.max_concurrent)
            ]

            # Start queue monitor
            monitor = asyncio.create_task(self._monitor())

            # Wait for shutdown
            await self.shutdown_event.wait()

            # Cancel workers
            for worker in workers:
                worker.cancel()
            monitor.cancel()

            # Wait for cleanup
            await asyncio.gather(*workers, monitor, return_exceptions=True)

        except Exception as e:
            logger.error(f"Dispatcher error: {e}")
        finally:
            self.running = False
            logger.info("JobDispatcher stopped")

    async def stop(self) -> None:
        """Stop the job dispatcher gracefully."""
        logger.info("Stopping JobDispatcher...")
        self.shutdown_event.set()

    async def _worker(self) -> None:
        """Worker task that processes jobs."""
        while not self.shutdown_event.is_set():
            try:
                # Acquire semaphore
                async with self.semaphore:
                    # Get next job
                    job = await self._get_next_job()
                    if not job:
                        await asyncio.sleep(0.1)  # Small delay when no jobs
                        continue

                    job_id = job['job_id']
                    job_data = job['data']

                    # Execute job
                    await self._execute_job(job_id, job_data)

            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)

    async def _get_next_job(self) -> Optional[Dict[str, Any]]:
        """Get the next job from the priority queue."""
        try:
            # Try to get job with timeout
            prioritized_job = await asyncio.wait_for(
                self._async_pop_job(),
                timeout=0.1
            )

            job = prioritized_job.job
            job_id = job['job_id']

            # Remove from map
            self.job_map.pop(job_id, None)

            return job

        except asyncio.TimeoutError:
            return None

    async def _async_pop_job(self) -> PrioritizedJob:
        """Async wrapper for heap pop operation."""
        # Run in thread pool since heapq operations are not async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, heapq.heappop, self.job_queue)

    async def _execute_job(self, job_id: str, job_data: Any) -> None:
        """Execute a single job."""
        start_time = time.time()

        # Notify job started
        if self.on_job_started:
            self.on_job_started(job_id, job_data)

        try:
            # Extract execution function and args
            if isinstance(job_data, dict) and 'func' in job_data:
                func = job_data['func']
                args = job_data.get('args', ())
                kwargs = job_data.get('kwargs', {})

                # Execute the job function
                result = await func(*args, **kwargs)
            else:
                # Assume job_data is a callable
                result = await job_data()

            # Record completion
            execution_time = time.time() - start_time
            self.completed_jobs[job_id] = {
                'result': result,
                'execution_time': execution_time,
                'completed_at': datetime.utcnow()
            }

            self.metrics['jobs_completed'] += 1
            self.metrics['total_execution_time'] += execution_time

            # Notify completion
            if self.on_job_completed:
                self.on_job_completed(job_id, job_data, result)

            logger.info(f"Job completed: {job_id} in {execution_time:.2f}s")

        except Exception as e:
            execution_time = time.time() - start_time
            self.failed_jobs[job_id] = {
                'error': str(e),
                'execution_time': execution_time,
                'failed_at': datetime.utcnow()
            }

            self.metrics['jobs_failed'] += 1

            # Notify failure
            if self.on_job_failed:
                self.on_job_failed(job_id, job_data, e)

            logger.error(f"Job failed: {job_id} after {execution_time:.2f}s: {e}")

        finally:
            # Remove from active jobs
            self.active_jobs.pop(job_id, None)

    async def _monitor(self) -> None:
        """Monitor queue and job status."""
        while not self.shutdown_event.is_set():
            try:
                queue_size = len(self.job_queue)
                active_count = len(self.active_jobs)

                logger.debug(f"Dispatcher status: {queue_size} queued, "
                           f"{active_count} active, "
                           f"{len(self.completed_jobs)} completed, "
                           f"{len(self.failed_jobs)} failed")

                # Check for stuck jobs (basic health check)
                stuck_jobs = []
                for job_id, job in self.active_jobs.items():
                    # This is a simple check - in production you'd want more sophisticated monitoring
                    pass

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(5)

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        if job_id in self.active_jobs:
            return {'status': 'running'}
        elif job_id in self.completed_jobs:
            return {
                'status': 'completed',
                **self.completed_jobs[job_id]
            }
        elif job_id in self.failed_jobs:
            return {
                'status': 'failed',
                **self.failed_jobs[job_id]
            }
        elif job_id in self.job_map:
            return {'status': 'queued'}
        else:
            return None

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            'queue_size': len(self.job_queue),
            'active_jobs': len(self.active_jobs),
            'completed_jobs': len(self.completed_jobs),
            'failed_jobs': len(self.failed_jobs),
            'max_concurrent': self.max_concurrent,
            'queue_capacity': self.queue_size,
            'queue_utilization': len(self.job_queue) / self.queue_size
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get dispatcher performance metrics."""
        total_jobs = self.metrics['jobs_completed'] + self.metrics['jobs_failed']
        success_rate = self.metrics['jobs_completed'] / max(1, total_jobs)
        avg_execution_time = self.metrics['total_execution_time'] / max(1, self.metrics['jobs_completed'])

        return {
            'jobs_dispatched': self.metrics['jobs_dispatched'],
            'jobs_completed': self.metrics['jobs_completed'],
            'jobs_failed': self.metrics['jobs_failed'],
            'success_rate': success_rate,
            'average_execution_time': avg_execution_time,
            'priority_distribution': self.metrics['priority_distribution'],
            'queue_stats': self.get_queue_stats()
        }

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job."""
        if job_id in self.job_map:
            prioritized_job = self.job_map[job_id]
            # Note: Removing from heapq is not straightforward
            # In production, you'd want a better data structure
            self.job_map.pop(job_id, None)
            logger.info(f"Job cancelled: {job_id}")
            return True

        logger.warning(f"Cannot cancel job {job_id}: not found in queue")
        return False

    def clear_completed_jobs(self, max_age_seconds: int = 3600) -> int:
        """Clear old completed jobs from memory."""
        cutoff_time = time.time() - max_age_seconds
        to_remove = []

        for job_id, job_info in self.completed_jobs.items():
            if job_info.get('completed_at', datetime.min).timestamp() < cutoff_time:
                to_remove.append(job_id)

        for job_id in to_remove:
            del self.completed_jobs[job_id]

        logger.info(f"Cleared {len(to_remove)} old completed jobs")
        return len(to_remove)

    async def wait_for_completion(self, job_id: str, timeout: float = 30.0) -> Optional[Any]:
        """
        Wait for a job to complete.

        Args:
            job_id: Job to wait for
            timeout: Maximum wait time in seconds

        Returns:
            Job result if completed successfully, None otherwise
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)

            if status and status['status'] == 'completed':
                return self.completed_jobs[job_id]['result']
            elif status and status['status'] == 'failed':
                return None

            await asyncio.sleep(0.1)

        logger.warning(f"Timeout waiting for job {job_id}")
        return None
