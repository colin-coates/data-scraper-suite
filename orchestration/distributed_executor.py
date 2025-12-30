# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Distributed Execution for MJ Data Scraper Suite

Provides multi-region scraping capabilities:
- Worker pool management
- Geographic distribution
- Load balancing
- Failover handling
- Result aggregation
"""

import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Awaitable
from enum import Enum
import random

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Status of a worker node."""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    DRAINING = "draining"  # Finishing current work, not accepting new


class Region(Enum):
    """Available regions for distributed execution."""
    US_EAST = "us-east"
    US_WEST = "us-west"
    EU_WEST = "eu-west"
    EU_CENTRAL = "eu-central"
    ASIA_EAST = "asia-east"
    ASIA_SOUTH = "asia-south"


@dataclass
class WorkerNode:
    """Represents a worker node in the distributed system."""
    worker_id: str
    region: Region
    endpoint: str
    status: WorkerStatus = WorkerStatus.IDLE
    
    # Capacity
    max_concurrent: int = 5
    current_jobs: int = 0
    
    # Performance metrics
    jobs_completed: int = 0
    jobs_failed: int = 0
    avg_response_time_ms: float = 0
    
    # Health
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    consecutive_failures: int = 0
    
    # Metadata
    capabilities: List[str] = field(default_factory=list)  # e.g., ["playwright", "captcha"]
    
    @property
    def available_capacity(self) -> int:
        """Get available job slots."""
        return max(0, self.max_concurrent - self.current_jobs)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.jobs_completed + self.jobs_failed
        if total == 0:
            return 1.0
        return self.jobs_completed / total
    
    @property
    def is_healthy(self) -> bool:
        """Check if worker is healthy."""
        if self.status == WorkerStatus.OFFLINE:
            return False
        if self.consecutive_failures >= 3:
            return False
        if (datetime.utcnow() - self.last_heartbeat).seconds > 60:
            return False
        return True


@dataclass
class DistributedJob:
    """A job distributed across workers."""
    job_id: str
    tasks: List[Dict[str, Any]]
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Distribution
    assigned_workers: Dict[str, List[int]] = field(default_factory=dict)  # worker_id -> task indices
    
    # Results
    completed_tasks: int = 0
    failed_tasks: int = 0
    results: Dict[int, Any] = field(default_factory=dict)  # task_index -> result
    errors: Dict[int, str] = field(default_factory=dict)  # task_index -> error
    
    # Status
    status: str = "pending"  # pending, running, completed, failed
    completed_at: Optional[datetime] = None
    
    @property
    def total_tasks(self) -> int:
        return len(self.tasks)
    
    @property
    def progress(self) -> float:
        if self.total_tasks == 0:
            return 1.0
        return (self.completed_tasks + self.failed_tasks) / self.total_tasks


class LoadBalancer:
    """
    Load balancer for distributing work across workers.
    
    Strategies:
    - Round robin
    - Least connections
    - Weighted (by success rate)
    - Geographic affinity
    """

    def __init__(self, strategy: str = "weighted"):
        """
        Initialize load balancer.

        Args:
            strategy: Balancing strategy (round_robin, least_conn, weighted, geo)
        """
        self.strategy = strategy
        self._round_robin_index = 0

    def select_worker(
        self,
        workers: List[WorkerNode],
        task: Dict[str, Any],
        preferred_region: Optional[Region] = None
    ) -> Optional[WorkerNode]:
        """
        Select best worker for a task.

        Args:
            workers: Available workers
            task: Task to assign
            preferred_region: Preferred region for task

        Returns:
            Selected worker or None
        """
        # Filter to healthy workers with capacity
        available = [
            w for w in workers
            if w.is_healthy and w.available_capacity > 0 and w.status != WorkerStatus.DRAINING
        ]
        
        if not available:
            return None
        
        # Check for required capabilities
        required_caps = task.get("required_capabilities", [])
        if required_caps:
            available = [
                w for w in available
                if all(cap in w.capabilities for cap in required_caps)
            ]
        
        if not available:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin(available)
        elif self.strategy == "least_conn":
            return self._least_connections(available)
        elif self.strategy == "weighted":
            return self._weighted(available)
        elif self.strategy == "geo":
            return self._geographic(available, preferred_region)
        else:
            return self._weighted(available)

    def _round_robin(self, workers: List[WorkerNode]) -> WorkerNode:
        """Round robin selection."""
        worker = workers[self._round_robin_index % len(workers)]
        self._round_robin_index += 1
        return worker

    def _least_connections(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with fewest current jobs."""
        return min(workers, key=lambda w: w.current_jobs)

    def _weighted(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select based on success rate and capacity."""
        def score(w: WorkerNode) -> float:
            # Higher score = better choice
            capacity_score = w.available_capacity / w.max_concurrent
            success_score = w.success_rate
            speed_score = 1.0 / (1.0 + w.avg_response_time_ms / 1000)
            return capacity_score * 0.3 + success_score * 0.5 + speed_score * 0.2
        
        return max(workers, key=score)

    def _geographic(
        self,
        workers: List[WorkerNode],
        preferred_region: Optional[Region]
    ) -> WorkerNode:
        """Select based on geographic affinity."""
        if preferred_region:
            regional = [w for w in workers if w.region == preferred_region]
            if regional:
                return self._weighted(regional)
        
        return self._weighted(workers)


class DistributedExecutor:
    """
    Manages distributed execution of scraping jobs.
    
    Features:
    - Worker pool management
    - Job distribution
    - Result aggregation
    - Automatic failover
    - Health monitoring
    """

    def __init__(
        self,
        load_balancer: Optional[LoadBalancer] = None,
        max_retries: int = 3,
        task_timeout: int = 300
    ):
        """
        Initialize distributed executor.

        Args:
            load_balancer: Load balancer instance
            max_retries: Max retries per task
            task_timeout: Task timeout in seconds
        """
        self.load_balancer = load_balancer or LoadBalancer()
        self.max_retries = max_retries
        self.task_timeout = task_timeout
        
        # Worker pool
        self.workers: Dict[str, WorkerNode] = {}
        
        # Jobs
        self.jobs: Dict[str, DistributedJob] = {}
        
        # Task executor (to be set by scraper engine)
        self.task_executor: Optional[Callable[[Dict[str, Any]], Awaitable[Any]]] = None
        
        # Background tasks
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

    def register_worker(
        self,
        endpoint: str,
        region: Region,
        max_concurrent: int = 5,
        capabilities: Optional[List[str]] = None
    ) -> WorkerNode:
        """
        Register a new worker node.

        Args:
            endpoint: Worker endpoint URL
            region: Worker region
            max_concurrent: Max concurrent jobs
            capabilities: Worker capabilities

        Returns:
            Registered WorkerNode
        """
        worker_id = str(uuid.uuid4())
        
        worker = WorkerNode(
            worker_id=worker_id,
            region=region,
            endpoint=endpoint,
            max_concurrent=max_concurrent,
            capabilities=capabilities or []
        )
        
        self.workers[worker_id] = worker
        logger.info(f"Registered worker {worker_id} in {region.value}")
        
        return worker

    def deregister_worker(self, worker_id: str) -> bool:
        """Remove a worker from the pool."""
        if worker_id not in self.workers:
            return False
        
        worker = self.workers[worker_id]
        worker.status = WorkerStatus.DRAINING
        
        # Wait for current jobs to complete (in production)
        del self.workers[worker_id]
        logger.info(f"Deregistered worker {worker_id}")
        return True

    def update_worker_heartbeat(self, worker_id: str) -> bool:
        """Update worker heartbeat."""
        if worker_id not in self.workers:
            return False
        
        self.workers[worker_id].last_heartbeat = datetime.utcnow()
        return True

    async def submit_job(
        self,
        tasks: List[Dict[str, Any]],
        preferred_region: Optional[Region] = None
    ) -> DistributedJob:
        """
        Submit a distributed job.

        Args:
            tasks: List of tasks to execute
            preferred_region: Preferred region for execution

        Returns:
            DistributedJob instance
        """
        job_id = str(uuid.uuid4())
        
        job = DistributedJob(
            job_id=job_id,
            tasks=tasks
        )
        
        self.jobs[job_id] = job
        
        # Distribute tasks
        await self._distribute_tasks(job, preferred_region)
        
        return job

    async def _distribute_tasks(
        self,
        job: DistributedJob,
        preferred_region: Optional[Region] = None
    ) -> None:
        """Distribute tasks across workers."""
        job.status = "running"
        workers = list(self.workers.values())
        
        for i, task in enumerate(job.tasks):
            worker = self.load_balancer.select_worker(
                workers, task, preferred_region
            )
            
            if worker:
                if worker.worker_id not in job.assigned_workers:
                    job.assigned_workers[worker.worker_id] = []
                job.assigned_workers[worker.worker_id].append(i)
                worker.current_jobs += 1
            else:
                # No worker available, queue for later
                logger.warning(f"No worker available for task {i} in job {job.job_id}")

    async def execute_job(
        self,
        job: DistributedJob,
        callback: Optional[Callable[[int, Any], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute a distributed job.

        Args:
            job: Job to execute
            callback: Optional callback for each completed task

        Returns:
            Aggregated results
        """
        if not self.task_executor:
            raise RuntimeError("Task executor not set")
        
        tasks_to_run = []
        
        for worker_id, task_indices in job.assigned_workers.items():
            worker = self.workers.get(worker_id)
            if not worker:
                continue
            
            for task_idx in task_indices:
                task = job.tasks[task_idx]
                tasks_to_run.append(
                    self._execute_task(job, task_idx, task, worker, callback)
                )
        
        # Execute all tasks concurrently
        await asyncio.gather(*tasks_to_run, return_exceptions=True)
        
        # Update job status
        if job.failed_tasks == 0:
            job.status = "completed"
        elif job.completed_tasks == 0:
            job.status = "failed"
        else:
            job.status = "partial"
        
        job.completed_at = datetime.utcnow()
        
        return {
            "job_id": job.job_id,
            "status": job.status,
            "total_tasks": job.total_tasks,
            "completed": job.completed_tasks,
            "failed": job.failed_tasks,
            "results": job.results,
            "errors": job.errors,
            "duration_seconds": (job.completed_at - job.created_at).total_seconds()
        }

    async def _execute_task(
        self,
        job: DistributedJob,
        task_idx: int,
        task: Dict[str, Any],
        worker: WorkerNode,
        callback: Optional[Callable[[int, Any], None]] = None
    ) -> None:
        """Execute a single task on a worker."""
        retries = 0
        
        while retries <= self.max_retries:
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self.task_executor(task),
                    timeout=self.task_timeout
                )
                
                # Success
                job.results[task_idx] = result
                job.completed_tasks += 1
                worker.jobs_completed += 1
                worker.consecutive_failures = 0
                
                if callback:
                    callback(task_idx, result)
                
                break
                
            except asyncio.TimeoutError:
                retries += 1
                logger.warning(f"Task {task_idx} timed out (retry {retries})")
                
            except Exception as e:
                retries += 1
                logger.error(f"Task {task_idx} failed: {e} (retry {retries})")
                
                if retries > self.max_retries:
                    job.errors[task_idx] = str(e)
                    job.failed_tasks += 1
                    worker.jobs_failed += 1
                    worker.consecutive_failures += 1
        
        # Release worker capacity
        worker.current_jobs = max(0, worker.current_jobs - 1)

    async def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Started distributed executor monitoring")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped distributed executor monitoring")

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                await self._check_worker_health()
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(5)

    async def _check_worker_health(self) -> None:
        """Check health of all workers."""
        now = datetime.utcnow()
        
        for worker in self.workers.values():
            # Check heartbeat
            if (now - worker.last_heartbeat).seconds > 60:
                if worker.status != WorkerStatus.OFFLINE:
                    worker.status = WorkerStatus.OFFLINE
                    logger.warning(f"Worker {worker.worker_id} marked offline (no heartbeat)")
            
            # Check consecutive failures
            if worker.consecutive_failures >= 3:
                if worker.status != WorkerStatus.OFFLINE:
                    worker.status = WorkerStatus.OFFLINE
                    logger.warning(f"Worker {worker.worker_id} marked offline (too many failures)")

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a job."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "total_tasks": job.total_tasks,
            "completed_tasks": job.completed_tasks,
            "failed_tasks": job.failed_tasks,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None
        }

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get statistics for all workers."""
        total_capacity = sum(w.max_concurrent for w in self.workers.values())
        used_capacity = sum(w.current_jobs for w in self.workers.values())
        healthy_workers = sum(1 for w in self.workers.values() if w.is_healthy)
        
        by_region = defaultdict(lambda: {"count": 0, "capacity": 0, "healthy": 0})
        for worker in self.workers.values():
            region = worker.region.value
            by_region[region]["count"] += 1
            by_region[region]["capacity"] += worker.max_concurrent
            if worker.is_healthy:
                by_region[region]["healthy"] += 1
        
        return {
            "total_workers": len(self.workers),
            "healthy_workers": healthy_workers,
            "total_capacity": total_capacity,
            "used_capacity": used_capacity,
            "utilization": used_capacity / max(1, total_capacity),
            "by_region": dict(by_region),
            "workers": [
                {
                    "worker_id": w.worker_id,
                    "region": w.region.value,
                    "status": w.status.value,
                    "current_jobs": w.current_jobs,
                    "max_concurrent": w.max_concurrent,
                    "success_rate": w.success_rate,
                    "is_healthy": w.is_healthy
                }
                for w in self.workers.values()
            ]
        }


# Global instance
distributed_executor = DistributedExecutor()
