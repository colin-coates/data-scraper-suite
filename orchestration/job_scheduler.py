# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Job Scheduler for MJ Data Scraper Suite

Provides scheduled and recurring scraping job management:
- Cron-based scheduling
- One-time scheduled jobs
- Recurring jobs with intervals
- Job queuing and prioritization
- Failure retry with backoff
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import json
import heapq

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a scheduled job."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class ScheduleType(Enum):
    """Types of job schedules."""
    ONCE = "once"  # Run once at specified time
    INTERVAL = "interval"  # Run every N seconds/minutes/hours
    CRON = "cron"  # Cron expression
    DAILY = "daily"  # Run daily at specified time
    WEEKLY = "weekly"  # Run weekly on specified days


@dataclass
class JobSchedule:
    """Schedule configuration for a job."""
    schedule_type: ScheduleType
    
    # For ONCE: specific datetime
    run_at: Optional[datetime] = None
    
    # For INTERVAL: interval in seconds
    interval_seconds: int = 3600
    
    # For CRON: cron expression (minute hour day month weekday)
    cron_expression: Optional[str] = None
    
    # For DAILY: time of day (HH:MM)
    time_of_day: Optional[str] = None
    
    # For WEEKLY: days of week (0=Monday, 6=Sunday)
    days_of_week: List[int] = field(default_factory=list)
    
    # Timezone
    timezone: str = "UTC"
    
    # Max runs (0 = unlimited)
    max_runs: int = 0
    
    # End date for recurring jobs
    end_date: Optional[datetime] = None


@dataclass
class ScheduledJob:
    """A scheduled scraping job."""
    job_id: str
    name: str
    scraper_type: str
    target: Dict[str, Any]
    schedule: JobSchedule
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    
    # Execution tracking
    run_count: int = 0
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    last_result: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    
    # Retry configuration
    max_retries: int = 3
    retry_count: int = 0
    retry_delay_seconds: int = 60
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Options
    options: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """Compare jobs for priority queue (higher priority first, then earlier time)."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        if self.next_run and other.next_run:
            return self.next_run < other.next_run
        return False


class JobScheduler:
    """
    Manages scheduled scraping jobs.
    
    Features:
    - Multiple schedule types (once, interval, cron, daily, weekly)
    - Priority-based execution
    - Automatic retry with exponential backoff
    - Job persistence
    - Concurrent job limits
    """

    def __init__(
        self,
        max_concurrent_jobs: int = 5,
        job_executor: Optional[Callable[[ScheduledJob], Awaitable[Dict[str, Any]]]] = None,
        persistence_path: Optional[str] = None
    ):
        """
        Initialize job scheduler.

        Args:
            max_concurrent_jobs: Maximum jobs to run concurrently
            job_executor: Async function to execute jobs
            persistence_path: Path to persist job state
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_executor = job_executor
        self.persistence_path = persistence_path
        
        # Job storage
        self.jobs: Dict[str, ScheduledJob] = {}
        
        # Priority queue for next runs
        self._job_queue: List[ScheduledJob] = []
        
        # Currently running jobs
        self._running_jobs: Dict[str, asyncio.Task] = {}
        
        # Scheduler state
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # Event hooks
        self.on_job_start: Optional[Callable[[ScheduledJob], None]] = None
        self.on_job_complete: Optional[Callable[[ScheduledJob, Dict[str, Any]], None]] = None
        self.on_job_error: Optional[Callable[[ScheduledJob, Exception], None]] = None

    def create_job(
        self,
        name: str,
        scraper_type: str,
        target: Dict[str, Any],
        schedule: JobSchedule,
        priority: JobPriority = JobPriority.NORMAL,
        max_retries: int = 3,
        options: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None
    ) -> ScheduledJob:
        """
        Create a new scheduled job.

        Args:
            name: Human-readable job name
            scraper_type: Type of scraper to use
            target: Scraping target configuration
            schedule: Job schedule configuration
            priority: Job priority
            max_retries: Max retry attempts on failure
            options: Additional scraper options
            tags: Job tags for filtering
            created_by: User who created the job

        Returns:
            The created ScheduledJob
        """
        job_id = str(uuid.uuid4())
        
        job = ScheduledJob(
            job_id=job_id,
            name=name,
            scraper_type=scraper_type,
            target=target,
            schedule=schedule,
            priority=priority,
            max_retries=max_retries,
            options=options or {},
            tags=tags or [],
            created_by=created_by
        )
        
        # Calculate next run time
        job.next_run = self._calculate_next_run(job)
        
        # Store job
        self.jobs[job_id] = job
        
        # Add to queue if has next run
        if job.next_run:
            heapq.heappush(self._job_queue, job)
        
        logger.info(f"Created job {job_id}: {name} (next run: {job.next_run})")
        
        self._persist_jobs()
        return job

    def _calculate_next_run(self, job: ScheduledJob) -> Optional[datetime]:
        """Calculate the next run time for a job."""
        now = datetime.utcnow()
        schedule = job.schedule
        
        # Check if max runs reached
        if schedule.max_runs > 0 and job.run_count >= schedule.max_runs:
            return None
        
        # Check if past end date
        if schedule.end_date and now > schedule.end_date:
            return None
        
        if schedule.schedule_type == ScheduleType.ONCE:
            if job.run_count == 0 and schedule.run_at and schedule.run_at > now:
                return schedule.run_at
            return None
        
        elif schedule.schedule_type == ScheduleType.INTERVAL:
            if job.last_run:
                return job.last_run + timedelta(seconds=schedule.interval_seconds)
            return now + timedelta(seconds=1)  # Start soon
        
        elif schedule.schedule_type == ScheduleType.DAILY:
            if schedule.time_of_day:
                hour, minute = map(int, schedule.time_of_day.split(":"))
                next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
                return next_run
            return None
        
        elif schedule.schedule_type == ScheduleType.WEEKLY:
            if schedule.time_of_day and schedule.days_of_week:
                hour, minute = map(int, schedule.time_of_day.split(":"))
                
                # Find next matching day
                for days_ahead in range(8):
                    check_date = now + timedelta(days=days_ahead)
                    if check_date.weekday() in schedule.days_of_week:
                        next_run = check_date.replace(
                            hour=hour, minute=minute, second=0, microsecond=0
                        )
                        if next_run > now:
                            return next_run
                return None
            return None
        
        elif schedule.schedule_type == ScheduleType.CRON:
            # Simple cron parsing (minute hour day month weekday)
            if schedule.cron_expression:
                return self._parse_cron_next(schedule.cron_expression, now)
            return None
        
        return None

    def _parse_cron_next(self, cron_expr: str, after: datetime) -> Optional[datetime]:
        """
        Parse cron expression and find next run time.
        
        Simplified cron format: minute hour day month weekday
        Supports: *, specific values, ranges (1-5), lists (1,3,5)
        """
        try:
            parts = cron_expr.split()
            if len(parts) != 5:
                return None
            
            minute, hour, day, month, weekday = parts
            
            # Simple implementation - check next 366 days
            check_time = after + timedelta(minutes=1)
            check_time = check_time.replace(second=0, microsecond=0)
            
            for _ in range(366 * 24 * 60):  # Max 1 year of minutes
                if self._cron_matches(check_time, minute, hour, day, month, weekday):
                    return check_time
                check_time += timedelta(minutes=1)
            
            return None
        except Exception:
            return None

    def _cron_matches(
        self,
        dt: datetime,
        minute: str,
        hour: str,
        day: str,
        month: str,
        weekday: str
    ) -> bool:
        """Check if datetime matches cron expression."""
        def matches_field(value: int, field: str) -> bool:
            if field == "*":
                return True
            if field.isdigit():
                return value == int(field)
            if "-" in field:
                start, end = map(int, field.split("-"))
                return start <= value <= end
            if "," in field:
                return value in map(int, field.split(","))
            return False
        
        return (
            matches_field(dt.minute, minute) and
            matches_field(dt.hour, hour) and
            matches_field(dt.day, day) and
            matches_field(dt.month, month) and
            matches_field(dt.weekday(), weekday)
        )

    async def start(self) -> None:
        """Start the job scheduler."""
        if self._running:
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Job scheduler started")

    async def stop(self) -> None:
        """Stop the job scheduler."""
        self._running = False
        
        # Cancel running jobs
        for task in self._running_jobs.values():
            task.cancel()
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        self._persist_jobs()
        logger.info("Job scheduler stopped")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                now = datetime.utcnow()
                
                # Check for jobs ready to run
                while self._job_queue and len(self._running_jobs) < self.max_concurrent_jobs:
                    # Peek at next job
                    if not self._job_queue:
                        break
                    
                    next_job = self._job_queue[0]
                    
                    if next_job.next_run and next_job.next_run <= now:
                        # Pop and run
                        heapq.heappop(self._job_queue)
                        
                        if next_job.status not in (JobStatus.CANCELLED, JobStatus.PAUSED):
                            await self._run_job(next_job)
                    else:
                        break
                
                # Wait before next check
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(5)

    async def _run_job(self, job: ScheduledJob) -> None:
        """Execute a job."""
        if job.job_id in self._running_jobs:
            return
        
        job.status = JobStatus.RUNNING
        job.updated_at = datetime.utcnow()
        
        if self.on_job_start:
            self.on_job_start(job)
        
        logger.info(f"Starting job {job.job_id}: {job.name}")
        
        # Create task
        task = asyncio.create_task(self._execute_job(job))
        self._running_jobs[job.job_id] = task

    async def _execute_job(self, job: ScheduledJob) -> None:
        """Execute job and handle result."""
        try:
            if self.job_executor:
                result = await self.job_executor(job)
            else:
                # Default executor - just log
                result = {"status": "executed", "job_id": job.job_id}
                await asyncio.sleep(1)
            
            # Success
            job.status = JobStatus.COMPLETED
            job.last_result = result
            job.last_error = None
            job.retry_count = 0
            job.run_count += 1
            job.last_run = datetime.utcnow()
            
            if self.on_job_complete:
                self.on_job_complete(job, result)
            
            logger.info(f"Job {job.job_id} completed successfully")
            
        except Exception as e:
            job.last_error = str(e)
            job.retry_count += 1
            
            if job.retry_count < job.max_retries:
                # Schedule retry
                job.status = JobStatus.PENDING
                retry_delay = job.retry_delay_seconds * (2 ** (job.retry_count - 1))
                job.next_run = datetime.utcnow() + timedelta(seconds=retry_delay)
                logger.warning(f"Job {job.job_id} failed, retry {job.retry_count}/{job.max_retries} in {retry_delay}s")
            else:
                job.status = JobStatus.FAILED
                logger.error(f"Job {job.job_id} failed after {job.max_retries} retries: {e}")
            
            if self.on_job_error:
                self.on_job_error(job, e)
        
        finally:
            job.updated_at = datetime.utcnow()
            self._running_jobs.pop(job.job_id, None)
            
            # Schedule next run if recurring
            if job.status == JobStatus.COMPLETED:
                job.next_run = self._calculate_next_run(job)
                if job.next_run:
                    job.status = JobStatus.PENDING
                    heapq.heappush(self._job_queue, job)
            elif job.status == JobStatus.PENDING and job.next_run:
                heapq.heappush(self._job_queue, job)
            
            self._persist_jobs()

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[ScheduledJob]:
        """List jobs with optional filtering."""
        jobs = list(self.jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        if tags:
            jobs = [j for j in jobs if any(t in j.tags for t in tags)]
        
        return sorted(jobs, key=lambda j: j.next_run or datetime.max)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        job.status = JobStatus.CANCELLED
        job.updated_at = datetime.utcnow()
        
        # Cancel if running
        if job_id in self._running_jobs:
            self._running_jobs[job_id].cancel()
        
        self._persist_jobs()
        logger.info(f"Cancelled job {job_id}")
        return True

    def pause_job(self, job_id: str) -> bool:
        """Pause a job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        job.status = JobStatus.PAUSED
        job.updated_at = datetime.utcnow()
        self._persist_jobs()
        logger.info(f"Paused job {job_id}")
        return True

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.PAUSED:
            return False
        
        job.status = JobStatus.PENDING
        job.next_run = self._calculate_next_run(job)
        job.updated_at = datetime.utcnow()
        
        if job.next_run:
            heapq.heappush(self._job_queue, job)
        
        self._persist_jobs()
        logger.info(f"Resumed job {job_id}")
        return True

    def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        if job_id not in self.jobs:
            return False
        
        # Cancel if running
        if job_id in self._running_jobs:
            self._running_jobs[job_id].cancel()
        
        del self.jobs[job_id]
        self._persist_jobs()
        logger.info(f"Deleted job {job_id}")
        return True

    def trigger_job(self, job_id: str) -> bool:
        """Manually trigger a job to run now."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        job.next_run = datetime.utcnow()
        job.status = JobStatus.PENDING
        heapq.heappush(self._job_queue, job)
        logger.info(f"Triggered job {job_id}")
        return True

    def get_metrics(self) -> Dict[str, Any]:
        """Get scheduler metrics."""
        status_counts = {}
        for job in self.jobs.values():
            status_counts[job.status.value] = status_counts.get(job.status.value, 0) + 1
        
        return {
            "total_jobs": len(self.jobs),
            "running_jobs": len(self._running_jobs),
            "queued_jobs": len(self._job_queue),
            "status_counts": status_counts,
            "max_concurrent": self.max_concurrent_jobs,
            "scheduler_running": self._running
        }

    def _persist_jobs(self) -> None:
        """Persist jobs to file."""
        if not self.persistence_path:
            return
        
        try:
            jobs_data = []
            for job in self.jobs.values():
                job_dict = {
                    "job_id": job.job_id,
                    "name": job.name,
                    "scraper_type": job.scraper_type,
                    "target": job.target,
                    "schedule": {
                        "schedule_type": job.schedule.schedule_type.value,
                        "run_at": job.schedule.run_at.isoformat() if job.schedule.run_at else None,
                        "interval_seconds": job.schedule.interval_seconds,
                        "cron_expression": job.schedule.cron_expression,
                        "time_of_day": job.schedule.time_of_day,
                        "days_of_week": job.schedule.days_of_week,
                        "timezone": job.schedule.timezone,
                        "max_runs": job.schedule.max_runs,
                        "end_date": job.schedule.end_date.isoformat() if job.schedule.end_date else None
                    },
                    "priority": job.priority.value,
                    "status": job.status.value,
                    "run_count": job.run_count,
                    "last_run": job.last_run.isoformat() if job.last_run else None,
                    "next_run": job.next_run.isoformat() if job.next_run else None,
                    "max_retries": job.max_retries,
                    "retry_count": job.retry_count,
                    "created_at": job.created_at.isoformat(),
                    "tags": job.tags,
                    "options": job.options
                }
                jobs_data.append(job_dict)
            
            with open(self.persistence_path, "w") as f:
                json.dump(jobs_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to persist jobs: {e}")

    def load_jobs(self) -> int:
        """Load jobs from persistence file."""
        if not self.persistence_path:
            return 0
        
        try:
            with open(self.persistence_path, "r") as f:
                jobs_data = json.load(f)
            
            loaded = 0
            for job_dict in jobs_data:
                schedule = JobSchedule(
                    schedule_type=ScheduleType(job_dict["schedule"]["schedule_type"]),
                    run_at=datetime.fromisoformat(job_dict["schedule"]["run_at"]) if job_dict["schedule"]["run_at"] else None,
                    interval_seconds=job_dict["schedule"]["interval_seconds"],
                    cron_expression=job_dict["schedule"]["cron_expression"],
                    time_of_day=job_dict["schedule"]["time_of_day"],
                    days_of_week=job_dict["schedule"]["days_of_week"],
                    timezone=job_dict["schedule"]["timezone"],
                    max_runs=job_dict["schedule"]["max_runs"],
                    end_date=datetime.fromisoformat(job_dict["schedule"]["end_date"]) if job_dict["schedule"]["end_date"] else None
                )
                
                job = ScheduledJob(
                    job_id=job_dict["job_id"],
                    name=job_dict["name"],
                    scraper_type=job_dict["scraper_type"],
                    target=job_dict["target"],
                    schedule=schedule,
                    priority=JobPriority(job_dict["priority"]),
                    status=JobStatus(job_dict["status"]),
                    run_count=job_dict["run_count"],
                    last_run=datetime.fromisoformat(job_dict["last_run"]) if job_dict["last_run"] else None,
                    next_run=datetime.fromisoformat(job_dict["next_run"]) if job_dict["next_run"] else None,
                    max_retries=job_dict["max_retries"],
                    retry_count=job_dict["retry_count"],
                    created_at=datetime.fromisoformat(job_dict["created_at"]),
                    tags=job_dict["tags"],
                    options=job_dict["options"]
                )
                
                self.jobs[job.job_id] = job
                
                if job.status == JobStatus.PENDING and job.next_run:
                    heapq.heappush(self._job_queue, job)
                
                loaded += 1
            
            logger.info(f"Loaded {loaded} jobs from persistence")
            return loaded
            
        except FileNotFoundError:
            return 0
        except Exception as e:
            logger.error(f"Failed to load jobs: {e}")
            return 0
