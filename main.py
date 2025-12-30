#!/usr/bin/env python3
# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
"""
Main entry point for the MJ Data Scraper Suite container.

This module provides:
1. HTTP API for job submission and status
2. Service Bus listener for async job processing
3. Health check endpoints
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from core.websocket_handler import websocket_endpoint, connection_manager, event_emitter
from orchestration.job_scheduler import JobScheduler, JobSchedule, ScheduleType, JobPriority

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="MJ Data Scraper Suite",
    description="Enterprise-grade web scraping platform for Mountain Jewels Intelligence",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
scraper_status = {
    "status": "idle",
    "jobs_completed": 0,
    "jobs_failed": 0,
    "current_job": None,
    "started_at": datetime.utcnow().isoformat()
}

# Job scheduler instance
job_scheduler = JobScheduler(
    max_concurrent_jobs=5,
    persistence_path="./data/scheduled_jobs.json"
)


class ScrapeRequest(BaseModel):
    """Request model for scraping jobs."""
    url: str = Field(..., description="URL to scrape")
    scraper_type: str = Field(default="web", description="Type of scraper to use")
    priority: str = Field(default="normal", description="Job priority")
    options: dict = Field(default_factory=dict, description="Additional scraper options")


class ScrapeResponse(BaseModel):
    """Response model for scraping jobs."""
    job_id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime: str
    jobs_completed: int
    jobs_failed: int


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime=scraper_status["started_at"],
        jobs_completed=scraper_status["jobs_completed"],
        jobs_failed=scraper_status["jobs_failed"]
    )


@app.get("/status")
async def get_status():
    """Get current scraper status."""
    return scraper_status


@app.post("/scrape", response_model=ScrapeResponse)
async def submit_scrape_job(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """Submit a new scraping job."""
    import uuid
    job_id = str(uuid.uuid4())
    
    logger.info(f"Received scrape request for {request.url}, job_id={job_id}")
    
    # Add to background tasks
    background_tasks.add_task(execute_scrape_job, job_id, request)
    
    return ScrapeResponse(
        job_id=job_id,
        status="queued",
        message=f"Job {job_id} queued for processing"
    )


async def execute_scrape_job(job_id: str, request: ScrapeRequest):
    """Execute a scraping job in the background."""
    global scraper_status
    
    try:
        scraper_status["status"] = "running"
        scraper_status["current_job"] = job_id
        
        logger.info(f"Starting scrape job {job_id} for {request.url}")
        
        # Import and run the scraper engine
        from core.scraper_engine import ScraperEngine
        
        engine = ScraperEngine()
        result = await engine.scrape(
            url=request.url,
            scraper_type=request.scraper_type,
            options=request.options
        )
        
        scraper_status["jobs_completed"] += 1
        logger.info(f"Completed scrape job {job_id}")
        
    except Exception as e:
        scraper_status["jobs_failed"] += 1
        logger.error(f"Failed scrape job {job_id}: {e}")
        
    finally:
        scraper_status["status"] = "idle"
        scraper_status["current_job"] = None


@app.get("/scrapers")
async def list_scrapers():
    """List available scrapers."""
    return {
        "scrapers": [
            {"id": "web", "name": "Web Scraper", "description": "General purpose web scraper"},
            {"id": "linkedin", "name": "LinkedIn Scraper", "description": "LinkedIn profile scraper"},
            {"id": "twitter", "name": "Twitter Scraper", "description": "Twitter/X data scraper"},
            {"id": "facebook", "name": "Facebook Scraper", "description": "Facebook page scraper"},
            {"id": "instagram", "name": "Instagram Scraper", "description": "Instagram profile scraper"},
            {"id": "news", "name": "News Scraper", "description": "News article scraper"},
            {"id": "public_records", "name": "Public Records", "description": "Government records scraper"},
            {"id": "business_directory", "name": "Business Directory", "description": "Business listing scraper"},
        ]
    }


@app.get("/sentinels")
async def get_sentinel_status():
    """Get status of all sentinels."""
    return {
        "sentinels": [
            {"id": "performance", "name": "Performance Sentinel", "status": "active"},
            {"id": "network", "name": "Network Sentinel", "status": "active"},
            {"id": "waf", "name": "WAF Sentinel", "status": "active"},
            {"id": "malware", "name": "Malware Sentinel", "status": "active"},
        ],
        "orchestrator": "ready"
    }


# WebSocket endpoint for real-time updates
@app.websocket("/ws/scraper-status")
async def scraper_status_ws(websocket: WebSocket):
    """WebSocket endpoint for real-time scraper status updates."""
    await websocket_endpoint(websocket)


# Scheduled jobs endpoints
class ScheduleJobRequest(BaseModel):
    """Request to create a scheduled job."""
    name: str
    scraper_type: str
    url: str
    schedule_type: str = "interval"  # once, interval, daily, weekly, cron
    interval_seconds: int = 3600
    time_of_day: Optional[str] = None  # HH:MM for daily/weekly
    days_of_week: list = []  # 0-6 for weekly
    cron_expression: Optional[str] = None
    priority: str = "normal"
    options: dict = Field(default_factory=dict)


@app.post("/jobs/schedule")
async def schedule_job(request: ScheduleJobRequest):
    """Create a new scheduled scraping job."""
    schedule_type_map = {
        "once": ScheduleType.ONCE,
        "interval": ScheduleType.INTERVAL,
        "daily": ScheduleType.DAILY,
        "weekly": ScheduleType.WEEKLY,
        "cron": ScheduleType.CRON
    }
    priority_map = {
        "low": JobPriority.LOW,
        "normal": JobPriority.NORMAL,
        "high": JobPriority.HIGH,
        "critical": JobPriority.CRITICAL
    }
    
    schedule = JobSchedule(
        schedule_type=schedule_type_map.get(request.schedule_type, ScheduleType.INTERVAL),
        interval_seconds=request.interval_seconds,
        time_of_day=request.time_of_day,
        days_of_week=request.days_of_week,
        cron_expression=request.cron_expression
    )
    
    job = job_scheduler.create_job(
        name=request.name,
        scraper_type=request.scraper_type,
        target={"url": request.url, **request.options},
        schedule=schedule,
        priority=priority_map.get(request.priority, JobPriority.NORMAL)
    )
    
    return {
        "job_id": job.job_id,
        "name": job.name,
        "next_run": job.next_run.isoformat() if job.next_run else None,
        "status": job.status.value
    }


@app.get("/jobs")
async def list_scheduled_jobs():
    """List all scheduled jobs."""
    jobs = job_scheduler.list_jobs()
    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "name": j.name,
                "scraper_type": j.scraper_type,
                "status": j.status.value,
                "next_run": j.next_run.isoformat() if j.next_run else None,
                "last_run": j.last_run.isoformat() if j.last_run else None,
                "run_count": j.run_count,
                "priority": j.priority.value
            }
            for j in jobs
        ],
        "total": len(jobs)
    }


@app.get("/jobs/{job_id}")
async def get_scheduled_job(job_id: str):
    """Get details of a scheduled job."""
    job = job_scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job.job_id,
        "name": job.name,
        "scraper_type": job.scraper_type,
        "target": job.target,
        "status": job.status.value,
        "next_run": job.next_run.isoformat() if job.next_run else None,
        "last_run": job.last_run.isoformat() if job.last_run else None,
        "run_count": job.run_count,
        "last_result": job.last_result,
        "last_error": job.last_error,
        "priority": job.priority.value,
        "schedule": {
            "type": job.schedule.schedule_type.value,
            "interval_seconds": job.schedule.interval_seconds,
            "time_of_day": job.schedule.time_of_day,
            "cron_expression": job.schedule.cron_expression
        }
    }


@app.post("/jobs/{job_id}/trigger")
async def trigger_job(job_id: str):
    """Manually trigger a scheduled job to run now."""
    if not job_scheduler.trigger_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "triggered", "job_id": job_id}


@app.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """Pause a scheduled job."""
    if not job_scheduler.pause_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "paused", "job_id": job_id}


@app.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """Resume a paused job."""
    if not job_scheduler.resume_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found or not paused")
    return {"status": "resumed", "job_id": job_id}


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a scheduled job."""
    if not job_scheduler.delete_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "deleted", "job_id": job_id}


@app.get("/jobs/metrics")
async def get_scheduler_metrics():
    """Get job scheduler metrics."""
    return job_scheduler.get_metrics()


@app.get("/ws/connections")
async def get_websocket_connections():
    """Get WebSocket connection metrics."""
    return connection_manager.get_metrics()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # Create data directory
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./screenshots", exist_ok=True)
    
    # Load persisted jobs
    job_scheduler.load_jobs()
    
    # Start scheduler
    await job_scheduler.start()
    
    # Start event emitter
    await event_emitter.start()
    
    logger.info("Scraper services initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await job_scheduler.stop()
    await event_emitter.stop()
    logger.info("Scraper services stopped")


def main():
    """Main entry point."""
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting MJ Data Scraper Suite on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
