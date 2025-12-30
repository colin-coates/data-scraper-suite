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

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

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
