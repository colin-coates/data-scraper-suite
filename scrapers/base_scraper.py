# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Base Scraper Class for Data Scraper Suite

Provides common functionality and interface for all scrapers.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Abstract base class for all scrapers."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.is_running = False

    @abstractmethod
    async def scrape(self, target: str, **kwargs) -> List[Dict[str, Any]]:
        """Abstract method to perform scraping.

        Args:
            target: The target to scrape (URL, search term, etc.)
            **kwargs: Additional parameters

        Returns:
            List of scraped data items
        """
        pass

    @abstractmethod
    def validate_target(self, target: str) -> bool:
        """Validate if target is suitable for this scraper.

        Args:
            target: Target to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    async def run(self, targets: List[str], **kwargs) -> Dict[str, Any]:
        """Run the scraper on multiple targets.

        Args:
            targets: List of targets to scrape
            **kwargs: Additional parameters

        Returns:
            Results summary
        """
        self.is_running = True
        start_time = datetime.now()

        results = {
            "scraper": self.name,
            "targets_processed": 0,
            "data_collected": 0,
            "errors": 0,
            "start_time": start_time.isoformat(),
            "results": []
        }

        try:
            for target in targets:
                if not self.validate_target(target):
                    self.logger.warning(f"Invalid target: {target}")
                    results["errors"] += 1
                    continue

                try:
                    self.logger.info(f"Scraping target: {target}")
                    data = await self.scrape(target, **kwargs)

                    results["results"].append({
                        "target": target,
                        "data_count": len(data),
                        "data": data,
                        "success": True
                    })

                    results["targets_processed"] += 1
                    results["data_collected"] += len(data)

                except Exception as e:
                    self.logger.error(f"Error scraping {target}: {e}")
                    results["results"].append({
                        "target": target,
                        "error": str(e),
                        "success": False
                    })
                    results["errors"] += 1

        finally:
            self.is_running = False
            results["end_time"] = datetime.now().isoformat()
            results["duration_seconds"] = (datetime.now() - start_time).total_seconds()

        self.logger.info(f"Scraping completed: {results['data_collected']} items from {results['targets_processed']} targets")
        return results

    def get_status(self) -> Dict[str, Any]:
        """Get current scraper status."""
        return {
            "name": self.name,
            "is_running": self.is_running,
            "config": self.config
        }

    async def cleanup(self):
        """Cleanup resources."""
        self.logger.info(f"Cleaning up {self.name} scraper")
        # Override in subclasses for specific cleanup
